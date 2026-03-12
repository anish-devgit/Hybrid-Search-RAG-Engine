"""
app/ingestion/loader.py
──────────────────────
Multi-format document loader for the Hybrid Search RAG Engine.

Supports:
    • PDF   — via pypdf (page-level granularity)
    • DOCX  — via python-docx (paragraph-level, mapped to pseudo-pages)
    • TXT   — plain text (no page concept; assigned page 1)

Each loader returns a list of ``(text, metadata)`` tuples where:
    - text     : str  — the raw extracted text for that page / chunk
    - metadata : dict — always contains ``filename`` (str) and
                        ``page_num`` (int, 1-based)

Design decisions
────────────────
* Page-level granularity for PDFs preserves citation accuracy; the LLM
  can say "see page 12 of report.pdf" instead of a vague passage reference.
* DOCX paragraphs are batched into pseudo-pages of ~DOCX_PAGE_SIZE words
  so downstream chunking sees a comparable input size to PDFs.
* Every function is pure (no side-effects beyond reading the file), making
  unit-testing trivial.
* All errors raise ``DocumentLoadError`` (a domain-specific exception) so
  callers don't have to catch bare OSError / ValueError etc.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import NamedTuple

import pypdf
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".txt"})

# Number of words batched into one "pseudo-page" for DOCX files.
# 300 words ≈ one A4 page of typical body text — keeps pseudo-pages
# roughly comparable to a real PDF page for downstream chunk sizing.
DOCX_PAGE_SIZE: int = 300


# ── Domain types ─────────────────────────────────────────────────────────────


class PageRecord(NamedTuple):
    """One extracted page / pseudo-page with its provenance metadata."""

    text: str
    metadata: dict  # {"filename": str, "page_num": int}


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or parsed."""


# ── Internal helpers ─────────────────────────────────────────────────────────


def _make_metadata(filename: str, page_num: int) -> dict:
    """Build a standardised metadata dict.

    Args:
        filename: The basename of the source file (e.g. ``"report.pdf"``).
        page_num: 1-based page number within that file.

    Returns:
        Dict with keys ``filename`` and ``page_num``.
    """
    return {"filename": filename, "page_num": page_num}


def _batch_words_into_pages(words: list[str], page_size: int) -> list[str]:
    """Split a flat word list into fixed-size page strings.

    Args:
        words:     All words extracted from the document.
        page_size: Maximum number of words per page string.

    Returns:
        List of page strings; the final page may be shorter than
        ``page_size``.
    """
    pages: list[str] = []
    for start in range(0, len(words), page_size):
        pages.append(" ".join(words[start : start + page_size]))
    return pages


# ── Format-specific loaders ──────────────────────────────────────────────────


def load_pdf(file_path: Path) -> list[PageRecord]:
    """Extract text from a PDF file, one ``PageRecord`` per PDF page.

    Empty pages (after stripping whitespace) are skipped so that
    cover-page blobs and blank separators don't pollute the index.

    Args:
        file_path: Absolute or relative path to a ``.pdf`` file.

    Returns:
        List of ``PageRecord`` objects, one per non-empty PDF page.

    Raises:
        DocumentLoadError: If the file cannot be opened or parsed by pypdf.
    """
    logger.info("Loading PDF: %s", file_path)
    filename = file_path.name
    records: list[PageRecord] = []

    try:
        reader = pypdf.PdfReader(str(file_path))
        total_pages = len(reader.pages)
        logger.debug("PDF '%s' has %d pages", filename, total_pages)

        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                # Some pages (images, forms) raise on extraction; skip them.
                logger.warning(
                    "Could not extract text from page %d of '%s': %s",
                    page_index,
                    filename,
                    exc,
                )
                continue

            text = text.strip()
            if not text:
                logger.debug("Skipping blank page %d in '%s'", page_index, filename)
                continue

            records.append(
                PageRecord(
                    text=text,
                    metadata=_make_metadata(filename, page_index),
                )
            )

    except pypdf.errors.PdfReadError as exc:
        raise DocumentLoadError(
            f"pypdf could not read '{file_path}': {exc}"
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(
            f"Cannot open file '{file_path}': {exc}"
        ) from exc

    logger.info(
        "PDF '%s': extracted %d non-empty pages (of %d total)",
        filename,
        len(records),
        total_pages,
    )
    return records


def load_docx(file_path: Path) -> list[PageRecord]:
    """Extract text from a DOCX file, batched into pseudo-pages.

    DOCX files have no true page concept in their XML representation,
    so paragraphs are accumulated into fixed-size word batches
    (``DOCX_PAGE_SIZE`` words each) that approximate a physical page.

    Args:
        file_path: Absolute or relative path to a ``.docx`` file.

    Returns:
        List of ``PageRecord`` objects, one per pseudo-page.

    Raises:
        DocumentLoadError: If the file cannot be opened or parsed.
    """
    logger.info("Loading DOCX: %s", file_path)
    filename = file_path.name

    try:
        doc = DocxDocument(str(file_path))
    except Exception as exc:
        raise DocumentLoadError(
            f"python-docx could not open '{file_path}': {exc}"
        ) from exc

    # Collect all non-empty paragraph text
    all_text = " ".join(
        para.text.strip()
        for para in doc.paragraphs
        if para.text.strip()
    )

    if not all_text:
        logger.warning("DOCX '%s' contained no extractable text.", filename)
        return []

    words = all_text.split()
    pseudo_pages = _batch_words_into_pages(words, DOCX_PAGE_SIZE)

    records = [
        PageRecord(
            text=page_text,
            metadata=_make_metadata(filename, page_num),
        )
        for page_num, page_text in enumerate(pseudo_pages, start=1)
    ]

    logger.info(
        "DOCX '%s': extracted %d pseudo-pages from %d words",
        filename,
        len(records),
        len(words),
    )
    return records


def load_txt(file_path: Path) -> list[PageRecord]:
    """Extract text from a plain-text file as a single page.

    The entire file is returned as ``page_num=1``.  If the file is
    very large it will be split by the chunker downstream; keeping it
    as one record here avoids artificial mid-sentence breaks.

    Args:
        file_path: Absolute or relative path to a ``.txt`` file.

    Returns:
        A list containing a single ``PageRecord``.

    Raises:
        DocumentLoadError: If the file cannot be read.
    """
    logger.info("Loading TXT: %s", file_path)
    filename = file_path.name

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        raise DocumentLoadError(
            f"Cannot read file '{file_path}': {exc}"
        ) from exc

    if not text:
        logger.warning("TXT file '%s' is empty.", filename)
        return []

    records = [PageRecord(text=text, metadata=_make_metadata(filename, 1))]
    logger.info("TXT '%s': loaded %d characters as 1 page", filename, len(text))
    return records


# ── Public API ───────────────────────────────────────────────────────────────


def load_document(file_path: str | Path) -> list[PageRecord]:
    """Load any supported document and return a list of ``PageRecord`` objects.

    This is the single entry-point for the ingestion pipeline.  It
    dispatches to the appropriate format-specific loader based on the
    file extension.

    Args:
        file_path: Path to a ``.pdf``, ``.docx``, or ``.txt`` file.

    Returns:
        List of ``PageRecord`` objects containing extracted text and
        metadata.  An empty list is returned if the document has no
        extractable content.

    Raises:
        DocumentLoadError: If the file does not exist, is an unsupported
            format, or cannot be parsed by the underlying library.

    Example::

        records = load_document("data/annual_report.pdf")
        for text, meta in records:
            print(meta["filename"], meta["page_num"], text[:80])
    """
    path = Path(file_path)

    if not path.exists():
        raise DocumentLoadError(f"File not found: '{path}'")

    if not path.is_file():
        raise DocumentLoadError(f"Path is not a file: '{path}'")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    dispatch = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
    }

    loader_fn = dispatch[ext]
    return loader_fn(path)


def load_directory(dir_path: str | Path) -> list[PageRecord]:
    """Recursively load all supported documents from a directory.

    Files with unsupported extensions are silently skipped.  Individual
    file errors are logged as warnings and do not abort the entire batch.

    Args:
        dir_path: Path to a directory containing documents.

    Returns:
        Concatenated list of ``PageRecord`` objects from all loaded files,
        in filesystem traversal order.

    Raises:
        DocumentLoadError: If ``dir_path`` does not exist or is not a
            directory.

    Example::

        records = load_directory("data/")
        print(f"Loaded {len(records)} pages from {dir_path}")
    """
    directory = Path(dir_path)

    if not directory.exists():
        raise DocumentLoadError(f"Directory not found: '{directory}'")
    if not directory.is_dir():
        raise DocumentLoadError(f"Path is not a directory: '{directory}'")

    all_records: list[PageRecord] = []
    found_files = sorted(directory.rglob("*"))  # deterministic order

    for file_path in found_files:
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            records = load_document(file_path)
            all_records.extend(records)
            logger.info(
                "Loaded '%s' → %d pages (running total: %d)",
                file_path.name,
                len(records),
                len(all_records),
            )
        except DocumentLoadError as exc:
            logger.warning("Skipping '%s': %s", file_path.name, exc)

    logger.info(
        "Directory '%s': loaded %d total pages from %d files",
        directory,
        len(all_records),
        sum(1 for f in found_files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS),
    )
    return all_records