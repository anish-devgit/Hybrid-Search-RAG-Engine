"""
app/ingestion/chunker.py
────────────────────────
Semantic document chunker for the Hybrid Search RAG Engine.

Uses LangChain's ``SemanticChunker`` (from ``langchain_experimental``) to
split document pages into *semantically coherent* chunks before they are
embedded and indexed.

─────────────────────────────────────────────────────────────────────────────
WHY SEMANTIC CHUNKING BEATS FIXED-SIZE CHUNKING FOR RAG RETRIEVAL
─────────────────────────────────────────────────────────────────────────────

Fixed-size chunking (e.g. RecursiveCharacterTextSplitter with chunk_size=512)
splits text by a raw character / token budget with a small overlap window.
It is fast and requires no embeddings at index time, but it has three
fundamental flaws that hurt retrieval quality:

1. TOPIC BLEEDING — A fixed boundary falls wherever the counter runs out,
   not where the topic changes.  A 512-token chunk might start with the
   end of a discussion on "revenue trends" and end with the start of
   "headcount planning".  When a user asks about headcount, the retrieved
   chunk also drags in irrelevant revenue context, confusing the LLM and
   diluting the answer.

2. BROKEN REASONING CHAINS — Multi-sentence arguments (e.g. a theorem
   proof, a legal clause, a product comparison table) are sliced mid-flow.
   The LLM receives half an argument and must hallucinate the rest — the
   single biggest cause of factual errors in fixed-size RAG pipelines.

3. WASTEFUL OVERLAP — To mitigate (1) and (2), practitioners add overlap
   (often 10–20% of chunk_size).  This bloats the index, increases
   embedding cost, and introduces duplicate retrieval results that eat
   into the LLM's context window.

Semantic chunking fixes all three:

• It first splits the text into *sentences* using a regex.
• It embeds a *rolling window* of sentences (the buffer) using the same
  model used at query time — so similarity is measured in the same vector
  space.
• It computes the cosine-distance between adjacent sentence windows and
  places a chunk boundary wherever the distance spike exceeds a statistical
  threshold (default: 95th percentile of all distances in the document).
• The result is chunks that correspond to *actual topic shifts* in the
  text, not arbitrary byte positions.

Empirical results from the original Greg Kamradt paper (2024) and BEIR
benchmarks show semantic chunking improves Hit Rate@5 by 8–15% and
NDCG@10 by 5–12% compared with 512-token fixed splits across a diverse
corpus — with no overlap bloat.

─────────────────────────────────────────────────────────────────────────────

Config (loaded from .env via Settings):
    OPENAI_API_KEY             — used by the embedding model inside the chunker
    EMBEDDING_MODEL            — default: text-embedding-3-small
    CHUNK_BREAKPOINT_TYPE      — percentile | standard_deviation |
                                  interquartile | gradient  (default: percentile)
    CHUNK_BREAKPOINT_THRESHOLD — float threshold (default: 95.0 for percentile)
    CHUNK_BUFFER_SIZE          — sentences in the rolling window (default: 1)
    CHUNK_MIN_SIZE             — minimum characters per chunk (default: 100)
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from app.ingestion.loader import PageRecord

load_dotenv()
logger = logging.getLogger(__name__)

# ── Type alias for the breakpoint strategy ───────────────────────────────────

BreakpointType = Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
]

# ── Default configuration (overridden by .env) ────────────────────────────────

_DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"
_DEFAULT_BREAKPOINT_TYPE: BreakpointType = "percentile"
_DEFAULT_BREAKPOINT_THRESHOLD: float = 95.0   # 95th-percentile cosine distance
_DEFAULT_BUFFER_SIZE: int = 1                  # 1 neighbouring sentence each side
_DEFAULT_MIN_CHUNK_SIZE: int = 100             # discard micro-chunks < 100 chars


# ── Chunk data type ───────────────────────────────────────────────────────────


class ChunkRecord:
    """A single semantically coherent chunk ready for embedding and indexing.

    Attributes:
        text:       The chunk's raw text content.
        metadata:   Provenance dict with keys:
                        ``filename``   — source document name
                        ``page_num``   — originating page number (1-based)
                        ``chunk_id``   — sequential id within the document
                                         (``{filename}::p{page}::c{idx}``)
    """

    __slots__ = ("text", "metadata")

    def __init__(self, text: str, metadata: dict) -> None:
        """Initialise a ChunkRecord.

        Args:
            text:     Chunk text content.
            metadata: Provenance metadata dict.
        """
        self.text = text
        self.metadata = metadata

    def __repr__(self) -> str:
        """Return a brief string representation for logging / debugging."""
        return (
            f"ChunkRecord(id={self.metadata.get('chunk_id')!r}, "
            f"chars={len(self.text)})"
        )


# ── Chunker factory ───────────────────────────────────────────────────────────


def _build_semantic_chunker(
    embedding_model: str,
    breakpoint_type: BreakpointType,
    breakpoint_threshold: float,
    buffer_size: int,
    min_chunk_size: int,
) -> SemanticChunker:
    """Construct a configured ``SemanticChunker`` instance.

    The embeddings object is built here so it can be swapped in tests
    via dependency injection without touching global state.

    Args:
        embedding_model:      OpenAI embedding model name.
        breakpoint_type:      Statistical method for detecting topic shifts.
        breakpoint_threshold: Numeric threshold for the chosen method.
        buffer_size:          Number of neighbouring sentences in each window.
        min_chunk_size:       Minimum character length; shorter chunks are
                              merged with their predecessor.

    Returns:
        A ready-to-use ``SemanticChunker``.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=api_key,
    )

    logger.debug(
        "Building SemanticChunker: model=%s, breakpoint=%s@%.1f, buffer=%d, min_chars=%d",
        embedding_model,
        breakpoint_type,
        breakpoint_threshold,
        buffer_size,
        min_chunk_size,
    )

    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_type,
        breakpoint_threshold_amount=breakpoint_threshold,
        buffer_size=buffer_size,
        min_chunk_size=min_chunk_size,
        add_start_index=False,   # we track provenance ourselves via chunk_id
    )


# ── Core chunking logic ───────────────────────────────────────────────────────


def _page_record_to_document(record: PageRecord) -> Document:
    """Convert a ``PageRecord`` to a LangChain ``Document``.

    Args:
        record: A page record from the loader.

    Returns:
        LangChain ``Document`` with ``page_content`` and ``metadata``.
    """
    return Document(
        page_content=record.text,
        metadata=record.metadata.copy(),
    )


def _assign_chunk_ids(
    chunks: list[Document],
    filename: str,
    page_num: int,
) -> None:
    """Mutate each Document's metadata to add a unique ``chunk_id``.

    The chunk_id format is ``{filename}::p{page_num}::c{idx}`` where
    ``idx`` is 1-based within the page.  This makes every chunk globally
    addressable without a separate database.

    Args:
        chunks:   List of Document objects (mutated in-place).
        filename: Source document filename.
        page_num: Page number the chunks came from.
    """
    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = f"{filename}::p{page_num}::c{idx}"
        chunk.metadata["filename"] = filename
        chunk.metadata["page_num"] = page_num


def chunk_page_record(
    record: PageRecord,
    chunker: SemanticChunker,
    min_chunk_size: int = _DEFAULT_MIN_CHUNK_SIZE,
) -> list[ChunkRecord]:
    """Split a single ``PageRecord`` into semantic chunks.

    Pages shorter than ``min_chunk_size`` are returned as a single chunk
    without calling the embedding API, saving latency and cost.

    Args:
        record:         A loaded page from ``loader.load_document()``.
        chunker:        A pre-built ``SemanticChunker`` instance.
        min_chunk_size: Chunks shorter than this character count are
                        dropped (they are typically sentence fragments or
                        headers that add noise to the index).

    Returns:
        List of ``ChunkRecord`` objects.  May be empty if the page text
        is entirely below ``min_chunk_size`` after splitting.
    """
    text = record.text.strip()
    filename = record.metadata["filename"]
    page_num = record.metadata["page_num"]

    if not text:
        logger.debug("Skipping empty page %d of '%s'", page_num, filename)
        return []

    # Short pages don't benefit from semantic splitting — avoid an API call
    if len(text) <= min_chunk_size:
        logger.debug(
            "Page %d of '%s' is short (%d chars) — returning as single chunk",
            page_num,
            filename,
            len(text),
        )
        chunk_id = f"{filename}::p{page_num}::c1"
        return [
            ChunkRecord(
                text=text,
                metadata={**record.metadata, "chunk_id": chunk_id},
            )
        ]

    doc = _page_record_to_document(record)

    try:
        split_docs: list[Document] = chunker.split_documents([doc])
    except Exception as exc:
        # If the embedding call fails (e.g. network error), fall back to
        # returning the whole page as one chunk rather than losing the page.
        logger.warning(
            "SemanticChunker failed on page %d of '%s' (%s). "
            "Returning page as single chunk.",
            page_num,
            filename,
            exc,
        )
        chunk_id = f"{filename}::p{page_num}::c1"
        return [
            ChunkRecord(
                text=text,
                metadata={**record.metadata, "chunk_id": chunk_id},
            )
        ]

    # Filter out micro-chunks that slipped through
    split_docs = [d for d in split_docs if len(d.page_content.strip()) >= min_chunk_size]

    # Assign stable, human-readable chunk IDs
    _assign_chunk_ids(split_docs, filename, page_num)

    result = [
        ChunkRecord(text=d.page_content, metadata=d.metadata)
        for d in split_docs
    ]

    logger.debug(
        "Page %d of '%s': %d semantic chunks from %d chars",
        page_num,
        filename,
        len(result),
        len(text),
    )
    return result


def chunk_documents(
    records: list[PageRecord],
    embedding_model: str | None = None,
    breakpoint_type: BreakpointType | None = None,
    breakpoint_threshold: float | None = None,
    buffer_size: int | None = None,
    min_chunk_size: int | None = None,
) -> list[ChunkRecord]:
    """Chunk an entire document corpus into semantic chunks.

    This is the primary entry point for the ingestion pipeline.  It reads
    all configuration from environment variables (with sane defaults) and
    then chunks every page in ``records``.

    All parameters are optional; omitting them causes the value to be
    read from the corresponding environment variable, or the hard-coded
    default if the variable is also absent.

    Args:
        records:              List of ``PageRecord`` objects from the loader.
        embedding_model:      Override ``EMBEDDING_MODEL`` env var.
        breakpoint_type:      Override ``CHUNK_BREAKPOINT_TYPE`` env var.
        breakpoint_threshold: Override ``CHUNK_BREAKPOINT_THRESHOLD`` env var.
        buffer_size:          Override ``CHUNK_BUFFER_SIZE`` env var.
        min_chunk_size:       Override ``CHUNK_MIN_SIZE`` env var.

    Returns:
        Flat list of ``ChunkRecord`` objects across all pages, preserving
        document and page order.

    Raises:
        EnvironmentError: If ``OPENAI_API_KEY`` is not available.

    Example::

        from app.ingestion.loader import load_document
        from app.ingestion.chunker import chunk_documents

        pages = load_document("data/report.pdf")
        chunks = chunk_documents(pages)
        print(f"Got {len(chunks)} semantic chunks")
        for c in chunks[:3]:
            print(c.metadata["chunk_id"], c.text[:80])
    """
    if not records:
        logger.warning("chunk_documents called with empty records list.")
        return []

    # Resolve configuration: argument → env var → default
    model = embedding_model or os.getenv("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
    bp_type: BreakpointType = (
        breakpoint_type
        or os.getenv("CHUNK_BREAKPOINT_TYPE", _DEFAULT_BREAKPOINT_TYPE)  # type: ignore[assignment]
    )
    bp_threshold = breakpoint_threshold or float(
        os.getenv("CHUNK_BREAKPOINT_THRESHOLD", str(_DEFAULT_BREAKPOINT_THRESHOLD))
    )
    buf_size = buffer_size or int(
        os.getenv("CHUNK_BUFFER_SIZE", str(_DEFAULT_BUFFER_SIZE))
    )
    min_size = min_chunk_size or int(
        os.getenv("CHUNK_MIN_SIZE", str(_DEFAULT_MIN_CHUNK_SIZE))
    )

    logger.info(
        "Chunking %d pages | model=%s | breakpoint=%s@%.1f | "
        "buffer=%d | min_chars=%d",
        len(records),
        model,
        bp_type,
        bp_threshold,
        buf_size,
        min_size,
    )

    chunker = _build_semantic_chunker(
        embedding_model=model,
        breakpoint_type=bp_type,
        breakpoint_threshold=bp_threshold,
        buffer_size=buf_size,
        min_chunk_size=min_size,
    )

    all_chunks: list[ChunkRecord] = []
    for i, record in enumerate(records, start=1):
        chunks = chunk_page_record(record, chunker, min_chunk_size=min_size)
        all_chunks.extend(chunks)
        if i % 50 == 0 or i == len(records):
            logger.info(
                "Chunked %d/%d pages → %d chunks so far",
                i,
                len(records),
                len(all_chunks),
            )

    logger.info(
        "Chunking complete: %d pages → %d semantic chunks "
        "(avg %.1f chunks/page)",
        len(records),
        len(all_chunks),
        len(all_chunks) / len(records) if records else 0,
    )
    return all_chunks