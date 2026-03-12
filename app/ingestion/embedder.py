"""
app/ingestion/embedder.py
─────────────────────────
OpenAI embedding generator for the Hybrid Search RAG Engine.

Responsibilities:
    1. Accept a list of ``ChunkRecord`` objects from the chunker.
    2. Send text to OpenAI ``text-embedding-3-small`` in batches of at
       most ``BATCH_SIZE`` chunks per API call (OpenAI's recommended max
       is 2048 inputs, but 100 keeps individual request latency low and
       makes retries cheap).
    3. Retry transient failures (rate limits, timeouts, connection drops)
       with exponential back-off using ``tenacity``.
    4. Return a flat list of ``EmbeddedChunk`` objects that bundle the
       original chunk text + metadata + the embedding vector.

Design decisions
────────────────
• **Synchronous** — ingestion is a one-time batch job, not a hot path.
  Sync code is easier to reason about and test without an event loop.
• **Batching** — a single API call with 100 inputs costs the same per
  token as 100 individual calls, but uses ~1 TCP connection vs 100.
  Batching also makes retry logic simpler: retry the whole batch, not
  individual items.
• **Tenacity over manual retry loops** — tenacity decorators compose
  cleanly with logging and don't pollute the business logic with
  while/sleep/counter boilerplate.
• **Preserve order** — the OpenAI API returns embeddings in the same
  order as the input; we assert this with the ``index`` field so the
  FAISS index and chunk list stay aligned.

Config (from .env):
    OPENAI_API_KEY        — required
    EMBEDDING_MODEL       — default: text-embedding-3-small
    EMBEDDING_BATCH_SIZE  — default: 100  (max chunks per API call)
    EMBEDDING_MAX_RETRIES — default: 5    (max retry attempts)
    EMBEDDING_DIMENSIONS  — default: 1536 (text-embedding-3-small native dim)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterator

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.ingestion.chunker import ChunkRecord

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"
DEFAULT_BATCH_SIZE: int = 100       # chunks per API request
DEFAULT_MAX_RETRIES: int = 5        # tenacity stop_after_attempt
DEFAULT_DIMENSIONS: int = 1536      # text-embedding-3-small native dimension

# Transient errors worth retrying — do NOT retry AuthenticationError,
# PermissionDeniedError, or NotFoundError (those are config bugs).
_RETRYABLE_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
)


# ── Output type ──────────────────────────────────────────────────────────────


@dataclass
class EmbeddedChunk:
    """A chunk record paired with its embedding vector.

    Attributes:
        text:      Raw text of the chunk.
        metadata:  Provenance dict (filename, page_num, chunk_id).
        embedding: Dense vector from OpenAI embeddings API.
                   Length == ``EMBEDDING_DIMENSIONS`` (default 1536).
    """

    text: str
    metadata: dict
    embedding: list[float] = field(repr=False)   # too long to print

    def __repr__(self) -> str:
        """Return a compact representation for logging."""
        return (
            f"EmbeddedChunk(id={self.metadata.get('chunk_id')!r}, "
            f"dim={len(self.embedding)})"
        )


# ── Retry-decorated API call ──────────────────────────────────────────────────


def _make_retry_decorator(max_retries: int):
    """Build a tenacity ``@retry`` decorator for OpenAI embedding calls.

    Uses exponential back-off starting at 2 s, capped at 60 s.
    Logs a warning before each sleep so operators can see what's happening.

    Args:
        max_retries: Maximum number of attempts before giving up.

    Returns:
        A configured ``tenacity.retry`` decorator.
    """
    return retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,   # raise the original exception after all attempts fail
    )


def _call_embeddings_api(
    client: OpenAI,
    texts: list[str],
    model: str,
) -> list[list[float]]:
    """Call the OpenAI embeddings endpoint for a single batch of texts.

    This function is deliberately thin — all retry and batching logic
    lives in the callers so this is easy to mock in tests.

    Args:
        client: Authenticated ``OpenAI`` client instance.
        texts:  List of strings to embed (max 2048 per OpenAI limit).
        model:  OpenAI embedding model name.

    Returns:
        List of embedding vectors in the same order as ``texts``.

    Raises:
        RateLimitError:      HTTP 429 — caller will retry.
        APITimeoutError:     Request timed out — caller will retry.
        APIConnectionError:  Network failure — caller will retry.
        APIStatusError:      Non-retryable HTTP errors (4xx/5xx).
    """
    response = client.embeddings.create(
        input=texts,
        model=model,
    )

    # OpenAI guarantees index-order preservation, but we verify defensively
    sorted_data = sorted(response.data, key=lambda e: e.index)
    if len(sorted_data) != len(texts):
        raise ValueError(
            f"OpenAI returned {len(sorted_data)} embeddings for "
            f"{len(texts)} inputs — unexpected mismatch."
        )

    logger.debug(
        "Embeddings batch: %d texts → model=%s, usage=%s",
        len(texts),
        response.model,
        response.usage,
    )
    return [e.embedding for e in sorted_data]


# ── Batching helper ───────────────────────────────────────────────────────────


def _iter_batches(
    items: list,
    batch_size: int,
) -> Iterator[tuple[int, list]]:
    """Yield ``(start_index, batch)`` tuples for a list in fixed-size windows.

    Args:
        items:      The full list to iterate over.
        batch_size: Maximum number of items per batch.

    Yields:
        Tuples of ``(start_index, sub_list)`` where ``start_index`` is the
        position of the first item in the original list.

    Example::

        for start, batch in _iter_batches(range(250), 100):
            print(start, len(batch))
        # 0 100 / 100 100 / 200 50
    """
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


# ── Public API ────────────────────────────────────────────────────────────────


def embed_chunks(
    chunks: list[ChunkRecord],
    *,
    model: str | None = None,
    batch_size: int | None = None,
    max_retries: int | None = None,
) -> list[EmbeddedChunk]:
    """Generate embeddings for a list of ``ChunkRecord`` objects.

    Sends chunks to the OpenAI embeddings API in batches, with automatic
    exponential-back-off retries on transient failures.  Returns the
    embeddings in the same order as the input list.

    Args:
        chunks:      List of ``ChunkRecord`` objects from the chunker.
        model:       Override the ``EMBEDDING_MODEL`` env var.
        batch_size:  Override the ``EMBEDDING_BATCH_SIZE`` env var.
                     Max 100 is recommended to keep per-request latency
                     under ~2 s on the free-tier rate limit.
        max_retries: Override the ``EMBEDDING_MAX_RETRIES`` env var.

    Returns:
        List of ``EmbeddedChunk`` objects, one per input chunk, in the
        same order as ``chunks``.

    Raises:
        EnvironmentError:  If ``OPENAI_API_KEY`` is not set.
        ValueError:        If ``chunks`` is empty.
        RateLimitError:    If all retry attempts are exhausted on 429.
        APIStatusError:    On non-retryable API errors.

    Example::

        from app.ingestion.loader import load_document
        from app.ingestion.chunker import chunk_documents
        from app.ingestion.embedder import embed_chunks

        pages  = load_document("data/report.pdf")
        chunks = chunk_documents(pages)
        embedded = embed_chunks(chunks)

        print(f"{len(embedded)} vectors, dim={len(embedded[0].embedding)}")
    """
    if not chunks:
        raise ValueError("embed_chunks received an empty chunk list.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    # Resolve config: argument → env var → default
    resolved_model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    resolved_batch = batch_size or int(
        os.getenv("EMBEDDING_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))
    )
    resolved_retries = max_retries or int(
        os.getenv("EMBEDDING_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))
    )

    client = OpenAI(api_key=api_key)
    retry_decorator = _make_retry_decorator(resolved_retries)

    # Wrap _call_embeddings_api with the retry decorator at call time
    # (not at definition time) so max_retries can vary per call
    retrying_call = retry_decorator(_call_embeddings_api)

    total = len(chunks)
    n_batches = (total + resolved_batch - 1) // resolved_batch
    logger.info(
        "Embedding %d chunks in %d batches (batch_size=%d, model=%s)",
        total,
        n_batches,
        resolved_batch,
        resolved_model,
    )

    results: list[EmbeddedChunk] = []
    t_start = time.monotonic()

    for batch_num, (start, batch) in enumerate(
        _iter_batches(chunks, resolved_batch), start=1
    ):
        texts = [c.text for c in batch]

        logger.debug(
            "Batch %d/%d: chunks %d–%d",
            batch_num,
            n_batches,
            start + 1,
            start + len(batch),
        )

        try:
            vectors = retrying_call(client, texts, resolved_model)
        except _RETRYABLE_ERRORS as exc:
            # All retries exhausted — log and re-raise for the caller to handle
            logger.error(
                "Batch %d/%d failed after %d retries: %s",
                batch_num,
                n_batches,
                resolved_retries,
                exc,
            )
            raise
        except APIStatusError as exc:
            logger.error(
                "Non-retryable API error on batch %d/%d (HTTP %d): %s",
                batch_num,
                n_batches,
                exc.status_code,
                exc.message,
            )
            raise

        for chunk, vector in zip(batch, vectors):
            results.append(
                EmbeddedChunk(
                    text=chunk.text,
                    metadata=chunk.metadata.copy(),
                    embedding=vector,
                )
            )

        elapsed = time.monotonic() - t_start
        logger.info(
            "Embedded batch %d/%d | %d/%d chunks done | %.1fs elapsed",
            batch_num,
            n_batches,
            len(results),
            total,
            elapsed,
        )

    elapsed_total = time.monotonic() - t_start
    logger.info(
        "Embedding complete: %d chunks in %.1fs (%.1f chunks/s)",
        total,
        elapsed_total,
        total / elapsed_total if elapsed_total > 0 else float("inf"),
    )
    return results


def get_query_embedding(
    query: str,
    *,
    model: str | None = None,
    max_retries: int | None = None,
) -> list[float]:
    """Generate an embedding vector for a single query string.

    Used at query time by the vector store to embed the user's question
    before performing the ANN search.

    Args:
        query:       The natural-language query string.
        model:       Override the ``EMBEDDING_MODEL`` env var.
        max_retries: Override the ``EMBEDDING_MAX_RETRIES`` env var.

    Returns:
        A single embedding vector (list of floats).

    Raises:
        EnvironmentError: If ``OPENAI_API_KEY`` is not set.
        ValueError:       If ``query`` is empty or whitespace-only.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query string must not be empty.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    resolved_model = model or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    resolved_retries = max_retries or int(
        os.getenv("EMBEDDING_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))
    )

    client = OpenAI(api_key=api_key)
    retry_decorator = _make_retry_decorator(resolved_retries)
    retrying_call = retry_decorator(_call_embeddings_api)

    logger.debug("Embedding query (%d chars) with model=%s", len(query), resolved_model)
    vectors = retrying_call(client, [query], resolved_model)
    return vectors[0]