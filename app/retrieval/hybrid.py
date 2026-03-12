"""
app/retrieval/hybrid.py
────────────────────────
Hybrid retrieval engine using Reciprocal Rank Fusion (RRF).

This module is the algorithmic heart of the RAG engine.  It:

1. Accepts a query string.
2. Embeds it once and passes the vector to both FAISS and BM25.
3. Collects top-K candidates from each retriever.
4. Merges the two ranked lists using RRF.
5. Returns the top-5 fused chunks to the generation layer.

─────────────────────────────────────────────────────────────────────────────
RECIPROCAL RANK FUSION — FULL EXPLANATION
─────────────────────────────────────────────────────────────────────────────

Problem: FAISS and BM25 each return a ranked list.  The scores are
incomparable — FAISS returns cosine similarities in [−1, 1] while BM25
returns non-normalised term-frequency-weighted IDF sums.  You cannot
simply average them.

RRF solution (Cormack, Clarke & Buettcher, SIGIR 2009):

    rrf_score(d) = Σ  1 / (k + rank_i(d))
                  i

where the sum is over every retriever i that returned document d,
rank_i(d) is d's 1-based rank in that list, and k is a smoothing
constant.

Key properties:
    • Score-agnostic  — only rank positions matter, not raw scores.
                        Cosine similarities and BM25 scores are never
                        compared directly.
    • Consensus bonus — a document appearing in BOTH lists earns
                        contributions from both terms.  A doc at rank 10
                        in each list (score ≈ 0.0143 + 0.0143 = 0.0286)
                        beats a doc at rank 1 in only one list (≈ 0.0164).
                        This rewards multi-evidence agreement.
    • Robust k=60     — Cormack's experiments across 20+ TREC benchmarks
                        showed k=60 is optimal across diverse corpus sizes
                        and result-list lengths.  k < 60 over-weights
                        top-ranked docs; k > 60 under-differentiates.
    • No calibration  — unlike weighted linear combination
                        (α·faiss_score + β·bm25_score), RRF needs no
                        per-corpus weight tuning.

─────────────────────────────────────────────────────────────────────────────
RETRIEVAL PIPELINE
─────────────────────────────────────────────────────────────────────────────

                        ┌─────────────────┐
                        │   Query string  │
                        └────────┬────────┘
                                 │ embed once
                                 ▼
                   ┌─────────────────────────┐
                   │   get_query_embedding() │  (text-embedding-3-small)
                   └──────┬──────────────────┘
                           │ query_vector (reused)
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌──────────────────┐      ┌──────────────────┐
   │  VectorStore     │      │  BM25Store       │
   │  .search(k=K)    │      │  .search(k=K)    │
   │  (cosine ANN)    │      │  (keyword TF-IDF)│
   └────────┬─────────┘      └────────┬─────────┘
            │ top-K chunks            │ top-K chunks
            │ with FAISS rank         │ with BM25 rank
            └────────────┬────────────┘
                         ▼
              ┌──────────────────────┐
              │   RRF Fusion         │
              │   score(d) =         │
              │   Σ 1/(rank_i(d)+60) │
              └──────────┬───────────┘
                         │ sorted by rrf_score desc
                         ▼
              ┌──────────────────────┐
              │  Top-N fused chunks  │  (default N=5)
              └──────────────────────┘

─────────────────────────────────────────────────────────────────────────────

Config (from .env):
    HYBRID_K        — candidates per retriever  (default: 10)
    HYBRID_TOP_N    — final results returned    (default: 5)
    HYBRID_RRF_K    — RRF smoothing constant    (default: 60)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Union

from dotenv import load_dotenv

from app.ingestion.embedder import EmbeddedChunk, get_query_embedding
from app.retrieval.bm25_store import BM25SearchResult, BM25Store
from app.retrieval.vector_store import VectorSearchResult, VectorStore

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_K: int = 10        # candidates fetched from each retriever
DEFAULT_TOP_N: int = 5     # final chunks returned after fusion
DEFAULT_RRF_K: int = 60    # RRF smoothing constant (Cormack 2009)

# Union type for results coming from either retriever
AnySearchResult = Union[VectorSearchResult, BM25SearchResult]


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class HybridSearchResult:
    """A single fused result from the hybrid retrieval pipeline.

    Attributes:
        chunk:         The retrieved ``EmbeddedChunk`` with text + metadata.
        rrf_score:     Combined Reciprocal Rank Fusion score.  Higher
                       is better; useful for ordering but not absolute.
        faiss_rank:    1-based rank from the FAISS vector search, or
                       ``None`` if the chunk did not appear in FAISS results.
        bm25_rank:     1-based rank from the BM25 keyword search, or
                       ``None`` if the chunk did not appear in BM25 results.
        final_rank:    1-based position in the fused result list.
        sources:       Set of retriever names that returned this chunk
                       (``{"faiss"}``, ``{"bm25"}``, or ``{"faiss","bm25"}``).
    """

    chunk: EmbeddedChunk
    rrf_score: float
    faiss_rank: int | None
    bm25_rank: int | None
    final_rank: int
    sources: set[str] = field(default_factory=set)

    def __repr__(self) -> str:
        """Return a compact string for logging."""
        return (
            f"HybridResult(rank={self.final_rank}, "
            f"rrf={self.rrf_score:.5f}, "
            f"faiss={self.faiss_rank}, bm25={self.bm25_rank}, "
            f"sources={self.sources}, "
            f"id={self.chunk.metadata.get('chunk_id')!r})"
        )


# ── Core RRF algorithm ────────────────────────────────────────────────────────


def _compute_rrf_scores(
    faiss_results: list[VectorSearchResult],
    bm25_results: list[BM25SearchResult],
    rrf_k: int,
) -> dict[str, dict]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Iterates through both result lists, accumulates RRF contributions
    per unique ``chunk_id``, and returns a dict keyed by chunk_id.

    The RRF formula applied here is:
        score(d) = Σ_i  1 / (k + rank_i(d))

    Args:
        faiss_results: Ranked list from the vector store.
        bm25_results:  Ranked list from the BM25 store.
        rrf_k:         Smoothing constant (default 60).

    Returns:
        Dict mapping ``chunk_id → {"chunk": ..., "rrf_score": float,
        "faiss_rank": int|None, "bm25_rank": int|None, "sources": set}``.
    """
    scores: dict[str, dict] = {}

    def _upsert(
        result: AnySearchResult,
        source: str,
        rank: int,
    ) -> None:
        """Add or update the RRF accumulator for a single result."""
        chunk_id: str = result.chunk.metadata.get("chunk_id", "")
        if not chunk_id:
            # Chunks without IDs cannot be deduplicated safely — skip
            logger.warning(
                "Chunk missing 'chunk_id' in metadata — skipping: %s",
                result.chunk.metadata,
            )
            return

        contribution = 1.0 / (rrf_k + rank)

        if chunk_id not in scores:
            scores[chunk_id] = {
                "chunk": result.chunk,
                "rrf_score": 0.0,
                "faiss_rank": None,
                "bm25_rank": None,
                "sources": set(),
            }

        scores[chunk_id]["rrf_score"] += contribution
        scores[chunk_id]["sources"].add(source)

        # Record the rank from each source (first write wins — they're unique)
        if source == "faiss":
            scores[chunk_id]["faiss_rank"] = rank
        elif source == "bm25":
            scores[chunk_id]["bm25_rank"] = rank

    for result in faiss_results:
        _upsert(result, "faiss", result.rank)

    for result in bm25_results:
        _upsert(result, "bm25", result.rank)

    return scores


def reciprocal_rank_fusion(
    faiss_results: list[VectorSearchResult],
    bm25_results: list[BM25SearchResult],
    top_n: int = DEFAULT_TOP_N,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[HybridSearchResult]:
    """Fuse two ranked lists with RRF and return the top-N results.

    This is a pure function (no I/O, no side effects) so it can be
    tested in complete isolation from the retriever implementations.

    Args:
        faiss_results: Output of ``VectorStore.search(query, k=K)``.
        bm25_results:  Output of ``BM25Store.search(query, k=K)``.
        top_n:         Number of fused results to return.
        rrf_k:         RRF smoothing constant.

    Returns:
        List of ``HybridSearchResult`` sorted by descending ``rrf_score``,
        length ≤ ``top_n``.  Empty if both input lists are empty.
    """
    if not faiss_results and not bm25_results:
        logger.warning("reciprocal_rank_fusion called with two empty lists.")
        return []

    raw = _compute_rrf_scores(faiss_results, bm25_results, rrf_k)

    # Sort by descending RRF score; break ties by chunk_id (deterministic)
    sorted_entries = sorted(
        raw.values(),
        key=lambda e: (-e["rrf_score"], e["chunk"].metadata.get("chunk_id", "")),
    )

    fused: list[HybridSearchResult] = []
    for final_rank, entry in enumerate(sorted_entries[:top_n], start=1):
        fused.append(
            HybridSearchResult(
                chunk=entry["chunk"],
                rrf_score=entry["rrf_score"],
                faiss_rank=entry["faiss_rank"],
                bm25_rank=entry["bm25_rank"],
                final_rank=final_rank,
                sources=entry["sources"],
            )
        )

    logger.debug(
        "RRF fusion: %d FAISS + %d BM25 → %d unique → top %d returned",
        len(faiss_results),
        len(bm25_results),
        len(raw),
        len(fused),
    )
    return fused


# ── HybridRetriever class ─────────────────────────────────────────────────────


class HybridRetriever:
    """End-to-end hybrid retrieval: embed → FAISS + BM25 → RRF fusion.

    Wraps a ``VectorStore`` and a ``BM25Store`` and exposes a single
    ``search()`` method that handles query embedding, parallel retrieval,
    and RRF fusion in one call.

    Typical usage::

        retriever = HybridRetriever(vector_store, bm25_store)
        results = retriever.search("What was the revenue in Q3 2023?")
        for r in results:
            print(r.final_rank, r.rrf_score, r.chunk.metadata["chunk_id"])
            print(r.sources)          # {"faiss", "bm25"} or just one
            print(r.chunk.text[:120])
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        k: int | None = None,
        top_n: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        """Initialise the hybrid retriever.

        Args:
            vector_store: A built (or loaded) ``VectorStore`` instance.
            bm25_store:   A built (or loaded) ``BM25Store`` instance.
            k:            Candidates fetched from each retriever.
                          Overrides ``HYBRID_K`` env var (default 10).
            top_n:        Final fused results to return.
                          Overrides ``HYBRID_TOP_N`` env var (default 5).
            rrf_k:        RRF smoothing constant.
                          Overrides ``HYBRID_RRF_K`` env var (default 60).

        Raises:
            ValueError: If ``vector_store`` or ``bm25_store`` is not built.
        """
        if not vector_store.is_built:
            raise ValueError(
                "VectorStore is not built. Call build_index() or load() first."
            )
        if not bm25_store.is_built:
            raise ValueError(
                "BM25Store is not built. Call build() or load() first."
            )

        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._k = k or int(os.getenv("HYBRID_K", str(DEFAULT_K)))
        self._top_n = top_n or int(os.getenv("HYBRID_TOP_N", str(DEFAULT_TOP_N)))
        self._rrf_k = rrf_k or int(os.getenv("HYBRID_RRF_K", str(DEFAULT_RRF_K)))

        logger.info(
            "HybridRetriever ready — k=%d, top_n=%d, rrf_k=%d, "
            "vector_store_size=%d, bm25_store_size=%d",
            self._k,
            self._top_n,
            self._rrf_k,
            self._vector_store.size,
            self._bm25_store.size,
        )

    def search(
        self,
        query: str,
        top_n: int | None = None,
    ) -> list[HybridSearchResult]:
        """Run hybrid search and return fused top-N results.

        Embeds the query once, runs both retrievers with the same vector,
        then fuses the ranked lists with RRF.

        Args:
            query:  Natural-language query string.
            top_n:  Override the instance-level ``top_n`` for this call.

        Returns:
            List of ``HybridSearchResult`` sorted by descending RRF score.
            Length ≤ ``top_n`` (or ``self._top_n`` if not overridden).

        Raises:
            ValueError:     If ``query`` is empty.
            EnvironmentError: If ``OPENAI_API_KEY`` is not set.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query string must not be empty.")

        effective_top_n = top_n or self._top_n

        logger.info(
            "Hybrid search: '%s…' (k=%d per retriever, top_n=%d)",
            query[:60],
            self._k,
            effective_top_n,
        )

        # ── Step 1: embed the query once ─────────────────────────────────────
        query_vector: list[float] = get_query_embedding(query)

        # ── Step 2: FAISS search (pass pre-computed vector — no 2nd API call)
        faiss_results: list[VectorSearchResult] = self._vector_store.search(
            query=query,
            k=self._k,
            query_vector=query_vector,
        )
        logger.debug("FAISS returned %d results", len(faiss_results))

        # ── Step 3: BM25 search (keyword-only — no embedding needed) ─────────
        bm25_results: list[BM25SearchResult] = self._bm25_store.search(
            query=query,
            k=self._k,
        )
        logger.debug("BM25 returned %d results", len(bm25_results))

        # ── Step 4: RRF fusion ─────────────────────────────────────────────
        fused = reciprocal_rank_fusion(
            faiss_results=faiss_results,
            bm25_results=bm25_results,
            top_n=effective_top_n,
            rrf_k=self._rrf_k,
        )

        # ── Log retrieval diagnostics ──────────────────────────────────────
        both_sources = sum(1 for r in fused if len(r.sources) == 2)
        logger.info(
            "Hybrid search complete: %d results | "
            "%d from both retrievers | "
            "top chunk: %s",
            len(fused),
            both_sources,
            fused[0].chunk.metadata.get("chunk_id") if fused else "none",
        )

        return fused

    def __repr__(self) -> str:
        """Return a compact string representation."""
        return (
            f"HybridRetriever(k={self._k}, top_n={self._top_n}, "
            f"rrf_k={self._rrf_k}, "
            f"vector_store_size={self._vector_store.size}, "
            f"bm25_store_size={self._bm25_store.size})"
        )