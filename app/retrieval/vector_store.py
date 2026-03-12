"""
app/retrieval/vector_store.py
──────────────────────────────
FAISS dense vector store for the Hybrid Search RAG Engine.

Responsibilities
────────────────
1. ``build_index()``  — accept ``EmbeddedChunk`` objects, normalise their
   vectors, and populate a FAISS index.
2. ``search()``       — embed a query, search the index, and return the top-k
   chunks with their cosine-similarity scores.
3. ``save()``         — persist the FAISS index + chunk metadata to disk so
   the server can restart without re-embedding.
4. ``load()``         — restore a previously saved ``VectorStore`` from disk.

Index strategy — two modes
───────────────────────────
• **Flat (exact)** — ``IndexFlatIP``.  Used when ``n_vectors < FLAT_THRESHOLD``
  (default 10 000).  Exact cosine search, O(n) scan, no training needed.
  For 500-page docs chunked at ~2 chunks/page → ~1 000 chunks — flat is
  perfectly fast (< 5 ms per query on CPU).

• **IVFFlat (approximate)** — ``IndexIVFFlat``.  Kicks in above the threshold.
  Partitions the space into ``nlist`` Voronoi cells and searches ``nprobe``
  of them.  Typical recall@10 ≈ 0.97 with nprobe = sqrt(nlist).

Cosine similarity via inner product
─────────────────────────────────────
All vectors are L2-normalised before being added to the index.  After
normalisation, ``inner_product(u, v) == cosine_similarity(u, v)`` because
both vectors lie on the unit sphere.  ``IndexFlatIP`` (inner product) then
returns scores in [−1, +1] where 1 = identical.

Persistence layout
───────────────────
``<index_dir>/
    faiss.index      ← the FAISS binary index file
    chunks.pkl       ← list[EmbeddedChunk], pickle protocol 5``

Config (from .env):
    OPENAI_API_KEY           — for query-time embedding
    EMBEDDING_MODEL          — default: text-embedding-3-small
    EMBEDDING_DIMENSIONS     — default: 1536
    VECTOR_STORE_DIR         — default: data/vector_store
    VECTOR_FLAT_THRESHOLD    — default: 10000  (switch to IVF above this)
    VECTOR_NLIST             — default: 100    (IVF Voronoi cells)
    VECTOR_NPROBE            — default: 10     (IVF cells searched per query)
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import faiss
import numpy as np
from dotenv import load_dotenv

from app.ingestion.embedder import DEFAULT_DIMENSIONS, EmbeddedChunk, get_query_embedding

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_FLAT_THRESHOLD: int = 10_000   # use exact search below this
_DEFAULT_NLIST: int = 100       # IVF Voronoi cells
_DEFAULT_NPROBE: int = 10       # IVF cells probed per query
_INDEX_FILENAME: str = "faiss.index"
_CHUNKS_FILENAME: str = "chunks.pkl"


# ── Result type ───────────────────────────────────────────────────────────────


class VectorSearchResult(NamedTuple):
    """A single result from a vector store search.

    Attributes:
        chunk:       The retrieved ``EmbeddedChunk`` (text + metadata).
        score:       Cosine similarity score in [−1, 1].  Higher is better.
        rank:        1-based position in the result list (1 = most similar).
    """

    chunk: EmbeddedChunk
    score: float
    rank: int


# ── VectorStore class ─────────────────────────────────────────────────────────


class VectorStore:
    """FAISS-backed dense vector store.

    Typical usage::

        store = VectorStore()
        store.build_index(embedded_chunks)
        store.save("data/vector_store")

        # later, after restart:
        store = VectorStore.load("data/vector_store")
        results = store.search("What is the revenue growth?", k=10)
        for r in results:
            print(r.rank, r.score, r.chunk.metadata["chunk_id"])
    """

    def __init__(self) -> None:
        """Initialise an empty VectorStore.

        Call ``build_index()`` to populate it, or ``VectorStore.load()``
        to restore from disk.
        """
        self._index: faiss.Index | None = None
        self._chunks: list[EmbeddedChunk] = []
        self._dim: int = int(os.getenv("EMBEDDING_DIMENSIONS", str(DEFAULT_DIMENSIONS)))
        self._flat_threshold: int = int(
            os.getenv("VECTOR_FLAT_THRESHOLD", str(_FLAT_THRESHOLD))
        )
        self._nlist: int = int(os.getenv("VECTOR_NLIST", str(_DEFAULT_NLIST)))
        self._nprobe: int = int(os.getenv("VECTOR_NPROBE", str(_DEFAULT_NPROBE)))

    # ── Build ────────────────────────────────────────────────────────────────

    def build_index(self, chunks: list[EmbeddedChunk]) -> None:
        """Build the FAISS index from a list of embedded chunks.

        Automatically chooses flat (exact) search for small corpora and
        IVFFlat (approximate) for large corpora based on
        ``VECTOR_FLAT_THRESHOLD``.  All vectors are L2-normalised in-place
        before being added so that inner-product == cosine similarity.

        Args:
            chunks: List of ``EmbeddedChunk`` objects from the embedder.
                    Must be non-empty and all embeddings must have the
                    same dimension.

        Raises:
            ValueError: If ``chunks`` is empty or has inconsistent
                        embedding dimensions.
        """
        if not chunks:
            raise ValueError("build_index received an empty chunk list.")

        dim = len(chunks[0].embedding)
        if any(len(c.embedding) != dim for c in chunks):
            raise ValueError(
                "All embeddings must have the same dimension. "
                "Found inconsistent sizes."
            )
        if dim != self._dim:
            logger.warning(
                "Embedding dim from chunks (%d) differs from config (%d). "
                "Using chunk dim.",
                dim,
                self._dim,
            )
            self._dim = dim

        n = len(chunks)
        logger.info("Building FAISS index: %d vectors, dim=%d", n, dim)

        # Stack all vectors into a float32 matrix and normalise
        matrix = np.array([c.embedding for c in chunks], dtype=np.float32)
        faiss.normalize_L2(matrix)   # in-place L2 normalisation

        if n < self._flat_threshold:
            logger.info(
                "Using IndexFlatIP (exact search) — %d vectors < threshold %d",
                n,
                self._flat_threshold,
            )
            index = faiss.IndexFlatIP(dim)
        else:
            # nlist must be ≤ n; cap it and warn if needed
            nlist = min(self._nlist, n // 10)
            nlist = max(nlist, 1)
            if nlist != self._nlist:
                logger.warning(
                    "Reduced nlist from %d to %d to match corpus size %d",
                    self._nlist,
                    nlist,
                    n,
                )
            logger.info(
                "Using IndexIVFFlat (approximate) — %d vectors ≥ threshold %d "
                "(nlist=%d, nprobe=%d)",
                n,
                self._flat_threshold,
                nlist,
                self._nprobe,
            )
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            # IVFFlat must be trained before adding vectors
            index.train(matrix)
            index.nprobe = self._nprobe

        index.add(matrix)

        self._index = index
        self._chunks = list(chunks)   # keep a copy for metadata lookup

        logger.info(
            "FAISS index built: %d vectors indexed (ntotal=%d)",
            n,
            self._index.ntotal,
        )

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        query_vector: list[float] | None = None,
    ) -> list[VectorSearchResult]:
        """Search the index for the top-k most similar chunks.

        Embeds ``query`` at call time (unless ``query_vector`` is provided),
        normalises the query vector, and returns the ``k`` nearest chunks
        by cosine similarity.

        Args:
            query:        Natural-language query string.
            k:            Number of results to return.
            query_vector: Optional pre-computed embedding.  When provided,
                          the embedding API call is skipped.  Useful for
                          testing and for hybrid search (reuse one embedding
                          across both retrieval systems).

        Returns:
            List of ``VectorSearchResult``, sorted by descending score
            (rank 1 = most similar).  May be shorter than ``k`` if the
            index contains fewer than ``k`` vectors.

        Raises:
            RuntimeError: If the index has not been built or loaded yet.
            ValueError:   If ``k`` is less than 1.
        """
        if self._index is None:
            raise RuntimeError(
                "VectorStore has no index. Call build_index() or load() first."
            )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")

        # Clamp k to the number of indexed vectors
        effective_k = min(k, self._index.ntotal)

        # Embed the query (or use the pre-computed vector)
        if query_vector is not None:
            vec = np.array([query_vector], dtype=np.float32)
        else:
            raw = get_query_embedding(query)
            vec = np.array([raw], dtype=np.float32)

        faiss.normalize_L2(vec)

        scores_matrix, ids_matrix = self._index.search(vec, effective_k)

        scores: list[float] = scores_matrix[0].tolist()
        ids: list[int] = ids_matrix[0].tolist()

        results: list[VectorSearchResult] = []
        for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
            if idx < 0:
                # FAISS returns −1 when fewer than k results exist
                continue
            results.append(
                VectorSearchResult(
                    chunk=self._chunks[idx],
                    score=float(score),
                    rank=rank,
                )
            )

        logger.debug(
            "Vector search '%s…': %d results, top score=%.4f",
            query[:50],
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str | Path | None = None) -> Path:
        """Persist the FAISS index and chunk list to disk.

        Creates the directory if it does not exist.  Two files are written:
            ``faiss.index`` — the FAISS binary index
            ``chunks.pkl``  — the ``list[EmbeddedChunk]`` via pickle

        Args:
            directory: Path to the output directory.  Defaults to the
                       ``VECTOR_STORE_DIR`` env var, then
                       ``data/vector_store``.

        Returns:
            The resolved ``Path`` of the directory that was written.

        Raises:
            RuntimeError: If the index has not been built yet.
            OSError:      If the directory cannot be created or the files
                          cannot be written.
        """
        if self._index is None:
            raise RuntimeError(
                "Cannot save: index has not been built. Call build_index() first."
            )

        save_dir = Path(
            directory
            or os.getenv("VECTOR_STORE_DIR", "data/vector_store")
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        index_path = save_dir / _INDEX_FILENAME
        chunks_path = save_dir / _CHUNKS_FILENAME

        faiss.write_index(self._index, str(index_path))
        logger.info("FAISS index saved → %s", index_path)

        with chunks_path.open("wb") as fh:
            pickle.dump(self._chunks, fh, protocol=5)
        logger.info(
            "Chunk list saved → %s (%d chunks)", chunks_path, len(self._chunks)
        )

        return save_dir

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "VectorStore":
        """Load a previously saved VectorStore from disk.

        Args:
            directory: Path to the directory produced by ``save()``.
                       Defaults to the ``VECTOR_STORE_DIR`` env var, then
                       ``data/vector_store``.

        Returns:
            A fully restored ``VectorStore`` ready for ``search()``.

        Raises:
            FileNotFoundError: If either ``faiss.index`` or ``chunks.pkl``
                               is missing from ``directory``.
            RuntimeError:      If the loaded index and chunk list have
                               mismatched lengths.
        """
        load_dir = Path(
            directory
            or os.getenv("VECTOR_STORE_DIR", "data/vector_store")
        )

        index_path = load_dir / _INDEX_FILENAME
        chunks_path = load_dir / _CHUNKS_FILENAME

        for p in (index_path, chunks_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"VectorStore file not found: '{p}'. "
                    "Has build_index() + save() been run?"
                )

        index = faiss.read_index(str(index_path))
        logger.info(
            "FAISS index loaded ← %s (ntotal=%d)", index_path, index.ntotal
        )

        with chunks_path.open("rb") as fh:
            chunks: list[EmbeddedChunk] = pickle.load(fh)
        logger.info(
            "Chunk list loaded ← %s (%d chunks)", chunks_path, len(chunks)
        )

        if index.ntotal != len(chunks):
            raise RuntimeError(
                f"Index/chunk mismatch: index has {index.ntotal} vectors "
                f"but chunk list has {len(chunks)} entries. "
                "The persisted files may be from different ingestion runs."
            )

        store = cls()
        store._index = index
        store._chunks = chunks
        store._dim = index.d
        logger.info(
            "VectorStore loaded: %d vectors, dim=%d", len(chunks), index.d
        )
        return store

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Return the number of vectors currently indexed.

        Returns:
            0 if the index has not been built yet.
        """
        return self._index.ntotal if self._index is not None else 0

    @property
    def is_built(self) -> bool:
        """Return True if the index has been built or loaded.

        Returns:
            Boolean indicating whether the store is ready for search.
        """
        return self._index is not None

    def __repr__(self) -> str:
        """Return a compact string representation."""
        index_type = type(self._index).__name__ if self._index else "None"
        return (
            f"VectorStore(size={self.size}, dim={self._dim}, "
            f"index_type={index_type})"
        )