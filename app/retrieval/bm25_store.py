"""
app/retrieval/bm25_store.py
────────────────────────────
BM25 sparse keyword index for the Hybrid Search RAG Engine.

Responsibilities
────────────────
1. ``build()``   — tokenise all chunks and populate a ``BM25Okapi`` index.
2. ``search()``  — tokenise a query and return the top-k chunks ranked by
                   BM25 score.
3. ``save()``    — pickle the BM25 object + chunk list to disk.
4. ``load()``    — restore a saved ``BM25Store`` from disk.

Why BM25 alongside FAISS?
──────────────────────────
FAISS (dense semantic search) excels at paraphrase matching — it retrieves
chunks that mean the same thing even with completely different words.
BM25 (sparse keyword search) excels at exact-term matching — it reliably
retrieves chunks that contain the precise terms the user typed, especially:

  • Proper nouns:   "GPT-4o", "Anthropic", "EBITDA"  — embeddings smear
                    these across the semantic space; BM25 pins them exactly.
  • Rare terms:     Technical jargon, product codes, regulation numbers.
  • Negations:      "NOT compliant" vs "compliant" look similar in embedding
                    space; BM25 distinguishes them by term presence.
  • Numbers:        "revenue of $4.2B in Q3 2023" — hard to embed precisely.

Combining both retrievers via Reciprocal Rank Fusion (Step 7) gets the best
of both worlds: semantic coverage + lexical precision.

BM25Okapi parameters
─────────────────────
  k1 = 1.5   (term frequency saturation)
              Controls how much repeated occurrences boost the score.
              k1=1.5 is the standard corpus default.
              Lower k1 → faster saturation (good for short chunks).

  b  = 0.75  (document length normalisation)
              Penalises long documents for containing more terms by chance.
              b=0.75 is Okapi BM25's canonical value.

  epsilon = 0.25  (IDF smoothing floor)
              Prevents negative IDF for very common terms.

Tokenisation
─────────────
A lightweight regex tokeniser is used instead of NLTK/spaCy to keep the
dependency footprint small.  It lowercases text, extracts alphanumeric
runs of length >= 2.  Stop-word removal is intentionally skipped: BM25's
IDF naturally down-weights high-frequency terms.

Persistence layout
───────────────────
``<index_dir>/bm25.pkl``  — BM25Okapi + chunk list, pickle protocol 5

Config (from .env):
    BM25_STORE_DIR   — default: data/bm25_store
    BM25_K1          — default: 1.5
    BM25_B           — default: 0.75
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from app.ingestion.chunker import ChunkRecord
from app.ingestion.embedder import EmbeddedChunk

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_BM25_FILENAME: str = "bm25.pkl"
_DEFAULT_K1: float = 1.5
_DEFAULT_B: float = 0.75
_DEFAULT_EPSILON: float = 0.25

# Matches alphanumeric runs of 2+ characters after lowercasing
_TOKEN_PATTERN: re.Pattern = re.compile(r"[a-z0-9]{2,}")


# ── Result type ───────────────────────────────────────────────────────────────


class BM25SearchResult(NamedTuple):
    """A single result from a BM25 keyword search.

    Attributes:
        chunk:  The retrieved chunk object (carries ``.text`` + ``.metadata``).
        score:  Raw BM25Okapi relevance score (non-negative float).
                Higher is better; not normalised — use only for ranking.
        rank:   1-based position in the result list (1 = highest score).
    """

    chunk: ChunkRecord | EmbeddedChunk
    score: float
    rank: int


# ── Tokeniser ─────────────────────────────────────────────────────────────────


def tokenize(text: str) -> list[str]:
    """Tokenise a string into lowercase alphanumeric tokens of length >= 2.

    Lowercases the input, then extracts all runs of alphanumeric characters
    with length >= 2.  Single characters are discarded as noise.
    No stop-word removal — BM25's IDF naturally down-weights high-frequency
    terms without needing an explicit stop list.

    Args:
        text: Raw text string (query or document chunk).

    Returns:
        List of lowercase tokens.  Empty list if ``text`` has no
        qualifying alphanumeric content.

    Example::

        tokenize("GPT-4o achieved 92.3% on MMLU (2024)!")
        # → ['gpt', '4o', 'achieved', '92', 'on', 'mmlu', '2024']
    """
    return _TOKEN_PATTERN.findall(text.lower())


# ── BM25Store class ───────────────────────────────────────────────────────────


class BM25Store:
    """BM25Okapi sparse keyword index.

    Accepts either ``ChunkRecord`` (from the chunker) or ``EmbeddedChunk``
    (from the embedder) so the same object list used for FAISS can be fed
    directly here — no conversion needed.

    Typical usage::

        store = BM25Store()
        store.build(embedded_chunks)
        store.save("data/bm25_store")

        # after restart:
        store = BM25Store.load("data/bm25_store")
        results = store.search("EBITDA margin Q3 2023", k=10)
        for r in results:
            print(r.rank, r.score, r.chunk.metadata["chunk_id"])
    """

    def __init__(self) -> None:
        """Initialise an empty BM25Store.

        Call ``build()`` to populate, or ``BM25Store.load()`` to restore.
        """
        self._bm25: BM25Okapi | None = None
        self._chunks: list[ChunkRecord | EmbeddedChunk] = []
        self._k1: float = float(os.getenv("BM25_K1", str(_DEFAULT_K1)))
        self._b: float = float(os.getenv("BM25_B", str(_DEFAULT_B)))

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, chunks: list[ChunkRecord | EmbeddedChunk]) -> None:
        """Build the BM25 index from a list of text chunks.

        Tokenises every chunk's ``.text`` attribute and passes the
        resulting corpus to ``BM25Okapi``.  Chunks that produce zero
        tokens receive a ``["__empty__"]`` placeholder so the index
        stays aligned with the chunk list (index ``i`` always corresponds
        to ``self._chunks[i]``).

        Args:
            chunks: Non-empty list of chunk objects.

        Raises:
            ValueError: If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError("build() received an empty chunk list.")

        logger.info(
            "Building BM25 index: %d chunks (k1=%.2f, b=%.2f)",
            len(chunks),
            self._k1,
            self._b,
        )

        tokenised_corpus: list[list[str]] = []
        empty_count = 0

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            if not tokens:
                empty_count += 1
                tokens = ["__empty__"]   # keeps index aligned with chunk list
            tokenised_corpus.append(tokens)

        if empty_count:
            logger.warning(
                "%d chunk(s) produced no tokens — placeholders inserted "
                "to preserve index alignment.",
                empty_count,
            )

        self._bm25 = BM25Okapi(
            tokenised_corpus,
            k1=self._k1,
            b=self._b,
            epsilon=_DEFAULT_EPSILON,
        )
        self._chunks = list(chunks)

        total_tokens = sum(len(t) for t in tokenised_corpus)
        avg_tokens = total_tokens / len(tokenised_corpus)
        logger.info(
            "BM25 index built: %d docs, avg %.1f tokens/doc, vocab_size=%d",
            self._bm25.corpus_size,
            avg_tokens,
            len(self._bm25.idf),
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[BM25SearchResult]:
        """Return the top-k keyword-matching chunks for a query.

        Tokenises ``query`` with the same tokeniser used at index time,
        scores every document via BM25Okapi, then returns the highest-
        scoring documents.  Documents scoring exactly 0 are excluded
        because they share no query terms and would only add noise to
        the hybrid fusion step.

        Args:
            query: Natural-language query string.
            k:     Maximum number of results to return.

        Returns:
            List of ``BM25SearchResult``, sorted by descending BM25 score.
            May be shorter than ``k`` if fewer documents score above zero,
            or if the query tokenises to nothing (returns ``[]``).

        Raises:
            RuntimeError: If the index has not been built or loaded.
            ValueError:   If ``k`` is less than 1.
        """
        if self._bm25 is None:
            raise RuntimeError(
                "BM25Store has no index. Call build() or load() first."
            )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")

        query_tokens = tokenize(query)
        if not query_tokens:
            logger.warning(
                "Query '%s' produced no tokens — returning empty results.",
                query,
            )
            return []

        scores: np.ndarray = self._bm25.get_scores(query_tokens)

        # Descending sort: negate scores and use argsort (O(n log n))
        sorted_indices = np.argsort(-scores)

        results: list[BM25SearchResult] = []
        rank = 1
        for idx in sorted_indices:
            if rank > k:
                break
            score = float(scores[idx])
            if score <= 0.0:
                # Remaining scores are also zero — short-circuit
                break
            results.append(
                BM25SearchResult(
                    chunk=self._chunks[int(idx)],
                    score=score,
                    rank=rank,
                )
            )
            rank += 1

        logger.debug(
            "BM25 search '%s…' (%d tokens): %d results, top_score=%.4f",
            query[:50],
            len(query_tokens),
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str | Path | None = None) -> Path:
        """Persist the BM25 index and chunk list to a single pickle file.

        The entire ``BM25Okapi`` object (including all precomputed IDF,
        TF, and length statistics) plus the chunk list are pickled together
        so a ``load()`` call returns an instantly-searchable store with no
        recomputation.

        Args:
            directory: Target directory.  Defaults to ``BM25_STORE_DIR``
                       env var, then ``data/bm25_store``.

        Returns:
            The resolved ``Path`` of the directory that was written.

        Raises:
            RuntimeError: If ``build()`` has not been called yet.
            OSError:      If the directory or file cannot be written.
        """
        if self._bm25 is None:
            raise RuntimeError(
                "Cannot save: index has not been built. Call build() first."
            )

        save_dir = Path(
            directory or os.getenv("BM25_STORE_DIR", "data/bm25_store")
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        pkl_path = save_dir / _BM25_FILENAME
        payload = {
            "bm25": self._bm25,
            "chunks": self._chunks,
            "k1": self._k1,
            "b": self._b,
        }

        with pkl_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=5)

        size_kb = pkl_path.stat().st_size / 1024
        logger.info(
            "BM25 index saved → %s (%d docs, %.1f KB)",
            pkl_path,
            len(self._chunks),
            size_kb,
        )
        return save_dir

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "BM25Store":
        """Restore a BM25Store from a pickle file produced by ``save()``.

        Args:
            directory: Directory containing ``bm25.pkl``.  Defaults to
                       ``BM25_STORE_DIR`` env var, then ``data/bm25_store``.

        Returns:
            A fully restored ``BM25Store`` ready for ``search()``.

        Raises:
            FileNotFoundError: If ``bm25.pkl`` is absent.
            KeyError:          If the pickle payload is missing expected
                               keys (indicates a corrupted or incompatible
                               file).
        """
        load_dir = Path(
            directory or os.getenv("BM25_STORE_DIR", "data/bm25_store")
        )
        pkl_path = load_dir / _BM25_FILENAME

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"BM25 index file not found: '{pkl_path}'. "
                "Has build() + save() been run?"
            )

        with pkl_path.open("rb") as fh:
            payload: dict = pickle.load(fh)

        required_keys = {"bm25", "chunks", "k1", "b"}
        missing = required_keys - payload.keys()
        if missing:
            raise KeyError(
                f"BM25 pickle payload is missing keys: {missing}. "
                "The file may be corrupted or from an incompatible version."
            )

        store = cls()
        store._bm25 = payload["bm25"]
        store._chunks = payload["chunks"]
        store._k1 = payload["k1"]
        store._b = payload["b"]

        logger.info(
            "BM25 index loaded ← %s (%d docs, vocab=%d)",
            pkl_path,
            len(store._chunks),
            len(store._bm25.idf),
        )
        return store

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Return the number of documents currently indexed.

        Returns:
            0 if the index has not been built yet.
        """
        return self._bm25.corpus_size if self._bm25 is not None else 0

    @property
    def is_built(self) -> bool:
        """Return True if the index is ready for search.

        Returns:
            Boolean indicating whether ``build()`` or ``load()`` succeeded.
        """
        return self._bm25 is not None

    @property
    def vocab_size(self) -> int:
        """Return the number of unique terms in the index vocabulary.

        Returns:
            0 if the index has not been built yet.
        """
        return len(self._bm25.idf) if self._bm25 is not None else 0

    def __repr__(self) -> str:
        """Return a compact string representation for logging."""
        return (
            f"BM25Store(size={self.size}, vocab={self.vocab_size}, "
            f"k1={self._k1}, b={self._b})"
        )