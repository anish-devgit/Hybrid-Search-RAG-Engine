"""
app/generation/llm.py
──────────────────────
GPT-4o generation layer for the Hybrid Search RAG Engine.

Builds a LangChain LCEL (LangChain Expression Language) RAG chain that:

    1. Formats retrieved chunks into a numbered context block.
    2. Constructs a structured prompt with a strict system instruction.
    3. Calls GPT-4o via ``ChatOpenAI``.
    4. Parses the response and extracts citations.
    5. Returns a structured ``GenerationResult`` to the API layer.

─────────────────────────────────────────────────────────────────────────────
SYSTEM PROMPT DESIGN
─────────────────────────────────────────────────────────────────────────────

Three hard rules are enforced in the system prompt:

Rule (a) — Context-only answers
    "Answer ONLY using the information in the provided context passages."
    The LLM is explicitly forbidden from using its parametric knowledge.
    This eliminates the main failure mode of RAG: the model confidently
    answering from training data when retrieved context is sparse or
    off-topic.

Rule (b) — Mandatory source citation
    Every factual claim must end with a [Source: filename, page N] marker.
    This gives users an auditable trail back to the exact document page
    from which each claim was drawn.  The format is machine-parseable so
    the API response can return ``sources`` as a structured list.

Rule (c) — Explicit refusal when context is insufficient
    If no context passage supports an answer, the model MUST reply with
    the sentinel string "I cannot find this in the provided documents"
    rather than hallucinating.  A deterministic sentinel is far easier to
    detect programmatically than hedged hallucinations like "I believe…"

─────────────────────────────────────────────────────────────────────────────
CHAIN ARCHITECTURE  (LCEL)
─────────────────────────────────────────────────────────────────────────────

    query ──► format_context() ──► ChatPromptTemplate
                                         │
                                    ChatOpenAI (GPT-4o, temp=0)
                                         │
                                    StrOutputParser
                                         │
                                    extract_citations()
                                         │
                                    GenerationResult

Why LCEL over RetrievalQA.from_chain_type():
    • RetrievalQA is deprecated as of LangChain 0.2 and removed in 1.x.
    • LCEL chains are composable, streamable, and trivially unit-testable
      (each step is a pure function or an Runnable with a clear interface).
    • The ``|`` pipe syntax makes data flow explicit — no magic kwargs.

─────────────────────────────────────────────────────────────────────────────

Config (from .env):
    OPENAI_API_KEY        — required
    GENERATION_MODEL      — default: gpt-4o
    GENERATION_TEMPERATURE — default: 0   (deterministic for factual QA)
    GENERATION_MAX_TOKENS — default: 1024
    GENERATION_TIMEOUT    — default: 60   (seconds)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.retrieval.hybrid import HybridSearchResult

load_dotenv()
logger = logging.getLogger(__name__)

# ── Model defaults ────────────────────────────────────────────────────────────

DEFAULT_MODEL: str = "gpt-4o"
DEFAULT_TEMPERATURE: float = 0.0    # fully deterministic for factual QA
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_TIMEOUT: int = 60           # seconds before request is abandoned

# Sentinel returned when the model cannot find an answer in context
CANNOT_ANSWER_SENTINEL: str = "I cannot find this in the provided documents"

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = """\
You are a precise, factual question-answering assistant.
You have been given a set of numbered context passages retrieved from a \
document corpus.

STRICT RULES — follow all of them without exception:

1. CONTEXT ONLY: Answer ONLY using information explicitly stated in the \
provided context passages below. Do NOT use your general training knowledge, \
do NOT infer beyond what the context says, and do NOT fill gaps with \
assumptions.

2. CITE YOUR SOURCES: Every sentence that states a fact must end with a \
citation in exactly this format:
   [Source: <filename>, page <N>]
   Use the filename and page number from the passage header. If a sentence \
draws on multiple passages, include all relevant citations.

3. INSUFFICIENT CONTEXT: If the context passages do not contain enough \
information to answer the question, you MUST respond with exactly:
   "{sentinel}"
   Do not attempt a partial answer, do not apologise, do not explain what \
you do not know. Just output that exact phrase and nothing else.

4. NO FABRICATION: Never invent names, numbers, dates, or facts. \
If a number is not in the context, do not state it.

5. CONCISE AND DIRECT: Lead with the answer. Do not repeat the question. \
Do not use filler phrases like "Based on the provided context…".

Context passages:
──────────────────
{{context}}
──────────────────
""".format(sentinel=CANNOT_ANSWER_SENTINEL)

_HUMAN_PROMPT: str = "{question}"

# ── Output type ───────────────────────────────────────────────────────────────


@dataclass
class SourceCitation:
    """A single parsed source citation from the model's answer.

    Attributes:
        filename:  The source document filename (e.g. ``"report.pdf"``).
        page_num:  The 1-based page number cited.
        chunk_id:  The full ``chunk_id`` from the retrieved chunk, if
                   available (may differ from the parsed citation).
    """

    filename: str
    page_num: int
    chunk_id: str = ""

    def __str__(self) -> str:
        """Return a human-readable citation string."""
        return f"{self.filename}, page {self.page_num}"


@dataclass
class GenerationResult:
    """The complete output from one RAG query.

    Attributes:
        answer:           The model's text answer (or the cannot-answer
                          sentinel if context was insufficient).
        sources:          Deduplicated list of ``SourceCitation`` objects
                          parsed from ``[Source: …]`` markers in the answer.
        retrieved_chunks: The ``HybridSearchResult`` objects that were
                          passed as context.  Kept for downstream
                          confidence scoring and debugging.
        model:            The model name used for generation.
        can_answer:       ``True`` if the model found relevant context;
                          ``False`` if it returned the cannot-answer
                          sentinel.
    """

    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    retrieved_chunks: list[HybridSearchResult] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    can_answer: bool = True

    def __repr__(self) -> str:
        """Return a compact representation for logging."""
        return (
            f"GenerationResult(can_answer={self.can_answer}, "
            f"sources={len(self.sources)}, "
            f"answer_len={len(self.answer)})"
        )


# ── Context formatter ─────────────────────────────────────────────────────────


def format_context(chunks: list[HybridSearchResult]) -> str:
    """Render retrieved chunks into a numbered context block for the prompt.

    Each chunk is formatted as::

        [1] Source: annual_report.pdf, page 14
        The company achieved revenue of $4.2B in Q3 2023, representing
        a 23% year-over-year increase…

    The numbering allows the model to cross-reference when citing.
    The ``Source:`` header on each passage is what the model uses to
    construct its ``[Source: …]`` citations in the answer.

    Args:
        chunks: Ordered list of ``HybridSearchResult`` objects from the
                hybrid retriever (already ranked by RRF score).

    Returns:
        A single formatted string with all passages separated by blank
        lines.  Returns ``"No context available."`` if ``chunks`` is empty.
    """
    if not chunks:
        return "No context available."

    parts: list[str] = []
    for i, result in enumerate(chunks, start=1):
        meta = result.chunk.metadata
        filename = meta.get("filename", "unknown")
        page_num = meta.get("page_num", "?")
        text = result.chunk.text.strip()
        parts.append(
            f"[{i}] Source: {filename}, page {page_num}\n{text}"
        )

    return "\n\n".join(parts)


# ── Citation extractor ────────────────────────────────────────────────────────

# Matches: [Source: filename.pdf, page 12]  or  [Source: filename.pdf, page 12]
_CITATION_RE: re.Pattern = re.compile(
    r"\[Source:\s*([^\],]+?),\s*page\s*(\d+)\]",
    re.IGNORECASE,
)


def extract_citations(
    answer: str,
    chunks: list[HybridSearchResult],
) -> list[SourceCitation]:
    """Parse ``[Source: …]`` markers from the model's answer text.

    Deduplicates by ``(filename, page_num)`` pair — if the same page is
    cited multiple times it appears only once in the output.

    Also attempts to match each citation back to a ``chunk_id`` from the
    retrieved chunks so the API response can include fully-qualified IDs.

    Args:
        answer: The raw text answer from the model.
        chunks: The retrieved chunks passed as context.

    Returns:
        Deduplicated list of ``SourceCitation`` objects in the order they
        first appear in the answer.
    """
    # Build a lookup: (filename, page_num) → chunk_id
    chunk_lookup: dict[tuple[str, int], str] = {}
    for result in chunks:
        meta = result.chunk.metadata
        key = (meta.get("filename", ""), meta.get("page_num", -1))
        if key not in chunk_lookup:
            chunk_lookup[key] = meta.get("chunk_id", "")

    seen: set[tuple[str, int]] = set()
    citations: list[SourceCitation] = []

    for match in _CITATION_RE.finditer(answer):
        filename = match.group(1).strip()
        try:
            page_num = int(match.group(2))
        except ValueError:
            continue

        key = (filename, page_num)
        if key in seen:
            continue
        seen.add(key)

        chunk_id = chunk_lookup.get(key, "")
        citations.append(
            SourceCitation(filename=filename, page_num=page_num, chunk_id=chunk_id)
        )

    return citations


# ── RAG chain builder ─────────────────────────────────────────────────────────


def build_rag_chain(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
):
    """Construct the LCEL RAG generation chain.

    The chain signature is::

        chain.invoke({"context": str, "question": str}) → str

    The returned chain is a ``RunnableSequence`` that:
        1. Passes ``context`` and ``question`` into the prompt template.
        2. Sends the formatted prompt to GPT-4o.
        3. Extracts the text content from the ``AIMessage`` response.

    The chain itself is stateless and reusable across requests.  Build it
    once at application startup and call ``.invoke()`` per query.

    Args:
        model:       GPT model name.  Overrides ``GENERATION_MODEL`` env var.
        temperature: Sampling temperature.  Overrides ``GENERATION_TEMPERATURE``.
        max_tokens:  Response token limit.  Overrides ``GENERATION_MAX_TOKENS``.
        timeout:     Request timeout in seconds.  Overrides ``GENERATION_TIMEOUT``.

    Returns:
        A LangChain ``RunnableSequence`` (prompt | llm | parser).

    Raises:
        EnvironmentError: If ``OPENAI_API_KEY`` is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    resolved_model = model or os.getenv("GENERATION_MODEL", DEFAULT_MODEL)
    resolved_temp = temperature if temperature is not None else float(
        os.getenv("GENERATION_TEMPERATURE", str(DEFAULT_TEMPERATURE))
    )
    resolved_tokens = max_tokens or int(
        os.getenv("GENERATION_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))
    )
    resolved_timeout = timeout or int(
        os.getenv("GENERATION_TIMEOUT", str(DEFAULT_TIMEOUT))
    )

    logger.debug(
        "Building RAG chain: model=%s, temp=%.1f, max_tokens=%d, timeout=%ds",
        resolved_model,
        resolved_temp,
        resolved_tokens,
        resolved_timeout,
    )

    llm = ChatOpenAI(
        model_name=resolved_model,
        temperature=resolved_temp,
        max_tokens=resolved_tokens,
        request_timeout=resolved_timeout,
        openai_api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human",  _HUMAN_PROMPT),
    ])

    chain = prompt | llm | StrOutputParser()
    logger.info("RAG chain built: %s | temp=%.1f", resolved_model, resolved_temp)
    return chain


# ── High-level query interface ────────────────────────────────────────────────


class RAGGenerator:
    """Stateful wrapper around the LCEL RAG chain.

    Holds a pre-built chain and exposes ``generate()`` which accepts
    hybrid retrieval results and a query, and returns a ``GenerationResult``.

    Typical usage::

        generator = RAGGenerator()
        result = generator.generate(
            query="What was Q3 revenue?",
            chunks=hybrid_results,
        )
        print(result.answer)
        for src in result.sources:
            print(f"  ↳ {src}")
        if not result.can_answer:
            print("Model could not find relevant context.")
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ) -> None:
        """Initialise and build the RAG chain.

        Args:
            model:       GPT model name override.
            temperature: Temperature override.
            max_tokens:  Max tokens override.
            timeout:     Request timeout override.

        Raises:
            EnvironmentError: If ``OPENAI_API_KEY`` is not set.
        """
        self._chain = build_rag_chain(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        self._model = model or os.getenv("GENERATION_MODEL", DEFAULT_MODEL)

    def generate(
        self,
        query: str,
        chunks: list[HybridSearchResult],
    ) -> GenerationResult:
        """Generate an answer for ``query`` grounded in ``chunks``.

        Formats the chunks into a context block, invokes the LLM chain,
        parses citations, and detects the cannot-answer sentinel.

        Args:
            query:  The user's natural-language question.
            chunks: Top-N results from ``HybridRetriever.search()``.

        Returns:
            A ``GenerationResult`` with the answer, parsed citations,
            source chunks, and a ``can_answer`` flag.

        Raises:
            ValueError:        If ``query`` is empty.
            EnvironmentError:  If the OpenAI key is missing at call time.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        context = format_context(chunks)

        logger.info(
            "Generating answer: query='%s…', chunks=%d, context_chars=%d",
            query[:60],
            len(chunks),
            len(context),
        )

        raw_answer: str = self._chain.invoke({
            "context": context,
            "question": query,
        })

        # Detect the cannot-answer sentinel
        can_answer = CANNOT_ANSWER_SENTINEL.lower() not in raw_answer.lower()

        citations = extract_citations(raw_answer, chunks) if can_answer else []

        result = GenerationResult(
            answer=raw_answer,
            sources=citations,
            retrieved_chunks=chunks,
            model=self._model,
            can_answer=can_answer,
        )

        logger.info(
            "Generation complete: can_answer=%s, citations=%d, tokens≈%d",
            result.can_answer,
            len(result.sources),
            len(raw_answer) // 4,   # rough token estimate
        )
        return result