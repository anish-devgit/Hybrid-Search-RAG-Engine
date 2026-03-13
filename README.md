# Hybrid Search RAG Engine

Production-grade Retrieval-Augmented Generation API for long-document QA.
Combines **FAISS dense search** and **BM25 keyword search** fused with
**Reciprocal Rank Fusion**, powered by GPT-4o.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                               │
│                                                                         │
│  PDF/DOCX/TXT ──► loader.py ──► chunker.py ──► embedder.py             │
│                   (PageRecord)  (SemanticChunker  (OpenAI               │
│                                  95th-pct split)   text-emb-3-small)    │
│                                       │                                 │
│                          ┌────────────┴────────────┐                   │
│                          ▼                         ▼                   │
│                   vector_store.py           bm25_store.py              │
│                   (FAISS IndexFlatIP)        (BM25Okapi)               │
│                   data/vector_store/         data/bm25_store/          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                                  │
│                                                                         │
│  Question ──► get_query_embedding()                                     │
│                    │ (one API call)                                      │
│         ┌──────────┴──────────┐                                         │
│         ▼                     ▼                                         │
│   VectorStore             BM25Store                                     │
│   .search(k=10)           .search(k=10)                                 │
│   cosine ANN              token TF-IDF                                  │
│         └──────────┬──────────┘                                         │
│                    ▼                                                     │
│          reciprocal_rank_fusion()                                        │
│          score(d) = Σ 1/(rank_i(d) + 60)                               │
│                    │                                                     │
│                    ▼                                                     │
│            top-5 fused chunks                                            │
│                    │                                                     │
│                    ▼                                                     │
│           RAGGenerator.generate()                                        │
│           GPT-4o · temp=0 · strict citation prompt                     │
│                    │                                                     │
│                    ▼                                                     │
│   {"answer": "...", "sources": [...], "confidence_score": 0.94}        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| Embedding | OpenAI text-embedding-3-small | via openai 1.75.0 |
| Generation | GPT-4o (temp=0) | via langchain-openai 1.1.11 |
| Dense index | FAISS IndexFlatIP / IVFFlat | faiss-cpu 1.9.0 |
| Sparse index | BM25Okapi (k1=1.5, b=0.75) | rank-bm25 0.2.2 |
| Chunking | SemanticChunker (95th-pct) | langchain-experimental 0.3.4 |
| Chain | LCEL RunnableSequence | langchain-core 1.2.18 |
| API | FastAPI + uvicorn | 0.115.14 / 0.34.3 |
| Validation | Pydantic v2 | 2.11.4 |
| Retry | tenacity | 9.1.4 |

---

## Quickstart

### 1. Clone and install

```bash
git clone <repo>
cd rag-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Run

```bash
uvicorn app.main:app --reload --port 8000
```

OpenAPI docs at `http://localhost:8000/docs`.

### 4. Docker (recommended for production)

```bash
cp .env.example .env        # fill in OPENAI_API_KEY
docker compose up --build -d
docker compose logs -f
docker compose down         # stop (data volume preserved)
docker compose down -v      # stop + wipe all index data
```

---

## API Reference

### POST /ingest

Upload a PDF, DOCX, or TXT file. Additive — each call accumulates into the
same corpus without replacing prior documents.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@annual_report.pdf"
```

```json
{
  "doc_id": "annual_report.pdf",
  "filename": "annual_report.pdf",
  "pages_loaded": 47,
  "chunks_indexed": 112,
  "message": "'annual_report.pdf' ingested successfully. 47 pages → 112 chunks indexed."
}
```

---

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the EBITDA margin in Q3 2023?"}'
```

```json
{
  "question": "What was the EBITDA margin in Q3 2023?",
  "answer": "The EBITDA margin in Q3 2023 was 18.5%, driven by operational efficiency improvements [Source: annual_report.pdf, page 15].",
  "sources": [
    {"filename": "annual_report.pdf", "page_num": 15, "chunk_id": "annual_report.pdf::p15::c2"}
  ],
  "confidence_score": 0.94,
  "can_answer": true,
  "model": "gpt-4o",
  "retrieved_chunks": []
}
```

Include retrieved chunks for debugging retrieval quality:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was revenue?", "top_n": 5, "include_chunks": true}'
```

When the question is outside the corpus:

```json
{
  "answer": "I cannot find this in the provided documents",
  "sources": [],
  "can_answer": false
}
```

---

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "vector_store_size": 1124,
  "bm25_store_size": 1124,
  "timestamp": "2024-01-15T14:32:07.114"
}
```

`status` is `"degraded"` when no documents have been ingested or when
the two index sizes are out of sync.

---

## Benchmark Results

Hit Rate@5 on a 20-query synthetic corpus (30 chunks, 5 topics):

```
────────────────────────────────────────────────────────────────────────────────
                       RETRIEVAL BENCHMARK — HIT RATE @ 5
────────────────────────────────────────────────────────────────────────────────
  HYBRID | BM25 | QUERY
────────────────────────────────────────────────────────────────────────────────
  HIT  | BM25=HIT  | Q3 2023 revenue 4.2 billion dollars           finance
  HIT  | BM25=HIT  | EBITDA margin 18.5 percent                    finance
  HIT  | BM25=HIT  | CRISPR-Cas9 genomic editing                   bio
  HIT  | BM25=HIT  | AlphaFold protein structure prediction        bio
  HIT  | BM25=HIT  | GDPR Article 17 right to erasure              legal
  HIT  | BM25=HIT  | Kubernetes microservices orchestration        tech
  HIT  | BM25=HIT  | CUDA GPU matrix multiplication                tech
  HIT  | BM25=HIT  | Gutenberg printing press 1440                 history
  HIT  | BM25=HIT  | messenger RNA spike protein antibodies        bio
  HIT  | BM25=HIT  | force majeure contractual impossibility       legal
  HIT  | BM25=MISS | How did company earnings perform last quarter finance  ← hybrid wins
  HIT  | BM25=HIT  | profitability efficiency measured by EBITDA   finance
  HIT  | BM25=HIT  | CRISPR bacterial gene editing technology      bio
  HIT  | BM25=HIT  | Predicting how proteins fold in 3D            bio
  HIT  | BM25=HIT  | Personal data deletion rights (European law)  legal
  HIT  | BM25=HIT  | Container scheduling across cloud infra       tech
  HIT  | BM25=HIT  | Parallel computation on graphics hardware     tech
  HIT  | BM25=HIT  | How knowledge spread before the internet      history
  MISS | BM25=MISS | Vaccine technology using genetic instructions  bio
  HIT  | BM25=HIT  | Contract clause excusing performance          legal
────────────────────────────────────────────────────────────────────────────────
  Hybrid=95% (19/20)   BM25-only=90% (18/20)   FAISS-only=85% (17/20)
  Hybrid gain: +1 vs BM25, +2 vs FAISS
────────────────────────────────────────────────────────────────────────────────
```

The hybrid-exclusive hit ("How did company earnings perform last quarter" →
revenue chunk) demonstrates the RRF value: FAISS finds it via semantic
similarity while BM25 misses due to zero lexical overlap with "Q3 revenue".

Run the benchmark:

```bash
pytest tests/test_retrieval.py -v -s
```

---

## Project Structure

```
rag-engine/
├── app/
│   ├── main.py                 # FastAPI app: /ingest, /query, /health
│   ├── schemas.py              # Pydantic v2 request/response models
│   ├── ingestion/
│   │   ├── loader.py           # PDF / DOCX / TXT → PageRecord list
│   │   ├── chunker.py          # SemanticChunker → ChunkRecord list
│   │   └── embedder.py         # Batch embed → EmbeddedChunk list
│   ├── retrieval/
│   │   ├── vector_store.py     # FAISS index (build, search, save, load)
│   │   ├── bm25_store.py       # BM25Okapi index (build, search, save, load)
│   │   └── hybrid.py           # RRF fusion + HybridRetriever
│   └── generation/
│       └── llm.py              # LCEL chain: prompt | GPT-4o | citation parser
├── data/
│   ├── vector_store/           # faiss.index + chunks.pkl (auto-created)
│   └── bm25_store/             # bm25.pkl (auto-created)
├── tests/
│   └── test_retrieval.py       # HR@5 benchmark: hybrid vs BM25 vs FAISS
├── Dockerfile                  # Multi-stage build (builder + runtime)
├── docker-compose.yml          # Single-service compose with named volume
├── .env.example                # Config template
├── requirements.txt            # Pinned dependencies
└── README.md
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key |
| `GENERATION_MODEL` | `gpt-4o` | LLM for answer generation |
| `GENERATION_TEMPERATURE` | `0` | Sampling temperature |
| `GENERATION_MAX_TOKENS` | `1024` | Max tokens in generated answer |
| `GENERATION_TIMEOUT` | `60` | OpenAI request timeout (seconds) |
| `VECTOR_STORE_DIR` | `data/vector_store` | FAISS persistence path |
| `BM25_STORE_DIR` | `data/bm25_store` | BM25 persistence path |
| `HYBRID_K` | `10` | Candidates per retriever |
| `HYBRID_TOP_N` | `5` | Final chunks passed to LLM |
| `HYBRID_RRF_K` | `60` | RRF smoothing constant |
| `BM25_K1` | `1.5` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 document length normalisation |
| `SEMANTIC_THRESHOLD_PERCENTILE` | `95` | Cosine distance percentile for splits |
| `LOG_LEVEL` | `info` | debug / info / warning / error |
| `APP_VERSION` | `1.0.0` | Version string in /health |

---

## Design Notes

**Why `--workers 1`?** FAISS's C++ index is not fork-safe. Scale horizontally
with multiple containers behind a load balancer instead of multiple workers
per container.

**Why semantic chunking?** Fixed 512-token windows slice sentences mid-thought.
`SemanticChunker` detects topic-shift boundaries via cosine distance spikes,
producing one-complete-idea chunks. The LLM receives coherent passages,
not sentence fragments.

**Why BM25 alongside FAISS?** Dense embeddings smear rare tokens —
"EBITDA", "CRISPR-Cas9", "Article 17" map to broad semantic regions shared
by adjacent-but-wrong terms. BM25's exact token matching catches them
precisely. RRF promotes chunks both retrievers agree on, the strongest
multi-evidence signal for relevance.

**Why the sentinel string?** Calibrating a confidence threshold requires
labelled queries. The sentinel `"I cannot find this in the provided documents"`
is model-enforced via the system prompt — deterministic, testable, and
requires no per-deployment calibration.

---

## License

MIT