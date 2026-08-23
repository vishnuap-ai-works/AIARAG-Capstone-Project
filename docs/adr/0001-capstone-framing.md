# ADR 0001: Capstone Framing (Locked — M1 Deliverable)

**Status:** Locked  
**Date:** 2025-07-10

---

## 1. Locked Use Case

**System:** RAG-powered Q&A assistant for the Basecamp (37signals) Employee Handbook.

**What it does:** Employees ask natural-language questions about company policies (benefits, PTO, retirement, insurance, sabbatical, etc.) and receive grounded, cited answers drawn exclusively from the handbook corpus.

**Who it is for:** 37signals employees — especially new hires — who need fast, accurate answers to HR and policy questions without manually searching the handbook.

**Constraint:** The system answers only from the provided handbook content. If the answer is not in the corpus, it says so rather than hallucinating.

---

## 2. Architecture

```
User (Browser)
     │
     ▼
Streamlit UI  (ui/app.py)
     │  POST /query
     ▼
FastAPI  (api/main.py)
     │
     ├── Embeddings  (src/rag/embeddings.py)
     │     └── text-embedding-ada-002  (OpenAI)
     │
     ├── Retriever  (src/rag/retriever.py)
     │     ├── DenseRetriever   — cosine similarity top-k
     │     └── HybridRetriever  — dense + BM25 sparse + cross-encoder re-rank
     │
     ├── Vector Store  (src/rag/vector_store.py)
     │     └── ChromaDB  (local, persisted to data/db/)
     │
     └── Generator  (src/rag/generator.py)
           └── GPT-4 / Claude  — context-grounded answer generation
```

**Ingestion pipeline** (`src/pipeline/store.py`):  
Raw markdown → chunking (chunk_size=500, overlap=50) → embedding → ChromaDB upsert.

**Config:** All model names, chunk sizes, and API keys are managed via `config/settings.py` + `.env`.

---

## 3. KPIs

| # | KPI | Target | Measurement |
|---|-----|--------|-------------|
| 1 | **Faithfulness Score** | ≥ 0.85 | LLM-as-judge (`src/eval/judge.py`) — answer claims must be grounded in retrieved context |
| 2 | **Answer Relevance (RAGAS)** | ≥ 0.80 | Cosine similarity between generated answer and original query embedding |
| 3 | **End-to-End Latency (p95)** | ≤ 4 s | Measured from POST /query receipt to first token returned, logged via `/metrics` endpoint |

Evaluation is run against the golden set (`data/golden_set.jsonl`) using `scripts/run_eval.py`.
