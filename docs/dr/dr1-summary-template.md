# DR1 (Design Review 1) Summary
**Date:** 2025-07-10
**Status:** ADR v1 Locked

---

## What We Are Building

A RAG-powered Q&A assistant for the Basecamp (37signals) Employee Handbook. Employees ask natural-language questions about HR policies — PTO, benefits, sabbatical, insurance — and receive grounded, cited answers drawn exclusively from the handbook corpus. The system refuses to answer outside the corpus rather than hallucinate.

Stack: Streamlit UI → FastAPI → ChromaDB (local vector store) → GPT-4 generator, with OpenAI `text-embedding-3-small` / `text-embedding-3-large` for embeddings and a hybrid retriever (dense + BM25 + cross-encoder re-rank).

---

## What We Measure

| KPI | Target | Method |
|-----|--------|--------|
| Faithfulness Score | ≥ 0.85 | LLM-as-judge — answer claims grounded in retrieved context |
| Answer Relevance (RAGAS) | ≥ 0.80 | Cosine similarity between generated answer and query embedding |
| End-to-End Latency (p95) | ≤ 4s | POST /query receipt to first token, logged via `/metrics` |

Evaluated against `data/golden_set.jsonl` using `scripts/run_eval.py`.

---

## Top 3 Things to Discuss

1. **Chunking strategy** — Fixed 500-token chunks with 50-token overlap is the baseline. Should we test semantic or recursive chunking to improve retrieval precision before locking the ingestion pipeline?

2. **Retriever choice** — Dense-only vs hybrid (BM25 + re-rank). Hybrid adds latency; does it meaningfully improve faithfulness on policy Q&A, or is dense sufficient given the structured handbook corpus?

3. **KPI thresholds** — Are 0.85 faithfulness and 0.80 relevance the right bars? We should align on whether these are pass/fail gates for the final eval run or directional targets.
