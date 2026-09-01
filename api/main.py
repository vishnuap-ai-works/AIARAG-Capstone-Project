"""
FastAPI main application.
Exposes endpoints for the RAG pipeline to be consumed by other services or the Streamlit UI.

Endpoints:
- GET /health: Basic health check.
- POST /upload: Upload documents for indexing.
- POST /query: Submit a question and get a generated answer with sources.
- GET /metrics: Retrieve system performance metrics.

Run with: `uvicorn api.main:app --reload`
"""

from fastapi import FastAPI

app = FastAPI(title="RAG API")


@app.get("/health")
def health_check():
    return {"status": "ok"}
