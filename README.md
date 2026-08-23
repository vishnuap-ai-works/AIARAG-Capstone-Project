# Capstone: RAG Pipeline

This repository contains the production-ready structure for an end-to-end Retrieval Augmented Generation (RAG) system.

## 📂 Detailed Folder Structure & Contents

### `src/` (Source Code)
The main codebase, containing all core logic for the RAG application.
- **`rag/`**: The core RAG engine. 
  - `chunking.py`: Logic for splitting documents (recursive, semantic, token-based).
  - `embeddings.py`: Wrappers for embedding models (OpenAI, HuggingFace).
  - `vector_store.py`: Abstraction layer for interacting with vector databases (ChromaDB, Pinecone).
  - `retriever.py`: Logic for finding relevant chunks (dense, sparse, hybrid search).
  - `generator.py`: Prompt construction and interaction with the LLM to generate the final answer.
  - `prompts/`: Contains `.txt` files with system and user prompt templates.
- **`eval/`**: Evaluation logic to score the RAG pipeline.
  - `judge.py`: LLM-as-a-judge implementation to grade answers.
  - `golden.py`: Utilities for evaluating against known golden datasets.
  - `pairwise.py`: A/B testing logic to compare two models.
  - `critic_creator.py`: Experimental agents designed to find flaws in generated text.
- **`pipeline/`**: Data persistence and orchestration.
  - `store.py`: SQLite-based experiment tracking to log queries, answers, and scores.
- **`schemas/`**: Pydantic models defining input/output structures for the API and internal data flow.
  - `api.py`: FastAPI request and response models.
  - `rag.py`: Internal structures like `DocumentChunk` and `RetrievalResult`.

### `api/` (Backend API)
- `main.py`: The FastAPI application exposing the RAG pipeline as RESTful endpoints (`/upload`, `/query`). Serves as the backend for the UI.

### `ui/` (User Interface)
- `app.py`: The Streamlit frontend providing a chat interface, document upload, and configuration sidebars.
- `components/`: Reusable Streamlit UI components (e.g., custom chat bubbles).

### `data/` (Datasets & Storage)
- **`uploads/`**: Temporary storage for files uploaded via the UI.
- **`raw_markdown/`**: Source knowledge base documents (`.md`, `.txt`) awaiting ingestion.
- **`db/`**: Persistent local storage, including SQLite databases for experiment tracking and ChromaDB vector indexes.
- **`temp/`**: Short-lived artifacts and JSON dumps used during processing.
- `golden_set.jsonl`: Benchmark datasets used for automated evaluation.

### `notebooks/`
- Jupyter notebooks (`.ipynb`) used for exploratory data analysis, testing chunking strategies, and visualization.

### `scripts/` (Execution)
Command-line Python scripts used for running evaluations and batch tasks.
- `run_eval.py`: Run standard evaluations.
- `run_rag_eval.py`: Orchestrate end-to-end pipeline evaluations.
- `run_pairwise.py`: Run A/B comparisons.

### `bin/` (Shell Utilities)
- `setup_env.sh` / `.ps1`: Scripts to install dependencies (via `pyproject.toml`) and create virtual environments.
- `run_app.sh` / `.ps1`: Scripts to concurrently launch the FastAPI server and Streamlit UI.

### `tests/`
Unit tests isolating specific components.
- `test_rag.py`: Tests for chunking, embedding, and retrieval.
- `test_judge.py`, `test_api.py`, `test_ui.py`: Asserts correctness for evaluation, API endpoints, and UI logic.

### `docs/`
Documentation, templates, and architecture decisions.
- `adr/`: Architecture Decision Records.
- `dr/`: Design Review materials.

### `config/`
- `settings.py`: Centralized configuration mapping to `.env` variables using `pydantic-settings` (API keys, DB paths, embedding models).
- `logging_config.py`: Standardized logging setup for the application.
