"""
Streamlit Web Interface.
This script launches a user-friendly chat interface for the RAG system using Streamlit.
It allows users to upload documents, ask questions, and visualize the retrieved chunks
alongside the AI-generated answer. It connects directly to the `src.rag` backend components or the FastAPI backend.

Features:
- Sidebar for configuration (model selection, chunk size).
- File uploader (PDF, TXT) for dynamic knowledge base ingestion.
- Chat interface with streaming text generation.
- Expandable sections showing retrieved context sources.

To run: `streamlit run ui/app.py`
"""
