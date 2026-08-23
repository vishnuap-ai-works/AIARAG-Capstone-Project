"""
Retrieval-Augmented Generation (RAG) Module Initialization.
This dedicated package encapsulates the entire end-to-end RAG workflow.
It is organized into distinct sub-modules handling chunking (text splitting),
embeddings (vectorization), vector store (database interfacing), retrieval (search),
and generation (LLM synthesis). This structure allows for independent testing and
swapping of components (e.g., changing the vector database without affecting the generator).
"""
