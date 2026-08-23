"""
Core Retrieval Logic.
This module is responsible for taking a user query, embedding it, and fetching the most
relevant document chunks from the vector store.
It implements advanced retrieval techniques including:
- Top-K Similarity Search (Cosine similarity, L2 distance)
- Hybrid Search (combining dense vector search with sparse keyword search like BM25)
- Query Re-writing and Expansion to improve recall.
- Re-ranking algorithms (e.g., using Cross-Encoders) to refine the fetched results.

Classes:
- BaseRetriever: Abstract interface.
- DenseRetriever: Standard vector similarity search.
- HybridRetriever: Combines sparse (BM25) and dense search.

Methods:
- retrieve(query, top_k=5): Executes the full retrieval pipeline including re-ranking.
"""
