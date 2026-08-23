"""
Embedding Model Wrappers.
This module acts as an abstraction layer over various embedding models (e.g., OpenAI's
text-embedding-ada-002, HuggingFace sentence-transformers, Cohere).
It provides a unified interface for converting text chunks into dense vector representations.
It also includes logic for batch processing, handling rate limits, and caching embeddings
to avoid redundant API calls and reduce costs during indexing.

Classes:
- BaseEmbeddingModel: Abstract base class for embedding models.
- OpenAIEmbeddings: Implementation for OpenAI API.
- HuggingFaceEmbeddings: Local embedding generation using sentence-transformers.

Functions:
- get_embedding(text): Returns the embedding vector for a single string.
- get_embeddings_batch(texts): Returns a list of vectors for a batch of strings.
"""
