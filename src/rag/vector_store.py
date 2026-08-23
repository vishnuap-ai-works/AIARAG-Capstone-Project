"""
Vector Database Interface.
This script manages the connection and interactions with the underlying vector database
(such as ChromaDB, FAISS, Qdrant, or Pinecone).
It provides standard methods for initializing the database, inserting document chunks and
their corresponding embeddings, deleting old records, and persisting the index to disk.
This abstraction ensures the core RAG logic remains decoupled from specific database vendors.

Classes:
- VectorStore: Abstract interface for vector databases.
- ChromaDBStore: Implementation using local ChromaDB.
- PineconeStore: Implementation using Pinecone cloud vector DB.

Methods:
- add_documents(documents, embeddings, metadata): Inserts records.
- search(query_embedding, top_k): Returns top_k similar documents.
- delete(document_ids): Removes documents from the index.
"""
