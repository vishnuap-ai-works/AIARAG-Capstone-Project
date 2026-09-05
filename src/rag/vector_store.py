"""
Vector Database Interface.
This script manages the connection and interactions with the underlying vector database
(such as ChromaDB, FAISS, Qdrant, or Pinecone).
It provides standard methods for initializing the database, inserting document chunks and
their corresponding embeddings, deleting old records, and persisting the index to disk.
This abstraction ensures the core RAG logic remains decoupled from specific database vendors.

Classes:
- BaseVectorStore: Abstract interface for vector databases.
- JSONVectorStore: Implementation using a simple JSON file.
- ChromaDBStore: Implementation using local ChromaDB.
- PineconeStore: Implementation using Pinecone cloud vector DB.
- QdrantStore: Implementation using Qdrant vector DB.
- VectorStoreFactory: Factory to initialize the configured vector store.

Methods:
- add_document(filename, file_type, chunks, embeddings): Inserts records.
- search(query_embedding, top_k): Searches for relevant chunks.
"""

import json
import math
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path


class BaseVectorStore(ABC):
    @abstractmethod
    def add_document(
        self, filename: str, file_type: str, chunks: list, embeddings: list
    ):
        """
        Adds a document's chunks and embeddings to the vector store.
        If the filename already exists, it overwrites the existing data to avoid duplication.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Searches the vector store for chunks matching the query embedding.
        Returns a list of dictionaries containing 'chunk' and 'metadata'.
        """
        pass


class JSONVectorStore(BaseVectorStore):
    def __init__(self, filepath: str = "data/vector_store.json"):
        self.filepath = filepath
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def add_document(
        self, filename: str, file_type: str, chunks: list, embeddings: list
    ):
        data = {}
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}

        # Update or add the document entry
        chunk_data = [{"chunk": c, "embedding": e} for c, e in zip(chunks, embeddings)]
        data[filename] = {
            "file_name": filename,
            "file_type": file_type,
            "chunks": chunk_data,
        }

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        data = {}
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return []

        results = []
        for filename, file_data in data.items():
            for chunk_data in file_data.get("chunks", []):
                chunk_embed = chunk_data.get("embedding")
                if chunk_embed and len(chunk_embed) == len(query_embedding):
                    # Compute cosine similarity
                    dot_product = sum(
                        a * b for a, b in zip(chunk_embed, query_embedding)
                    )
                    norm_a = math.sqrt(sum(a * a for a in chunk_embed))
                    norm_b = math.sqrt(sum(b * b for b in query_embedding))
                    similarity = (
                        dot_product / (norm_a * norm_b) if norm_a and norm_b else 0
                    )

                    results.append(
                        {
                            "chunk": chunk_data["chunk"],
                            "metadata": {
                                "filename": filename,
                                "file_type": file_data.get("file_type"),
                            },
                            "score": similarity,
                        }
                    )

        # Sort by similarity score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# TODO: Implement ChromaDBStore
class ChromaDBStore(BaseVectorStore):
    def add_document(
        self, filename: str, file_type: str, chunks: list, embeddings: list
    ):
        pass

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        pass


# TODO: Implement PineconeStore
class PineconeStore(BaseVectorStore):
    def add_document(
        self, filename: str, file_type: str, chunks: list, embeddings: list
    ):
        pass

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        pass


# TODO: Implement QdrantStore
class QdrantStore(BaseVectorStore):
    def add_document(
        self, filename: str, file_type: str, chunks: list, embeddings: list
    ):
        pass

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        pass


class VectorStoreFactory:
    @staticmethod
    def get_vector_store() -> BaseVectorStore:
        from config.settings import settings

        store_type = getattr(settings, "VECTOR_STORE_TYPE", "json").lower()

        if store_type == "json":
            return JSONVectorStore(filepath=settings.VECTOR_STORE_PATH)
        elif store_type == "chromadb":
            return ChromaDBStore(persist_directory=settings.CHROMA_PERSIST_DIR)
        elif store_type == "pinecone":
            return PineconeStore(
                api_key=settings.PINECONE_API_KEY,
                index_name=settings.PINECONE_INDEX_NAME,
            )
        elif store_type == "qdrant":
            return QdrantStore(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                collection_name=settings.QDRANT_COLLECTION,
            )
        else:
            raise ValueError(f"Unknown vector store type: {store_type}")
