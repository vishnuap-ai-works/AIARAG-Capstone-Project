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
from config.settings import settings

class BaseVectorStore(ABC):
    @abstractmethod
    def add_document(
        self, filename: str, file_type: str, chunks: list, embeddings: list, sparse_embeddings: list = None
    ):
        """
        Adds a document's chunks and embeddings to the vector store.
        If the filename already exists, it overwrites the existing data to avoid duplication.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5, sparse_embedding: dict = None) -> list[dict]:
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
        self, filename: str, file_type: str, chunks: list, embeddings: list, sparse_embeddings: list = None
    ):
        from rag.scratch import delete_from_json
        delete_from_json(filename)
        from config.settings import settings
        data = {}
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}

        # Update or add the document entry
        chunk_data = []
        for i, (c, e) in enumerate(zip(chunks, embeddings)):
            chunk_dict = {"chunk": c, "embedding": e}
            if sparse_embeddings:
                chunk_dict["sparse_embedding"] = {
                    "indices": sparse_embeddings[i].indices.tolist(),
                    "values": sparse_embeddings[i].values.tolist()
                }
            chunk_data.append(chunk_dict)
            
        data[filename] = {
            "file_name": filename,
            "file_type": file_type,
            "chunks": chunk_data,
        }

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def search(self, query_embedding: list[float], top_k: int = 5, sparse_embedding: dict = None) -> list[dict]:
        data = {}
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return []

        dense_results = []
        sparse_results = []
        
        for filename, file_data in data.items():
            for chunk_data in file_data.get("chunks", []):
                chunk_embed = chunk_data.get("embedding")
                if chunk_embed and len(chunk_embed) == len(query_embedding):
                    # Compute dense cosine similarity
                    dot_product = sum(a * b for a, b in zip(chunk_embed, query_embedding))
                    norm_a = math.sqrt(sum(a * a for a in chunk_embed))
                    norm_b = math.sqrt(sum(b * b for b in query_embedding))
                    dense_similarity = dot_product / (norm_a * norm_b) if norm_a and norm_b else 0
                    
                    item = {
                        "chunk": chunk_data["chunk"],
                        "metadata": {"file_name": filename, "file_type": file_data.get("file_type")},
                    }
                    dense_results.append((dense_similarity, item))
                    
                    # Compute sparse similarity if requested
                    if sparse_embedding and "sparse_embedding" in chunk_data:
                        query_indices = sparse_embedding["indices"]
                        query_values = sparse_embedding["values"]
                        
                        doc_indices = chunk_data["sparse_embedding"]["indices"]
                        doc_values = chunk_data["sparse_embedding"]["values"]
                        
                        # Calculate dot product of sparse vectors
                        query_dict = dict(zip(query_indices, query_values))
                        doc_dict = dict(zip(doc_indices, doc_values))
                        
                        sparse_similarity = sum(query_dict.get(k, 0) * doc_dict[k] for k in doc_dict)
                        sparse_results.append((sparse_similarity, item))

        if not sparse_embedding or not sparse_results:
            # Sort by dense only
            dense_results.sort(key=lambda x: x[0], reverse=True)
            return [res[1] for res in dense_results[:top_k]]
            
        # Perform Reciprocal Rank Fusion (RRF)
        dense_results.sort(key=lambda x: x[0], reverse=True)
        sparse_results.sort(key=lambda x: x[0], reverse=True)
        
        rrf_scores = {}
        
        for rank, (_, item) in enumerate(dense_results):
            chunk_text = item["chunk"]
            rrf_scores[chunk_text] = rrf_scores.get(chunk_text, 0) + (1.0 / (60 + rank + 1))
            
        for rank, (_, item) in enumerate(sparse_results):
            chunk_text = item["chunk"]
            rrf_scores[chunk_text] = rrf_scores.get(chunk_text, 0) + (1.0 / (60 + rank + 1))
            
        # Map chunks back to items
        chunk_to_item = {item["chunk"]: item for _, item in dense_results}
        
        # Sort by combined RRF score
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [chunk_to_item[chunk_text] for chunk_text, score in fused[:top_k]]


class ChromaDBStore(BaseVectorStore):
    def __init__(self, persist_directory: str = "data/db/chroma"):
        import chromadb
        self.persist_directory=persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=settings.CHROMA_COLLECTION)
    def add_document(
        self, filename: str, file_type: str, chunks: list, embeddings: list, sparse_embeddings: list = None
    ):
        if not chunks or not embeddings:
            return

        from rag.scratch import delete_from_chroma
        delete_from_chroma(filename)

        from config.settings import settings

        ids = []
        metadata = []
        for i in range(len(chunks)):
            ids.append(f"{filename}_{i}")
            meta = {"file_name": filename, "file_type": file_type}
            if sparse_embeddings:
                import json
                meta["sparse_embedding"] = json.dumps({
                    "indices": sparse_embeddings[i].indices.tolist(),
                    "values": sparse_embeddings[i].values.tolist()
                })
            metadata.append(meta)

        self.collection.upsert(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )

    def search(self, query_embedding: list[float], top_k: int = 5, sparse_embedding: dict = None) -> list[dict]:
        if not self.collection:
            return []

        # Note: ChromaDB doesn't natively support sparse vectors seamlessly yet. 
        # We fall back to standard dense retrieval.
        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            results = []
            for i in range(len(result["documents"][0])):
                doc = result["documents"][0][i]
                metadata = result["metadatas"][0][i]
                score = result["distances"][0][i]

                results.append(
                    {
                        "chunk": doc,
                        "metadata": metadata,
                        "score": score
                    }
                )
            return results
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"ChromaDB search failed: {e}")
            return []

# TODO: Implement PineconeStore
class PineconeStore(BaseVectorStore):
    def add_document(
        self, filename: str, file_type: str, chunks: list, embeddings: list
    ):
        pass

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        pass



class QdrantStore(BaseVectorStore):
    def __init__(self, url: str, api_key: str, collection_name: str):
        from qdrant_client import QdrantClient
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)
        # Aticipate Collection does not exist
        self.collection_exist=False

    def add_document(
        self, filename: str, file_type: str, chunks: list, embeddings: list, sparse_embeddings: list = None
    ):
        if not chunks or not embeddings:
            return
        from qdrant_client.models import PointStruct, VectorParams, Distance, SparseVectorParams

        if not self.collection_exist:
            # Check if collection exists
            if self.client.collection_exists(self.collection_name):
                col = self.client.get_collection(self.collection_name)
                # Check if it has the new named vector schema
                if not isinstance(col.config.params.vectors, dict) or "dense" not in col.config.params.vectors:
                    # Old schema: overwrite as requested
                    self.client.delete_collection(self.collection_name)
                else:
                    self.collection_exist = True
            
            if not self.collection_exist:
                vector_size = len(embeddings[0]) # 384, 762, 1058
                # Create collection with named vectors for dense and sparse
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams()
                    }
                )
                self.collection_exist = True

        # Adding documents in to qdrant
        from rag.scratch import delete_from_qdrant
        delete_from_qdrant(filename)

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{i}"))
            
            vector_payload = {"dense": embedding}
            if sparse_embeddings:
                s_emb = sparse_embeddings[i]
                # Convert fastembed SparseEmbedding to dict expected by Qdrant
                vector_payload["sparse"] = {"indices": s_emb.indices.tolist(), "values": s_emb.values.tolist()}

            points.append(
                PointStruct(
                    id = point_id,
                    vector=vector_payload,
                    payload={"chunk" : chunk, "file_name" : filename, "file_type" : file_type}
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_embedding: list[float], top_k: int = 5, sparse_embedding: dict = None) -> list[dict]:
        from qdrant_client import models
        try:
            if sparse_embedding:
                # Hybrid search using Prefetch and Fusion
                prefetch = [
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_embedding["indices"],
                            values=sparse_embedding["values"]
                        ),
                        using="sparse",
                        limit=top_k
                    ),
                    models.Prefetch(
                        query=query_embedding,
                        using="dense",
                        limit=top_k
                    )
                ]
                temp = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k
                )
            else:
                # Try with named vector "dense" first (new schema)
                temp = self.client.query_points(
                    collection_name = self.collection_name,
                    query = query_embedding,
                    using = "dense",
                    limit = top_k
                )
        except Exception as e:
            try:
                # Fallback to unnamed vector (old schema)
                temp = self.client.query_points(
                    collection_name = self.collection_name,
                    query = query_embedding,
                    limit = top_k
                )
            except Exception as e2:
                import logging
                logging.getLogger(__name__).error(f"Qdrant search failed on both named and unnamed schemas: {e2}")
                return []

        results = []
        for item in temp.points:
            results.append(
                {
               "chunk": item.payload["chunk"],
               "metadata": {
                   'file_name' : item.payload["file_name"],
                   'file_type' : item.payload["file_type"]
               },
               "score": item.score
                }
            )
        return results


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
