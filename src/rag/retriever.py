"""
Core Retrieval Logic.
This module is responsible for taking a user query, embedding it, and fetching the most
relevant document chunks from the vector store.
"""

from abc import ABC, abstractmethod

from config.logging_config import setup_logger
from rag.embeddings import ModelSelector
from rag.vector_store import BaseVectorStore

logger = setup_logger(__name__)


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        pass


class DenseRetriever(BaseRetriever):
    def __init__(self, vector_store: BaseVectorStore, reranker=None):
        self.vector_store = vector_store
        self.reranker = reranker

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        logger.info(f"Retrieving results for query: '{query}'")
        try:
            # If using a reranker, fetch a larger pool initially
            fetch_k = top_k * 4 if self.reranker else top_k

            query_embedding = await ModelSelector.get_single_embedding(query)
            results = self.vector_store.search(query_embedding, top_k=fetch_k)
            logger.info(f"Initial retrieval fetched {len(results)} results")

            # Apply reranker if configured
            if self.reranker and results:
                logger.info(f"Applying reranker to refine top {top_k} results")
                results = self.reranker.rerank(query, results, top_k=top_k)
                logger.info(f"Reranking complete. Final results: {len(results)}")

            return results
        except Exception as e:
            logger.error(f"Failed to retrieve results: {e}")
            raise
