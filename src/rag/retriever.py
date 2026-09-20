"""
Core Retrieval Logic.
This module is responsible for taking a user query, embedding it, and fetching the most
relevant document chunks from the vector store.
"""

from abc import ABC, abstractmethod

from config.logging_config import setup_logger
from config.settings import settings
from rag.embeddings import ModelSelector
from rag.generator import LLMGenerator
from rag.prompts.prompts import build_multi_query_prompt, build_query_decomposition_prompt, build_hyde_prompt
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

            queries_to_embed = [query]
            
            if getattr(settings, "USE_MULTI_QUERY", False) or getattr(settings, "USE_QUERY_DECOMPOSITION", False) or getattr(settings, "USE_HYDE", False):
                llm = LLMGenerator()
                augmented_queries = []
                
                if getattr(settings, "USE_MULTI_QUERY", False):
                    logger.info("Generating multi-queries")
                    prompt = build_multi_query_prompt(query)
                    res = await llm.generate_text(prompt)
                    augmented_queries.extend([q.strip() for q in res.split('\n') if q.strip()])
                    
                if getattr(settings, "USE_QUERY_DECOMPOSITION", False):
                    logger.info("Generating decomposed queries")
                    prompt = build_query_decomposition_prompt(query)
                    res = await llm.generate_text(prompt)
                    augmented_queries.extend([q.strip() for q in res.split('\n') if q.strip()])
                    
                if getattr(settings, "USE_HYDE", False):
                    logger.info("Generating hypothetical document for HyDE")
                    prompt = build_hyde_prompt(query)
                    res = await llm.generate_text(prompt)
                    print("###########")
                    print(res)
                    if res and res.strip():
                        augmented_queries.append(res.strip())
                
                # Add augmented queries, avoiding duplicates
                for q in augmented_queries:
                    if q and q not in queries_to_embed:
                        queries_to_embed.append(q)
                
                logger.info(f"Total queries to embed: {len(queries_to_embed)}")

            all_results = []
            seen_chunks = set()
            
            for q in queries_to_embed:
                query_embedding = await ModelSelector.get_single_embedding(q)
                
                sparse_embedding = None
                if getattr(settings, "USE_HYBRID_SEARCH", False):
                    try:
                        sparse_obj = ModelSelector.get_single_sparse_embedding(q)
                        if sparse_obj:
                            sparse_embedding = {
                                "indices": sparse_obj.indices.tolist(),
                                "values": sparse_obj.values.tolist()
                            }
                    except Exception as e:
                        logger.warning(f"Failed to generate sparse query embedding: {e}")
                
                results = self.vector_store.search(
                    query_embedding, 
                    top_k=fetch_k, 
                    sparse_embedding=sparse_embedding
                )
                
                for r in results:
                    chunk_text = r.get("chunk", str(r))
                    if chunk_text not in seen_chunks:
                        seen_chunks.add(chunk_text)
                        all_results.append(r)
            
            logger.info(f"Initial retrieval fetched {len(all_results)} unique results across all queries")

            # Apply reranker if configured
            if self.reranker and all_results:
                logger.info(f"Applying reranker to refine top {top_k} results")
                all_results = self.reranker.rerank(query, all_results, top_k=top_k)
                logger.info(f"Reranking complete. Final results: {len(all_results)}")
            elif all_results and len(all_results) > top_k:
                # If no reranker, just take the first top_k
                all_results = all_results[:top_k]

            return all_results
        except Exception as e:
            logger.error(f"Failed to retrieve results: {e}")
            raise
