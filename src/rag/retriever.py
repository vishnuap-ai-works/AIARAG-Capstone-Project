"""
Core Retrieval Logic.
This module is responsible for taking a user query, embedding it, and fetching the most
relevant document chunks from the vector store.
"""

import json
from abc import ABC, abstractmethod

from config.logging_config import setup_logger
from config.settings import settings
from rag.embeddings import ModelSelector
from rag.generator import LLMGenerator
from rag.prompts.prompts import (
    build_hyde_prompt,
    build_multi_query_prompt,
    build_query_decomposition_prompt,
    build_query_rewriting_prompt,
)
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

            if (
                getattr(settings, "USE_MULTI_QUERY", False)
                or getattr(settings, "USE_QUERY_DECOMPOSITION", False)
                or getattr(settings, "USE_HYDE", False)
                or getattr(settings, "USE_QUERY_REWRITING", False)
            ):
                llm = LLMGenerator()
                augmented_queries = []

                if getattr(settings, "USE_MULTI_QUERY", False):
                    logger.info("Generating multi-queries")
                    prompt = build_multi_query_prompt(query)
                    res = await llm.generate_text(prompt)
                    augmented_queries.extend(
                        [q.strip() for q in res.split("\n") if q.strip()]
                    )

                if getattr(settings, "USE_QUERY_DECOMPOSITION", False):
                    logger.info("Generating decomposed queries")
                    prompt = build_query_decomposition_prompt(query)
                    res = await llm.generate_text(prompt)
                    augmented_queries.extend(
                        [q.strip() for q in res.split("\n") if q.strip()]
                    )

                if getattr(settings, "USE_HYDE", False):
                    logger.info("Generating hypothetical document for HyDE")
                    prompt = build_hyde_prompt(query)
                    res = await llm.generate_text(prompt)
                    if res and res.strip():
                        augmented_queries.append(res.strip())

                if getattr(settings, "USE_QUERY_REWRITING", False):
                    logger.info("Generating rewritten query")
                    prompt = build_query_rewriting_prompt(query)
                    res = await llm.generate_text(prompt)
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
                                "values": sparse_obj.values.tolist(),
                            }
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate sparse query embedding: {e}"
                        )

                results = self.vector_store.search(
                    query_embedding, top_k=fetch_k, sparse_embedding=sparse_embedding
                )

                for r in results:
                    chunk_text = r.get("chunk", str(r))
                    if chunk_text not in seen_chunks:
                        seen_chunks.add(chunk_text)
                        all_results.append(r)

            logger.info(
                f"Initial retrieval fetched {len(all_results)} unique results across all queries"
            )

            # Tag initial retrieval rank on all fetched candidate chunks
            for idx, r in enumerate(all_results, start=1):
                if isinstance(r, dict):
                    r["retrieval_rank"] = idx

            # Apply reranker if configured
            if self.reranker and all_results:
                initial_chunks_payload = [
                    {
                        "rank": idx,
                        "similarity_score": (
                            round(r.get("score"), 4)
                            if isinstance(r.get("score"), (int, float))
                            else None
                        ),
                        "metadata": r.get("metadata", {}),
                    }
                    for idx, r in enumerate(all_results, start=1)
                ]

                initial_score_entries = [
                    (
                        f"chunk {c['rank']}. - {c['similarity_score']:.2f}"
                        if isinstance(c["similarity_score"], (int, float))
                        else f"chunk {c['rank']}. - N/A"
                    )
                    for c in initial_chunks_payload
                ]
                logger.info(
                    f"Initial retrieved chunk similarity scores: {', '.join(initial_score_entries)}"
                )

                logger.info(f"Applying reranker to refine top {top_k} results")
                all_results = self.reranker.rerank(query, all_results, top_k=top_k)
                logger.info(f"Reranking complete. Final results: {len(all_results)}")

                reranked_chunks_payload = [
                    {
                        "rank": idx,
                        "initial_rank": r.get("retrieval_rank"),
                        "rerank_score": (
                            round(r.get("rerank_score"), 4)
                            if isinstance(r.get("rerank_score"), (int, float))
                            else None
                        ),
                        "initial_score": (
                            round(r.get("score"), 4)
                            if isinstance(r.get("score"), (int, float))
                            else None
                        ),
                        "metadata": r.get("metadata", {}),
                    }
                    for idx, r in enumerate(all_results, start=1)
                ]

                rerank_score_entries = [
                    (
                        f"chunk {c['rank']}. - {c['rerank_score']:.2f}"
                        if isinstance(c["rerank_score"], (int, float))
                        else f"chunk {c['rank']}. - N/A"
                    )
                    for c in reranked_chunks_payload
                ]
                logger.info(
                    f"Top {len(all_results)} reranked chunk scores: {', '.join(rerank_score_entries)}"
                )

                # Log ranks and scores of retriever and reranker as JSON
                logger.info(
                    "Retriever and Reranked Chunks (JSON):\n"
                    + json.dumps(
                        {
                            "query": query,
                            "retrieved_chunks": initial_chunks_payload,
                            "reranked_chunks": reranked_chunks_payload,
                        },
                        indent=2,
                        default=str,
                    )
                )
            elif all_results:
                if len(all_results) > top_k:
                    # If no reranker, just take the first top_k
                    all_results = all_results[:top_k]

                retrieved_chunks_payload = [
                    {
                        "rank": idx,
                        "similarity_score": (
                            round(r.get("score"), 4)
                            if isinstance(r.get("score"), (int, float))
                            else None
                        ),
                        "metadata": r.get("metadata", {}),
                    }
                    for idx, r in enumerate(all_results, start=1)
                ]

                score_entries = [
                    (
                        f"chunk {c['rank']}. - {c['similarity_score']:.2f}"
                        if isinstance(c["similarity_score"], (int, float))
                        else f"chunk {c['rank']}. - N/A"
                    )
                    for c in retrieved_chunks_payload
                ]
                logger.info(
                    f"Top {len(all_results)} retrieved chunk similarity scores: {', '.join(score_entries)}"
                )

                # Log ranks and scores of retriever as JSON
                logger.info(
                    "Retrieved Chunks (JSON):\n"
                    + json.dumps(
                        {
                            "query": query,
                            "retrieved_chunks": retrieved_chunks_payload,
                        },
                        indent=2,
                        default=str,
                    )
                )
            else:
                logger.info("No chunks retrieved.")

            return all_results
        except Exception as e:
            logger.error(f"Failed to retrieve results: {e}")
            raise
