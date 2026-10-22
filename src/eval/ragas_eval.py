"""
Ragas Evaluation Module.
Provides standalone Ragas evaluation running alongside the LLM Judge.
Computes Faithfulness, Answer Relevancy, Context Precision, and Context Recall
using the configured JUDGE_SOURCE (ollama, openai, or huggingface).
"""

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class RagasJudge:
    """
    Ragas Evaluation Judge.
    Runs standardized Ragas metrics for RAG pipelines.
    """

    def __init__(self, model: Optional[str] = None, source: Optional[str] = None):
        self.source = (source or getattr(settings, "JUDGE_SOURCE", "openai")).lower()
        self.model_name = model

        self._init_llm_and_embeddings()
        self._init_metrics()

    def _init_llm_and_embeddings(self):
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        raw_llm = None
        raw_embeddings = None

        if self.source == "openai":
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            model = self.model_name or getattr(settings, "OPENAI_JUDGE_MODEL", "gpt-4o-mini")
            base_url = getattr(settings, "OPENAI_API_BASE_URL", None) or None
            api_key = getattr(settings, "OPENAI_API_KEY", None) or None
            temperature = getattr(settings, "OPENAI_TEMPERATURE", 0.0)

            raw_llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
            )

            emb_model = getattr(settings, "OPENAI_API_EMBEDDING_MODEL", None) or "text-embedding-3-small"
            try:
                if api_key:
                    raw_embeddings = OpenAIEmbeddings(
                        model=emb_model,
                        api_key=api_key,
                        base_url=base_url,
                    )
            except Exception as e:
                logger.warning(f"Could not initialize OpenAIEmbeddings: {e}")
                raw_embeddings = None

        elif self.source == "ollama":
            from langchain_ollama import ChatOllama

            model = self.model_name or getattr(settings, "OLLAMA_JUDGE_MODEL", "qwen2.5:7b")
            base_url = getattr(settings, "OLLAMA_BASE_URL", None) or "http://localhost:11434"
            temperature = getattr(settings, "OLLAMA_TEMPERATURE", 0.0)

            raw_llm = ChatOllama(
                model=model,
                base_url=base_url,
                temperature=temperature,
                client_kwargs={"timeout": 300.0},
                async_client_kwargs={"timeout": 300.0},
            )

            ollama_emb_model = getattr(settings, "OLLAMA_EMBEDDING_MODEL", None)
            if ollama_emb_model:
                try:
                    from langchain_ollama import OllamaEmbeddings

                    raw_embeddings = OllamaEmbeddings(
                        model=ollama_emb_model,
                        base_url=base_url,
                    )
                except Exception as e:
                    logger.warning(f"Could not initialize OllamaEmbeddings ({ollama_emb_model}): {e}")
                    raw_embeddings = None

        elif self.source == "huggingface":
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

            model = self.model_name or getattr(
                settings,
                "HUGGINGFACE_JUDGE_MODEL",
                "meta-llama/Meta-Llama-3-8B-Instruct",
            )
            api_key = getattr(settings, "HUGGINGFACE_API_KEY", None)

            endpoint = HuggingFaceEndpoint(
                repo_id=model,
                huggingfacehub_api_token=api_key,
                temperature=0.01,
            )
            raw_llm = ChatHuggingFace(llm=endpoint)

        else:
            raise ValueError(f"Unsupported Judge Source for Ragas: {self.source}")

        # Local sentence-transformers fallback for embeddings
        if raw_embeddings is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings

                raw_embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            except Exception as e:
                logger.error(f"Failed to initialize HuggingFaceEmbeddings fallback: {e}")
                raise RuntimeError(f"Unable to initialize embeddings for Ragas: {e}")

        # For local Ollama models, worker concurrency must be limited to prevent
        # GPU/CPU saturation and request queue timeouts in async evaluation jobs.
        from ragas.run_config import RunConfig

        max_workers = 1 if self.source == "ollama" else 4
        self.run_config = RunConfig(
            timeout=300,
            max_workers=max_workers,
            max_retries=3,
            max_wait=60,
        )

        self.llm = LangchainLLMWrapper(raw_llm, run_config=self.run_config)
        self.embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

    def _init_metrics(self):
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def _sanitize_score(self, value: Any) -> float:
        """Sanitize raw metric score into a valid float [0.0, 1.0]."""
        if value is None:
            return 0.0
        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0

    async def evaluate_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Evaluate a list of test questions with Ragas concurrently.
        Each item contains: 'query' (or 'question'), 'context',
        'generated_answer' (or 'answer'), 'expected_answer' (or 'ideal_answer').
        """
        if not items:
            return []

        from datasets import Dataset
        from ragas import evaluate

        user_inputs = []
        responses = []
        retrieved_contexts_list = []
        references = []

        for item in items:
            q = item.get("query") or item.get("question") or ""
            ans = item.get("generated_answer") or item.get("answer") or ""
            ref = item.get("expected_answer") or item.get("ideal_answer") or ""
            raw_ctx = item.get("context", [])

            ctx_chunks = []
            if isinstance(raw_ctx, list):
                for c in raw_ctx:
                    if isinstance(c, dict):
                        chunk_str = c.get("chunk") or c.get("text") or str(c)
                    else:
                        chunk_str = str(c)
                    if chunk_str.strip():
                        ctx_chunks.append(chunk_str.strip())
            elif isinstance(raw_ctx, str) and raw_ctx.strip():
                ctx_chunks = [raw_ctx.strip()]

            if not ctx_chunks:
                ctx_chunks = ["No context retrieved."]

            user_inputs.append(q)
            responses.append(ans)
            retrieved_contexts_list.append(ctx_chunks)
            references.append(ref)

        dataset = Dataset.from_dict(
            {
                "user_input": user_inputs,
                "response": responses,
                "retrieved_contexts": retrieved_contexts_list,
                "reference": references,
            }
        )

        loop = asyncio.get_running_loop()
        eval_result = await loop.run_in_executor(
            None,
            lambda: evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                run_config=self.run_config,
                raise_exceptions=False,
                show_progress=False,
            ),
        )

        df = eval_result.to_pandas()
        scores_list = []

        for _, row in df.iterrows():
            f_score = self._sanitize_score(row.get("faithfulness"))
            ar_score = self._sanitize_score(row.get("answer_relevancy"))
            cp_score = self._sanitize_score(row.get("context_precision"))
            cr_score = self._sanitize_score(row.get("context_recall"))

            scores_list.append(
                {
                    "ragas_faithfulness": f_score,
                    "ragas_answer_relevancy": ar_score,
                    "ragas_context_precision": cp_score,
                    "ragas_context_recall": cr_score,
                }
            )

        return scores_list

    async def evaluate_all(
        self, query: str, context: list, generated_answer: str, expected_answer: str
    ) -> dict:
        """Single-query Ragas evaluation."""
        results = await self.evaluate_batch(
            [
                {
                    "query": query,
                    "context": context,
                    "generated_answer": generated_answer,
                    "expected_answer": expected_answer,
                }
            ]
        )
        if results:
            return results[0]
        return {
            "ragas_faithfulness": 0.0,
            "ragas_answer_relevancy": 0.0,
            "ragas_context_precision": 0.0,
            "ragas_context_recall": 0.0,
        }
