import math
from abc import ABC, abstractmethod

from config.logging_config import setup_logger
from config.settings import settings

logger = setup_logger(__name__)


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """
        Reranks the given documents against the query.
        Each document is a dict containing at least a 'chunk' string.
        Returns the top_k sorted list of documents.
        """
        pass


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        logger.info(f"Initializing CrossEncoderReranker with model: {self.model_name}")
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
        except ImportError:
            logger.error(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}")
            raise

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []

        logger.info(f"Reranking {len(documents)} documents for query: '{query}'")

        # CrossEncoder expects pairs of [query, chunk]
        pairs = [[query, doc["chunk"]] for doc in documents]

        # Predict scores
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Failed to compute reranking scores: {e}")
            raise

        # Update documents with their new scores
        for idx, doc in enumerate(documents):
            raw_score = float(scores[idx])
            # Normalize to [0, 1] via sigmoid if raw score is outside [0, 1]
            if 0.0 <= raw_score <= 1.0:
                norm_score = raw_score
            else:
                try:
                    norm_score = 1.0 / (1.0 + math.exp(-raw_score))
                except OverflowError:
                    norm_score = 1.0 if raw_score > 0 else 0.0

            doc["raw_rerank_score"] = raw_score
            doc["rerank_score"] = round(norm_score, 4)

        # Sort documents by score in descending order
        reranked_docs = sorted(
            documents, key=lambda x: x["raw_rerank_score"], reverse=True
        )
        top_docs = reranked_docs[:top_k]

        score_entries = [
            f"chunk {i}. - {doc['rerank_score']:.2f}"
            for i, doc in enumerate(top_docs, start=1)
        ]
        logger.info(
            f"Top {len(top_docs)} reranked chunk scores: {', '.join(score_entries)}"
        )

        return top_docs


class CohereReranker(BaseReranker):
    def __init__(self, model_name: str = "rerank-english-v3.0"):
        self.model_name = model_name
        self.api_key = getattr(settings, "COHERE_API_KEY", "")
        logger.info(f"Initializing CohereReranker with model: {self.model_name}")

        if not self.api_key:
            logger.warning("COHERE_API_KEY is not set. Cohere reranking will fail.")

        try:
            import cohere

            self.client = cohere.Client(self.api_key)
        except ImportError:
            logger.error("cohere is not installed. Run: pip install cohere")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            raise

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []

        logger.info(f"Reranking {len(documents)} documents for query: '{query}'")

        # Cohere expects just the strings
        texts = [doc["chunk"] for doc in documents]

        try:
            results = self.client.rerank(
                query=query, documents=texts, top_n=top_k, model=self.model_name
            )
        except Exception as e:
            logger.error(f"Failed to compute Cohere reranking scores: {e}")
            raise

        reranked_docs = []
        for result in results.results:
            idx = result.index
            doc = documents[idx]
            doc["rerank_score"] = round(float(result.relevance_score), 4)
            reranked_docs.append(doc)

        score_entries = [
            f"chunk {i}. - {doc['rerank_score']:.2f}"
            for i, doc in enumerate(reranked_docs, start=1)
        ]
        logger.info(
            f"Top {len(reranked_docs)} reranked chunk scores: {', '.join(score_entries)}"
        )

        return reranked_docs


class RerankerFactory:
    @staticmethod
    def get_reranker() -> BaseReranker:
        reranker_type = getattr(settings, "RERANKER_TYPE", "cross_encoder").lower()
        if reranker_type == "cohere":
            return CohereReranker()
        elif reranker_type == "cross_encoder":
            return CrossEncoderReranker()
        else:
            logger.warning(
                f"Unsupported RERANKER_TYPE: {reranker_type}. Defaulting to CrossEncoderReranker."
            )
            return CrossEncoderReranker()
