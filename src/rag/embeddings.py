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

from abc import ABC, abstractmethod

import ollama
from openai import AsyncOpenAI

from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger(__name__)


# Template for our file loaders
class BaseEmbeddingModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def get_embedding(self, chunk: str) -> list[float]:
        pass


class OpenAIEmbeddings(BaseEmbeddingModel):

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_API_BASE_URL
        )

    async def get_embedding(self, chunk: str) -> list[float]:
        try:
            response = await self.client.embeddings.create(
                input=[chunk], model=settings.OPENAI_API_EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise


class OllamaEmbeddings(BaseEmbeddingModel):
    async def get_embedding(self, chunk: str) -> list[float]:
        try:
            response = ollama.embed(model=settings.OLLAMA_EMBEDDING_MODEL, input=chunk)
            return response["embeddings"][0]
        except Exception as e:
            logger.error(f"Ollama embedding generation failed: {e}")
            raise

#TODO
class HuggingFaceEmbeddings(BaseEmbeddingModel):
    pass


class ModelSelector:
    @staticmethod
    async def get_embedded(chunks: list[str]) -> list[list[float]]:
        vector = []

        # Instantiate model once
        if settings.EMBEDDING_MODEL_SOURCE == "openai":
            model = OpenAIEmbeddings()
        elif settings.EMBEDDING_MODEL_SOURCE == "ollama":
            model = OllamaEmbeddings()
        else:
            raise ValueError(
                f"Unknown embedding model source: {settings.EMBEDDING_MODEL_SOURCE}"
            )

        logger.info(f"Generating embeddings for {len(chunks)} chunks using {settings.EMBEDDING_MODEL_SOURCE} model")
        
        try:
            for chunk in chunks:
                embedded = await model.get_embedding(chunk)
                vector.append(embedded)
            logger.info(f"Successfully generated {len(vector)} embeddings.")
        except Exception as e:
            logger.error(f"Error during batch embedding generation: {e}")
            raise

        return vector
