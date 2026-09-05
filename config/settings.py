"""
Configuration Settings.
This module uses Pydantic Settings to load and validate environment variables from the .env file.
It centralizes all configuration management, ensuring type safety for API keys,
database paths, and other system-wide constants.
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the absolute path to the root of the project (going up one level from config folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE_PATH = os.path.join(BASE_DIR, ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH, env_file_encoding="utf-8", extra="ignore"
    )
    DATA_DIRECTORY: str = ""
    EMBEDDING_MODEL_SOURCE: str = ""

    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE_URL: str = ""
    OPENAI_API_EMBEDDING_MODEL: str = ""

    OLLAMA_BASE_URL: str = ""
    OLLAMA_EMBEDDING_MODEL: str = ""

    DOCKER_BASE_URL: str = ""
    DOCKER_EMBEDDING_MODEL: str = ""

    HUGGINGFACE_API_KEY: str = ""
    DB_PATH: str = "data/db/eval_runs.db"

    VECTOR_STORE_TYPE: str = "json"
    VECTOR_STORE_PATH: str = "data/vector_store.json"
    CHROMA_PERSIST_DIR: str = "data/chroma"
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = ""
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = ""

    USE_RERANKER: bool = False
    RERANKER_TYPE: str = "cross_encoder"  # Options: cross_encoder, cohere
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    COHERE_API_KEY: str = ""

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5


settings = Settings()
