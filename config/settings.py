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
    LLM_SOURCE: str = "openai"

    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE_URL: str = ""
    OPENAI_API_EMBEDDING_MODEL: str = ""
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.0
    JUDGE_SOURCE: str = "openai"
    OPENAI_JUDGE_MODEL: str = "gpt-4o-mini"
    HUGGINGFACE_JUDGE_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    OLLAMA_JUDGE_MODEL: str = "llama3"

    EMBEDDING_COST_PER_1M_TOKENS: float = 0.1
    LLM_INPUT_COST_PER_1M_TOKENS: float = 0.15
    LLM_OUTPUT_COST_PER_1M_TOKENS: float = 0.6

    OLLAMA_BASE_URL: str = ""
    OLLAMA_EMBEDDING_MODEL: str = ""
    OLLAMA_LLM_MODEL: str = "llama3"
    OLLAMA_TEMPERATURE: float = 0.0

    DOCKER_BASE_URL: str = ""
    DOCKER_EMBEDDING_MODEL: str = ""

    HUGGINGFACE_API_KEY: str = ""
    DB_PATH: str = "data/db/eval_runs.db"

    VECTOR_STORE_TYPE: str = "json"
    VECTOR_STORE_PATH: str = "data/vector_store.json"
    CHROMA_PERSIST_DIR: str = "data/chroma"
    CHROMA_COLLECTION: str = "rag_documents"
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = ""
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = ""

    USE_HYBRID_SEARCH: bool = False
    USE_HYBRID_EMBEDDING: bool = True
    SPARSE_EMBEDDING_MODEL: str = "splade"  # Options: splade, bm25
    SPLADE_MODEL: str = "prithvida/Splade_PP_en_v1"
    BM25_MODEL: str = "Qdrant/bm25"

    USE_RERANKER: bool = False
    RERANKER_TYPE: str = "cross_encoder"  # Options: cross_encoder, cohere
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    COHERE_API_KEY: str = ""

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5

    USE_MULTI_QUERY: bool = False
    USE_QUERY_DECOMPOSITION: bool = False
    USE_HYDE: bool = False
    USE_QUERY_REWRITING: bool = False
    USE_PII_REDACTION: bool = False
    PII_NLP_ENGINE_NAME: str = "spacy" # Options: spacy, stanza, transformers
    PII_NLP_MODEL_NAME: str = "en_core_web_sm" # e.g. en_core_web_sm, dslim/bert-base-NER


settings = Settings()
