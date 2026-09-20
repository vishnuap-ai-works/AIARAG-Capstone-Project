"""
Document Ingestion Pipeline.
This module manages the pipeline for ingesting documents, chunking them,
and generating embeddings for vector storage.

Classes:
- DocumentIngestionPipeline: Manages document processing workflow.
"""

import asyncio
import os
from pathlib import Path

from config.logging_config import setup_logger
from config.settings import settings
from rag.chunking import SlidingWindowChunking
from rag.document_loader import DocumentLoadFactory
from rag.embeddings import ModelSelector
from rag.vector_store import VectorStoreFactory

logger = setup_logger(__name__)


class DocumentIngestionPipeline:

    def __init__(self, chunker=None, embedder=None, vector_store=None):
        logger.info("Initializing DocumentIngestionPipeline")
        self.chunker = chunker or SlidingWindowChunking(
            chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP
        )
        self.embedder = embedder or ModelSelector
        # Configure Vector Store from settings via Factory
        self.vector_store = vector_store or VectorStoreFactory.get_vector_store()

        self.redactor = None
        if getattr(settings, "USE_PII_REDACTION", False):
            from rag.pii_redactor import PIIRedactor
            self.redactor = PIIRedactor()

    async def ingest_file(self, path) -> list[str]:
        try:
            path_obj = Path(path)
            file_name = path_obj.name
            file_type = path_obj.suffix.lstrip(".")

            logger.info(f"1. Loading Document.....{file_name}")
            loader = DocumentLoadFactory.get_loader(path_obj)
            text_content = await loader.load()
            logger.info(f"Successfully loaded document: {file_name}")

            if self.redactor:
                logger.info(f"Redacting PII from document: {file_name}")
                text_content = self.redactor.redact_text(text_content)

            logger.info(f"2. Document Chunking.....{file_name}")
            chunks = await self.chunker.chunk(text_content)
            logger.info(f"Successfully chunked document into {len(chunks)} pieces.")

            logger.info(f"3. Embeddings.....{file_name}")
            embedded = await self.embedder.get_embedded(chunks)
            logger.info(f"Successfully generated {len(embedded)} dense embeddings.")

            sparse_embedded = None
            if getattr(settings, "USE_HYBRID_EMBEDDING", True):
                sparse_embedded = self.embedder.get_sparse_embedded(chunks)
                if sparse_embedded:
                    logger.info(f"Successfully generated {len(sparse_embedded)} sparse embeddings.")

            logger.info(f"4. Vector Storage.....{file_name}")

            # Store the data in the vector store
            self.vector_store.add_document(
                filename=file_name,
                file_type=file_type,
                chunks=chunks,
                embeddings=embedded,
                sparse_embeddings=sparse_embedded,
            )
            logger.info(f"Successfully completed vector storage for: {file_name}")
            return chunks
        except Exception as e:
            logger.error(f"Failed to ingest file {file_name}: {e}")
            raise

    async def ingest_directory(self, path) -> list[str]:
        try:
            path = Path(path)
            all_files = []
            files = path.rglob("*")

            for file in files:
                if file.is_file():
                    all_files.append(await self.ingest_file(file))

            return all_files
        except Exception as e:
            logger.error(f"Failed to ingest directory {path}: {e}")
            raise


async def main():
    pipeline = DocumentIngestionPipeline()
    await pipeline.ingest_directory(settings.DATA_DIRECTORY)


if __name__ == "__main__":
    asyncio.run(main())
