"""
Text Splitting and Chunking Strategies.
This file contains algorithms for breaking down large documents into smaller, semantically
meaningful chunks before embedding. It includes implementations for:
- Recursive Character Text Splitting: Splitting by paragraphs, sentences, and words.
- Semantic Chunking: Grouping text based on embedding similarity to maintain context.
- Token-based Splitting: Ensuring chunks fit strictly within LLM context windows.
Proper chunking is critical for effective retrieval and minimizing noise in the context.

Functions:
- chunk_by_characters(text, chunk_size, overlap): Recursive splitting.
- chunk_by_tokens(text, max_tokens): Token-aware splitting.
- semantic_chunk(text, embedding_model): Groups sentences by semantic similarity.
"""

from abc import ABC, abstractmethod

from config.logging_config import setup_logger

logger = setup_logger(__name__)


class BaseChunker:

    @abstractmethod
    async def chunk(self, text: str) -> list[str]:
        pass


class SlidingWindowChunking(BaseChunker):
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def chunk(self, text):
        logger.info(f"Starting sliding window chunking. Text length: {len(text) if text else 0}, Chunk size: {self.chunk_size}, Overlap: {self.overlap}")
        chunks = []
        try:
            start = 0
            text_length = len(text) if text else 0

            if not text:
                logger.warning("Received empty text for chunking.")
                return chunks

            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunks.append(text[start:end])
                if end == text_length:
                    break
                start += self.chunk_size - self.overlap
                
            logger.info(f"Chunking completed. Generated {len(chunks)} chunks.")
        except Exception as e:
            logger.error(f"Error during chunking: {e}")
            raise
            
        return chunks


# TODO
class SectionChunking(BaseChunker):
    def __init__(self):
        pass

    async def chunk(self, text):
        pass

# TODO
class SemanticChunker(BaseChunker):
    def __init__(self):
        pass

    async def chunk(self, text):
        pass
