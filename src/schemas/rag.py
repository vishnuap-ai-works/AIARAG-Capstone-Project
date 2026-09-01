"""
RAG Core Schemas.
Defines internal data structures passed between chunker, retriever, and generator.

Classes:
- DocumentChunk: Represents a single piece of text with metadata and an optional embedding.
- RetrievalResult: Represents a chunk fetched from the vector store with a similarity score.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DocumentChunk(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
