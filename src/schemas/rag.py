"""
RAG Core Schemas.
Defines internal data structures passed between chunker, retriever, and generator.

Classes:
- DocumentChunk: Represents a single piece of text with metadata and an optional embedding.
- RetrievalResult: Represents a chunk fetched from the vector store with a similarity score.
"""
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class DocumentChunk(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
