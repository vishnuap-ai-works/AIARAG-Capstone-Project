"""
API Schemas.
Defines the Pydantic models for FastAPI request and response bodies.

Classes:
- QueryRequest: User query and optional filters.
- QueryResponse: Generated answer, retrieved context, and metadata.
- UploadResponse: Status of document ingestion.
"""
from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
