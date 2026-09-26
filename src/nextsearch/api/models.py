"""Request and response models for the nextsearch HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nextsearch.vector_store.base import SearchResult


class AskRequest(BaseModel):
    """Payload accepted by ``POST /api/ask``."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)


class Citation(BaseModel):
    """A grounded source excerpt returned with an answer."""

    id: str
    title: str
    source: str
    excerpt: str
    score: float

    @classmethod
    def from_search_result(cls, result: SearchResult) -> Citation:
        source = str(result.metadata.get("source", result.metadata.get("file_name", result.chunk_id)))
        title = str(result.metadata.get("title", source))
        excerpt = result.text[:300] + ("…" if len(result.text) > 300 else "")
        return cls(
            id=result.chunk_id,
            title=title,
            source=source,
            excerpt=excerpt,
            score=result.score,
        )


class AskResponse(BaseModel):
    """Answer and the retrieved sources used to ground it."""

    answer: str
    citations: list[Citation]

