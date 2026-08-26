"""Abstract base class for vector stores.

This makes it easy to swap ChromaDB for FAISS, Qdrant, Pinecone, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """A single vector search result."""

    chunk_id: str
    text: str
    score: float          # higher = more relevant (cosine similarity)
    metadata: dict[str, Any]
    doc_type: str


class VectorStore(ABC):
    """Minimal interface every vector store must implement."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        """Insert or upsert documents into the store."""

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return the top-k most similar chunks."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored vectors."""

    @abstractmethod
    def reset(self) -> None:
        """Delete all documents from the store."""
