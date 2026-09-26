"""Shared embedder interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nextsearch.ingestion.models import Chunk


@runtime_checkable
class Embedder(Protocol):
    """Minimal interface every embedder must implement."""

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Return a list of embedding vectors, one per input text."""

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Convenience wrapper: embed the .text of each Chunk."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
