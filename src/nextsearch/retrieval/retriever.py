"""Retriever: embed a query and fetch the top-k chunks from the vector store."""

from __future__ import annotations

from nextsearch.embedding.base import Embedder
from nextsearch.vector_store.base import SearchResult, VectorStore


class Retriever:
    """Embeds a natural-language query and returns ranked chunks.

    Parameters
    ----------
    embedder:
        Any object implementing the Embedder protocol.
    vector_store:
        The vector store to search.
    top_k:
        Default number of results to return.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Return ranked SearchResult objects for *query*."""
        k = top_k if top_k is not None else self.top_k
        query_vec = self._embedder.embed_query(query)
        return self._store.search(query_vec, top_k=k)

    def retrieve_texts(self, query: str, top_k: int | None = None) -> list[str]:
        """Convenience method: return just the text of each result."""
        return [r.text for r in self.retrieve(query, top_k=top_k)]
