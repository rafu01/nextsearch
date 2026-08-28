"""ChromaDB-backed vector store implementation."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from nextsearch.vector_store.base import SearchResult, VectorStore


class ChromaVectorStore(VectorStore):
    """Persistent ChromaDB collection.

    Parameters
    ----------
    persist_dir:
        Directory where ChromaDB will save its SQLite + index files.
    collection_name:
        Name of the ChromaDB collection to use.
    """

    def __init__(self, persist_dir: Path | str, collection_name: str = "nextsearch") -> None:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # get_or_create so re-runs are idempotent
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        """Upsert chunks (safe to call multiple times)."""
        # Chroma requires metadata values to be str/int/float/bool
        clean_metas = [_sanitize_metadata(m) for m in metadatas]
        self._col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=clean_metas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._col.count()),
            include=["documents", "metadatas", "distances"],
        )
        output: list[SearchResult] = []
        for chunk_id, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Chroma returns cosine *distance* (0=identical, 2=opposite)
            # Convert to similarity score in [0, 1]
            score = 1.0 - dist / 2.0
            output.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    score=score,
                    metadata=meta,
                    doc_type=meta.get("doc_type", "unknown"),
                )
            )
        return output

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        self._client.delete_collection(self._col.name)
        self._col = self._client.get_or_create_collection(
            name=self._col.name,
            metadata={"hnsw:space": "cosine"},
        )

    def delete(self, ids: list[str]) -> None:
        """Delete chunks by ID."""
        if not ids:
            return
        self._col.delete(ids=ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_metadata(meta: dict) -> dict:
    """Chroma only supports str/int/float/bool metadata values."""
    clean: dict = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = ", ".join(str(x) for x in v)
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    return clean
