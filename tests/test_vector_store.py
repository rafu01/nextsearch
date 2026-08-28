"""Tests for the vector store implementations."""

from pathlib import Path

from nextsearch.vector_store.chroma_store import ChromaVectorStore


def test_chroma_delete_by_ids(tmp_path: Path) -> None:
    store = ChromaVectorStore(persist_dir=tmp_path, collection_name="test_delete")

    store.add(
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        texts=["first", "second", "third"],
        metadatas=[{"doc_type": "test"}, {"doc_type": "test"}, {"doc_type": "test"}],
    )

    assert store.count() == 3

    store.delete(["a", "b"])

    assert store.count() == 1


def test_chroma_delete_empty_id_list_is_noop(tmp_path: Path) -> None:
    store = ChromaVectorStore(persist_dir=tmp_path, collection_name="test_noop")

    store.add(
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        texts=["only"],
        metadatas=[{"doc_type": "test"}],
    )

    store.delete([])

    assert store.count() == 1
