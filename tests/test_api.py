"""Isolated tests for the HTTP ask endpoint."""

from importlib import import_module
from unittest.mock import Mock

from fastapi.testclient import TestClient

api_app = import_module("nextsearch.api.app")

from nextsearch.vector_store.base import SearchResult

client = TestClient(api_app.app)


def test_ask_returns_answer_and_grounded_citations(monkeypatch) -> None:
    result = SearchResult(
        chunk_id="attention_abc123__0",
        text="Multi-head attention lets a model attend to multiple representation subspaces.",
        score=0.94,
        metadata={
            "title": "Attention Is All You Need",
            "file_name": "transformers/attention.md",
        },
        doc_type="markdown",
    )
    pipeline = Mock()
    pipeline.ask_with_sources.return_value = ("Attention uses multiple heads.", [result])
    monkeypatch.setattr(api_app, "_pipeline", pipeline)

    response = client.post("/api/ask", json={"query": "How does attention work?", "top_k": 3})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Attention uses multiple heads.",
        "citations": [
            {
                "id": "attention_abc123__0",
                "title": "Attention Is All You Need",
                "source": "transformers/attention.md",
                "excerpt": (
                    "Multi-head attention lets a model attend to multiple representation subspaces."
                ),
                "score": 0.94,
            }
        ],
    }
    pipeline.ask_with_sources.assert_called_once_with("How does attention work?", top_k=3)


def test_ask_returns_empty_citations_when_retrieval_is_empty(monkeypatch) -> None:
    pipeline = Mock()
    pipeline.ask_with_sources.return_value = ("No matching notes found.", [])
    monkeypatch.setattr(api_app, "_pipeline", pipeline)

    response = client.post("/api/ask", json={"query": "Unknown topic"})

    assert response.status_code == 200
    assert response.json() == {"answer": "No matching notes found.", "citations": []}
    pipeline.ask_with_sources.assert_called_once_with("Unknown topic", top_k=5)


def test_ask_rejects_invalid_requests(monkeypatch) -> None:
    pipeline = Mock()
    monkeypatch.setattr(api_app, "_pipeline", pipeline)

    empty_query = client.post("/api/ask", json={"query": ""})
    invalid_top_k = client.post("/api/ask", json={"query": "valid", "top_k": 0})

    assert empty_query.status_code == 422
    assert invalid_top_k.status_code == 422
    pipeline.ask_with_sources.assert_not_called()


def test_ask_hides_pipeline_errors(monkeypatch) -> None:
    pipeline = Mock()
    pipeline.ask_with_sources.side_effect = RuntimeError("secret provider details")
    monkeypatch.setattr(api_app, "_pipeline", pipeline)

    response = client.post("/api/ask", json={"query": "valid"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Unable to answer the question."}
    assert "secret provider details" not in response.text
