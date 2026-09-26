"""HTTP API for querying nextsearch."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from nextsearch.api.models import AskRequest, AskResponse, Citation

app = FastAPI(title="nextsearch API", version="0.1.0")
_pipeline: Any = None


def _get_pipeline() -> Any:
    """Return the lazily initialized application pipeline."""
    global _pipeline
    if _pipeline is None:
        from nextsearch.pipeline import RAGPipeline

        _pipeline = RAGPipeline()
    return _pipeline


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question and return the retrieved grounding citations."""
    try:
        answer, results = _get_pipeline().ask_with_sources(request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to answer the question.") from exc

    return AskResponse(
        answer=answer,
        citations=[Citation.from_search_result(result) for result in results],
    )
