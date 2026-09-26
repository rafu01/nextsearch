"""Generate answers with Gemini using retrieved context chunks.

The generator assembles a prompt that includes:
  1. A system instruction (persona + task).
  2. The retrieved context chunks with source citations.
  3. The user's question.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from nextsearch.vector_store.base import SearchResult

_SYSTEM_PROMPT = """\
You are a knowledgeable study assistant. You answer technical questions \
based ONLY on the provided context from the user's notes and textbooks. \
If the answer cannot be found in the context, say so clearly. \
Always cite the source document when you reference information from the context.
"""


def _build_context_block(results: list[SearchResult]) -> str:
    """Format retrieved chunks into a readable context block."""
    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        source = r.metadata.get("file_name", r.chunk_id)
        title = r.metadata.get("title", source)
        parts.append(
            f"[{i}] Source: {title} ({r.doc_type})\n"
            f"Relevance score: {r.score:.2f}\n"
            f"{r.text}\n"
        )
    return "\n---\n".join(parts)


class GeminiGenerator:
    """Wrap the Gemini generative API for RAG-style Q&A.

    Parameters
    ----------
    api_key:
        Your Google AI API key.
    model:
        Gemini model name, e.g. "gemini-1.5-pro" or "gemini-1.5-flash".
    temperature:
        Sampling temperature (0 = deterministic).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.2,
        api_version: str = "v1",
    ) -> None:
        self._client = genai.Client(
            api_key=api_key,
            http_options={"api_version": api_version},
        )
        self._model = model
        self._config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=temperature,
        )

    def generate(self, query: str, context_results: list[SearchResult]) -> str:
        """Return a generated answer string.

        Parameters
        ----------
        query:
            The user's question.
        context_results:
            Ranked list of SearchResult objects from the retriever.
        """
        if not context_results:
            return (
                "I couldn't find any relevant information in your notes or textbooks "
                "to answer this question."
            )

        context_block = _build_context_block(context_results)
        prompt = (
            f"## Context from your knowledge base\n\n{context_block}\n\n"
            f"## Question\n\n{query}"
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._config,
        )
        return response.text
