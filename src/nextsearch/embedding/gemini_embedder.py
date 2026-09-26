"""Generate embeddings using Google's Gemini embedding API via google-genai."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from google import genai
from google.genai import types
from tqdm import tqdm

from nextsearch.ingestion.models import Chunk

_logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """Wrap the Gemini embeddings endpoint with batching + retry logic.

    Parameters
    ----------
    api_key:
        Your Google AI API key.
    model:
        Embedding model name, e.g. "models/text-embedding-004".
    batch_size:
        Number of texts to embed per API call.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "models/embedding-001",
        batch_size: int = 100,
        api_version: str = "v1",
    ) -> None:
        self._client = genai.Client(
            api_key=api_key,
            http_options={"api_version": api_version},
        )
        self.model = model
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(
        self,
        texts: Sequence[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Return a list of embedding vectors, one per input text."""
        all_embeddings: list[list[float]] = []
        batches = list(self._batched(texts, self.batch_size))

        iterator = tqdm(batches, desc="Embedding", unit="batch") if show_progress else batches

        for batch in iterator:
            all_embeddings.extend(self._embed_batch(batch))

        return all_embeddings

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Convenience wrapper: embed the .text of each Chunk."""
        return self.embed_texts([c.text for c in chunks], show_progress=show_progress)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self._embed_batch([query], task_type="RETRIEVAL_QUERY")[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_batch(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
        retries: int = 3,
    ) -> list[list[float]]:
        config = types.EmbedContentConfig(task_type=task_type)

        for attempt in range(retries):
            try:
                response = self._client.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config=config,
                )
                return [embedding.values for embedding in response.embeddings]
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                _logger.warning("Embedding batch failed (%s); retrying in %ss…", exc, wait)
                time.sleep(wait)

        raise RuntimeError("Unreachable")

    @staticmethod
    def _batched(seq: Sequence, size: int):
        for i in range(0, len(seq), size):
            yield list(seq[i : i + size])
