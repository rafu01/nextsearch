"""Generate embeddings using OpenAI's text-embedding API.

Batches requests automatically to respect the API's token-per-minute limits.
"""

from __future__ import annotations

import time
from typing import Sequence

from openai import OpenAI
from tqdm import tqdm

from nextsearch.ingestion.models import Chunk


class OpenAIEmbedder:
    """Wrap the OpenAI embeddings endpoint with batching + retry logic.

    Parameters
    ----------
    api_key:
        Your OpenAI API key.
    model:
        Embedding model name, e.g. "text-embedding-3-small".
    batch_size:
        Number of texts to embed per API call (max 2048 for OpenAI).
    dimensions:
        Optional reduced dimension (only supported by v3 models).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 256,
        dimensions: int | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: Sequence[str], show_progress: bool = True) -> list[list[float]]:
        """Return a list of embedding vectors, one per input text."""
        all_embeddings: list[list[float]] = []
        batches = list(self._batched(texts, self.batch_size))

        iterator = tqdm(batches, desc="Embedding", unit="batch") if show_progress else batches

        for batch in iterator:
            all_embeddings.extend(self._embed_batch(batch))

        return all_embeddings

    def embed_chunks(self, chunks: list[Chunk], show_progress: bool = True) -> list[list[float]]:
        """Convenience wrapper: embed the .text of each Chunk."""
        return self.embed_texts([c.text for c in chunks], show_progress=show_progress)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string (no batching needed)."""
        return self._embed_batch([query])[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str], retries: int = 3) -> list[list[float]]:
        kwargs: dict = {"input": texts, "model": self.model}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions

        for attempt in range(retries):
            try:
                response = self._client.embeddings.create(**kwargs)
                # Results come back sorted by index — preserve order.
                return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            except Exception as exc:  # noqa: BLE001
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"[WARN] Embedding batch failed ({exc}); retrying in {wait}s…")
                time.sleep(wait)

        raise RuntimeError("Unreachable")

    @staticmethod
    def _batched(seq: Sequence, size: int):
        for i in range(0, len(seq), size):
            yield list(seq[i : i + size])
