"""Generate embeddings locally with sentence-transformers.

No API calls and no quota — the model runs on your CPU/GPU and is downloaded
on first use.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from nextsearch.ingestion.models import Chunk

_logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Run a local sentence-transformers model for embeddings.

    Parameters
    ----------
    model:
        Name of a sentence-transformers model, e.g. "all-MiniLM-L6-v2".
    batch_size:
        Number of texts to encode at once.
    device:
        "cpu", "cuda", "mps", or None for auto-selection.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        _logger.info("Loading local embedding model %s on %s", model, device or "auto")
        self._model = SentenceTransformer(model, device=device)
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
        if not texts:
            return []

        embeddings = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32).tolist()

    def embed_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Convenience wrapper: embed the .text of each Chunk."""
        return self.embed_texts([c.text for c in chunks], show_progress=show_progress)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query], show_progress=False)[0]
