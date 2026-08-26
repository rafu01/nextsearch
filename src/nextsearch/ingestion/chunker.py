"""Token-aware text chunker.

Uses tiktoken to measure chunk sizes in tokens so the numbers map directly
to what the embedding model sees (rather than characters or words).
"""

from __future__ import annotations

from pathlib import Path

import tiktoken

from nextsearch.ingestion.models import Chunk, Document


class TextChunker:
    """Split a Document into overlapping token-bounded Chunks.

    Parameters
    ----------
    chunk_size:
        Maximum number of tokens per chunk.
    chunk_overlap:
        Number of tokens of overlap between consecutive chunks.
    encoding_name:
        The tiktoken encoding to use for tokenisation.
        "cl100k_base" works for OpenAI text-embedding-3-* and gpt-4.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._enc = tiktoken.get_encoding(encoding_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Return a list of Chunks for the given Document."""
        tokens = self._enc.encode(doc.content)
        raw_chunks = self._split_tokens(tokens)

        chunks: list[Chunk] = []
        for i, token_chunk in enumerate(raw_chunks):
            text = self._enc.decode(token_chunk)
            chunks.append(
                Chunk(
                    text=text,
                    source=doc.source,
                    chunk_index=i,
                    metadata={**doc.metadata, "total_chunks": len(raw_chunks)},
                    doc_type=doc.doc_type,
                )
            )
        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Chunk all documents and return a flat list of Chunks."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_tokens(self, tokens: list[int]) -> list[list[int]]:
        """Sliding-window split of a token list into overlapping sublists."""
        if not tokens:
            return []

        chunks: list[list[int]] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunks.append(tokens[start:end])
            if end == len(tokens):
                break
            start += step

        return chunks
