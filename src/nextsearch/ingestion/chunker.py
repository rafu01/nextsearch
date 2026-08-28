"""Token-aware text chunker.

Uses tiktoken to measure chunk sizes in tokens so the numbers map directly
to what the embedding model sees (rather than characters or words).
"""

from __future__ import annotations

import logging
from typing import ClassVar

import tiktoken

from nextsearch.ingestion.models import Chunk, Document

_logger = logging.getLogger(__name__)


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

    def chunk_document(self, doc: Document, start_index: int = 0) -> list[Chunk]:
        """Return a list of Chunks for the given Document."""
        tokens = self._enc.encode(doc.content)
        raw_chunks = self._split_tokens(tokens)

        chunks: list[Chunk] = []
        for i, token_chunk in enumerate(raw_chunks):
            text = self._enc.decode(token_chunk)
            chunk = Chunk(
                text=text,
                source=doc.source,
                chunk_index=start_index + i,
                metadata={**doc.metadata, "total_chunks": len(raw_chunks)},
                doc_type=doc.doc_type,
            )
            _logger.debug(
                "Chunk %d from %s: %d tokens",
                i + start_index,
                doc.source,
                len(token_chunk),
            )
            chunks.append(chunk)
        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Chunk all documents and return a flat list of Chunks.

        Chunk indices are offset across documents so IDs stay unique even when
        several files share the same stem.
        """
        all_chunks: list[Chunk] = []
        offset = 0
        for doc in docs:
            chunks = self.chunk_document(doc, start_index=offset)
            all_chunks.extend(chunks)
            offset += len(chunks)
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


class MarkdownChunker:
    """Structure-aware chunker for Obsidian / Markdown documents.

    Strategy
    --------
    1. Split on heading hierarchy (H1 → H2 → H3) using
       ``MarkdownHeaderTextSplitter`` from langchain-text-splitters.
       Each section becomes its own logical unit, carrying its full
       heading breadcrumb as metadata.

    2. If a section still exceeds ``chunk_size`` tokens, fall back to
       the token-window splitter (same as TextChunker) so no section
       ever overflows the embedding model's context.

    3. The heading breadcrumb is *prepended* to every chunk's text so
       the LLM always knows which section the content belongs to, even
       when the chunk is retrieved in isolation.

    Parameters
    ----------
    chunk_size:
        Maximum number of tokens per final chunk.
    chunk_overlap:
        Token overlap used only during the overflow fallback split.
    encoding_name:
        tiktoken encoding name.
    """

    _HEADERS: ClassVar[list[tuple[str, str]]] = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        from langchain_text_splitters import MarkdownHeaderTextSplitter

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._enc = tiktoken.get_encoding(encoding_name)
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._HEADERS,
            strip_headers=True,   # headings go into metadata, not the text body
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, doc: Document, start_index: int = 0) -> list[Chunk]:
        """Return semantically-chunked Chunks for a markdown Document."""
        sections = self._header_splitter.split_text(doc.content)

        chunks: list[Chunk] = []
        global_index = start_index

        for section in sections:
            breadcrumb = self._breadcrumb(section.metadata)
            body = section.page_content.strip()

            if not body:
                continue

            # Prepend breadcrumb so every chunk is self-contained
            full_text = f"{breadcrumb}\n\n{body}" if breadcrumb else body

            token_count = len(self._enc.encode(full_text))

            if token_count <= self.chunk_size:
                # Section fits in one chunk — no further splitting needed
                chunk = Chunk(
                    text=full_text,
                    source=doc.source,
                    chunk_index=global_index,
                    metadata={
                        **doc.metadata,
                        **self._heading_meta(section.metadata),
                        "breadcrumb": breadcrumb,
                    },
                    doc_type=doc.doc_type,
                )
                _logger.debug(
                    "Chunk %d from %s: %d tokens (breadcrumb: %s)",
                    global_index,
                    doc.source,
                    token_count,
                    breadcrumb,
                )
                chunks.append(chunk)
                global_index += 1
            else:
                # Section too large — fall back to token-window overflow split
                sub_chunks = self._overflow_split(full_text)
                for sub_text in sub_chunks:
                    chunks.append(
                        Chunk(
                            text=sub_text,
                            source=doc.source,
                            chunk_index=global_index,
                            metadata={
                                **doc.metadata,
                                **self._heading_meta(section.metadata),
                                "breadcrumb": breadcrumb,
                            },
                            doc_type=doc.doc_type,
                        )
                    )
                    global_index += 1

        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Chunk all markdown documents and return a flat list of Chunks.

        Chunk indices are offset across documents so IDs stay unique even when
        several files share the same stem.
        """
        all_chunks: list[Chunk] = []
        offset = 0
        for doc in docs:
            chunks = self.chunk_document(doc, start_index=offset)
            all_chunks.extend(chunks)
            offset += len(chunks)
        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _breadcrumb(self, header_meta: dict) -> str:
        """Build a heading breadcrumb string, e.g. 'Chapter 1 > Attention > Multi-Head'."""
        parts = [header_meta[k] for k in ("h1", "h2", "h3") if header_meta.get(k)]
        return " > ".join(parts)

    def _heading_meta(self, header_meta: dict) -> dict:
        """Extract only the heading keys that are present."""
        return {k: v for k, v in header_meta.items() if k in ("h1", "h2", "h3") and v}

    def _overflow_split(self, text: str) -> list[str]:
        """Token-window split for sections that exceed chunk_size."""
        tokens = self._enc.encode(text)
        step = self.chunk_size - self.chunk_overlap
        result: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            result.append(self._enc.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += step
        return result
