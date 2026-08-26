"""Shared data models for ingested documents."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """A single parsed document before chunking."""

    source: Path           # absolute path to the source file
    content: str           # raw text content
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_type: str = "unknown"  # "markdown" | "pdf"


@dataclass
class Chunk:
    """A text chunk ready for embedding."""

    text: str
    source: Path
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_type: str = "unknown"

    @property
    def chunk_id(self) -> str:
        """Stable unique ID for this chunk."""
        return f"{self.source.stem}__{self.chunk_index}"
