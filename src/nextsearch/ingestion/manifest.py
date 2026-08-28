"""Incremental ingestion manifest.

Tracks which source files have been indexed and which chunk IDs belong to each
file, so re-ingestion can skip unchanged files and delete chunks for removed
or modified files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


def hash_file(path: Path) -> str:
    """Return a stable SHA-256 hash for a file's contents."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha.update(block)
    return f"sha256:{sha.hexdigest()}"


@dataclass
class FileEntry:
    """Manifest entry for a single source file."""

    hash: str
    chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"hash": self.hash, "chunk_ids": self.chunk_ids}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileEntry:
        return cls(hash=data["hash"], chunk_ids=list(data.get("chunk_ids", [])))


@dataclass
class IngestionManifest:
    """Persisted map of source files → chunk IDs + content hashes."""

    version: int = MANIFEST_VERSION
    files: dict[str, FileEntry] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "files": {path: entry.to_dict() for path, entry in self.files.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestionManifest:
        return cls(
            version=data.get("version", MANIFEST_VERSION),
            files={
                path: FileEntry.from_dict(entry)
                for path, entry in data.get("files", {}).items()
            },
        )

    def load(self, path: Path) -> None:
        """Load manifest from disk, or start empty if it doesn't exist."""
        if not path.exists():
            self.files = {}
            return
        with path.open("r", encoding="utf-8") as f:
            loaded = self.from_dict(json.load(f))
        self.version = loaded.version
        self.files = loaded.files

    def save(self, path: Path) -> None:
        """Persist manifest to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # Diffing
    # ------------------------------------------------------------------

    def diff(
        self,
        current_hashes: dict[str, str],
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        """Compare the manifest against the current filesystem.

        Returns
        -------
        unchanged, new, changed, deleted
            Sets of absolute file paths.
        """
        existing = set(self.files.keys())
        current = set(current_hashes.keys())

        new = current - existing
        deleted = existing - current
        unchanged = {
            path
            for path in existing & current
            if self.files[path].hash == current_hashes[path]
        }
        changed = (existing & current) - unchanged

        return unchanged, new, changed, deleted
