"""Parse Obsidian Markdown (.md) files into Documents.

Handles:
- YAML frontmatter (tags, aliases, creation date, etc.)
- Wikilinks  [[Note Title]]  →  stripped to plain text
- Block references  ^block-id  →  stripped
- Callouts  > [!NOTE] ...  →  kept as plain text
"""

import logging
import re
from pathlib import Path

import frontmatter  # python-frontmatter

from nextsearch.ingestion.models import Document

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_BLOCK_REF_RE = re.compile(r"\s*\^[a-zA-Z0-9\-]+\s*$", re.MULTILINE)
_CALLOUT_RE = re.compile(r"^>\s*\[![A-Z]+\]\s*", re.MULTILINE)


def _clean_obsidian(text: str) -> str:
    """Remove / normalise Obsidian-specific syntax."""
    # Wikilinks: [[Page|Alias]] → Alias, [[Page]] → Page
    text = _WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    # Block references
    text = _BLOCK_REF_RE.sub("", text)
    # Callout markers → empty
    text = _CALLOUT_RE.sub("> ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_markdown_file(path: Path) -> Document:
    """Parse a single .md file and return a Document."""
    raw = path.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)

    metadata = dict(post.metadata)
    metadata.setdefault("title", path.stem)
    metadata["file_name"] = path.name

    content = _clean_obsidian(post.content)

    return Document(
        source=path.resolve(),
        content=content,
        metadata=metadata,
        doc_type="markdown",
    )


def parse_markdown_dir(directory: Path) -> list[Document]:
    """Recursively parse all .md files in a directory."""
    docs: list[Document] = []
    for md_path in sorted(directory.rglob("*.md")):
        try:
            docs.append(parse_markdown_file(md_path))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Skipping %s: %s", md_path, exc)
    return docs
