"""Unit tests for the Markdown / Obsidian parser."""

import textwrap
from pathlib import Path

import pytest

from nextsearch.ingestion.markdown_parser import parse_markdown_file


@pytest.fixture()
def tmp_md(tmp_path):
    """Write a markdown file and return its path."""
    def _write(content: str) -> Path:
        p = tmp_path / "test_note.md"
        p.write_text(content, encoding="utf-8")
        return p
    return _write


def test_basic_frontmatter(tmp_md):
    content = textwrap.dedent("""\
        ---
        title: My Note
        tags: [python, rag]
        ---
        Hello world
    """)
    doc = parse_markdown_file(tmp_md(content))
    assert doc.metadata["title"] == "My Note"
    assert "Hello world" in doc.content


def test_wikilink_single(tmp_md):
    doc = parse_markdown_file(tmp_md("See [[Other Note]] for details."))
    assert "Other Note" in doc.content
    assert "[[" not in doc.content


def test_wikilink_with_alias(tmp_md):
    doc = parse_markdown_file(tmp_md("See [[Other Note|this note]] for details."))
    assert "this note" in doc.content
    assert "[[" not in doc.content


def test_block_reference_stripped(tmp_md):
    doc = parse_markdown_file(tmp_md("Some content ^my-block-ref"))
    assert "^my-block-ref" not in doc.content


def test_no_frontmatter(tmp_md):
    doc = parse_markdown_file(tmp_md("Just a plain note."))
    assert "Just a plain note." in doc.content
    assert doc.metadata["title"] == "test_note"
