"""Unit tests for the TextChunker."""

from pathlib import Path

from nextsearch.ingestion.chunker import TextChunker
from nextsearch.ingestion.models import Document


def make_doc(content: str) -> Document:
    return Document(source=Path("/fake/note.md"), content=content, doc_type="markdown")


def test_empty_document():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(make_doc(""))
    assert chunks == []


def test_single_chunk_for_short_text():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(make_doc("Hello world"))
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_multiple_chunks():
    # 600 tokens > chunk_size=512, so we should get 2 chunks
    chunker = TextChunker(chunk_size=50, chunk_overlap=5)
    long_text = " ".join(["word"] * 200)
    chunks = chunker.chunk_document(make_doc(long_text))
    assert len(chunks) > 1


def test_overlap_makes_first_tokens_of_chunk2_same_as_last_tokens_of_chunk1():
    """Verify overlap by comparing at the tiktoken token level, not word level.

    Tiktoken may split a word like 'tok10' into multiple tokens, so naive
    .split() comparisons across chunk boundaries are not reliable.
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    chunk_size = 10
    chunk_overlap = 3

    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Use simple single-token words to make token boundaries predictable
    words = ["hi"] * 40          # "hi" is reliably one token in cl100k_base
    text = " ".join(words)
    doc = make_doc(text)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2

    tokens_c0 = enc.encode(chunks[0].text)
    tokens_c1 = enc.encode(chunks[1].text)

    # The last `chunk_overlap` tokens of chunk 0 should match
    # the first `chunk_overlap` tokens of chunk 1.
    assert tokens_c0[-chunk_overlap:] == tokens_c1[:chunk_overlap]


def test_chunk_id_format():
    import hashlib

    chunker = TextChunker(chunk_size=50, chunk_overlap=5)
    doc = make_doc("Some content " * 30)
    chunks = chunker.chunk_document(doc)
    path_hash = hashlib.sha256(str(doc.source).encode("utf-8")).hexdigest()[:8]
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"note_{path_hash}__{i}"
