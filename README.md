# nextsearch 🔍

A **RAG (Retrieval-Augmented Generation)** system built from scratch to let you query your **Obsidian notes** and **PDF textbooks** using natural language.

## Stack

| Layer | Technology |
|---|---|
| **Parsing** | `pymupdf` (PDFs) · `python-frontmatter` (Markdown) |
| **Chunking** | Token-aware sliding window via `tiktoken` |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector Store** | ChromaDB (local, persistent) |
| **Generation** | Google Gemini 1.5 Pro |
| **CLI** | Typer + Rich |

## Project Structure

```
nextsearch/
├── src/nextsearch/
│   ├── config.py               # All settings via .env
│   ├── pipeline.py             # High-level RAGPipeline
│   ├── ingestion/
│   │   ├── models.py           # Document & Chunk dataclasses
│   │   ├── markdown_parser.py  # Obsidian .md parser
│   │   ├── pdf_parser.py       # PDF parser (PyMuPDF)
│   │   └── chunker.py          # Token-aware chunker
│   ├── embedding/
│   │   └── openai_embedder.py  # OpenAI embedding API
│   ├── vector_store/
│   │   ├── base.py             # Abstract VectorStore interface
│   │   └── chroma_store.py     # ChromaDB implementation
│   ├── retrieval/
│   │   └── retriever.py        # Query → top-k chunks
│   └── generation/
│       └── gemini_generator.py # Gemini answer generation
├── data/
│   ├── raw/obsidian/           # Drop your .md vault files here
│   └── raw/pdfs/               # Drop your PDF textbooks here
├── tests/                      # pytest test suite
├── scripts/
│   ├── ingest.py               # Standalone ingest runner
│   └── ask.py                  # Standalone query runner
├── .env.example                # Copy to .env and fill in API keys
└── pyproject.toml
```

## Quick Start

### 1. Set up environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY and GOOGLE_API_KEY
```

### 3. Add your documents

```
data/raw/obsidian/   ← copy your .md files / Obsidian vault here
data/raw/pdfs/       ← copy your PDF textbooks here
```

### 4. Ingest

```bash
nextsearch ingest
# or with custom paths:
nextsearch ingest --obsidian ~/my-vault --pdfs ~/books
```

### 5. Ask questions

```bash
# One-shot
nextsearch ask "What is the attention mechanism in transformers?"

# Interactive mode
nextsearch ask

# With verbose context
nextsearch ask "Explain backpropagation" --verbose
```

### Other commands

```bash
nextsearch stats    # show vector store size
nextsearch reset    # wipe all indexed data
```

## Running Tests

```bash
pytest tests/ -v
```

## Extending the Stack

The `VectorStore` base class (`src/nextsearch/vector_store/base.py`) defines a minimal interface. To swap to FAISS or Qdrant, implement the four abstract methods and update `pipeline.py`.
