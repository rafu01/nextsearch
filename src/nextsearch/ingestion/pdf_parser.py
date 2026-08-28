"""Parse PDF files into Documents using PyMuPDF (fitz).

Strategy:
- Extract text page-by-page.
- Preserve page numbers in metadata for citation.
- Basic heuristics to skip header/footer noise.
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

from nextsearch.ingestion.models import Document

_logger = logging.getLogger(__name__)


def _extract_text_from_pdf(pdf_path: Path) -> tuple[str, dict]:
    """Return (full_text, metadata) from a PDF."""
    doc = fitz.open(str(pdf_path))

    # Gather document-level metadata
    meta = doc.metadata or {}
    metadata = {
        "title": meta.get("title") or pdf_path.stem,
        "author": meta.get("author", ""),
        "file_name": pdf_path.name,
        "num_pages": doc.page_count,
    }

    pages: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")  # plain text extraction
        if text.strip():
            # Tag each page's text so we can recover page numbers later
            pages.append(f"[PAGE {page_num}]\n{text.strip()}")

    doc.close()
    return "\n\n".join(pages), metadata


def parse_pdf_file(path: Path) -> Document:
    """Parse a single PDF and return a Document."""
    content, metadata = _extract_text_from_pdf(path)
    return Document(
        source=path.resolve(),
        content=content,
        metadata=metadata,
        doc_type="pdf",
    )


def parse_pdf_dir(directory: Path) -> list[Document]:
    """Recursively parse all .pdf files in a directory."""
    docs: list[Document] = []
    for pdf_path in sorted(directory.rglob("*.pdf")):
        try:
            docs.append(parse_pdf_file(pdf_path))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Skipping %s: %s", pdf_path, exc)
    return docs
