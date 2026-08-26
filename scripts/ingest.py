#!/usr/bin/env python
"""Standalone ingest script — useful if you don't want the CLI.

Edit the paths below, then run:
    python scripts/ingest.py
"""

from pathlib import Path
from nextsearch.pipeline import RAGPipeline

OBSIDIAN_DIR = Path("data/raw/obsidian")
PDF_DIR = Path("data/raw/pdfs")

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.ingest(obsidian_dir=OBSIDIAN_DIR, pdf_dir=PDF_DIR, reset=False)
