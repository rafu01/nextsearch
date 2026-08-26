#!/usr/bin/env python
"""Standalone ask script — useful for quick one-off queries.

Run:
    python scripts/ask.py "What is the attention mechanism?"
"""

import sys
from nextsearch.pipeline import RAGPipeline
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    pipeline = RAGPipeline()
    answer = pipeline.ask(query, verbose=True)
    console.print(Panel(Markdown(answer), title="Answer", border_style="green"))
