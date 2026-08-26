"""Command-line interface for nextsearch.

Commands
--------
  nextsearch ingest   — parse & embed your docs into the vector store
  nextsearch ask      — ask a question (interactive or one-shot)
  nextsearch stats    — show vector store stats
  nextsearch reset    — wipe the vector store
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from nextsearch.pipeline import RAGPipeline

app = typer.Typer(
    name="nextsearch",
    help="RAG over your Obsidian notes and PDF textbooks.",
    add_completion=False,
)
console = Console()

# Lazy-loaded pipeline singleton
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@app.command()
def ingest(
    obsidian_dir: Path = typer.Option(
        Path("data/raw/obsidian"),
        "--obsidian", "-o",
        help="Path to your Obsidian vault (or subfolder).",
        exists=False,
    ),
    pdf_dir: Path = typer.Option(
        Path("data/raw/pdfs"),
        "--pdfs", "-p",
        help="Path to your PDF textbooks directory.",
        exists=False,
    ),
    reset: bool = typer.Option(False, "--reset", help="Wipe vector store before ingesting."),
) -> None:
    """Parse, chunk, embed, and index your documents."""
    _get_pipeline().ingest(obsidian_dir=obsidian_dir, pdf_dir=pdf_dir, reset=reset)


@app.command()
def ask(
    question: str = typer.Argument(None, help="Your question (omit for interactive mode)."),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show retrieved context."),
) -> None:
    """Ask a question against your indexed knowledge base."""
    pipeline = _get_pipeline()

    if question:
        answer = pipeline.ask(question, top_k=top_k, verbose=verbose)
        console.print(Panel(Markdown(answer), title="[bold green]Answer[/bold green]", border_style="green"))
    else:
        # Interactive mode
        console.print("[bold cyan]nextsearch interactive mode[/bold cyan] — type 'exit' to quit.\n")
        while True:
            try:
                q = typer.prompt("You")
            except (EOFError, KeyboardInterrupt):
                break
            if q.strip().lower() in {"exit", "quit", "q"}:
                break
            answer = pipeline.ask(q, top_k=top_k, verbose=verbose)
            console.print(Panel(Markdown(answer), title="[bold green]Answer[/bold green]", border_style="green"))


@app.command()
def stats() -> None:
    """Show vector store statistics."""
    from nextsearch.config import settings
    from nextsearch.vector_store.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
    count = store.count()
    console.print(f"[bold]Vector store:[/bold] {settings.chroma_persist_dir}")
    console.print(f"[bold]Collection:[/bold] {settings.chroma_collection_name}")
    console.print(f"[bold]Chunks indexed:[/bold] {count}")


@app.command()
def reset() -> None:
    """Wipe all indexed data from the vector store."""
    confirm = typer.confirm("This will delete ALL indexed documents. Are you sure?")
    if not confirm:
        raise typer.Abort()
    from nextsearch.config import settings
    from nextsearch.vector_store.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )
    store.reset()
    console.print("[yellow]Vector store wiped.[/yellow]")
