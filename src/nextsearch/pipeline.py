"""High-level RAG pipeline: wires ingestion → embedding → storage → retrieval → generation."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from nextsearch.config import settings
from nextsearch.embedding.openai_embedder import OpenAIEmbedder
from nextsearch.generation.gemini_generator import GeminiGenerator
from nextsearch.ingestion.chunker import TextChunker
from nextsearch.ingestion.markdown_parser import parse_markdown_dir
from nextsearch.ingestion.pdf_parser import parse_pdf_dir
from nextsearch.retrieval.retriever import Retriever
from nextsearch.vector_store.chroma_store import ChromaVectorStore

console = Console()


class RAGPipeline:
    """End-to-end RAG pipeline.

    Usage
    -----
    >>> pipeline = RAGPipeline()
    >>> pipeline.ingest(obsidian_dir=Path("data/raw/obsidian"),
    ...                 pdf_dir=Path("data/raw/pdfs"))
    >>> answer = pipeline.ask("What is backpropagation?")
    >>> print(answer)
    """

    def __init__(self) -> None:
        self._embedder = OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
        self._store = ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        self._retriever = Retriever(
            embedder=self._embedder,
            vector_store=self._store,
            top_k=settings.top_k,
        )
        self._generator = GeminiGenerator(
            api_key=settings.google_api_key,
            model=settings.gemini_model,
        )
        self._chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        obsidian_dir: Path | None = None,
        pdf_dir: Path | None = None,
        reset: bool = False,
    ) -> None:
        """Parse, chunk, embed, and store all documents.

        Parameters
        ----------
        obsidian_dir:
            Root directory of your Obsidian vault (or a subfolder).
        pdf_dir:
            Directory containing PDF textbooks.
        reset:
            If True, wipe the vector store before ingesting.
        """
        if reset:
            console.print("[yellow]Resetting vector store…[/yellow]")
            self._store.reset()

        docs = []

        if obsidian_dir and obsidian_dir.exists():
            console.print(f"[cyan]Parsing Obsidian notes from {obsidian_dir}…[/cyan]")
            docs.extend(parse_markdown_dir(obsidian_dir))

        if pdf_dir and pdf_dir.exists():
            console.print(f"[cyan]Parsing PDFs from {pdf_dir}…[/cyan]")
            docs.extend(parse_pdf_dir(pdf_dir))

        if not docs:
            console.print("[red]No documents found. Check your data directories.[/red]")
            return

        console.print(f"[green]Loaded {len(docs)} document(s). Chunking…[/green]")
        chunks = self._chunker.chunk_documents(docs)
        console.print(f"[green]Created {len(chunks)} chunk(s). Embedding…[/green]")

        embeddings = self._embedder.embed_chunks(chunks)

        self._store.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            texts=[c.text for c in chunks],
            metadatas=[{**c.metadata, "doc_type": c.doc_type} for c in chunks],
        )

        console.print(
            f"[bold green]✓ Ingestion complete.[/bold green] "
            f"Vector store now has {self._store.count()} chunk(s)."
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def ask(self, query: str, top_k: int | None = None, verbose: bool = False) -> str:
        """Ask a question and return the generated answer.

        Parameters
        ----------
        query:
            Your technical question.
        top_k:
            Override the default number of retrieved chunks.
        verbose:
            If True, print the retrieved context before the answer.
        """
        results = self._retriever.retrieve(query, top_k=top_k)

        if verbose:
            console.print(Panel.fit("[bold]Retrieved Context[/bold]", style="blue"))
            for i, r in enumerate(results, 1):
                source = r.metadata.get("file_name", r.chunk_id)
                console.print(f"[dim][{i}] {source}  (score={r.score:.3f})[/dim]")
                console.print(r.text[:300] + ("…" if len(r.text) > 300 else ""))
                console.rule()

        answer = self._generator.generate(query, results)
        return answer
