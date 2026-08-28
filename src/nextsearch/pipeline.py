"""High-level RAG pipeline: wires ingestion → embedding → storage → retrieval → generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from nextsearch.config import settings
from nextsearch.embedding.openai_embedder import OpenAIEmbedder
from nextsearch.generation.gemini_generator import GeminiGenerator
from nextsearch.ingestion.chunker import MarkdownChunker, TextChunker
from nextsearch.ingestion.manifest import FileEntry, IngestionManifest, hash_file
from nextsearch.ingestion.markdown_parser import parse_markdown_dir
from nextsearch.ingestion.models import Chunk, Document
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
        self._markdown_chunker = MarkdownChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._text_chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._manifest_path = settings.chroma_persist_dir / "ingestion_manifest.json"
        self._manifest = IngestionManifest()
        self._manifest.load(self._manifest_path)

    # ------------------------------------------------------------------
    # Ingestion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_files(obsidian_dir: Path | None, pdf_dir: Path | None) -> dict[str, str]:
        """Return a map of resolved file path → content hash."""
        files: dict[str, str] = {}
        if obsidian_dir and obsidian_dir.exists():
            for path in sorted(obsidian_dir.rglob("*.md")):
                files[str(path.resolve())] = hash_file(path)
        if pdf_dir and pdf_dir.exists():
            for path in sorted(pdf_dir.rglob("*.pdf")):
                files[str(path.resolve())] = hash_file(path)
        return files

    @staticmethod
    def _is_under_any(path: str, roots: list[Path]) -> bool:
        """Check whether *path* lives under any of the provided root directories."""
        path_obj = Path(path)
        return any(path_obj.is_relative_to(root) for root in roots)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        obsidian_dir: Path | None = None,
        pdf_dir: Path | None = None,
        reset: bool = False,
    ) -> None:
        """Parse, chunk, embed, and store documents incrementally.

        Unchanged files are skipped. New, changed, and deleted files are handled
        automatically by comparing file content hashes against the ingestion
        manifest.

        Parameters
        ----------
        obsidian_dir:
            Root directory of your Obsidian vault (or a subfolder).
        pdf_dir:
            Directory containing PDF textbooks.
        reset:
            If True, wipe the vector store and manifest before ingesting.
        """
        if reset:
            console.print("[yellow]Resetting vector store and manifest…[/yellow]")
            self._store.reset()
            self._manifest = IngestionManifest()
            self._manifest.save(self._manifest_path)

        # Roots currently being synced; files outside these roots are left alone.
        roots: list[Path] = []
        if obsidian_dir and obsidian_dir.exists():
            roots.append(obsidian_dir.resolve())
        if pdf_dir and pdf_dir.exists():
            roots.append(pdf_dir.resolve())

        current_files = self._scan_files(obsidian_dir, pdf_dir)

        # Only diff files that live under the directories we are ingesting now.
        relevant_existing = {
            path
            for path in self._manifest.files
            if self._is_under_any(path, roots)
        }
        view = IngestionManifest(
            files={path: self._manifest.files[path] for path in relevant_existing}
        )
        _unchanged, new, changed, deleted = view.diff(current_files)

        # Remove chunks for deleted files.
        if deleted:
            ids_to_delete: list[str] = []
            for path in deleted:
                ids_to_delete.extend(self._manifest.files[path].chunk_ids)
            if ids_to_delete:
                console.print(
                    f"[yellow]Removing chunks for {len(deleted)} deleted file(s)…[/yellow]"
                )
                self._store.delete(ids_to_delete)
            for path in deleted:
                del self._manifest.files[path]

        to_process = new | changed
        if not to_process:
            self._manifest.save(self._manifest_path)
            console.print("[green]No changes detected. Nothing to ingest.[/green]")
            return

        console.print(
            f"[cyan]Processing {len(to_process)} file(s) "
            f"({len(new)} new, {len(changed)} changed)…[/cyan]"
        )

        docs: list[Document] = []
        if obsidian_dir and obsidian_dir.exists():
            docs.extend(parse_markdown_dir(obsidian_dir))
        if pdf_dir and pdf_dir.exists():
            docs.extend(parse_pdf_dir(pdf_dir))

        docs = [d for d in docs if str(d.source) in to_process]

        if not docs:
            console.print("[red]No documents found. Check your data directories.[/red]")
            self._manifest.save(self._manifest_path)
            return

        console.print(f"[green]Loaded {len(docs)} document(s). Chunking…[/green]")
        markdown_docs = [d for d in docs if d.doc_type == "markdown"]
        pdf_docs = [d for d in docs if d.doc_type != "markdown"]

        chunks: list[Chunk] = []
        if markdown_docs:
            chunks.extend(self._markdown_chunker.chunk_documents(markdown_docs))
        if pdf_docs:
            chunks.extend(self._text_chunker.chunk_documents(pdf_docs))
        console.print(f"[green]Created {len(chunks)} chunk(s). Embedding…[/green]")

        embeddings = self._embedder.embed_chunks(chunks)

        self._store.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            texts=[c.text for c in chunks],
            metadatas=[{**c.metadata, "doc_type": c.doc_type} for c in chunks],
        )

        # Update manifest with the new chunk IDs for each processed file.
        file_chunk_ids: dict[str, list[str]] = defaultdict(list)
        for chunk in chunks:
            file_chunk_ids[str(chunk.source)].append(chunk.chunk_id)

        for path in to_process:
            self._manifest.files[path] = FileEntry(
                hash=current_files[path],
                chunk_ids=file_chunk_ids.get(path, []),
            )

        self._manifest.save(self._manifest_path)

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
