"""Tests for the incremental ingestion manifest."""

from pathlib import Path

from nextsearch.ingestion.manifest import FileEntry, IngestionManifest, hash_file


def test_hash_file_detects_changes(tmp_path: Path) -> None:
    note = tmp_path / "note.md"
    note.write_text("hello world", encoding="utf-8")

    first_hash = hash_file(note)
    assert first_hash.startswith("sha256:")

    note.write_text("goodbye world", encoding="utf-8")
    second_hash = hash_file(note)

    assert second_hash != first_hash


def test_manifest_diff() -> None:
    manifest = IngestionManifest(
        files={
            "/vault/old.md": FileEntry(hash="aaa", chunk_ids=["old__0"]),
            "/vault/unchanged.md": FileEntry(hash="bbb", chunk_ids=["unchanged__0"]),
            "/vault/changed.md": FileEntry(hash="ccc-old", chunk_ids=["changed__0"]),
        }
    )

    current = {
        "/vault/unchanged.md": "bbb",
        "/vault/changed.md": "ccc-new",
        "/vault/new.md": "ddd",
    }

    unchanged, new, changed, deleted = manifest.diff(current)

    assert unchanged == {"/vault/unchanged.md"}
    assert new == {"/vault/new.md"}
    assert changed == {"/vault/changed.md"}
    assert deleted == {"/vault/old.md"}


def test_manifest_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = IngestionManifest(
        files={
            "/vault/note.md": FileEntry(
                hash="sha256:abc", chunk_ids=["note__0", "note__1"]
            )
        }
    )

    manifest.save(path)

    loaded = IngestionManifest()
    loaded.load(path)

    assert loaded.version == manifest.version
    assert "/vault/note.md" in loaded.files
    assert loaded.files["/vault/note.md"].hash == "sha256:abc"
    assert loaded.files["/vault/note.md"].chunk_ids == ["note__0", "note__1"]


def test_manifest_load_missing_file_starts_empty(tmp_path: Path) -> None:
    manifest = IngestionManifest()
    manifest.load(tmp_path / "does_not_exist.json")
    assert manifest.files == {}
