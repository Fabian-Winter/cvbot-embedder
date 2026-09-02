"""Tests for reading the documents."""

from __future__ import annotations

from pathlib import Path

import pytest

from cvbot_embedder.loader import load_documents


def test_loads_supported_files_recursively(documents_dir: Path) -> None:
    documents = load_documents(documents_dir)

    sources = {document.metadata["source"] for document in documents}
    assert sources == {"a.txt", "nested/b.md"}


def test_skips_empty_files(documents_dir: Path) -> None:
    documents = load_documents(documents_dir)

    assert all("empty" not in document.metadata["source"] for document in documents)


def test_metadata_contains_relative_source_and_filename(documents_dir: Path) -> None:
    documents = load_documents(documents_dir)

    nested = next(d for d in documents if d.metadata["filename"] == "b.md")
    assert nested.metadata["source"] == "nested/b.md"


def test_content_is_read_verbatim(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("# Heading\n\nContent.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert documents[0].page_content == "# Heading\n\nContent."


def test_skips_non_utf8_files(tmp_path: Path) -> None:
    (tmp_path / "latin.txt").write_bytes(b"\xff\xfe invalid")

    assert load_documents(tmp_path) == []


def test_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    assert load_documents(tmp_path) == []


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")


def test_file_instead_of_directory_raises(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("content", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        load_documents(path)
