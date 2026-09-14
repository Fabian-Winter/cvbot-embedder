"""Shared fixtures and test doubles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from cvbot_core.testing import FakeEmbeddings
from langchain_core.documents import Document

from cvbot_embedder.config import Settings


class FakeStore:
    """Vector store double that records the batches it receives."""

    def __init__(self) -> None:
        """Initializes the empty store."""
        self.batches: list[list[Document]] = []

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Accepts a batch.

        Args:
            documents: The chunks to write.

        Returns:
            The generated pseudo IDs.
        """
        self.batches.append(list(documents))
        return [str(i) for i in range(len(documents))]

    @property
    def documents(self) -> list[Document]:
        """All written chunks in write order."""
        return [doc for batch in self.batches for doc in batch]


class FakeChromaClient:
    """Chroma client double that records delete calls."""

    def __init__(self, existing: set[str] | None = None) -> None:
        """Initializes the client.

        Args:
            existing: Names of already existing collections.
        """
        self.existing = set(existing or ())
        self.deleted: list[str] = []

    def delete_collection(self, name: str) -> None:
        """Deletes a collection.

        Args:
            name: Name of the collection.

        Raises:
            ValueError: If the collection does not exist.
        """
        self.deleted.append(name)
        if name not in self.existing:
            raise ValueError(f"Collection {name} does not exist")
        self.existing.remove(name)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    """Provides a deterministic embedding model."""
    return FakeEmbeddings()


@pytest.fixture
def documents_dir(tmp_path: Path) -> Path:
    """Creates a documents directory holding sample files.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        The path of the created directory.
    """
    root = tmp_path / "documents"
    (root / "nested").mkdir(parents=True)
    (root / "a.txt").write_text(
        "The first sentence introduces the topic. The second one expands on it.",
        encoding="utf-8",
    )
    (root / "nested" / "b.md").write_text(
        "# Title\n\nA paragraph with content. Another paragraph with content.",
        encoding="utf-8",
    )
    (root / "ignore.pdf").write_bytes(b"%PDF-1.4 binary content")
    (root / "empty.txt").write_text("   \n", encoding="utf-8")
    return root


@pytest.fixture
def settings(documents_dir: Path) -> Settings:
    """Provides a configuration for tests.

    Args:
        documents_dir: The temporary documents directory.

    Returns:
        The configuration.
    """
    return Settings(
        documents_dir=documents_dir,
        chroma_host="chroma.internal",
        chroma_port=8000,
        collection_name="test_collection",
        max_chunk_tokens=64,
        token_chunk_overlap=8,
        batch_size=2,
    )


def make_documents(*contents: str, source: str = "doc.txt") -> list[Document]:
    """Creates documents with uniform metadata.

    Args:
        *contents: The text contents.
        source: Value of the ``source`` metadata field.

    Returns:
        The created documents.
    """
    return [
        Document(page_content=content, metadata={"source": source, "filename": source})
        for content in contents
    ]


def collect_metadata(documents: list[Document], key: str) -> list[Any]:
    """Reads a metadata field from several documents.

    Args:
        documents: The documents.
        key: The metadata key.

    Returns:
        The values in document order.
    """
    return [document.metadata.get(key) for document in documents]
