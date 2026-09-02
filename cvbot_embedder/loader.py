"""Loading of local text documents."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

LOGGER = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = frozenset({".txt", ".md"})


def load_documents(directory: Path) -> list[Document]:
    """Reads all supported text files of a directory recursively.

    Supported extensions are ``.txt`` and ``.md``. Files without content and
    files that cannot be decoded as UTF-8 are skipped.

    Args:
        directory: Root directory that is searched.

    Returns:
        The loaded documents carrying the metadata ``source`` (path relative to
        the root directory) and ``filename``, sorted by path.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"documents directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"path is not a directory: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            LOGGER.warning("file is not UTF-8 encoded, skipped: %s", path)
            continue
        if not content.strip():
            LOGGER.warning("file is empty, skipped: %s", path)
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(path.relative_to(directory)),
                    "filename": path.name,
                },
            )
        )

    LOGGER.info("loaded %d document(s) from %s", len(documents), directory)
    return documents
