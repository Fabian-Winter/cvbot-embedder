"""Access to the ChromaDB instance (container running on AWS Fargate)."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import Settings

LOGGER = logging.getLogger(__name__)


def create_client(settings: Settings) -> chromadb.ClientAPI:
    """Creates an HTTP client for the ChromaDB instance.

    Args:
        settings: Runtime configuration holding host and port.

    Returns:
        The connected Chroma client.
    """
    LOGGER.info(
        "connecting to ChromaDB: %s:%d",
        settings.chroma_host,
        settings.chroma_port,
    )
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )


def recreate_collection(
    client: chromadb.ClientAPI,
    collection_name: str,
    embeddings: Embeddings,
) -> Chroma:
    """Drops an existing collection and creates it again.

    This guarantees that after each run the store contains only the chunks of
    the current document set.

    Args:
        client: The Chroma client.
        collection_name: Name of the collection.
        embeddings: Embedding model used by the store.

    Returns:
        The empty, writable vector store.
    """
    try:
        client.delete_collection(collection_name)
        LOGGER.info("deleted existing collection %r", collection_name)
    except Exception:  # Chroma raises different errors depending on version.
        LOGGER.info("collection %r did not exist yet", collection_name)

    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def index_chunks(store: Any, chunks: list[Document], batch_size: int) -> int:
    """Writes chunks to the vector store in batches.

    Args:
        store: Target store exposing an ``add_documents`` method.
        chunks: The chunks to index.
        batch_size: Number of chunks per write.

    Returns:
        The number of chunks written.
    """
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        store.add_documents(batch)
        LOGGER.debug("batch written: %d chunk(s)", len(batch))

    LOGGER.info("indexed %d chunk(s)", len(chunks))
    return len(chunks)
