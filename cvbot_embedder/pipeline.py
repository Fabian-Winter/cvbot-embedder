"""Orchestration of the ingestion pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .chunking import DocumentChunker
from .config import Settings
from .embeddings import build_embeddings
from .loader import load_documents
from .vector_store import create_client, index_chunks, recreate_collection

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingResult:
    """Result of a pipeline run.

    Attributes:
        documents: Number of documents read.
        chunks: Number of chunks indexed.
        collection: Name of the collection that was written.
    """

    documents: int
    chunks: int
    collection: str


def run(settings: Settings) -> IndexingResult:
    """Runs the complete ingestion.

    Steps: load documents, chunk them by format and token count, recreate the
    target collection and index the chunks in batches.

    Args:
        settings: Runtime configuration.

    Returns:
        The metrics of the run.
    """
    documents = load_documents(settings.documents_dir)
    if not documents:
        LOGGER.warning("no documents found in %s", settings.documents_dir)

    embeddings = build_embeddings(settings)
    chunker = DocumentChunker(
        max_tokens=settings.max_chunk_tokens,
        token_overlap=settings.token_chunk_overlap,
    )
    chunks = chunker.split(documents)

    client = create_client(settings)
    store = recreate_collection(
        client, settings.collection_name, embeddings, settings.embedding_model_id
    )
    indexed = index_chunks(store, chunks, settings.batch_size)

    LOGGER.info(
        "ingestion finished: %d document(s), %d chunk(s), collection %r",
        len(documents),
        indexed,
        settings.collection_name,
    )
    return IndexingResult(
        documents=len(documents),
        chunks=indexed,
        collection=settings.collection_name,
    )
