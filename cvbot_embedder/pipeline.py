"""Orchestration of the ingestion pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .chunking import DocumentChunker
from .config import Settings
from .embeddings import build_embeddings
from .loader import load_documents
from .metadata import collect_schema
from .vector_store import create_client, index_chunks, recreate_collection

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingResult:
    """Result of a pipeline run.

    Attributes:
        documents: Number of documents read.
        chunks: Number of chunks indexed.
        collection: Name of the collection that was written.
        metadata_fields: Number of section metadata fields published as the
            filterable schema.
    """

    documents: int
    chunks: int
    collection: str
    metadata_fields: int = 0


def run(settings: Settings) -> IndexingResult:
    """Runs the complete ingestion.

    Steps: load documents, chunk them by format and token count, collect the
    metadata schema the sections carry, recreate the target collection and
    index the chunks in batches.

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
    metadata_schema = collect_schema(chunks)

    client = create_client(settings)
    store = recreate_collection(
        client,
        settings.collection_name,
        embeddings,
        settings.embedding_model_id,
        metadata_schema,
    )
    indexed = index_chunks(store, chunks, settings.batch_size)

    LOGGER.info(
        "ingestion finished: %d document(s), %d chunk(s), %d metadata field(s), "
        "collection %r",
        len(documents),
        indexed,
        len(metadata_schema),
        settings.collection_name,
    )
    return IndexingResult(
        documents=len(documents),
        chunks=indexed,
        collection=settings.collection_name,
        metadata_fields=len(metadata_schema),
    )
