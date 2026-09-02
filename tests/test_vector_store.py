"""Tests for the ChromaDB access layer."""

from __future__ import annotations

import pytest

from cvbot_embedder import vector_store
from cvbot_embedder.config import Settings
from tests.conftest import FakeChromaClient, FakeEmbeddings, FakeStore, make_documents


class RecordingHttpClient:
    """Captures the arguments passed to ``chromadb.HttpClient``."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_create_client_passes_connection_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        vector_store.chromadb, "HttpClient", lambda **kw: captured.update(kw)
    )
    settings = Settings(
        chroma_host="chroma.internal", chroma_port=8443, chroma_ssl=True
    )

    vector_store.create_client(settings)

    assert captured["host"] == "chroma.internal"
    assert captured["port"] == 8443
    assert captured["ssl"] is True
    assert captured["headers"] is None


def test_create_client_sets_bearer_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        vector_store.chromadb, "HttpClient", lambda **kw: captured.update(kw)
    )

    vector_store.create_client(Settings(chroma_auth_token="secret"))

    assert captured["headers"] == {"Authorization": "Bearer secret"}


def test_recreate_collection_deletes_existing(
    monkeypatch: pytest.MonkeyPatch, fake_embeddings: FakeEmbeddings
) -> None:
    monkeypatch.setattr(vector_store, "Chroma", lambda **kw: kw)
    client = FakeChromaClient(existing={"jobs"})

    store = vector_store.recreate_collection(client, "jobs", fake_embeddings)

    assert client.deleted == ["jobs"]
    assert store["collection_name"] == "jobs"
    assert store["embedding_function"] is fake_embeddings


def test_recreate_collection_tolerates_missing_collection(
    monkeypatch: pytest.MonkeyPatch, fake_embeddings: FakeEmbeddings
) -> None:
    monkeypatch.setattr(vector_store, "Chroma", lambda **kw: kw)
    client = FakeChromaClient(existing=set())

    store = vector_store.recreate_collection(client, "jobs", fake_embeddings)

    assert store["collection_name"] == "jobs"


def test_index_chunks_writes_in_batches() -> None:
    store = FakeStore()
    chunks = make_documents(*[f"Chunk {i}" for i in range(5)])

    written = vector_store.index_chunks(store, chunks, batch_size=2)

    assert written == 5
    assert [len(batch) for batch in store.batches] == [2, 2, 1]
    assert store.documents == chunks


def test_index_chunks_without_data_does_not_write() -> None:
    store = FakeStore()

    assert vector_store.index_chunks(store, [], batch_size=2) == 0
    assert store.batches == []
