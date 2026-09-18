"""End-to-end tests of the pipeline and the command line."""

from __future__ import annotations

from pathlib import Path

import pytest

from cvbot_embedder import __main__ as cli
from cvbot_embedder import pipeline
from cvbot_embedder.config import Settings
from tests.conftest import FakeChromaClient, FakeEmbeddings, FakeStore


@pytest.fixture
def patched_pipeline(
    monkeypatch: pytest.MonkeyPatch, fake_embeddings: FakeEmbeddings
) -> FakeStore:
    """Replaces AWS and ChromaDB access with test doubles.

    Args:
        monkeypatch: pytest fixture for temporarily replacing attributes.
        fake_embeddings: Deterministic embedding model.

    Returns:
        The store the pipeline writes to.
    """
    store = FakeStore()
    client = FakeChromaClient(existing={"test_collection"})
    monkeypatch.setattr(pipeline, "build_embeddings", lambda s: fake_embeddings)
    monkeypatch.setattr(pipeline, "create_client", lambda s: client)
    monkeypatch.setattr(
        pipeline,
        "recreate_collection",
        lambda c, name, emb, model_id, schema=None: store,
    )
    return store


def test_run_indexes_all_chunks(
    settings: Settings, patched_pipeline: FakeStore
) -> None:
    result = pipeline.run(settings)

    assert result.documents == 2
    assert result.chunks == len(patched_pipeline.documents)
    assert result.chunks > 0
    assert result.collection == "test_collection"


def test_run_preserves_source_metadata(
    settings: Settings, patched_pipeline: FakeStore
) -> None:
    pipeline.run(settings)

    sources = {doc.metadata["source"] for doc in patched_pipeline.documents}
    assert sources == {"a.txt", "nested/b.md"}
    assert all("chunk_index" in doc.metadata for doc in patched_pipeline.documents)


def test_run_respects_batch_size(
    settings: Settings, patched_pipeline: FakeStore
) -> None:
    pipeline.run(settings)

    assert all(len(batch) <= settings.batch_size for batch in patched_pipeline.batches)


def test_run_without_documents_writes_nothing(
    settings: Settings, patched_pipeline: FakeStore, tmp_path: Path
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = pipeline.run(settings.with_overrides(documents_dir=empty_dir))

    assert result.documents == 0
    assert result.chunks == 0
    assert patched_pipeline.batches == []


def test_run_propagates_missing_directory(settings: Settings) -> None:
    with pytest.raises(FileNotFoundError):
        pipeline.run(settings.with_overrides(documents_dir=Path("/does/not/exist")))


def test_cli_applies_overrides(
    monkeypatch: pytest.MonkeyPatch, documents_dir: Path
) -> None:
    captured: list[Settings] = []
    monkeypatch.setattr(
        cli,
        "run",
        lambda s: captured.append(s)
        or pipeline.IndexingResult(1, 2, s.collection_name),
    )

    exit_code = cli.main(
        [
            "--documents-dir",
            str(documents_dir),
            "--collection",
            "cli_collection",
            "--chroma-host",
            "cli-host",
            "--chroma-port",
            "9000",
        ]
    )

    assert exit_code == 0
    assert captured[0].documents_dir == documents_dir
    assert captured[0].collection_name == "cli_collection"
    assert captured[0].chroma_host == "cli-host"
    assert captured[0].chroma_port == 9000


def test_cli_reads_log_level_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Settings] = []
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setattr(
        cli,
        "run",
        lambda s: captured.append(s)
        or pipeline.IndexingResult(1, 2, s.collection_name),
    )

    exit_code = cli.main([])

    assert exit_code == 0
    assert captured[0].log_level == "DEBUG"


def test_cli_log_level_argument_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Settings] = []
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setattr(
        cli,
        "run",
        lambda s: captured.append(s)
        or pipeline.IndexingResult(1, 2, s.collection_name),
    )

    exit_code = cli.main(["--log-level", "ERROR"])

    assert exit_code == 0
    assert captured[0].log_level == "ERROR"


def test_cli_returns_error_code_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: Settings) -> None:
        raise RuntimeError("Bedrock unreachable")

    monkeypatch.setattr(cli, "run", fail)

    assert cli.main([]) == 1
