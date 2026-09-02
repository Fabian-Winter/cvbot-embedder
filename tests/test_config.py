"""Tests for the configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from cvbot_embedder.config import (
    DEFAULT_CHROMA_PORT,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DOCUMENTS_DIR,
    DEFAULT_EMBEDDING_MODEL_ID,
    DEFAULT_LOG_LEVEL,
    Settings,
)


def test_from_env_uses_defaults_when_unset() -> None:
    settings = Settings.from_env(env={})

    assert settings.documents_dir == DEFAULT_DOCUMENTS_DIR
    assert settings.documents_dir.exists()
    assert settings.chroma_port == DEFAULT_CHROMA_PORT
    assert settings.collection_name == DEFAULT_COLLECTION_NAME
    assert settings.embedding_model_id == DEFAULT_EMBEDDING_MODEL_ID
    assert settings.chroma_auth_token is None
    assert settings.chroma_ssl is False
    assert settings.log_level == DEFAULT_LOG_LEVEL


def test_from_env_reads_all_values() -> None:
    settings = Settings.from_env(
        env={
            "DOCUMENTS_DIR": "/data/docs",
            "CHROMA_HOST": "chroma.internal",
            "CHROMA_PORT": "8443",
            "CHROMA_SSL": "true",
            "CHROMA_AUTH_TOKEN": "secret",
            "CHROMA_COLLECTION": "jobs",
            "AWS_REGION": "eu-west-1",
            "EMBEDDING_MODEL_ID": "amazon.titan-embed-text-v1",
            "MAX_CHUNK_TOKENS": "256",
            "TOKEN_CHUNK_OVERLAP": "16",
            "BATCH_SIZE": "10",
            "LOG_LEVEL": "debug",
        }
    )

    assert settings.documents_dir == Path("/data/docs")
    assert settings.chroma_host == "chroma.internal"
    assert settings.chroma_port == 8443
    assert settings.chroma_ssl is True
    assert settings.chroma_auth_token == "secret"
    assert settings.collection_name == "jobs"
    assert settings.aws_region == "eu-west-1"
    assert settings.max_chunk_tokens == 256
    assert settings.token_chunk_overlap == 16
    assert settings.batch_size == 10
    assert settings.log_level == "DEBUG"


def test_from_env_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHROMA_HOST", "from-process-env")

    assert Settings.from_env().chroma_host == "from-process-env"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("TRUE", True), ("yes", True), ("on", True), ("0", False),
     ("false", False), ("", False)],
)
def test_bool_parsing(raw: str, expected: bool) -> None:
    assert Settings.from_env(env={"CHROMA_SSL": raw}).chroma_ssl is expected


def test_empty_auth_token_becomes_none() -> None:
    assert Settings.from_env(env={"CHROMA_AUTH_TOKEN": ""}).chroma_auth_token is None


def test_non_numeric_port_raises() -> None:
    with pytest.raises(ValueError, match="CHROMA_PORT"):
        Settings.from_env(env={"CHROMA_PORT": "eight"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"chroma_host": ""},
        {"collection_name": ""},
        {"chroma_port": 0},
        {"chroma_port": 70000},
        {"max_chunk_tokens": 0},
        {"max_chunk_tokens": 10, "token_chunk_overlap": 10},
        {"token_chunk_overlap": -1},
        {"batch_size": 0},
        {"log_level": "TRACE"},
    ],
)
def test_invalid_values_raise(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Settings(**overrides)  # type: ignore[arg-type]


def test_with_overrides_ignores_none() -> None:
    settings = Settings(chroma_host="original", collection_name="original")

    updated = settings.with_overrides(chroma_host=None, collection_name="new")

    assert updated.chroma_host == "original"
    assert updated.collection_name == "new"
    assert settings.collection_name == "original"
