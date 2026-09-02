"""Pipeline configuration read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

DEFAULT_DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
DEFAULT_COLLECTION_NAME = "cvbot_documents"
DEFAULT_CHROMA_HOST = "localhost"
DEFAULT_CHROMA_PORT = 8000
DEFAULT_AWS_REGION = "eu-central-1"
DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_MAX_CHUNK_TOKENS = 512
DEFAULT_TOKEN_CHUNK_OVERLAP = 50
DEFAULT_BATCH_SIZE = 50
DEFAULT_LOG_LEVEL = "INFO"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})


@dataclass(frozen=True)
class Settings:
    """Runtime configuration of the ingestion pipeline.

    Attributes:
        documents_dir: Local directory the documents are read from.
        chroma_host: Hostname of the ChromaDB container (AWS Fargate).
        chroma_port: Port of the ChromaDB container.
        chroma_ssl: True if the connection uses HTTPS.
        chroma_auth_token: Optional bearer token for ChromaDB.
        collection_name: Name of the collection that is recreated.
        aws_region: AWS region of the Bedrock client.
        embedding_model_id: Bedrock model ID used for the embeddings.
        max_chunk_tokens: Maximum number of tokens per chunk.
        token_chunk_overlap: Overlap used when splitting oversized chunks.
        batch_size: Number of chunks per write to ChromaDB.
        log_level: Verbosity of the log output.
    """

    documents_dir: Path = DEFAULT_DOCUMENTS_DIR
    chroma_host: str = DEFAULT_CHROMA_HOST
    chroma_port: int = DEFAULT_CHROMA_PORT
    chroma_ssl: bool = False
    chroma_auth_token: str | None = None
    collection_name: str = DEFAULT_COLLECTION_NAME
    aws_region: str = DEFAULT_AWS_REGION
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS
    token_chunk_overlap: int = DEFAULT_TOKEN_CHUNK_OVERLAP
    batch_size: int = DEFAULT_BATCH_SIZE
    log_level: str = DEFAULT_LOG_LEVEL

    def __post_init__(self) -> None:
        """Validates the configuration.

        Raises:
            ValueError: If a value is outside the accepted range.
        """
        if not self.chroma_host:
            raise ValueError("chroma_host must not be empty")
        if not self.collection_name:
            raise ValueError("collection_name must not be empty")
        if not 1 <= self.chroma_port <= 65535:
            raise ValueError(f"chroma_port outside 1-65535: {self.chroma_port}")
        if self.max_chunk_tokens < 1:
            raise ValueError(
                f"max_chunk_tokens must be positive: {self.max_chunk_tokens}"
            )
        if not 0 <= self.token_chunk_overlap < self.max_chunk_tokens:
            raise ValueError(
                "token_chunk_overlap must be smaller than max_chunk_tokens: "
                f"{self.token_chunk_overlap} >= {self.max_chunk_tokens}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive: {self.batch_size}")
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ValueError(f"unknown log_level: {self.log_level}")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        """Builds the configuration from environment variables.

        Variables that are not set fall back to the module defaults.

        Args:
            env: Optional mapping used instead of ``os.environ`` (for tests).

        Returns:
            The validated configuration.

        Raises:
            ValueError: If a variable cannot be parsed or is outside the
                accepted range.
        """
        source = os.environ if env is None else env
        return cls(
            documents_dir=Path(
                source.get("DOCUMENTS_DIR", DEFAULT_DOCUMENTS_DIR)
            ),
            chroma_host=source.get("CHROMA_HOST", DEFAULT_CHROMA_HOST),
            chroma_port=_int(source, "CHROMA_PORT", DEFAULT_CHROMA_PORT),
            chroma_ssl=_bool(source, "CHROMA_SSL", False),
            chroma_auth_token=source.get("CHROMA_AUTH_TOKEN") or None,
            collection_name=source.get(
                "CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME
            ),
            aws_region=source.get("AWS_REGION", DEFAULT_AWS_REGION),
            embedding_model_id=source.get(
                "EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID
            ),
            max_chunk_tokens=_int(
                source, "MAX_CHUNK_TOKENS", DEFAULT_MAX_CHUNK_TOKENS
            ),
            token_chunk_overlap=_int(
                source, "TOKEN_CHUNK_OVERLAP", DEFAULT_TOKEN_CHUNK_OVERLAP
            ),
            batch_size=_int(source, "BATCH_SIZE", DEFAULT_BATCH_SIZE),
            log_level=source.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        )

    def with_overrides(self, **overrides: Any) -> "Settings":
        """Returns a copy with the given fields replaced.

        ``None`` values are ignored so that unset CLI arguments do not override
        the configuration coming from the environment.

        Args:
            **overrides: Field names and their new values.

        Returns:
            A new, validated ``Settings`` instance.
        """
        effective = {
            key: value for key, value in overrides.items() if value is not None
        }
        return replace(self, **effective)


def _int(env: dict[str, str] | Any, key: str, default: int) -> int:
    """Reads an integer from the environment.

    Args:
        env: Mapping of variable names to values.
        key: Name of the variable.
        default: Value used if the variable is not set.

    Returns:
        The parsed value or ``default``.

    Raises:
        ValueError: If the value is not an integer.
    """
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} is not an integer: {raw!r}") from exc


def _float(env: dict[str, str] | Any, key: str, default: float) -> float:
    """Reads a floating point number from the environment.

    Args:
        env: Mapping of variable names to values.
        key: Name of the variable.
        default: Value used if the variable is not set.

    Returns:
        The parsed value or ``default``.

    Raises:
        ValueError: If the value is not a number.
    """
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} is not a number: {raw!r}") from exc


def _bool(env: dict[str, str] | Any, key: str, default: bool) -> bool:
    """Reads a boolean flag from the environment.

    Args:
        env: Mapping of variable names to values.
        key: Name of the variable.
        default: Value used if the variable is not set.

    Returns:
        ``True`` for "1", "true", "yes" or "on" (case-insensitive), otherwise
        ``False``.
    """
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in _TRUE_VALUES
