"""Pipeline configuration read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cvbot_core.logging_config import VALID_LOG_LEVELS
from cvbot_core.env import read_int, read_path, read_str
from cvbot_core.overrides import apply_overrides
from cvbot_core.validation import (
    require_choice,
    require_in_range,
    require_non_empty,
    require_port,
    require_positive,
)
from cvbot_core.vector_store import DEFAULT_COLLECTION_NAME

DEFAULT_DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
DEFAULT_CHROMA_HOST = "localhost"
DEFAULT_CHROMA_PORT = 8000
DEFAULT_AWS_REGION = "eu-central-1"
DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_MAX_CHUNK_TOKENS = 512
DEFAULT_TOKEN_CHUNK_OVERLAP = 50
DEFAULT_BATCH_SIZE = 50
DEFAULT_LOG_LEVEL = "INFO"


@dataclass(frozen=True)
class Settings:
    """Runtime configuration of the ingestion pipeline.

    Attributes:
        documents_dir: Local directory the documents are read from.
        chroma_host: Hostname of the ChromaDB container (AWS Fargate).
        chroma_port: Port of the ChromaDB container.
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
        require_non_empty(self.chroma_host, "chroma_host")
        require_non_empty(self.collection_name, "collection_name")
        require_port(self.chroma_port, "chroma_port")
        require_positive(self.max_chunk_tokens, "max_chunk_tokens")
        require_in_range(
            self.token_chunk_overlap,
            self.max_chunk_tokens,
            "token_chunk_overlap",
            "max_chunk_tokens",
        )
        require_positive(self.batch_size, "batch_size")
        require_choice(self.log_level, VALID_LOG_LEVELS, "log_level")

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
            documents_dir=read_path(
                source, "DOCUMENTS_DIR", DEFAULT_DOCUMENTS_DIR
            ),
            chroma_host=read_str(source, "CHROMA_HOST", DEFAULT_CHROMA_HOST),
            chroma_port=read_int(source, "CHROMA_PORT", DEFAULT_CHROMA_PORT),
            collection_name=read_str(
                source, "CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME
            ),
            aws_region=read_str(source, "AWS_REGION", DEFAULT_AWS_REGION),
            embedding_model_id=read_str(
                source, "EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID
            ),
            max_chunk_tokens=read_int(
                source, "MAX_CHUNK_TOKENS", DEFAULT_MAX_CHUNK_TOKENS
            ),
            token_chunk_overlap=read_int(
                source, "TOKEN_CHUNK_OVERLAP", DEFAULT_TOKEN_CHUNK_OVERLAP
            ),
            batch_size=read_int(source, "BATCH_SIZE", DEFAULT_BATCH_SIZE),
            log_level=read_str(source, "LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
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
        return apply_overrides(self, **overrides)

