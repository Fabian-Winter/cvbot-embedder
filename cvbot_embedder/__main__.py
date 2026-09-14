"""Command line entry point: ``python -m cvbot_embedder``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cvbot_core.logging_config import configure_logging

from .config import Settings
from .pipeline import run

LOGGER = logging.getLogger("cvbot_embedder")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line arguments.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv``.

    Returns:
        The parsed arguments. Options that are not given are ``None`` and leave
        the value from the environment untouched.
    """
    parser = argparse.ArgumentParser(
        prog="cvbot_embedder",
        description=(
            "Reads local text documents and indexes them as embeddings in a "
            "ChromaDB. The collection is recreated on every run."
        ),
    )
    parser.add_argument(
        "--documents-dir", type=Path, help="directory holding .txt/.md files"
    )
    parser.add_argument(
        "--collection", dest="collection_name", help="target collection name"
    )
    parser.add_argument("--chroma-host", help="hostname of the ChromaDB")
    parser.add_argument("--chroma-port", type=int, help="port of the ChromaDB")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="verbosity of the log output (default: LOG_LEVEL or INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Starts the ingestion.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv``.

    Returns:
        ``0`` on success, ``1`` on failure.
    """
    args = _parse_args(argv)

    try:
        settings = Settings.from_env().with_overrides(
            documents_dir=args.documents_dir,
            collection_name=args.collection_name,
            chroma_host=args.chroma_host,
            chroma_port=args.chroma_port,
            log_level=args.log_level,
        )
        configure_logging(settings.log_level)
        result = run(settings)
    except Exception:
        LOGGER.exception("ingestion failed")
        return 1

    LOGGER.info(
        "done: %d document(s) -> %d chunk(s) in collection %r",
        result.documents,
        result.chunks,
        result.collection,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
