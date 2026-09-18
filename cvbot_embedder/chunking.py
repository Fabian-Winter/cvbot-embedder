"""Two-stage chunking: format-aware first, token-based for oversized chunks."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

from cvbot_core.tokens import ENCODING_NAME, count_tokens
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, TokenTextSplitter

from .metadata import (
    YEARS_KEY,
    derive_year_values,
    render_metadata_line,
    split_metadata_block,
)

LOGGER = logging.getLogger(__name__)

MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class DocumentChunker:
    """Splits documents into format-aware, token-bounded chunks.

    Markdown files are split by header hierarchy and text files by paragraph.
    Chunks that still exceed ``max_tokens`` are additionally split by token
    count.
    """

    def __init__(
        self,
        max_tokens: int,
        token_overlap: int,
    ) -> None:
        """Initializes both splitters.

        Args:
            max_tokens: Maximum number of tokens per chunk.
            token_overlap: Overlap used when splitting oversized chunks.
        """
        self._max_tokens = max_tokens
        self._markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=MARKDOWN_HEADERS,
        )
        self._token_splitter = TokenTextSplitter(
            encoding_name=ENCODING_NAME,
            chunk_size=max_tokens,
            chunk_overlap=token_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Splits documents into chunks.

        Args:
            documents: The documents to split.

        Returns:
            The chunks in document order. The metadata of the source document
            is preserved and extended by ``chunk_index``.
        """
        if not documents:
            return []

        chunks: list[Document] = []
        for document in documents:
            for chunk in self._split_by_format(document):
                if count_tokens(chunk.page_content) <= self._max_tokens:
                    chunks.append(chunk)
                else:
                    chunks.extend(self._token_splitter.split_documents([chunk]))

        _assign_chunk_indices(chunks)
        LOGGER.info(
            "split %d document(s) into %d chunk(s)", len(documents), len(chunks)
        )
        return chunks

    def _split_by_format(self, document: Document) -> list[Document]:
        """Splits a document according to its file format.

        Markdown files keep their header context, while text files keep the
        previous paragraph as overlap when it fits the token budget.

        Args:
            document: The document to split.

        Returns:
            The format-aware chunks of the document.
        """
        source = str(document.metadata.get("source", ""))
        suffix = Path(source).suffix.lower()
        if suffix == ".md":
            return self._split_markdown(document)
        return self._split_plaintext(document)

    def _split_markdown(self, document: Document) -> list[Document]:
        """Splits Markdown and keeps the active header path with each chunk.

        Metadata blocks are consumed per section and inherited downwards, so a
        field set once below the ``#`` heading applies to every section of the
        file unless a deeper section overrides it.

        Args:
            document: The Markdown document to split.

        Returns:
            The chunks with original, inherited and header metadata.
        """
        split_documents = self._markdown_splitter.split_text(document.page_content)
        inherited: dict[int, dict[str, str]] = {}
        chunks: list[Document] = []
        for split_document in split_documents:
            section_metadata, body = split_metadata_block(
                split_document.page_content
            )
            depth = _header_depth(split_document.metadata)
            _reset_below(inherited, depth)
            inherited[depth] = section_metadata

            metadata: dict[str, object] = {**document.metadata}
            for level in sorted(inherited):
                metadata.update(inherited[level])
            metadata.update(split_document.metadata)
            _add_derived_years(metadata)

            chunks.append(
                Document(
                    page_content=_build_page_content(metadata, body),
                    metadata=metadata,
                )
            )
        return chunks or [document]

    def _split_plaintext(self, document: Document) -> list[Document]:
        """Splits plain text by paragraphs with previous-paragraph overlap.

        Args:
            document: The text document to split.

        Returns:
            The paragraph chunks with original metadata.
        """
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", document.page_content)
            if paragraph.strip()
        ]
        if not paragraphs:
            return [document]

        chunks: list[Document] = []
        for index, paragraph in enumerate(paragraphs):
            page_content = paragraph
            if index > 0:
                candidate = f"{paragraphs[index - 1]}\n\n{paragraph}"
                if count_tokens(candidate) <= self._max_tokens:
                    page_content = candidate
            chunks.append(
                Document(
                    page_content=page_content,
                    metadata=dict(document.metadata),
                )
            )
        return chunks


def _assign_chunk_indices(chunks: list[Document]) -> None:
    """Numbers the chunks consecutively per source document, starting at zero.

    Args:
        chunks: The chunks whose metadata is extended.
    """
    counters: defaultdict[str, int] = defaultdict(int)
    for chunk in chunks:
        source = str(chunk.metadata.get("source", ""))
        chunk.metadata["chunk_index"] = counters[source]
        counters[source] += 1


def _format_header_context(metadata: dict[str, object]) -> str:
    """Formats Markdown header metadata as header lines.

    Args:
        metadata: Header metadata from ``MarkdownHeaderTextSplitter``.

    Returns:
        The header path as Markdown text.
    """
    lines: list[str] = []
    for marker, key in MARKDOWN_HEADERS:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            lines.append(f"{marker} {value}")
    return "\n".join(lines)


def _header_depth(header_metadata: dict[str, str]) -> int:
    """Determines how deep a split sits in the header hierarchy.

    Args:
        header_metadata: Header metadata from ``MarkdownHeaderTextSplitter``.

    Returns:
        The depth, where ``0`` is the document preamble before the first
        heading.
    """
    return sum(1 for _, key in MARKDOWN_HEADERS if header_metadata.get(key))


def _reset_below(inherited: dict[int, dict[str, str]], depth: int) -> None:
    """Drops the metadata of the previous branch when a sibling section starts.

    Args:
        inherited: Metadata per header depth.
        depth: Depth of the section that is about to be processed.
    """
    for level in [level for level in inherited if level >= depth]:
        del inherited[level]


def _add_derived_years(metadata: dict[str, object]) -> None:
    """Adds the ``jahre`` field derived from an inherited ``von``/``bis`` period.

    Args:
        metadata: The merged metadata of the chunk, modified in place.
    """
    period = {
        key: value for key, value in metadata.items() if isinstance(value, str)
    }
    years = derive_year_values(period)
    if years:
        metadata[YEARS_KEY] = years


def _build_page_content(metadata: dict[str, object], body: str) -> str:
    """Prepends the header path and the metadata line to the section body.

    Args:
        metadata: The merged metadata of the chunk.
        body: The section text without its metadata block.

    Returns:
        The embeddable chunk text.
    """
    fields = {
        key: value for key, value in metadata.items() if isinstance(value, str)
    }
    parts = [
        part
        for part in (_format_header_context(metadata), render_metadata_line(fields))
        if part
    ]
    parts.append(body)
    return "\n\n".join(part for part in parts if part)
