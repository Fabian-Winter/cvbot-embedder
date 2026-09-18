"""Tests for the two-stage chunking."""

from __future__ import annotations

import tiktoken

from cvbot_embedder.chunking import ENCODING_NAME, DocumentChunker
from tests.conftest import collect_metadata, make_documents

ENCODING = tiktoken.get_encoding(ENCODING_NAME)


def token_count(text: str) -> int:
    """Counts the tokens of a text the same way the chunker does."""
    return len(ENCODING.encode(text))


def build_chunker(max_tokens: int = 64) -> DocumentChunker:
    """Creates a chunker with the test configuration."""
    return DocumentChunker(
        max_tokens=max_tokens,
        token_overlap=4,
    )


def test_empty_input_returns_empty_list() -> None:
    assert build_chunker().split([]) == []


def test_short_document_stays_in_one_chunk() -> None:
    documents = make_documents("A short sentence about the topic.")

    chunks = build_chunker().split(documents)

    assert len(chunks) == 1
    assert chunks[0].page_content == "A short sentence about the topic."


def test_markdown_splits_by_headers_with_header_context() -> None:
    documents = make_documents(
        "# Handbook\n\nIntro text.\n\n## Vacation\n\nVacation policy.\n\n"
        "## Remote Work\n\nRemote policy.",
        source="handbook.md",
    )

    chunks = build_chunker().split(documents)

    assert [chunk.metadata["h1"] for chunk in chunks] == [
        "Handbook",
        "Handbook",
        "Handbook",
    ]
    assert chunks[1].metadata["h2"] == "Vacation"
    assert chunks[1].page_content.startswith("# Handbook\n## Vacation")
    assert "Remote policy" not in chunks[1].page_content


def test_plaintext_splits_by_paragraphs_with_overlap() -> None:
    documents = make_documents(
        "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        source="policy.txt",
    )

    chunks = build_chunker().split(documents)

    assert [chunk.page_content for chunk in chunks] == [
        "First paragraph.",
        "First paragraph.\n\nSecond paragraph.",
        "Second paragraph.\n\nThird paragraph.",
    ]


def test_plaintext_skips_overlap_when_it_exceeds_token_limit() -> None:
    first = " ".join(f"First{i}" for i in range(30))
    second = "Second paragraph."
    documents = make_documents(f"{first}\n\n{second}", source="policy.txt")

    chunks = build_chunker(max_tokens=16).split(documents)

    assert second in chunks[-1].page_content
    assert "First0" not in chunks[-1].page_content


def test_oversized_chunk_is_split_by_tokens() -> None:
    long_text = " ".join(f"Word{i} to make the text longer." for i in range(120))
    documents = make_documents(long_text)

    chunks = build_chunker(max_tokens=32).split(documents)

    assert len(chunks) > 1
    assert all(token_count(chunk.page_content) <= 32 for chunk in chunks)


def test_all_chunks_respect_token_limit() -> None:
    paragraphs = [
        "The vacation policy grants thirty days. " * 20,
        "Remote work is possible on three days. " * 20,
    ]
    documents = make_documents("\n\n".join(paragraphs))

    chunks = build_chunker(max_tokens=48).split(documents)

    assert all(token_count(chunk.page_content) <= 48 for chunk in chunks)


def test_metadata_is_preserved_on_split_chunks() -> None:
    long_text = " ".join(f"Sentence number {i} with content." for i in range(100))
    documents = make_documents(long_text, source="handbook.md")

    chunks = build_chunker(max_tokens=32).split(documents)

    assert all(chunk.metadata["source"] == "handbook.md" for chunk in chunks)
    assert all(chunk.metadata["filename"] == "handbook.md" for chunk in chunks)


def test_chunk_index_is_sequential_per_source() -> None:
    long_text = " ".join(f"Sentence number {i} with content." for i in range(100))
    documents = make_documents(long_text, source="a.txt")
    documents += make_documents(long_text, source="b.txt")

    chunks = build_chunker(max_tokens=32).split(documents)

    for source in ("a.txt", "b.txt"):
        indices = collect_metadata(
            [c for c in chunks if c.metadata["source"] == source], "chunk_index"
        )
        assert indices == list(range(len(indices)))


def test_content_is_not_lost() -> None:
    documents = make_documents("First sentence. Second sentence. Third sentence.")

    chunks = build_chunker().split(documents)

    joined = " ".join(chunk.page_content for chunk in chunks)
    assert "First sentence" in joined
    assert "Third sentence" in joined


def test_markdown_without_metadata_lines_is_unchanged() -> None:
    documents = make_documents(
        "# Handbook\n\nIntro text.\n\n## Vacation\n\nVacation policy.",
        source="handbook.md",
    )

    chunks = build_chunker().split(documents)

    assert chunks[1].page_content == "# Handbook\n## Vacation\n\nVacation policy."
    assert set(chunks[1].metadata) == {"source", "filename", "h1", "h2", "chunk_index"}


def test_section_metadata_becomes_chunk_metadata() -> None:
    documents = make_documents(
        "# Arbeitgeber\n\n## IAV GmbH\n> status: Historisch\n> ort: Berlin\n\nText.",
        source="cv.md",
    )

    chunks = build_chunker().split(documents)

    assert chunks[-1].metadata["status"] == "Historisch"
    assert chunks[-1].metadata["ort"] == "Berlin"


def test_metadata_lines_are_replaced_by_a_compact_line() -> None:
    documents = make_documents(
        "# Arbeitgeber\n> status: Historisch\n> ort: Berlin\n\nText.",
        source="cv.md",
    )

    chunks = build_chunker().split(documents)

    assert chunks[0].page_content == (
        "# Arbeitgeber\n\nort: Berlin | status: Historisch\n\nText."
    )


def test_section_metadata_is_inherited_by_subsections() -> None:
    documents = make_documents(
        "# Projekte\n> typ: projekt\n\n## Beruf\n> kontext: beruf\n\n"
        "### Erstes\n> rolle: Architekt\n\nText.",
        source="cv.md",
    )

    chunks = build_chunker().split(documents)

    assert chunks[-1].metadata["typ"] == "projekt"
    assert chunks[-1].metadata["kontext"] == "beruf"
    assert chunks[-1].metadata["rolle"] == "Architekt"


def test_sibling_sections_do_not_inherit_from_each_other() -> None:
    documents = make_documents(
        "# Projekte\n\n## Erstes\n> rolle: Architekt\n\nText.\n\n"
        "## Zweites\n> ort: Berlin\n\nText.",
        source="cv.md",
    )

    chunks = build_chunker().split(documents)

    assert "rolle" not in chunks[-1].metadata
    assert chunks[-1].metadata["ort"] == "Berlin"


def test_deeper_section_overrides_an_inherited_field() -> None:
    documents = make_documents(
        "# Projekte\n> status: Historisch\n\n## Aktuelles\n> status: Aktuell\n\nText.",
        source="cv.md",
    )

    chunks = build_chunker().split(documents)

    assert chunks[-1].metadata["status"] == "Aktuell"


def test_years_are_derived_from_an_inherited_period() -> None:
    documents = make_documents(
        "# Projekte\n> von: 2011-10\n> bis: 2013-05\n\n## Erstes\n\nText.",
        source="cv.md",
    )

    chunks = build_chunker().split(documents)

    assert chunks[-1].metadata["jahre"] == "2011, 2012, 2013"
    assert "jahre: 2011, 2012, 2013" in chunks[-1].page_content
