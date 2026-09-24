"""Tests for the schema-on-write metadata parsing."""

from __future__ import annotations

import pytest

from cvbot_embedder import metadata
from cvbot_embedder.metadata import (
    collect_schema,
    derive_year_values,
    render_metadata_line,
    split_metadata_block,
)
from tests.conftest import make_documents


def test_split_metadata_block_reads_the_leading_block() -> None:
    fields, body = split_metadata_block(
        "> status: Historisch\n> ort: Berlin\n\nFließtext."
    )

    assert fields == {"status": "Historisch", "ort": "Berlin"}
    assert body == "Fließtext."


def test_split_metadata_block_normalizes_keys() -> None:
    fields, _ = split_metadata_block("> Tech-Stack: Java, Maven")

    assert fields == {"tech_stack": "Java, Maven"}


def test_split_metadata_block_tolerates_blank_lines_inside_the_block() -> None:
    fields, body = split_metadata_block("\n> status: Aktuell\n\n> ort: Berlin\n\nText")

    assert fields == {"status": "Aktuell", "ort": "Berlin"}
    assert body == "Text"


def test_split_metadata_block_stops_at_the_first_body_line() -> None:
    fields, body = split_metadata_block(
        "> status: Aktuell\nFließtext.\n> ort: Berlin"
    )

    assert fields == {"status": "Aktuell"}
    assert body == "Fließtext.\n> ort: Berlin"


def test_split_metadata_block_skips_malformed_lines() -> None:
    fields, body = split_metadata_block(
        "> kaputt ohne Doppelpunkt\n> status: Aktuell\n\nText"
    )

    assert fields == {"status": "Aktuell"}
    assert body == "Text"


def test_split_metadata_block_skips_lines_without_a_value() -> None:
    fields, _ = split_metadata_block("> status:\n> ort: Berlin")

    assert fields == {"ort": "Berlin"}


def test_split_metadata_block_keeps_the_last_duplicate() -> None:
    fields, _ = split_metadata_block("> status: Aktuell\n> status: Historisch")

    assert fields == {"status": "Historisch"}


def test_split_metadata_block_returns_the_text_unchanged_without_metadata() -> None:
    fields, body = split_metadata_block("Nur Fließtext.\n\nZweiter Absatz.")

    assert fields == {}
    assert body == "Nur Fließtext.\n\nZweiter Absatz."


def test_render_metadata_line_sorts_the_fields() -> None:
    line = render_metadata_line({"status": "Aktuell", "ort": "Berlin"})

    assert line == "ort: Berlin | status: Aktuell"


def test_render_metadata_line_skips_reserved_keys() -> None:
    line = render_metadata_line({"source": "a.md", "h1": "Titel", "ort": "Berlin"})

    assert line == "ort: Berlin"


def test_render_metadata_line_is_empty_without_fields() -> None:
    assert render_metadata_line({}) == ""


def test_derive_year_values_expands_a_closed_period() -> None:
    years = derive_year_values({"startdate": "2011-10", "enddate": "2014-10"})

    assert years == "2011, 2012, 2013, 2014"


def test_derive_year_values_accepts_plain_years_and_full_dates() -> None:
    assert derive_year_values({"startdate": "2011", "enddate": "2012-03-01"}) == "2011, 2012"


def test_derive_year_values_reads_years_out_of_prose() -> None:
    assert derive_year_values({"startdate": "Oktober 2011", "enddate": "Mai 2012"}) == (
        "2011, 2012"
    )


def test_derive_year_values_runs_until_the_current_year_when_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metadata, "_current_year", lambda: 2013)

    assert derive_year_values({"startdate": "2011", "enddate": "laufend"}) == "2011, 2012, 2013"


def test_derive_year_values_treats_a_missing_end_as_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metadata, "_current_year", lambda: 2012)

    assert derive_year_values({"startdate": "2011"}) == "2011, 2012"


def test_derive_year_values_falls_back_to_a_single_year_for_an_unparsable_end() -> None:
    assert derive_year_values({"startdate": "2011", "enddate": "irgendwann"}) == "2011"


def test_derive_year_values_swaps_reversed_bounds() -> None:
    assert derive_year_values({"startdate": "2014", "enddate": "2012"}) == "2012, 2013, 2014"


def test_derive_year_values_caps_an_implausible_period() -> None:
    years = derive_year_values({"startdate": "1900", "enddate": "2020"})

    assert years is not None
    assert len(years.split(", ")) == metadata.MAX_DERIVED_YEARS
    assert years.endswith("2020")


def test_derive_year_values_keeps_a_manually_maintained_field() -> None:
    assert derive_year_values({"years": "2011", "startdate": "2011", "enddate": "2014"}) is None


def test_derive_year_values_returns_none_without_a_start() -> None:
    assert derive_year_values({"enddate": "2014"}) is None
    assert derive_year_values({"startdate": "unbekannt", "enddate": "2014"}) is None


def test_collect_schema_gathers_the_observed_values() -> None:
    chunks = make_documents("a", "b")
    chunks[0].metadata.update({"status": "Aktuell", "tech": "Java, Maven"})
    chunks[1].metadata.update({"status": "Historisch", "tech": "Java"})

    assert collect_schema(chunks) == {
        "status": ["aktuell", "historisch"],
        "tech": ["java", "maven"],
    }


def test_collect_schema_skips_reserved_keys() -> None:
    chunks = make_documents("a")
    chunks[0].metadata.update({"h1": "Titel", "chunk_index": 0, "ort": "Berlin"})

    assert collect_schema(chunks) == {"ort": ["berlin"]}


def test_collect_schema_is_empty_without_metadata() -> None:
    assert collect_schema(make_documents("a", "b")) == {}


def test_collect_schema_caps_the_value_list() -> None:
    chunks = make_documents(*[str(index) for index in range(60)])
    for index, chunk in enumerate(chunks):
        chunk.metadata["years"] = str(1900 + index)

    assert len(collect_schema(chunks)["years"]) == metadata.MAX_VALUES_PER_FIELD


def test_collect_schema_caps_the_field_count() -> None:
    chunks = make_documents("a")
    for index in range(40):
        chunks[0].metadata[f"feld_{index}"] = "wert"

    assert len(collect_schema(chunks)) == metadata.MAX_SCHEMA_FIELDS
