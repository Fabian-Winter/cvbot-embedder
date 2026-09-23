"""Schema-on-write parsing of the ``> key: value`` blocks in the Markdown files.

Each section may carry its metadata in blockquote lines directly below its
heading, one field per line. The block is consumed into structured chunk
metadata so that cvbot-retriever can filter on it, and a compact rendering is
written back into the chunk text so the values stay semantically searchable.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime

from cvbot_core.metadata import (
    MAX_SCHEMA_FIELDS,
    MAX_VALUES_PER_FIELD,
    PERIOD_END_KEY,
    PERIOD_START_KEY,
    RESERVED_METADATA_KEYS,
    normalize_key,
    parse_period_year,
    period_end_year,
    split_values,
)
from langchain_core.documents import Document

LOGGER = logging.getLogger(__name__)

METADATA_LINE = re.compile(r"^>\s*(?P<key>[^:]+?)\s*:\s*(?P<value>.*?)\s*$")

YEARS_KEY = "years"

# Guards against a typo turning one section into a hundred filterable years.
MAX_DERIVED_YEARS = 60


def split_metadata_block(text: str) -> tuple[dict[str, str], str]:
    """Separates the leading metadata block from the section body.

    Only the leading run of blockquote lines is consumed; a blockquote further
    down stays part of the body. Malformed lines are skipped instead of failing
    the ingestion, so a formatting slip never costs a whole section.

    Args:
        text: The section text, headings already stripped by the splitter.

    Returns:
        The parsed fields keyed by their normalized name, and the remaining
        body text.
    """
    lines = text.splitlines()
    metadata: dict[str, str] = {}
    index = 0
    consumed = 0

    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if not stripped.startswith(">"):
            break

        index += 1
        consumed = index
        match = METADATA_LINE.match(stripped)
        if match is None:
            LOGGER.debug("ignoring malformed metadata line: %r", stripped)
            continue

        key = normalize_key(match.group("key"))
        value = match.group("value").strip()
        if not key or not value:
            LOGGER.debug("ignoring metadata line without key or value: %r", stripped)
            continue
        if key in metadata:
            LOGGER.warning("duplicate metadata key %r, keeping the last value", key)
        metadata[key] = value

    body = "\n".join(lines[consumed:]).strip()
    _warn_about_trailing_metadata(body)
    return metadata, body


def render_metadata_line(metadata: Mapping[str, str]) -> str:
    """Renders the metadata back into one line of embeddable text.

    Without this the values would only live in the Chroma payload and a
    question like "Woran hat er 2013 gearbeitet?" could no longer match
    semantically.

    Args:
        metadata: The parsed fields of the chunk.

    Returns:
        The rendered line, or an empty string if there is nothing to render.
    """
    fields = sorted(
        (key, value)
        for key, value in metadata.items()
        if key not in RESERVED_METADATA_KEYS and isinstance(value, str) and value
    )
    return " | ".join(f"{key}: {value}" for key, value in fields)


def derive_year_values(metadata: Mapping[str, str]) -> str | None:
    """Expands a ``from``/``to`` period into the list of covered years.

    Filtering only compares values, so a period is unusable as a filter while a
    year list matches a question about a single year directly.

    Args:
        metadata: The parsed fields of the chunk.

    Returns:
        The comma-separated years, or ``None`` if nothing can be derived or a
        ``years`` field is already maintained by hand.
    """
    if metadata.get(YEARS_KEY):
        return None

    start = parse_period_year(metadata.get(PERIOD_START_KEY))
    if start is None:
        LOGGER.debug("no usable %r field, skipping year derivation", PERIOD_START_KEY)
        return None

    end = _resolve_period_end(metadata, start)
    if end < start:
        LOGGER.warning("period ends before it starts (%d..%d), swapping", start, end)
        start, end = end, start
    if end - start + 1 > MAX_DERIVED_YEARS:
        LOGGER.warning(
            "period %d..%d exceeds %d years, capping it",
            start,
            end,
            MAX_DERIVED_YEARS,
        )
        start = end - MAX_DERIVED_YEARS + 1

    return ", ".join(str(year) for year in range(start, end + 1))


def collect_schema(chunks: list[Document]) -> dict[str, list[str]]:
    """Collects the metadata schema actually present in the indexed chunks.

    The schema is published on the collection and injected into the
    condensation prompt, so it is capped on both axes.

    Args:
        chunks: The chunks about to be indexed.

    Returns:
        Field name mapped onto its observed values.
    """
    observed: dict[str, list[str]] = {}
    for chunk in chunks:
        for key, value in chunk.metadata.items():
            if key in RESERVED_METADATA_KEYS or not isinstance(value, str):
                continue
            field = normalize_key(key)
            if not field:
                continue
            values = observed.setdefault(field, [])
            for candidate in split_values(value):
                if candidate not in values:
                    values.append(candidate)

    return _apply_caps(observed)


def _apply_caps(observed: dict[str, list[str]]) -> dict[str, list[str]]:
    """Trims an observed schema to the published size limits.

    Fields with few distinct values are kept first, since a high-cardinality
    field is a poor filter anyway.

    Args:
        observed: The raw field-to-values mapping.

    Returns:
        The capped schema.
    """
    ranked = sorted(observed.items(), key=lambda item: (len(item[1]), item[0]))
    if len(ranked) > MAX_SCHEMA_FIELDS:
        LOGGER.warning(
            "observed %d metadata field(s), publishing only %d",
            len(ranked),
            MAX_SCHEMA_FIELDS,
        )
        ranked = ranked[:MAX_SCHEMA_FIELDS]

    schema: dict[str, list[str]] = {}
    for field, values in ranked:
        if len(values) > MAX_VALUES_PER_FIELD:
            LOGGER.warning(
                "field %r has %d value(s), publishing only %d",
                field,
                len(values),
                MAX_VALUES_PER_FIELD,
            )
        schema[field] = sorted(values)[:MAX_VALUES_PER_FIELD]
    return schema


def _resolve_period_end(metadata: Mapping[str, str], start: int) -> int:
    """Determines the last year of a period.

    Delegates to the shared rule in cvbot_core so that the published year
    lists and the query-time recency ranking of cvbot-retriever always agree
    on when a period ends.

    Args:
        metadata: The parsed fields of the chunk.
        start: The already parsed first year, used as the conservative fallback.

    Returns:
        The last year of the period.
    """
    end = period_end_year(metadata, _current_year())
    if end is None:
        # Unreachable while a start year exists; kept as the conservative
        # fallback the single-year rule always had.
        return start
    return end


def _current_year() -> int:
    """Returns the current year; separated out so tests can pin it."""
    return datetime.now(UTC).year


def _warn_about_trailing_metadata(body: str) -> None:
    """Warns about metadata lines that are not part of the leading block.

    The most likely cause is a block under an ``####`` heading, which is not a
    split boundary and therefore cannot carry its own metadata.

    Args:
        body: The section body after the leading block was removed.
    """
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith(">") and METADATA_LINE.match(stripped):
            LOGGER.warning(
                "metadata line outside the leading block is ignored: %r", stripped
            )
            return
