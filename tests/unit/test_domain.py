from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from src.news.domain.entities import NewsItem
from src.news.domain.exceptions import DomainError
from src.news.domain.hash import calculate_raw_content_hash


def test_hash_is_stable_for_same_source_and_content() -> None:
    first = calculate_raw_content_hash(" Test Source ", "Original text")
    second = calculate_raw_content_hash("test   source", "Original text")

    assert first == second


def test_hash_changes_when_raw_content_changes() -> None:
    first = calculate_raw_content_hash("source", "Original text")
    second = calculate_raw_content_hash("source", "Original text ")

    assert first != second


def test_raw_content_is_preserved_without_normalization() -> None:
    raw_content = "  Original text\nwith spacing  "
    item = NewsItem.create(
        source_id="source",
        source_name="Source",
        source_url="https://example.com/news/1",
        title="Title",
        raw_content=raw_content,
        language="en",
        published_at=datetime(2026, 8, 6, 8, tzinfo=UTC),
    )

    assert item.raw_content == raw_content


def test_empty_raw_content_is_rejected() -> None:
    with pytest.raises(DomainError, match="raw_content"):
        NewsItem.create(
            source_id="source",
            source_name="Source",
            source_url="https://example.com/news/1",
            title="Title",
            raw_content="   ",
            language="en",
            published_at=datetime(2026, 8, 6, 8, tzinfo=UTC),
        )


def test_timezone_naive_published_at_is_rejected() -> None:
    with pytest.raises(DomainError, match="published_at"):
        NewsItem.create(
            source_id="source",
            source_name="Source",
            source_url="https://example.com/news/1",
            title="Title",
            raw_content="text",
            language="en",
            published_at=datetime(2026, 8, 6, 8),
        )


def test_timestamps_are_converted_to_utc() -> None:
    plus_five = timezone(timedelta(hours=5))
    item = NewsItem.create(
        source_id="source",
        source_name="Source",
        source_url="https://example.com/news/1",
        title="Title",
        raw_content="text",
        language="en",
        published_at=datetime(2026, 8, 6, 13, tzinfo=plus_five),
        received_at=datetime(2026, 8, 6, 13, 1, tzinfo=plus_five),
    )

    assert item.published_at == datetime(2026, 8, 6, 8, tzinfo=UTC)
    assert item.received_at == datetime(2026, 8, 6, 8, 1, tzinfo=UTC)
