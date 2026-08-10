from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from apps.cli.historical_news_common import parse_range_datetime
from src.historical_news.application.use_cases import allowed_content, effective_storage_policy
from src.historical_news.domain.enums import ContentStoragePolicy, HistoricalNewsSourceKind
from src.historical_news.domain.time import parse_publication_timestamp
from src.historical_news.infrastructure.issuer_feed import (
    IssuerFeedNewsSource,
    feed_retry_delay,
)
from src.historical_news.infrastructure.local_archive import (
    HistoricalSourceContractError,
    LocalArchiveNewsSource,
)
from src.historical_news.infrastructure.schemas import HistoricalNewsSourceItemV1
from src.news.domain.enums import PublicationTimestampQuality

FROM = datetime(2026, 1, 1, tzinfo=UTC)
TO = datetime(2026, 12, 31, 23, 59, tzinfo=UTC)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "historical-news-source-v1",
        "source_item_id": "item-1",
        "source_url": "https://example.invalid/item-1",
        "title": "SBER test",
        "published_at": "2026-07-01T10:00:00+03:00",
        "source_timezone": "Europe/Moscow",
        "content": "SBER test content",
        "content_storage_policy": "FULL_TEXT_ALLOWED",
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_local_jsonl_schema_validation() -> None:
    item = HistoricalNewsSourceItemV1.model_validate(_row())
    assert item.schema_version == "historical-news-source-v1"
    assert item.to_source_item().source_item_id == "item-1"


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", "v2"), ("source_item_id", " "), ("source_url", "not-a-url")],
)
def test_local_jsonl_schema_rejects_invalid_identity(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        HistoricalNewsSourceItemV1.model_validate(_row(**{field: value}))


def test_exact_aware_timestamp_converts_to_utc() -> None:
    parsed = parse_publication_timestamp(
        "2026-07-01T10:00:00+03:00", source_timezone="Europe/Moscow"
    )
    assert parsed.quality == PublicationTimestampQuality.EXACT
    assert parsed.published_at == datetime(2026, 7, 1, 7, 0, tzinfo=UTC)


def test_exact_naive_timestamp_requires_configured_timezone() -> None:
    parsed = parse_publication_timestamp("2026-07-01T10:00:00", source_timezone="Europe/Moscow")
    assert parsed.quality == PublicationTimestampQuality.EXACT
    assert parsed.published_at == datetime(2026, 7, 1, 7, 0, tzinfo=UTC)


def test_date_only_timestamp_is_not_upgraded_to_exact() -> None:
    parsed = parse_publication_timestamp("2026-07-01", source_timezone="Europe/Moscow")
    assert parsed.quality == PublicationTimestampQuality.DATE_ONLY
    assert parsed.published_at == datetime(2026, 7, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("value", "timezone", "error"),
    [
        ("2026-07-01T10:00:00", None, "timezone_unknown"),
        ("2026-07-01T10:00:00", "Mars/Olympus", "timezone_invalid"),
        ("not-a-time", "Europe/Moscow", "timestamp_invalid"),
        ("", None, "timestamp_empty"),
    ],
)
def test_unknown_timestamp_is_never_guessed(value: str, timezone: str | None, error: str) -> None:
    parsed = parse_publication_timestamp(value, source_timezone=timezone)
    assert parsed.quality == PublicationTimestampQuality.UNKNOWN
    assert parsed.published_at is None
    assert parsed.error == error


def test_range_date_is_utc_and_end_is_inclusive() -> None:
    assert parse_range_datetime("2026-07-01", end_of_day=False).hour == 0
    assert parse_range_datetime("2026-07-01", end_of_day=True).date().isoformat() == "2026-07-01"


@pytest.mark.parametrize(
    ("source", "item", "expected"),
    [
        ("FULL_TEXT_ALLOWED", "FULL_TEXT_ALLOWED", "FULL_TEXT_ALLOWED"),
        ("FULL_TEXT_ALLOWED", "EXCERPT_ALLOWED", "EXCERPT_ALLOWED"),
        ("FULL_TEXT_ALLOWED", "METADATA_ONLY", "METADATA_ONLY"),
        ("FULL_TEXT_ALLOWED", "UNKNOWN", "UNKNOWN"),
    ],
)
def test_most_restrictive_storage_policy_wins(source: str, item: str, expected: str) -> None:
    assert (
        effective_storage_policy(ContentStoragePolicy(source), ContentStoragePolicy(item))
        == expected
    )


@pytest.mark.parametrize(
    ("policy", "is_excerpt", "expected"),
    [
        ("FULL_TEXT_ALLOWED", False, "text"),
        ("EXCERPT_ALLOWED", True, "text"),
        ("EXCERPT_ALLOWED", False, None),
        ("METADATA_ONLY", True, None),
        ("UNKNOWN", False, None),
    ],
)
def test_content_storage_enforcement(policy: str, is_excerpt: bool, expected: str | None) -> None:
    assert (
        allowed_content("text", policy=ContentStoragePolicy(policy), content_is_excerpt=is_excerpt)
        == expected
    )


async def test_local_archive_paginates_with_hard_bound(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    _write_jsonl(path, [_row(source_item_id=f"item-{index}") for index in range(3)])
    source = LocalArchiveNewsSource(path, max_items=3)
    first = await source.fetch_items(from_datetime=FROM, to_datetime=TO, cursor=None, limit=2)
    second = await source.fetch_items(
        from_datetime=FROM, to_datetime=TO, cursor=first.next_cursor, limit=2
    )
    assert [item.source_item_id for item in first.items] == ["item-0", "item-1"]
    assert [item.source_item_id for item in second.items] == ["item-2"]
    assert second.next_cursor is None


async def test_local_archive_filters_timestamp_range(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    _write_jsonl(
        path,
        [
            _row(source_item_id="inside"),
            _row(source_item_id="outside", published_at="2025-07-01T10:00:00Z"),
        ],
    )
    page = await LocalArchiveNewsSource(path).fetch_items(
        from_datetime=FROM, to_datetime=TO, cursor=None, limit=10
    )
    assert [item.source_item_id for item in page.items] == ["inside"]


@pytest.mark.parametrize("cursor", ["bad", "-1"])
async def test_local_archive_rejects_invalid_cursor(tmp_path: Path, cursor: str) -> None:
    path = tmp_path / "source.jsonl"
    _write_jsonl(path, [_row()])
    with pytest.raises(HistoricalSourceContractError):
        await LocalArchiveNewsSource(path).fetch_items(
            from_datetime=FROM, to_datetime=TO, cursor=cursor, limit=1
        )


async def test_local_archive_rejects_non_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(HistoricalSourceContractError, match="JSONL"):
        await LocalArchiveNewsSource(path).fetch_items(
            from_datetime=FROM, to_datetime=TO, cursor=None, limit=1
        )


async def test_local_archive_reports_invalid_line(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(HistoricalSourceContractError, match="line 1"):
        await LocalArchiveNewsSource(path).fetch_items(
            from_datetime=FROM, to_datetime=TO, cursor=None, limit=1
        )


async def test_local_archive_rejects_oversized_input(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    _write_jsonl(path, [_row(source_item_id="a"), _row(source_item_id="b")])
    with pytest.raises(HistoricalSourceContractError, match="max_items"):
        await LocalArchiveNewsSource(path, max_items=1).fetch_items(
            from_datetime=FROM, to_datetime=TO, cursor=None, limit=1
        )


async def test_local_archive_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    _write_jsonl(path, [_row()])
    with pytest.raises(HistoricalSourceContractError, match="max_file_bytes"):
        await LocalArchiveNewsSource(path, max_file_bytes=1).fetch_items(
            from_datetime=FROM, to_datetime=TO, cursor=None, limit=1
        )


async def test_rss_parsing_uses_no_live_http() -> None:
    rss = (
        b"<rss><channel><item><guid>rss-1</guid>"
        b"<link>https://issuer.invalid/1</link><title>SBER RSS</title>"
        b"<pubDate>Wed, 01 Jul 2026 10:00:00 +0300</pubDate>"
        b"<description>SBER body</description></item></channel></rss>"
    )
    source, client = _feed(httpx.Response(200, content=rss), HistoricalNewsSourceKind.ISSUER_RSS)
    try:
        page = await source.fetch_items(from_datetime=FROM, to_datetime=TO, cursor=None, limit=10)
    finally:
        await client.aclose()
    assert page.items[0].source_item_id == "rss-1"
    assert page.items[0].published_at_text.endswith("+03:00")


async def test_atom_parsing_uses_no_live_http() -> None:
    atom = (
        b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>atom-1</id>'
        b'<link href="https://issuer.invalid/1"/><title>GAZP Atom</title>'
        b"<published>2026-07-01T10:00:00+03:00</published>"
        b"<content>GAZP body</content></entry></feed>"
    )
    source, client = _feed(httpx.Response(200, content=atom), HistoricalNewsSourceKind.ISSUER_ATOM)
    try:
        page = await source.fetch_items(from_datetime=FROM, to_datetime=TO, cursor=None, limit=10)
    finally:
        await client.aclose()
    assert page.items[0].source_item_id == "atom-1"


async def test_conditional_request_handles_not_modified() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(304, headers={"ETag": "v2"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = _feed_source(
        client, HistoricalNewsSourceKind.ISSUER_RSS, etag="v1", last_modified="yesterday"
    )
    try:
        page = await source.fetch_items(from_datetime=FROM, to_datetime=TO, cursor=None, limit=10)
    finally:
        await client.aclose()
    assert page.not_modified is True
    assert captured[0].headers["If-None-Match"] == "v1"
    assert captured[0].headers["If-Modified-Since"] == "yesterday"


async def test_response_cache_validators_are_reused() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if len(captured) == 1:
            return httpx.Response(
                200,
                content=b"<rss><channel/></rss>",
                headers={"ETag": "fresh", "Last-Modified": "today"},
                request=request,
            )
        return httpx.Response(304, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = _feed_source(client, HistoricalNewsSourceKind.ISSUER_RSS)
    try:
        await source.fetch_items(from_datetime=FROM, to_datetime=TO, cursor=None, limit=10)
        await source.fetch_items(from_datetime=FROM, to_datetime=TO, cursor=None, limit=10)
    finally:
        await client.aclose()
    assert captured[1].headers["If-None-Match"] == "fresh"
    assert captured[1].headers["If-Modified-Since"] == "today"


async def test_retry_is_bounded_and_recovers() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, request=request)
        return httpx.Response(200, content=b"<rss><channel/></rss>", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = _feed_source(client, HistoricalNewsSourceKind.ISSUER_RSS, max_retries=2)
    try:
        await source.fetch_items(from_datetime=FROM, to_datetime=TO, cursor=None, limit=10)
    finally:
        await client.aclose()
    assert attempts == 3


async def test_retry_exhaustion_is_reported() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    source = _feed_source(client, HistoricalNewsSourceKind.ISSUER_RSS, max_retries=1)
    try:
        with pytest.raises(HistoricalSourceContractError, match="retries exhausted"):
            await source.fetch_items(from_datetime=FROM, to_datetime=TO, cursor=None, limit=10)
    finally:
        await client.aclose()
    assert attempts == 2


async def test_malformed_feed_is_rejected() -> None:
    source, client = _feed(
        httpx.Response(200, content=b"<rss>"), HistoricalNewsSourceKind.ISSUER_RSS
    )
    try:
        with pytest.raises(HistoricalSourceContractError, match="malformed XML"):
            await source.fetch_items(from_datetime=FROM, to_datetime=TO, cursor=None, limit=10)
    finally:
        await client.aclose()


async def test_oversized_feed_is_rejected() -> None:
    source, client = _feed(
        httpx.Response(200, content=b"<rss><channel/></rss>"),
        HistoricalNewsSourceKind.ISSUER_RSS,
        max_response_bytes=1,
    )
    try:
        with pytest.raises(HistoricalSourceContractError, match="max_response_bytes"):
            await source.fetch_items(from_datetime=FROM, to_datetime=TO, cursor=None, limit=10)
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("http://issuer.invalid/feed", HistoricalNewsSourceKind.ISSUER_RSS),
        ("https://user:secret@issuer.invalid/feed", HistoricalNewsSourceKind.ISSUER_RSS),
        ("https://issuer.invalid/feed", HistoricalNewsSourceKind.LOCAL_ARCHIVE),
    ],
)
def test_feed_configuration_is_restricted(url: str, kind: HistoricalNewsSourceKind) -> None:
    with pytest.raises(ValueError):
        IssuerFeedNewsSource(
            feed_url=url,
            source_kind=kind,
            content_storage_policy=ContentStoragePolicy.EXCERPT_ALLOWED,
            source_timezone="Europe/Moscow",
            timeout_seconds=1,
            max_retries=0,
            max_items=10,
            user_agent="test",
        )


def test_feed_retries_cannot_exceed_hard_bound() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        IssuerFeedNewsSource(
            feed_url="https://issuer.invalid/feed",
            source_kind=HistoricalNewsSourceKind.ISSUER_RSS,
            content_storage_policy=ContentStoragePolicy.EXCERPT_ALLOWED,
            source_timezone=None,
            timeout_seconds=1,
            max_retries=6,
            max_items=10,
            user_agent="test",
        )


@pytest.mark.parametrize(
    ("attempt", "retry_after", "expected"),
    [(0, None, 0.25), (2, None, 1.0), (0, "99", 30.0), (0, "bad", 1.0)],
)
def test_retry_delay_is_capped(attempt: int, retry_after: str | None, expected: float) -> None:
    assert feed_retry_delay(attempt, retry_after) == expected


def _feed(
    response: httpx.Response,
    kind: HistoricalNewsSourceKind,
    *,
    max_response_bytes: int = 10_000_000,
) -> tuple[IssuerFeedNewsSource, httpx.AsyncClient]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            response.status_code,
            content=response.content,
            headers=response.headers,
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return _feed_source(client, kind, max_response_bytes=max_response_bytes), client


def _feed_source(
    client: httpx.AsyncClient,
    kind: HistoricalNewsSourceKind,
    *,
    max_retries: int = 0,
    etag: str | None = None,
    last_modified: str | None = None,
    max_response_bytes: int = 10_000_000,
) -> IssuerFeedNewsSource:
    return IssuerFeedNewsSource(
        feed_url="https://issuer.invalid/feed",
        source_kind=kind,
        content_storage_policy=ContentStoragePolicy.EXCERPT_ALLOWED,
        source_timezone="Europe/Moscow",
        timeout_seconds=1,
        max_retries=max_retries,
        max_items=10,
        user_agent="historical-news-test",
        etag=etag,
        last_modified=last_modified,
        client=client,
        sleep=False,
        max_response_bytes=max_response_bytes,
    )
