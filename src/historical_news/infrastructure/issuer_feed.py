from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from time import monotonic
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx

from src.historical_news.domain.entities import HistoricalNewsPage, HistoricalSourceItem
from src.historical_news.domain.enums import ContentStoragePolicy, HistoricalNewsSourceKind
from src.historical_news.infrastructure.local_archive import HistoricalSourceContractError


class IssuerFeedNewsSource:
    def __init__(
        self,
        *,
        feed_url: str,
        source_kind: HistoricalNewsSourceKind,
        content_storage_policy: ContentStoragePolicy,
        source_timezone: str | None,
        timeout_seconds: float,
        max_retries: int,
        max_items: int,
        user_agent: str,
        etag: str | None = None,
        last_modified: str | None = None,
        client: httpx.AsyncClient | None = None,
        sleep: bool = True,
        min_request_interval_seconds: float = 0.0,
        max_response_bytes: int = 10_000_000,
    ) -> None:
        if source_kind not in {
            HistoricalNewsSourceKind.ISSUER_RSS,
            HistoricalNewsSourceKind.ISSUER_ATOM,
        }:
            raise ValueError("issuer feed source_kind must be ISSUER_RSS or ISSUER_ATOM")
        parsed_url = urlparse(feed_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("issuer feed URL must use HTTPS")
        if parsed_url.username is not None or parsed_url.password is not None:
            raise ValueError("issuer feed URL must not contain credentials")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if not 1 <= max_items <= 100_000:
            raise ValueError("max_items must be between 1 and 100000")
        if not user_agent.strip():
            raise ValueError("user_agent must not be blank")
        if not 1 <= max_response_bytes <= 100_000_000:
            raise ValueError("max_response_bytes must be between 1 and 100000000")
        self._feed_url = feed_url
        self._source_kind = source_kind
        self._content_storage_policy = content_storage_policy
        self._source_timezone = source_timezone
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._max_items = max_items
        self._user_agent = user_agent
        self._etag = etag
        self._last_modified = last_modified
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self._owns_client = client is None
        self._sleep = sleep
        self._min_request_interval_seconds = max(0.0, min_request_interval_seconds)
        self._last_request_started: float | None = None
        self._max_response_bytes = max_response_bytes

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch_items(
        self,
        *,
        from_datetime: datetime,
        to_datetime: datetime,
        cursor: str | None,
        limit: int,
    ) -> HistoricalNewsPage:
        if cursor is not None:
            return HistoricalNewsPage(items=[], next_cursor=None)
        response = await self._request()
        if response.status_code == 304:
            return HistoricalNewsPage(
                items=[],
                next_cursor=None,
                etag=response.headers.get("ETag", self._etag),
                last_modified=response.headers.get("Last-Modified", self._last_modified),
                not_modified=True,
            )
        if len(response.content) > self._max_response_bytes:
            raise HistoricalSourceContractError("issuer feed exceeds configured max_response_bytes")
        self._etag = response.headers.get("ETag", self._etag)
        self._last_modified = response.headers.get("Last-Modified", self._last_modified)
        items = (
            _parse_rss(response.content, self._content_storage_policy, self._source_timezone)
            if self._source_kind == HistoricalNewsSourceKind.ISSUER_RSS
            else _parse_atom(response.content, self._content_storage_policy, self._source_timezone)
        )
        bounded = [
            item
            for item in items
            if _feed_item_in_range(item, from_datetime=from_datetime, to_datetime=to_datetime)
        ][: min(limit, self._max_items)]
        return HistoricalNewsPage(
            items=bounded,
            next_cursor=None,
            etag=self._etag,
            last_modified=self._last_modified,
        )

    async def _request(self) -> httpx.Response:
        headers = {"User-Agent": self._user_agent}
        if self._etag:
            headers["If-None-Match"] = self._etag
        if self._last_modified:
            headers["If-Modified-Since"] = self._last_modified
        for attempt in range(self._max_retries + 1):
            await self._respect_rate_limit()
            try:
                response = await self._client.get(
                    self._feed_url,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self._max_retries:
                    raise HistoricalSourceContractError("issuer feed request failed") from exc
                await self._backoff(attempt, None)
                continue
            if response.status_code == 304:
                return response
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt >= self._max_retries:
                    raise HistoricalSourceContractError("issuer feed retries exhausted")
                await self._backoff(attempt, response.headers.get("Retry-After"))
                continue
            if response.is_error:
                raise HistoricalSourceContractError(
                    f"issuer feed returned HTTP {response.status_code}"
                )
            return response
        raise HistoricalSourceContractError("issuer feed retries exhausted")

    async def _respect_rate_limit(self) -> None:
        now = monotonic()
        if self._last_request_started is not None:
            remaining = self._min_request_interval_seconds - (now - self._last_request_started)
            if remaining > 0 and self._sleep:
                await asyncio.sleep(remaining)
        self._last_request_started = monotonic()

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = feed_retry_delay(attempt, retry_after)
        if self._sleep:
            await asyncio.sleep(delay)


def feed_retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), 30.0)
        except ValueError:
            return 1.0
    return min(0.25 * (2**attempt), 5.0)


def _parse_rss(
    payload: bytes,
    policy: ContentStoragePolicy,
    source_timezone: str | None,
) -> list[HistoricalSourceItem]:
    root = _xml_root(payload)
    return [
        _feed_source_item(
            source_item_id=_child_text(item, "guid") or _child_text(item, "link"),
            source_url=_child_text(item, "link"),
            title=_child_text(item, "title"),
            published_at=_child_text(item, "pubDate"),
            content=_child_text(item, "description"),
            policy=policy,
            source_timezone=source_timezone,
        )
        for item in root.findall("./channel/item")
    ]


def _parse_atom(
    payload: bytes,
    policy: ContentStoragePolicy,
    source_timezone: str | None,
) -> list[HistoricalSourceItem]:
    root = _xml_root(payload)
    namespace = "{http://www.w3.org/2005/Atom}"
    result: list[HistoricalSourceItem] = []
    for entry in root.findall(f"{namespace}entry"):
        link = entry.find(f"{namespace}link")
        url = "" if link is None else link.attrib.get("href", "")
        result.append(
            _feed_source_item(
                source_item_id=_element_text(entry.find(f"{namespace}id")) or url,
                source_url=url,
                title=_element_text(entry.find(f"{namespace}title")),
                published_at=_element_text(entry.find(f"{namespace}published"))
                or _element_text(entry.find(f"{namespace}updated")),
                content=_element_text(entry.find(f"{namespace}content"))
                or _element_text(entry.find(f"{namespace}summary")),
                policy=policy,
                source_timezone=source_timezone,
            )
        )
    return result


def _xml_root(payload: bytes) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise HistoricalSourceContractError("issuer feed returned malformed XML") from exc


def _feed_source_item(
    *,
    source_item_id: str,
    source_url: str,
    title: str,
    published_at: str,
    content: str,
    policy: ContentStoragePolicy,
    source_timezone: str | None,
) -> HistoricalSourceItem:
    if not source_item_id.strip() or not source_url.strip() or not title.strip():
        raise HistoricalSourceContractError("issuer feed item misses identity fields")
    normalized_timestamp = _normalize_feed_timestamp(published_at)
    return HistoricalSourceItem(
        source_item_id=source_item_id.strip(),
        source_url=source_url.strip(),
        title=title.strip(),
        published_at_text=normalized_timestamp,
        source_timezone=source_timezone,
        content=content or None,
        content_storage_policy=policy,
        content_is_excerpt=policy == ContentStoragePolicy.EXCERPT_ALLOWED,
        original_timestamp_text=published_at,
        corrects_source_item_id=None,
        fetched_at=datetime.now(UTC),
    )


def _normalize_feed_timestamp(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return stripped
    try:
        parsed = parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return stripped
    return parsed.isoformat()


def _feed_item_in_range(
    item: HistoricalSourceItem,
    *,
    from_datetime: datetime,
    to_datetime: datetime,
) -> bool:
    from src.historical_news.domain.time import parse_publication_timestamp

    parsed = parse_publication_timestamp(
        item.published_at_text,
        source_timezone=item.source_timezone,
    )
    return parsed.published_at is None or from_datetime <= parsed.published_at <= to_datetime


def _child_text(element: ElementTree.Element, name: str) -> str:
    return _element_text(element.find(name))


def _element_text(element: ElementTree.Element | None) -> str:
    return "" if element is None or element.text is None else element.text.strip()
