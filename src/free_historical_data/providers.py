from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from html.parser import HTMLParser
from time import monotonic
from typing import Any, Protocol, cast
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx

from src.historical_news.domain.entities import HistoricalNewsPage, HistoricalSourceItem
from src.historical_news.domain.enums import ContentStoragePolicy
from src.historical_news.domain.time import parse_publication_timestamp
from src.historical_news.infrastructure.local_archive import HistoricalSourceContractError


class ArticleParser(Protocol):
    def __call__(
        self, source_url: str, payload: bytes, fetched_at: datetime
    ) -> HistoricalSourceItem | None: ...


class ArchivePageParser(Protocol):
    def __call__(
        self, page_url: str, payload: bytes, fetched_at: datetime
    ) -> tuple[list[HistoricalSourceItem], bool]: ...


class JsonPageParser(Protocol):
    def __call__(
        self, payload: bytes, fetched_at: datetime
    ) -> tuple[list[HistoricalSourceItem], bool]: ...


class BoundedHttpClient:
    def __init__(
        self,
        *,
        allowed_hosts: tuple[str, ...],
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        min_request_interval_seconds: float = 0.5,
        concurrency: int = 1,
        max_response_bytes: int = 5_000_000,
        user_agent: str = "trade-ai-zero-cost-acquisition/1.0",
        client: httpx.AsyncClient | None = None,
        sleep: bool = True,
    ) -> None:
        hosts = tuple(host.strip().lower() for host in allowed_hosts if host.strip())
        if not hosts:
            raise ValueError("at least one allowed host is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if min_request_interval_seconds < 0.1:
            raise ValueError("request interval must be at least 0.1 seconds")
        if not 1 <= concurrency <= 2:
            raise ValueError("concurrency must be between 1 and 2")
        if not 1 <= max_response_bytes <= 20_000_000:
            raise ValueError("max_response_bytes must be between 1 and 20000000")
        if not user_agent.strip():
            raise ValueError("user_agent must not be blank")
        self._allowed_hosts = hosts
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._min_interval = min_request_interval_seconds
        self._max_response_bytes = max_response_bytes
        self._user_agent = user_agent
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self._owns_client = client is None
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(concurrency)
        self._rate_lock = asyncio.Lock()
        self._last_request_started: float | None = None
        self.request_count = 0

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get(self, url: str) -> bytes:
        self._validate_url(url)
        async with self._semaphore:
            for attempt in range(self._max_retries + 1):
                await self._wait_for_slot()
                self.request_count += 1
                try:
                    response = await self._client.get(
                        url,
                        headers={"User-Agent": self._user_agent},
                        timeout=self._timeout_seconds,
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt >= self._max_retries:
                        raise HistoricalSourceContractError("bounded request failed") from exc
                    await self._backoff(attempt)
                    continue
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    if attempt >= self._max_retries:
                        raise HistoricalSourceContractError("bounded request retries exhausted")
                    await self._backoff(attempt)
                    continue
                if response.is_error:
                    raise HistoricalSourceContractError(
                        f"bounded request returned HTTP {response.status_code}"
                    )
                if len(response.content) > self._max_response_bytes:
                    raise HistoricalSourceContractError("bounded response exceeds size limit")
                return response.content
        raise HistoricalSourceContractError("bounded request retries exhausted")

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise HistoricalSourceContractError("provider URL must be credential-free HTTPS")
        host = parsed.hostname.lower()
        if not any(
            host == allowed or host.endswith(f".{allowed}") for allowed in self._allowed_hosts
        ):
            raise HistoricalSourceContractError("provider URL is outside the domain allowlist")

    async def _wait_for_slot(self) -> None:
        async with self._rate_lock:
            now = monotonic()
            if self._last_request_started is not None:
                remaining = self._min_interval - (now - self._last_request_started)
                if remaining > 0 and self._sleep:
                    await asyncio.sleep(remaining)
            self._last_request_started = monotonic()

    async def _backoff(self, attempt: int) -> None:
        if self._sleep:
            await asyncio.sleep(min(0.25 * (2**attempt), 5.0))


class SitemapArchiveNewsSource:
    def __init__(
        self,
        *,
        sitemap_urls: tuple[str, ...],
        article_parser: ArticleParser,
        transport: BoundedHttpClient,
        news_url_filter: Callable[[str], bool],
        max_sitemaps: int = 10,
        max_article_requests: int = 200,
    ) -> None:
        if not sitemap_urls:
            raise ValueError("at least one sitemap URL is required")
        if not 1 <= max_sitemaps <= 20:
            raise ValueError("max_sitemaps must be between 1 and 20")
        if not 1 <= max_article_requests <= 200:
            raise ValueError("max_article_requests must be between 1 and 200")
        self._sitemap_urls = sitemap_urls
        self._article_parser = article_parser
        self._transport = transport
        self._news_url_filter = news_url_filter
        self._max_sitemaps = max_sitemaps
        self._max_article_requests = max_article_requests

    async def fetch_items(
        self,
        *,
        from_datetime: datetime,
        to_datetime: datetime,
        cursor: str | None,
        limit: int,
    ) -> HistoricalNewsPage:
        offset = _cursor_offset(cursor)
        urls = await self._discover_urls()
        selected = urls[offset : offset + min(limit, self._max_article_requests)]
        items: list[HistoricalSourceItem] = []
        for url in selected:
            fetched_at = datetime.now(UTC)
            parsed = self._article_parser(url, await self._transport.get(url), fetched_at)
            if parsed is not None and _in_range(parsed, from_datetime, to_datetime):
                items.append(parsed)
        next_offset = offset + len(selected)
        return HistoricalNewsPage(
            items=items,
            next_cursor=str(next_offset) if next_offset < len(urls) else None,
        )

    async def _discover_urls(self) -> list[str]:
        pending = list(self._sitemap_urls)
        seen_sitemaps: set[str] = set()
        article_urls: set[str] = set()
        while pending and len(seen_sitemaps) < self._max_sitemaps:
            sitemap_url = pending.pop(0)
            if sitemap_url in seen_sitemaps:
                continue
            seen_sitemaps.add(sitemap_url)
            nested, urls = parse_sitemap(await self._transport.get(sitemap_url))
            pending.extend(url for url in nested if url not in seen_sitemaps)
            article_urls.update(url for url in urls if self._news_url_filter(url))
        return sorted(article_urls)


class PaginatedIssuerArchiveNewsSource:
    def __init__(
        self,
        *,
        page_url: Callable[[int], str],
        page_parser: ArchivePageParser,
        transport: BoundedHttpClient,
        max_pages: int = 20,
    ) -> None:
        if not 1 <= max_pages <= 100:
            raise ValueError("max_pages must be between 1 and 100")
        self._page_url = page_url
        self._page_parser = page_parser
        self._transport = transport
        self._max_pages = max_pages

    async def fetch_items(
        self,
        *,
        from_datetime: datetime,
        to_datetime: datetime,
        cursor: str | None,
        limit: int,
    ) -> HistoricalNewsPage:
        page_number = 1 if cursor is None else _positive_cursor(cursor)
        if page_number > self._max_pages:
            return HistoricalNewsPage(items=[], next_cursor=None)
        url = self._page_url(page_number)
        items, has_more = self._page_parser(
            url,
            await self._transport.get(url),
            datetime.now(UTC),
        )
        bounded = [item for item in items if _in_range(item, from_datetime, to_datetime)][:limit]
        next_cursor = str(page_number + 1) if has_more and page_number < self._max_pages else None
        return HistoricalNewsPage(items=bounded, next_cursor=next_cursor)


class PublicJsonNewsSource:
    def __init__(
        self,
        *,
        page_url: Callable[[int, int], str],
        page_parser: JsonPageParser,
        transport: BoundedHttpClient,
        max_pages: int = 20,
    ) -> None:
        if not 1 <= max_pages <= 100:
            raise ValueError("max_pages must be between 1 and 100")
        self._page_url = page_url
        self._page_parser = page_parser
        self._transport = transport
        self._max_pages = max_pages

    async def fetch_items(
        self,
        *,
        from_datetime: datetime,
        to_datetime: datetime,
        cursor: str | None,
        limit: int,
    ) -> HistoricalNewsPage:
        page_number = 1 if cursor is None else _positive_cursor(cursor)
        if page_number > self._max_pages:
            return HistoricalNewsPage(items=[], next_cursor=None)
        items, has_more = self._page_parser(
            await self._transport.get(self._page_url(page_number, limit)),
            datetime.now(UTC),
        )
        bounded = [item for item in items if _in_range(item, from_datetime, to_datetime)][:limit]
        next_cursor = str(page_number + 1) if has_more and page_number < self._max_pages else None
        return HistoricalNewsPage(items=bounded, next_cursor=next_cursor)


class JsonLdArticleParser:
    def __init__(
        self,
        *,
        storage_policy: ContentStoragePolicy,
        source_timezone: str | None,
    ) -> None:
        self._storage_policy = storage_policy
        self._source_timezone = source_timezone

    def __call__(
        self, source_url: str, payload: bytes, fetched_at: datetime
    ) -> HistoricalSourceItem | None:
        parser = _JsonLdHtmlParser()
        try:
            parser.feed(payload.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise HistoricalSourceContractError("article is not UTF-8") from exc
        article = next((_article_node(node) for node in parser.nodes if _article_node(node)), None)
        if article is None:
            return None
        title = _string(article.get("headline")) or _string(article.get("name"))
        published = _string(article.get("datePublished"))
        canonical = _string(article.get("url")) or source_url
        if not title or not published:
            return None
        description = _string(article.get("description"))
        return HistoricalSourceItem(
            source_item_id=canonical,
            source_url=canonical,
            title=title,
            published_at_text=published,
            source_timezone=self._source_timezone,
            content=description or None,
            content_storage_policy=self._storage_policy,
            content_is_excerpt=True,
            original_timestamp_text=published,
            corrects_source_item_id=None,
            fetched_at=fetched_at,
        )


def parse_sitemap(payload: bytes) -> tuple[list[str], list[str]]:
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise HistoricalSourceContractError("sitemap returned malformed XML") from exc
    nested: list[str] = []
    urls: list[str] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "loc" or element.text is None:
            continue
        value = element.text.strip()
        parent_kind = root.tag.rsplit("}", 1)[-1]
        if parent_kind == "sitemapindex":
            nested.append(value)
        elif parent_kind == "urlset":
            urls.append(value)
    return nested, urls


class _JsonLdHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._capture = False
        self._buffer: list[str] = []
        self.nodes: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        content_type = attributes.get("type") or ""
        if tag.lower() == "script" and content_type.lower() == "application/ld+json":
            self._capture = True
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "script" or not self._capture:
            return
        self._capture = False
        try:
            value = cast("object", json.loads("".join(self._buffer)))
        except json.JSONDecodeError:
            return
        self.nodes.extend(_jsonld_nodes(value))


def _jsonld_nodes(value: object) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        typed = cast("dict[str, Any]", value)
        graph = typed.get("@graph")
        if isinstance(graph, list):
            items = cast("list[object]", graph)
            return [cast("dict[str, Any]", item) for item in items if isinstance(item, dict)]
        return [typed]
    if isinstance(value, list):
        items = cast("list[object]", value)
        return [cast("dict[str, Any]", item) for item in items if isinstance(item, dict)]
    return []


def _article_node(node: dict[str, Any]) -> dict[str, Any] | None:
    kind = node.get("@type")
    kinds = (
        {str(item) for item in cast("list[object]", kind)}
        if isinstance(kind, list)
        else {str(kind)}
    )
    return node if kinds & {"Article", "NewsArticle", "PressRelease"} else None


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _cursor_offset(cursor: str | None) -> int:
    if cursor is None:
        return 0
    try:
        value = int(cursor)
    except ValueError as exc:
        raise HistoricalSourceContractError("invalid provider cursor") from exc
    if value < 0:
        raise HistoricalSourceContractError("invalid provider cursor")
    return value


def _positive_cursor(cursor: str) -> int:
    value = _cursor_offset(cursor)
    if value < 1:
        raise HistoricalSourceContractError("invalid provider cursor")
    return value


def _in_range(item: HistoricalSourceItem, date_from: datetime, date_to: datetime) -> bool:
    parsed = parse_publication_timestamp(
        item.published_at_text,
        source_timezone=item.source_timezone,
    )
    return parsed.published_at is None or date_from <= parsed.published_at <= date_to
