from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin

import httpx

from src.exact_event_corpus.domain import ExactEvent
from src.exact_event_diversity.domain import parse_explicit_utc

USER_AGENT = "trade-ai-news-mvp/0.1"
MAX_RESPONSE_BYTES = 8_000_000
MAX_RETRIES = 3


@dataclass(frozen=True, slots=True)
class OfficialSourceProfile:
    source_code: str
    ticker: str
    issuer: str
    instrument_uid: str
    source_url: str
    allowed_host: str
    timestamp_field: str


async def acquire_x5_wordpress(
    profile: OfficialSourceProfile,
    *,
    date_from: date,
    date_to: date,
    item_limit: int,
    cache_dir: Path,
    client: httpx.AsyncClient | None = None,
) -> list[ExactEvent]:
    _bounded(item_limit, maximum=200)
    rows: list[dict[str, object]] = []
    page_size = min(100, item_limit)
    page_limit = min(5, (item_limit + page_size - 1) // page_size)
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    try:
        for page in range(1, page_limit + 1):
            payload = await _get_json_cached(
                http_client,
                url=profile.source_url,
                params={
                    "per_page": page_size,
                    "page": page,
                    "orderby": "date",
                    "order": "desc",
                    "lang": "ru",
                    "_fields": "id,date,date_gmt,link,slug,title",
                },
                allowed_host=profile.allowed_host,
                cache_path=cache_dir / f"page-{page}.json",
            )
            if not isinstance(payload, list):
                raise ValueError("X5_WORDPRESS_RESPONSE_INVALID")
            page_rows = cast("list[dict[str, object]]", payload)
            rows.extend(page_rows)
            if len(page_rows) < page_size:
                break
    finally:
        if owns_client:
            await http_client.aclose()
    events: list[ExactEvent] = []
    for row in rows:
        raw = str(row.get("date_gmt", ""))
        published = parse_explicit_utc(f"{raw}Z")
        title_payload = row.get("title")
        title = (
            str(cast("dict[str, object]", title_payload).get("rendered", ""))
            if isinstance(title_payload, dict)
            else ""
        )
        event = ExactEvent.create(
            source_code=profile.source_code,
            source_item_id=str(row["id"]),
            canonical_url=str(row["link"]),
            ticker=profile.ticker,
            issuer=profile.issuer,
            instrument_uid=profile.instrument_uid,
            title=html.unescape(re.sub(r"<[^>]+>", " ", title)),
            publication_timestamp_raw=raw,
            publication_timestamp_utc=published,
            timestamp_source_field=profile.timestamp_field,
        )
        if date_from <= event.publication_date <= date_to:
            events.append(event)
    return _ordered_unique(events)[:item_limit]


async def acquire_tbank_public_news(
    profile: OfficialSourceProfile,
    *,
    date_from: date,
    date_to: date,
    item_limit: int,
    cache_dir: Path,
    client: httpx.AsyncClient | None = None,
) -> list[ExactEvent]:
    _bounded(item_limit, maximum=200)
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    try:
        payload = await _get_json_cached(
            http_client,
            url=profile.source_url,
            params={
                "pageOffset": 0,
                "pageSize": item_limit,
                "lang": "ru-RU",
                "partTitle": "Новости",
            },
            allowed_host=profile.allowed_host,
            cache_path=cache_dir / "page-0.json",
        )
    finally:
        if owns_client:
            await http_client.aclose()
    response = _require_dict(_require_dict(payload).get("response"))
    items = response.get("items")
    if not isinstance(items, list):
        raise ValueError("TBANK_PUBLIC_NEWS_RESPONSE_INVALID")
    events: list[ExactEvent] = []
    for row in cast("list[dict[str, object]]", items):
        raw = str(row.get("publishedAt", ""))
        published = parse_explicit_utc(raw)
        title = " ".join(
            part for part in (str(row.get("title", "")), str(row.get("textshort", ""))) if part
        )
        event = ExactEvent.create(
            source_code=profile.source_code,
            source_item_id=str(row["id"]),
            canonical_url=f"https://www.tbank.ru/about/news/{row['slug']}/",
            ticker=profile.ticker,
            issuer=profile.issuer,
            instrument_uid=profile.instrument_uid,
            title=html.unescape(re.sub(r"<[^>]+>", " ", title)),
            publication_timestamp_raw=raw,
            publication_timestamp_utc=published,
            timestamp_source_field=profile.timestamp_field,
        )
        if date_from <= event.publication_date <= date_to:
            events.append(event)
    return _ordered_unique(events)[:item_limit]


async def acquire_vk_next_state(
    profile: OfficialSourceProfile,
    *,
    date_from: date,
    date_to: date,
    item_limit: int,
    cache_dir: Path,
    client: httpx.AsyncClient | None = None,
) -> list[ExactEvent]:
    _bounded(item_limit, maximum=30)
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    try:
        text = await _get_text_cached(
            http_client,
            url=profile.source_url,
            allowed_host=profile.allowed_host,
            cache_path=cache_dir / "page.html",
        )
    finally:
        if owns_client:
            await http_client.aclose()
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("VK_PUBLIC_NEXT_STATE_NOT_FOUND")
    payload = _require_dict(json.loads(match.group(1)))
    props = _require_dict(_require_dict(payload["props"])["pageProps"])
    rows = props.get("publications")
    if not isinstance(rows, list):
        raise ValueError("VK_PUBLICATIONS_INVALID")
    events: list[ExactEvent] = []
    for row in cast("list[dict[str, object]]", rows):
        raw = str(row.get("pub_date", ""))
        published = parse_explicit_utc(raw)
        event = ExactEvent.create(
            source_code=profile.source_code,
            source_item_id=str(row["id"]),
            canonical_url=urljoin("https://vk.company/ru/", str(row["public_url"])),
            ticker=profile.ticker,
            issuer=profile.issuer,
            instrument_uid=profile.instrument_uid,
            title=str(row["title"]),
            publication_timestamp_raw=raw,
            publication_timestamp_utc=published,
            timestamp_source_field=profile.timestamp_field,
        )
        if date_from <= event.publication_date <= date_to:
            events.append(event)
    return _ordered_unique(events)[:item_limit]


async def acquire_embedded_app_state(
    profile: OfficialSourceProfile,
    *,
    date_from: date,
    date_to: date,
    item_limit: int,
    cache_dir: Path,
    client: httpx.AsyncClient | None = None,
) -> list[ExactEvent]:
    _bounded(item_limit, maximum=100)
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    try:
        text = await _get_text_cached(
            http_client,
            url=profile.source_url,
            allowed_host=profile.allowed_host,
            cache_path=cache_dir / "page.html",
        )
    finally:
        if owns_client:
            await http_client.aclose()
    match = re.search(r"\bApp\s*=\s*", text)
    if match is None:
        raise ValueError("OFFICIAL_APP_STATE_NOT_FOUND")
    payload = json.JSONDecoder().raw_decode(text[match.end() :])[0]
    rows = _require_dict(_require_dict(payload).get("news", {})).get("items")
    if not isinstance(rows, list):
        raise ValueError("OFFICIAL_APP_NEWS_INVALID")
    events: list[ExactEvent] = []
    for row in cast("list[dict[str, object]]", rows):
        timestamp = row.get("activeFrom")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            continue
        published = datetime.fromtimestamp(timestamp, UTC)
        if (published + timedelta(hours=3)).time() == datetime.min.time():
            continue
        event = ExactEvent.create(
            source_code=profile.source_code,
            source_item_id=str(row["detailPageUrl"]),
            canonical_url=urljoin(profile.source_url, str(row["detailPageUrl"])),
            ticker=profile.ticker,
            issuer=profile.issuer,
            instrument_uid=profile.instrument_uid,
            title=str(row["name"]),
            publication_timestamp_raw=str(timestamp),
            publication_timestamp_utc=published,
            timestamp_source_field=profile.timestamp_field,
        )
        if date_from <= event.publication_date <= date_to:
            events.append(event)
    return _ordered_unique(events)[:item_limit]


async def acquire_moex_rss(
    profile: OfficialSourceProfile,
    *,
    date_from: date,
    date_to: date,
    item_limit: int,
    cache_dir: Path,
    required_phrases: tuple[str, ...] = (),
    rejected_phrases: tuple[str, ...] = (),
    client: httpx.AsyncClient | None = None,
) -> list[ExactEvent]:
    _bounded(item_limit, maximum=100)
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=httpx.Timeout(90.0))
    try:
        content = await _get_bytes_cached(
            http_client,
            url=profile.source_url,
            allowed_host=profile.allowed_host,
            cache_path=cache_dir / "feed.xml",
        )
    finally:
        if owns_client:
            await http_client.aclose()
    root = ET.fromstring(content)
    events: list[ExactEvent] = []
    for item in root.findall("./channel/item"):
        title = html.unescape(item.findtext("title") or "")
        description = html.unescape(item.findtext("description") or "")
        searchable = f"{title} {description}"
        if required_phrases and not any(phrase in searchable for phrase in required_phrases):
            continue
        if any(phrase in searchable for phrase in rejected_phrases):
            continue
        raw = item.findtext("pubDate") or ""
        published = parsedate_to_datetime(raw)
        if published.tzinfo is None:
            raise ValueError("TIMESTAMP_TIMEZONE_UNRESOLVED")
        link = item.findtext("link") or ""
        event = ExactEvent.create(
            source_code=profile.source_code,
            source_item_id=link,
            canonical_url=link,
            ticker=profile.ticker,
            issuer=profile.issuer,
            instrument_uid=profile.instrument_uid,
            title=title,
            publication_timestamp_raw=raw,
            publication_timestamp_utc=published,
            timestamp_source_field=profile.timestamp_field,
        )
        if date_from <= event.publication_date <= date_to:
            events.append(event)
    return sorted(
        _ordered_unique(events),
        key=lambda item: (item.publication_timestamp_utc, item.source_item_id),
        reverse=True,
    )[:item_limit]


async def _get_json_cached(
    client: httpx.AsyncClient,
    *,
    url: str,
    params: dict[str, str | int],
    allowed_host: str,
    cache_path: Path,
) -> object:
    content = await _get_bytes_cached(
        client,
        url=url,
        params=params,
        allowed_host=allowed_host,
        cache_path=cache_path,
    )
    return json.loads(content)


async def _get_text_cached(
    client: httpx.AsyncClient,
    *,
    url: str,
    allowed_host: str,
    cache_path: Path,
) -> str:
    return (
        await _get_bytes_cached(
            client,
            url=url,
            allowed_host=allowed_host,
            cache_path=cache_path,
        )
    ).decode("utf-8-sig")


async def _get_bytes_cached(
    client: httpx.AsyncClient,
    *,
    url: str,
    allowed_host: str,
    cache_path: Path,
    params: dict[str, str | int] | None = None,
) -> bytes:
    parsed_url = httpx.URL(url)
    if parsed_url.scheme != "https" or parsed_url.host != allowed_host:
        raise RuntimeError("OFFICIAL_SOURCE_URL_REJECTED")
    request_url = str(httpx.URL(url, params=params))
    metadata_path = cache_path.with_suffix(f"{cache_path.suffix}.meta.json")
    cache_exists, metadata_exists = await asyncio.gather(
        asyncio.to_thread(cache_path.exists),
        asyncio.to_thread(metadata_path.exists),
    )
    if cache_exists and metadata_exists:
        content = await asyncio.to_thread(cache_path.read_bytes)
        metadata = json.loads(await asyncio.to_thread(metadata_path.read_text, encoding="utf-8"))
        if (
            metadata.get("request_url") == request_url
            and metadata.get("sha256") == hashlib.sha256(content).hexdigest()
        ):
            return content
    response: httpx.Response | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(url, params=params, headers={"User-Agent": USER_AGENT})
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError):
            if attempt + 1 >= MAX_RETRIES:
                raise RuntimeError("OFFICIAL_SOURCE_UNAVAILABLE") from None
            await asyncio.sleep(0.25 * (2**attempt))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            if attempt + 1 >= MAX_RETRIES:
                raise RuntimeError(f"OFFICIAL_SOURCE_HTTP_{response.status_code}")
            await asyncio.sleep(0.25 * (2**attempt))
            continue
        break
    if response is None or response.status_code != 200:
        status = "NO_RESPONSE" if response is None else str(response.status_code)
        raise RuntimeError(f"OFFICIAL_SOURCE_HTTP_{status}")
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise RuntimeError("OFFICIAL_SOURCE_PAYLOAD_TOO_LARGE")
    resolved = response.url
    if resolved.scheme != "https" or resolved.host != allowed_host:
        raise RuntimeError("OFFICIAL_SOURCE_DOMAIN_REDIRECT_REJECTED")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(cache_path.write_bytes, response.content)
    await asyncio.to_thread(
        metadata_path.write_text,
        json.dumps(
            {
                "request_url": request_url,
                "resolved_url": str(resolved),
                "sha256": hashlib.sha256(response.content).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return response.content


def _bounded(item_limit: int, *, maximum: int) -> None:
    if not 1 <= item_limit <= maximum:
        raise ValueError("OFFICIAL_SOURCE_ACQUISITION_MUST_BE_BOUNDED")


def _require_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("OFFICIAL_SOURCE_OBJECT_REQUIRED")
    return cast("dict[str, Any]", value)


def _ordered_unique(events: Iterable[ExactEvent]) -> list[ExactEvent]:
    unique = {event.source_item_id: event for event in events}
    return sorted(
        unique.values(), key=lambda item: (item.publication_timestamp_utc, item.source_item_id)
    )
