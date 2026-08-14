from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin

import httpx

from src.event_market_dataset.domain import AcquiredEvent
from src.news.domain.enums import PublicationTimestampQuality

USER_AGENT = "trade-ai-news-mvp/0.1"
MAX_RESPONSE_BYTES = 8_000_000
MAX_RETRIES = 3
PARSER_VERSION = "official-archive-parser-v2"


@dataclass(frozen=True, slots=True)
class ArchiveSourceConfig:
    source_code: str
    source_name: str
    official_owner: str
    ticker: str
    issuer_name: str
    instrument_uid: str
    figi: str | None
    url_template: str
    page_values: tuple[int, ...]
    source_type: str
    collection_method: str
    historical_range: str
    live_supported: bool
    parser_profile: str = "NOVATEK"
    collector_family: str = "BOUNDED_OFFICIAL_HTML_ARCHIVE"


@dataclass(frozen=True, slots=True)
class ParsedPublication:
    source_item_id: str
    source_url: str
    title: str
    publication_date: date


async def acquire_archive(
    config: ArchiveSourceConfig,
    *,
    date_from: date,
    date_to: date,
    limit: int,
    cache_dir: Path | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[AcquiredEvent]:
    if not 1 <= limit <= 10_000:
        raise ValueError("archive limit must be bounded")
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    events: list[AcquiredEvent] = []
    try:
        for page in config.page_values:
            url = config.url_template.format(page=page)
            payload, resolved_url = await _get_cached(
                http_client,
                url,
                cache_dir=cache_dir,
                cache_key=str(page),
            )
            parsed = parse_official_archive(
                payload,
                base_url=resolved_url,
                profile=config.parser_profile,
            )
            if not parsed:
                raise RuntimeError(f"OFFICIAL_ARCHIVE_PARSE_EMPTY:{config.source_code}:{page}")
            for item in parsed:
                if not date_from <= item.publication_date <= date_to:
                    continue
                events.append(
                    AcquiredEvent.create(
                        source_code=config.source_code,
                        source_item_id=item.source_item_id,
                        source_url=item.source_url,
                        ticker=config.ticker,
                        issuer_name=config.issuer_name,
                        instrument_uid=config.instrument_uid,
                        figi=config.figi,
                        title=item.title,
                        publication_date=item.publication_date,
                        published_at=None,
                        timestamp_quality=PublicationTimestampQuality.DATE_ONLY,
                    )
                )
                if len(events) >= limit:
                    return events
    finally:
        if owns_client:
            await http_client.aclose()
    return events


def parse_official_archive(
    payload: str,
    *,
    base_url: str,
    profile: str,
) -> list[ParsedPublication]:
    parsers = {
        "YANDEX": _parse_yandex,
        "NOVATEK": _parse_novatek,
        "LUKOIL": _parse_lukoil,
        "TATNEFT": _parse_tatneft,
        "PHOSAGRO": _parse_phosagro,
        "POLYUS": _parse_polyus,
        "INTERRAO": _parse_interrao,
        "ALROSA": _parse_alrosa,
        "NORNICKEL_APP": _parse_nornickel_app,
        "MAGNIT_APP": _parse_magnit_app,
    }
    parser = parsers.get(profile)
    if parser is None:
        raise ValueError(f"unsupported official archive parser profile: {profile}")
    parsed = parser(payload, base_url)
    unique = {item.source_item_id: item for item in parsed}
    return sorted(
        unique.values(),
        key=lambda item: (item.publication_date, item.source_item_id),
        reverse=True,
    )


def parse_yandex_archive(payload: str, *, base_url: str) -> list[tuple[str, str, str, date]]:
    return [
        (item.source_item_id, item.source_url, item.title, item.publication_date)
        for item in _parse_yandex(payload, base_url)
    ]


def parse_novatek_archive(payload: str, *, base_url: str) -> list[tuple[str, str, str, date]]:
    return [
        (item.source_item_id, item.source_url, item.title, item.publication_date)
        for item in _parse_novatek(payload, base_url)
    ]


def _parse_yandex(payload: str, base_url: str) -> list[ParsedPublication]:
    pattern = re.compile(
        r'<a\s+class="press-release-item__link"\s+href="(?P<href>[^"]+)"[^>]*>'
        r'.*?<span\s+class="date">(?P<date>.*?)</span>'
        r'.*?<span\s+class="press-release-item__title">(?P<title>.*?)</span>',
        re.IGNORECASE | re.DOTALL,
    )
    return [
        _publication(
            base_url,
            html.unescape(match.group("href")),
            _plain(match.group("title")),
            _parse_russian_date(_plain(match.group("date"))),
        )
        for match in pattern.finditer(payload)
    ]


def _parse_novatek(payload: str, base_url: str) -> list[ParsedPublication]:
    pattern = re.compile(
        r'<div\s+class="date">(?P<date>.*?)</div>.*?'
        r'<a\s+href="/en/press/releases/index\.php\?id_4=(?P<id>\d+)[^"]*">'
        r"(?P<title>.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    result: list[ParsedPublication] = []
    for match in pattern.finditer(payload):
        href = f"/en/press/releases/index.php?id_4={match.group('id')}"
        result.append(
            _publication(
                base_url,
                href,
                _plain(match.group("title")),
                datetime.strptime(_plain(match.group("date")), "%d %B %Y").date(),
            )
        )
    return result


def _parse_lukoil(payload: str, base_url: str) -> list[ParsedPublication]:
    match = re.search(
        r'<script[^>]+class="pressreleases-data"[^>]*>\s*(?P<data>\{.*?\})\s*</script>',
        payload,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return []
    data = cast("dict[str, object]", json.loads(match.group("data")))
    items = data.get("Items")
    if not isinstance(items, list):
        return []
    result: list[ParsedPublication] = []
    for raw_item in cast("list[object]", items):
        if not isinstance(raw_item, dict):
            continue
        item = cast("dict[str, object]", raw_item)
        slug = str(item.get("FriendlyUrl", ""))
        title = str(item.get("Name", ""))
        published = str(item.get("PublicationDate", ""))[:10]
        if slug and title and published:
            result.append(
                _publication(
                    base_url,
                    f"/PressCenter/Pressreleases/Pressrelease/{slug}",
                    title,
                    date.fromisoformat(published),
                )
            )
    return result


def _parse_tatneft(payload: str, base_url: str) -> list[ParsedPublication]:
    pattern = re.compile(
        r'<div\s+class="material\b.*?'
        r'<span\s+class="material__date">(?P<date>.*?)</span>.*?'
        r'<a\s+href="(?P<href>[^"]+/more/(?P<id>\d+)/[^\"]*)"\s+'
        r'class="material__title">(?P<title>.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    return [
        _publication(
            base_url,
            html.unescape(match.group("href")),
            _plain(match.group("title")),
            _parse_russian_date(_plain(match.group("date"))),
            source_item_id=match.group("id"),
        )
        for match in pattern.finditer(payload)
    ]


def _parse_phosagro(payload: str, base_url: str) -> list[ParsedPublication]:
    pattern = re.compile(
        r'<a\s+href="(?P<href>/press/company/[^"]+)"\s+'
        r'class="press__card[^\"]*">.*?<h4[^>]*>(?P<title>.*?)</h4>.*?'
        r'<div\s+class="press__card-date">(?P<date>.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    )
    return [
        _publication(
            base_url,
            match.group("href"),
            _plain(match.group("title")),
            _parse_russian_date(_plain(match.group("date"))),
        )
        for match in pattern.finditer(payload)
    ]


def _parse_polyus(payload: str, base_url: str) -> list[ParsedPublication]:
    pattern = re.compile(
        r'<article\s+class="news-item">.*?<a\s+href="(?P<href>[^"]+)">'
        r'(?P<title>.*?)</a>.*?<time[^>]+datetime="(?P<date>\d{4}-\d{2}-\d{2})"',
        re.IGNORECASE | re.DOTALL,
    )
    return [
        _publication(
            base_url,
            match.group("href"),
            _plain(match.group("title")),
            date.fromisoformat(match.group("date")),
        )
        for match in pattern.finditer(payload)
    ]


def _parse_interrao(payload: str, base_url: str) -> list[ParsedPublication]:
    pattern = re.compile(
        r'<div\s+class="news-item[^\"]*"[^>]*>\s*'
        r'<div\s+class="date">(?P<date>.*?)</div>\s*'
        r'<div\s+class="name"><a\s+href="(?P<href>detail\.php\?ID=(?P<id>\d+))"[^>]*>'
        r"(?P<title>.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    return [
        _publication(
            base_url,
            match.group("href"),
            _plain(match.group("title")),
            _parse_russian_date(_plain(match.group("date"))),
            source_item_id=match.group("id"),
        )
        for match in pattern.finditer(payload)
    ]


def _parse_alrosa(payload: str, base_url: str) -> list[ParsedPublication]:
    marker = ':news="'
    start = payload.find(marker)
    if start < 0:
        return []
    decoded = html.unescape(payload[start + len(marker) :])
    try:
        values = cast("object", json.JSONDecoder().raw_decode(decoded)[0])
    except json.JSONDecodeError:
        return []
    result: list[ParsedPublication] = []
    raw_items = cast("list[object]", values) if isinstance(values, list) else []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        item = cast("dict[str, object]", raw_item)
        href = str(item.get("url", ""))
        title = str(item.get("caption", ""))
        published = str(item.get("date", ""))
        source_id = str(item.get("id", ""))
        if href and title and published and source_id:
            result.append(
                _publication(
                    base_url,
                    href,
                    title,
                    _parse_short_russian_date(published),
                    source_item_id=source_id,
                )
            )
    return result


def _parse_nornickel_app(payload: str, base_url: str) -> list[ParsedPublication]:
    app = _app_payload(payload)
    result: list[ParsedPublication] = []
    for item in _walk_dicts(app):
        if not {"name", "detailPageUrl", "activeFrom"} <= item.keys():
            continue
        href = item.get("detailPageUrl")
        title = item.get("name")
        timestamp = item.get("activeFrom")
        if isinstance(href, str) and isinstance(title, str) and isinstance(timestamp, int):
            result.append(
                _publication(
                    base_url,
                    href,
                    title,
                    datetime.fromtimestamp(timestamp, UTC).date(),
                    source_item_id=str(item.get("code") or href),
                )
            )
    return result


def _parse_magnit_app(payload: str, base_url: str) -> list[ParsedPublication]:
    app = _app_payload(payload)
    result: list[ParsedPublication] = []
    for item in _walk_dicts(app):
        if not {"date", "name", "link"} <= item.keys():
            continue
        href = item.get("link")
        title = item.get("name")
        timestamp = item.get("date")
        if isinstance(href, str) and isinstance(title, str) and isinstance(timestamp, int):
            result.append(
                _publication(
                    base_url,
                    href,
                    title,
                    datetime.fromtimestamp(timestamp, UTC).date(),
                    source_item_id=href,
                )
            )
    return result


def _app_payload(payload: str) -> object:
    match = re.search(r"\bApp\s*=\s*", payload)
    if match is None:
        return {}
    try:
        value = cast("object", json.JSONDecoder().raw_decode(payload[match.end() :])[0])
    except json.JSONDecodeError:
        return {}
    return value


def _walk_dicts(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        yield mapping
        for child in mapping.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in cast("list[object]", value):
            yield from _walk_dicts(child)


def _publication(
    base_url: str,
    href: str,
    title: str,
    published: date,
    *,
    source_item_id: str | None = None,
) -> ParsedPublication:
    canonical = urljoin(base_url, html.unescape(href))
    return ParsedPublication(
        source_item_id=source_item_id or canonical,
        source_url=canonical,
        title=_plain(title),
        publication_date=published,
    )


async def _get_cached(
    client: httpx.AsyncClient,
    url: str,
    *,
    cache_dir: Path | None,
    cache_key: str,
) -> tuple[str, str]:
    payload_path = None if cache_dir is None else cache_dir / f"{cache_key}.html"
    metadata_path = None if cache_dir is None else cache_dir / f"{cache_key}.json"
    if cache_dir is not None:
        assert payload_path is not None and metadata_path is not None
        if payload_path.exists() and metadata_path.exists():
            payload = payload_path.read_text(encoding="utf-8")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if metadata.get("sha256") == digest and metadata.get("url") == url:
                return payload, str(metadata["resolved_url"])
    response = await _get(client, url)
    payload = response.text
    if cache_dir is not None:
        assert payload_path is not None and metadata_path is not None
        await asyncio.to_thread(
            _write_cache,
            cache_dir,
            payload_path,
            metadata_path,
            payload,
            url,
            str(response.url),
        )
    return payload, str(response.url)


def _write_cache(
    cache_dir: Path,
    payload_path: Path,
    metadata_path: Path,
    payload: str,
    url: str,
    resolved_url: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(payload, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "url": url,
                "resolved_url": resolved_url,
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


async def _get(client: httpx.AsyncClient, url: str) -> httpx.Response:
    if not url.startswith("https://"):
        raise ValueError("official archive URL must use HTTPS")
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await client.get(url, headers={"User-Agent": USER_AGENT})
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt >= MAX_RETRIES:
                raise RuntimeError("OFFICIAL_ARCHIVE_UNAVAILABLE") from exc
            await asyncio.sleep(min(0.5 * (2**attempt), 4.0))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            if attempt >= MAX_RETRIES:
                raise RuntimeError(f"OFFICIAL_ARCHIVE_HTTP_{response.status_code}")
            await asyncio.sleep(min(0.5 * (2**attempt), 4.0))
            continue
        if response.is_error:
            raise RuntimeError(f"OFFICIAL_ARCHIVE_HTTP_{response.status_code}")
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise RuntimeError("OFFICIAL_ARCHIVE_PAYLOAD_TOO_LARGE")
        return response
    raise RuntimeError("OFFICIAL_ARCHIVE_UNAVAILABLE")


def _plain(value: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(no_tags).split())


def _parse_russian_date(value: str) -> date:
    cleaned = value.lower().replace("\u0433.", "").replace(",", " ")
    day, month, year, *_rest = cleaned.split()
    return date(int(year), _RUSSIAN_MONTHS[month], int(day))


def _parse_short_russian_date(value: str) -> date:
    day, month, year = value.lower().split()
    return date(int(year), _SHORT_RUSSIAN_MONTHS[month.rstrip(".")], int(day))


_RUSSIAN_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

_SHORT_RUSSIAN_MONTHS = {
    "янв": 1,
    "фев": 2,
    "мар": 3,
    "апр": 4,
    "мая": 5,
    "июн": 6,
    "июл": 7,
    "авг": 8,
    "сен": 9,
    "окт": 10,
    "ноя": 11,
    "дек": 12,
}
