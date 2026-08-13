from __future__ import annotations

import asyncio
import hashlib
import html
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urljoin

import httpx

from src.event_market_dataset.domain import AcquiredEvent
from src.news.domain.enums import PublicationTimestampQuality

USER_AGENT = "trade-ai-news-mvp/0.1"
MAX_RESPONSE_BYTES = 8_000_000
MAX_RETRIES = 3


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
            parsed = (
                parse_yandex_archive(payload, base_url=resolved_url)
                if config.source_code == "YANDEX_IR_ARCHIVE_DATE_ONLY"
                else parse_novatek_archive(payload, base_url=resolved_url)
            )
            if not parsed:
                raise RuntimeError(f"OFFICIAL_ARCHIVE_PARSE_EMPTY:{config.source_code}:{page}")
            for source_item_id, source_url, title, publication_date in parsed:
                if not date_from <= publication_date <= date_to:
                    continue
                events.append(
                    AcquiredEvent.create(
                        source_code=config.source_code,
                        source_item_id=source_item_id,
                        source_url=source_url,
                        ticker=config.ticker,
                        issuer_name=config.issuer_name,
                        instrument_uid=config.instrument_uid,
                        figi=config.figi,
                        title=title,
                        publication_date=publication_date,
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


def parse_yandex_archive(payload: str, *, base_url: str) -> list[tuple[str, str, str, date]]:
    pattern = re.compile(
        r'<a\s+class="press-release-item__link"\s+href="(?P<href>[^"]+)"[^>]*>'
        r'.*?<span\s+class="date">(?P<date>.*?)</span>'
        r'.*?<span\s+class="press-release-item__title">(?P<title>.*?)</span>',
        re.IGNORECASE | re.DOTALL,
    )
    result: list[tuple[str, str, str, date]] = []
    for match in pattern.finditer(payload):
        href = html.unescape(match.group("href"))
        title = _plain(match.group("title"))
        published = _parse_russian_date(_plain(match.group("date")))
        canonical = urljoin(base_url, href)
        result.append((canonical, canonical, title, published))
    return result


def parse_novatek_archive(payload: str, *, base_url: str) -> list[tuple[str, str, str, date]]:
    pattern = re.compile(
        r'<div\s+class="date">(?P<date>.*?)</div>.*?'
        r'<a\s+href="/en/press/releases/index\.php\?id_4=(?P<id>\d+)[^"]*">'
        r"(?P<title>.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    result: list[tuple[str, str, str, date]] = []
    for match in pattern.finditer(payload):
        title = _plain(match.group("title"))
        published = datetime.strptime(_plain(match.group("date")), "%d %B %Y").date()
        canonical = urljoin(
            base_url,
            f"/en/press/releases/index.php?id_4={match.group('id')}",
        )
        result.append((canonical, canonical, title, published))
    return result


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
    months = {
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
    day, month, year = value.lower().split()
    return date(int(year), months[month], int(day))
