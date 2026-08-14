from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast
from urllib.parse import urljoin

import httpx

from src.exact_event_corpus.domain import ExactEvent

USER_AGENT = "trade-ai-news-mvp/0.1"
MAX_RESPONSE_BYTES = 2_000_000
MAX_RETRIES = 3


@dataclass(frozen=True, slots=True)
class ExactAppStateProfile:
    source_code: str
    ticker: str
    issuer: str
    instrument_uid: str
    base_url: str
    timestamp_field: str
    title_field: str
    url_field: str
    identity_field: str | None = None
    reject_source_local_midnight: bool = False
    source_utc_offset_minutes: int = 180


NORNICKEL_PROFILE = "NORNICKEL_APP_EXACT"
MAGNIT_PROFILE = "MAGNIT_APP_EXACT"


def parse_exact_app_state(payload: str, *, profile: ExactAppStateProfile) -> list[ExactEvent]:
    app = _app_payload(payload)
    events: list[ExactEvent] = []
    for item in _walk_dicts(app):
        timestamp = item.get(profile.timestamp_field)
        title = item.get(profile.title_field)
        href = item.get(profile.url_field)
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            continue
        if not isinstance(title, str) or not isinstance(href, str):
            continue
        published = datetime.fromtimestamp(timestamp, UTC)
        source_local = published + timedelta(minutes=profile.source_utc_offset_minutes)
        if profile.reject_source_local_midnight and source_local.time() == datetime.min.time():
            continue
        identity = item.get(profile.identity_field) if profile.identity_field else None
        source_item_id = str(identity) if isinstance(identity, str) and identity else href
        events.append(
            ExactEvent.create(
                source_code=profile.source_code,
                source_item_id=source_item_id,
                canonical_url=urljoin(profile.base_url, href),
                ticker=profile.ticker,
                issuer=profile.issuer,
                instrument_uid=profile.instrument_uid,
                title=title,
                publication_timestamp_raw=str(timestamp),
                publication_timestamp_utc=published,
                timestamp_source_field=f"embedded App.{profile.timestamp_field} Unix epoch seconds",
            )
        )
    unique = {event.source_item_id: event for event in events}
    return sorted(
        unique.values(), key=lambda item: (item.publication_timestamp_utc, item.source_item_id)
    )


def load_exact_app_state(path: Path, *, profile: ExactAppStateProfile) -> list[ExactEvent]:
    return parse_exact_app_state(path.read_text(encoding="utf-8"), profile=profile)


async def acquire_exact_json_pages(
    *,
    profile: ExactAppStateProfile,
    url_template: str,
    date_from: date,
    date_to: date,
    page_limit: int,
    item_limit: int,
    cache_dir: Path,
    client: httpx.AsyncClient | None = None,
) -> list[ExactEvent]:
    if not 1 <= page_limit <= 50 or not 1 <= item_limit <= 400:
        raise ValueError("official JSON acquisition must be bounded")
    if date_to < date_from:
        raise ValueError("date_to must not be before date_from")
    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    events: list[ExactEvent] = []
    try:
        for page in range(1, page_limit + 1):
            url = url_template.format(page=page)
            payload = await _get_json_cached(
                http_client, url=url, cache_dir=cache_dir, cache_key=str(page)
            )
            parsed = parse_exact_json_items(payload, profile=profile)
            for event in parsed:
                if date_from <= event.publication_date <= date_to:
                    events.append(event)
                    if len(events) >= item_limit:
                        return _unique_events(events)
            nav = payload.get("nav")
            if isinstance(nav, dict):
                typed_nav = cast("dict[str, object]", nav)
                current = typed_nav.get("current")
                total = typed_nav.get("total")
                if isinstance(current, int) and isinstance(total, int) and current >= total:
                    break
    finally:
        if owns_client:
            await http_client.aclose()
    return _unique_events(events)


def parse_exact_json_items(
    payload: dict[str, object], *, profile: ExactAppStateProfile
) -> list[ExactEvent]:
    rows = payload.get("items")
    if not isinstance(rows, list):
        raise ValueError("OFFICIAL_JSON_ITEMS_INVALID")
    wrapped = json.dumps({"items": rows}, ensure_ascii=False)
    return parse_exact_app_state(f"App = {wrapped};", profile=profile)


def _app_payload(payload: str) -> object:
    match = re.search(r"\bApp\s*=\s*", payload)
    if match is None:
        raise ValueError("OFFICIAL_APP_STATE_NOT_FOUND")
    try:
        return cast("object", json.JSONDecoder().raw_decode(payload[match.end() :])[0])
    except json.JSONDecodeError as exc:
        raise ValueError("OFFICIAL_APP_STATE_INVALID") from exc


def _walk_dicts(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        yield mapping
        for child in mapping.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in cast("list[object]", value):
            yield from _walk_dicts(child)


async def _get_json_cached(
    client: httpx.AsyncClient,
    *,
    url: str,
    cache_dir: Path,
    cache_key: str,
) -> dict[str, object]:
    payload_path = cache_dir / f"{cache_key}.json"
    metadata_path = cache_dir / f"{cache_key}.meta.json"
    if payload_path.exists() and metadata_path.exists():
        text = await asyncio.to_thread(payload_path.read_text, encoding="utf-8")
        metadata = json.loads(await asyncio.to_thread(metadata_path.read_text, encoding="utf-8"))
        if isinstance(metadata, dict):
            typed_metadata = cast("dict[str, object]", metadata)
            if (
                typed_metadata.get("url") == url
                and typed_metadata.get("sha256") == hashlib.sha256(text.encode("utf-8")).hexdigest()
            ):
                return _json_object(text)
    response: httpx.Response | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(url, headers={"User-Agent": USER_AGENT})
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError):
            if attempt + 1 >= MAX_RETRIES:
                raise RuntimeError("OFFICIAL_JSON_SOURCE_UNAVAILABLE") from None
            await asyncio.sleep(0.25 * (2**attempt))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            if attempt + 1 >= MAX_RETRIES:
                raise RuntimeError(f"OFFICIAL_JSON_HTTP_{response.status_code}")
            await asyncio.sleep(0.25 * (2**attempt))
            continue
        break
    if response is None or response.status_code != 200:
        status = "NO_RESPONSE" if response is None else str(response.status_code)
        raise RuntimeError(f"OFFICIAL_JSON_HTTP_{status}")
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise RuntimeError("OFFICIAL_JSON_PAYLOAD_TOO_LARGE")
    resolved = response.url
    if resolved.scheme != "https" or resolved.host != "www.magnit.com":
        raise RuntimeError("OFFICIAL_JSON_DOMAIN_REDIRECT_REJECTED")
    text = response.text
    payload = _json_object(text)
    await asyncio.to_thread(
        _write_json_cache,
        cache_dir,
        payload_path,
        metadata_path,
        text,
        url,
        str(resolved),
    )
    return payload


def _json_object(text: str) -> dict[str, object]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("OFFICIAL_JSON_RESPONSE_INVALID")
    return cast("dict[str, object]", value)


def _write_json_cache(
    cache_dir: Path,
    payload_path: Path,
    metadata_path: Path,
    text: str,
    url: str,
    resolved_url: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(text, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "url": url,
                "resolved_url": resolved_url,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _unique_events(events: list[ExactEvent]) -> list[ExactEvent]:
    unique = {event.source_item_id: event for event in events}
    return sorted(
        unique.values(), key=lambda item: (item.publication_timestamp_utc, item.source_item_id)
    )
