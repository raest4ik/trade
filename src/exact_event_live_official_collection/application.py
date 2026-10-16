from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar, cast
from urllib.parse import urljoin
from uuid import UUID

from src.events.domain.v3 import (
    EVENT_ANALYSIS_V3_VERSION,
    EventAnalyzerV3,
    rules_v3_fingerprint,
)
from src.exact_event_corpus.domain import ExactEvent
from src.exact_event_corpus.holdout import is_future_holdout
from src.exact_event_live_official_collection.domain import (
    ARTIFACT_VERSION,
    DEFAULT_SOURCE_REGISTRY_PATH,
    MAX_ITEMS_PER_SOURCE,
    NETWORK_LIMITS,
    PARSER_VERSION,
    RAW_PUBLICATION_SNAPSHOT_VERSION,
    SEMANTIC_MATERIAL_PROVENANCE_VERSION,
    SOURCE_REGISTRY_VERSION,
    LiveExactSource,
    SourceStatus,
    collection_safety_flags,
    parse_publication_timestamp_exact,
    parse_rss_pubdate_exact,
    publication_material,
    publication_material_sha,
    sha256_bytes,
    sha256_payload,
    sha256_text,
)
from src.exact_event_live_official_collection.http_client import (
    BoundedHttpClient,
    FetchResult,
    HttpClient,
)

EXPECTED_AUDIT_ARTIFACT_SHA = "7db63e5e642eee7470c6e51807f66c21a9ddb26bdd3bfce5cda5a82919a74ec6"
EXPECTED_AUDIT_SOURCE_MECHANISM_REGISTRY_SHA = (
    "f287a9b3e5859b97c03ccc40732252b0ff43636beba06a79369ef6bac9ebe16d"
)


@dataclass(frozen=True, slots=True)
class ParsedItem:
    source_item_id: str
    canonical_url: str
    title: str
    description: str
    content: str
    link: str
    guid: str
    raw_payload: dict[str, str | None]
    raw_item: str
    source_format: str
    publication_timestamp_raw: str
    publication_timestamp_utc: datetime


def build_live_official_collection_artifact(
    *,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    input_events_path: Path,
    source_registry_path: Path = Path(DEFAULT_SOURCE_REGISTRY_PATH),
    audit_manifest_path: Path | None = None,
    state_path: Path | None = None,
    client: HttpClient | None = None,
    created_at: datetime | None = None,
    event_origin_filter: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable live official collection artifact output already exists")
    _verify_audit_gate(audit_manifest_path)
    output_root.mkdir(parents=True, exist_ok=False)
    raw_root = output_root / "raw-snapshots"
    raw_root.mkdir()

    sources = read_source_registry(source_registry_path)
    if event_origin_filter is not None:
        allowed_origins = set(event_origin_filter)
        sources = [source for source in sources if source.event_origin in allowed_origins]
    source_registry_payload = [source.payload() for source in sources]
    enabled_sources = [source for source in sources if source.enabled]
    previous_state = _read_state(state_path)
    existing_identities = _existing_identities(input_events_path)
    seen_identities = set(existing_identities) | set(previous_state["seen_source_identities"])
    existing_exact_total = _existing_exact_total(input_events_path)
    now = created_at or datetime.now(UTC)
    http = client or BoundedHttpClient()

    network_records: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    raw_snapshots: list[dict[str, Any]] = []
    raw_publication_snapshots: list[dict[str, Any]] = []
    semantic_material_provenance: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    invalid_items: list[dict[str, Any]] = []
    semantic_extraction_results: list[dict[str, Any]] = []
    blockers: Counter[str] = Counter()

    attempted = success = failed = 0
    items_fetched = timestamp_valid = timestamp_invalid = 0
    items_with_publication_material = items_without_publication_material = 0
    duplicates = new_items = 0
    historical_new = future_new = 0
    semantic_ready = analyzer_unknown = 0
    analyzer = EventAnalyzerV3()
    semantic_computed_at = now

    for source in sources:
        if not source.enabled:
            blockers[SourceStatus.SOURCE_DISABLED.value] += 1
            source_reports.append(_source_report(source, SourceStatus.SOURCE_DISABLED, 0, 0, 0))
            continue
        attempted += 1
        result = http.get(source.source_url)
        network_records.append(_network_record(source, result, now))
        if result.blocker is not None or result.status is None or result.status >= 400:
            status = _status_from_fetch(result)
            blockers[status.value] += 1
            failed += 1
            source_reports.append(_source_report(source, status, 0, 0, 0))
            continue
        raw_path = raw_root / f"{source.source_id}.xml"
        raw_path.write_bytes(result.body)
        raw_snapshots.append(
            {
                "source_id": source.source_id,
                "source_url": source.source_url,
                "raw_snapshot_path": str(raw_path.as_posix()),
                "content_sha256": sha256_bytes(result.body),
                "bytes_received": len(result.body),
                "fetched_at": now.isoformat(),
            }
        )
        try:
            parsed_items = _parse_items(source, result.body)
        except ValueError as exc:
            status = _rss_parse_status(exc)
            blockers[status.value] += 1
            failed += 1
            source_reports.append(_source_report(source, status, 0, 0, 0))
            continue

        source_new = source_duplicates = 0
        for item in parsed_items[:MAX_ITEMS_PER_SOURCE]:
            items_fetched += 1
            timestamp_valid += 1
            material = publication_material(_snapshot_material_view(item))
            if material is None:
                items_without_publication_material += 1
                invalid_items.append(
                    {
                        "source_id": source.source_id,
                        "source_item_id": item.source_item_id,
                        "blocker": SourceStatus.PUBLICATION_MATERIAL_MISSING.value,
                    }
                )
                continue
            items_with_publication_material += 1
            identity = f"{source.source_family}|{item.source_item_id}"
            if identity in seen_identities:
                duplicates += 1
                source_duplicates += 1
                continue
            try:
                event = ExactEvent.create(
                    source_code=source.source_family,
                    source_item_id=item.source_item_id,
                    canonical_url=item.canonical_url,
                    ticker=source.ticker,
                    issuer=source.issuer,
                    instrument_uid=source.instrument_uid,
                    title=item.title or item.description or item.content,
                    publication_timestamp_raw=item.publication_timestamp_raw,
                    publication_timestamp_utc=item.publication_timestamp_utc,
                    timestamp_source_field=source.timestamp_field,
                )
            except ValueError as exc:
                timestamp_invalid += 1
                invalid_items.append(
                    {
                        "source_id": source.source_id,
                        "source_item_id": item.source_item_id,
                        "blocker": str(exc),
                    }
                )
                continue
            seen_identities.add(identity)
            new_items += 1
            source_new += 1
            future = is_future_holdout(event.publication_date)
            if future:
                future_new += 1
            else:
                historical_new += 1
            snapshot = _raw_publication_snapshot(source, item, event, result, now)
            semantic_result = _semantic_extraction_record(
                snapshot=snapshot,
                event=event,
                analyzer=analyzer,
                computed_at=semantic_computed_at,
            )
            features = cast("dict[str, Any]", semantic_result["event_features"])
            semantic_ready += 1
            if features["primary_event_type"] == "UNKNOWN":
                analyzer_unknown += 1
            raw_publication_snapshots.append(snapshot)
            semantic_material_provenance.append(_semantic_material_record(snapshot, event))
            semantic_extraction_results.append(semantic_result)
            event_rows.append(
                _event_metadata_row(source, event, result, now, future, snapshot, features)
            )
        status = SourceStatus.SUCCESS if source_new else SourceStatus.NO_NEW_ITEMS
        success += 1 if status in {SourceStatus.SUCCESS, SourceStatus.NO_NEW_ITEMS} else 0
        blockers[status.value] += 1
        source_reports.append(
            _source_report(source, status, len(parsed_items), source_new, source_duplicates)
        )

    event_rows = sorted(
        event_rows,
        key=lambda row: (
            str(cast("dict[str, Any]", row["metadata"])["publication_timestamp_utc"]),
            str(cast("dict[str, Any]", row["metadata"])["event_id"]),
        ),
    )
    dedupe_state = {
        "dedupe_state_version": "exact-event-live-official-dedupe-state-v1",
        "seen_source_identities": sorted(seen_identities),
        "state_inputs": {
            "existing_event_identity_count": len(existing_identities),
            "previous_state_identity_count": len(previous_state["seen_source_identities"]),
            "new_identity_count": new_items,
        },
    }
    raw_snapshot_payload = sorted(raw_snapshots, key=lambda row: str(row["source_id"]))
    raw_publication_snapshot_payload = sorted(
        raw_publication_snapshots,
        key=lambda row: (str(row["source_id"]), str(row["source_item_id"])),
    )
    semantic_material_payload = sorted(
        semantic_material_provenance,
        key=lambda row: (str(row["source_id"]), str(row["source_item_id"])),
    )
    semantic_extraction_payload = sorted(
        semantic_extraction_results,
        key=lambda row: (str(row["source_id"]), str(row["source_item_id"])),
    )
    network_payload = sorted(network_records, key=lambda row: str(row["source_id"]))
    safety = collection_safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": now.isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "SOURCE_REGISTRY_VERSION": SOURCE_REGISTRY_VERSION,
        "PARSER_VERSION": PARSER_VERSION,
        "SOURCE_REGISTRY_SHA": sha256_payload(source_registry_payload),
        "NETWORK_LIMITS_SHA": sha256_payload(NETWORK_LIMITS),
        "NETWORK_PROVENANCE_SHA": sha256_payload(network_payload),
        "RAW_SNAPSHOT_SHA": sha256_payload(raw_snapshot_payload),
        "RAW_PUBLICATION_SNAPSHOT_SHA": sha256_payload(raw_publication_snapshot_payload),
        "PUBLICATION_MATERIAL_PROVENANCE_SHA": sha256_payload(semantic_material_payload),
        "SEMANTIC_EXTRACTION_RESULTS_SHA": sha256_payload(semantic_extraction_payload),
        "COLLECTED_EVENT_METADATA_SHA": sha256_payload(event_rows),
        "DEDUPE_STATE_SHA": sha256_payload(dedupe_state),
        "LIVE_EXACT_SOURCES_ENABLED": len(enabled_sources),
        "LIVE_EXACT_SOURCES_ATTEMPTED": attempted,
        "LIVE_EXACT_SOURCES_SUCCESS": success,
        "LIVE_EXACT_SOURCES_FAILED": failed,
        "ITEMS_FETCHED": items_fetched,
        "ITEMS_TIMESTAMP_VALID": timestamp_valid,
        "ITEMS_TIMESTAMP_INVALID": timestamp_invalid,
        "ITEMS_WITH_PUBLICATION_MATERIAL": items_with_publication_material,
        "ITEMS_WITHOUT_PUBLICATION_MATERIAL": items_without_publication_material,
        "ITEMS_NEW": new_items,
        "ITEMS_DUPLICATE": duplicates,
        "SNAPSHOTS_WRITTEN": len(raw_publication_snapshot_payload),
        "DUPLICATE_SNAPSHOTS": 0,
        "EXACT_EVENTS_TOTAL_BEFORE": existing_exact_total,
        "EXACT_EVENTS_TOTAL_AFTER": existing_exact_total + new_items,
        "NEW_EXACT_EVENTS": new_items,
        "ITEMS_SEMANTIC_READY": semantic_ready,
        "SEMANTIC_READY_EVENTS": semantic_ready,
        "ANALYZER_UNKNOWN": analyzer_unknown,
        "NEW_HISTORICAL_EXACT_EVENTS": historical_new,
        "NEW_FUTURE_METADATA_ONLY_EVENTS": future_new,
        "FUTURE_METADATA_ONLY_EVENTS_ADDED": future_new,
        "TINVEST_REQUESTS": 0,
        "MARKET_PRICE_LOOKUPS": 0,
        "FUTURE_PRICE_LOOKUPS": 0,
        "FUTURE_REACTIONS_COMPUTED": 0,
        "FUTURE_TARGETS_COMPUTED": 0,
        "FUTURE_OUTCOMES_READ": 0,
        "FUTURE_OUTCOMES_READ_ATTEMPTED": False,
        "WINDOWS_SCHEDULER_CHANGED": False,
        "BACKGROUND_AUTOMATION_ENABLED": False,
        "RULES_V3_FINGERPRINT": rules_v3_fingerprint(),
        "ANALYZER_VERSION": EVENT_ANALYSIS_V3_VERSION,
        "FEATURE_DEFINITION_CHANGED": False,
        "DATE_ONLY_COERCIONS": 0,
        "FETCH_TIME_USED_AS_PUBLICATION_TIME": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "PER_SOURCE_STATUS": source_reports,
        "BLOCKERS_BY_TYPE": dict(sorted(blockers.items())),
        "RECURRING_OPERATION": None,
        "safety": safety,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = _artifact_sha(manifest)
    _write_json(output_root / "manifest.json", manifest)
    _write_jsonl(output_root / "network-provenance.jsonl", network_payload)
    _write_jsonl(output_root / "raw-snapshot-manifest.jsonl", raw_snapshot_payload)
    _write_jsonl(output_root / "raw-publication-snapshots.jsonl", raw_publication_snapshot_payload)
    _write_jsonl(output_root / "semantic-material-provenance.jsonl", semantic_material_payload)
    _write_jsonl(output_root / "semantic-extraction-results.jsonl", semantic_extraction_payload)
    _write_jsonl(output_root / "collected-event-metadata.jsonl", event_rows)
    _write_jsonl(output_root / "invalid-items.jsonl", invalid_items)
    _write_json(output_root / "dedupe-state.json", dedupe_state)
    _write_jsonl(output_root / "source-registry.jsonl", source_registry_payload)
    _write_report(output_root / "report.md", manifest)
    return manifest


def read_source_registry(path: Path) -> list[LiveExactSource]:
    payload = _read_json(path)
    if payload.get("source_registry_version") != SOURCE_REGISTRY_VERSION:
        raise ValueError("SOURCE_REGISTRY_VERSION_MISMATCH")
    rows = payload.get("sources")
    if not isinstance(rows, list):
        raise ValueError("SOURCE_REGISTRY_SOURCES_MISSING")
    raw_rows = cast("list[object]", rows)
    return [LiveExactSource.from_payload(cast("dict[str, Any]", row)) for row in raw_rows]


def _parse_items(source: LiveExactSource, body: bytes) -> list[ParsedItem]:
    if source.mechanism_type == "JSON":
        return _parse_json_items(source, body)
    if source.mechanism_type == "HTML":
        return _parse_html_items(source, body)
    return _parse_xml_feed_items(source, body)


def _parse_xml_feed_items(source: LiveExactSource, body: bytes) -> list[ParsedItem]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ValueError("INVALID_RSS") from exc
    root_name = _local(root.tag)
    if root_name not in {"rss", "feed"}:
        raise ValueError("INVALID_RSS")
    if source.mechanism_type == "RSS" and root_name != "rss":
        raise ValueError("INVALID_RSS")
    if source.mechanism_type == "ATOM" and root_name != "feed":
        raise ValueError("INVALID_RSS")
    item_names = {"item"} if root_name == "rss" else {"entry"}
    items: list[ParsedItem] = []
    for item in (element for element in root.iter() if _local(element.tag) in item_names):
        title = _text(item, "title")
        description = _text(item, "description")
        content = _text(item, "encoded") or _text(item, "content") or _text(item, "summary")
        link = _text(item, "link")
        if not link:
            link = _atom_link(item) or ""
        guid = _text(item, "guid")
        pubdate = _text(item, "pubDate") if root_name == "rss" else _text(item, "published")
        if source.item_match_any and not _item_matches(item, source.item_match_any):
            continue
        if not pubdate:
            raise ValueError(SourceStatus.MISSING_EXACT_TIMESTAMP.value)
        try:
            published = (
                parse_rss_pubdate_exact(pubdate)
                if root_name == "rss"
                else parse_publication_timestamp_exact(pubdate, field_name="Atom entry published")
            )
        except ValueError as exc:
            status = (
                SourceStatus.INVALID_TIMEZONE.value
                if str(exc) == SourceStatus.INVALID_TIMEZONE.value
                else SourceStatus.MISSING_EXACT_TIMESTAMP.value
            )
            raise ValueError(status) from exc
        source_item_id = guid or _text(item, "id") or link
        if not source_item_id:
            raise ValueError("RSS_ITEM_ID_MISSING")
        if source.item_match_any:
            source_item_id = f"{source.ticker}:{source_item_id}"
        canonical_url = link or urljoin(source.source_url, f"#{source_item_id}")
        items.append(
            ParsedItem(
                source_item_id=source_item_id,
                canonical_url=canonical_url,
                title=title,
                description=description,
                content=content,
                link=link,
                guid=guid,
                raw_payload={
                    "title": title or None,
                    "description": description or None,
                    "content": content or None,
                    "pubDate": pubdate or None,
                    "link": link or None,
                    "guid": guid or None,
                    "id": _text(item, "id") or None,
                },
                raw_item=ET.tostring(item, encoding="unicode"),
                source_format="RSS_ITEM" if root_name == "rss" else "ATOM_ENTRY",
                publication_timestamp_raw=pubdate,
                publication_timestamp_utc=published,
            )
        )
    return items


def _parse_json_items(source: LiveExactSource, body: bytes) -> list[ParsedItem]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("INVALID_RSS") from exc
    rows = _json_candidate_rows(payload)
    items: list[ParsedItem] = []
    for row in rows[:MAX_ITEMS_PER_SOURCE]:
        title = _clean_text(str(row.get("title") or row.get("name") or row.get("headline") or ""))
        description = _clean_text(str(row.get("description") or row.get("summary") or ""))
        content = _clean_text(str(row.get("content") or row.get("body") or row.get("text") or ""))
        timestamp = row.get(_json_timestamp_key(source.timestamp_field))
        if not isinstance(timestamp, str):
            raise ValueError(SourceStatus.MISSING_EXACT_TIMESTAMP.value)
        try:
            published = parse_publication_timestamp_exact(
                timestamp, field_name=source.timestamp_field
            )
        except ValueError as exc:
            status = (
                SourceStatus.INVALID_TIMEZONE.value
                if str(exc) == SourceStatus.INVALID_TIMEZONE.value
                else SourceStatus.MISSING_EXACT_TIMESTAMP.value
            )
            raise ValueError(status) from exc
        link = str(row.get("url") or row.get("link") or source.source_url)
        source_item_id = str(row.get("id") or row.get("guid") or link)
        if source.item_match_any and not _contains_any_item_token(
            " ".join((title, description, content)), source.item_match_any
        ):
            continue
        items.append(
            ParsedItem(
                source_item_id=source_item_id,
                canonical_url=urljoin(source.source_url, link),
                title=title,
                description=description,
                content=content,
                link=link,
                guid=str(row.get("guid") or row.get("id") or ""),
                raw_payload={
                    "title": title or None,
                    "description": description or None,
                    "content": content or None,
                    source.timestamp_field: timestamp,
                    "link": link or None,
                    "guid": str(row.get("guid") or "") or None,
                    "id": str(row.get("id") or "") or None,
                },
                raw_item=json.dumps(row, ensure_ascii=False, sort_keys=True),
                source_format="JSON_ITEM",
                publication_timestamp_raw=timestamp,
                publication_timestamp_utc=published,
            )
        )
    return items


def _parse_html_items(source: LiveExactSource, body: bytes) -> list[ParsedItem]:
    content = body.decode("utf-8", errors="replace")
    parser = _HtmlMetadataParser()
    parser.feed(content)
    timestamp, field = parser.publication_timestamp(source.timestamp_field)
    if not timestamp or not field:
        if parser.has_clock_without_publication_timezone():
            raise ValueError(SourceStatus.INVALID_TIMEZONE.value)
        raise ValueError(SourceStatus.MISSING_EXACT_TIMESTAMP.value)
    try:
        published = parse_publication_timestamp_exact(timestamp, field_name=field)
    except ValueError as exc:
        status = (
            SourceStatus.INVALID_TIMEZONE.value
            if str(exc) == SourceStatus.INVALID_TIMEZONE.value
            else SourceStatus.MISSING_EXACT_TIMESTAMP.value
        )
        raise ValueError(status) from exc
    title = parser.title() or _html_title(content) or source.issuer
    material = parser.text()
    if source.item_match_any and not _contains_any_item_token(
        " ".join((title, material)), source.item_match_any
    ):
        return []
    link = parser.canonical_url() or source.source_url
    return [
        ParsedItem(
            source_item_id=link,
            canonical_url=link,
            title=title,
            description=material,
            content="",
            link=link,
            guid=link,
            raw_payload={
                "title": title or None,
                "description": material or None,
                "content": None,
                field: timestamp,
                "link": link,
                "guid": link,
            },
            raw_item=content,
            source_format="HTML_PUBLICATION",
            publication_timestamp_raw=timestamp,
            publication_timestamp_utc=published,
        )
    ]


def _event_metadata_row(
    source: LiveExactSource,
    event: ExactEvent,
    result: FetchResult,
    fetched_at: datetime,
    future_metadata_only: bool,
    snapshot: dict[str, Any],
    event_features: dict[str, Any],
) -> dict[str, Any]:
    metadata = event.metadata_payload()
    metadata["future_holdout"] = future_metadata_only
    metadata["future_holdout_metadata_only"] = future_metadata_only
    metadata["raw_source_sha256"] = sha256_bytes(result.body)
    metadata["raw_source_fetched_at"] = fetched_at.isoformat()
    metadata["source_id"] = source.source_id
    metadata["source_family"] = source.source_family
    metadata["parser_version"] = source.parser_version
    metadata["source_registry_version"] = source.source_registry_version
    metadata["source_registry_hash"] = sha256_payload(source.payload())
    metadata["publication_snapshot_id"] = snapshot["snapshot_id"]
    metadata["publication_material_available"] = snapshot["publication_material_available"]
    metadata["publication_material_sha"] = snapshot["publication_material_sha"]
    metadata["event_origin"] = _event_origin(source)
    return {
        "metadata": metadata,
        "event_features": event_features,
        "pre_event_market_features": None,
        "quality": {
            "metadata_only": True,
            "outcome_read_guard": "FUTURE_HOLDOUT_METADATA_ONLY"
            if future_metadata_only
            else "HISTORICAL_METADATA_ONLY",
            "source_timestamp_provided": True,
            "explicit_timezone_required": True,
            "date_only_coerced": False,
        },
        "target_availability": {
            "feature_ready": False,
            "reaction_ready": False,
            "research_outcomes_visible": False,
            "status": "FUTURE_METADATA_ONLY" if future_metadata_only else "METADATA_ONLY",
        },
    }


def _snapshot_material_view(item: ParsedItem) -> dict[str, str]:
    return {
        "title": item.title,
        "description": item.description,
        "content": item.content,
    }


def _raw_publication_snapshot(
    source: LiveExactSource,
    item: ParsedItem,
    event: ExactEvent,
    result: FetchResult,
    fetched_at: datetime,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "snapshot_version": RAW_PUBLICATION_SNAPSHOT_VERSION,
        "snapshot_id": _snapshot_id(source, item.source_item_id),
        "event_id": str(event.event_id),
        "source_id": source.source_id,
        "source_family": source.source_family,
        "source_url": source.source_url,
        "source_item_id": item.source_item_id,
        "ticker": source.ticker,
        "issuer": source.issuer,
        "event_origin": _event_origin(source),
        "fetched_at_utc": fetched_at.isoformat(),
        "publication_timestamp_utc": item.publication_timestamp_utc.isoformat(),
        "publication_timestamp_quality": "EXACT",
        "publication_timestamp_raw": item.publication_timestamp_raw,
        "title": item.title or None,
        "description": item.description or None,
        "content": item.content or None,
        "raw_payload": item.raw_payload,
        "content_type": result.content_type,
        "source_format": item.source_format,
        "link": item.link or None,
        "guid": item.guid or None,
        "canonical_url": item.canonical_url,
        "raw_content_hash": sha256_text(item.raw_item),
        "normalized_content_hash": sha256_payload(item.raw_payload),
        "collection_version": ARTIFACT_VERSION,
        "parser_version": source.parser_version,
    }
    snapshot["publication_material_available"] = publication_material(snapshot) is not None
    snapshot["publication_material_sha"] = publication_material_sha(snapshot)
    return snapshot


def _semantic_material_record(snapshot: dict[str, Any], event: ExactEvent) -> dict[str, Any]:
    return {
        "provenance_version": SEMANTIC_MATERIAL_PROVENANCE_VERSION,
        "event_id": str(event.event_id),
        "snapshot_id": snapshot["snapshot_id"],
        "source_id": snapshot["source_id"],
        "source_family": snapshot["source_family"],
        "source_item_id": snapshot["source_item_id"],
        "publication_material_available": snapshot["publication_material_available"],
        "publication_material_sha": snapshot["publication_material_sha"],
        "publication_material_fields": [
            key for key in ("title", "description", "content") if snapshot.get(key)
        ],
        "semantic_replay_builder": "src.events.domain.v3:EventAnalyzerV3",
        "rules_v3_changed": False,
        "qwen_changed": False,
        "uses_market_data": False,
        "uses_reaction_data": False,
        "uses_target_data": False,
    }


def _semantic_extraction_record(
    *,
    snapshot: dict[str, Any],
    event: ExactEvent,
    analyzer: EventAnalyzerV3,
    computed_at: datetime,
) -> dict[str, Any]:
    material = publication_material(snapshot)
    if material is None:
        raise ValueError("PUBLICATION_MATERIAL_MISSING")
    analysis = analyzer.analyze(news_id=UUID(str(event.event_id)), raw_content=material)
    event_features = {
        "primary_event_type": analysis.primary_event_type.value,
        "event_count": len(analysis.events),
        "fact_count": len(analysis.financial_facts),
    }
    return {
        "provenance_version": "semantic-extraction-result-v1",
        "event_id": str(event.event_id),
        "snapshot_id": snapshot["snapshot_id"],
        "source_id": snapshot["source_id"],
        "source_family": snapshot["source_family"],
        "source_item_id": snapshot["source_item_id"],
        "publication_material_sha": snapshot["publication_material_sha"],
        "semantic_features_sha": sha256_payload(event_features),
        "rules_v3_fingerprint": rules_v3_fingerprint(),
        "analyzer_version": analysis.analysis_version,
        "semantic_computed_at": computed_at.isoformat(),
        "publication_timestamp_utc": snapshot["publication_timestamp_utc"],
        "snapshot_sha": sha256_payload(snapshot),
        "semantic_input_fields": [
            key for key in ("title", "description", "content") if snapshot.get(key)
        ],
        "event_features": event_features,
        "uses_market_data": False,
        "uses_reaction_data": False,
        "uses_target_data": False,
    }


def _snapshot_id(source: LiveExactSource, source_item_id: str) -> str:
    return sha256_payload(
        {
            "snapshot_version": RAW_PUBLICATION_SNAPSHOT_VERSION,
            "source_family": source.source_family,
            "source_item_id": source_item_id,
        }
    )


def _event_origin(source: LiveExactSource) -> str:
    return source.event_origin


def _network_record(
    source: LiveExactSource,
    result: FetchResult,
    fetched_at: datetime,
) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "ticker": source.ticker,
        "issuer": source.issuer,
        "official_domain": source.official_domain,
        "source_url": source.source_url,
        "source_type": source.mechanism_type,
        "event_origin": source.event_origin,
        "ticker_attribution_method": source.ticker_attribution_method,
        "fetch_timestamp": fetched_at.isoformat(),
        "http_status": result.status,
        "redirects": result.redirects,
        "redirect_chain": list(result.redirect_chain),
        "final_url": result.final_url,
        "response_sha256": sha256_bytes(result.body) if result.body else None,
        "bytes_received": len(result.body),
        "parser_version": source.parser_version,
        "source_registry_version": source.source_registry_version,
        "source_registry_hash": sha256_payload(source.payload()),
        "blocker": result.blocker,
    }


def _source_report(
    source: LiveExactSource,
    status: SourceStatus,
    items_fetched: int,
    items_new: int,
    items_duplicate: int,
) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "ticker": source.ticker,
        "issuer": source.issuer,
        "source_url": source.source_url,
        "source_family": source.source_family,
        "event_origin": source.event_origin,
        "ticker_attribution_method": source.ticker_attribution_method,
        "status": status.value,
        "blocker": None
        if status in {SourceStatus.SUCCESS, SourceStatus.NO_NEW_ITEMS}
        else status.value,
        "items_fetched": items_fetched,
        "items_new": items_new,
        "items_duplicate": items_duplicate,
    }


def _status_from_fetch(result: FetchResult) -> SourceStatus:
    if result.blocker in {item.value for item in SourceStatus}:
        return SourceStatus(str(result.blocker))
    if result.status == 429:
        return SourceStatus.RATE_LIMITED
    if result.status == 403:
        return SourceStatus.POLICY_BLOCKED
    if result.status is not None and result.status >= 400:
        return SourceStatus.HTTP_FAILURE
    return SourceStatus.TECHNICAL_FAILURE


def _rss_parse_status(exc: ValueError) -> SourceStatus:
    value = str(exc)
    if value == SourceStatus.MISSING_EXACT_TIMESTAMP.value:
        return SourceStatus.MISSING_EXACT_TIMESTAMP
    if value == SourceStatus.INVALID_TIMEZONE.value:
        return SourceStatus.INVALID_TIMEZONE
    return SourceStatus.INVALID_RSS


def _existing_identities(path: Path) -> set[str]:
    if not path.exists():
        return set()
    identities: set[str] = set()
    for row in _read_jsonl(path):
        metadata = cast("dict[str, Any]", row.get("metadata") or {})
        source_code = metadata.get("source_code")
        source_item_id = metadata.get("source_item_id")
        if source_code and source_item_id:
            identities.add(f"{source_code}|{source_item_id}")
    return identities


def _existing_exact_total(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _row in _read_jsonl(path))


def _read_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"seen_source_identities": []}
    payload = _read_json(path)
    raw_identities: object = payload.get("seen_source_identities") or []
    if not isinstance(raw_identities, list):
        raise ValueError("INVALID_DEDUPE_STATE")
    identities = cast("list[object]", raw_identities)
    return {"seen_source_identities": [str(item) for item in identities]}


def _verify_audit_gate(path: Path | None) -> None:
    if path is None:
        return
    payload = _read_json(path)
    if payload.get("ARTIFACT_SHA") != EXPECTED_AUDIT_ARTIFACT_SHA:
        raise ValueError("INPUT_AUDIT_ARTIFACT_SHA_MISMATCH")
    if payload.get("SOURCE_MECHANISM_REGISTRY_SHA") != EXPECTED_AUDIT_SOURCE_MECHANISM_REGISTRY_SHA:
        raise ValueError("INPUT_AUDIT_SOURCE_MECHANISM_REGISTRY_SHA_MISMATCH")
    if payload.get("EXACT_LIVE_ONLY_COUNT") != 1 or payload.get("NEW_EXACT_CAPABLE_SOURCES") != 1:
        raise ValueError("INPUT_AUDIT_EXACT_LIVE_ONLY_MISMATCH")
    if payload.get("DECISION") != "LIVE_COLLECTION_IS_MAIN_PATH":
        raise ValueError("INPUT_AUDIT_DECISION_MISMATCH")


def _artifact_sha(manifest: dict[str, Any]) -> str:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"ARTIFACT_SHA", "created_at", "git_sha"}
    }
    return sha256_payload(core)


def _text(item: ET.Element, tag: str) -> str:
    value = next(
        (child.text for child in item if _local(child.tag) == tag and child.text is not None),
        None,
    )
    return " ".join(value.split()) if value else ""


def _item_matches(item: ET.Element, tokens: tuple[str, ...]) -> bool:
    haystack = " ".join(
        value
        for value in (
            _text(item, "title"),
            _text(item, "description"),
            _text(item, "encoded"),
        )
        if value
    )
    return _contains_any_item_token(haystack, tokens)


def _contains_any_item_token(haystack: str, tokens: tuple[str, ...]) -> bool:
    return any(_contains_atomic_token(haystack, token) for token in tokens)


def _contains_atomic_token(haystack: str, token: str) -> bool:
    escaped = re.escape(token.strip())
    if not escaped:
        return False
    return re.search(rf"(?<![A-Z0-9]){escaped}(?![A-Z0-9])", haystack, re.IGNORECASE) is not None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _atom_link(node: ET.Element) -> str | None:
    for child in list(node):
        if _local(child.tag) == "link" and child.attrib.get("href"):
            return str(child.attrib["href"])
    return None


def _json_candidate_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        payload_rows = cast("list[Any]", payload)
        return [cast("dict[str, Any]", row) for row in payload_rows if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    data = cast("dict[str, Any]", payload)
    for key in ("items", "data", "news", "results", "publications"):
        value = data.get(key)
        if isinstance(value, list):
            value_rows = cast("list[Any]", value)
            return [cast("dict[str, Any]", row) for row in value_rows if isinstance(row, dict)]
    return [data]


def _json_timestamp_key(timestamp_field: str) -> str:
    return {
        "JSON published_at": "published_at",
        "JSON publication_date": "publication_date",
    }.get(timestamp_field, timestamp_field.removeprefix("JSON "))


def _clean_text(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def _html_title(content: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return _clean_text(match.group(1))


class _HtmlMetadataParser(HTMLParser):
    _publication_fields: ClassVar[dict[str, str]] = {
        "datepublished": "HTML datePublished",
        "article:published_time": "HTML article:published_time",
        "pubdate": "HTML pubdate",
    }
    _modification_fields: ClassVar[set[str]] = {
        "datemodified",
        "article:modified_time",
        "dateupdated",
        "updated",
    }

    def __init__(self) -> None:
        super().__init__()
        self._publication_values: list[tuple[str, str]] = []
        self._modification_values: list[tuple[str, str]] = []
        self._links: list[str] = []
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._in_title = False
        self._in_script = False
        self._jsonld_parts: list[str] = []
        self._in_jsonld = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = {key.lower(): value for key, value in attrs if value is not None}
        lower = tag.lower()
        if lower in {"title", "h1"}:
            self._in_title = True
        if lower == "script":
            self._in_script = True
        if lower == "script" and normalized.get("type", "").lower() == "application/ld+json":
            self._in_jsonld = True
            self._jsonld_parts = []
        if lower == "link" and normalized.get("rel", "").lower() == "canonical":
            href = normalized.get("href")
            if href:
                self._links.append(str(href))
        if lower != "meta":
            return
        marker = (
            normalized.get("property") or normalized.get("name") or normalized.get("itemprop") or ""
        ).lower()
        content = normalized.get("content")
        if not content:
            return
        if marker in self._publication_fields:
            self._publication_values.append((self._publication_fields[marker], content.strip()))
        if marker in self._modification_fields:
            self._modification_values.append((marker, content.strip()))

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in {"title", "h1"}:
            self._in_title = False
        if lower == "script" and self._in_jsonld:
            self._in_jsonld = False
            self._consume_jsonld("".join(self._jsonld_parts))
        if lower == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_jsonld:
            self._jsonld_parts.append(data)
            return
        if self._in_script:
            return
        self._parts.append(text)
        if self._in_title:
            self._title_parts.append(text)

    def publication_timestamp(self, expected_field: str) -> tuple[str | None, str | None]:
        matching = [
            (field, value) for field, value in self._publication_values if field == expected_field
        ]
        if not matching:
            return None, None
        field, value = matching[0]
        return value, field

    def has_clock_without_publication_timezone(self) -> bool:
        text = self.text()
        return bool(
            self._modification_values
            or re.search(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?\b", text)
            or re.search(r"\b\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}\b", text)
            or re.search(r"\b\d{1,2}:\d{2}\b", text)
        )

    def canonical_url(self) -> str | None:
        return self._links[0] if self._links else None

    def text(self) -> str:
        return _clean_text(" ".join(self._parts))

    def title(self) -> str | None:
        title = _clean_text(" ".join(self._title_parts))
        return title or None

    def _consume_jsonld(self, value: str) -> None:
        try:
            payload = json.loads(html.unescape(value).strip())
        except json.JSONDecodeError:
            return
        for field, raw in _jsonld_dates(payload):
            marker = field.lower()
            if marker == "datepublished":
                self._publication_values.append(("HTML datePublished", raw))
            elif marker in self._modification_fields:
                self._modification_values.append((field, raw))


def _jsonld_dates(payload: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(payload, list):
        for item in cast("list[Any]", payload):
            rows.extend(_jsonld_dates(item))
        return rows
    if not isinstance(payload, dict):
        return rows
    typed = cast("dict[str, Any]", payload)
    for key in ("datePublished", "dateModified", "dateUpdated", "published"):
        value = typed.get(key)
        if isinstance(value, str):
            rows.append((key, value))
    graph = typed.get("@graph")
    if isinstance(graph, list):
        for item in cast("list[Any]", graph):
            rows.extend(_jsonld_dates(item))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        f"ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        f"SOURCE_REGISTRY_SHA={manifest['SOURCE_REGISTRY_SHA']}",
        f"NETWORK_PROVENANCE_SHA={manifest['NETWORK_PROVENANCE_SHA']}",
        f"RAW_SNAPSHOT_SHA={manifest['RAW_SNAPSHOT_SHA']}",
        f"RAW_PUBLICATION_SNAPSHOT_SHA={manifest['RAW_PUBLICATION_SNAPSHOT_SHA']}",
        f"PUBLICATION_MATERIAL_PROVENANCE_SHA={manifest['PUBLICATION_MATERIAL_PROVENANCE_SHA']}",
        f"COLLECTED_EVENT_METADATA_SHA={manifest['COLLECTED_EVENT_METADATA_SHA']}",
        f"DEDUPE_STATE_SHA={manifest['DEDUPE_STATE_SHA']}",
        "",
        f"LIVE_EXACT_SOURCES_ENABLED={manifest['LIVE_EXACT_SOURCES_ENABLED']}",
        f"LIVE_EXACT_SOURCES_ATTEMPTED={manifest['LIVE_EXACT_SOURCES_ATTEMPTED']}",
        f"LIVE_EXACT_SOURCES_SUCCESS={manifest['LIVE_EXACT_SOURCES_SUCCESS']}",
        f"LIVE_EXACT_SOURCES_FAILED={manifest['LIVE_EXACT_SOURCES_FAILED']}",
        f"ITEMS_FETCHED={manifest['ITEMS_FETCHED']}",
        f"ITEMS_TIMESTAMP_VALID={manifest['ITEMS_TIMESTAMP_VALID']}",
        f"ITEMS_TIMESTAMP_INVALID={manifest['ITEMS_TIMESTAMP_INVALID']}",
        f"ITEMS_WITH_PUBLICATION_MATERIAL={manifest['ITEMS_WITH_PUBLICATION_MATERIAL']}",
        f"ITEMS_WITHOUT_PUBLICATION_MATERIAL={manifest['ITEMS_WITHOUT_PUBLICATION_MATERIAL']}",
        f"ITEMS_NEW={manifest['ITEMS_NEW']}",
        f"ITEMS_DUPLICATE={manifest['ITEMS_DUPLICATE']}",
        f"SNAPSHOTS_WRITTEN={manifest['SNAPSHOTS_WRITTEN']}",
        f"DUPLICATE_SNAPSHOTS={manifest['DUPLICATE_SNAPSHOTS']}",
        "",
        f"EXACT_EVENTS_TOTAL_BEFORE={manifest['EXACT_EVENTS_TOTAL_BEFORE']}",
        f"EXACT_EVENTS_TOTAL_AFTER={manifest['EXACT_EVENTS_TOTAL_AFTER']}",
        f"NEW_EXACT_EVENTS={manifest['NEW_EXACT_EVENTS']}",
        f"NEW_HISTORICAL_EXACT_EVENTS={manifest['NEW_HISTORICAL_EXACT_EVENTS']}",
        f"NEW_FUTURE_METADATA_ONLY_EVENTS={manifest['NEW_FUTURE_METADATA_ONLY_EVENTS']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
