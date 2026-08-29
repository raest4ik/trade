from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin

from src.exact_event_corpus.domain import ExactEvent
from src.exact_event_corpus.holdout import is_future_holdout
from src.exact_event_live_official_collection.domain import (
    ARTIFACT_VERSION,
    DEFAULT_SOURCE_REGISTRY_PATH,
    MAX_ITEMS_PER_SOURCE,
    NETWORK_LIMITS,
    PARSER_VERSION,
    SOURCE_REGISTRY_VERSION,
    LiveExactSource,
    SourceStatus,
    collection_safety_flags,
    parse_rss_pubdate_exact,
    sha256_bytes,
    sha256_payload,
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
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable live official collection artifact output already exists")
    _verify_audit_gate(audit_manifest_path)
    output_root.mkdir(parents=True, exist_ok=False)
    raw_root = output_root / "raw-snapshots"
    raw_root.mkdir()

    sources = read_source_registry(source_registry_path)
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
    event_rows: list[dict[str, Any]] = []
    invalid_items: list[dict[str, Any]] = []
    blockers: Counter[str] = Counter()

    attempted = success = failed = 0
    items_fetched = timestamp_valid = timestamp_invalid = 0
    duplicates = new_items = 0
    historical_new = future_new = 0

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
            parsed_items = _parse_rss_items(source, result.body)
        except ValueError as exc:
            status = _rss_parse_status(exc)
            blockers[status.value] += 1
            failed += 1
            source_reports.append(_source_report(source, status, 0, 0, 0))
            continue

        source_new = source_duplicates = 0
        for item in parsed_items[:MAX_ITEMS_PER_SOURCE]:
            items_fetched += 1
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
                    title=item.title,
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
            timestamp_valid += 1
            seen_identities.add(identity)
            new_items += 1
            source_new += 1
            future = is_future_holdout(event.publication_date)
            if future:
                future_new += 1
            else:
                historical_new += 1
            event_rows.append(_event_metadata_row(source, event, result, now, future))
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
        "COLLECTED_EVENT_METADATA_SHA": sha256_payload(event_rows),
        "DEDUPE_STATE_SHA": sha256_payload(dedupe_state),
        "LIVE_EXACT_SOURCES_ENABLED": len(enabled_sources),
        "LIVE_EXACT_SOURCES_ATTEMPTED": attempted,
        "LIVE_EXACT_SOURCES_SUCCESS": success,
        "LIVE_EXACT_SOURCES_FAILED": failed,
        "ITEMS_FETCHED": items_fetched,
        "ITEMS_TIMESTAMP_VALID": timestamp_valid,
        "ITEMS_TIMESTAMP_INVALID": timestamp_invalid,
        "ITEMS_NEW": new_items,
        "ITEMS_DUPLICATE": duplicates,
        "EXACT_EVENTS_TOTAL_BEFORE": existing_exact_total,
        "EXACT_EVENTS_TOTAL_AFTER": existing_exact_total + new_items,
        "NEW_EXACT_EVENTS": new_items,
        "NEW_HISTORICAL_EXACT_EVENTS": historical_new,
        "NEW_FUTURE_METADATA_ONLY_EVENTS": future_new,
        "FUTURE_METADATA_ONLY_EVENTS_ADDED": future_new,
        "DATE_ONLY_COERCIONS": 0,
        "FETCH_TIME_USED_AS_PUBLICATION_TIME": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "PER_SOURCE_STATUS": source_reports,
        "BLOCKERS_BY_TYPE": dict(sorted(blockers.items())),
        "RECURRING_OPERATION": {
            "cadence_minutes": 60,
            "scheduler_command": (
                "uv run python -m apps.cli.acquire_exact_event_live_official "
                "--base-main-sha <BASE_MAIN_SHA> --output-dir <RUN_ARTIFACT_DIR>"
            ),
            "scheduler_policy": "environment-specific scheduler may invoke one-shot CLI",
        },
        "safety": safety,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = _artifact_sha(manifest)
    _write_json(output_root / "manifest.json", manifest)
    _write_jsonl(output_root / "network-provenance.jsonl", network_payload)
    _write_jsonl(output_root / "raw-snapshot-manifest.jsonl", raw_snapshot_payload)
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


def _parse_rss_items(source: LiveExactSource, body: bytes) -> list[ParsedItem]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ValueError("INVALID_RSS") from exc
    if _local(root.tag) != "rss":
        raise ValueError("INVALID_RSS")
    items: list[ParsedItem] = []
    for item in (element for element in root.iter() if _local(element.tag) == "item"):
        title = _text(item, "title")
        link = _text(item, "link")
        guid = _text(item, "guid")
        pubdate = _text(item, "pubDate")
        if source.item_match_any and not _item_matches(item, source.item_match_any):
            continue
        if not pubdate:
            raise ValueError(SourceStatus.MISSING_EXACT_TIMESTAMP.value)
        try:
            published = parse_rss_pubdate_exact(pubdate)
        except ValueError as exc:
            status = (
                SourceStatus.INVALID_TIMEZONE.value
                if str(exc) == SourceStatus.INVALID_TIMEZONE.value
                else SourceStatus.MISSING_EXACT_TIMESTAMP.value
            )
            raise ValueError(status) from exc
        source_item_id = guid or link
        if not source_item_id:
            raise ValueError("RSS_ITEM_ID_MISSING")
        if source.item_match_any:
            source_item_id = f"{source.ticker}:{source_item_id}"
        canonical_url = link or urljoin(source.source_url, f"#{source_item_id}")
        items.append(
            ParsedItem(
                source_item_id=source_item_id,
                canonical_url=canonical_url,
                title=title or "Official RSS item",
                publication_timestamp_raw=pubdate,
                publication_timestamp_utc=published,
            )
        )
    return items


def _event_metadata_row(
    source: LiveExactSource,
    event: ExactEvent,
    result: FetchResult,
    fetched_at: datetime,
    future_metadata_only: bool,
) -> dict[str, Any]:
    metadata = event.metadata_payload()
    metadata["future_holdout"] = future_metadata_only
    metadata["future_holdout_metadata_only"] = future_metadata_only
    metadata["raw_source_sha256"] = sha256_bytes(result.body)
    metadata["raw_source_fetched_at"] = fetched_at.isoformat()
    metadata["source_id"] = source.source_id
    metadata["parser_version"] = source.parser_version
    metadata["source_registry_version"] = source.source_registry_version
    metadata["source_registry_hash"] = sha256_payload(source.payload())
    return {
        "metadata": metadata,
        "event_features": None,
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
            _text(item, "link"),
            _text(item, "guid"),
            _text(item, "description"),
            _text(item, "encoded"),
        )
        if value
    )
    return any(_contains_atomic_token(haystack, token) for token in tokens)


def _contains_atomic_token(haystack: str, token: str) -> bool:
    escaped = re.escape(token.strip())
    if not escaped:
        return False
    return re.search(rf"(?<![A-Z0-9]){escaped}(?![A-Z0-9])", haystack, re.IGNORECASE) is not None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


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
        f"ITEMS_NEW={manifest['ITEMS_NEW']}",
        f"ITEMS_DUPLICATE={manifest['ITEMS_DUPLICATE']}",
        "",
        f"EXACT_EVENTS_TOTAL_BEFORE={manifest['EXACT_EVENTS_TOTAL_BEFORE']}",
        f"EXACT_EVENTS_TOTAL_AFTER={manifest['EXACT_EVENTS_TOTAL_AFTER']}",
        f"NEW_EXACT_EVENTS={manifest['NEW_EXACT_EVENTS']}",
        f"NEW_HISTORICAL_EXACT_EVENTS={manifest['NEW_HISTORICAL_EXACT_EVENTS']}",
        f"NEW_FUTURE_METADATA_ONLY_EVENTS={manifest['NEW_FUTURE_METADATA_ONLY_EVENTS']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
