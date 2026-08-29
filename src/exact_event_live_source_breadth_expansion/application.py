from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from src.exact_event_live_official_collection.application import (
    build_live_official_collection_artifact,
)
from src.exact_event_live_official_collection.domain import (
    SOURCE_REGISTRY_VERSION,
    LiveExactSource,
    parse_rss_pubdate_exact,
)
from src.exact_event_live_official_collection.http_client import BoundedHttpClient, HttpClient
from src.exact_event_live_source_breadth_expansion.domain import (
    ARTIFACT_VERSION,
    DEFAULT_ELIGIBILITY_MANIFEST_PATH,
    DEFAULT_INPUT_EVENTS_PATH,
    DEFAULT_LIVE_REGISTRY_PATH,
    DEFAULT_UNIVERSE_PATH,
    DISCOVERY_LIMITS,
    MAX_ITEMS_PER_FEED_DISCOVERY,
    MAX_NEW_LIVE_SOURCES,
    MAX_RESPONSE_BYTES,
    MAX_TARGET_TICKERS,
    MOEX_RISK_PARAMETERS_RSS_URL,
    MOEX_RISK_PARAMETERS_SOURCE_FAMILY,
    CandidateStatus,
    sha256_payload,
    source_breadth_safety_flags,
)


@dataclass(frozen=True, slots=True)
class RssEvidenceItem:
    title: str
    link: str
    guid: str
    pubdate: str
    published_at_utc: datetime
    matched_tickers: tuple[str, ...]


def build_live_source_breadth_expansion_artifact(
    *,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    universe_path: Path = Path(DEFAULT_UNIVERSE_PATH),
    input_events_path: Path = Path(DEFAULT_INPUT_EVENTS_PATH),
    eligibility_manifest_path: Path = Path(DEFAULT_ELIGIBILITY_MANIFEST_PATH),
    live_registry_path: Path = Path(DEFAULT_LIVE_REGISTRY_PATH),
    client: HttpClient | None = None,
    created_at: datetime | None = None,
    write_registry: bool = True,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable live source breadth artifact output already exists")
    now = created_at or datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=False)

    eligibility_manifest = _read_json(eligibility_manifest_path)
    _require_tradability_gate(eligibility_manifest)
    universe_rows = _read_universe(universe_path)
    events_before = _read_jsonl(input_events_path)
    registry_before = _read_live_registry_payload(live_registry_path)
    existing_live_tickers = {
        str(row["ticker"]) for row in cast("list[dict[str, Any]]", registry_before["sources"])
    }
    exact_counts = _exact_counts(events_before)
    feature_counts = _feature_counts(events_before)
    before = _event_metrics(events_before)

    http = client or BoundedHttpClient(max_response_bytes=MAX_RESPONSE_BYTES)
    fetch = http.get(MOEX_RISK_PARAMETERS_RSS_URL)
    evidence_items, endpoint_status, endpoint_blocker = _rss_evidence_items(
        fetch.body, universe_rows
    )
    evidence_rows = _evidence_rows(
        fetch_status=fetch.status,
        fetch_blocker=fetch.blocker,
        endpoint_status=endpoint_status,
        endpoint_blocker=endpoint_blocker,
        evidence_items=evidence_items,
    )
    target_rows = _target_rows(
        universe_rows=universe_rows,
        evidence_items=evidence_items,
        exact_counts=exact_counts,
        feature_counts=feature_counts,
        existing_live_tickers=existing_live_tickers,
    )
    candidate_rows = _candidate_rows(target_rows)
    new_sources = _new_source_rows(candidate_rows)
    registry_after = {
        "source_registry_version": SOURCE_REGISTRY_VERSION,
        "sources": [
            *_normalize_registry_rows(cast("list[dict[str, Any]]", registry_before["sources"])),
            *new_sources,
        ],
    }

    _write_json(output_root / "discovery-limits.json", DISCOVERY_LIMITS)
    _write_jsonl(output_root / "target-universe.jsonl", target_rows)
    _write_jsonl(output_root / "source-discovery-evidence.jsonl", evidence_rows)
    _write_jsonl(output_root / "source-candidates.jsonl", candidate_rows)
    _write_json(output_root / "new-source-registry.json", registry_after)
    collection_registry = {
        "source_registry_version": SOURCE_REGISTRY_VERSION,
        "sources": new_sources,
    }
    _write_json(output_root / "collection-source-registry.json", collection_registry)
    if write_registry:
        _write_json(live_registry_path, registry_after)

    first_collection = build_live_official_collection_artifact(
        output_root=output_root / "live-collection",
        base_main_sha=base_main_sha,
        git_sha=git_sha,
        input_events_path=input_events_path,
        source_registry_path=output_root / "collection-source-registry.json",
        audit_manifest_path=None,
        client=http,
        created_at=now,
    )
    replay_collection = build_live_official_collection_artifact(
        output_root=output_root / "live-collection-replay",
        base_main_sha=base_main_sha,
        git_sha=git_sha,
        input_events_path=input_events_path,
        source_registry_path=output_root / "collection-source-registry.json",
        state_path=output_root / "live-collection" / "dedupe-state.json",
        audit_manifest_path=None,
        client=http,
        created_at=now,
    )
    collection_rows = _collection_rows(output_root, first_collection, replay_collection)
    _write_jsonl(output_root / "collection-results.jsonl", collection_rows)

    collected_events = _read_jsonl(
        output_root / "live-collection" / "collected-event-metadata.jsonl"
    )
    after = _event_metrics([*events_before, *collected_events])
    status_counts = Counter(str(row["status"]) for row in candidate_rows)
    safety = source_breadth_safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": now.isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_ELIGIBILITY_ARTIFACT_SHA": eligibility_manifest["ARTIFACT_SHA"],
        "TARGET_UNIVERSE_SHA": sha256_payload(target_rows),
        "DISCOVERY_LIMITS_SHA": sha256_payload(DISCOVERY_LIMITS),
        "SOURCE_DISCOVERY_EVIDENCE_SHA": sha256_payload(evidence_rows),
        "SOURCE_CANDIDATES_SHA": sha256_payload(candidate_rows),
        "SOURCE_REGISTRY_SHA": sha256_payload(registry_after),
        "COLLECTION_SOURCE_REGISTRY_SHA": sha256_payload(collection_registry),
        "COLLECTION_RESULT_SHA": sha256_payload(collection_rows),
        "TARGET_TICKERS": len(target_rows),
        "TRADABLE_TARGET_TICKERS": sum(
            row["target_selection_status"] == "TRADABLE_TARGET" for row in target_rows
        ),
        "NON_TRADABLE_SKIPPED": sum(
            row["target_selection_status"] == CandidateStatus.CURRENTLY_NON_TRADABLE.value
            for row in target_rows
        ),
        "IDENTITY_BLOCKED": sum(
            row["target_selection_status"] == CandidateStatus.IDENTITY_AMBIGUOUS.value
            for row in target_rows
        ),
        "OFFICIAL_SOURCES_DISCOVERED": sum(
            row["status"] == CandidateStatus.EXACT_LIVE_READY.value for row in candidate_rows
        ),
        "EXACT_LIVE_READY_SOURCES": status_counts[CandidateStatus.EXACT_LIVE_READY.value],
        "EXACT_ARCHIVE_READY_SOURCES": status_counts[CandidateStatus.EXACT_ARCHIVE_READY.value],
        "DATE_ONLY_SOURCES": status_counts[CandidateStatus.DATE_ONLY.value],
        "NEW_EXACT_LIVE_SOURCES": len(new_sources),
        "NEW_SOURCE_FAMILIES": sorted({row["source_family"] for row in new_sources}),
        "NEW_TICKERS_WITH_EXACT_SOURCE": sorted({row["ticker"] for row in new_sources}),
        "ITEMS_FETCHED": first_collection["ITEMS_FETCHED"],
        "ITEMS_NEW": first_collection["ITEMS_NEW"],
        "ITEMS_DUPLICATE": first_collection["ITEMS_DUPLICATE"],
        "ITEMS_TIMESTAMP_INVALID": first_collection["ITEMS_TIMESTAMP_INVALID"],
        "NEW_CANONICAL_EXACT_EVENTS": first_collection["NEW_EXACT_EVENTS"],
        "NEW_HISTORICAL_EXACT_EVENTS": first_collection["NEW_HISTORICAL_EXACT_EVENTS"],
        "NEW_FUTURE_METADATA_ONLY_EVENTS": first_collection["NEW_FUTURE_METADATA_ONLY_EVENTS"],
        "REPLAY_ITEMS_NEW": replay_collection["ITEMS_NEW"],
        "REPLAY_ITEMS_DUPLICATE": replay_collection["ITEMS_DUPLICATE"],
        "CANONICAL_EXACT_EVENTS_TOTAL_BEFORE": eligibility_manifest["CANONICAL_EXACT_EVENTS_TOTAL"],
        "CANONICAL_EXACT_EVENTS_TOTAL_AFTER": eligibility_manifest["CANONICAL_EXACT_EVENTS_TOTAL"]
        + first_collection["NEW_EXACT_EVENTS"],
        "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_BEFORE": eligibility_manifest[
            "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS"
        ],
        "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_AFTER": eligibility_manifest[
            "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS"
        ],
        "MARKET_REACTION_INELIGIBLE_EXACT_EVENTS_BEFORE": eligibility_manifest[
            "MARKET_REACTION_INELIGIBLE_EXACT_EVENTS"
        ],
        "MARKET_REACTION_INELIGIBLE_EXACT_EVENTS_AFTER": eligibility_manifest[
            "MARKET_REACTION_INELIGIBLE_EXACT_EVENTS"
        ],
        "REACTION_READY_EVENTS_BEFORE": eligibility_manifest["REACTION_READY_EVENTS"],
        "REACTION_READY_EVENTS_AFTER": eligibility_manifest["REACTION_READY_EVENTS"],
        "FEATURE_READY_EVENTS_BEFORE": eligibility_manifest["FEATURE_READY_EVENTS"],
        "FEATURE_READY_EVENTS_AFTER": eligibility_manifest["FEATURE_READY_EVENTS"],
        "before": before,
        "after": after,
        "EVENTS_BY_TICKER_BEFORE": before["events_by_ticker"],
        "EVENTS_BY_TICKER_AFTER": after["events_by_ticker"],
        "ELIGIBLE_EXACT_BY_TICKER_BEFORE": before["events_by_ticker"],
        "ELIGIBLE_EXACT_BY_TICKER_AFTER": before["events_by_ticker"],
        "TICKER_TOP1_BEFORE": before["ticker_concentration"]["top1_share"],
        "TICKER_TOP1_AFTER": after["ticker_concentration"]["top1_share"],
        "TICKER_TOP3_BEFORE": before["ticker_concentration"]["top3_share"],
        "TICKER_TOP3_AFTER": after["ticker_concentration"]["top3_share"],
        "ISSUER_HHI_BEFORE": before["issuer_concentration"]["hhi"],
        "ISSUER_HHI_AFTER": after["issuer_concentration"]["hhi"],
        "EFFECTIVE_ISSUER_COUNT_BEFORE": before["issuer_concentration"]["effective_count"],
        "EFFECTIVE_ISSUER_COUNT_AFTER": after["issuer_concentration"]["effective_count"],
        "SOURCE_TOP1_BEFORE": before["source_concentration"]["top1_share"],
        "SOURCE_TOP1_AFTER": after["source_concentration"]["top1_share"],
        "SOURCE_TOP3_BEFORE": before["source_concentration"]["top3_share"],
        "SOURCE_TOP3_AFTER": after["source_concentration"]["top3_share"],
        "SOURCE_HHI_BEFORE": before["source_concentration"]["hhi"],
        "SOURCE_HHI_AFTER": after["source_concentration"]["hhi"],
        "EFFECTIVE_SOURCE_COUNT_BEFORE": before["source_concentration"]["effective_count"],
        "EFFECTIVE_SOURCE_COUNT_AFTER": after["source_concentration"]["effective_count"],
        "FINAL_DECISION": _final_decision(first_collection, len(new_sources)),
        "safety": safety,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = _artifact_sha(manifest)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)
    return manifest


def _require_tradability_gate(manifest: dict[str, Any]) -> None:
    if manifest.get("ARTIFACT_SHA") != (
        "106a32fbc732b6e0813827993d60af80d75a53dcea06a36a208ffb0d04f1669d"
    ):
        raise ValueError("INPUT_TRADABILITY_ELIGIBILITY_ARTIFACT_SHA_MISMATCH")
    expected = {
        "CANONICAL_EXACT_EVENTS_TOTAL": 761,
        "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS": 685,
        "MARKET_REACTION_INELIGIBLE_EXACT_EVENTS": 44,
        "REACTION_READY_EVENTS": 565,
        "FEATURE_READY_EVENTS": 564,
        "FINAL_DECISION": "SOURCE_BREADTH_EXPANSION_NEXT",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"INPUT_TRADABILITY_{key}_MISMATCH")


def _rss_evidence_items(
    body: bytes,
    universe_rows: list[dict[str, Any]],
) -> tuple[list[RssEvidenceItem], str, str | None]:
    if not body:
        return [], CandidateStatus.TECHNICAL_FAILURE.value, CandidateStatus.TECHNICAL_FAILURE.value
    tickers = sorted({str(row["ticker"]) for row in universe_rows})
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return [], CandidateStatus.TECHNICAL_FAILURE.value, CandidateStatus.TECHNICAL_FAILURE.value
    items: list[RssEvidenceItem] = []
    for item in (element for element in root.iter() if _local(element.tag) == "item"):
        if len(items) >= MAX_ITEMS_PER_FEED_DISCOVERY:
            break
        pubdate = _text(item, "pubDate")
        if not pubdate:
            continue
        try:
            published = parse_rss_pubdate_exact(pubdate)
        except ValueError:
            continue
        haystack = _item_haystack(item)
        matched = tuple(ticker for ticker in tickers if _contains_atomic_token(haystack, ticker))
        if not matched:
            continue
        items.append(
            RssEvidenceItem(
                title=_text(item, "title"),
                link=_text(item, "link"),
                guid=_text(item, "guid"),
                pubdate=pubdate,
                published_at_utc=published,
                matched_tickers=matched,
            )
        )
    return items, CandidateStatus.EXACT_LIVE_READY.value, None


def _target_rows(
    *,
    universe_rows: list[dict[str, Any]],
    evidence_items: list[RssEvidenceItem],
    exact_counts: Counter[str],
    feature_counts: Counter[str],
    existing_live_tickers: set[str],
) -> list[dict[str, Any]]:
    universe_by_ticker = _unique_universe_by_ticker(universe_rows)
    ambiguous = _ambiguous_tickers(universe_rows)
    matched_counts = Counter(ticker for item in evidence_items for ticker in item.matched_tickers)
    candidates = sorted(
        matched_counts,
        key=lambda ticker: (
            exact_counts[ticker],
            ticker in existing_live_tickers,
            -matched_counts[ticker],
            ticker,
        ),
    )
    rows: list[dict[str, Any]] = []
    for index, ticker in enumerate(candidates[:MAX_TARGET_TICKERS]):
        source = universe_by_ticker.get(ticker)
        if ticker in ambiguous:
            status = CandidateStatus.IDENTITY_AMBIGUOUS.value
        elif source is None or not _is_currently_tradable(source):
            status = CandidateStatus.CURRENTLY_NON_TRADABLE.value
        else:
            status = "TRADABLE_TARGET"
        rows.append(
            {
                "ticker": ticker,
                "issuer": str(source.get("name") if source else ticker),
                "instrument_uid": str(source.get("instrument_uid") if source else ""),
                "figi": str(source.get("figi") if source else ""),
                "class_code": str(source.get("class_code") if source else ""),
                "currency": str(source.get("currency") if source else ""),
                "exchange": str(source.get("exchange") if source else ""),
                "exact_event_count_before": exact_counts[ticker],
                "feature_ready_count_before": feature_counts[ticker],
                "live_feed_matched_item_count": matched_counts[ticker],
                "already_in_live_registry": ticker in existing_live_tickers,
                "target_selection_status": status,
                "batch_index": index // 5,
                "priority_inputs_used": [
                    "current_project_tqbr_universe",
                    "official_live_rss_item_mentions",
                    "exact_count_before",
                    "existing_live_registry",
                ],
            }
        )
    return rows


def _candidate_rows(target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in target_rows:
        if row["target_selection_status"] != "TRADABLE_TARGET":
            status = str(row["target_selection_status"])
        elif row["already_in_live_registry"]:
            status = CandidateStatus.EXACT_LIVE_READY.value
        else:
            status = CandidateStatus.EXACT_LIVE_READY.value
        rows.append(
            {
                "ticker": row["ticker"],
                "issuer": row["issuer"],
                "instrument_uid": row["instrument_uid"],
                "status": status,
                "official_domain": "www.moex.com"
                if status == CandidateStatus.EXACT_LIVE_READY.value
                else None,
                "source_url": MOEX_RISK_PARAMETERS_RSS_URL
                if status == CandidateStatus.EXACT_LIVE_READY.value
                else None,
                "source_family": MOEX_RISK_PARAMETERS_SOURCE_FAMILY
                if status == CandidateStatus.EXACT_LIVE_READY.value
                else None,
                "mechanism_type": "RSS"
                if status == CandidateStatus.EXACT_LIVE_READY.value
                else None,
                "timestamp_field": "RSS item pubDate"
                if status == CandidateStatus.EXACT_LIVE_READY.value
                else None,
                "timestamp_policy": (
                    "Require item-level RSS pubDate with publication date, clock time, and "
                    "explicit numeric timezone such as +0300; item text must contain ticker token."
                )
                if status == CandidateStatus.EXACT_LIVE_READY.value
                else None,
                "item_match_any": [row["ticker"]]
                if status == CandidateStatus.EXACT_LIVE_READY.value
                else [],
                "items_matched": row["live_feed_matched_item_count"],
                "exact_event_count_before": row["exact_event_count_before"],
                "already_in_live_registry": row["already_in_live_registry"],
                "selection_reason": (
                    "Active TQBR/RUB share with exact item-level official MOEX risk-parameter "
                    "RSS publication timestamp."
                )
                if status == CandidateStatus.EXACT_LIVE_READY.value
                else "Skipped by tradability or identity gate.",
            }
        )
    return rows


def _new_source_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        if len(rows) >= MAX_NEW_LIVE_SOURCES:
            break
        if row["status"] != CandidateStatus.EXACT_LIVE_READY.value:
            continue
        if row["already_in_live_registry"]:
            continue
        source = {
            "source_registry_version": SOURCE_REGISTRY_VERSION,
            "source_id": f"{row['ticker']}_MOEX_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "ticker": row["ticker"],
            "issuer": row["issuer"],
            "instrument_uid": row["instrument_uid"],
            "source_family": MOEX_RISK_PARAMETERS_SOURCE_FAMILY,
            "source_url": MOEX_RISK_PARAMETERS_RSS_URL,
            "official_domain": "www.moex.com",
            "mechanism_type": "RSS",
            "timestamp_field": "RSS item pubDate",
            "timestamp_policy": row["timestamp_policy"],
            "archive_capability": False,
            "live_capability": True,
            "provenance_evidence_url": (
                "artifacts/exact-event-live-source-breadth-expansion-v1/"
                f"source-candidates.jsonl#{row['ticker']}"
            ),
            "provenance_evidence_sha": sha256_payload(row),
            "enabled": True,
            "parser_version": "rss-item-pubdate-exact-v1",
            "item_match_any": row["item_match_any"],
        }
        LiveExactSource.from_payload(source)
        rows.append(source)
    return rows


def _collection_rows(
    output_root: Path,
    first_manifest: dict[str, Any],
    replay_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    first_events = _read_jsonl(output_root / "live-collection" / "collected-event-metadata.jsonl")
    replay_events = _read_jsonl(
        output_root / "live-collection-replay" / "collected-event-metadata.jsonl"
    )
    return [
        *_collection_source_rows("collection", first_manifest, first_events),
        *_collection_source_rows("replay", replay_manifest, replay_events),
    ]


def _collection_source_rows(
    run: str,
    manifest: dict[str, Any],
    event_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events_by_source = Counter(
        str(cast("dict[str, Any]", row["metadata"])["source_id"]) for row in event_rows
    )
    historical_by_source = Counter(
        str(cast("dict[str, Any]", row["metadata"])["source_id"])
        for row in event_rows
        if not bool(cast("dict[str, Any]", row["metadata"]).get("future_holdout"))
    )
    future_by_source = Counter(
        str(cast("dict[str, Any]", row["metadata"])["source_id"])
        for row in event_rows
        if bool(cast("dict[str, Any]", row["metadata"]).get("future_holdout"))
    )
    rows: list[dict[str, Any]] = []
    for report in cast("list[dict[str, Any]]", manifest["PER_SOURCE_STATUS"]):
        source_id = str(report["source_id"])
        rows.append(
            {
                "run": run,
                "source_id": source_id,
                "ticker": report["ticker"],
                "source_url": report["source_url"],
                "status": report["status"],
                "items_fetched": report["items_fetched"],
                "items_valid": int(report["items_new"]) + int(report["items_duplicate"]),
                "items_invalid": 0,
                "items_new": report["items_new"],
                "items_duplicate": report["items_duplicate"],
                "new_canonical_exact_events": events_by_source[source_id],
                "new_historical_exact_events": historical_by_source[source_id],
                "new_future_metadata_only_events": future_by_source[source_id],
                "errors": report["blocker"],
            }
        )
    return rows


def _event_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ticker_counts: Counter[str] = Counter()
    issuer_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    for row in rows:
        metadata = cast("dict[str, Any]", row["metadata"])
        ticker = str(metadata.get("ticker") or "")
        issuer = str(metadata.get("issuer") or ticker)
        source = str(metadata.get("source_code") or metadata.get("source_family") or "UNKNOWN")
        if ticker:
            ticker_counts[ticker] += 1
        if issuer:
            issuer_counts[issuer] += 1
        if source:
            source_counts[source] += 1
        target = cast("dict[str, Any]", row.get("target_availability") or {})
        if bool(target.get("feature_ready")) and ticker:
            feature_counts[ticker] += 1
    return {
        "EXACT_TOTAL": sum(ticker_counts.values()),
        "EXACT_UNIQUE_TICKERS": len(ticker_counts),
        "EXACT_UNIQUE_ISSUERS": len(issuer_counts),
        "FEATURE_READY": sum(feature_counts.values()),
        "events_by_ticker": dict(sorted(ticker_counts.items())),
        "feature_ready_by_ticker": dict(sorted(feature_counts.items())),
        "ticker_concentration": _concentration(ticker_counts),
        "issuer_concentration": _concentration(issuer_counts),
        "source_concentration": _concentration(source_counts),
    }


def _evidence_rows(
    *,
    fetch_status: int | None,
    fetch_blocker: str | None,
    endpoint_status: str,
    endpoint_blocker: str | None,
    evidence_items: list[RssEvidenceItem],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "evidence_type": "OFFICIAL_RSS_ENDPOINT",
            "source_url": MOEX_RISK_PARAMETERS_RSS_URL,
            "official_domain": "www.moex.com",
            "http_status": fetch_status,
            "fetch_blocker": fetch_blocker,
            "status": endpoint_status,
            "blocker": endpoint_blocker,
            "timestamp_field": "RSS item pubDate",
            "timezone_semantics": "EXPLICIT_NUMERIC_OFFSET",
            "source_documentation_url": "https://www.moex.com/a40",
        }
    ]
    for item in evidence_items:
        rows.append(
            {
                "evidence_type": "RSS_ITEM_EXACT_TIMESTAMP",
                "source_url": MOEX_RISK_PARAMETERS_RSS_URL,
                "canonical_url": item.link,
                "guid": item.guid,
                "title": item.title,
                "publication_timestamp_raw": item.pubdate,
                "publication_timestamp_utc": item.published_at_utc.isoformat(),
                "matched_tickers": list(item.matched_tickers),
                "status": CandidateStatus.EXACT_LIVE_READY.value,
            }
        )
    return rows


def _read_universe(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = cast("list[dict[str, Any]]", payload["instruments"])
    return [
        row
        for row in rows
        if str(row.get("class_code")) == "TQBR"
        and str(row.get("currency")) == "rub"
        and str(row.get("instrument_type")) == "INSTRUMENT_TYPE_SHARE"
    ]


def _is_currently_tradable(row: dict[str, Any]) -> bool:
    return str(row.get("exchange") or "").lower() not in {"", "unknown"}


def _unique_universe_by_ticker(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts = Counter(str(row["ticker"]) for row in rows)
    return {str(row["ticker"]): row for row in rows if counts[str(row["ticker"])] == 1}


def _ambiguous_tickers(rows: list[dict[str, Any]]) -> set[str]:
    counts = Counter(str(row["ticker"]) for row in rows)
    return {ticker for ticker, count in counts.items() if count > 1}


def _normalize_registry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [LiveExactSource.from_payload(row).payload() for row in rows]


def _exact_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(cast("dict[str, Any]", row["metadata"])["ticker"]) for row in rows)


def _feature_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        target = cast("dict[str, Any]", row.get("target_availability") or {})
        if bool(target.get("feature_ready")):
            ticker = str(cast("dict[str, Any]", row["metadata"])["ticker"])
            counts[ticker] += 1
    return counts


def _concentration(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    if total == 0:
        return {
            "counts": {},
            "top1_share": 0.0,
            "top3_share": 0.0,
            "hhi": 0.0,
            "effective_count": 0.0,
        }
    shares = sorted((count / total for count in counts.values()), reverse=True)
    hhi = sum(share * share for share in shares)
    return {
        "counts": dict(sorted(counts.items())),
        "top1_share": shares[0],
        "top3_share": sum(shares[:3]),
        "hhi": hhi,
        "effective_count": 1 / hhi if hhi else 0.0,
    }


def _final_decision(collection: dict[str, Any], new_source_count: int) -> str:
    if new_source_count > 0 and collection["NEW_EXACT_EVENTS"] > 0:
        return "SOURCE_BREADTH_GAINED"
    if new_source_count > 0:
        return "LIVE_ACCUMULATION_MAIN_PATH"
    blockers = cast("dict[str, Any]", collection.get("BLOCKERS_BY_TYPE") or {})
    if blockers.get("TECHNICAL_FAILURE"):
        return "SOURCE_DISCOVERY_TECHNICAL_BLOCKERS_DOMINATE"
    return "MORE_OFFICIAL_SOURCE_DISCOVERY"


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


def _item_haystack(item: ET.Element) -> str:
    return " ".join(
        value
        for value in (
            _text(item, "title"),
            _text(item, "description"),
            _text(item, "encoded"),
        )
        if value
    )


def _contains_atomic_token(haystack: str, token: str) -> bool:
    escaped = re.escape(token.strip())
    if not escaped:
        return False
    return re.search(rf"(?<![A-Z0-9]){escaped}(?![A-Z0-9])", haystack, re.IGNORECASE) is not None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_live_registry_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("source_registry_version") != SOURCE_REGISTRY_VERSION:
        raise ValueError("SOURCE_REGISTRY_VERSION_MISMATCH")
    return payload


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
        "Data-acquisition-only live official EXACT source breadth expansion.",
        "",
        f"- ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        f"- TARGET_UNIVERSE_SHA={manifest['TARGET_UNIVERSE_SHA']}",
        f"- DISCOVERY_LIMITS_SHA={manifest['DISCOVERY_LIMITS_SHA']}",
        f"- SOURCE_DISCOVERY_EVIDENCE_SHA={manifest['SOURCE_DISCOVERY_EVIDENCE_SHA']}",
        f"- SOURCE_CANDIDATES_SHA={manifest['SOURCE_CANDIDATES_SHA']}",
        f"- SOURCE_REGISTRY_SHA={manifest['SOURCE_REGISTRY_SHA']}",
        f"- COLLECTION_RESULT_SHA={manifest['COLLECTION_RESULT_SHA']}",
        "",
        f"- TARGET_TICKERS={manifest['TARGET_TICKERS']}",
        f"- TRADABLE_TARGET_TICKERS={manifest['TRADABLE_TARGET_TICKERS']}",
        f"- NEW_EXACT_LIVE_SOURCES={manifest['NEW_EXACT_LIVE_SOURCES']}",
        f"- NEW_TICKERS_WITH_EXACT_SOURCE={manifest['NEW_TICKERS_WITH_EXACT_SOURCE']}",
        f"- ITEMS_FETCHED={manifest['ITEMS_FETCHED']}",
        f"- ITEMS_NEW={manifest['ITEMS_NEW']}",
        f"- ITEMS_DUPLICATE={manifest['ITEMS_DUPLICATE']}",
        f"- ITEMS_TIMESTAMP_INVALID={manifest['ITEMS_TIMESTAMP_INVALID']}",
        "",
        f"- NEW_CANONICAL_EXACT_EVENTS={manifest['NEW_CANONICAL_EXACT_EVENTS']}",
        f"- NEW_HISTORICAL_EXACT_EVENTS={manifest['NEW_HISTORICAL_EXACT_EVENTS']}",
        f"- NEW_FUTURE_METADATA_ONLY_EVENTS={manifest['NEW_FUTURE_METADATA_ONLY_EVENTS']}",
        f"- REPLAY_ITEMS_NEW={manifest['REPLAY_ITEMS_NEW']}",
        f"- REPLAY_ITEMS_DUPLICATE={manifest['REPLAY_ITEMS_DUPLICATE']}",
        "",
        f"FINAL_DECISION={manifest['FINAL_DECISION']}",
        "",
        "No model training, TEST outcome use, future outcome read, market maturation, backtest, "
        "paper trading, real trading, orders, or broker mutation was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
