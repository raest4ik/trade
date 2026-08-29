from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
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
from src.exact_event_live_official_collection.http_client import (
    BoundedHttpClient,
    FetchResult,
    HttpClient,
)
from src.exact_event_live_source_breadth_expansion.domain import (
    MAX_RESPONSE_BYTES,
    CandidateStatus,
)
from src.exact_event_live_source_breadth_expansion_v2.domain import (
    ARTIFACT_VERSION,
    DEFAULT_BASE_EVENTS_PATH,
    DEFAULT_LIVE_REGISTRY_PATH,
    DEFAULT_V1_ARTIFACT_ROOT,
    EXCLUDED_TICKERS,
    EXPECTED_V1_ARTIFACT_SHA,
    EXPECTED_V1_CANDIDATE_SET_SHA,
    MAX_NEW_LIVE_SOURCES,
    V1_ONBOARDED_TICKERS,
    artifact_sha,
    safety_flags,
    sha256_payload,
)


def build_live_source_breadth_expansion_v2_artifact(
    *,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    v1_artifact_root: Path = Path(DEFAULT_V1_ARTIFACT_ROOT),
    base_events_path: Path = Path(DEFAULT_BASE_EVENTS_PATH),
    live_registry_path: Path = Path(DEFAULT_LIVE_REGISTRY_PATH),
    client: HttpClient | None = None,
    created_at: datetime | None = None,
    write_registry: bool = True,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable live source breadth v2 artifact output already exists")
    now = created_at or datetime.now(UTC)
    output_root.mkdir(parents=True, exist_ok=False)

    v1_manifest = _read_json(v1_artifact_root / "manifest.json")
    _require_v1_artifact(v1_manifest)
    input_candidate_set = _read_jsonl(v1_artifact_root / "source-candidates.jsonl")
    input_candidate_set_sha = sha256_payload(input_candidate_set)
    if (
        input_candidate_set_sha != EXPECTED_V1_CANDIDATE_SET_SHA
        or v1_manifest.get("SOURCE_CANDIDATES_SHA") != EXPECTED_V1_CANDIDATE_SET_SHA
    ):
        raise ValueError("V2_CANDIDATE_SET_NOT_REPRODUCIBLE")
    v1_target_rows = _read_jsonl(v1_artifact_root / "target-universe.jsonl")
    feature_counts = {
        str(row["ticker"]): int(row.get("feature_ready_count_before") or 0)
        for row in v1_target_rows
    }
    registry_before = _read_live_registry(live_registry_path)
    existing_live_tickers = {str(row["ticker"]) for row in registry_before}
    base_events = _read_jsonl(base_events_path)
    v1_events = _read_jsonl(v1_artifact_root / "live-collection" / "collected-event-metadata.jsonl")
    events_before = [*base_events, *v1_events]
    _write_jsonl(output_root / "input-candidate-set.jsonl", input_candidate_set)
    _write_jsonl(output_root / "existing-events-before.jsonl", events_before)

    ordered_candidates = _ordered_v2_candidates(
        input_candidate_set=input_candidate_set,
        feature_counts=feature_counts,
        existing_live_tickers=existing_live_tickers,
    )
    http = _SnapshotHttpClient(client or BoundedHttpClient(max_response_bytes=MAX_RESPONSE_BYTES))
    validated, validation_rows = _validate_next_batch(
        ordered_candidates=ordered_candidates,
        http=http,
    )
    selected_sources = [_source_from_candidate(row) for row in validated]
    selected_source_ids = {str(row["source_id"]) for row in selected_sources}
    selected_cohort = [
        row for row in validation_rows if str(row.get("source_id") or "") in selected_source_ids
    ]
    registry_after = {
        "source_registry_version": SOURCE_REGISTRY_VERSION,
        "sources": [*_normalize_registry(registry_before), *selected_sources],
    }
    collection_registry = {
        "source_registry_version": SOURCE_REGISTRY_VERSION,
        "sources": selected_sources,
    }
    _write_jsonl(output_root / "selected-source-cohort.jsonl", selected_cohort)
    _write_jsonl(output_root / "source-validation.jsonl", validation_rows)
    _write_json(output_root / "new-source-registry.json", registry_after)
    _write_json(output_root / "collection-source-registry.json", collection_registry)
    if write_registry:
        _write_json(live_registry_path, registry_after)

    collection = build_live_official_collection_artifact(
        output_root=output_root / "live-collection",
        base_main_sha=base_main_sha,
        git_sha=git_sha,
        input_events_path=output_root / "existing-events-before.jsonl",
        source_registry_path=output_root / "collection-source-registry.json",
        audit_manifest_path=None,
        client=http,
        created_at=now,
    )
    replay = build_live_official_collection_artifact(
        output_root=output_root / "live-collection-replay",
        base_main_sha=base_main_sha,
        git_sha=git_sha,
        input_events_path=output_root / "existing-events-before.jsonl",
        source_registry_path=output_root / "collection-source-registry.json",
        state_path=output_root / "live-collection" / "dedupe-state.json",
        audit_manifest_path=None,
        client=http,
        created_at=now,
    )
    collection_events = _read_jsonl(
        output_root / "live-collection" / "collected-event-metadata.jsonl"
    )
    replay_events = _read_jsonl(
        output_root / "live-collection-replay" / "collected-event-metadata.jsonl"
    )
    collection_rows = _collection_source_rows("collection", collection, collection_events)
    replay_rows = _collection_source_rows("replay", replay, replay_events)
    _write_jsonl(output_root / "collection-results.jsonl", collection_rows)
    _write_jsonl(output_root / "replay-results.jsonl", replay_rows)

    before = _event_metrics(events_before)
    after = _event_metrics([*events_before, *collection_events])
    remaining = [
        row
        for row in ordered_candidates
        if str(row["ticker"]) not in {str(item["ticker"]) for item in selected_sources}
    ]
    flags = safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": now.isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_V1_ARTIFACT_SHA": v1_manifest["ARTIFACT_SHA"],
        "INPUT_CANDIDATE_SET_SHA": input_candidate_set_sha,
        "SELECTED_SOURCE_COHORT_SHA": sha256_payload(selected_cohort),
        "SOURCE_VALIDATION_SHA": sha256_payload(validation_rows),
        "SOURCE_REGISTRY_SHA": sha256_payload(registry_after),
        "COLLECTION_SOURCE_REGISTRY_SHA": sha256_payload(collection_registry),
        "COLLECTION_RESULT_SHA": sha256_payload(collection_rows),
        "REPLAY_RESULT_SHA": sha256_payload(replay_rows),
        "CANDIDATES_AVAILABLE": len(ordered_candidates),
        "CANDIDATES_ATTEMPTED": len(validation_rows),
        "V1_ONBOARDED_TICKERS": list(V1_ONBOARDED_TICKERS),
        "NEW_EXACT_LIVE_SOURCES": len(selected_sources),
        "NEW_TICKERS_WITH_EXACT_SOURCE": sorted({str(row["ticker"]) for row in selected_sources}),
        "NEW_SOURCE_FAMILIES": sorted({str(row["source_family"]) for row in selected_sources}),
        "V1_CANDIDATES_REMAINING_AFTER_BATCH": len(remaining),
        "ITEMS_FETCHED": collection["ITEMS_FETCHED"],
        "ITEMS_NEW": collection["ITEMS_NEW"],
        "ITEMS_DUPLICATE": collection["ITEMS_DUPLICATE"],
        "ITEMS_TIMESTAMP_INVALID": collection["ITEMS_TIMESTAMP_INVALID"],
        "NEW_CANONICAL_EXACT_EVENTS": collection["NEW_EXACT_EVENTS"],
        "NEW_HISTORICAL_EXACT_EVENTS": collection["NEW_HISTORICAL_EXACT_EVENTS"],
        "NEW_FUTURE_METADATA_ONLY_EVENTS": collection["NEW_FUTURE_METADATA_ONLY_EVENTS"],
        "REPLAY_ITEMS_NEW": replay["ITEMS_NEW"],
        "REPLAY_ITEMS_DUPLICATE": replay["ITEMS_DUPLICATE"],
        "CANONICAL_EXACT_EVENTS_BEFORE": v1_manifest["CANONICAL_EXACT_EVENTS_TOTAL_AFTER"],
        "CANONICAL_EXACT_EVENTS_AFTER": v1_manifest["CANONICAL_EXACT_EVENTS_TOTAL_AFTER"]
        + collection["NEW_EXACT_EVENTS"],
        "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_BEFORE": v1_manifest[
            "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_AFTER"
        ],
        "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_AFTER": v1_manifest[
            "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_AFTER"
        ],
        "EVENTS_BY_TICKER_BEFORE": before["events_by_ticker"],
        "EVENTS_BY_TICKER_AFTER": after["events_by_ticker"],
        "TOP1_TICKER_SHARE_BEFORE": before["ticker_concentration"]["top1_share"],
        "TOP1_TICKER_SHARE_AFTER": after["ticker_concentration"]["top1_share"],
        "TOP3_TICKER_SHARE_BEFORE": before["ticker_concentration"]["top3_share"],
        "TOP3_TICKER_SHARE_AFTER": after["ticker_concentration"]["top3_share"],
        "ISSUER_HHI_BEFORE": before["issuer_concentration"]["hhi"],
        "ISSUER_HHI_AFTER": after["issuer_concentration"]["hhi"],
        "EFFECTIVE_ISSUER_COUNT_BEFORE": before["issuer_concentration"]["effective_count"],
        "EFFECTIVE_ISSUER_COUNT_AFTER": after["issuer_concentration"]["effective_count"],
        "UNRELATED_ITEMS_REJECTED": sum(
            int(row["unrelated_items_rejected"]) for row in validation_rows
        ),
        "DATE_ONLY_COERCIONS": 0,
        "FETCH_TIME_USED_AS_PUBLICATION_TIME": False,
        "FINAL_DECISION": _decision(collection, selected_sources, before, after),
        "safety": flags,
        **flags,
    }
    manifest["ARTIFACT_SHA"] = artifact_sha(manifest)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)
    return manifest


def _require_v1_artifact(manifest: dict[str, Any]) -> None:
    expected = {
        "ARTIFACT_SHA": EXPECTED_V1_ARTIFACT_SHA,
        "NEW_EXACT_LIVE_SOURCES": 5,
        "NEW_TICKERS_WITH_EXACT_SOURCE": ["AFKS", "ASTR", "ELMT", "OZON", "RUAL"],
        "NEW_CANONICAL_EXACT_EVENTS": 56,
        "NEW_HISTORICAL_EXACT_EVENTS": 45,
        "NEW_FUTURE_METADATA_ONLY_EVENTS": 11,
        "REPLAY_ITEMS_NEW": 0,
        "REPLAY_ITEMS_DUPLICATE": 56,
        "FINAL_DECISION": "SOURCE_BREADTH_GAINED",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"INPUT_V1_{key}_MISMATCH")


class _SnapshotHttpClient:
    def __init__(self, inner: HttpClient) -> None:
        self._inner = inner
        self._cache: dict[str, FetchResult] = {}

    def get(self, url: str) -> FetchResult:
        cached = self._cache.get(url)
        if cached is not None:
            return cached
        result = self._inner.get(url)
        self._cache[url] = result
        return result


def _ordered_v2_candidates(
    *,
    input_candidate_set: list[dict[str, Any]],
    feature_counts: dict[str, int],
    existing_live_tickers: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in input_candidate_set:
        ticker = str(row["ticker"])
        if row.get("status") != CandidateStatus.EXACT_LIVE_READY.value:
            continue
        if ticker in EXCLUDED_TICKERS or ticker in existing_live_tickers:
            continue
        if not row.get("instrument_uid"):
            continue
        candidate = dict(row)
        candidate["feature_ready_count_before"] = feature_counts.get(ticker, 0)
        rows.append(candidate)
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("exact_event_count_before") or 0),
            int(row.get("feature_ready_count_before") or 0),
            str(row["ticker"]),
        ),
    )


def _validate_next_batch(
    *,
    ordered_candidates: list[dict[str, Any]],
    http: HttpClient,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validated: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for candidate in ordered_candidates:
        result = http.get(str(candidate["source_url"]))
        exact_matches, unrelated_rejected, blocker = _validate_candidate(candidate, result.body)
        passed = blocker is None and exact_matches > 0
        source_id = _source_id(candidate)
        rows.append(
            {
                "ticker": candidate["ticker"],
                "issuer": candidate["issuer"],
                "source_id": source_id,
                "source_url": candidate["source_url"],
                "mechanism": candidate["mechanism_type"],
                "item_match_any": candidate["item_match_any"],
                "http_status": result.status,
                "fetch_blocker": result.blocker,
                "validation_status": "CURRENTLY_TRADABLE_SOURCE_TARGET"
                if passed
                else "CANDIDATE_VALIDATION_BLOCKED",
                "blocker": blocker,
                "exact_matching_items": exact_matches,
                "unrelated_items_rejected": unrelated_rejected,
                "source_evidence_remains_valid": passed,
            }
        )
        if passed:
            validated.append(candidate)
        if len(validated) >= MAX_NEW_LIVE_SOURCES:
            break
    return validated, rows


def _validate_candidate(candidate: dict[str, Any], body: bytes) -> tuple[int, int, str | None]:
    if not body:
        return 0, 0, CandidateStatus.TECHNICAL_FAILURE.value
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return 0, 0, CandidateStatus.TECHNICAL_FAILURE.value
    token = str(cast("list[Any]", candidate["item_match_any"])[0])
    exact_matches = 0
    unrelated_rejected = 0
    timestamp_blocker: str | None = None
    for item in (element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "item"):
        haystack = _item_haystack(item)
        if not _contains_atomic_token(haystack, token):
            unrelated_rejected += 1
            continue
        pubdate = _text(item, "pubDate")
        try:
            parse_result = _parse_pubdate(pubdate)
        except ValueError as exc:
            timestamp_blocker = str(exc)
            continue
        if parse_result:
            exact_matches += 1
    return exact_matches, unrelated_rejected, timestamp_blocker


def _parse_pubdate(pubdate: str) -> bool:
    parse_rss_pubdate_exact(pubdate)
    return True


def _source_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    source = {
        "source_registry_version": SOURCE_REGISTRY_VERSION,
        "source_id": _source_id(candidate),
        "ticker": candidate["ticker"],
        "issuer": candidate["issuer"],
        "instrument_uid": candidate["instrument_uid"],
        "source_family": candidate["source_family"],
        "source_url": candidate["source_url"],
        "official_domain": candidate["official_domain"],
        "mechanism_type": candidate["mechanism_type"],
        "timestamp_field": candidate["timestamp_field"],
        "timestamp_policy": candidate["timestamp_policy"],
        "archive_capability": False,
        "live_capability": True,
        "provenance_evidence_url": (
            f"artifacts/{ARTIFACT_VERSION}/selected-source-cohort.jsonl#{candidate['ticker']}"
        ),
        "provenance_evidence_sha": sha256_payload(candidate),
        "enabled": True,
        "parser_version": "rss-item-pubdate-exact-v1",
        "item_match_any": candidate["item_match_any"],
    }
    LiveExactSource.from_payload(source)
    return source


def _source_id(candidate: dict[str, Any]) -> str:
    return f"{candidate['ticker']}_MOEX_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"


def _read_live_registry(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if payload.get("source_registry_version") != SOURCE_REGISTRY_VERSION:
        raise ValueError("SOURCE_REGISTRY_VERSION_MISMATCH")
    return cast("list[dict[str, Any]]", payload.get("sources") or [])


def _normalize_registry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [LiveExactSource.from_payload(row).payload() for row in rows]


def _decision(
    collection: dict[str, Any],
    selected_sources: list[dict[str, Any]],
    before: dict[str, Any],
    after: dict[str, Any],
) -> str:
    if not selected_sources:
        return "CANDIDATE_SET_EXHAUSTED"
    if collection["NEW_EXACT_EVENTS"] == 0:
        return "CANDIDATE_VALIDATION_BLOCKED"
    if (
        after["ticker_concentration"]["top1_share"] <= before["ticker_concentration"]["top1_share"]
        and after["ticker_concentration"]["top3_share"]
        <= before["ticker_concentration"]["top3_share"]
        and after["issuer_concentration"]["hhi"] <= before["issuer_concentration"]["hhi"]
    ):
        return "SOURCE_BREADTH_GAINED"
    return "SOURCE_DIVERSITY_GAIN_TOO_SMALL"


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
    for row in rows:
        metadata = cast("dict[str, Any]", row["metadata"])
        ticker = str(metadata.get("ticker") or "")
        issuer = str(metadata.get("issuer") or ticker)
        if ticker:
            ticker_counts[ticker] += 1
        if issuer:
            issuer_counts[issuer] += 1
    return {
        "events_by_ticker": dict(sorted(ticker_counts.items())),
        "ticker_concentration": _concentration(ticker_counts),
        "issuer_concentration": _concentration(issuer_counts),
    }


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


def _text(item: ET.Element, tag: str) -> str:
    value = next(
        (child.text for child in item if _local(child.tag) == tag and child.text is not None),
        None,
    )
    return " ".join(value.split()) if value else ""


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
        "Second deterministic batch from the v1 EXACT_LIVE_READY candidate set.",
        "",
        f"- ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        f"- INPUT_V1_ARTIFACT_SHA={manifest['INPUT_V1_ARTIFACT_SHA']}",
        f"- INPUT_CANDIDATE_SET_SHA={manifest['INPUT_CANDIDATE_SET_SHA']}",
        f"- SELECTED_SOURCE_COHORT_SHA={manifest['SELECTED_SOURCE_COHORT_SHA']}",
        f"- SOURCE_VALIDATION_SHA={manifest['SOURCE_VALIDATION_SHA']}",
        f"- SOURCE_REGISTRY_SHA={manifest['SOURCE_REGISTRY_SHA']}",
        f"- COLLECTION_RESULT_SHA={manifest['COLLECTION_RESULT_SHA']}",
        f"- REPLAY_RESULT_SHA={manifest['REPLAY_RESULT_SHA']}",
        "",
        f"- CANDIDATES_AVAILABLE={manifest['CANDIDATES_AVAILABLE']}",
        f"- CANDIDATES_ATTEMPTED={manifest['CANDIDATES_ATTEMPTED']}",
        f"- NEW_EXACT_LIVE_SOURCES={manifest['NEW_EXACT_LIVE_SOURCES']}",
        f"- NEW_TICKERS_WITH_EXACT_SOURCE={manifest['NEW_TICKERS_WITH_EXACT_SOURCE']}",
        f"- NEW_SOURCE_FAMILIES={manifest['NEW_SOURCE_FAMILIES']}",
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
        f"- TOP1_TICKER_SHARE={manifest['TOP1_TICKER_SHARE_BEFORE']} -> "
        f"{manifest['TOP1_TICKER_SHARE_AFTER']}",
        f"- TOP3_TICKER_SHARE={manifest['TOP3_TICKER_SHARE_BEFORE']} -> "
        f"{manifest['TOP3_TICKER_SHARE_AFTER']}",
        f"- ISSUER_HHI={manifest['ISSUER_HHI_BEFORE']} -> {manifest['ISSUER_HHI_AFTER']}",
        "",
        f"FINAL_DECISION={manifest['FINAL_DECISION']}",
        "",
        "No model training, TEST outcome use, future outcome read, market maturation, backtest, "
        "paper trading, real trading, orders, or broker mutation was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
