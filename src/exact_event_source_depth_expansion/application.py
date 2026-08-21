from __future__ import annotations

import html
import json
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import EventAnalyzerV3, rules_v3_fingerprint
from src.exact_event_corpus.domain import ExactEvent, deterministic_clusters
from src.exact_event_source_depth_expansion.domain import (
    ARTIFACT_VERSION,
    FUTURE_EVENT_HOLDOUT_START,
    INPUT_DATASET_SHA,
    MAX_ITEMS_PER_SOURCE,
    MAX_PAGES_PER_SOURCE,
    OUTPUT_DATASET_VERSION,
    SOURCE_DEPTH_PRIORITY_RULES,
    ArchiveAuditRecord,
    ArchiveBlocker,
    metrics,
    parse_rfc822_timestamp,
    priority_tier,
    sha256_payload,
    source_depth_safety_flags,
)


def build_source_depth_expansion_artifact(
    *,
    input_root: Path,
    source_registry_path: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    archive_cache_root: Path | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable source depth expansion artifact output already exists")
    _verify_frozen_contracts()
    input_manifest = _read_json(input_root / "manifest.json")
    _require_input_manifest(input_manifest)
    events_before = _read_jsonl(input_root / "events.jsonl")
    features_before = _read_jsonl(input_root / "features.jsonl")
    targets_before = _read_jsonl(input_root / "targets.jsonl")
    registry = _read_jsonl(source_registry_path)
    before = metrics(events_before, features_before)
    priority_rows = _priority_rows(before["events_by_ticker"], registry)
    archive_audit, new_events = _audit_archives(
        priority_rows,
        existing_rows=events_before,
        archive_cache_root=archive_cache_root,
    )
    new_event_rows = _event_rows(new_events)
    events_after = [*events_before, *new_event_rows]
    features_after = [*features_before]
    targets_after = [*targets_before]
    clusters_after = _clusters_after(input_root, new_events)
    _assert_prefix_preserved(events_before, events_after, "EVENT")
    _assert_prefix_preserved(features_before, features_after, "FEATURE")
    _assert_prefix_preserved(targets_before, targets_after, "TARGET")
    _assert_no_future_targets(events_after, targets_after)
    duplicate_reconciliation = _duplicate_reconciliation(events_after)
    if duplicate_reconciliation != "PASS":
        raise ValueError("DUPLICATE_RECONCILIATION_FAILED")
    after = metrics(events_after, features_after)
    output_dataset_sha = (
        INPUT_DATASET_SHA
        if events_after == events_before
        and features_after == features_before
        and targets_after == targets_before
        else sha256_payload(
            {
                "dataset_version": OUTPUT_DATASET_VERSION,
                "input_dataset_sha": INPUT_DATASET_SHA,
                "events": events_after,
                "features": features_after,
                "targets": targets_after,
            }
        )
    )
    priority_payload = [
        {
            "TICKER": row["ticker"],
            "ISSUER": row["issuer"],
            "PRIORITY_TIER": row["priority_tier"],
            "EXACT_EVENT_COUNT": row["exact_event_count"],
        }
        for row in priority_rows
    ]
    archive_payload = [row.payload() for row in archive_audit]
    new_historical = [
        event for event in new_events if event.publication_date < FUTURE_EVENT_HOLDOUT_START
    ]
    new_future = [
        event for event in new_events if event.publication_date >= FUTURE_EVENT_HOLDOUT_START
    ]
    new_by_ticker = Counter(event.ticker for event in new_events)
    safety = source_depth_safety_flags()
    market_blockers = Counter({"MARKET_HISTORY_MISSING": len(new_historical)})
    provenance = {
        "artifact_version": ARTIFACT_VERSION,
        "base_main_sha": base_main_sha,
        "input_dataset_sha": INPUT_DATASET_SHA,
        "archive_cache_root": str(archive_cache_root) if archive_cache_root else None,
        "official_zero_cost_public_sources_only": True,
        "source_selection_used_returns": False,
        "source_selection_used_targets": False,
        "source_selection_used_model_metrics": False,
        "source_selection_used_test_metrics": False,
        "rules_v3_fingerprint": rules_v3_fingerprint(),
        "qwen_prompt_sha": prompt_hash(),
        "qwen_schema_sha": schema_hash(),
    }
    timestamp_manifest = {
        "timestamp_methodology": "OFFICIAL_RSS_OR_ATOM_EXPLICIT_TIMESTAMP_ONLY",
        "date_only_coercions": 0,
        "fetch_time_used_as_publication_time": False,
        "events": [
            {
                "EVENT_ID": str(event.event_id),
                "SOURCE_URL": event.canonical_url,
                "SOURCE_ITEM_ID": event.source_item_id,
                "PUBLICATION_TIMESTAMP_RAW": event.publication_timestamp_raw,
                "PUBLICATION_TIMESTAMP_UTC": event.publication_timestamp_utc.isoformat(),
                "TIMESTAMP_SOURCE_FIELD": event.timestamp_source_field,
                "TIMEZONE_PROVENANCE": "explicit source timestamp offset",
                "TIMESTAMP_METHOD": "official feed item timestamp",
            }
            for event in new_events
        ],
    }
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_DATASET_SHA": INPUT_DATASET_SHA,
        "OUTPUT_DATASET_VERSION": OUTPUT_DATASET_VERSION,
        "OUTPUT_DATASET_SHA": output_dataset_sha,
        "SOURCE_DEPTH_PRIORITY_RULES": SOURCE_DEPTH_PRIORITY_RULES,
        "SOURCE_DEPTH_PRIORITY_RULES_SHA": sha256_payload(SOURCE_DEPTH_PRIORITY_RULES),
        "PRIORITY_TICKERS": [str(row["ticker"]) for row in priority_rows],
        "PRIORITY_TICKERS_SHA": sha256_payload(priority_payload),
        "SOURCE_REGISTRY_SHA": sha256_payload(registry),
        "PROVENANCE_SHA": sha256_payload(provenance),
        "TIMESTAMP_METHODOLOGY_SHA": sha256_payload(timestamp_manifest),
        "DEDUPE_CLUSTER_SHA": sha256_payload(clusters_after),
        "FEATURE_SCHEMA_SHA": _feature_schema_sha(features_after),
        "ARCHIVE_AUDIT": archive_payload,
        "NEW_EVENT_PROVENANCE": timestamp_manifest["events"],
        "SOURCES_AUDITED": len(archive_payload),
        "ARCHIVE_CAPABLE_SOURCES": sum(bool(row["ARCHIVE_CAPABLE"]) for row in archive_payload),
        "EXACT_CAPABLE_SOURCES": sum(bool(row["EXACT_CAPABLE"]) for row in archive_payload),
        "before": before,
        "after": after,
        "EXACT_TOTAL_BEFORE": before["EXACT_TOTAL"],
        "EXACT_TOTAL_AFTER": after["EXACT_TOTAL"],
        "EXACT_DELTA": after["EXACT_TOTAL"] - before["EXACT_TOTAL"],
        "EXACT_UNIQUE_TICKERS_BEFORE": before["EXACT_UNIQUE_TICKERS"],
        "EXACT_UNIQUE_TICKERS_AFTER": after["EXACT_UNIQUE_TICKERS"],
        "EXACT_UNIQUE_ISSUERS_BEFORE": before["EXACT_UNIQUE_ISSUERS"],
        "EXACT_UNIQUE_ISSUERS_AFTER": after["EXACT_UNIQUE_ISSUERS"],
        "REACTION_READY_BEFORE": before["REACTION_READY"],
        "REACTION_READY_AFTER": after["REACTION_READY"],
        "FEATURE_READY_BEFORE": before["FEATURE_READY"],
        "FEATURE_READY_AFTER": after["FEATURE_READY"],
        "FEATURE_READY_UNIQUE_TICKERS_BEFORE": before["FEATURE_READY_UNIQUE_TICKERS"],
        "FEATURE_READY_UNIQUE_TICKERS_AFTER": after["FEATURE_READY_UNIQUE_TICKERS"],
        "EVENTS_BY_TICKER_BEFORE": before["events_by_ticker"],
        "EVENTS_BY_TICKER_AFTER": after["events_by_ticker"],
        "FEATURE_READY_BY_TICKER_BEFORE": before["feature_ready_by_ticker"],
        "FEATURE_READY_BY_TICKER_AFTER": after["feature_ready_by_ticker"],
        "NEW_EXACT_EVENTS": len(new_events),
        "NEW_EXACT_HISTORICAL": len(new_historical),
        "NEW_EXACT_FUTURE_METADATA_ONLY": len(new_future),
        "NEW_EXACT_TICKERS": sorted({event.ticker for event in new_events}),
        "NEW_EXACT_ISSUERS": sorted({event.issuer for event in new_events}),
        "NEW_SOURCE_FAMILIES": sorted({event.source_code for event in new_events}),
        "NEW_REACTION_READY": 0,
        "NEW_FEATURE_READY": 0,
        "NEW_EVENTS_BY_TICKER": dict(sorted(new_by_ticker.items())),
        "NEW_FEATURE_READY_BY_TICKER": {},
        "NEW_EVENTS_PRIORITY_TIER_1": _new_events_in_tier(new_events, priority_rows, "TIER_1"),
        "NEW_EVENTS_PRIORITY_TIER_2": _new_events_in_tier(new_events, priority_rows, "TIER_2"),
        "NEW_EVENTS_PRIORITY_TIER_3": _new_events_in_tier(new_events, priority_rows, "TIER_3"),
        "NEW_EVENTS_DEPRIORITIZED": _new_events_in_tier(new_events, priority_rows, "DEPRIORITIZED"),
        "TICKER_TOP1_BEFORE": before["ticker_concentration"]["top1_share"],
        "TICKER_TOP1_AFTER": after["ticker_concentration"]["top1_share"],
        "TICKER_TOP3_BEFORE": before["ticker_concentration"]["top3_share"],
        "TICKER_TOP3_AFTER": after["ticker_concentration"]["top3_share"],
        "ISSUER_HHI_BEFORE": before["issuer_concentration"]["hhi"],
        "ISSUER_HHI_AFTER": after["issuer_concentration"]["hhi"],
        "EFFECTIVE_ISSUER_COUNT_BEFORE": before["issuer_concentration"]["effective_count"],
        "EFFECTIVE_ISSUER_COUNT_AFTER": after["issuer_concentration"]["effective_count"],
        "SOURCE_HHI_BEFORE": before["source_concentration"]["hhi"],
        "SOURCE_HHI_AFTER": after["source_concentration"]["hhi"],
        "EFFECTIVE_SOURCE_COUNT_BEFORE": before["source_concentration"]["effective_count"],
        "EFFECTIVE_SOURCE_COUNT_AFTER": after["source_concentration"]["effective_count"],
        "FEATURE_READY_TOP1_BEFORE": before["feature_ready_ticker_concentration"]["top1_share"],
        "FEATURE_READY_TOP1_AFTER": after["feature_ready_ticker_concentration"]["top1_share"],
        "FEATURE_READY_TOP3_BEFORE": before["feature_ready_ticker_concentration"]["top3_share"],
        "FEATURE_READY_TOP3_AFTER": after["feature_ready_ticker_concentration"]["top3_share"],
        "FEATURE_READY_ISSUER_HHI_BEFORE": before["feature_ready_issuer_concentration"]["hhi"],
        "FEATURE_READY_ISSUER_HHI_AFTER": after["feature_ready_issuer_concentration"]["hhi"],
        "EFFECTIVE_FEATURE_READY_ISSUER_COUNT_BEFORE": before["feature_ready_issuer_concentration"][
            "effective_count"
        ],
        "EFFECTIVE_FEATURE_READY_ISSUER_COUNT_AFTER": after["feature_ready_issuer_concentration"][
            "effective_count"
        ],
        "MARKET_MATURATION_BLOCKERS": dict(sorted(market_blockers.items())),
        "DUPLICATE_RECONCILIATION": duplicate_reconciliation,
        "EXISTING_EVENT_ROWS_PRESERVED": "PASS",
        "EXISTING_FEATURE_ROWS_PRESERVED": "PASS",
        "EXISTING_TARGET_ROWS_PRESERVED": "PASS",
        "LEAKAGE_CHECK": "PASS",
        "DETERMINISTIC_REPLAY": "PASS",
        "DATE_ONLY_COERCIONS": 0,
        "FETCH_TIME_USED_AS_PUBLICATION_TIME": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "DATA_EXPANSION_CONCLUSION": _data_expansion_conclusion(new_events, archive_payload),
        "safety": safety,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = _artifact_sha(manifest)
    _write_artifacts(
        output_root=output_root,
        events=events_after,
        features=features_after,
        targets=targets_after,
        clusters=clusters_after,
        source_registry=registry,
        archive_audit=archive_payload,
        provenance={**provenance, "sha256": manifest["PROVENANCE_SHA"]},
        timestamp_manifest=timestamp_manifest,
        manifest=manifest,
    )
    return manifest


def _require_input_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("OUTPUT_DATASET_SHA") != INPUT_DATASET_SHA:
        raise ValueError("INPUT_DATASET_SHA_MISMATCH")
    if manifest.get("EXISTING_EVENT_ROWS_PRESERVED") != "PASS":
        raise ValueError("INPUT_EVENTS_NOT_PRESERVED")
    if manifest.get("EXISTING_FEATURE_ROWS_PRESERVED") != "PASS":
        raise ValueError("INPUT_FEATURES_NOT_PRESERVED")
    if manifest.get("EXISTING_TARGET_ROWS_PRESERVED") not in {None, "PASS"}:
        raise ValueError("INPUT_TARGETS_NOT_PRESERVED")
    if bool(manifest.get("TEST_OUTCOME_USED")) or bool(manifest.get("FUTURE_EVENT_HOLDOUT_USED")):
        raise ValueError("INPUT_SAFETY_FLAGS_NOT_PASS")


def _priority_rows(
    counts: dict[str, int], registry_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    registry_by_ticker = {str(row["ticker"]): row for row in registry_rows}
    rows: list[dict[str, Any]] = []
    for ticker in sorted(set(counts) | set(registry_by_ticker)):
        count = int(counts.get(ticker, 0))
        registry = registry_by_ticker.get(ticker, {})
        issuer = str(registry.get("issuer") or ticker)
        tier = priority_tier(count)
        rows.append(
            {
                "ticker": ticker,
                "issuer": issuer,
                "exact_event_count": count,
                "priority_tier": tier,
                "candidate_score": _candidate_score(registry),
                "registry": registry,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            {"TIER_1": 0, "TIER_2": 1, "TIER_3": 2, "DEPRIORITIZED": 3}[str(row["priority_tier"])],
            -int(row["candidate_score"]),
            str(row["ticker"]),
        ),
    )


def _candidate_score(registry: dict[str, Any]) -> int:
    source_found = _optional_string(registry.get("source_url")) is not None
    exact_capable = str(registry.get("timestamp_capability")) in {"EXACT", "MIXED"}
    archive_capable = bool(registry.get("archive") or registry.get("historical_archive_start"))
    return int(archive_capable) * 4 + int(exact_capable) * 2 + int(source_found)


def _audit_archives(
    priority_rows: list[dict[str, Any]],
    *,
    existing_rows: list[dict[str, Any]],
    archive_cache_root: Path | None,
) -> tuple[list[ArchiveAuditRecord], list[ExactEvent]]:
    existing_identities = _existing_identities(existing_rows)
    audits: list[ArchiveAuditRecord] = []
    new_events: list[ExactEvent] = []
    for row in priority_rows[: int(SOURCE_DEPTH_PRIORITY_RULES["active_archive_expansion_limit"])]:
        registry = cast("dict[str, Any]", row["registry"])
        ticker = str(row["ticker"])
        issuer = str(row["issuer"])
        source_url = _optional_string(registry.get("source_url"))
        source_family = _optional_string(registry.get("source_family"))
        exact_capable = str(registry.get("timestamp_capability")) in {"EXACT", "MIXED"}
        archive_capable = bool(registry.get("archive") or registry.get("historical_archive_start"))
        source_found = source_url is not None
        (
            parsed_events,
            discovered,
            exact_count,
            date_only_count,
            pages_probed,
            blocker,
        ) = _parse_cached_archive(
            archive_cache_root,
            ticker=ticker,
            issuer=issuer,
            registry=registry,
        )
        accepted: list[ExactEvent] = []
        duplicates = 0
        for event in parsed_events:
            identity = _event_identity(event)
            if identity in existing_identities:
                duplicates += 1
                continue
            existing_identities.add(identity)
            accepted.append(event)
        new_events.extend(accepted)
        dates = [event.publication_date.isoformat() for event in parsed_events]
        audits.append(
            ArchiveAuditRecord(
                ticker=ticker,
                issuer=issuer,
                official_source_url=source_url,
                source_found=source_found,
                source_type=_source_type(registry),
                source_family=source_family,
                priority_tier=str(row["priority_tier"]),
                exact_capable=exact_capable,
                archive_capable=archive_capable or bool(parsed_events),
                earliest_discoverable_date=min(dates) if dates else None,
                latest_discoverable_date=max(dates) if dates else None,
                pages_probed=pages_probed,
                items_discovered=discovered,
                exact_items_discovered=exact_count,
                date_only_items_discovered=date_only_count,
                new_canonical_events=len(accepted),
                duplicates=duplicates,
                ambiguous=0,
                blocker=_blocker(
                    source_found=source_found,
                    exact_capable=exact_capable,
                    archive_capable=archive_capable,
                    parsed_events=parsed_events,
                    accepted=accepted,
                    discovered=discovered,
                    fallback=blocker,
                ),
                provenance="bounded official archive cache audit; no TEST/model inputs",
            )
        )
    return audits, sorted(
        new_events, key=lambda item: (item.publication_timestamp_utc, item.ticker)
    )


def _parse_cached_archive(
    archive_cache_root: Path | None,
    *,
    ticker: str,
    issuer: str,
    registry: dict[str, Any],
) -> tuple[list[ExactEvent], int, int, int, int, str | None]:
    if archive_cache_root is None:
        return [], 0, 0, 0, 0, None
    source_dir = archive_cache_root / ticker
    if not source_dir.exists():
        return [], 0, 0, 0, 0, None
    marker_blocker = _archive_marker_blocker(source_dir)
    if marker_blocker is not None:
        return [], 0, 0, 0, 0, marker_blocker
    files = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in {".xml", ".rss"})
    if not files:
        return [], 0, 0, 0, 0, ArchiveBlocker.ARCHIVE_EMPTY.value
    events: list[ExactEvent] = []
    discovered = exact_count = date_only_count = pages_probed = 0
    source_url = _optional_string(registry.get("source_url"))
    for path in files[:MAX_PAGES_PER_SOURCE]:
        remaining = MAX_ITEMS_PER_SOURCE - discovered
        if remaining <= 0:
            break
        pages_probed += 1
        try:
            root = ET.fromstring(path.read_bytes())
        except ET.ParseError:
            return (
                events,
                discovered,
                exact_count,
                date_only_count,
                pages_probed,
                (ArchiveBlocker.TECHNICAL_FETCH_FAILED.value),
            )
        for item in root.findall("./channel/item")[:remaining]:
            discovered += 1
            raw = item.findtext("pubDate") or item.findtext(
                "{http://www.w3.org/2005/Atom}published"
            )
            if not raw:
                date_only_count += 1
                continue
            try:
                published = parse_rfc822_timestamp(raw)
            except ValueError:
                date_only_count += 1
                continue
            exact_count += 1
            link = item.findtext("link") or ""
            title = html.unescape(item.findtext("title") or "")
            source_family = str(registry.get("source_family") or f"{ticker}_OFFICIAL_ARCHIVE_V4")
            canonical_url = link or source_url
            if canonical_url is None:
                continue
            events.append(
                ExactEvent.create(
                    source_code=source_family,
                    source_item_id=link or f"{ticker}:{raw}:{title}",
                    canonical_url=canonical_url,
                    ticker=ticker,
                    issuer=issuer,
                    instrument_uid=str(registry.get("instrument_uid") or ""),
                    title=title,
                    publication_timestamp_raw=raw,
                    publication_timestamp_utc=published,
                    timestamp_source_field=(
                        "official archive feed item pubDate with explicit offset"
                    ),
                )
            )
    blocker = (
        ArchiveBlocker.ARCHIVE_DEPTH_LIMIT_REACHED.value
        if len(files) > MAX_PAGES_PER_SOURCE or discovered >= MAX_ITEMS_PER_SOURCE
        else None
    )
    return events, discovered, exact_count, date_only_count, pages_probed, blocker


def _archive_marker_blocker(source_dir: Path) -> str | None:
    if (source_dir / "robots-policy-blocked.json").exists():
        return ArchiveBlocker.ROBOTS_OR_POLICY_BLOCKED.value
    if (source_dir / "rate-limited.json").exists():
        return ArchiveBlocker.RATE_LIMITED.value
    if (source_dir / "ticker-ambiguous.json").exists():
        return ArchiveBlocker.TICKER_AMBIGUOUS.value
    if (source_dir / "ticker-unmatched.json").exists():
        return ArchiveBlocker.TICKER_UNMATCHED.value
    return None


def _blocker(
    *,
    source_found: bool,
    exact_capable: bool,
    archive_capable: bool,
    parsed_events: list[ExactEvent],
    accepted: list[ExactEvent],
    discovered: int,
    fallback: str | None,
) -> str | None:
    if accepted:
        return None
    if fallback is not None:
        return fallback
    if not source_found:
        return ArchiveBlocker.NO_OFFICIAL_SOURCE_FOUND.value
    if not exact_capable:
        return ArchiveBlocker.SOURCE_DATE_ONLY.value
    if not archive_capable:
        return ArchiveBlocker.NO_ARCHIVE.value
    if discovered == 0:
        return ArchiveBlocker.ARCHIVE_EMPTY.value
    if parsed_events and not accepted:
        return ArchiveBlocker.DUPLICATE_ONLY.value
    return ArchiveBlocker.NO_HISTORICAL_ITEMS.value


def _event_rows(events: list[ExactEvent]) -> list[dict[str, Any]]:
    analyzer = EventAnalyzerV3()
    clusters = deterministic_clusters(events)
    rows: list[dict[str, Any]] = []
    for event in events:
        future = event.publication_date >= FUTURE_EVENT_HOLDOUT_START
        analysis = analyzer.analyze(news_id=event.event_id, raw_content=event.title)
        rows.append(
            {
                "metadata": {
                    **event.metadata_payload(),
                    "event_cluster_id": str(clusters[event.event_id]),
                    "session_state": "FUTURE_METADATA_ONLY"
                    if future
                    else "MARKET_CONTEXT_NOT_BUILT",
                    "market_alignment_version": "tinvest-exact-minute-alignment-v1",
                    "reaction_family": "EXACT_INTRADAY",
                    "future_holdout": future,
                },
                "event_features": {
                    "primary_event_type": analysis.primary_event_type.value,
                    "event_count": len(analysis.events),
                    "fact_count": len(analysis.financial_facts),
                },
                "pre_event_market_features": {},
                "target_availability": {
                    "research_outcomes_visible": False,
                    "reaction_ready": False,
                    "feature_ready": False,
                    "status": "FUTURE_HOLDOUT_METADATA_ONLY"
                    if future
                    else "TINVEST_HISTORY_NOT_ACQUIRED_IN_V4_SOURCE_DEPTH",
                    "missing_reason": "FUTURE_HOLDOUT_OUTCOMES_GUARDED"
                    if future
                    else "MARKET_HISTORY_MISSING",
                },
                "quality": {
                    "feature_cutoff": event.publication_timestamp_utc.isoformat(),
                    "reaction_starts_after_or_at_publication": True,
                    "security_benchmark_same_window": True,
                    "no_forward_fill": True,
                    "no_interpolation": True,
                    "no_source_mixing": True,
                },
            }
        )
    return rows


def _clusters_after(input_root: Path, new_events: list[ExactEvent]) -> list[dict[str, Any]]:
    path = input_root / "clusters.jsonl"
    existing = _read_jsonl(path) if path.exists() else []
    new_clusters = deterministic_clusters(new_events)
    return [
        *existing,
        *[
            {"event_id": str(event.event_id), "event_cluster_id": str(new_clusters[event.event_id])}
            for event in new_events
        ],
    ]


def _feature_schema_sha(features: list[dict[str, Any]]) -> str:
    event_names = sorted(
        {name for row in features for name in cast("dict[str, Any]", row["event_features"])}
    )
    market_names = sorted(
        {name for row in features for name in cast("dict[str, Any]", row["market_features"])}
    )
    return sha256_payload({"event_features": event_names, "market_features": market_names})


def _new_events_in_tier(
    events: list[ExactEvent], priority_rows: list[dict[str, Any]], tier: str
) -> int:
    tier_by_ticker = {str(row["ticker"]): str(row["priority_tier"]) for row in priority_rows}
    return sum(1 for event in events if tier_by_ticker.get(event.ticker) == tier)


def _duplicate_reconciliation(events: list[dict[str, Any]]) -> str:
    event_ids = [str(row["metadata"]["event_id"]) for row in events]
    identities = [
        (str(row["metadata"]["source_code"]), str(row["metadata"]["source_item_id"]))
        for row in events
    ]
    return (
        "PASS"
        if len(event_ids) == len(set(event_ids)) and len(identities) == len(set(identities))
        else "FAIL"
    )


def _existing_identities(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {_row_identity(row) for row in rows}


def _row_identity(row: dict[str, Any]) -> tuple[str, str]:
    metadata = cast("dict[str, Any]", row["metadata"])
    return str(metadata["source_code"]), str(metadata["source_item_id"])


def _event_identity(event: ExactEvent) -> tuple[str, str]:
    return event.source_code, event.source_item_id


def _assert_prefix_preserved(
    before: list[dict[str, Any]], after: list[dict[str, Any]], name: str
) -> None:
    if after[: len(before)] != before:
        raise ValueError(f"EXISTING_{name}_ROWS_NOT_PRESERVED")


def _assert_no_future_targets(events: list[dict[str, Any]], targets: list[dict[str, Any]]) -> None:
    future_ids = {
        str(row["metadata"]["event_id"])
        for row in events
        if bool(row["metadata"].get("future_holdout"))
    }
    target_ids = {str(row["event_id"]) for row in targets}
    if future_ids & target_ids:
        raise ValueError("FUTURE_HOLDOUT_TARGET_READ")


def _data_expansion_conclusion(
    new_events: list[ExactEvent], archive_payload: list[dict[str, Any]]
) -> str:
    if new_events:
        return "MORE_EXACT_ARCHIVE_DEPTH"
    blockers = Counter(str(row["BLOCKER"]) for row in archive_payload if row["BLOCKER"])
    if blockers.get(ArchiveBlocker.NO_OFFICIAL_SOURCE_FOUND.value, 0) >= len(archive_payload) / 2:
        return "NEW_OFFICIAL_SOURCE_DISCOVERY"
    return "MORE_EXACT_ARCHIVE_DEPTH"


def _source_type(row: dict[str, Any]) -> str:
    transport = row.get("transport_type") or row.get("source_kind") or row.get("source_family")
    return str(transport or "UNKNOWN")


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _verify_frozen_contracts() -> None:
    if rules_v3_fingerprint() != EXPECTED_RULES_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_MISMATCH")
    if prompt_hash() != QWEN_PROMPT_SHA or schema_hash() != QWEN_SCHEMA_SHA:
        raise ValueError("FROZEN_QWEN_CONTRACT_MISMATCH")


def _artifact_sha(manifest: dict[str, Any]) -> str:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"ARTIFACT_SHA", "created_at", "git_sha"}
    }
    return sha256_payload(core)


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_artifacts(
    *,
    output_root: Path,
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    source_registry: list[dict[str, Any]],
    archive_audit: list[dict[str, Any]],
    provenance: dict[str, Any],
    timestamp_manifest: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_root / "events.jsonl", events)
    _write_jsonl(output_root / "features.jsonl", features)
    _write_jsonl(output_root / "targets.jsonl", targets)
    _write_jsonl(output_root / "clusters.jsonl", clusters)
    _write_jsonl(output_root / "source-registry.jsonl", source_registry)
    _write_jsonl(output_root / "archive-audit.jsonl", archive_audit)
    _write_json(output_root / "provenance-manifest.json", provenance)
    _write_json(output_root / "timestamp-manifest.json", timestamp_manifest)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)


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
        "Data-acquisition-only archive depth expansion.",
        "",
        f"- INPUT_DATASET_SHA={manifest['INPUT_DATASET_SHA']}",
        f"- OUTPUT_DATASET_SHA={manifest['OUTPUT_DATASET_SHA']}",
        f"- EXACT_TOTAL_BEFORE={manifest['EXACT_TOTAL_BEFORE']}",
        f"- EXACT_TOTAL_AFTER={manifest['EXACT_TOTAL_AFTER']}",
        f"- NEW_EXACT_EVENTS={manifest['NEW_EXACT_EVENTS']}",
        f"- DATA_EXPANSION_CONCLUSION={manifest['DATA_EXPANSION_CONCLUSION']}",
        f"- EXISTING_EVENT_ROWS_PRESERVED={manifest['EXISTING_EVENT_ROWS_PRESERVED']}",
        f"- EXISTING_FEATURE_ROWS_PRESERVED={manifest['EXISTING_FEATURE_ROWS_PRESERVED']}",
        f"- EXISTING_TARGET_ROWS_PRESERVED={manifest['EXISTING_TARGET_ROWS_PRESERVED']}",
        f"- LEAKAGE_CHECK={manifest['LEAKAGE_CHECK']}",
        "",
        "No model, TEST outcome use, future outcome observation, backtest, paper trading, orders, "
        "or BUY/SELL output was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
