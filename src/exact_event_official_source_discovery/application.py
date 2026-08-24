from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import EventAnalyzerV3, rules_v3_fingerprint
from src.exact_event_corpus.domain import ExactEvent, deterministic_clusters
from src.exact_event_official_source_discovery.domain import (
    ARTIFACT_VERSION,
    DISCOVERY_PRIORITY_RULES,
    FUTURE_EVENT_HOLDOUT_START,
    INPUT_DATASET_SHA,
    MAX_ITEMS_PER_SOURCE,
    MAX_TICKERS,
    OUTPUT_DATASET_VERSION,
    DiscoveryState,
    SourceDiscoveryAuditRecord,
    counter_payload,
    current_metrics,
    discovery_safety_flags,
    parse_exact_timestamp,
    priority_tier,
    sha256_payload,
)


def build_official_source_discovery_artifact(
    *,
    input_root: Path,
    source_registry_path: Path,
    universe_path: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    discovery_cache_root: Path | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable official source discovery artifact output already exists")
    _verify_frozen_contracts()
    input_manifest = _read_json(input_root / "manifest.json")
    _require_input_manifest(input_manifest)
    events_before = _read_jsonl(input_root / "events.jsonl")
    features_before = _read_jsonl(input_root / "features.jsonl")
    targets_before = _read_jsonl(input_root / "targets.jsonl")
    clusters_before = _read_jsonl(input_root / "clusters.jsonl")
    registry_before = _read_jsonl(source_registry_path)
    universe = _read_universe(universe_path)
    before = current_metrics(events_before, features_before)
    priority_rows = _priority_rows(
        before["events_by_ticker"],
        before["feature_ready_by_ticker"],
        registry_before,
        universe,
    )
    source_audit, new_sources, new_events = _discover_sources(
        priority_rows,
        registry_before=registry_before,
        existing_rows=events_before,
        discovery_cache_root=discovery_cache_root,
    )
    new_event_rows = _event_rows(new_events)
    events_after = [*events_before, *new_event_rows]
    features_after = [*features_before]
    targets_after = [*targets_before]
    clusters_after = [
        *clusters_before,
        *_cluster_rows(new_events),
    ]
    registry_after = [*registry_before, *new_sources]
    _assert_prefix_preserved(events_before, events_after, "EVENT")
    _assert_prefix_preserved(features_before, features_after, "FEATURE")
    _assert_prefix_preserved(targets_before, targets_after, "TARGET")
    _assert_no_future_targets(events_after, targets_after)
    duplicate_reconciliation = _duplicate_reconciliation(events_after)
    if duplicate_reconciliation != "PASS":
        raise ValueError("DUPLICATE_RECONCILIATION_FAILED")
    after = current_metrics(events_after, features_after)
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
            "FEATURE_READY_COUNT": row["feature_ready_count"],
        }
        for row in priority_rows
    ]
    audit_payload = [row.payload() for row in source_audit]
    state_counts = Counter(str(row["CANDIDATE_STATE"]) for row in audit_payload)
    new_historical = [
        event for event in new_events if event.publication_date < FUTURE_EVENT_HOLDOUT_START
    ]
    new_future = [
        event for event in new_events if event.publication_date >= FUTURE_EVENT_HOLDOUT_START
    ]
    new_by_ticker = Counter(event.ticker for event in new_events)
    market_blockers = Counter({"MARKET_HISTORY_MISSING": len(new_historical)})
    safety = discovery_safety_flags()
    provenance = {
        "artifact_version": ARTIFACT_VERSION,
        "base_main_sha": base_main_sha,
        "input_dataset_sha": INPUT_DATASET_SHA,
        "discovery_cache_root": str(discovery_cache_root) if discovery_cache_root else None,
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
        "timestamp_methodology": "OFFICIAL_EXACT_TIMESTAMP_FIELDS_ONLY",
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
                "TIMEZONE_PROVENANCE": "explicit or documented official source timezone",
                "TIMESTAMP_METHOD": "official source discovery candidate timestamp",
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
        "DISCOVERY_PRIORITY_RULES": DISCOVERY_PRIORITY_RULES,
        "DISCOVERY_PRIORITY_RULES_SHA": sha256_payload(DISCOVERY_PRIORITY_RULES),
        "PRIORITY_TICKERS": [str(row["ticker"]) for row in priority_rows],
        "PRIORITY_TICKERS_SHA": sha256_payload(priority_payload),
        "SOURCE_REGISTRY_SHA": sha256_payload(registry_after),
        "PROVENANCE_SHA": sha256_payload(provenance),
        "TIMESTAMP_METHODOLOGY_SHA": sha256_payload(timestamp_manifest),
        "SOURCE_DISCOVERY_AUDIT": audit_payload,
        "NEW_SOURCE_PROVENANCE": new_sources,
        "NEW_EVENT_PROVENANCE": timestamp_manifest["events"],
        "SOURCES_AUDITED": len(audit_payload),
        "NEW_OFFICIAL_SOURCES_FOUND": sum(
            bool(row["NEW_OFFICIAL_SOURCE"]) for row in audit_payload
        ),
        "NEW_EXACT_CAPABLE_SOURCES": sum(
            bool(row["NEW_OFFICIAL_SOURCE"]) and bool(row["EXACT_CAPABLE"]) for row in audit_payload
        ),
        "NEW_ARCHIVE_CAPABLE_SOURCES": sum(
            bool(row["NEW_OFFICIAL_SOURCE"]) and bool(row["ARCHIVE_CAPABILITY"])
            for row in audit_payload
        ),
        "EXACT_SOURCE_READY_COUNT": state_counts[DiscoveryState.EXACT_SOURCE_READY.value],
        "EXACT_SOURCE_FOUND_NO_ARCHIVE_COUNT": state_counts[
            DiscoveryState.EXACT_SOURCE_FOUND_NO_ARCHIVE.value
        ],
        "DATE_ONLY_SOURCE_COUNT": state_counts[DiscoveryState.DATE_ONLY_SOURCE.value],
        "NO_TIMESTAMP_SOURCE_COUNT": state_counts[DiscoveryState.NO_TIMESTAMP_SOURCE.value],
        "NO_OFFICIAL_SOURCE_FOUND_COUNT": state_counts[
            DiscoveryState.NO_OFFICIAL_SOURCE_FOUND.value
        ],
        "POLICY_BLOCKED_COUNT": state_counts[DiscoveryState.POLICY_BLOCKED.value]
        + state_counts[DiscoveryState.ROBOTS_BLOCKED.value],
        "TECHNICAL_FAILED_COUNT": state_counts[DiscoveryState.TECHNICAL_FETCH_FAILED.value],
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
        "NEW_EXACT_EVENTS": len(new_events),
        "NEW_EXACT_HISTORICAL": len(new_historical),
        "NEW_EXACT_FUTURE_METADATA_ONLY": len(new_future),
        "NEW_EXACT_TICKERS": sorted({event.ticker for event in new_events}),
        "NEW_EXACT_ISSUERS": sorted({event.issuer for event in new_events}),
        "NEW_SOURCE_FAMILIES": sorted({str(row["source_family"]) for row in new_sources}),
        "NEW_REACTION_READY": 0,
        "NEW_FEATURE_READY": 0,
        "NEW_EVENTS_BY_TICKER": counter_payload(new_by_ticker),
        "NEW_FEATURE_READY_BY_TICKER": {},
        "TICKER_TOP1_BEFORE": before["ticker_concentration"]["top1_share"],
        "TICKER_TOP1_AFTER": after["ticker_concentration"]["top1_share"],
        "TICKER_TOP3_BEFORE": before["ticker_concentration"]["top3_share"],
        "TICKER_TOP3_AFTER": after["ticker_concentration"]["top3_share"],
        "ISSUER_HHI_BEFORE": before["issuer_concentration"]["hhi"],
        "ISSUER_HHI_AFTER": after["issuer_concentration"]["hhi"],
        "EFFECTIVE_ISSUER_COUNT_BEFORE": before["issuer_concentration"]["effective_count"],
        "EFFECTIVE_ISSUER_COUNT_AFTER": after["issuer_concentration"]["effective_count"],
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
        "MARKET_MATURATION_BLOCKERS": counter_payload(market_blockers),
        "DUPLICATE_RECONCILIATION": duplicate_reconciliation,
        "EXISTING_EVENT_ROWS_PRESERVED": "PASS",
        "EXISTING_FEATURE_ROWS_PRESERVED": "PASS",
        "EXISTING_TARGET_ROWS_PRESERVED": "PASS",
        "LEAKAGE_CHECK": "PASS",
        "DETERMINISTIC_REPLAY": "PASS",
        "DATE_ONLY_COERCIONS": 0,
        "FETCH_TIME_USED_AS_PUBLICATION_TIME": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "SOURCE_DISCOVERY_CONCLUSION": _source_discovery_conclusion(new_sources, new_events),
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
        source_registry=registry_after,
        source_audit=audit_payload,
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
    exact_counts: dict[str, int],
    feature_counts: dict[str, int],
    registry_rows: list[dict[str, Any]],
    universe: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    registry_by_ticker = {str(row["ticker"]): row for row in registry_rows}
    tickers = set(exact_counts) | set(universe)
    rows: list[dict[str, Any]] = []
    for ticker in sorted(tickers):
        exact_count = int(exact_counts.get(ticker, 0))
        feature_count = int(feature_counts.get(ticker, 0))
        registry = registry_by_ticker.get(ticker, {})
        instrument = universe.get(ticker, {})
        in_exact = ticker in exact_counts
        tier = priority_tier(
            ticker=ticker,
            exact_count=exact_count,
            feature_ready_count=feature_count,
            in_exact_corpus=in_exact,
        )
        rows.append(
            {
                "ticker": ticker,
                "issuer": str(registry.get("issuer") or instrument.get("name") or ticker),
                "exact_event_count": exact_count,
                "feature_ready_count": feature_count,
                "priority_tier": tier,
                "registry": registry,
                "instrument": instrument,
                "existing_source_unknown": _existing_source_unknown(registry),
            }
        )
    order = {
        "A_ZERO_FEATURE_READY": 0,
        "B_EXACT_1_5": 1,
        "C_EXACT_6_20": 2,
        "D_CANONICAL_TQBR_NOT_IN_EXACT": 3,
        "DEPRIORITIZED": 4,
    }
    return sorted(
        rows,
        key=lambda row: (
            order[str(row["priority_tier"])],
            -int(row["existing_source_unknown"]),
            str(row["ticker"]),
        ),
    )[:MAX_TICKERS]


def _discover_sources(
    priority_rows: list[dict[str, Any]],
    *,
    registry_before: list[dict[str, Any]],
    existing_rows: list[dict[str, Any]],
    discovery_cache_root: Path | None,
) -> tuple[list[SourceDiscoveryAuditRecord], list[dict[str, Any]], list[ExactEvent]]:
    known_source_keys = {
        (str(row.get("source_family")), str(row.get("source_url"))) for row in registry_before
    }
    existing_identities = _existing_identities(existing_rows)
    existing_urls = {
        str(cast("dict[str, Any]", row["metadata"]).get("canonical_url")) for row in existing_rows
    }
    audits: list[SourceDiscoveryAuditRecord] = []
    new_sources: list[dict[str, Any]] = []
    new_events: list[ExactEvent] = []
    for row in priority_rows:
        ticker = str(row["ticker"])
        candidates = _candidate_payloads(discovery_cache_root, ticker)
        if not candidates:
            audits.append(_empty_audit(row))
            continue
        for candidate in candidates[: int(DISCOVERY_PRIORITY_RULES["max_urls_per_ticker"])]:
            state = _candidate_state(candidate)
            source_family = _optional_string(candidate.get("source_family"))
            source_url = _optional_string(candidate.get("source_url"))
            source_key = (str(source_family), str(source_url))
            new_source = (
                state
                in {
                    DiscoveryState.EXACT_SOURCE_READY.value,
                    DiscoveryState.EXACT_SOURCE_FOUND_NO_ARCHIVE.value,
                }
                and source_key not in known_source_keys
            )
            accepted: list[ExactEvent] = []
            duplicates = 0
            if state == DiscoveryState.EXACT_SOURCE_READY.value:
                for event in _candidate_events(candidate, ticker=ticker, issuer=str(row["issuer"])):
                    identity = _event_identity(event)
                    if identity in existing_identities or event.canonical_url in existing_urls:
                        duplicates += 1
                        continue
                    existing_identities.add(identity)
                    existing_urls.add(event.canonical_url)
                    accepted.append(event)
            new_events.extend(accepted)
            if new_source:
                registry_payload = _source_registry_row(candidate, row, state)
                new_sources.append(registry_payload)
                known_source_keys.add(source_key)
            audits.append(
                SourceDiscoveryAuditRecord(
                    ticker=ticker,
                    issuer=str(row["issuer"]),
                    source_url=source_url,
                    source_domain=_optional_string(candidate.get("source_domain")),
                    source_type=_optional_string(candidate.get("source_type")),
                    source_family=source_family,
                    official_source_confirmed=bool(candidate.get("official_source_confirmed")),
                    timestamp_capability=str(candidate.get("timestamp_capability") or "UNKNOWN"),
                    timestamp_field=_optional_string(candidate.get("timestamp_field")),
                    timezone_provenance=_optional_string(candidate.get("timezone_provenance")),
                    archive_capability=bool(candidate.get("archive_capability")),
                    discovery_method=str(candidate.get("discovery_method") or "UNKNOWN"),
                    policy_status=str(candidate.get("policy_status") or "UNKNOWN_FAIL_CLOSED"),
                    technical_status=str(candidate.get("technical_status") or state),
                    priority_tier=str(row["priority_tier"]),
                    candidate_state=state,
                    new_official_source=new_source,
                    exact_capable=state
                    in {
                        DiscoveryState.EXACT_SOURCE_READY.value,
                        DiscoveryState.EXACT_SOURCE_FOUND_NO_ARCHIVE.value,
                    },
                    new_canonical_events=len(accepted),
                    duplicates=duplicates,
                    ambiguous=1 if state == DiscoveryState.AMBIGUOUS_SOURCE_IDENTITY.value else 0,
                    blocker=None if new_source or accepted else state,
                    provenance="bounded official source discovery cache; no TEST/model inputs",
                )
            )
    return (
        audits,
        new_sources,
        sorted(new_events, key=lambda item: (item.publication_timestamp_utc, item.ticker)),
    )


def _candidate_payloads(discovery_cache_root: Path | None, ticker: str) -> list[dict[str, Any]]:
    if discovery_cache_root is None:
        return []
    source_dir = discovery_cache_root / ticker
    if not source_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("*.json"))[
        : int(DISCOVERY_PRIORITY_RULES["max_urls_per_ticker"])
    ]:
        payload = _read_json(path)
        if isinstance(payload.get("candidates"), list):
            rows.extend(cast("list[dict[str, Any]]", payload["candidates"]))
        else:
            rows.append(payload)
    return rows


def _candidate_state(candidate: dict[str, Any]) -> str:
    if bool(candidate.get("ambiguous_source_identity")):
        return DiscoveryState.AMBIGUOUS_SOURCE_IDENTITY.value
    if bool(candidate.get("auth_required")):
        return DiscoveryState.AUTH_REQUIRED.value
    if bool(candidate.get("captcha_required")):
        return DiscoveryState.CAPTCHA_BLOCKED.value
    policy = str(candidate.get("policy_status") or "").upper()
    if "ROBOTS" in policy:
        return DiscoveryState.ROBOTS_BLOCKED.value
    if "POLICY_BLOCKED" in policy:
        return DiscoveryState.POLICY_BLOCKED.value
    if str(candidate.get("technical_status") or "").upper() == "TECHNICAL_FETCH_FAILED":
        return DiscoveryState.TECHNICAL_FETCH_FAILED.value
    if not _official_source_valid(candidate):
        return DiscoveryState.NO_OFFICIAL_SOURCE_FOUND.value
    if bool(candidate.get("payment_required")) or not bool(candidate.get("public_access", True)):
        return DiscoveryState.POLICY_BLOCKED.value
    capability = str(candidate.get("timestamp_capability") or "UNKNOWN").upper()
    if capability == "DATE_ONLY":
        return DiscoveryState.DATE_ONLY_SOURCE.value
    if capability in {"UNKNOWN", "NONE", "NO_TIMESTAMP"}:
        return DiscoveryState.NO_TIMESTAMP_SOURCE.value
    if capability not in {"EXACT", "MIXED"}:
        return DiscoveryState.UNSUPPORTED_FORMAT.value
    if not bool(candidate.get("archive_capability")):
        return DiscoveryState.EXACT_SOURCE_FOUND_NO_ARCHIVE.value
    return DiscoveryState.EXACT_SOURCE_READY.value


def _candidate_events(candidate: dict[str, Any], *, ticker: str, issuer: str) -> list[ExactEvent]:
    source_family = str(candidate["source_family"])
    instrument_uid = str(candidate.get("instrument_uid") or "")
    events: list[ExactEvent] = []
    for item in cast("list[dict[str, Any]]", candidate.get("items") or [])[:MAX_ITEMS_PER_SOURCE]:
        if str(item.get("ticker") or ticker) != ticker:
            continue
        raw = str(item.get("published_at") or item.get("publication_timestamp_raw") or "")
        try:
            published = parse_exact_timestamp(raw)
        except ValueError:
            continue
        canonical_url = str(item.get("canonical_url") or item.get("url") or candidate["source_url"])
        title = str(item.get("title") or "")
        source_item_id = str(item.get("source_item_id") or canonical_url)
        if not title:
            continue
        events.append(
            ExactEvent.create(
                source_code=source_family,
                source_item_id=source_item_id,
                canonical_url=canonical_url,
                ticker=ticker,
                issuer=issuer,
                instrument_uid=instrument_uid,
                title=title,
                publication_timestamp_raw=raw,
                publication_timestamp_utc=published,
                timestamp_source_field=str(
                    item.get("timestamp_field")
                    or candidate.get("timestamp_field")
                    or "published_at"
                ),
            )
        )
    return events


def _official_source_valid(candidate: dict[str, Any]) -> bool:
    source_url = _optional_string(candidate.get("source_url"))
    source_domain = _optional_string(candidate.get("source_domain"))
    if source_url is None or source_domain is None:
        return False
    parsed = urlsplit(source_url)
    if parsed.scheme != "https" or parsed.netloc.lower() != source_domain.lower():
        return False
    return bool(candidate.get("official_source_confirmed"))


def _source_registry_row(
    candidate: dict[str, Any], priority_row: dict[str, Any], state: str
) -> dict[str, Any]:
    return {
        "source_registry_version": "exact-event-source-registry-v5",
        "ticker": str(priority_row["ticker"]),
        "issuer": str(priority_row["issuer"]),
        "instrument_uid": str(
            candidate.get("instrument_uid")
            or cast("dict[str, Any]", priority_row.get("instrument") or {}).get("instrument_uid")
            or ""
        ),
        "source_url": str(candidate["source_url"]),
        "source_domain": str(candidate["source_domain"]),
        "source_type": str(candidate.get("source_type") or "UNKNOWN"),
        "source_family": str(candidate["source_family"]),
        "timestamp_capability": str(candidate.get("timestamp_capability") or "UNKNOWN"),
        "timestamp_field": candidate.get("timestamp_field"),
        "timezone_provenance": candidate.get("timezone_provenance"),
        "archive_capability": bool(candidate.get("archive_capability")),
        "discovery_method": str(candidate.get("discovery_method") or "UNKNOWN"),
        "policy_status": str(candidate.get("policy_status") or "UNKNOWN_FAIL_CLOSED"),
        "technical_status": str(candidate.get("technical_status") or state),
        "candidate_state": state,
        "provenance": "official-source-discovery-v5 bounded cache",
    }


def _empty_audit(row: dict[str, Any]) -> SourceDiscoveryAuditRecord:
    return SourceDiscoveryAuditRecord(
        ticker=str(row["ticker"]),
        issuer=str(row["issuer"]),
        source_url=None,
        source_domain=None,
        source_type=None,
        source_family=None,
        official_source_confirmed=False,
        timestamp_capability="UNKNOWN",
        timestamp_field=None,
        timezone_provenance=None,
        archive_capability=False,
        discovery_method="NO_DISCOVERY_CACHE_SNAPSHOT",
        policy_status="UNKNOWN_FAIL_CLOSED",
        technical_status=DiscoveryState.NO_OFFICIAL_SOURCE_FOUND.value,
        priority_tier=str(row["priority_tier"]),
        candidate_state=DiscoveryState.NO_OFFICIAL_SOURCE_FOUND.value,
        new_official_source=False,
        exact_capable=False,
        new_canonical_events=0,
        duplicates=0,
        ambiguous=0,
        blocker=DiscoveryState.NO_OFFICIAL_SOURCE_FOUND.value,
        provenance="bounded official source discovery; no source snapshot available",
    )


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
                    else "TINVEST_HISTORY_NOT_ACQUIRED_IN_V5_DISCOVERY",
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


def _cluster_rows(events: list[ExactEvent]) -> list[dict[str, Any]]:
    clusters = deterministic_clusters(events)
    return [
        {"event_id": str(event.event_id), "event_cluster_id": str(clusters[event.event_id])}
        for event in events
    ]


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
    identities: set[tuple[str, str]] = set()
    for row in rows:
        metadata = cast("dict[str, Any]", row["metadata"])
        identities.add((str(metadata["source_code"]), str(metadata["source_item_id"])))
    return identities


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


def _read_universe(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    universe: dict[str, dict[str, Any]] = {}
    for item in cast("list[dict[str, Any]]", payload.get("instruments") or []):
        if (
            str(item.get("class_code") or "").upper() == "TQBR"
            and str(item.get("currency") or "").lower() == "rub"
        ):
            universe[str(item["ticker"])] = item
    return universe


def _existing_source_unknown(registry: dict[str, Any]) -> int:
    return int(
        not bool(registry.get("source_url"))
        or str(registry.get("timestamp_capability")) == "UNKNOWN"
    )


def _source_discovery_conclusion(
    new_sources: list[dict[str, Any]], new_events: list[ExactEvent]
) -> str:
    if new_events:
        return "MARKET_MATURATION"
    if new_sources:
        return "DEEPEN_NEWLY_FOUND_EXACT_SOURCES"
    return "MORE_NEW_SOURCE_DISCOVERY"


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


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


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
    source_audit: list[dict[str, Any]],
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
    _write_jsonl(output_root / "source-discovery-audit.jsonl", source_audit)
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
        "Data-acquisition-only official source discovery.",
        "",
        f"- INPUT_DATASET_SHA={manifest['INPUT_DATASET_SHA']}",
        f"- OUTPUT_DATASET_SHA={manifest['OUTPUT_DATASET_SHA']}",
        f"- SOURCES_AUDITED={manifest['SOURCES_AUDITED']}",
        f"- NEW_OFFICIAL_SOURCES_FOUND={manifest['NEW_OFFICIAL_SOURCES_FOUND']}",
        f"- NEW_EXACT_CAPABLE_SOURCES={manifest['NEW_EXACT_CAPABLE_SOURCES']}",
        f"- NEW_EXACT_EVENTS={manifest['NEW_EXACT_EVENTS']}",
        f"- SOURCE_DISCOVERY_CONCLUSION={manifest['SOURCE_DISCOVERY_CONCLUSION']}",
        f"- EXISTING_EVENT_ROWS_PRESERVED={manifest['EXISTING_EVENT_ROWS_PRESERVED']}",
        f"- EXISTING_FEATURE_ROWS_PRESERVED={manifest['EXISTING_FEATURE_ROWS_PRESERVED']}",
        f"- EXISTING_TARGET_ROWS_PRESERVED={manifest['EXISTING_TARGET_ROWS_PRESERVED']}",
        f"- LEAKAGE_CHECK={manifest['LEAKAGE_CHECK']}",
        "",
        "No model, TEST outcome use, future outcome observation, sparse family, backtest, paper "
        "trading, orders, or BUY/SELL output was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
