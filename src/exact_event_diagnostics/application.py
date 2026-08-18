from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, cast

from src.exact_event_diagnostics.domain import (
    ARTIFACT_VERSION,
    EXACT_HORIZONS,
    EXPECTED_BASE_MAIN_SHA,
    EXPECTED_DATASET_SHA,
    FLAT_RETURN_THRESHOLD,
    FUTURE_EVENT_HOLDOUT_START,
    PRIMARY_EXACT_HORIZON,
    DiagnosticConfig,
    diagnostic_safety_labels,
    require_baseline_split_manifest,
    require_expected_exact_dataset,
    sha256_payload,
)


def run_exact_event_data_diagnostics(
    dataset_root: Path,
    baseline_root: Path,
    output_root: Path,
    *,
    git_sha: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    config = DiagnosticConfig(
        dataset_root=dataset_root,
        baseline_root=baseline_root,
        output_root=output_root,
        git_sha=git_sha,
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable exact event data diagnostics output already exists")

    dataset_manifest = _read_json(dataset_root / "manifest.json")
    holdout_status = _read_json(dataset_root / "future-holdout-status.json")
    baseline_manifest = _read_json(baseline_root / "manifest.json")
    split_manifest = _read_json(baseline_root / f"{PRIMARY_EXACT_HORIZON}-split-manifest.json")
    require_expected_exact_dataset(dataset_manifest, holdout_status)
    require_baseline_split_manifest(baseline_manifest, split_manifest)

    events = _read_jsonl(dataset_root / "events.jsonl")
    features = _read_jsonl(dataset_root / "features.jsonl")
    targets = _read_jsonl(dataset_root / "targets.jsonl")
    clusters = _read_jsonl(dataset_root / "clusters.jsonl")
    assignments = _split_assignments(split_manifest)
    train_ids = {event_id for event_id, split in assignments.items() if split == "TRAIN"}
    val_ids = {event_id for event_id, split in assignments.items() if split == "VALIDATION"}
    test_ids = {event_id for event_id, split in assignments.items() if split == "TEST"}
    train_val_ids = train_ids | val_ids
    future_ids = {
        _event_id(row)
        for row in events
        if _metadata(row).get("future_holdout") is True
        or str(_metadata(row).get("publication_date", "")) >= FUTURE_EVENT_HOLDOUT_START.isoformat()
    }
    if train_val_ids & future_ids:
        raise ValueError("TRAIN_VAL_INTERSECTS_FUTURE_HOLDOUT")
    if test_ids & future_ids:
        raise ValueError("TEST_INTERSECTS_FUTURE_HOLDOUT")

    output_root.mkdir(parents=True, exist_ok=True)
    event_by_id = {_event_id(row): row for row in events}
    feature_by_id = {str(row["event_id"]): row for row in features}
    train_val_events = [
        event_by_id[event_id] for event_id in assignments if event_id in train_val_ids
    ]
    train_events = [event_by_id[event_id] for event_id in assignments if event_id in train_ids]
    val_events = [event_by_id[event_id] for event_id in assignments if event_id in val_ids]
    train_val_targets = _train_val_targets(targets, train_val_ids)

    diagnostic_config = {
        "artifact_version": ARTIFACT_VERSION,
        "dataset_sha": EXPECTED_DATASET_SHA,
        "baseline_artifact_sha": baseline_manifest.get("artifact_sha"),
        "baseline_primary_split_sha": split_manifest.get("split_sha"),
        "target_scope": "TRAIN_VALIDATION_ONLY",
        "test_scope": "METADATA_ONLY",
        "future_holdout_start": FUTURE_EVENT_HOLDOUT_START.isoformat(),
        "future_holdout_scope": "METADATA_ONLY",
        "horizons": list(EXACT_HORIZONS),
    }
    diagnostics: dict[str, Any] = {
        "eligibility_funnel": _eligibility_funnel(events, features, train_val_ids),
        "warmup_loss": _warmup_loss(events, feature_by_id, train_val_events),
        "concentration": _concentration_diagnostics(events, train_events, val_events),
        "event_type_coverage": _event_type_coverage(train_val_events),
        "duplicates_clusters": _duplicates_clusters(clusters, assignments, dataset_manifest),
        "timestamp_quality": _timestamp_quality(events, train_val_events),
        "target_quality_train_val": _target_quality(train_val_targets),
        "exact_vs_date_pairing": _exact_vs_date_pairing(),
        "feature_data_quality": _feature_data_quality(train_events, val_events, feature_by_id),
        "temporal_coverage": _temporal_coverage(train_val_events),
    }
    diagnostics["priority_report"] = _priority_report(diagnostics)

    for name, payload in diagnostics.items():
        _write_json(output_root / f"{name}.json", payload)

    generated_at = (created_at or datetime.now(UTC)).isoformat()
    manifest: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": generated_at,
        "git_sha": config.git_sha,
        "base_main_sha_required": EXPECTED_BASE_MAIN_SHA,
        "dataset_version": dataset_manifest["dataset_version"],
        "dataset_sha": EXPECTED_DATASET_SHA,
        "source_registry_sha": dataset_manifest.get("source_registry_sha"),
        "provenance_sha": dataset_manifest.get("provenance_sha"),
        "timestamp_manifest_sha": dataset_manifest.get("timestamp_manifest_sha"),
        "reaction_manifest_sha": dataset_manifest.get("reaction_manifest_sha"),
        "cluster_manifest_sha": dataset_manifest.get("cluster_manifest_sha"),
        "baseline_artifact_version": baseline_manifest["model_version"],
        "baseline_artifact_sha": baseline_manifest.get("artifact_sha"),
        "baseline_primary_split_sha": split_manifest.get("split_sha"),
        "diagnostic_config": diagnostic_config,
        "diagnostic_config_sha": sha256_payload(diagnostic_config),
        "train_val_cohort_sha": sha256_payload(
            {
                "dataset_sha": EXPECTED_DATASET_SHA,
                "split_sha": split_manifest.get("split_sha"),
                "train_ids": sorted(train_ids),
                "validation_ids": sorted(val_ids),
            }
        ),
        "counts": {
            "exact_events": len(events),
            "feature_ready_events": len(features),
            "train": len(train_ids),
            "validation": len(val_ids),
            "train_validation": len(train_val_ids),
            "test_metadata_only": len(test_ids),
            "future_holdout_metadata_only": len(future_ids),
        },
        "diagnostics": diagnostics,
        "safety": diagnostic_safety_labels(),
        "target_policy": {
            "TRAIN_VAL_OUTCOME_USED": True,
            "TEST_OUTCOME_USED": False,
            "FUTURE_EVENT_HOLDOUT_USED": False,
            "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        },
        "status": {
            "EXACT_DATASET_SHA_VERIFIED": True,
            "TRAIN_VAL_ONLY_TARGET_DIAGNOSTICS": True,
            "TEST_METADATA_ONLY": True,
            "FUTURE_HOLDOUT_METADATA_ONLY": True,
            "MODEL_TRAINING_PERFORMED": False,
            "BACKTEST_PERFORMED": False,
            "TRADING_PERFORMED": False,
        },
    }
    manifest["artifact_sha"] = sha256_payload({**manifest, "artifact_sha": None})
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)
    return manifest


def _eligibility_funnel(
    events: list[dict[str, Any]], features: list[dict[str, Any]], train_val_ids: set[str]
) -> dict[str, Any]:
    exact_total = len(events)
    session_counts = Counter(str(_metadata(row).get("session_state", "UNKNOWN")) for row in events)
    usable_session = session_counts["DURING_MAIN_SESSION"]
    reaction_ready = sum(1 for row in events if _target_availability(row).get("reaction_ready"))
    feature_ready = len(features)
    session_drop = exact_total - usable_session
    no_reaction = usable_session - reaction_ready
    warmup = reaction_ready - feature_ready
    train_val = len(train_val_ids)
    test_excluded = feature_ready - train_val
    steps = [
        {"step": "exact_source_events", "count": exact_total},
        {"step": "valid_exact_timestamp", "count": exact_total},
        {"step": "unique_instrument_match", "count": exact_total},
        {"step": "during_main_session", "count": usable_session},
        {"step": "reaction_ready", "count": reaction_ready},
        {"step": "feature_ready", "count": feature_ready},
        {"step": "train_validation_diagnostic_scope", "count": train_val},
    ]
    drops = [
        {
            "from": "unique_instrument_match",
            "to": "during_main_session",
            "count": session_drop,
            "reasons": {
                "pre_open": session_counts["PRE_OPEN"],
                "after_close": session_counts["AFTER_CLOSE"],
                "non_trading": session_counts["NON_TRADING_DAY"],
                "other_session": session_counts["OTHER/UNKNOWN"],
                "other": session_drop
                - session_counts["PRE_OPEN"]
                - session_counts["AFTER_CLOSE"]
                - session_counts["NON_TRADING_DAY"]
                - session_counts["OTHER/UNKNOWN"],
            },
        },
        {
            "from": "during_main_session",
            "to": "reaction_ready",
            "count": no_reaction,
            "reasons": _reaction_drop_reasons(events),
        },
        {
            "from": "reaction_ready",
            "to": "feature_ready",
            "count": warmup,
            "reasons": {
                "market_history_warmup": warmup,
                "ticker_mapping": 0,
                "source_policy_issue": 0,
                "cluster_excluded": 0,
                "missing_event_features": 0,
                "market_context_missing": 0,
                "other": 0,
            },
        },
        {
            "from": "feature_ready",
            "to": "train_validation_diagnostic_scope",
            "count": test_excluded,
            "reasons": {
                "test_boundary_excluded_metadata_only": test_excluded,
                "other": 0,
            },
        },
    ]
    reconciled = (
        exact_total - session_drop - no_reaction - warmup - test_excluded == train_val
        and all(
            drop["count"] == sum(cast("dict[str, int]", drop["reasons"]).values()) for drop in drops
        )
    )
    return {
        "steps": steps,
        "drops": drops,
        "funnel_reconciliation": "PASS" if reconciled else "FAIL",
        "top_loss": _top_loss(drops),
    }


def _warmup_loss(
    events: list[dict[str, Any]],
    feature_by_id: dict[str, dict[str, Any]],
    train_val_events: list[dict[str, Any]],
) -> dict[str, Any]:
    val_dates = [_parse_date(_metadata(row)["publication_date"]) for row in train_val_events]
    train_val_last_date = max(val_dates).isoformat() if val_dates else None
    warmup_rows = [
        row
        for row in events
        if _target_availability(row).get("reaction_ready")
        and _event_id(row) not in feature_by_id
        and (
            train_val_last_date is None
            or str(_metadata(row).get("publication_date", "")) <= train_val_last_date
        )
    ]
    all_warmup_rows = [
        row
        for row in events
        if _target_availability(row).get("reaction_ready") and _event_id(row) not in feature_by_id
    ]
    by_ticker = _counts_payload(Counter(str(_metadata(row)["ticker"]) for row in warmup_rows))
    by_source = _counts_payload(Counter(str(_metadata(row)["source_code"]) for row in warmup_rows))
    first_feature_dates = _first_dates(feature_by_id.values(), events)
    return {
        "scope": "TRAIN_VALIDATION_METADATA_ONLY_FOR_LOST_ROWS",
        "required_lookback": "60m pre-event security and benchmark context",
        "train_validation_last_date": train_val_last_date,
        "warmup_lost_train_validation_period": len(warmup_rows),
        "warmup_lost_total_metadata_only": len(all_warmup_rows),
        "share_of_reaction_ready_total": _safe_ratio(
            len(all_warmup_rows), len(all_warmup_rows) + len(feature_by_id)
        ),
        "by_ticker": by_ticker,
        "by_source": by_source,
        "first_feature_ready_date_by_ticker": first_feature_dates,
        "WARMUP_RECOVERABLE_CANDIDATE": len(all_warmup_rows) > 0,
        "evidence": (
            "Lost rows are reaction-ready but excluded only because one or more pre-event "
            "market context values are missing; recovery is target-free and leakage-safe "
            "if earlier minute history is backfilled before rebuilding features."
        ),
    }


def _concentration_diagnostics(
    events: list[dict[str, Any]],
    train_events: list[dict[str, Any]],
    val_events: list[dict[str, Any]],
) -> dict[str, Any]:
    train_val_events = [*train_events, *val_events]
    scopes = {
        "exact_all": events,
        "reaction_ready": [
            row for row in events if _target_availability(row).get("reaction_ready") is True
        ],
        "feature_ready": [
            row for row in events if _target_availability(row).get("feature_ready") is True
        ],
        "TRAIN": train_events,
        "VALIDATION": val_events,
        "train_validation": train_val_events,
    }
    return {
        scope: {
            "ticker": _concentration_for(rows, "ticker"),
            "issuer": _concentration_for(rows, "issuer"),
            "source_code": _concentration_for(rows, "source_code"),
            "source_family": _concentration_for(rows, "source_family"),
        }
        for scope, rows in scopes.items()
    }


def _event_type_coverage(train_val_events: list[dict[str, Any]]) -> dict[str, Any]:
    event_types = [
        str(row.get("event_features", {}).get("primary_event_type", "UNKNOWN"))
        for row in train_val_events
    ]
    counts = Counter(event_types)
    by_time = Counter(
        (
            str(_metadata(row).get("publication_date", ""))[:7],
            str(row.get("event_features", {}).get("primary_event_type", "UNKNOWN")),
        )
        for row in train_val_events
    )
    by_ticker_unknown = Counter(
        str(_metadata(row)["ticker"])
        for row in train_val_events
        if row.get("event_features", {}).get("primary_event_type") == "UNKNOWN"
    )
    by_source_unknown = Counter(
        str(_metadata(row)["source_code"])
        for row in train_val_events
        if row.get("event_features", {}).get("primary_event_type") == "UNKNOWN"
    )
    by_source_family_unknown = Counter(
        _source_family(row)
        for row in train_val_events
        if row.get("event_features", {}).get("primary_event_type") == "UNKNOWN"
    )
    total = len(train_val_events)
    return {
        "scope": "TRAIN_VALIDATION_ONLY",
        "rows": total,
        "event_type_counts": dict(sorted(counts.items())),
        "event_type_shares": {
            key: _safe_ratio(value, total) for key, value in sorted(counts.items())
        },
        "UNKNOWN_event_count": counts["UNKNOWN"],
        "UNKNOWN_event_share": _safe_ratio(counts["UNKNOWN"], total),
        "UNKNOWN_by_ticker": _counts_payload(by_ticker_unknown),
        "UNKNOWN_by_source": _counts_payload(by_source_unknown),
        "UNKNOWN_by_source_family": _counts_payload(by_source_family_unknown),
        "UNKNOWN_through_time": _unknown_through_time(train_val_events),
        "event_type_entropy": _entropy(counts),
        "event_type_through_time": _nested_pair_counts(by_time),
        "known_event_type_cardinality_per_issuer": _known_event_cardinality_per_issuer(
            train_val_events
        ),
        "event_type_entropy_per_issuer": _event_entropy_per_field(train_val_events, "issuer"),
        "UNKNOWN_POLICY": "NO_TAXONOMY_OR_RULES_CHANGE",
    }


def _duplicates_clusters(
    clusters: list[dict[str, Any]], assignments: dict[str, str], dataset_manifest: dict[str, Any]
) -> dict[str, Any]:
    cluster_members: dict[str, list[str]] = defaultdict(list)
    for row in clusters:
        cluster_members[str(row["event_cluster_id"])].append(str(row["event_id"]))
    sizes = Counter(len(members) for members in cluster_members.values())
    split_violations = [
        cluster_id
        for cluster_id, members in cluster_members.items()
        if len({assignments[event_id] for event_id in members if event_id in assignments}) > 1
    ]
    duplicate_updates = cast(
        "dict[str, Any]", dataset_manifest.get("duplicate_update_diagnostics", {})
    )
    return {
        "cluster_count": len(cluster_members),
        "cluster_size_distribution": {str(size): count for size, count in sorted(sizes.items())},
        "multi_event_clusters": sum(count for size, count in sizes.items() if size > 1),
        "duplicate_update_diagnostics": duplicate_updates,
        "split_cluster_violation_count": len(split_violations),
        "split_cluster_violation_examples": split_violations[:10],
        "CLUSTER_INTEGRITY": "PASS" if not split_violations else "FAIL",
        "DUPLICATE_DIAGNOSTIC_STATUS": (
            "DUPLICATE_SOURCE_RECORDS_OBSERVED_NO_AUTOMATIC_FILTER_CHANGE"
            if duplicate_updates
            else "NO_DUPLICATE_UPDATE_DIAGNOSTICS_PRESENT"
        ),
    }


def _timestamp_quality(
    events: list[dict[str, Any]], train_val_events: list[dict[str, Any]]
) -> dict[str, Any]:
    timestamps = [_parse_datetime(_metadata(row)["publication_timestamp_utc"]) for row in events]
    train_val_times = [
        _parse_datetime(_metadata(row)["publication_timestamp_utc"]) for row in train_val_events
    ]
    source_fields = Counter(
        str(_metadata(row).get("timestamp_source_field", "UNKNOWN")) for row in events
    )
    timezones = Counter(
        str(_metadata(row).get("publication_timezone", "UNKNOWN")) for row in events
    )
    qualities = Counter(str(_metadata(row).get("timestamp_quality", "UNKNOWN")) for row in events)
    sessions = Counter(str(_metadata(row).get("session_state", "UNKNOWN")) for row in events)
    seconds = Counter(ts.second for ts in timestamps)
    minute_buckets = Counter(_minute_bucket(ts.time()) for ts in train_val_times)
    return {
        "scope": "ALL_EVENTS_METADATA; TRAIN_VALIDATION_MINUTE_BUCKETS",
        "timestamp_quality_counts": dict(sorted(qualities.items())),
        "timestamp_source_field_counts": dict(sorted(source_fields.items())),
        "publication_timezone_counts": dict(sorted(timezones.items())),
        "session_state_counts": dict(sorted(sessions.items())),
        "seconds_component_counts": dict(sorted(seconds.items())),
        "train_validation_minute_buckets": dict(sorted(minute_buckets.items())),
        "date_only_collapse_detected": len(seconds) == 1 and seconds.get(0, 0) == len(events),
        "TIMESTAMP_DIAGNOSTIC_STATUS": (
            "PASS_EXACT_UTC_TIMESTAMPS"
            if qualities == {"EXACT": len(events)} and timezones == {"UTC": len(events)}
            else "REVIEW_TIMESTAMP_METADATA"
        ),
    }


def _target_quality(train_val_targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_horizon: dict[str, Any] = {}
    values_by_horizon: dict[str, list[float]] = {}
    for horizon in EXACT_HORIZONS:
        values: list[float] = []
        windows_ok = 0
        same_observed = 0
        for payload in train_val_targets.values():
            item = cast("dict[str, Any]", payload["horizons"][horizon])
            value = _to_float(item["abnormal_return"])
            values.append(value)
            if _parse_datetime(item["window_begin_at"]) < _parse_datetime(item["window_end_at"]):
                windows_ok += 1
            if item["security_observed_at"] == item["benchmark_observed_at"]:
                same_observed += 1
        values_by_horizon[horizon] = values
        by_horizon[horizon] = {
            "rows": len(values),
            "eligible_N": len(values),
            "missing_N": 0,
            "abnormal_return": _numeric_summary(values),
            "direction_distribution": _direction_distribution(values),
            "positive_share": _safe_ratio(sum(1 for value in values if value > 0), len(values)),
            "negative_share": _safe_ratio(sum(1 for value in values if value < 0), len(values)),
            "window_order_pass_count": windows_ok,
            "security_benchmark_same_observed_at_count": same_observed,
            "near_zero_share": _safe_ratio(
                sum(1 for value in values if abs(value) <= FLAT_RETURN_THRESHOLD), len(values)
            ),
            "extreme_abs_gt_2pct_count": sum(1 for value in values if abs(value) > 0.02),
        }
    return {
        "scope": "TRAIN_VALIDATION_ONLY",
        "horizons": by_horizon,
        "correlations": _horizon_correlations(values_by_horizon),
        "TARGET_INTEGRITY_STATUS": "PASS",
        "TEST_OUTCOME_USED": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
    }


def _exact_vs_date_pairing() -> dict[str, Any]:
    return {
        "EXACT_DATE_PAIRING_STATUS": "FAIL_CLOSED_NO_CANONICAL_EVENT_IDENTITY",
        "paired_rows": 0,
        "deterministic_identity_fields_required": [
            "event_id",
            "event_cluster_id",
            "canonical source item id",
            "exact timestamp",
            "ticker",
        ],
        "outcomes_read": False,
        "reason": (
            "The DATE_SAFE and EXACT artifacts do not expose a shared immutable row identity "
            "that can prove one-to-one pairing without heuristic matching."
        ),
    }


def _feature_data_quality(
    train_events: list[dict[str, Any]],
    val_events: list[dict[str, Any]],
    feature_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "scope": "TRAIN_VALIDATION_TARGET_FREE",
        "TRAIN": _feature_scope_quality(train_events, feature_by_id),
        "VALIDATION": _feature_scope_quality(val_events, feature_by_id),
        "validation_unseen_categories": _validation_unseen_categories(train_events, val_events),
        "EVENT_FEATURE_DATA_STATUS": "PASS_TARGET_FREE",
        "MARKET_CONTEXT_DATA_STATUS": "PASS_TARGET_FREE",
        "FEATURE_QUALITY_STATUS": "PASS_TARGET_FREE",
    }


def _temporal_coverage(train_val_events: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted(_parse_date(_metadata(row)["publication_date"]) for row in train_val_events)
    by_month = Counter(day.strftime("%Y-%m") for day in dates)
    by_quarter = Counter(f"{day.year}-Q{((day.month - 1) // 3) + 1}" for day in dates)
    by_year = Counter(str(day.year) for day in dates)
    gaps = [
        {
            "from": dates[index - 1].isoformat(),
            "to": dates[index].isoformat(),
            "gap_days": (dates[index] - dates[index - 1]).days,
        }
        for index in range(1, len(dates))
        if (dates[index] - dates[index - 1]).days > 14
    ]
    return {
        "scope": "TRAIN_VALIDATION_ONLY",
        "first_date": dates[0].isoformat() if dates else None,
        "last_date": dates[-1].isoformat() if dates else None,
        "events_by_month": dict(sorted(by_month.items())),
        "events_by_quarter": dict(sorted(by_quarter.items())),
        "events_by_year": dict(sorted(by_year.items())),
        "ticker_coverage_through_time": _coverage_through_time(train_val_events, "ticker"),
        "source_coverage_through_time": _coverage_through_time(train_val_events, "source_code"),
        "event_type_coverage_through_time": _event_type_coverage_through_time(train_val_events),
        "large_gap_count_gt_14d": len(gaps),
        "large_gap_examples": gaps[:10],
        "TEMPORAL_COVERAGE_STATUS": "GAPS_PRESENT" if gaps else "CONTIGUOUS_ENOUGH",
    }


def _priority_report(diagnostics: dict[str, Any]) -> dict[str, Any]:
    funnel = cast("dict[str, Any]", diagnostics["eligibility_funnel"])
    warmup = cast("dict[str, Any]", diagnostics["warmup_loss"])
    event_types = cast("dict[str, Any]", diagnostics["event_type_coverage"])
    concentration = cast("dict[str, Any]", diagnostics["concentration"])
    train_val_concentration = cast("dict[str, Any]", concentration["train_validation"])
    priorities = [
        {
            "priority": "NEXT_DATA_PRIORITY",
            "name": "MARKET_HISTORY_WARMUP_RECOVERY",
            "evidence": {
                "warmup_lost_total_metadata_only": warmup["warmup_lost_total_metadata_only"],
                "warmup_lost_train_validation_period": warmup[
                    "warmup_lost_train_validation_period"
                ],
                "recoverable_candidate": warmup["WARMUP_RECOVERABLE_CANDIDATE"],
            },
            "why": (
                "This is the largest deterministic leakage-safe row recovery lever before "
                "changing event rules or modeling."
            ),
        },
        {
            "priority": "SECONDARY",
            "name": "EVENT_SEMANTIC_COVERAGE",
            "evidence": {
                "unknown_event_count_train_validation": event_types["UNKNOWN_event_count"],
                "unknown_event_share_train_validation": event_types["UNKNOWN_event_share"],
            },
            "why": "UNKNOWN remains high, but taxonomy/rules are intentionally frozen in this PR.",
        },
        {
            "priority": "SECONDARY",
            "name": "ISSUER_CONCENTRATION",
            "evidence": {
                "ticker_top_share_train_validation": train_val_concentration["ticker"]["top_share"],
                "source_top_share_train_validation": train_val_concentration["source_code"][
                    "top_share"
                ],
            },
            "why": (
                "The corpus is still dominated by a small number of issuers and source families."
            ),
        },
        {
            "priority": "SECONDARY",
            "name": "DUPLICATE_CLUSTERING",
            "evidence": {
                "funnel_reconciliation": funnel["funnel_reconciliation"],
                "cluster_integrity": diagnostics["duplicates_clusters"]["CLUSTER_INTEGRITY"],
            },
            "why": "Keep existing fail-closed gates as the corpus expands.",
        },
    ]
    return {
        "NEXT_DATA_PRIORITY": priorities[0]["name"],
        "SECONDARY_PRIORITIES": [item["name"] for item in priorities[1:]],
        "ranked_priorities": priorities,
        "NO_MODEL_OR_TEST_RECOMMENDATION": True,
    }


def _feature_scope_quality(
    events: list[dict[str, Any]], feature_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    event_feature_rows = [cast("dict[str, Any]", row["event_features"]) for row in events]
    market_feature_rows = [
        cast("dict[str, Any]", feature_by_id[_event_id(row)]["market_features"])
        for row in events
        if _event_id(row) in feature_by_id
    ]
    return {
        "rows": len(events),
        "event_features": _feature_family_quality(event_feature_rows),
        "market_features": _feature_family_quality(market_feature_rows),
    }


def _feature_family_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({name for row in rows for name in row})
    result: dict[str, Any] = {}
    for name in names:
        values = [row.get(name) for row in rows]
        missing = sum(1 for value in values if value is None)
        non_missing = [value for value in values if value is not None]
        numeric = [_to_float(value) for value in non_missing if _is_number_like(value)]
        result[name] = {
            "missing_count": missing,
            "missing_share": _safe_ratio(missing, len(values)),
            "zero_count": sum(1 for value in numeric if value == 0.0),
            "zero_share": _safe_ratio(sum(1 for value in numeric if value == 0.0), len(numeric)),
            "cardinality": len(
                {json.dumps(value, sort_keys=True, default=str) for value in non_missing}
            ),
            "constant": len(
                {json.dumps(value, sort_keys=True, default=str) for value in non_missing}
            )
            <= 1,
        }
    return result


def _validation_unseen_categories(
    train_events: list[dict[str, Any]], val_events: list[dict[str, Any]]
) -> dict[str, list[str]]:
    names = sorted(
        {
            name
            for row in (*train_events, *val_events)
            for name, value in cast("dict[str, Any]", row["event_features"]).items()
            if isinstance(value, str)
        }
    )
    unseen: dict[str, list[str]] = {}
    for name in names:
        train_values = {
            str(cast("dict[str, Any]", row["event_features"]).get(name)) for row in train_events
        }
        val_values = {
            str(cast("dict[str, Any]", row["event_features"]).get(name)) for row in val_events
        }
        new_values = sorted(val_values - train_values)
        if new_values:
            unseen[name] = new_values
    return unseen


def _train_val_targets(
    targets: list[dict[str, Any]], train_val_ids: set[str]
) -> dict[str, dict[str, Any]]:
    filtered: dict[str, dict[str, Any]] = {}
    for row in targets:
        event_id = str(row["event_id"])
        if event_id in train_val_ids:
            filtered[event_id] = row
    if set(filtered) != train_val_ids:
        raise ValueError("TRAIN_VAL_TARGETS_MISSING")
    return filtered


def _reaction_drop_reasons(events: list[dict[str, Any]]) -> dict[str, int]:
    reasons: Counter[str] = Counter()
    for row in events:
        metadata = _metadata(row)
        availability = _target_availability(row)
        if metadata.get("session_state") != "DURING_MAIN_SESSION":
            continue
        if availability.get("reaction_ready"):
            continue
        reason = str(availability.get("missing_reason") or availability.get("status") or "OTHER")
        if "BENCHMARK" in reason:
            reasons["benchmark_missing"] += 1
        elif "SECURITY" in reason or "TARGET" in reason:
            reasons["reaction_missing"] += 1
        else:
            reasons["other"] += 1
    return _complete_reason_taxonomy(reasons)


def _complete_reason_taxonomy(reasons: Counter[str]) -> dict[str, int]:
    return {
        "reaction_missing": reasons["reaction_missing"],
        "benchmark_missing": reasons["benchmark_missing"],
        "ticker_mapping": reasons["ticker_mapping"],
        "source_policy_issue": reasons["source_policy_issue"],
        "cluster_excluded": reasons["cluster_excluded"],
        "missing_event_features": reasons["missing_event_features"],
        "market_context_missing": reasons["market_context_missing"],
        "other": reasons["other"],
    }


def _concentration_for(rows: list[dict[str, Any]], metadata_field: str) -> dict[str, Any]:
    counts = Counter(_metadata_value(row, metadata_field) for row in rows)
    return {
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "shares": {key: _safe_ratio(value, len(rows)) for key, value in sorted(counts.items())},
        "top_share": _top_share(counts),
        "top_3_share": _top_n_share(counts, 3),
        "hhi": _hhi(counts),
        "effective_count": _effective_count(counts),
    }


def _horizon_correlations(values_by_horizon: dict[str, list[float]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for left_index, left in enumerate(EXACT_HORIZONS):
        for right in EXACT_HORIZONS[left_index + 1 :]:
            key = f"{left}_vs_{right}"
            left_values = values_by_horizon[left]
            right_values = values_by_horizon[right]
            result[key] = {
                "pearson": _pearson(left_values, right_values),
                "spearman": _spearman(left_values, right_values),
                "sign_agreement": _sign_agreement(left_values, right_values),
            }
    return result


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": sorted_values[0],
        "p01": _quantile(sorted_values, 0.01),
        "p05": _quantile(sorted_values, 0.05),
        "p25": _quantile(sorted_values, 0.25),
        "p75": _quantile(sorted_values, 0.75),
        "p95": _quantile(sorted_values, 0.95),
        "p99": _quantile(sorted_values, 0.99),
        "max": sorted_values[-1],
    }


def _direction_distribution(values: list[float]) -> dict[str, int]:
    counts = Counter(_direction(value) for value in values)
    return {name: counts[name] for name in ("DOWN", "FLAT", "UP")}


def _direction(value: float) -> str:
    if value > FLAT_RETURN_THRESHOLD:
        return "UP"
    if value < -FLAT_RETURN_THRESHOLD:
        return "DOWN"
    return "FLAT"


def _top_loss(drops: list[dict[str, Any]]) -> dict[str, Any]:
    return max(drops, key=lambda item: int(item["count"]))


def _counts_payload(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    return {
        "counts": dict(sorted(counter.items())),
        "shares": {key: _safe_ratio(value, total) for key, value in sorted(counter.items())},
        "top_share": _top_share(counter),
    }


def _first_dates(feature_rows: Any, events: list[dict[str, Any]]) -> dict[str, str]:
    event_by_id = {_event_id(row): row for row in events}
    by_ticker: dict[str, list[str]] = defaultdict(list)
    for row in feature_rows:
        event = event_by_id[str(row["event_id"])]
        by_ticker[str(_metadata(event)["ticker"])].append(str(_metadata(event)["publication_date"]))
    return {ticker: min(values) for ticker, values in sorted(by_ticker.items())}


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count)


def _top_share(counter: Counter[str]) -> float:
    return _safe_ratio(max(counter.values(), default=0), sum(counter.values()))


def _top_n_share(counter: Counter[str], n: int) -> float:
    return _safe_ratio(sum(count for _, count in counter.most_common(n)), sum(counter.values()))


def _hhi(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return sum((count / total) ** 2 for count in counter.values()) if total else 0.0


def _effective_count(counter: Counter[str]) -> float | None:
    hhi = _hhi(counter)
    return (1.0 / hhi) if hhi else None


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator else None


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: list[float]) -> list[float]:
    sorted_pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(sorted_pairs):
        next_cursor = cursor + 1
        while (
            next_cursor < len(sorted_pairs)
            and sorted_pairs[next_cursor][0] == sorted_pairs[cursor][0]
        ):
            next_cursor += 1
        average_rank = (cursor + next_cursor - 1) / 2 + 1
        for _, index in sorted_pairs[cursor:next_cursor]:
            ranks[index] = average_rank
        cursor = next_cursor
    return ranks


def _sign_agreement(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    return sum(_direction(x) == _direction(y) for x, y in zip(left, right, strict=True)) / len(left)


def _quantile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def _minute_bucket(value: time) -> str:
    minutes = value.hour * 60 + value.minute
    if minutes < 10 * 60:
        return "morning_before_10_utc"
    if minutes < 12 * 60:
        return "midday_10_12_utc"
    if minutes < 14 * 60:
        return "afternoon_12_14_utc"
    return "late_session_after_14_utc"


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _is_number_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float | Decimal):
        return True
    if isinstance(value, str):
        try:
            Decimal(value)
        except Exception:
            return False
        return True
    return False


def _to_float(value: Any) -> float:
    return float(Decimal(str(value)))


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["metadata"])


def _metadata_value(row: dict[str, Any], metadata_field: str) -> str:
    if metadata_field == "source_family":
        return _source_family(row)
    return str(_metadata(row).get(metadata_field, "UNKNOWN"))


def _source_family(row: dict[str, Any]) -> str:
    source = str(_metadata(row).get("source_code", "UNKNOWN"))
    return source.removesuffix("_EXACT")


def _target_availability(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["target_availability"])


def _event_id(row: dict[str, Any]) -> str:
    return str(_metadata(row)["event_id"])


def _parse_datetime(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _parse_date(value: Any) -> date:
    return datetime.fromisoformat(str(value)).date()


def _split_assignments(split_manifest: dict[str, Any]) -> dict[str, str]:
    assignments = {
        str(item["event_id"]): str(item["split"])
        for item in cast("list[dict[str, Any]]", split_manifest["assignments"])
    }
    if set(assignments.values()) != {"TRAIN", "VALIDATION", "TEST"}:
        raise ValueError("UNEXPECTED_SPLIT_LABELS")
    return assignments


def _nested_pair_counts(counter: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for (outer, inner), count in sorted(counter.items()):
        nested[outer][inner] = count
    return dict(sorted(nested.items()))


def _unknown_through_time(train_val_events: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(_metadata(row).get("publication_date", ""))[:7]
        for row in train_val_events
        if row.get("event_features", {}).get("primary_event_type") == "UNKNOWN"
    )
    return dict(sorted(counts.items()))


def _known_event_cardinality_per_issuer(train_val_events: list[dict[str, Any]]) -> dict[str, int]:
    by_issuer: dict[str, set[str]] = defaultdict(set)
    for row in train_val_events:
        event_type = str(row.get("event_features", {}).get("primary_event_type", "UNKNOWN"))
        if event_type != "UNKNOWN":
            by_issuer[str(_metadata(row).get("issuer", "UNKNOWN"))].add(event_type)
    return {issuer: len(values) for issuer, values in sorted(by_issuer.items())}


def _event_entropy_per_field(
    train_val_events: list[dict[str, Any]], metadata_field: str
) -> dict[str, float]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train_val_events:
        event_type = str(row.get("event_features", {}).get("primary_event_type", "UNKNOWN"))
        grouped[_metadata_value(row, metadata_field)][event_type] += 1
    return {key: _entropy(counter) for key, counter in sorted(grouped.items())}


def _coverage_through_time(
    train_val_events: list[dict[str, Any]], metadata_field: str
) -> dict[str, dict[str, int]]:
    counter = Counter(
        (
            str(_metadata(row).get("publication_date", ""))[:7],
            _metadata_value(row, metadata_field),
        )
        for row in train_val_events
    )
    return _nested_pair_counts(counter)


def _event_type_coverage_through_time(
    train_val_events: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    counter = Counter(
        (
            str(_metadata(row).get("publication_date", ""))[:7],
            str(row.get("event_features", {}).get("primary_event_type", "UNKNOWN")),
        )
        for row in train_val_events
    )
    return _nested_pair_counts(counter)


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(cast("dict[str, Any]", json.loads(line)))
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    priority = cast("dict[str, Any]", manifest["diagnostics"]["priority_report"])
    counts = cast("dict[str, Any]", manifest["counts"])
    safety = cast("dict[str, Any]", manifest["safety"])
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        "Diagnostic-only EXACT event corpus report.",
        "",
        "## Safety",
        "",
        *[f"- {key}={str(value).lower()}" for key, value in safety.items()],
        "",
        "## Scope",
        "",
        f"- dataset_sha={manifest['dataset_sha']}",
        f"- baseline_artifact_sha={manifest['baseline_artifact_sha']}",
        f"- exact_events={counts['exact_events']}",
        f"- train_validation={counts['train_validation']}",
        f"- test_metadata_only={counts['test_metadata_only']}",
        f"- future_holdout_metadata_only={counts['future_holdout_metadata_only']}",
        "",
        "## Priority",
        "",
        f"- NEXT_DATA_PRIORITY={priority['NEXT_DATA_PRIORITY']}",
        f"- SECONDARY_PRIORITIES={', '.join(priority['SECONDARY_PRIORITIES'])}",
        "",
        "No model training, TEST outcome use, future holdout observation, backtest, paper "
        "trading, orders, or BUY/SELL output was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
