from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

from src.event_predictive_baseline.domain import (
    DATASET_VERSION,
    EXACT_HORIZONS,
    EXPECTED_CLUSTER_SHA,
    EXPECTED_DATASET_SHA,
    EXPECTED_PROVENANCE_SHA,
    EXPECTED_REACTION_SHA,
    EXPECTED_SOURCE_REGISTRY_SHA,
    EXPECTED_TIMESTAMP_SHA,
    FUTURE_EVENT_HOLDOUT_START,
    REACTION_FAMILY,
    EventFeatureRow,
    EventTargetRow,
    HorizonCohort,
    classify_abnormal_return,
    guard_future_holdout_outcome_read,
    sha256_payload,
)


def load_exact_horizon_cohorts(root: Path) -> tuple[dict[str, HorizonCohort], dict[str, Any]]:
    manifest = _json(root / "manifest.json")
    holdout = _json(root / "future-holdout-status.json")
    validate_exact_manifest(manifest, holdout)
    events = _events_by_id(root / "events.jsonl")
    features = _features_by_id(root / "features.jsonl")
    targets = _targets_by_id(root / "targets.jsonl", events)
    event_schema = _event_schema(features)
    market_schema = _market_schema(features)
    event_schema_sha = sha256_payload(event_schema)
    market_schema_sha = sha256_payload(market_schema)
    target_schema = (
        "abnormal_return",
        "security_return",
        "benchmark_return",
        "window_begin_at",
        "window_end_at",
        "security_observed_at",
        "benchmark_observed_at",
    )
    target_schema_sha = sha256_payload(target_schema)
    cohorts: dict[str, HorizonCohort] = {}
    excluded: dict[str, dict[str, int]] = {}
    for horizon in EXACT_HORIZONS:
        rows: list[EventFeatureRow] = []
        horizon_targets: dict[str, EventTargetRow] = {}
        reasons: Counter[str] = Counter()
        for event_id, feature in features.items():
            event = events.get(event_id)
            if event is None:
                reasons["EVENT_METADATA_MISSING"] += 1
                continue
            metadata = cast("dict[str, Any]", event["metadata"])
            quality = cast("dict[str, Any]", event["quality"])
            reason = _eligibility_exclusion(metadata, quality, feature, targets, horizon)
            if reason is not None:
                reasons[reason] += 1
                continue
            publication_date = date.fromisoformat(str(metadata["publication_date"]))
            guard_future_holdout_outcome_read(publication_date, context=f"load_target:{horizon}")
            target = _target_row(event_id, horizon, targets[event_id][horizon])
            rows.append(_feature_row(event, feature, event_schema, market_schema))
            horizon_targets[event_id] = target
        rows.sort(
            key=lambda row: (
                row.publication_date,
                row.publication_timestamp_utc,
                row.ticker,
                row.event_id,
            )
        )
        if not rows:
            raise ValueError(f"empty exact horizon cohort: {horizon}")
        event_ids = [row.event_id for row in rows]
        if set(event_ids) != set(horizon_targets):
            raise ValueError("target/cohort event ids drifted")
        cohort_payload = {
            "dataset_sha": EXPECTED_DATASET_SHA,
            "reaction_family": REACTION_FAMILY,
            "horizon": horizon,
            "eligibility": "EXACT_DURING_SESSION_FEATURE_AND_TARGET_READY_HISTORICAL",
            "event_schema_sha": event_schema_sha,
            "market_schema_sha": market_schema_sha,
            "target_schema_sha": target_schema_sha,
            "event_ids": event_ids,
        }
        cohorts[horizon] = HorizonCohort(
            horizon=horizon,
            rows=tuple(rows),
            targets=horizon_targets,
            event_feature_names=tuple(event_schema),
            market_feature_names=tuple(market_schema),
            cohort_sha=sha256_payload(cohort_payload),
            event_schema_sha=event_schema_sha,
            market_schema_sha=market_schema_sha,
            target_schema_sha=target_schema_sha,
        )
        excluded[horizon] = dict(sorted(reasons.items()))
    metadata = {
        "dataset_counts": {
            "EXACT_EVENTS": manifest["exact_timestamp_events"],
            "EXACT_REACTION_READY": manifest["exact_reaction_ready"],
            "EXACT_FEATURE_READY": manifest["exact_feature_ready"],
            "EXACT_UNIQUE_TICKERS": manifest["exact_unique_tickers"],
            "EXACT_UNIQUE_ISSUERS": manifest["exact_unique_issuers"],
        },
        "excluded_by_horizon": excluded,
        "future_holdout": holdout,
        "EVENT_MARKET_LEAKAGE_CHECK": manifest["EVENT_MARKET_LEAKAGE_CHECK"],
        "FUTURE_EVENT_HOLDOUT_OBSERVED": manifest["FUTURE_EVENT_HOLDOUT_OBSERVED"],
        "holdout_guard": manifest["holdout_guard"],
        "rules_changed": manifest["rules_changed"],
        "qwen_changed": manifest["qwen_changed"],
        "qwen_run": manifest["qwen_run"],
        "NLP_FROZEN": manifest["NLP_FROZEN"],
    }
    return cohorts, metadata


def validate_exact_manifest(manifest: dict[str, Any], holdout: dict[str, Any]) -> None:
    expected = {
        "dataset_version": DATASET_VERSION,
        "exact_dataset_sha": EXPECTED_DATASET_SHA,
        "source_registry_sha": EXPECTED_SOURCE_REGISTRY_SHA,
        "provenance_sha": EXPECTED_PROVENANCE_SHA,
        "timestamp_manifest_sha": EXPECTED_TIMESTAMP_SHA,
        "reaction_manifest_sha": EXPECTED_REACTION_SHA,
        "cluster_manifest_sha": EXPECTED_CLUSTER_SHA,
        "EVENT_MARKET_LEAKAGE_CHECK": "PASS",
        "EXACT_V1_PRESERVED": "YES",
        "EXACT_MODEL_DATA_STATUS": "READY_FOR_EXACT_BASELINE_EXPERIMENT",
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "holdout_guard": "PASS",
        "rules_changed": False,
        "qwen_changed": False,
        "qwen_run": False,
        "NLP_FROZEN": True,
        "model_trained": False,
        "abc_evaluated": False,
        "backtest_executed": False,
        "paper_trading_executed": False,
        "orders_submitted": False,
        "buy_sell_generated": False,
        "real_trading_executed": False,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"frozen exact dataset mismatch: {name}")
    if holdout.get("FUTURE_EVENT_HOLDOUT_START") != FUTURE_EVENT_HOLDOUT_START.isoformat():
        raise ValueError("future holdout start changed")
    if holdout.get("FUTURE_EVENT_HOLDOUT_OBSERVED") is not False:
        raise ValueError("future holdout observed")
    if int(holdout.get("outcome_fields_exported_for_future", -1)) != 0:
        raise ValueError("future holdout outcome fields exported")


def build_temporal_split(cohort: HorizonCohort) -> dict[str, Any]:
    groups = _atomic_groups(cohort.rows)
    preferred = _preferred_assignments(groups)
    assignments = (
        preferred if _split_is_usable(cohort, preferred) else _fallback_assignments(groups)
    )
    _assert_split_integrity(cohort, assignments)
    payload: dict[str, Any] = {
        "protocol": (
            "PREFERRED_CALENDAR_SPLIT_V1"
            if preferred == assignments
            else "DETERMINISTIC_CHRONOLOGICAL_60_20_20_GROUPED_V1"
        ),
        "horizon": cohort.horizon,
        "grouping": ["publication_date", "event_cluster_id"],
        "assignments": [
            {"event_id": row.event_id, "split": assignments[row.event_id]} for row in cohort.rows
        ],
        "counts": _split_counts(cohort, assignments),
        "ticker_counts": _split_tickers(cohort, assignments),
        "date_ranges": _split_date_ranges(cohort, assignments),
        "target_outcomes_inspected_before_lock": False,
        "cluster_integrity": "PASS",
        "leakage_check": "PASS",
    }
    payload["split_sha"] = sha256_payload(payload)
    return payload


def _eligibility_exclusion(
    metadata: dict[str, Any],
    quality: dict[str, Any],
    feature: dict[str, Any],
    targets: dict[str, dict[str, Any]],
    horizon: str,
) -> str | None:
    if date.fromisoformat(str(metadata["publication_date"])) >= FUTURE_EVENT_HOLDOUT_START:
        return "FUTURE_HOLDOUT_EXCLUDED"
    if metadata.get("reaction_family") != REACTION_FAMILY:
        return "REACTION_FAMILY_NOT_EXACT_INTRADAY"
    if metadata.get("timestamp_quality") != "EXACT":
        return "TIMESTAMP_NOT_EXACT"
    if metadata.get("session_state") != "DURING_MAIN_SESSION":
        return "SESSION_NOT_DURING_MAIN_SESSION"
    if bool(metadata.get("future_holdout")):
        return "FUTURE_HOLDOUT_EXCLUDED"
    if not metadata.get("ticker") or not metadata.get("instrument_uid"):
        return "TICKER_NOT_UNIQUELY_MATCHED"
    if quality.get("reaction_starts_after_or_at_publication") is not True:
        return "REACTION_STARTS_BEFORE_PUBLICATION"
    if quality.get("security_benchmark_same_window") is not True:
        return "SECURITY_BENCHMARK_WINDOW_MISMATCH"
    market = cast("dict[str, Any]", feature["market_features"])
    if market.get("post_event_values_in_features") is not False:
        return "MARKET_FEATURE_LEAKAGE"
    if feature.get("event_features") is None or feature.get("market_features") is None:
        return "FEATURES_MISSING"
    horizon_payload = targets.get(str(metadata["event_id"]), {}).get(horizon)
    if not bool(horizon_payload and horizon_payload.get("available")):
        return "TARGET_HORIZON_UNAVAILABLE"
    return None


def _feature_row(
    event: dict[str, Any],
    feature: dict[str, Any],
    event_schema: list[str],
    market_schema: list[str],
) -> EventFeatureRow:
    metadata = cast("dict[str, Any]", event["metadata"])
    event_values = cast("dict[str, Any]", feature["event_features"])
    market_values = cast("dict[str, Any]", feature["market_features"])
    if set(event_values) != set(event_schema):
        raise ValueError("event feature schema drift")
    numeric_market: dict[str, float] = {}
    for name in market_schema:
        value = market_values[name]
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("non-finite market feature")
        numeric_market[name] = numeric
    return EventFeatureRow(
        event_id=str(metadata["event_id"]),
        event_cluster_id=str(metadata["event_cluster_id"]),
        ticker=str(metadata["ticker"]),
        issuer_name=str(metadata["issuer"]),
        publication_date=date.fromisoformat(str(metadata["publication_date"])),
        publication_timestamp_utc=datetime.fromisoformat(
            str(metadata["publication_timestamp_utc"])
        ),
        source_family=str(metadata["source_code"]),
        event_features={str(name): event_values[name] for name in event_schema},
        market_features=numeric_market,
    )


def _target_row(event_id: str, horizon: str, payload: dict[str, Any]) -> EventTargetRow:
    abnormal_return = float(payload["abnormal_return"])
    security_return = float(payload["security_return"])
    benchmark_return = float(payload["benchmark_return"])
    if not all(
        math.isfinite(value) for value in (abnormal_return, security_return, benchmark_return)
    ):
        raise ValueError("non-finite exact target")
    window_begin = str(payload["window_begin_at"])
    window_end = str(payload["window_end_at"])
    security_observed = str(payload["security_observed_at"])
    benchmark_observed = str(payload["benchmark_observed_at"])
    if security_observed != benchmark_observed:
        raise ValueError("security/benchmark target window drift")
    if window_end != security_observed:
        raise ValueError("target observed timestamp drift")
    return EventTargetRow(
        event_id=event_id,
        horizon=horizon,
        direction=classify_abnormal_return(abnormal_return),
        abnormal_return=abnormal_return,
        security_return=security_return,
        benchmark_return=benchmark_return,
        window_begin_at=window_begin,
        window_end_at=window_end,
        security_observed_at=security_observed,
        benchmark_observed_at=benchmark_observed,
    )


def _events_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(cast("dict[str, Any]", payload["metadata"])["event_id"]): payload
        for payload in _jsonl(path)
    }


def _features_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(payload["event_id"]): payload for payload in _jsonl(path)}


def _targets_by_id(
    path: Path, events: dict[str, dict[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for payload in _jsonl(path):
        if payload.get("reaction_family") != REACTION_FAMILY:
            continue
        event_id = str(payload["event_id"])
        event = events.get(event_id)
        if event is None:
            continue
        metadata = cast("dict[str, Any]", event["metadata"])
        publication_date = date.fromisoformat(str(metadata["publication_date"]))
        guard_future_holdout_outcome_read(publication_date, context="target_file")
        result[event_id] = cast("dict[str, dict[str, Any]]", payload["horizons"])
    return result


def _event_schema(features: dict[str, dict[str, Any]]) -> list[str]:
    schemas = {
        tuple(sorted(cast("dict[str, Any]", row["event_features"]))) for row in features.values()
    }
    if len(schemas) != 1:
        raise ValueError("event feature schema is not stable")
    return list(next(iter(schemas)))


def _market_schema(features: dict[str, dict[str, Any]]) -> list[str]:
    names = {
        name
        for row in features.values()
        for name, value in cast("dict[str, Any]", row["market_features"]).items()
        if name not in {"feature_cutoff", "post_event_values_in_features"} and value is not None
    }
    schema = sorted(names)
    if not schema:
        raise ValueError("market feature schema is empty")
    return schema


def _atomic_groups(rows: tuple[EventFeatureRow, ...]) -> list[list[EventFeatureRow]]:
    parent: dict[str, str] = {row.event_id: row.event_id for row in rows}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_date: dict[date, list[str]] = {}
    by_cluster: dict[str, list[str]] = {}
    for row in rows:
        by_date.setdefault(row.publication_date, []).append(row.event_id)
        by_cluster.setdefault(row.event_cluster_id, []).append(row.event_id)
    for ids in (*by_date.values(), *by_cluster.values()):
        anchor = ids[0]
        for event_id in ids[1:]:
            union(anchor, event_id)
    grouped: dict[str, list[EventFeatureRow]] = {}
    for row in rows:
        grouped.setdefault(find(row.event_id), []).append(row)
    return sorted(
        grouped.values(),
        key=lambda group: (
            min(row.publication_date for row in group),
            min(row.publication_timestamp_utc for row in group),
            min(row.event_id for row in group),
        ),
    )


def _preferred_assignments(groups: list[list[EventFeatureRow]]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for group in groups:
        group_date = min(row.publication_date for row in group)
        split = (
            "TRAIN"
            if group_date <= date(2024, 12, 31)
            else "VALIDATION"
            if group_date <= date(2025, 12, 31)
            else "TEST"
        )
        for row in group:
            assignments[row.event_id] = split
    return assignments


def _fallback_assignments(groups: list[list[EventFeatureRow]]) -> dict[str, str]:
    total = sum(len(group) for group in groups)
    train_limit = total * 0.6
    validation_limit = total * 0.8
    assignments: dict[str, str] = {}
    observed = 0
    for group in groups:
        next_count = observed + len(group)
        if next_count <= train_limit or not any(value == "TRAIN" for value in assignments.values()):
            split = "TRAIN"
        elif next_count <= validation_limit or not any(
            value == "VALIDATION" for value in assignments.values()
        ):
            split = "VALIDATION"
        else:
            split = "TEST"
        for row in group:
            assignments[row.event_id] = split
        observed = next_count
    return assignments


def _split_is_usable(cohort: HorizonCohort, assignments: dict[str, str]) -> bool:
    counts = _split_counts(cohort, assignments)
    return counts["TRAIN"] >= 30 and counts["VALIDATION"] >= 10 and counts["TEST"] >= 10


def _assert_split_integrity(cohort: HorizonCohort, assignments: dict[str, str]) -> None:
    for key in ("publication_date", "event_cluster_id"):
        grouped: dict[object, set[str]] = {}
        for row in cohort.rows:
            value = getattr(row, key)
            grouped.setdefault(value, set()).add(assignments[row.event_id])
        if any(len(values) != 1 for values in grouped.values()):
            raise ValueError(f"{key} crosses temporal split")
    if set(assignments) != {row.event_id for row in cohort.rows}:
        raise ValueError("temporal split does not cover cohort")


def _split_counts(cohort: HorizonCohort, assignments: dict[str, str]) -> dict[str, int]:
    return {
        split: sum(assignments[row.event_id] == split for row in cohort.rows)
        for split in ("TRAIN", "VALIDATION", "TEST")
    }


def _split_tickers(cohort: HorizonCohort, assignments: dict[str, str]) -> dict[str, list[str]]:
    return {
        split: sorted({row.ticker for row in cohort.rows if assignments[row.event_id] == split})
        for split in ("TRAIN", "VALIDATION", "TEST")
    }


def _split_date_ranges(
    cohort: HorizonCohort, assignments: dict[str, str]
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for split in ("TRAIN", "VALIDATION", "TEST"):
        dates = [row.publication_date for row in cohort.rows if assignments[row.event_id] == split]
        if not dates:
            raise ValueError(f"empty temporal split: {split}")
        result[split] = {"from": min(dates).isoformat(), "to": max(dates).isoformat()}
    return result


def _json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
