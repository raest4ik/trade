from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, cast

from src.event_predictive_baseline.domain import (
    DATASET_VERSION,
    EXPECTED_DATASET_SHA,
    EXPECTED_FEATURE_SCHEMA_SHA,
    EXPECTED_PROVENANCE_SHA,
    EXPECTED_SOURCE_REGISTRY_SHA,
    PREDICTIVE_UNIT,
    REACTION_FAMILY,
    TRAIN_END,
    VALIDATION_END,
    ComparisonCohort,
    EventFeatureRow,
    EventTargetRow,
    sha256_payload,
)


def load_comparison_cohort(root: Path) -> tuple[ComparisonCohort, dict[str, Any]]:
    manifest = _json(root / "manifest.json")
    schema = _json(root / "feature-schema.json")
    validate_frozen_manifest(manifest)
    if sha256_payload(schema) != EXPECTED_FEATURE_SCHEMA_SHA:
        raise ValueError("frozen feature schema SHA changed")
    rows: list[EventFeatureRow] = []
    excluded: Counter[str] = Counter()
    for payload in _jsonl(root / "features.jsonl"):
        metadata = cast("dict[str, Any]", payload["metadata"])
        quality = cast("dict[str, Any]", payload["quality"])
        reason = _exclusion_reason(metadata, quality)
        if reason is not None:
            excluded[reason] += 1
            continue
        rows.append(_feature_row(payload))
    rows.sort(key=lambda row: (row.publication_date, row.ticker, row.event_id))
    if not rows:
        raise ValueError("comparison cohort is empty")
    event_names = tuple(str(name) for name in cast("list[Any]", schema["event_features"]))
    observed_market_names = {name for row in rows for name in row.market_features}
    market_names = tuple(
        str(name)
        for name in cast("list[Any]", schema["market_features"])
        if name in observed_market_names
    )
    for row in rows:
        if set(row.event_features) != set(event_names):
            raise ValueError("event feature row schema changed")
        if set(row.market_features) != set(market_names):
            raise ValueError("DATE_SAFE_DAILY market feature row schema changed")
    event_schema_sha = sha256_payload(list(event_names))
    market_schema_sha = sha256_payload(list(market_names))
    cohort_sha = sha256_payload(
        {
            "dataset_sha": EXPECTED_DATASET_SHA,
            "reaction_family": REACTION_FAMILY,
            "event_schema_sha": event_schema_sha,
            "market_schema_sha": market_schema_sha,
            "event_ids": [row.event_id for row in rows],
        }
    )
    return (
        ComparisonCohort(
            rows=tuple(rows),
            event_feature_names=event_names,
            market_feature_names=market_names,
            cohort_sha=cohort_sha,
            event_schema_sha=event_schema_sha,
            market_schema_sha=market_schema_sha,
        ),
        {"excluded_by_reason": dict(sorted(excluded.items()))},
    )


def build_temporal_split(cohort: ComparisonCohort) -> dict[str, Any]:
    assignments: list[dict[str, str]] = []
    for row in cohort.rows:
        split = (
            "TRAIN"
            if row.publication_date <= TRAIN_END
            else "VALIDATION"
            if row.publication_date <= VALIDATION_END
            else "TEST"
        )
        assignments.append({"event_id": row.event_id, "split": split})
    assignment_map = {item["event_id"]: item["split"] for item in assignments}
    if len(assignment_map) != len(assignments):
        raise ValueError("duplicate event_id in temporal split")
    dates: dict[date, set[str]] = {}
    issuer_dates: dict[tuple[str, date], set[str]] = {}
    stories: dict[tuple[str, date, str], set[str]] = {}
    for row in cohort.rows:
        split = assignment_map[row.event_id]
        dates.setdefault(row.publication_date, set()).add(split)
        issuer_dates.setdefault((row.ticker, row.publication_date), set()).add(split)
        if row.title_hash:
            stories.setdefault((row.ticker, row.publication_date, row.title_hash), set()).add(split)
    if any(
        len(values) != 1 for values in (*dates.values(), *issuer_dates.values(), *stories.values())
    ):
        raise ValueError("temporal grouping leakage detected")
    split_payload: dict[str, Any] = {
        "protocol": "CHRONOLOGICAL_PUBLICATION_DATE_V1",
        "grouping": ["publication_date", "ticker+publication_date", "same_story_if_available"],
        "boundaries": {
            "TRAIN": {"from": "2022-01-01", "to": TRAIN_END.isoformat()},
            "VALIDATION": {"from": "2025-01-01", "to": VALIDATION_END.isoformat()},
            "TEST": {
                "from": "2026-01-01",
                "to": max(row.publication_date for row in cohort.rows).isoformat(),
            },
        },
        "assignments": assignments,
        "counts": _split_counts(cohort, assignment_map),
        "ticker_counts": _split_tickers(cohort, assignment_map),
        "date_ranges": _split_date_ranges(cohort, assignment_map),
        "target_outcomes_inspected_before_lock": False,
        "leakage_check": "PASS",
    }
    split_payload["split_sha"] = sha256_payload(split_payload)
    return split_payload


def load_targets(
    path: Path, event_ids: set[str], reaction_family: str = REACTION_FAMILY
) -> dict[str, EventTargetRow]:
    targets: dict[str, EventTargetRow] = {}
    for payload in _jsonl(path):
        if payload.get("reaction_family") != reaction_family:
            continue
        event_id = str(payload["event_id"])
        if event_id not in event_ids:
            continue
        target = EventTargetRow(
            event_id=event_id,
            direction=str(payload["classification"]),
            abnormal_return=float(payload["abnormal_return"]),
            security_return=float(payload["security_return"]),
        )
        if target.direction not in {"DOWN", "FLAT", "UP"}:
            raise ValueError("unknown target direction")
        if not math.isfinite(target.abnormal_return) or not math.isfinite(target.security_return):
            raise ValueError("non-finite target")
        targets[event_id] = target
    if set(targets) != event_ids:
        raise ValueError("targets do not cover the requested comparison cohort")
    return targets


def exact_event_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features = [
        payload
        for payload in _jsonl(root / "features.jsonl")
        if cast("dict[str, Any]", payload["metadata"])["reaction_family"] == "EXACT_INTRADAY"
    ]
    ids = {str(cast("dict[str, Any]", row["metadata"])["event_id"]) for row in features}
    targets = [
        payload
        for payload in _jsonl(root / "targets.jsonl")
        if payload.get("reaction_family") == "EXACT_INTRADAY" and str(payload["event_id"]) in ids
    ]
    return features, targets


def validate_frozen_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "dataset_version": DATASET_VERSION,
        "event_market_dataset_sha": EXPECTED_DATASET_SHA,
        "source_registry_sha": EXPECTED_SOURCE_REGISTRY_SHA,
        "provenance_manifest_sha": EXPECTED_PROVENANCE_SHA,
        "feature_schema_sha": EXPECTED_FEATURE_SCHEMA_SHA,
        "event_market_leakage_check": "PASS",
        "predictive_unit": PREDICTIVE_UNIT,
        "new_total_real_events": 1276,
        "event_market_feature_ready": 1260,
        "unverified_events": 0,
        "rules_changed": False,
        "qwen_changed": False,
        "model_trained": False,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"frozen event dataset mismatch: {name}")


def _exclusion_reason(metadata: dict[str, Any], quality: dict[str, Any]) -> str | None:
    if metadata.get("reaction_family") != REACTION_FAMILY:
        return "REACTION_FAMILY_NOT_DATE_SAFE_DAILY"
    if metadata.get("publication_time_quality") != "DATE_ONLY":
        return "PUBLICATION_TIME_NOT_DATE_ONLY"
    if metadata.get("predictive_unit") != PREDICTIVE_UNIT:
        return "PREDICTIVE_UNIT_NOT_EVENT"
    if not metadata.get("ticker") or not metadata.get("instrument_uid"):
        return "TICKER_NOT_UNIQUELY_MATCHED"
    if metadata.get("source_rights_status") != "PRIVATE_INTERNAL_RESEARCH_ONLY":
        return "SOURCE_POLICY_NOT_ACCEPTED"
    if quality.get("post_event_values_in_features") is not False:
        return "LEAKAGE_AUDIT_FAILED"
    if not quality.get("event_available_at_cutoff") or not quality.get(
        "market_context_available_at_cutoff"
    ):
        return "POINT_IN_TIME_AVAILABILITY_FAILED"
    return None


def _feature_row(payload: dict[str, Any]) -> EventFeatureRow:
    metadata = cast("dict[str, Any]", payload["metadata"])
    event_values = cast("dict[str, Any]", payload["event_features"])
    market_values = cast("dict[str, Any]", payload["market_features"])
    numeric_market = {str(name): float(value) for name, value in market_values.items()}
    if not all(math.isfinite(value) for value in numeric_market.values()):
        raise ValueError("non-finite market feature")
    return EventFeatureRow(
        event_id=str(metadata["event_id"]),
        ticker=str(metadata["ticker"]),
        issuer_name=str(metadata["issuer_name"]),
        publication_date=date.fromisoformat(str(metadata["publication_date"])),
        source_family=str(metadata["source_code"]),
        title_hash=str(metadata["title_hash"]) if metadata.get("title_hash") else None,
        event_features={str(name): value for name, value in event_values.items()},
        market_features=numeric_market,
    )


def _split_counts(cohort: ComparisonCohort, assignments: dict[str, str]) -> dict[str, int]:
    return {
        split: sum(assignments[row.event_id] == split for row in cohort.rows)
        for split in ("TRAIN", "VALIDATION", "TEST")
    }


def _split_tickers(cohort: ComparisonCohort, assignments: dict[str, str]) -> dict[str, list[str]]:
    return {
        split: sorted({row.ticker for row in cohort.rows if assignments[row.event_id] == split})
        for split in ("TRAIN", "VALIDATION", "TEST")
    }


def _split_date_ranges(
    cohort: ComparisonCohort, assignments: dict[str, str]
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
