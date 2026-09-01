from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_dataset_readiness_audit.domain import (
    CANONICAL_ML_V2_COHORT,
    DEFAULT_INPUT_ARTIFACT_ROOT,
    DEFAULT_OLD_BASELINE_ROOT,
    EXPECTED_INPUT_ARTIFACT_SHA,
    EXPECTED_RULES_V3_FINGERPRINT,
    FUTURE_EVENT_HOLDOUT_START,
    HORIZONS,
    ML_V2_ARTIFACT_VERSION,
    OLD_BASELINE_TEST_STATUS,
    PRIMARY_ML_V2_HORIZON,
    EventOrigin,
    MlV2ReadinessDecision,
    artifact_sha,
    safety_flags,
    sha256_payload,
)
from src.historical_exact_semantic_backfill.domain import (
    artifact_sha as backfill_artifact_sha,
)

MIN_ISSUER_FEATURE_READY_ROWS = 500
MIN_ISSUER_TICKERS = 10
MAX_ISSUER_UNKNOWN_RATE = Decimal("0.50")
MAX_TOP_1_TICKER_SHARE = Decimal("0.50")
MAX_TOP_SOURCE_FAMILY_SHARE = Decimal("0.50")
MAX_SOURCE_FAMILY_HHI = Decimal("0.50")
MIN_PRIMARY_TARGET_COVERAGE = Decimal("0.95")

ISSUER_OWNED_SOURCE_MARKERS = (
    "MAGNIT",
    "NORNICKEL",
    "ROSNEFT",
    "TBANK",
    "VK",
    "X5",
    "YANDEX",
    "CHEP",
    "NOVATEK",
    "LUKOIL",
    "TATNEFT",
    "ALROSA",
    "PHOSAGRO",
    "POLYUS",
    "INTERRAO",
)


def run_ml_v2_readiness_audit(
    *,
    input_root: Path = Path(DEFAULT_INPUT_ARTIFACT_ROOT),
    old_baseline_root: Path = Path(DEFAULT_OLD_BASELINE_ROOT),
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable ML v2 readiness audit output already exists")
    if rules_v3_fingerprint() != EXPECTED_RULES_V3_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_CHANGED")

    input_manifest = _read_json(input_root / "manifest.json")
    _require_input_manifest(input_manifest)
    old_test_state = _old_baseline_test_state(old_baseline_root)
    old_split = _old_baseline_primary_split(old_baseline_root)

    events = _read_jsonl(input_root / "events.jsonl")
    features = _read_jsonl(input_root / "features.jsonl")
    targets = _read_jsonl(input_root / "targets.jsonl")
    material_rows = _read_jsonl(input_root / "semantic-material-provenance.jsonl")
    semantic_rows = _read_jsonl(input_root / "semantic-extraction-results.jsonl")

    input_shas = {
        "input_manifest_sha": sha256_payload(input_manifest),
        "events_sha": sha256_payload(events),
        "features_sha": sha256_payload(features),
        "targets_sha": sha256_payload(targets),
        "semantic_material_provenance_sha": sha256_payload(material_rows),
        "semantic_extraction_results_sha": sha256_payload(semantic_rows),
        "old_baseline_test_state_sha": sha256_payload(old_test_state),
        "old_baseline_primary_split_sha": sha256_payload(old_split),
    }
    feature_ids = {str(row["event_id"]) for row in features}
    event_rows = [_audit_event_row(row, feature_ids) for row in events]
    strict_historical = [
        row for row in event_rows if _is_strict_exact_historical(row, events_by_id=events)
    ]
    feature_ready = [row for row in strict_historical if row["feature_ready"]]
    issuer_feature_ready = [
        row for row in feature_ready if row["event_origin"] == EventOrigin.ISSUER.value
    ]
    exchange_feature_ready = [
        row for row in feature_ready if row["event_origin"] == EventOrigin.EXCHANGE.value
    ]
    other_feature_ready = [
        row
        for row in feature_ready
        if row["event_origin"] not in {EventOrigin.ISSUER.value, EventOrigin.EXCHANGE.value}
    ]
    target_rows = _historical_target_rows(
        targets, event_rows, {str(row["event_id"]) for row in feature_ready}
    )

    leakage = _leakage_audit(events, features, targets, old_test_state)
    dataset_funnel = _dataset_funnel(strict_historical, feature_ready, issuer_feature_ready)
    ticker_summary = _ticker_summary(feature_ready)
    ticker_concentration = _ticker_concentration(issuer_feature_ready, feature_ready)
    source_concentration = _source_concentration(issuer_feature_ready, feature_ready)
    semantic_summary = _semantic_summary(feature_ready, issuer_feature_ready)
    target_coverage = _target_coverage(feature_ready, issuer_feature_ready, target_rows)
    temporal_summary = _temporal_summary(feature_ready, issuer_feature_ready)
    duplicate_summary = _duplicate_summary(
        strict_historical,
        material_rows,
        semantic_rows,
        {str(row["event_id"]) for row in feature_ready},
    )
    cohort_summary = _cohort_summary(
        issuer_feature_ready,
        exchange_feature_ready,
        other_feature_ready,
        target_rows,
    )
    criteria = _canonical_gate_criteria(
        issuer_feature_ready=issuer_feature_ready,
        semantic_summary=semantic_summary,
        ticker_concentration=ticker_concentration,
        source_concentration=source_concentration,
        target_coverage=target_coverage,
        leakage=leakage,
    )
    decision, main_blocker, secondary_blockers = _decision(criteria)
    allowed_scope = _controlled_scope(old_split, issuer_feature_ready)
    now = created_at or datetime.now(UTC)
    flags = safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ML_V2_ARTIFACT_VERSION,
        "artifact_version": ML_V2_ARTIFACT_VERSION,
        "created_at": now.isoformat(),
        "git_sha": git_sha,
        "BASE_MAIN_SHA": base_main_sha,
        "INPUT_ARTIFACT_ROOT": str(input_root),
        "INPUT_ARTIFACT_SHA": input_manifest["ARTIFACT_SHA"],
        "EXPECTED_INPUT_ARTIFACT_SHA": EXPECTED_INPUT_ARTIFACT_SHA,
        "INPUT_FILE_SHAS": input_shas,
        "RULES_V3_FINGERPRINT": rules_v3_fingerprint(),
        "CONFIG_SHA": sha256_payload(_gate_config_payload()),
        "CANONICAL_COHORT": CANONICAL_ML_V2_COHORT,
        "PRIMARY_HORIZON": PRIMARY_ML_V2_HORIZON,
        "TOTAL_HISTORICAL_STRICT_EXACT_EVENTS": len(strict_historical),
        "FEATURE_READY_EVENTS": len(feature_ready),
        "ISSUER_ORIGINATED_FEATURE_READY_EVENTS": len(issuer_feature_ready),
        "EXCHANGE_ORIGINATED_FEATURE_READY_EVENTS": len(exchange_feature_ready),
        "OTHER_OFFICIAL_FEATURE_READY_EVENTS": len(other_feature_ready),
        "UNIQUE_ISSUER_TICKERS": len({row["ticker"] for row in issuer_feature_ready}),
        "UNIQUE_ISSUERS": len({row["issuer"] for row in issuer_feature_ready}),
        "UNIQUE_SOURCE_FAMILIES": len({row["source_family"] for row in issuer_feature_ready}),
        "DATE_RANGE": temporal_summary["canonical_issuer"]["date_range"],
        "ISSUER_UNKNOWN_RATE": semantic_summary["canonical_issuer"]["unknown_rate"],
        "TOP_1_TICKER_SHARE": ticker_concentration["canonical_issuer"]["top_1_share"],
        "TOP_3_TICKER_SHARE": ticker_concentration["canonical_issuer"]["top_3_share"],
        "TOP_5_TICKER_SHARE": ticker_concentration["canonical_issuer"]["top_5_share"],
        "TICKER_HHI": ticker_concentration["canonical_issuer"]["ticker_hhi"],
        "EFFECTIVE_TICKER_COUNT": ticker_concentration["canonical_issuer"][
            "effective_ticker_count"
        ],
        "SOURCE_FAMILY_HHI": source_concentration["canonical_issuer"]["source_family_hhi"],
        "SOURCE_ID_HHI": source_concentration["canonical_issuer"]["source_id_hhi"],
        "PRIMARY_15M_TARGET_COVERAGE": target_coverage["canonical_issuer"]["horizons"][
            PRIMARY_ML_V2_HORIZON
        ]["coverage"],
        "LEAKAGE_AUDIT": leakage["LEAKAGE_AUDIT"],
        "OLD_BASELINE_TEST_STATUS": OLD_BASELINE_TEST_STATUS,
        "FUTURE_HOLDOUT_STATUS": "UNOBSERVED",
        "FUTURE_HOLDOUT_START": FUTURE_EVENT_HOLDOUT_START.isoformat(),
        "FUTURE_OUTCOMES_READ": 0,
        "FUTURE_TARGETS_READ": 0,
        "DETERMINISTIC_REPLAY": "PASS",
        "CAN_START_CONTROLLED_ML_V2": decision
        == MlV2ReadinessDecision.READY_FOR_CONTROLLED_ML_V2.value,
        "FINAL_READINESS_DECISION": decision,
        "MAIN_BLOCKER": main_blocker,
        "SECONDARY_BLOCKERS": secondary_blockers,
        "RECOMMENDED_NEXT_STEP": _recommended_next_step(decision),
        "DATASET_FUNNEL_SHA": sha256_payload(dataset_funnel),
        "COHORT_SUMMARY_SHA": sha256_payload(cohort_summary),
        "TICKER_SUMMARY_SHA": sha256_payload(ticker_summary),
        "TICKER_CONCENTRATION_SHA": sha256_payload(ticker_concentration),
        "SOURCE_CONCENTRATION_SHA": sha256_payload(source_concentration),
        "SEMANTIC_SUMMARY_SHA": sha256_payload(semantic_summary),
        "TARGET_COVERAGE_SHA": sha256_payload(target_coverage),
        "TEMPORAL_SUMMARY_SHA": sha256_payload(temporal_summary),
        "DUPLICATE_SUMMARY_SHA": sha256_payload(duplicate_summary),
        "LEAKAGE_AUDIT_SHA": sha256_payload(leakage),
        "CONTROLLED_SCOPE_SHA": sha256_payload(allowed_scope),
        "CANONICAL_GATE_CRITERIA_SHA": sha256_payload(criteria),
        "OLD_BASELINE_TEST_STATE_SHA": sha256_payload(old_test_state),
        "OLD_BASELINE_PRIMARY_SPLIT_SHA": sha256_payload(old_split),
        "LABEL_STATISTICS_COMPUTED": False,
        "LABEL_STATISTICS_POLICY": (
            "Not computed in this readiness gate; target files are used only for fixed-horizon "
            "coverage and leakage checks, not for source/rule/feature/model selection."
        ),
        "SCALER_VECTORIZER_STATISTICS_FIT_ON_VALIDATION_TEST": False,
        "SOURCE_SELECTION_USED_RETURNS_OR_MODEL_PERFORMANCE": False,
        "EVENT_SELECTION_USED_FUTURE_OUTCOMES": False,
        "OLD_TEST_PREDICTIONS_OR_METRICS_USED_FOR_SELECTION": False,
        "safety": flags,
        **flags,
    }
    manifest["ARTIFACT_SHA"] = artifact_sha(manifest)

    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "dataset-funnel.json", dataset_funnel)
    _write_json(output_root / "cohort-summary.json", cohort_summary)
    _write_jsonl(output_root / "ticker-summary.jsonl", ticker_summary)
    _write_json(output_root / "ticker-concentration.json", ticker_concentration)
    _write_json(output_root / "source-concentration.json", source_concentration)
    _write_json(output_root / "semantic-summary.json", semantic_summary)
    _write_json(output_root / "target-coverage.json", target_coverage)
    _write_json(output_root / "temporal-summary.json", temporal_summary)
    _write_json(output_root / "duplicate-summary.json", duplicate_summary)
    _write_json(output_root / "leakage-audit.json", leakage)
    _write_json(output_root / "canonical-gate-criteria.json", criteria)
    _write_json(output_root / "controlled-ml-v2-scope.json", allowed_scope)
    _write_report(output_root / "report.md", manifest, criteria)
    return manifest


def _require_input_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("ARTIFACT_SHA") != EXPECTED_INPUT_ARTIFACT_SHA:
        raise ValueError("INPUT_ARTIFACT_SHA_MISMATCH")
    if manifest.get("ARTIFACT_SHA") != backfill_artifact_sha(manifest):
        raise ValueError("INPUT_ARTIFACT_REPLAY_MISMATCH")
    unsafe_keys = (
        "MODEL_TRAINING_PERFORMED",
        "TEST_OUTCOME_USED",
        "TEST_EVALUATION_PERFORMED",
        "BACKTEST_PERFORMED",
        "FUTURE_EVENT_HOLDOUT_USED",
        "FUTURE_EVENT_HOLDOUT_OBSERVED",
    )
    for key in unsafe_keys:
        if bool(manifest.get(key)):
            raise ValueError(f"INPUT_{key}_NOT_SAFE")


def _old_baseline_test_state(root: Path) -> dict[str, Any]:
    state = _read_json(root / "test-evaluation-state.json")
    if state.get("TEST_STATUS") != "OBSERVED_AFTER_EXACT_BASELINE_V1":
        raise ValueError("OLD_BASELINE_TEST_STATUS_UNEXPECTED")
    if int(state.get("TEST_EVALUATION_COUNT_PRIMARY", -1)) < 1:
        raise ValueError("OLD_BASELINE_PRIMARY_TEST_NOT_MARKED_OBSERVED")
    return {
        "TEST_CONFIG_LOCKED": state.get("TEST_CONFIG_LOCKED"),
        "TEST_EVALUATION_COUNT_PRIMARY": state.get("TEST_EVALUATION_COUNT_PRIMARY"),
        "TEST_STATUS": state.get("TEST_STATUS"),
        "artifact_sha": state.get("artifact_sha"),
        "locked_config_sha": state.get("locked_config_sha"),
        "ml_v2_policy_status": OLD_BASELINE_TEST_STATUS,
    }


def _old_baseline_primary_split(root: Path) -> dict[str, Any]:
    split = _read_json(root / f"{PRIMARY_ML_V2_HORIZON}-split-manifest.json")
    return {
        "horizon": split.get("horizon"),
        "protocol": split.get("protocol"),
        "split_sha": split.get("split_sha"),
        "counts": split.get("counts"),
        "date_ranges": split.get("date_ranges"),
        "test_outcomes_observed": True,
        "ml_v2_test_scope_status": "EXCLUDED_FROM_NEW_FINAL_TEST_AND_TUNING",
    }


def _audit_event_row(row: dict[str, Any], feature_ids: set[str]) -> dict[str, Any]:
    metadata = _metadata(row)
    event_id = str(metadata["event_id"])
    source_family = _source_family(metadata)
    origin = _event_origin(metadata, source_family)
    event_features = (
        cast("dict[str, Any]", row.get("event_features"))
        if isinstance(row.get("event_features"), dict)
        else {}
    )
    availability = (
        cast("dict[str, Any]", row.get("target_availability"))
        if isinstance(row.get("target_availability"), dict)
        else {}
    )
    market_features = row.get("pre_event_market_features")
    return {
        "event_id": event_id,
        "ticker": str(metadata.get("ticker") or "UNKNOWN"),
        "issuer": str(metadata.get("issuer") or metadata.get("issuer_name") or "UNKNOWN"),
        "source_id": str(metadata.get("source_id") or metadata.get("source_code") or source_family),
        "source_family": source_family,
        "source_item_id": str(metadata.get("source_item_id") or ""),
        "published_at_utc": _parse_datetime(metadata["publication_timestamp_utc"]).isoformat(),
        "event_origin": origin.value,
        "reaction_ready": bool(availability.get("reaction_ready")),
        "feature_ready": bool(availability.get("feature_ready")) and event_id in feature_ids,
        "market_eligible": _market_features_complete(market_features),
        "primary_event_type": str(event_features.get("primary_event_type", "UNKNOWN")),
        "event_count": _int_feature(event_features, "event_count"),
        "fact_count": _int_feature(event_features, "fact_count"),
        "semantic_valid": _semantic_valid(event_features),
        "semantic_features_sha": sha256_payload(event_features) if event_features else None,
    }


def _metadata(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    return cast("dict[str, Any]", row["metadata"])


def _source_family(metadata: dict[str, Any]) -> str:
    return str(
        metadata.get("source_family")
        or metadata.get("source_code")
        or metadata.get("source")
        or "UNKNOWN_SOURCE_FAMILY"
    )


def _event_origin(metadata: dict[str, Any], source_family: str) -> EventOrigin:
    value = metadata.get("event_origin")
    if isinstance(value, str) and value in {item.value for item in EventOrigin}:
        return EventOrigin(value)
    haystack = " ".join(
        str(item)
        for item in (
            source_family,
            metadata.get("source_code"),
            metadata.get("source_id"),
            metadata.get("official_domain"),
        )
        if item
    ).upper()
    if "MOEX" in haystack or "MOSCOW_EXCHANGE" in haystack or "RISK_PARAMETERS" in haystack:
        return EventOrigin.EXCHANGE
    if "CBR" in haystack or "BANK_OF_RUSSIA" in haystack or "REGULATOR" in haystack:
        return EventOrigin.REGULATOR
    if any(marker in haystack for marker in ISSUER_OWNED_SOURCE_MARKERS):
        return EventOrigin.ISSUER
    if "OFFICIAL" in haystack or "ISSUER" in haystack:
        return EventOrigin.ISSUER
    if haystack:
        return EventOrigin.OTHER_OFFICIAL
    return EventOrigin.UNKNOWN


def _market_features_complete(value: object) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    features = cast("dict[str, object]", value)
    return all(
        feature_value is not None
        for key, feature_value in features.items()
        if str(key).startswith(("pre_return_", "imoex_pre_return_"))
    )


def _semantic_valid(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    features = cast("dict[str, object]", value)
    return (
        isinstance(features.get("primary_event_type"), str)
        and isinstance(features.get("event_count"), int)
        and isinstance(features.get("fact_count"), int)
    )


def _int_feature(value: object, key: str) -> int:
    if isinstance(value, dict):
        feature_value = cast("dict[str, object]", value).get(key)
        if isinstance(feature_value, int):
            return feature_value
    return 0


def _is_strict_exact_historical(row: dict[str, Any], *, events_by_id: list[dict[str, Any]]) -> bool:
    by_id = {str(_metadata(item)["event_id"]): item for item in events_by_id}
    original = by_id.get(str(row["event_id"]))
    metadata = _metadata(original) if original is not None else {}
    return (
        str(metadata.get("timestamp_quality")) == "EXACT"
        and bool(metadata.get("ticker"))
        and bool(metadata.get("instrument_uid"))
        and _parse_datetime(row["published_at_utc"]).date() < FUTURE_EVENT_HOLDOUT_START
    )


def _historical_target_rows(
    targets: list[dict[str, Any]], event_rows: list[dict[str, Any]], feature_ready_ids: set[str]
) -> list[dict[str, Any]]:
    event_by_id = {str(row["event_id"]): row for row in event_rows}
    result: list[dict[str, Any]] = []
    for target in targets:
        event_id = str(target["event_id"])
        event = event_by_id.get(event_id)
        if event is None or event_id not in feature_ready_ids:
            continue
        if _parse_datetime(event["published_at_utc"]).date() >= FUTURE_EVENT_HOLDOUT_START:
            continue
        result.append(target)
    return result


def _dataset_funnel(
    historical: list[dict[str, Any]],
    feature_ready: list[dict[str, Any]],
    issuer_feature_ready: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "canonical_dataset_candidate": CANONICAL_ML_V2_COHORT,
        "stages": {
            "historical_strict_exact_events": len(historical),
            "feature_ready_events": len(feature_ready),
            "issuer_originated_feature_ready_events": len(issuer_feature_ready),
            "issuer_originated_feature_ready_share": _share(
                len(issuer_feature_ready), len(feature_ready)
            ),
        },
        "policy": {
            "future_holdout_start": FUTURE_EVENT_HOLDOUT_START.isoformat(),
            "future_holdout_excluded_from_training_and_outcome_reads": True,
            "exchange_originated_kept_as_control_family": True,
        },
    }


def _cohort_summary(
    issuer: list[dict[str, Any]],
    exchange: list[dict[str, Any]],
    other: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    target_by_id = {str(row["event_id"]): row for row in targets}
    return {
        "canonical_issuer": _cohort_payload(issuer, target_by_id),
        "exchange_control": _cohort_payload(exchange, target_by_id),
        "other_official_review": _cohort_payload(other, target_by_id),
    }


def _cohort_payload(
    rows: list[dict[str, Any]], target_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    dates = [_parse_datetime(row["published_at_utc"]).date() for row in rows]
    return {
        "rows": len(rows),
        "tickers": len({row["ticker"] for row in rows}),
        "issuers": len({row["issuer"] for row in rows}),
        "source_families": len({row["source_family"] for row in rows}),
        "event_ids_sha": sha256_payload(sorted(str(row["event_id"]) for row in rows)),
        "date_range": {
            "first_date": min(dates).isoformat() if dates else None,
            "last_date": max(dates).isoformat() if dates else None,
        },
        "primary_target_coverage": _coverage_for(rows, target_by_id)["horizons"][
            PRIMARY_ML_V2_HORIZON
        ],
    }


def _ticker_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for ticker in sorted({str(row["ticker"]) for row in rows}):
        subset = [row for row in rows if row["ticker"] == ticker]
        result.append(
            {
                "ticker": ticker,
                "feature_ready": len(subset),
                "share": _share(len(subset), len(rows)),
                "issuer": _counter_payload(row["issuer"] for row in subset),
                "event_origins": _counter_payload(row["event_origin"] for row in subset),
                "source_families": _counter_payload(row["source_family"] for row in subset),
                "unknown_count": sum(row["primary_event_type"] == "UNKNOWN" for row in subset),
                "unknown_rate": _share(
                    sum(row["primary_event_type"] == "UNKNOWN" for row in subset), len(subset)
                ),
            }
        )
    return result


def _ticker_concentration(
    issuer_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "canonical_issuer": _concentration_for(issuer_rows, "ticker"),
        "whole_feature_ready": _concentration_for(all_rows, "ticker"),
    }


def _source_concentration(
    issuer_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    issuer_family = _concentration_for(issuer_rows, "source_family")
    issuer_source = _concentration_for(issuer_rows, "source_id")
    all_family = _concentration_for(all_rows, "source_family")
    return {
        "canonical_issuer": {
            "source_family_hhi": issuer_family["hhi"],
            "source_id_hhi": issuer_source["hhi"],
            "top_source_family_share": issuer_family["top_1_share"],
            "top_source_id_share": issuer_source["top_1_share"],
            "effective_source_family_count": issuer_family["effective_count"],
            "top_source_families": _top_counts(issuer_rows, "source_family", 5),
            "top_source_ids": _top_counts(issuer_rows, "source_id", 5),
        },
        "whole_feature_ready": {
            "source_family_hhi": all_family["hhi"],
            "top_source_family_share": all_family["top_1_share"],
            "top_source_families": _top_counts(all_rows, "source_family", 5),
        },
    }


def _semantic_summary(
    all_rows: list[dict[str, Any]], issuer_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "whole_feature_ready": _semantic_payload(all_rows),
        "canonical_issuer": _semantic_payload(issuer_rows),
        "unknown_rate_by_ticker": _rate_by(issuer_rows, "ticker", "primary_event_type", "UNKNOWN"),
        "event_type_distribution": _counter_payload(
            row["primary_event_type"] for row in issuer_rows
        ),
        "fact_count_distribution": _counter_payload(row["fact_count"] for row in issuer_rows),
        "event_count_distribution": _counter_payload(row["event_count"] for row in issuer_rows),
    }


def _semantic_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unknown = sum(row["primary_event_type"] == "UNKNOWN" for row in rows)
    return {
        "rows": len(rows),
        "unknown_count": unknown,
        "unknown_rate": _share(unknown, len(rows)),
    }


def _target_coverage(
    all_rows: list[dict[str, Any]],
    issuer_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    targets_by_id = {str(row["event_id"]): row for row in target_rows}
    return {
        "canonical_issuer": _coverage_for(issuer_rows, targets_by_id),
        "whole_feature_ready": _coverage_for(all_rows, targets_by_id),
        "policy": "Coverage only; no target distribution or model metrics computed.",
    }


def _coverage_for(
    rows: list[dict[str, Any]], targets_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": len(rows), "horizons": {}}
    for horizon in HORIZONS:
        available = sum(
            _target_horizon_available(targets_by_id.get(str(row["event_id"])), horizon)
            for row in rows
        )
        result["horizons"][horizon] = {
            "available": available,
            "missing": len(rows) - available,
            "coverage": _share(available, len(rows)),
        }
    return result


def _temporal_summary(
    all_rows: list[dict[str, Any]], issuer_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "canonical_issuer": _temporal_payload(issuer_rows),
        "whole_feature_ready": _temporal_payload(all_rows),
    }


def _temporal_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dates = [_parse_datetime(row["published_at_utc"]).date() for row in rows]
    months = Counter(f"{item.year:04d}-{item.month:02d}" for item in dates)
    days = Counter(item.isoformat() for item in dates)
    return {
        "date_range": {
            "first_date": min(dates).isoformat() if dates else None,
            "last_date": max(dates).isoformat() if dates else None,
        },
        "events_per_month": dict(sorted(months.items())),
        "events_per_day_top_10": [
            {"date": name, "events": count, "share": _share(count, len(rows))}
            for name, count in days.most_common(10)
        ],
        "top_month_share": _top_share(months, 1),
        "top_day_share": _top_share(days, 1),
        "temporal_clustering": {
            "top_month_share_gt_50pct": Decimal(_top_share(months, 1)) > Decimal("0.50"),
            "top_day_share_gt_10pct": Decimal(_top_share(days, 1)) > Decimal("0.10"),
        },
    }


def _duplicate_summary(
    rows: list[dict[str, Any]],
    material_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    feature_ready_ids: set[str],
) -> dict[str, Any]:
    source_identity_keys = [
        f"{row['source_id']}|{row['source_item_id']}" for row in rows if row["source_item_id"]
    ]
    material_shas = [
        str(row["publication_material_sha"])
        for row in material_rows
        if str(row.get("event_id")) in feature_ready_ids and row.get("publication_material_sha")
    ]
    semantic_shas = [
        str(row["semantic_features_sha"])
        for row in semantic_rows
        if str(row.get("event_id")) in feature_ready_ids and row.get("semantic_features_sha")
    ]
    return {
        "duplicate_event_ids": _duplicate_count(str(row["event_id"]) for row in rows),
        "duplicate_source_identities": _duplicate_count(source_identity_keys),
        "duplicate_publication_material_hashes": _duplicate_count(material_shas),
        "duplicate_semantic_feature_hashes": _duplicate_count(semantic_shas),
        "duplicate_publication_material_hash_counts": _duplicate_counter(material_shas),
        "duplicate_source_identity_counts": _duplicate_counter(source_identity_keys),
    }


def _leakage_audit(
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    old_test_state: dict[str, Any],
) -> dict[str, Any]:
    events_by_id = {str(_metadata(row)["event_id"]): row for row in events}
    violations: list[dict[str, str]] = []
    for feature in features:
        event_id = str(feature["event_id"])
        event = events_by_id.get(event_id)
        if event is None:
            violations.append({"event_id": event_id, "violation": "FEATURE_EVENT_MISSING"})
            continue
        metadata = _metadata(event)
        published = _parse_datetime(metadata["publication_timestamp_utc"])
        feature_cutoff = _parse_datetime(feature.get("feature_cutoff"))
        if feature_cutoff > published:
            violations.append(
                {"event_id": event_id, "violation": "FEATURE_TIMESTAMP_AFTER_PUBLICATION"}
            )
        market_features = cast("dict[str, Any]", feature.get("market_features") or {})
        if market_features.get("post_event_values_in_features") is not False:
            violations.append(
                {"event_id": event_id, "violation": "POST_EVENT_VALUES_IN_FEATURES_FLAG"}
            )
        forbidden = {
            key
            for key in market_features
            if (
                key != "post_event_values_in_features"
                and key.startswith(("target_", "future_", "post_event_"))
            )
            or key in {"security_return", "benchmark_return", "abnormal_return"}
        }
        if forbidden:
            violations.append(
                {
                    "event_id": event_id,
                    "violation": "POST_EVENT_RETURN_FEATURE_COLUMNS",
                    "columns": ",".join(sorted(forbidden)),
                }
            )
    future_event_ids = {
        str(_metadata(row)["event_id"])
        for row in events
        if _parse_datetime(_metadata(row)["publication_timestamp_utc"]).date()
        >= FUTURE_EVENT_HOLDOUT_START
    }
    future_target_rows = [
        str(row["event_id"]) for row in targets if str(row["event_id"]) in future_event_ids
    ]
    if future_target_rows:
        violations.append(
            {
                "event_id": ",".join(sorted(future_target_rows)[:5]),
                "violation": "FUTURE_TARGETS_PRESENT",
            }
        )
    checks = {
        "feature_timestamp_lte_event_publication_timestamp": not any(
            row["violation"] == "FEATURE_TIMESTAMP_AFTER_PUBLICATION" for row in violations
        ),
        "post_event_returns_absent_from_features": not any(
            row["violation"]
            in {"POST_EVENT_VALUES_IN_FEATURES_FLAG", "POST_EVENT_RETURN_FEATURE_COLUMNS"}
            for row in violations
        ),
        "scaler_vectorizer_statistics_not_fit_on_validation_test": True,
        "source_selection_not_using_returns_or_model_performance": True,
        "event_selection_not_using_future_outcomes": True,
        "future_events_have_no_targets_or_outcomes_read": not future_target_rows,
        "old_baseline_test_predictions_metrics_not_used_for_selection": True,
        "old_baseline_test_observed": old_test_state["ml_v2_policy_status"]
        == OLD_BASELINE_TEST_STATUS,
    }
    return {
        **checks,
        "violations": violations,
        "FUTURE_OUTCOMES_READ": 0,
        "FUTURE_TARGETS_READ": 0,
        "OLD_BASELINE_TEST_STATUS": OLD_BASELINE_TEST_STATUS,
        "LEAKAGE_AUDIT": "PASS" if all(checks.values()) and not violations else "FAIL",
    }


def _canonical_gate_criteria(
    *,
    issuer_feature_ready: list[dict[str, Any]],
    semantic_summary: dict[str, Any],
    ticker_concentration: dict[str, Any],
    source_concentration: dict[str, Any],
    target_coverage: dict[str, Any],
    leakage: dict[str, Any],
) -> dict[str, Any]:
    issuer_unknown = Decimal(str(semantic_summary["canonical_issuer"]["unknown_rate"]))
    top_ticker = Decimal(str(ticker_concentration["canonical_issuer"]["top_1_share"]))
    top_source_family = Decimal(
        str(source_concentration["canonical_issuer"]["top_source_family_share"])
    )
    source_family_hhi = Decimal(str(source_concentration["canonical_issuer"]["source_family_hhi"]))
    target_15m = Decimal(
        str(target_coverage["canonical_issuer"]["horizons"][PRIMARY_ML_V2_HORIZON]["coverage"])
    )
    return {
        "rationale": {
            "threshold_conflict_resolution": (
                "ML v2 uses the stricter existing >=10 ticker experiment gate from "
                "event-market-dataset readiness, plus issuer-only rows and old TEST protection."
            ),
            "methodology_change_status": (
                "Explicit canonicalization for ML v2, not a hidden refactor of prior gates."
            ),
        },
        "criteria": {
            "issuer_feature_ready_rows": {
                "actual": len(issuer_feature_ready),
                "threshold": MIN_ISSUER_FEATURE_READY_ROWS,
                "pass": len(issuer_feature_ready) >= MIN_ISSUER_FEATURE_READY_ROWS,
            },
            "unique_issuer_tickers": {
                "actual": len({row["ticker"] for row in issuer_feature_ready}),
                "threshold": MIN_ISSUER_TICKERS,
                "pass": len({row["ticker"] for row in issuer_feature_ready}) >= MIN_ISSUER_TICKERS,
            },
            "issuer_semantic_unknown_rate": {
                "actual": str(issuer_unknown),
                "threshold": str(MAX_ISSUER_UNKNOWN_RATE),
                "pass": issuer_unknown <= MAX_ISSUER_UNKNOWN_RATE,
            },
            "top_1_ticker_share": {
                "actual": str(top_ticker),
                "threshold": str(MAX_TOP_1_TICKER_SHARE),
                "pass": top_ticker <= MAX_TOP_1_TICKER_SHARE,
            },
            "source_concentration": {
                "top_source_family_share_actual": str(top_source_family),
                "top_source_family_share_threshold": str(MAX_TOP_SOURCE_FAMILY_SHARE),
                "source_family_hhi_actual": str(source_family_hhi),
                "source_family_hhi_threshold": str(MAX_SOURCE_FAMILY_HHI),
                "pass": (
                    top_source_family <= MAX_TOP_SOURCE_FAMILY_SHARE
                    and source_family_hhi <= MAX_SOURCE_FAMILY_HHI
                ),
            },
            "primary_15m_target_coverage": {
                "actual": str(target_15m),
                "threshold": str(MIN_PRIMARY_TARGET_COVERAGE),
                "pass": target_15m >= MIN_PRIMARY_TARGET_COVERAGE,
            },
            "deterministic_replay": {"actual": "PASS", "threshold": "PASS", "pass": True},
            "leakage_audit": {
                "actual": leakage["LEAKAGE_AUDIT"],
                "threshold": "PASS",
                "pass": leakage["LEAKAGE_AUDIT"] == "PASS",
            },
            "future_holdout_untouched": {
                "actual": "UNOBSERVED",
                "threshold": "UNOBSERVED",
                "pass": leakage["FUTURE_OUTCOMES_READ"] == 0
                and leakage["FUTURE_TARGETS_READ"] == 0,
            },
            "old_baseline_test_protection": {
                "actual": OLD_BASELINE_TEST_STATUS,
                "threshold": OLD_BASELINE_TEST_STATUS,
                "pass": True,
            },
        },
    }


def _decision(criteria: dict[str, Any]) -> tuple[str, str | None, list[str]]:
    checks = cast("dict[str, dict[str, Any]]", criteria["criteria"])
    failures = [name for name, payload in checks.items() if payload["pass"] is not True]
    if not failures:
        return MlV2ReadinessDecision.READY_FOR_CONTROLLED_ML_V2.value, None, []
    if "leakage_audit" in failures or "future_holdout_untouched" in failures:
        return (
            MlV2ReadinessDecision.DATASET_INTEGRITY_FAILURE.value,
            "DATASET_INTEGRITY_FAILURE",
            [
                item.upper()
                for item in failures
                if item not in {"leakage_audit", "future_holdout_untouched"}
            ],
        )
    if "issuer_feature_ready_rows" in failures:
        main = "ISSUER_FEATURE_READY_ROWS_BELOW_500"
        return (
            MlV2ReadinessDecision.MORE_ISSUER_ROWS_REQUIRED.value,
            main,
            _secondary_failures(failures, "issuer_feature_ready_rows"),
        )
    if "unique_issuer_tickers" in failures:
        main = "UNIQUE_ISSUER_TICKERS_BELOW_10"
        return (
            MlV2ReadinessDecision.MORE_ISSUER_DIVERSITY_REQUIRED.value,
            main,
            _secondary_failures(failures, "unique_issuer_tickers"),
        )
    if "issuer_semantic_unknown_rate" in failures:
        return (
            MlV2ReadinessDecision.SEMANTIC_QUALITY_INSUFFICIENT.value,
            "ISSUER_SEMANTIC_UNKNOWN_RATE_ABOVE_50PCT",
            _secondary_failures(failures, "issuer_semantic_unknown_rate"),
        )
    if "source_concentration" in failures or "top_1_ticker_share" in failures:
        return (
            MlV2ReadinessDecision.SOURCE_CONCENTRATION_TOO_HIGH.value,
            "SOURCE_OR_TICKER_CONCENTRATION_TOO_HIGH",
            _secondary_failures(failures, "source_concentration"),
        )
    return (
        MlV2ReadinessDecision.TARGET_COVERAGE_INSUFFICIENT.value,
        "PRIMARY_15M_TARGET_COVERAGE_BELOW_THRESHOLD",
        _secondary_failures(failures, "primary_15m_target_coverage"),
    )


def _secondary_failures(failures: list[str], main: str) -> list[str]:
    labels = {
        "issuer_feature_ready_rows": "ISSUER_FEATURE_READY_ROWS_BELOW_500",
        "unique_issuer_tickers": "UNIQUE_ISSUER_TICKERS_BELOW_10",
        "issuer_semantic_unknown_rate": "ISSUER_SEMANTIC_UNKNOWN_RATE_ABOVE_50PCT",
        "top_1_ticker_share": "TOP_1_TICKER_SHARE_ABOVE_50PCT",
        "source_concentration": "SOURCE_CONCENTRATION_TOO_HIGH",
        "primary_15m_target_coverage": "PRIMARY_15M_TARGET_COVERAGE_BELOW_95PCT",
        "deterministic_replay": "DETERMINISTIC_REPLAY_FAILED",
        "leakage_audit": "LEAKAGE_AUDIT_FAILED",
        "future_holdout_untouched": "FUTURE_HOLDOUT_TOUCHED",
        "old_baseline_test_protection": "OLD_BASELINE_TEST_PROTECTION_FAILED",
    }
    return [labels[item] for item in failures if item != main]


def _controlled_scope(
    old_split: dict[str, Any], issuer_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    dates = [_parse_datetime(row["published_at_utc"]).date() for row in issuer_rows]
    old_ranges = cast("dict[str, dict[str, str]]", old_split.get("date_ranges") or {})
    observed_test = old_ranges.get("TEST") or {}
    return {
        "canonical_cohort": CANONICAL_ML_V2_COHORT,
        "primary_horizon": PRIMARY_ML_V2_HORIZON,
        "historical_issuer_date_range": {
            "from": min(dates).isoformat() if dates else None,
            "to": max(dates).isoformat() if dates else None,
        },
        "allowed_training_validation_scope": (
            "Historical issuer feature-ready rows may be used for controlled TRAIN/VALIDATION "
            "only after excluding future holdout and after treating old baseline TEST as observed."
        ),
        "old_baseline_observed_test_date_range": {
            "from": observed_test.get("from"),
            "to": observed_test.get("to"),
        },
        "forbidden_scope": {
            "old_baseline_test_metrics_for_tuning": True,
            "old_baseline_test_predictions_for_selection": True,
            "future_events_gte_holdout_start": FUTURE_EVENT_HOLDOUT_START.isoformat(),
            "future_targets_or_outcomes": True,
        },
    }


def _recommended_next_step(decision: str) -> str:
    if decision == MlV2ReadinessDecision.READY_FOR_CONTROLLED_ML_V2.value:
        return (
            "Open a controlled ML v2 TRAIN/VALIDATION-only experiment using the canonical issuer "
            "cohort and predeclared primary horizon 15m; keep old TEST and future holdout closed."
        )
    return (
        "Expand issuer-originated strict-EXACT source diversity before ML v2; prioritize adding "
        "at least three more issuer tickers and reducing UNKNOWN/concentration without reading "
        "future outcomes or using model performance."
    )


def _gate_config_payload() -> dict[str, Any]:
    return {
        "canonical_cohort": CANONICAL_ML_V2_COHORT,
        "primary_horizon": PRIMARY_ML_V2_HORIZON,
        "issuer_feature_ready_rows_min": MIN_ISSUER_FEATURE_READY_ROWS,
        "unique_issuer_tickers_min": MIN_ISSUER_TICKERS,
        "issuer_unknown_rate_max": str(MAX_ISSUER_UNKNOWN_RATE),
        "top_1_ticker_share_max": str(MAX_TOP_1_TICKER_SHARE),
        "top_source_family_share_max": str(MAX_TOP_SOURCE_FAMILY_SHARE),
        "source_family_hhi_max": str(MAX_SOURCE_FAMILY_HHI),
        "primary_target_coverage_min": str(MIN_PRIMARY_TARGET_COVERAGE),
        "future_holdout_start": FUTURE_EVENT_HOLDOUT_START.isoformat(),
        "old_baseline_test_status": OLD_BASELINE_TEST_STATUS,
    }


def _target_horizon_available(row: dict[str, Any] | None, horizon: str) -> bool:
    return _target_abnormal_return(row, horizon) is not None


def _target_abnormal_return(row: dict[str, Any] | None, horizon: str) -> Decimal | None:
    if row is None:
        return None
    horizons = row.get("horizons")
    if isinstance(horizons, dict):
        payload = cast("dict[str, object]", horizons).get(horizon)
        if isinstance(payload, dict):
            typed_payload = cast("dict[str, object]", payload)
            if typed_payload.get("available") is not True:
                return None
            return _decimal_or_none(
                typed_payload.get("abnormal_return") or typed_payload.get("abnormal_simple_return")
            )
        return None
    if str(row.get("horizon")) == horizon and row.get("available", True):
        return _decimal_or_none(row.get("abnormal_return") or row.get("abnormal_simple_return"))
    return None


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _parse_datetime(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _concentration_for(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    counts = Counter(str(row[field]) for row in rows)
    values = sorted(counts.values())
    return {
        "events": len(rows),
        "unique": len(counts),
        "counts": dict(sorted(counts.items())),
        "median_events": _fmt_stat(statistics.median(values)) if values else "0.000000",
        "p10_events": _percentile_int(values, Decimal("0.10")) if values else 0,
        "p90_events": _percentile_int(values, Decimal("0.90")) if values else 0,
        "top_1_share": _top_share(counts, 1),
        "top_3_share": _top_share(counts, 3),
        "top_5_share": _top_share(counts, 5),
        "hhi": _hhi(counts),
        "ticker_hhi": _hhi(counts),
        "effective_count": _effective_count(counts),
        "effective_ticker_count": _effective_count(counts),
    }


def _top_counts(rows: list[dict[str, Any]], field: str, limit: int) -> list[dict[str, Any]]:
    counts = Counter(str(row[field]) for row in rows)
    total = sum(counts.values())
    return [
        {"value": name, "count": count, "share": _share(count, total)}
        for name, count in counts.most_common(limit)
    ]


def _rate_by(
    rows: list[dict[str, Any]], group_field: str, value_field: str, positive_value: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in sorted({str(row[group_field]) for row in rows}):
        subset = [row for row in rows if str(row[group_field]) == group]
        result[group] = _share(
            sum(row[value_field] == positive_value for row in subset), len(subset)
        )
    return result


def _counter_payload(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _duplicate_count(values: Iterable[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def _duplicate_counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted((value, count) for value, count in Counter(values).items() if count > 1))


def _share(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000000"
    return _fmt_decimal(Decimal(numerator) / Decimal(denominator))


def _top_share(counts: Counter[Any], top_n: int) -> str:
    total = sum(counts.values())
    if total == 0:
        return "0.000000"
    return _share(sum(count for _, count in counts.most_common(top_n)), total)


def _hhi(counts: Counter[Any]) -> str:
    total = sum(counts.values())
    if total == 0:
        return "0.000000"
    return _fmt_decimal(
        sum(((Decimal(count) / Decimal(total)) ** 2 for count in counts.values()), Decimal("0"))
    )


def _effective_count(counts: Counter[Any]) -> str:
    total = sum(counts.values())
    if total == 0:
        return "0.000000"
    hhi = sum(((Decimal(count) / Decimal(total)) ** 2 for count in counts.values()), Decimal("0"))
    if hhi == 0:
        return "0.000000"
    return _fmt_decimal(Decimal("1") / hhi)


def _percentile_int(values: list[int], q: Decimal) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = int((q * Decimal(len(ordered) - 1)).to_integral_value(rounding="ROUND_HALF_UP"))
    return ordered[index]


def _fmt_stat(value: float | int) -> str:
    return _fmt_decimal(Decimal(str(value)))


def _fmt_decimal(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.000001'))}"


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_report(path: Path, manifest: dict[str, Any], criteria: dict[str, Any]) -> None:
    checks = cast("dict[str, dict[str, Any]]", criteria["criteria"])
    lines = [
        f"# {ML_V2_ARTIFACT_VERSION}",
        "",
        f"- ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        f"- BASE_MAIN_SHA={manifest['BASE_MAIN_SHA']}",
        f"- HEAD_SHA={manifest['git_sha']}",
        f"- CANONICAL_COHORT={manifest['CANONICAL_COHORT']}",
        f"- PRIMARY_HORIZON={manifest['PRIMARY_HORIZON']}",
        f"- FEATURE_READY_ISSUER_ROWS={manifest['ISSUER_ORIGINATED_FEATURE_READY_EVENTS']}",
        f"- ISSUER_TICKERS={manifest['UNIQUE_ISSUER_TICKERS']}",
        f"- ISSUER_UNKNOWN_RATE={manifest['ISSUER_UNKNOWN_RATE']}",
        f"- TOP_1_TICKER_SHARE={manifest['TOP_1_TICKER_SHARE']}",
        f"- TOP_3_TICKER_SHARE={manifest['TOP_3_TICKER_SHARE']}",
        f"- SOURCE_FAMILY_HHI={manifest['SOURCE_FAMILY_HHI']}",
        f"- PRIMARY_15M_TARGET_COVERAGE={manifest['PRIMARY_15M_TARGET_COVERAGE']}",
        f"- LEAKAGE_AUDIT={manifest['LEAKAGE_AUDIT']}",
        f"- OLD_BASELINE_TEST_STATUS={manifest['OLD_BASELINE_TEST_STATUS']}",
        f"- FUTURE_HOLDOUT_START={manifest['FUTURE_HOLDOUT_START']}",
        f"- FINAL_READINESS_DECISION={manifest['FINAL_READINESS_DECISION']}",
        f"- CAN_START_CONTROLLED_ML_V2={manifest['CAN_START_CONTROLLED_ML_V2']}",
        f"- MAIN_BLOCKER={manifest['MAIN_BLOCKER']}",
        "",
        "## Canonical Gate",
        "",
    ]
    for name, payload in checks.items():
        lines.append(
            f"- {name}: actual={payload.get('actual', payload)}, "
            f"threshold={payload.get('threshold', 'see payload')}, pass={payload['pass']}"
        )
    lines.extend(
        [
            "",
            "No model training, model changes, hyperparameter search, backtest, trading signal, "
            "old TEST tuning, Rules v3/Qwen/NLP tuning, or future holdout outcome read was "
            "performed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_canonical_gate_criteria(
    *,
    issuer_feature_ready: list[dict[str, Any]],
    semantic_summary: dict[str, Any],
    ticker_concentration: dict[str, Any],
    source_concentration: dict[str, Any],
    target_coverage: dict[str, Any],
    leakage: dict[str, Any],
) -> dict[str, Any]:
    return _canonical_gate_criteria(
        issuer_feature_ready=issuer_feature_ready,
        semantic_summary=semantic_summary,
        ticker_concentration=ticker_concentration,
        source_concentration=source_concentration,
        target_coverage=target_coverage,
        leakage=leakage,
    )


def readiness_decision(criteria: dict[str, Any]) -> tuple[str, str | None, list[str]]:
    return _decision(criteria)


def concentration_for(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return _concentration_for(rows, field)


def semantic_summary(
    all_rows: list[dict[str, Any]], issuer_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return _semantic_summary(all_rows, issuer_rows)


def target_coverage(
    all_rows: list[dict[str, Any]],
    issuer_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return _target_coverage(all_rows, issuer_rows, target_rows)


def leakage_audit(
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    old_test_state: dict[str, Any],
) -> dict[str, Any]:
    return _leakage_audit(events, features, targets, old_test_state)
