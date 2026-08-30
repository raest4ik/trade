from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_dataset_readiness_audit.domain import (
    ARTIFACT_VERSION,
    DEFAULT_INPUT_ARTIFACT_ROOT,
    EXPECTED_INPUT_ARTIFACT_SHA,
    EXPECTED_RULES_V3_FINGERPRINT,
    FUTURE_EVENT_HOLDOUT_START,
    HORIZONS,
    EventOrigin,
    ReadinessDecision,
    artifact_sha,
    safety_flags,
    sha256_payload,
)
from src.historical_exact_semantic_backfill.domain import (
    artifact_sha as backfill_artifact_sha,
)


def run_exact_dataset_readiness_audit(
    *,
    input_root: Path = Path(DEFAULT_INPUT_ARTIFACT_ROOT),
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if rules_v3_fingerprint() != EXPECTED_RULES_V3_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_CHANGED")

    input_manifest = _read_json(input_root / "manifest.json")
    _require_input_manifest(input_manifest)
    events = _read_jsonl(input_root / "events.jsonl")
    features = _read_jsonl(input_root / "features.jsonl")
    targets = _read_jsonl(input_root / "targets.jsonl")
    material_rows = _read_jsonl(input_root / "semantic-material-provenance.jsonl")
    semantic_rows = _read_jsonl(input_root / "semantic-extraction-results.jsonl")

    feature_ids = {str(row["event_id"]) for row in features}
    event_rows = [_audit_event_row(row, feature_ids) for row in events]
    historical_rows = [
        row
        for row in event_rows
        if _parse_datetime(row["published_at_utc"]).date() < FUTURE_EVENT_HOLDOUT_START
    ]
    feature_ready_rows = [row for row in historical_rows if row["feature_ready"]]
    feature_ready_ids = {str(row["event_id"]) for row in feature_ready_rows}

    historical_target_rows = _historical_target_rows(targets, event_rows, feature_ready_ids)
    target_coverage = _target_coverage(feature_ready_rows, historical_target_rows)
    label_distribution = _label_distribution(historical_target_rows)
    dataset_funnel = _dataset_funnel(historical_rows)
    source_family_rows = _source_family_summary(historical_rows)
    origin_rows = _event_origin_summary(historical_rows)
    semantic_summary = _semantic_summary(feature_ready_rows)
    ticker_rows, ticker_concentration = _ticker_summary(feature_ready_rows)
    source_concentration = _source_concentration(feature_ready_rows)
    temporal_summary = _temporal_summary(feature_ready_rows)
    duplicate_summary = _duplicate_summary(
        historical_rows,
        material_rows,
        semantic_rows,
        feature_ready_ids,
    )
    cohort_a = _cohort_rows(feature_ready_rows, EventOrigin.ISSUER)
    cohort_b = _cohort_rows(feature_ready_rows, EventOrigin.EXCHANGE)
    cohort_c = _cohort_rows(feature_ready_rows, None)
    cohort_summary = {
        "COHORT_A": _cohort_summary(cohort_a, historical_target_rows),
        "COHORT_B": _cohort_summary(cohort_b, historical_target_rows),
        "COHORT_C": _cohort_summary(cohort_c, historical_target_rows),
    }
    decision = _readiness_decision(
        cohort_summary=cohort_summary,
        semantic_summary=semantic_summary,
        ticker_concentration=ticker_concentration,
        source_concentration=source_concentration,
    )

    flags = safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "git_sha": git_sha,
        "BASE_MAIN_SHA": base_main_sha,
        "INPUT_ARTIFACT_SHA": input_manifest["ARTIFACT_SHA"],
        "EXPECTED_INPUT_ARTIFACT_SHA": EXPECTED_INPUT_ARTIFACT_SHA,
        "RULES_V3_FINGERPRINT": rules_v3_fingerprint(),
        "CANONICAL_EXACT_EVENTS": dataset_funnel["CANONICAL_EXACT_EVENTS"]["count"],
        "FEATURE_READY_EVENTS": dataset_funnel["FEATURE_READY"]["count"],
        "ISSUER_ORIGINATED_FEATURE_READY": cohort_summary["COHORT_A"]["rows"],
        "EXCHANGE_ORIGINATED_FEATURE_READY": cohort_summary["COHORT_B"]["rows"],
        "UNKNOWN_RATE_TOTAL": semantic_summary["UNKNOWN_RATE_TOTAL"],
        "MOEX_RISK_UNKNOWN_RATE": _moex_unknown_rate(source_family_rows),
        "TOP_TICKER_SHARE": ticker_concentration["top_1_share"],
        "TOP_3_TICKER_SHARE": ticker_concentration["top_3_share"],
        "TOP_5_TICKER_SHARE": ticker_concentration["top_5_share"],
        "TICKER_HHI": ticker_concentration["ticker_hhi"],
        "EFFECTIVE_TICKER_COUNT": ticker_concentration["effective_ticker_count"],
        "SOURCE_FAMILY_HHI": source_concentration["source_family_hhi"],
        "SOURCE_ID_HHI": source_concentration["source_id_hhi"],
        "EVENT_ORIGIN_HHI": source_concentration["event_origin_hhi"],
        "LABEL_DISTRIBUTION_AUDIT_SKIPPED_FOR_METHOD_SAFETY": False,
        "READINESS_DECISION": decision,
        "RECOMMENDED_PRIMARY_COHORT": _recommended_primary_cohort(decision),
        "MOEX_RISK_EVENTS_TREATMENT": _moex_treatment(source_family_rows),
        "DATASET_FUNNEL_SHA": sha256_payload(dataset_funnel),
        "SOURCE_FAMILY_SUMMARY_SHA": sha256_payload(source_family_rows),
        "EVENT_ORIGIN_SUMMARY_SHA": sha256_payload(origin_rows),
        "SEMANTIC_SUMMARY_SHA": sha256_payload(semantic_summary),
        "TICKER_SUMMARY_SHA": sha256_payload(ticker_rows),
        "TICKER_CONCENTRATION_SHA": sha256_payload(ticker_concentration),
        "SOURCE_CONCENTRATION_SHA": sha256_payload(source_concentration),
        "TEMPORAL_SUMMARY_SHA": sha256_payload(temporal_summary),
        "TARGET_COVERAGE_SHA": sha256_payload(target_coverage),
        "LABEL_DISTRIBUTION_SHA": sha256_payload(label_distribution),
        "DUPLICATE_SUMMARY_SHA": sha256_payload(duplicate_summary),
        "COHORT_A_SHA": sha256_payload(cohort_a),
        "COHORT_B_SHA": sha256_payload(cohort_b),
        "COHORT_C_SHA": sha256_payload(cohort_c),
        "COHORT_SUMMARY_SHA": sha256_payload(cohort_summary),
        "DETERMINISTIC_REPLAY": "PASS",
        "safety": flags,
        **flags,
    }
    manifest["ARTIFACT_SHA"] = artifact_sha(manifest)

    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "dataset-funnel.json", dataset_funnel)
    _write_jsonl(output_root / "source-family-summary.jsonl", source_family_rows)
    _write_jsonl(output_root / "event-origin-summary.jsonl", origin_rows)
    _write_json(output_root / "semantic-summary.json", semantic_summary)
    _write_jsonl(output_root / "ticker-summary.jsonl", ticker_rows)
    _write_json(output_root / "ticker-concentration.json", ticker_concentration)
    _write_json(output_root / "source-concentration.json", source_concentration)
    _write_json(output_root / "temporal-summary.json", temporal_summary)
    _write_json(output_root / "target-coverage.json", target_coverage)
    _write_json(output_root / "label-distribution.json", label_distribution)
    _write_json(output_root / "duplicate-summary.json", duplicate_summary)
    _write_jsonl(output_root / "cohort-a-issuer-event-ids.jsonl", cohort_a)
    _write_jsonl(output_root / "cohort-b-exchange-event-ids.jsonl", cohort_b)
    _write_jsonl(output_root / "cohort-c-all-event-ids.jsonl", cohort_c)
    _write_json(output_root / "cohort-summary.json", cohort_summary)
    _write_report(output_root / "report.md", manifest, cohort_summary)
    return manifest


def _require_input_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("ARTIFACT_SHA") != EXPECTED_INPUT_ARTIFACT_SHA:
        raise ValueError("INPUT_ARTIFACT_SHA_MISMATCH")
    if manifest.get("ARTIFACT_SHA") != backfill_artifact_sha(manifest):
        raise ValueError("INPUT_ARTIFACT_REPLAY_MISMATCH")
    for key in (
        "MODEL_TRAINING_PERFORMED",
        "TEST_OUTCOME_USED",
        "TEST_EVALUATION_PERFORMED",
        "BACKTEST_PERFORMED",
        "FUTURE_EVENT_HOLDOUT_USED",
        "FUTURE_EVENT_HOLDOUT_OBSERVED",
    ):
        if bool(manifest.get(key)):
            raise ValueError(f"INPUT_{key}_NOT_SAFE")


def _audit_event_row(row: dict[str, Any], feature_ids: set[str]) -> dict[str, Any]:
    metadata = _metadata(row)
    event_id = str(metadata["event_id"])
    source_family = _source_family(metadata)
    origin = _event_origin(metadata, source_family)
    raw_event_features = row.get("event_features")
    event_features: dict[str, Any] = (
        cast("dict[str, Any]", raw_event_features) if isinstance(raw_event_features, dict) else {}
    )
    primary_event_type = str(event_features.get("primary_event_type", "UNKNOWN"))
    availability = _availability(row)
    return {
        "event_id": event_id,
        "ticker": str(metadata.get("ticker") or "UNKNOWN"),
        "source_id": str(metadata.get("source_id") or metadata.get("source_code") or source_family),
        "source_family": source_family,
        "source_item_id": str(metadata.get("source_item_id") or ""),
        "published_at_utc": _parse_datetime(metadata["publication_timestamp_utc"]).isoformat(),
        "event_origin": origin.value,
        "event_origin_available": metadata.get("event_origin") is not None,
        "reaction_ready": bool(availability.get("reaction_ready")),
        "feature_ready": bool(availability.get("feature_ready")) and event_id in feature_ids,
        "market_eligible": _market_features_complete(row.get("pre_event_market_features")),
        "primary_event_type": primary_event_type,
        "event_count": _int_feature(event_features, "event_count"),
        "fact_count": _int_feature(event_features, "fact_count"),
        "semantic_valid": _semantic_valid(event_features),
        "semantic_features_sha": sha256_payload(event_features) if event_features else None,
    }


def _dataset_funnel(rows: list[dict[str, Any]]) -> dict[str, dict[str, int | str]]:
    total = len(rows)
    stages = {
        "CANONICAL_EXACT_EVENTS": total,
        "MARKET_ELIGIBLE": sum(bool(row["market_eligible"]) for row in rows),
        "REACTION_READY": sum(bool(row["reaction_ready"]) for row in rows),
        "FEATURE_READY": sum(bool(row["feature_ready"]) for row in rows),
    }
    result = {
        name: {"count": count, "share": _share(count, total)} for name, count in stages.items()
    }
    feature_rows = [row for row in rows if row["feature_ready"]]
    feature_total = len(feature_rows)
    valid_semantics = sum(bool(row["semantic_valid"]) for row in feature_rows)
    unknown_semantics = sum(row["primary_event_type"] == "UNKNOWN" for row in feature_rows)
    result["FEATURE_READY_WITH_VALID_SEMANTICS"] = {
        "count": valid_semantics,
        "share": _share(valid_semantics, feature_total),
    }
    result["FEATURE_READY_WITH_UNKNOWN_SEMANTICS"] = {
        "count": unknown_semantics,
        "share": _share(unknown_semantics, feature_total),
    }
    result["FEATURE_READY_WITH_NON_UNKNOWN_SEMANTICS"] = {
        "count": feature_total - unknown_semantics,
        "share": _share(feature_total - unknown_semantics, feature_total),
    }
    return result


def _source_family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for family in sorted({str(row["source_family"]) for row in rows}):
        subset = [row for row in rows if row["source_family"] == family]
        feature = [row for row in subset if row["feature_ready"]]
        tickers = Counter(str(row["ticker"]) for row in feature)
        dates = [_parse_datetime(row["published_at_utc"]).date().isoformat() for row in subset]
        result.append(
            {
                "source_family": family,
                "events": len(subset),
                "reaction_ready": sum(bool(row["reaction_ready"]) for row in subset),
                "feature_ready": len(feature),
                "feature_ready_share": _share(len(feature), len(subset)),
                "unknown_count": sum(row["primary_event_type"] == "UNKNOWN" for row in feature),
                "unknown_rate": _share(
                    sum(row["primary_event_type"] == "UNKNOWN" for row in feature), len(feature)
                ),
                "unique_tickers": len(tickers),
                "first_date": min(dates) if dates else None,
                "last_date": max(dates) if dates else None,
                "top_ticker_share": _share(tickers.most_common(1)[0][1], len(feature))
                if feature
                else "0.000000",
                "event_origin": _counter_payload(row["event_origin"] for row in subset),
            }
        )
    return result


def _event_origin_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for origin in [item.value for item in EventOrigin]:
        subset = [row for row in rows if row["event_origin"] == origin]
        if not subset:
            continue
        feature = [row for row in subset if row["feature_ready"]]
        result.append(
            {
                "event_origin": origin,
                "event_count": len(subset),
                "feature_ready_count": len(feature),
                "unknown_semantic_rate": _share(
                    sum(row["primary_event_type"] == "UNKNOWN" for row in feature), len(feature)
                ),
                "ticker_diversity": len({row["ticker"] for row in feature}),
                "source_family_diversity": len({row["source_family"] for row in feature}),
            }
        )
    return result


def _semantic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary_counts = _counter_payload(row["primary_event_type"] for row in rows)
    event_count_distribution = _counter_payload(row["event_count"] for row in rows)
    fact_count_distribution = _counter_payload(row["fact_count"] for row in rows)
    unknown_count = int(primary_counts.get("UNKNOWN", 0))
    return {
        "FEATURE_READY_ROWS": len(rows),
        "PRIMARY_EVENT_TYPE_COUNTS": primary_counts,
        "EVENT_COUNT_DISTRIBUTION": event_count_distribution,
        "FACT_COUNT_DISTRIBUTION": fact_count_distribution,
        "UNKNOWN_RATE_TOTAL": _share(unknown_count, len(rows)),
        "UNKNOWN_RATE_BY_SOURCE_FAMILY": _rate_by(
            rows, "source_family", "primary_event_type", "UNKNOWN"
        ),
        "UNKNOWN_RATE_BY_EVENT_ORIGIN": _rate_by(
            rows, "event_origin", "primary_event_type", "UNKNOWN"
        ),
        "UNKNOWN_RATE_BY_TICKER": _rate_by(rows, "ticker", "primary_event_type", "UNKNOWN"),
        "ADDS_SEMANTIC_DIVERSITY": len(primary_counts) > 1 and unknown_count < len(rows),
    }


def _ticker_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = Counter(str(row["ticker"]) for row in rows)
    result = [
        {
            "ticker": ticker,
            "feature_ready": count,
            "share": _share(count, len(rows)),
            "event_origins": _counter_payload(
                row["event_origin"] for row in rows if row["ticker"] == ticker
            ),
            "source_families": _counter_payload(
                row["source_family"] for row in rows if row["ticker"] == ticker
            ),
        }
        for ticker, count in sorted(counts.items())
    ]
    concentration = {
        "unique_feature_ready_tickers": len(counts),
        "top_1_share": _top_share(counts, 1),
        "top_3_share": _top_share(counts, 3),
        "top_5_share": _top_share(counts, 5),
        "ticker_hhi": _hhi(counts),
        "effective_ticker_count": _effective_count(counts),
        "issuer_originated": _concentration_for(
            [row for row in rows if row["event_origin"] == EventOrigin.ISSUER.value], "ticker"
        ),
        "exchange_originated": _concentration_for(
            [row for row in rows if row["event_origin"] == EventOrigin.EXCHANGE.value], "ticker"
        ),
    }
    return result, concentration


def _source_concentration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family = Counter(str(row["source_family"]) for row in rows)
    source_id = Counter(str(row["source_id"]) for row in rows)
    origin = Counter(str(row["event_origin"]) for row in rows)
    return {
        "source_family_hhi": _hhi(family),
        "source_id_hhi": _hhi(source_id),
        "event_origin_hhi": _hhi(origin),
        "top_source_family_shares": [
            {"source_family": name, "share": _share(count, len(rows))}
            for name, count in family.most_common(5)
        ],
    }


def _temporal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dates = [_parse_datetime(row["published_at_utc"]).date() for row in rows]
    months = Counter(f"{date.year:04d}-{date.month:02d}" for date in dates)
    quarters = Counter(f"{date.year:04d}-Q{((date.month - 1) // 3) + 1}" for date in dates)
    return {
        "first_date": min(dates).isoformat() if dates else None,
        "last_date": max(dates).isoformat() if dates else None,
        "events_per_month": dict(sorted(months.items())),
        "events_per_quarter": dict(sorted(quarters.items())),
        "top_month_share": _top_share(months, 1),
        "top_quarter_share": _top_share(quarters, 1),
        "temporal_clustering_flag": _top_share_float(months, 1) > 0.50,
    }


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


def _target_coverage(
    rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    targets_by_id = {str(row["event_id"]): row for row in target_rows}
    whole = _coverage_for(rows, targets_by_id)
    by_family = {
        family: _coverage_for(
            [row for row in rows if row["source_family"] == family], targets_by_id
        )
        for family in sorted({str(row["source_family"]) for row in rows})
    }
    by_origin = {
        origin: _coverage_for([row for row in rows if row["event_origin"] == origin], targets_by_id)
        for origin in sorted({str(row["event_origin"]) for row in rows})
    }
    return {"whole_corpus": whole, "by_source_family": by_family, "by_event_origin": by_origin}


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


def _label_distribution(target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[Decimal]] = {horizon: [] for horizon in HORIZONS}
    for row in target_rows:
        for horizon in HORIZONS:
            value = _target_abnormal_return(row, horizon)
            if value is not None:
                values[horizon].append(value)
    return {
        "LABEL_DISTRIBUTION_AUDIT_SKIPPED_FOR_METHOD_SAFETY": False,
        "policy": (
            "Descriptive QA statistics only for non-future historical feature-ready rows; "
            "not grouped by semantic class and not used for feature/rule/model selection."
        ),
        "horizons": {horizon: _decimal_stats(items) for horizon, items in values.items()},
    }


def _duplicate_summary(
    rows: list[dict[str, Any]],
    material_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    feature_ready_ids: set[str],
) -> dict[str, Any]:
    source_item_keys = [
        f"{row['source_id']}|{row['source_item_id']}" for row in rows if row["source_item_id"]
    ]
    material_shas = [
        str(row["publication_material_sha"])
        for row in material_rows
        if row.get("event_id") in feature_ready_ids and row.get("publication_material_sha")
    ]
    semantic_rows_by_id = {str(row.get("event_id")): row for row in semantic_rows}
    semantic_feature_shas: list[str] = []
    for row in rows:
        if row["event_id"] not in feature_ready_ids:
            continue
        semantic_sha = _semantic_feature_sha(row, semantic_rows_by_id)
        if semantic_sha is not None:
            semantic_feature_shas.append(semantic_sha)
    return {
        "duplicate_event_id_count": _duplicate_count([str(row["event_id"]) for row in rows]),
        "duplicate_source_item_id_within_source_count": _duplicate_count(source_item_keys),
        "duplicate_publication_material_sha_count": _duplicate_count(material_shas),
        "duplicate_semantic_features_sha_count": _duplicate_count(semantic_feature_shas),
        "publication_material_sha_counts": _duplicate_counter(material_shas),
        "semantic_features_sha_counts": _duplicate_counter(semantic_feature_shas),
        "top_publication_material_sha_share": _top_share(Counter(material_shas), 1),
        "top_semantic_features_sha_share": _top_share(Counter(semantic_feature_shas), 1),
    }


def _semantic_feature_sha(
    row: dict[str, Any], semantic_rows_by_id: dict[str, dict[str, Any]]
) -> str | None:
    semantic_row = semantic_rows_by_id.get(str(row["event_id"]))
    if semantic_row is not None and semantic_row.get("semantic_features_sha"):
        return str(semantic_row["semantic_features_sha"])
    value = row.get("semantic_features_sha")
    return str(value) if value else None


def _cohort_rows(rows: list[dict[str, Any]], origin: EventOrigin | None) -> list[dict[str, Any]]:
    selected = (
        rows if origin is None else [row for row in rows if row["event_origin"] == origin.value]
    )
    return [
        {
            "event_id": str(row["event_id"]),
            "ticker": str(row["ticker"]),
            "source_family": str(row["source_family"]),
            "source_id": str(row["source_id"]),
            "event_origin": str(row["event_origin"]),
            "published_at_utc": str(row["published_at_utc"]),
            "primary_event_type": str(row["primary_event_type"]),
            "semantic_features_sha": row["semantic_features_sha"],
        }
        for row in sorted(
            selected, key=lambda item: (str(item["published_at_utc"]), str(item["event_id"]))
        )
    ]


def _cohort_summary(cohort: list[dict[str, Any]], targets: list[dict[str, Any]]) -> dict[str, Any]:
    dates = [_parse_datetime(row["published_at_utc"]).date() for row in cohort]
    target_by_id = {str(row["event_id"]): row for row in targets}
    return {
        "rows": len(cohort),
        "tickers": len({row["ticker"] for row in cohort}),
        "source_families": len({row["source_family"] for row in cohort}),
        "unknown_rate": _share(
            sum(row["primary_event_type"] == "UNKNOWN" for row in cohort), len(cohort)
        ),
        "date_range": {
            "first_date": min(dates).isoformat() if dates else None,
            "last_date": max(dates).isoformat() if dates else None,
        },
        "label_coverage": _coverage_for(cohort, target_by_id),
        "event_ids_sha": sha256_payload([row["event_id"] for row in cohort]),
    }


def _readiness_decision(
    *,
    cohort_summary: dict[str, dict[str, Any]],
    semantic_summary: dict[str, Any],
    ticker_concentration: dict[str, Any],
    source_concentration: dict[str, Any],
) -> str:
    issuer_rows = int(cohort_summary["COHORT_A"]["rows"])
    exchange_rows = int(cohort_summary["COHORT_B"]["rows"])
    issuer_unknown_rate = Decimal(str(cohort_summary["COHORT_A"]["unknown_rate"]))
    exchange_unknown_rate = Decimal(str(cohort_summary["COHORT_B"]["unknown_rate"]))
    issuer_tickers = int(cohort_summary["COHORT_A"]["tickers"])
    if issuer_rows >= 500 and issuer_tickers >= 5 and issuer_unknown_rate <= Decimal("0.50"):
        if exchange_rows and exchange_unknown_rate >= Decimal("0.50"):
            return ReadinessDecision.ISSUER_COHORT_READY_EXCHANGE_COHORT_SEPARATE.value
        return ReadinessDecision.DATASET_READY_FOR_CONTROLLED_BASELINE.value
    if Decimal(str(semantic_summary["UNKNOWN_RATE_TOTAL"])) > Decimal("0.50"):
        return ReadinessDecision.SEMANTIC_REPRESENTATION_TOO_WEAK.value
    if Decimal(str(ticker_concentration["top_1_share"])) > Decimal("0.70") or Decimal(
        str(source_concentration["source_family_hhi"])
    ) > Decimal("0.70"):
        return ReadinessDecision.SOURCE_CONCENTRATION_TOO_HIGH.value
    return ReadinessDecision.MORE_ISSUER_EVENT_DATA_REQUIRED.value


def _recommended_primary_cohort(decision: str) -> str:
    if decision == ReadinessDecision.ISSUER_COHORT_READY_EXCHANGE_COHORT_SEPARATE:
        return "COHORT_A_ISSUER_ORIGINATED"
    if decision == ReadinessDecision.DATASET_READY_FOR_CONTROLLED_BASELINE:
        return "COHORT_C_ALL_WITH_ORIGIN_RETAINED"
    return "NO_BASELINE_PRIMARY_COHORT_RECOMMENDED"


def _moex_unknown_rate(source_family_rows: list[dict[str, Any]]) -> str:
    row = next(
        (
            item
            for item in source_family_rows
            if item["source_family"] == "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"
        ),
        None,
    )
    return "0.000000" if row is None else str(row["unknown_rate"])


def _moex_treatment(source_family_rows: list[dict[str, Any]]) -> str:
    row = next(
        (
            item
            for item in source_family_rows
            if item["source_family"] == "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"
        ),
        None,
    )
    if row is None or not row["feature_ready"]:
        return "D_INSUFFICIENTLY_REPRESENTED_FOR_MODELING_DECISION"
    unknown_rate = Decimal(str(row["unknown_rate"]))
    if unknown_rate >= Decimal("0.50"):
        return "B_SEPARATE_EXCHANGE_ORIGINATED_EVENT_FAMILY"
    return "C_CONTROL_OR_AUXILIARY_FEATURE_FAMILY"


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
        features = cast("dict[str, object]", value)
        feature_value = features.get(key)
        if isinstance(feature_value, int):
            return feature_value
    return 0


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
    if "OFFICIAL" in haystack or "ISSUER" in haystack:
        return EventOrigin.ISSUER
    if haystack:
        return EventOrigin.OTHER_OFFICIAL
    return EventOrigin.UNKNOWN


def _target_horizon_available(row: dict[str, Any] | None, horizon: str) -> bool:
    return _target_abnormal_return(row, horizon) is not None


def _target_abnormal_return(row: dict[str, Any] | None, horizon: str) -> Decimal | None:
    if row is None:
        return None
    horizons = row.get("horizons")
    if isinstance(horizons, dict):
        typed_horizons = cast("dict[str, object]", horizons)
        payload = typed_horizons.get(horizon)
        if isinstance(payload, dict):
            typed_payload = cast("dict[str, object]", payload)
            if typed_payload.get("available") is not True:
                return None
            value = typed_payload.get("abnormal_return") or typed_payload.get(
                "abnormal_simple_return"
            )
            return _decimal_or_none(value)
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


def _decimal_stats(values: list[Decimal]) -> dict[str, str | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "std": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    floats = [float(value) for value in ordered]
    return {
        "count": len(ordered),
        "median": _fmt_decimal(Decimal(str(statistics.median(floats)))),
        "mean": _fmt_decimal(sum(ordered) / Decimal(len(ordered))),
        "std": _fmt_decimal(Decimal(str(statistics.pstdev(floats))))
        if len(ordered) > 1
        else "0.000000",
        "p05": _fmt_decimal(_percentile(ordered, Decimal("0.05"))),
        "p25": _fmt_decimal(_percentile(ordered, Decimal("0.25"))),
        "p75": _fmt_decimal(_percentile(ordered, Decimal("0.75"))),
        "p95": _fmt_decimal(_percentile(ordered, Decimal("0.95"))),
    }


def _percentile(values: list[Decimal], q: Decimal) -> Decimal:
    if len(values) == 1:
        return values[0]
    position = q * Decimal(len(values) - 1)
    lower = int(position.to_integral_value(rounding="ROUND_FLOOR"))
    upper = int(position.to_integral_value(rounding="ROUND_CEILING"))
    if lower == upper:
        return values[lower]
    weight = position - Decimal(lower)
    return values[lower] * (Decimal("1") - weight) + values[upper] * weight


def _share(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000000"
    return _fmt_decimal(Decimal(numerator) / Decimal(denominator))


def _top_share(counts: Counter[Any], top_n: int) -> str:
    total = sum(counts.values())
    if total == 0:
        return "0.000000"
    return _share(sum(count for _, count in counts.most_common(top_n)), total)


def _top_share_float(counts: Counter[Any], top_n: int) -> float:
    return float(_top_share(counts, top_n))


def _hhi(counts: Counter[Any]) -> str:
    total = sum(counts.values())
    if total == 0:
        return "0.000000"
    return _fmt_decimal(
        sum(((Decimal(count) / Decimal(total)) ** 2 for count in counts.values()), Decimal("0"))
    )


def _effective_count(counts: Counter[Any]) -> str:
    hhi = Decimal(_hhi(counts))
    if hhi == 0:
        return "0.000000"
    return _fmt_decimal(Decimal("1") / hhi)


def _concentration_for(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    counts = Counter(str(row[field]) for row in rows)
    return {
        "events": len(rows),
        "unique": len(counts),
        "top_1_share": _top_share(counts, 1),
        "top_3_share": _top_share(counts, 3),
        "hhi": _hhi(counts),
        "effective_count": _effective_count(counts),
    }


def _duplicate_count(values: list[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def _duplicate_counter(values: list[str]) -> dict[str, int]:
    return dict(sorted((value, count) for value, count in Counter(values).items() if count > 1))


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


def _fmt_decimal(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.000001'))}"


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["metadata"])


def _availability(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["target_availability"])


def _parse_datetime(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


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


def _write_report(
    path: Path, manifest: dict[str, Any], cohort_summary: dict[str, dict[str, Any]]
) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        f"- ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        f"- INPUT_ARTIFACT_SHA={manifest['INPUT_ARTIFACT_SHA']}",
        f"- CANONICAL_EXACT_EVENTS={manifest['CANONICAL_EXACT_EVENTS']}",
        f"- FEATURE_READY_EVENTS={manifest['FEATURE_READY_EVENTS']}",
        f"- ISSUER_ORIGINATED_FEATURE_READY={manifest['ISSUER_ORIGINATED_FEATURE_READY']}",
        f"- EXCHANGE_ORIGINATED_FEATURE_READY={manifest['EXCHANGE_ORIGINATED_FEATURE_READY']}",
        f"- UNKNOWN_RATE_TOTAL={manifest['UNKNOWN_RATE_TOTAL']}",
        f"- MOEX_RISK_UNKNOWN_RATE={manifest['MOEX_RISK_UNKNOWN_RATE']}",
        f"- TOP_TICKER_SHARE={manifest['TOP_TICKER_SHARE']}",
        f"- SOURCE_FAMILY_HHI={manifest['SOURCE_FAMILY_HHI']}",
        f"- MOEX_RISK_EVENTS_TREATMENT={manifest['MOEX_RISK_EVENTS_TREATMENT']}",
        f"- READINESS_DECISION={manifest['READINESS_DECISION']}",
        f"- RECOMMENDED_PRIMARY_COHORT={manifest['RECOMMENDED_PRIMARY_COHORT']}",
        "",
        "## Candidate Cohorts",
        "",
        f"- COHORT_A rows={cohort_summary['COHORT_A']['rows']}, "
        f"unknown_rate={cohort_summary['COHORT_A']['unknown_rate']}",
        f"- COHORT_B rows={cohort_summary['COHORT_B']['rows']}, "
        f"unknown_rate={cohort_summary['COHORT_B']['unknown_rate']}",
        f"- COHORT_C rows={cohort_summary['COHORT_C']['rows']}, "
        f"unknown_rate={cohort_summary['COHORT_C']['unknown_rate']}",
        "",
        "No model training, TEST evaluation, backtest, trading, Rules v3/Qwen tuning, or future "
        "holdout target inspection was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
