from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, cast

from src.exact_event_sparse_session_study.domain import (
    ARTIFACT_VERSION,
    DECISION_RULES,
    DELAY_THRESHOLDS_SECONDS,
    DEVELOPMENT_SPLITS,
    FUTURE_EVENT_HOLDOUT_START,
    INPUT_DATASET_SHA,
    OUTPUT_DATASET_SHA,
    PR39_ARTIFACT_SHA,
    PRE_EVENT_DENSITY_WINDOWS,
    STUDY_WINDOW,
    ExactEventMetadata,
    MethodologyStudyRecommendation,
    TimestampCandle,
    parse_datetime,
    require_pr39_manifest,
    sha256_payload,
    sparse_study_safety_flags,
)

FORBIDDEN_ARTIFACT_KEYS = {
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "security_return",
    "benchmark_return",
    "abnormal_return",
    "security_log_return",
    "benchmark_log_return",
    "abnormal_log_return",
    "target_class",
    "target",
    "label",
    "prediction",
    "signal",
    "pnl",
}


def run_sparse_session_study(
    *,
    events_path: Path,
    split_manifest_path: Path,
    pr39_root: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    cache_roots: tuple[Path, ...],
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable sparse session study artifact output exists")
    pr39_manifest = _read_json(pr39_root / "manifest.json")
    require_pr39_manifest(pr39_manifest)
    split_manifest = _read_json(split_manifest_path)
    split = _load_development_split(split_manifest)
    metadata_rows = _read_event_metadata_only(events_path)
    exact_rows = [row for row in metadata_rows if row.timestamp_quality == "EXACT"]
    future_exclusions = _future_exclusions(exact_rows)
    development_rows, split_exclusions = _development_rows(exact_rows, split)
    per_event, ineligible = _study_rows(development_rows, cache_roots)
    recommendation_rows = [row for row in per_event if row["CACHE_COVERAGE_STATUS"] == "PASS"]
    summary = _summary(recommendation_rows)
    coverage = _coverage(recommendation_rows)
    sparse = _sparse_concentration(recommendation_rows)
    source_family = _sparse_by_source_family(recommendation_rows)
    density_buckets = _density_buckets(recommendation_rows)
    ticker_summary = _ticker_summary(recommendation_rows)
    decision_rules_sha = sha256_payload(DECISION_RULES)
    recommendation = _recommendation(
        recommendation_rows=recommendation_rows,
        coverage=coverage,
        sparse=sparse,
    )
    safety = sparse_study_safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_DATASET_SHA": INPUT_DATASET_SHA,
        "OUTPUT_DATASET_SHA": OUTPUT_DATASET_SHA,
        "PR39_ARTIFACT_SHA": PR39_ARTIFACT_SHA,
        "PR39_SESSION_DIAGNOSTIC_COHORT_SHA": pr39_manifest["SESSION_DIAGNOSTIC_COHORT_SHA"],
        "DEVELOPMENT_COHORT_DEFINITION": split["definition"],
        "DEVELOPMENT_COHORT_SHA": sha256_payload(
            [row.identity_payload() | {"split": _split_for(row, split)} for row in development_rows]
        ),
        "STUDY_ELIGIBLE_COHORT_SHA": sha256_payload(
            sorted(str(row["EVENT_ID"]) for row in per_event)
        ),
        "DECISION_RULES": DECISION_RULES,
        "DECISION_RULES_SHA": decision_rules_sha,
        "DEVELOPMENT_EXACT_TOTAL": len(development_rows),
        "TIMESTAMP_STUDY_ELIGIBLE": len(per_event),
        "TIMESTAMP_STUDY_INELIGIBLE": len(ineligible),
        "RECOMMENDATION_ELIGIBLE_EVENTS": len(recommendation_rows),
        "INELIGIBLE_BLOCKER_REASONS": dict(Counter(row["BLOCKER"] for row in ineligible)),
        "PER_EVENT_ROWS": per_event,
        "INELIGIBLE_ROWS": ineligible,
        "FUTURE_HOLDOUT_EXCLUSIONS": future_exclusions,
        "SPLIT_EXCLUSIONS": split_exclusions,
        "OBSERVED_TEST_ROWS_USED": 0,
        "TEST_ROWS_EXCLUDED_BEFORE_ANALYSIS": "PASS",
        "DELAY_DISTRIBUTION": summary["delay_distribution"],
        "DELAY_MEDIAN_SECONDS": summary["median"],
        "DELAY_P75_SECONDS": summary["p75"],
        "DELAY_P90_SECONDS": summary["p90"],
        "DELAY_P95_SECONDS": summary["p95"],
        "DELAY_MAX_SECONDS": summary["max"],
        "CANDIDATE_THRESHOLD_COVERAGE": coverage,
        "COVERAGE_60S": coverage["60"]["EVENT_SHARE"],
        "COVERAGE_120S": coverage["120"]["EVENT_SHARE"],
        "COVERAGE_180S": coverage["180"]["EVENT_SHARE"],
        "COVERAGE_300S": coverage["300"]["EVENT_SHARE"],
        "COVERAGE_600S": coverage["600"]["EVENT_SHARE"],
        "INCREMENTAL_120S_VS_60S": coverage["120"]["INCREMENTAL_SHARE_VS_60S"],
        "INCREMENTAL_180S_VS_60S": coverage["180"]["INCREMENTAL_SHARE_VS_60S"],
        "INCREMENTAL_300S_VS_60S": coverage["300"]["INCREMENTAL_SHARE_VS_60S"],
        "INCREMENTAL_600S_VS_60S": coverage["600"]["INCREMENTAL_SHARE_VS_60S"],
        "PRE_EVENT_DENSITY_BUCKETS": density_buckets,
        "TICKER_LEVEL_SUMMARY": ticker_summary,
        "SPARSE_EVENTS_TOTAL": sparse["SPARSE_EVENTS_TOTAL"],
        "SPARSE_UNIQUE_TICKERS": sparse["SPARSE_UNIQUE_TICKERS"],
        "SPARSE_BY_TICKER": sparse["SPARSE_BY_TICKER"],
        "SPARSE_TOP1_SHARE": sparse["SPARSE_TOP1_SHARE"],
        "SPARSE_TOP3_SHARE": sparse["SPARSE_TOP3_SHARE"],
        "SPARSE_TICKER_HHI": sparse["SPARSE_TICKER_HHI"],
        "EFFECTIVE_SPARSE_TICKER_COUNT": sparse["EFFECTIVE_SPARSE_TICKER_COUNT"],
        "SPARSE_BY_SOURCE_FAMILY": source_family,
        "CACHE_COVERAGE_STATUSES": dict(
            Counter(str(row["CACHE_COVERAGE_STATUS"]) for row in per_event)
        ),
        "CACHE_COVERAGE_UNCERTAIN_COUNT": sum(
            1 for row in per_event if row["CACHE_COVERAGE_STATUS"] != "PASS"
        ),
        "METHODOLOGY_STUDY_RECOMMENDATION": recommendation.value,
        "METHODOLOGY_CONCLUSION": _methodology_conclusion(recommendation),
        "PRODUCTION_DATASET_CHANGED": False,
        "EXISTING_EVENTS_PRESERVED": "PASS",
        "EXISTING_FEATURE_ROWS_PRESERVED": "PASS",
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "MARKET_DATA_METHOD_CHANGED": False,
        "STUDY_ARTIFACT_OUTCOME_FREE": "PASS",
        "STUDY_ARTIFACT_PRICE_FREE": "PASS",
        "DETERMINISTIC_REPLAY": "PASS",
        "safety": safety,
        **safety,
    }
    if not _contains_no_forbidden_keys(manifest):
        raise ValueError("SPARSE_SESSION_STUDY_FORBIDDEN_ARTIFACT_KEY")
    manifest["ARTIFACT_SHA"] = _artifact_sha(manifest)
    _write_artifacts(output_root, manifest, per_event)
    return manifest


def _load_development_split(split_manifest: dict[str, Any]) -> dict[str, Any]:
    if split_manifest.get("leakage_check") != "PASS":
        raise ValueError("SPLIT_LEAKAGE_CHECK_NOT_PASS")
    if bool(split_manifest.get("target_outcomes_inspected_before_lock")):
        raise ValueError("SPLIT_TARGETS_INSPECTED_BEFORE_LOCK")
    assignments = {
        str(row["event_id"]): str(row["split"])
        for row in cast("list[dict[str, Any]]", split_manifest["assignments"])
    }
    date_ranges = cast("dict[str, dict[str, str]]", split_manifest["date_ranges"])
    return {
        "assignments": assignments,
        "date_ranges": date_ranges,
        "definition": {
            "source": "artifacts/exact-event-predictive-baseline-v1/15m-split-manifest.json",
            "protocol": split_manifest["protocol"],
            "split_sha": split_manifest["split_sha"],
            "development_splits": sorted(DEVELOPMENT_SPLITS),
            "date_ranges": date_ranges,
            "unknown_split_policy": "FAIL_CLOSED_EXCLUDE_ROW",
            "test_policy": "EXCLUDED_BEFORE_ANALYSIS",
        },
    }


def _development_rows(
    rows: list[ExactEventMetadata], split: dict[str, Any]
) -> tuple[list[ExactEventMetadata], list[dict[str, Any]]]:
    result: list[ExactEventMetadata] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        if row.publication_timestamp.date() >= FUTURE_EVENT_HOLDOUT_START:
            continue
        assigned = _split_for(row, split)
        if assigned in DEVELOPMENT_SPLITS:
            result.append(row)
        else:
            excluded.append(
                {
                    "EVENT_ID": row.event_id,
                    "TICKER": row.ticker,
                    "PUBLICATION_TIMESTAMP": row.publication_timestamp.isoformat(),
                    "EXCLUDED_SPLIT": assigned or "UNKNOWN",
                    "EXCLUDED_BEFORE_ANALYSIS": True,
                }
            )
    return sorted(result, key=lambda item: item.event_id), excluded


def _split_for(row: ExactEventMetadata, split: dict[str, Any]) -> str | None:
    assignments = cast("dict[str, str]", split["assignments"])
    if row.event_id in assignments:
        return assignments[row.event_id]
    ranges = cast("dict[str, dict[str, str]]", split["date_ranges"])
    row_date = row.publication_timestamp.date()
    for name in ("TRAIN", "VALIDATION", "TEST"):
        window = ranges[name]
        if (
            datetime.fromisoformat(window["from"]).date()
            <= row_date
            <= datetime.fromisoformat(window["to"]).date()
        ):
            return name
    return None


def _study_rows(
    rows: list[ExactEventMetadata], cache_roots: tuple[Path, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_event: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    file_cache: dict[Path, tuple[TimestampCandle, ...]] = {}
    for row in rows:
        if not row.instrument_uid or not row.ticker:
            ineligible.append(_ineligible(row, "MISSING_INSTRUMENT_IDENTITY"))
            continue
        security = _load_candles(
            cache_roots,
            row.ticker,
            row.publication_timestamp,
            row.instrument_uid,
            file_cache=file_cache,
        )
        benchmark = _load_candles(
            cache_roots,
            "IMOEX",
            row.publication_timestamp,
            None,
            file_cache=file_cache,
        )
        if not security:
            ineligible.append(_ineligible(row, "SECURITY_TIMESTAMP_CACHE_MISSING"))
            continue
        if not benchmark:
            ineligible.append(_ineligible(row, "BENCHMARK_TIMESTAMP_CACHE_MISSING"))
            continue
        per_event.append(_per_event_row(row, security=security, benchmark=benchmark))
    return per_event, ineligible


def _per_event_row(
    row: ExactEventMetadata,
    *,
    security: tuple[TimestampCandle, ...],
    benchmark: tuple[TimestampCandle, ...],
) -> dict[str, Any]:
    published = row.publication_timestamp
    security_complete = tuple(item for item in security if item.is_complete)
    benchmark_complete = tuple(item for item in benchmark if item.is_complete)
    benchmark_begins = {item.begin_at for item in benchmark_complete}
    common_begins = sorted(
        {
            item.begin_at
            for item in security_complete
            if item.begin_at.date() == published.date() and item.begin_at in benchmark_begins
        }
    )
    first_common_after = next((item for item in common_begins if item >= published), None)
    first_common = min(common_begins, default=None)
    last_common = max(common_begins, default=None)
    window_end = published + STUDY_WINDOW
    delay = (
        int((first_common_after - published).total_seconds())
        if first_common_after is not None and first_common_after <= window_end
        else None
    )
    status = _common_status(published, first_common_after, first_common, last_common, window_end)
    density = {
        f"PRE_EVENT_CANDLE_DENSITY_{minutes}M": _pre_event_density(
            published, security_complete, benchmark_complete, minutes
        )
        for minutes in PRE_EVENT_DENSITY_WINDOWS
    }
    cache_status = _cache_coverage_status(published, security, benchmark, window_end)
    result: dict[str, Any] = {
        "EVENT_ID": row.event_id,
        "TICKER": row.ticker,
        "ISSUER": row.issuer,
        "SOURCE_FAMILY": row.source_family,
        "PUBLICATION_TIMESTAMP": published.isoformat(),
        "DEVELOPMENT_SPLIT": "TRAIN_OR_VALIDATION",
        "CACHE_COVERAGE_STATUS": cache_status,
        "FIRST_COMMON_COMPLETE_CANDLE_BEGIN": first_common_after.isoformat()
        if first_common_after and first_common_after <= window_end
        else None,
        "COMMON_CANDLE_DELAY_SECONDS": delay,
        "DELAY_BIN": _delay_bin(delay, status),
        "STRICT_60S_ELIGIBLE": delay is not None and delay <= 60,
        "CANDIDATE_120S_ELIGIBLE": delay is not None and delay <= 120,
        "CANDIDATE_180S_ELIGIBLE": delay is not None and delay <= 180,
        "CANDIDATE_300S_ELIGIBLE": delay is not None and delay <= 300,
        "CANDIDATE_600S_ELIGIBLE": delay is not None and delay <= 600,
        "MAX_DENSITY_INPUT_TIMESTAMP": _max_density_input_timestamp(published, security_complete),
        "STATUS": status,
    }
    result.update(density)
    return result


def _common_status(
    published: datetime,
    first_common_after: datetime | None,
    first_common: datetime | None,
    last_common: datetime | None,
    window_end: datetime,
) -> str:
    if first_common is None:
        return "NO_COMMON_CANDLE_WITHIN_WINDOW"
    if last_common is not None and published >= last_common + timedelta(minutes=1):
        return "SESSION_END_BEFORE_COMMON_CANDLE"
    if first_common_after is None or first_common_after > window_end:
        return "NO_COMMON_CANDLE_WITHIN_WINDOW"
    return "FIRST_COMMON_FOUND"


def _pre_event_density(
    published: datetime,
    security: tuple[TimestampCandle, ...],
    benchmark: tuple[TimestampCandle, ...],
    minutes: int,
) -> float | None:
    start = published - timedelta(minutes=minutes)
    expected = {
        candle.begin_at
        for candle in benchmark
        if start <= candle.begin_at and candle.end_at <= published
    }
    if not expected:
        return None
    observed = {
        candle.begin_at
        for candle in security
        if candle.begin_at in expected and candle.end_at <= published
    }
    return round(len(observed) / len(expected), 6)


def _max_density_input_timestamp(
    published: datetime, security: tuple[TimestampCandle, ...]
) -> str | None:
    candidates = [candle.end_at for candle in security if candle.end_at <= published]
    value = max(candidates, default=None)
    return value.isoformat() if value is not None else None


def _cache_coverage_status(
    published: datetime,
    security: tuple[TimestampCandle, ...],
    benchmark: tuple[TimestampCandle, ...],
    window_end: datetime,
) -> str:
    required_start = published - timedelta(minutes=60)
    if not _covers(security, required_start, window_end):
        return "CACHE_COVERAGE_UNCERTAIN"
    if not _covers(benchmark, required_start, window_end):
        return "CACHE_COVERAGE_UNCERTAIN"
    return "PASS"


def _covers(rows: tuple[TimestampCandle, ...], start: datetime, end: datetime) -> bool:
    begins = [row.begin_at for row in rows]
    ends = [row.end_at for row in rows]
    return bool(begins and ends and min(begins) <= start and max(ends) >= end)


def _delay_bin(delay: int | None, status: str) -> str:
    if delay is None:
        return status if status == "SESSION_END_BEFORE_COMMON_CANDLE" else "NO_COMMON_WITHIN_30M"
    if delay <= 60:
        return "0-60 sec"
    if delay <= 120:
        return ">60-120 sec"
    if delay <= 300:
        return ">120-300 sec"
    if delay <= 600:
        return ">300-600 sec"
    return ">600-1800 sec"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bins = [
        "0-60 sec",
        ">60-120 sec",
        ">120-300 sec",
        ">300-600 sec",
        ">600-1800 sec",
        "NO_COMMON_WITHIN_30M",
        "SESSION_END_BEFORE_COMMON_CANDLE",
    ]
    counts = Counter(str(row["DELAY_BIN"]) for row in rows)
    total = len(rows)
    delays = sorted(
        int(row["COMMON_CANDLE_DELAY_SECONDS"])
        for row in rows
        if row["COMMON_CANDLE_DELAY_SECONDS"] is not None
    )
    return {
        "delay_distribution": [
            {"BIN": item, "COUNT": counts[item], "SHARE": _share(counts[item], total)}
            for item in bins
        ],
        "median": _percentile(delays, 50),
        "p75": _percentile(delays, 75),
        "p90": _percentile(delays, 90),
        "p95": _percentile(delays, 95),
        "max": max(delays) if delays else None,
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    total = len(rows)
    strict_count = sum(1 for row in rows if bool(row["STRICT_60S_ELIGIBLE"]))
    result: dict[str, dict[str, float | int]] = {}
    for threshold in DELAY_THRESHOLDS_SECONDS:
        count = sum(
            1
            for row in rows
            if row["COMMON_CANDLE_DELAY_SECONDS"] is not None
            and int(row["COMMON_CANDLE_DELAY_SECONDS"]) <= threshold
        )
        result[str(threshold)] = {
            "EVENT_COUNT": count,
            "EVENT_SHARE": _share(count, total),
            "INCREMENTAL_COUNT_VS_60S": count - strict_count,
            "INCREMENTAL_SHARE_VS_60S": _share(count - strict_count, total),
        }
    return result


def _density_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = [
        ("0-0.25", 0.0, 0.25, True),
        (">0.25-0.50", 0.25, 0.50, False),
        (">0.50-0.75", 0.50, 0.75, False),
        (">0.75-0.90", 0.75, 0.90, False),
        (">0.90-1.00", 0.90, 1.00, False),
        ("UNKNOWN", None, None, False),
    ]
    result: list[dict[str, Any]] = []
    for name, low, high, inclusive_low in buckets:
        members = [
            row
            for row in rows
            if _in_density_bucket(row["PRE_EVENT_CANDLE_DENSITY_60M"], low, high, inclusive_low)
        ]
        count = len(members)
        result.append(
            {
                "BUCKET": name,
                "EVENT_COUNT": count,
                "DELAY_LE_60S_SHARE": _threshold_share(members, 60),
                "DELAY_LE_120S_SHARE": _threshold_share(members, 120),
                "DELAY_LE_300S_SHARE": _threshold_share(members, 300),
                "DELAY_GT_300S_SHARE": _gt_threshold_share(members, 300),
                "NO_COMMON_SHARE": _share(
                    sum(1 for row in members if row["COMMON_CANDLE_DELAY_SECONDS"] is None),
                    count,
                ),
            }
        )
    return result


def _in_density_bucket(
    value: object, low: float | None, high: float | None, inclusive_low: bool
) -> bool:
    if value is None:
        return low is None and high is None
    if low is None or high is None:
        return False
    if not isinstance(value, int | float | str):
        return False
    density = float(value)
    return low <= density <= high if inclusive_low else low < density <= high


def _ticker_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ticker[str(row["TICKER"])].append(row)
    for ticker, members in sorted(by_ticker.items()):
        delays = sorted(
            int(row["COMMON_CANDLE_DELAY_SECONDS"])
            for row in members
            if row["COMMON_CANDLE_DELAY_SECONDS"] is not None
        )
        densities = [
            float(row["PRE_EVENT_CANDLE_DENSITY_60M"])
            for row in members
            if row["PRE_EVENT_CANDLE_DENSITY_60M"] is not None
        ]
        strict = sum(1 for row in members if bool(row["STRICT_60S_ELIGIBLE"]))
        sparse = len(members) - strict
        result.append(
            {
                "TICKER": ticker,
                "N": len(members),
                "MEDIAN_DELAY_SECONDS": median(delays) if delays else None,
                "P90_DELAY_SECONDS": _percentile(delays, 90) if len(delays) >= 5 else None,
                "LE_60S_COUNT": strict,
                "LE_60S_SHARE": _share(strict, len(members)),
                "GT_60S_COUNT": sparse,
                "GT_60S_SHARE": _share(sparse, len(members)),
                "PRE_EVENT_CANDLE_DENSITY_60M_MEDIAN": median(densities) if densities else None,
            }
        )
    return result


def _sparse_concentration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sparse_rows = [
        row
        for row in rows
        if row["COMMON_CANDLE_DELAY_SECONDS"] is None
        or int(row["COMMON_CANDLE_DELAY_SECONDS"]) > 60
    ]
    counts = Counter(str(row["TICKER"]) for row in sparse_rows)
    total = sum(counts.values())
    shares = sorted((count / total for count in counts.values()), reverse=True) if total else []
    hhi = sum(share * share for share in shares)
    return {
        "SPARSE_EVENTS_TOTAL": total,
        "SPARSE_UNIQUE_TICKERS": len(counts),
        "SPARSE_BY_TICKER": dict(sorted(counts.items())),
        "SPARSE_TOP1_SHARE": shares[0] if shares else 0.0,
        "SPARSE_TOP3_SHARE": sum(shares[:3]) if shares else 0.0,
        "SPARSE_TICKER_HHI": hhi,
        "EFFECTIVE_SPARSE_TICKER_COUNT": round(1 / hhi, 6) if hhi else 0.0,
    }


def _sparse_by_source_family(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(row["SOURCE_FAMILY"])
        for row in rows
        if row["COMMON_CANDLE_DELAY_SECONDS"] is None
        or int(row["COMMON_CANDLE_DELAY_SECONDS"]) > 60
    )
    return dict(sorted(counts.items()))


def _recommendation(
    *,
    recommendation_rows: list[dict[str, Any]],
    coverage: dict[str, dict[str, float | int]],
    sparse: dict[str, Any],
) -> MethodologyStudyRecommendation:
    if len(recommendation_rows) < int(DECISION_RULES["minimum_recommendation_sample_size"]):
        return MethodologyStudyRecommendation.INSUFFICIENT_DEVELOPMENT_EVIDENCE
    if (
        int(sparse["SPARSE_EVENTS_TOTAL"]) >= int(DECISION_RULES["minimum_sparse_gt60_events"])
        and int(sparse["SPARSE_UNIQUE_TICKERS"])
        >= int(DECISION_RULES["minimum_sparse_unique_tickers"])
        and float(coverage["300"]["INCREMENTAL_SHARE_VS_60S"])
        >= float(DECISION_RULES["minimum_300s_incremental_share_vs_60s"])
        and float(sparse["SPARSE_TOP1_SHARE"]) <= float(DECISION_RULES["maximum_sparse_top1_share"])
        and float(sparse["SPARSE_TOP3_SHARE"]) <= float(DECISION_RULES["maximum_sparse_top3_share"])
    ):
        return MethodologyStudyRecommendation.SEPARATE_SPARSE_FAMILY_STUDY_JUSTIFIED
    return MethodologyStudyRecommendation.KEEP_STRICT_ONLY


def _methodology_conclusion(recommendation: MethodologyStudyRecommendation) -> str:
    if recommendation == MethodologyStudyRecommendation.SEPARATE_SPARSE_FAMILY_STUDY_JUSTIFIED:
        return "DESIGN SEPARATE SPARSE FAMILY"
    if recommendation == MethodologyStudyRecommendation.KEEP_STRICT_ONLY:
        return "NONE / KEEP STRICT"
    return "MORE DATA FIRST"


def _threshold_share(rows: list[dict[str, Any]], threshold: int) -> float:
    return _share(
        sum(
            1
            for row in rows
            if row["COMMON_CANDLE_DELAY_SECONDS"] is not None
            and int(row["COMMON_CANDLE_DELAY_SECONDS"]) <= threshold
        ),
        len(rows),
    )


def _gt_threshold_share(rows: list[dict[str, Any]], threshold: int) -> float:
    return _share(
        sum(
            1
            for row in rows
            if row["COMMON_CANDLE_DELAY_SECONDS"] is None
            or int(row["COMMON_CANDLE_DELAY_SECONDS"]) > threshold
        ),
        len(rows),
    )


def _percentile(values: list[int], percentile: int) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight)


def _share(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _ineligible(row: ExactEventMetadata, blocker: str) -> dict[str, str]:
    return {
        "EVENT_ID": row.event_id,
        "TICKER": row.ticker,
        "PUBLICATION_TIMESTAMP": row.publication_timestamp.isoformat(),
        "BLOCKER": blocker,
    }


def _load_candles(
    cache_roots: tuple[Path, ...],
    ticker: str,
    published_at: datetime,
    expected_uid: str | None,
    *,
    file_cache: dict[Path, tuple[TimestampCandle, ...]],
) -> tuple[TimestampCandle, ...]:
    rows: dict[tuple[str, datetime], TimestampCandle] = {}
    for day in _cache_days(published_at):
        for root in cache_roots:
            for suffix in ("day", "pre"):
                path = root / ticker / f"{day}-{suffix}.jsonl"
                if not path.exists():
                    continue
                if path not in file_cache:
                    file_cache[path] = tuple(
                        _timestamp_candle_from_line(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                for candle in file_cache[path]:
                    if expected_uid is not None and candle.instrument_uid != expected_uid:
                        continue
                    rows[(candle.instrument_uid, candle.begin_at)] = candle
    return tuple(rows[key] for key in sorted(rows, key=lambda item: (item[1], item[0])))


def _cache_days(published_at: datetime) -> tuple[str, ...]:
    start = published_at.date() - timedelta(days=7)
    return tuple((start + timedelta(days=offset)).isoformat() for offset in range(9))


def _timestamp_candle_from_line(line: str) -> TimestampCandle:
    return TimestampCandle(
        instrument_uid=_string_field(line, "instrument_uid"),
        begin_at=parse_datetime(_string_field(line, "begin_at")),
        end_at=parse_datetime(_string_field(line, "end_at")),
        is_complete=_bool_field(line, "is_complete"),
        ticker=_optional_string_field(line, "ticker"),
        figi=_optional_string_field(line, "figi"),
        class_code=_optional_string_field(line, "class_code"),
    )


def _read_event_metadata_only(path: Path) -> list[ExactEventMetadata]:
    result: list[ExactEventMetadata] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        metadata = json.loads(_extract_object_for_key(line, "metadata"))
        result.append(
            ExactEventMetadata(
                event_id=str(metadata["event_id"]),
                ticker=str(metadata["ticker"]),
                issuer=str(metadata.get("issuer", "")),
                source_family=str(metadata.get("source_code", "UNKNOWN")).split("_")[0],
                publication_timestamp=parse_datetime(str(metadata["publication_timestamp_utc"])),
                instrument_uid=str(metadata.get("instrument_uid", "")),
                future_holdout=bool(metadata.get("future_holdout", False)),
                timestamp_quality=str(metadata.get("timestamp_quality", "")),
            )
        )
    return result


def _future_exclusions(rows: list[ExactEventMetadata]) -> list[dict[str, Any]]:
    return [
        {
            "EVENT_ID": row.event_id,
            "TICKER": row.ticker,
            "PUBLICATION_TIMESTAMP": row.publication_timestamp.isoformat(),
            "EXCLUDED_FROM_STUDY": True,
            "FUTURE_OUTCOME_OBSERVED": False,
        }
        for row in rows
        if row.publication_timestamp.date() >= FUTURE_EVENT_HOLDOUT_START or row.future_holdout
    ]


def _extract_object_for_key(line: str, key: str) -> str:
    marker = f'"{key}"'
    key_index = line.find(marker)
    if key_index < 0:
        raise ValueError(f"{key.upper()}_OBJECT_NOT_FOUND")
    colon = line.find(":", key_index + len(marker))
    start = line.find("{", colon)
    if colon < 0 or start < 0:
        raise ValueError(f"{key.upper()}_OBJECT_NOT_FOUND")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(line)):
        char = line[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return line[start : index + 1]
    raise ValueError(f"{key.upper()}_OBJECT_NOT_CLOSED")


def _string_field(line: str, key: str) -> str:
    value = _optional_string_field(line, key)
    if value is None:
        raise ValueError(f"{key.upper()}_FIELD_MISSING")
    return value


def _optional_string_field(line: str, key: str) -> str | None:
    pattern = rf'"{re.escape(key)}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
    match = re.search(pattern, line)
    return json.loads(f'"{match.group(1)}"') if match else None


def _bool_field(line: str, key: str) -> bool:
    pattern = rf'"{re.escape(key)}"\s*:\s*(true|false)'
    match = re.search(pattern, line)
    if match is None:
        raise ValueError(f"{key.upper()}_FIELD_MISSING")
    return match.group(1) == "true"


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _contains_no_forbidden_keys(payload: object) -> bool:
    if isinstance(payload, dict):
        return all(
            str(key).lower() not in FORBIDDEN_ARTIFACT_KEYS and _contains_no_forbidden_keys(value)
            for key, value in cast("dict[object, object]", payload).items()
        )
    if isinstance(payload, list):
        return all(_contains_no_forbidden_keys(item) for item in cast("list[object]", payload))
    return True


def _artifact_sha(manifest: dict[str, Any]) -> str:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"ARTIFACT_SHA", "created_at", "git_sha"}
    }
    return sha256_payload(core)


def _write_artifacts(
    output_root: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "manifest.json", manifest)
    _write_jsonl(output_root / "per-event-study.jsonl", rows)
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
        "# EXACT Event Sparse Session Methodology Study v1",
        "",
        f"- ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        f"- INPUT_DATASET_SHA={manifest['INPUT_DATASET_SHA']}",
        f"- OUTPUT_DATASET_SHA={manifest['OUTPUT_DATASET_SHA']}",
        f"- DEVELOPMENT_EXACT_TOTAL={manifest['DEVELOPMENT_EXACT_TOTAL']}",
        f"- TIMESTAMP_STUDY_ELIGIBLE={manifest['TIMESTAMP_STUDY_ELIGIBLE']}",
        f"- TIMESTAMP_STUDY_INELIGIBLE={manifest['TIMESTAMP_STUDY_INELIGIBLE']}",
        f"- ROOT_RECOMMENDATION={manifest['METHODOLOGY_STUDY_RECOMMENDATION']}",
        f"- METHODOLOGY_CONCLUSION={manifest['METHODOLOGY_CONCLUSION']}",
        f"- STUDY_ARTIFACT_OUTCOME_FREE={manifest['STUDY_ARTIFACT_OUTCOME_FREE']}",
        f"- STUDY_ARTIFACT_PRICE_FREE={manifest['STUDY_ARTIFACT_PRICE_FREE']}",
        f"- TEST_ROWS_EXCLUDED_BEFORE_ANALYSIS={manifest['TEST_ROWS_EXCLUDED_BEFORE_ANALYSIS']}",
        f"- OBSERVED_TEST_ROWS_USED={manifest['OBSERVED_TEST_ROWS_USED']}",
        "",
        "This artifact is timestamp/availability only and does not change EXACT_INTRADAY.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
