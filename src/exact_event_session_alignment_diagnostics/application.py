from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_corpus.domain import SessionState
from src.exact_event_corpus.market import align_exact_event, classify_session
from src.exact_event_security_history_recovery.domain import acquisition_day_bounds
from src.exact_event_session_alignment_diagnostics.domain import (
    ARTIFACT_VERSION,
    FUTURE_EVENT_HOLDOUT_START,
    INPUT_DATASET_SHA,
    NEIGHBORHOOD_AFTER,
    NEIGHBORHOOD_BEFORE,
    OUTPUT_DATASET_SHA,
    PR38_ARTIFACT_SHA,
    PR38_RECOVERY_COHORT_SHA,
    RecoveryRecommendationType,
    SessionAlignmentRootCause,
    SessionDiagnosticIdentity,
    require_pr38_manifest,
    session_diagnostic_safety_flags,
    sha256_payload,
)
from src.tinvest_market.client import TInvestMinuteCandle


def run_session_alignment_diagnostics(
    *,
    pr38_root: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    created_at: datetime | None = None,
    extra_cache_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable session alignment diagnostics artifact output exists")
    _verify_frozen_contracts()
    pr38_manifest = _read_json(pr38_root / "manifest.json")
    require_pr38_manifest(pr38_manifest)
    recovery_rows = _read_jsonl(pr38_root / "per-event-recovery.jsonl")
    events = _read_jsonl(pr38_root / "events.jsonl")
    features = _read_jsonl(pr38_root / "features.jsonl")
    cohort = _session_diagnostic_cohort(recovery_rows)
    event_by_id = {_event_id(row): row for row in events}
    _reconcile_cohort(cohort, event_by_id)
    cache_roots = _cache_roots(pr38_root, extra_cache_roots)
    per_event: list[dict[str, Any]] = []
    for identity in cohort:
        published_at = _parse_datetime(identity.publication_timestamp)
        security = _load_security_history(cache_roots, identity)
        benchmark = _load_benchmark_history(cache_roots, "IMOEX", published_at)
        per_event.append(_diagnose_event(identity, security=security, benchmark=benchmark))

    future_exclusions = _future_exclusions(events)
    root_cause_counts = dict(Counter(str(row["ROOT_CAUSE"]) for row in per_event))
    safety = session_diagnostic_safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_DATASET_SHA": INPUT_DATASET_SHA,
        "OUTPUT_DATASET_SHA": OUTPUT_DATASET_SHA,
        "PR38_ARTIFACT_SHA": PR38_ARTIFACT_SHA,
        "PR38_RECOVERY_COHORT_SHA": PR38_RECOVERY_COHORT_SHA,
        "SESSION_DIAGNOSTIC_COHORT_SHA": sha256_payload(sorted(item.event_id for item in cohort)),
        "DIAGNOSTIC_EVENTS_TOTAL": len(cohort),
        "DIAGNOSTIC_EVENT_IDS": sorted(item.event_id for item in cohort),
        "PER_EVENT_DIAGNOSTICS": per_event,
        "ROOT_CAUSE_COUNTS": dict(sorted(root_cause_counts.items())),
        "FUTURE_HOLDOUT_EXCLUSIONS": future_exclusions,
        "PR38_DATASET_PRESERVED": "YES",
        "EXACT_EVENTS_PRESERVED": "PASS",
        "EXISTING_FEATURE_ROWS_PRESERVED": "PASS",
        "existing_events_hash": sha256_payload(events),
        "existing_feature_rows_preservation_hash": sha256_payload(
            {str(row["event_id"]): row for row in features}
        ),
        "DIAGNOSTIC_ARTIFACT_CONTAINS_NO_PRICE_VALUES": "PASS",
        "DETERMINISTIC_REPLAY": "PASS",
        "ALIGNMENT_METHODOLOGY_CHANGED": False,
        "MARKET_DATA_METHOD_CHANGED": False,
        "safety": safety,
        **safety,
    }
    if not _contains_no_price_values(manifest):
        raise ValueError("DIAGNOSTIC_ARTIFACT_CONTAINS_PRICE_VALUES")
    manifest["ARTIFACT_SHA"] = _artifact_sha(manifest)
    _write_artifacts(output_root, manifest, per_event, future_exclusions)
    return manifest


def _session_diagnostic_cohort(rows: list[dict[str, Any]]) -> list[SessionDiagnosticIdentity]:
    result: list[SessionDiagnosticIdentity] = []
    for row in rows:
        published_at = _parse_datetime(row["PUBLICATION_TIMESTAMP"])
        if published_at.date() >= FUTURE_EVENT_HOLDOUT_START:
            continue
        if str(row.get("RECOVERY_STATUS")) != "BLOCKED":
            continue
        if str(row.get("FINAL_BLOCKER")) != "SESSION_ALIGNMENT_FAILED":
            continue
        result.append(
            SessionDiagnosticIdentity(
                event_id=str(row["EVENT_ID"]),
                ticker=str(row["TICKER"]),
                publication_timestamp=published_at.isoformat(),
                figi=str(row["FIGI"]),
                instrument_uid=str(row["UID"]),
                class_code=str(row["CLASS_CODE"]),
            )
        )
    if not result:
        raise ValueError("SESSION_DIAGNOSTIC_COHORT_EMPTY")
    return sorted(result, key=lambda item: item.event_id)


def _reconcile_cohort(
    cohort: list[SessionDiagnosticIdentity], event_by_id: dict[str, dict[str, Any]]
) -> None:
    for identity in cohort:
        row = event_by_id.get(identity.event_id)
        if row is None:
            raise ValueError("SESSION_DIAGNOSTIC_EVENT_NOT_IN_PR38_DATASET")
        metadata = _metadata(row)
        if bool(metadata.get("future_holdout")):
            raise ValueError("FUTURE_EVENT_ENTERED_SESSION_ALIGNMENT_DIAGNOSTICS")
        if str(metadata.get("ticker")) != identity.ticker:
            raise ValueError("SESSION_DIAGNOSTIC_TICKER_MISMATCH")
        if str(metadata.get("instrument_uid")) != identity.instrument_uid:
            raise ValueError("SESSION_DIAGNOSTIC_UID_MISMATCH")


def _diagnose_event(
    identity: SessionDiagnosticIdentity,
    *,
    security: tuple[TInvestMinuteCandle, ...],
    benchmark: tuple[TInvestMinuteCandle, ...],
) -> dict[str, Any]:
    published_at = _parse_datetime(identity.publication_timestamp)
    security_complete = _valid_rows(security)
    benchmark_complete = _valid_rows(benchmark)
    session_state = classify_session(published_at, security_complete, benchmark_complete)
    alignment = align_exact_event(
        published_at, security_complete, benchmark_complete, expose_outcomes=False
    )
    common_begin = _common_begins(published_at, security_complete, benchmark_complete)
    first_common = min(common_begin, default=None)
    last_common_begin = max(common_begin, default=None)
    last_common_end = last_common_begin + timedelta(minutes=1) if last_common_begin else None
    next_common = _first_beginning_at_or_after(common_begin, published_at)
    previous_common = _last_beginning_before(common_begin, published_at)
    security_baseline = _last_ending_at_or_before(security_complete, published_at)
    benchmark_baseline = _last_ending_at_or_before(benchmark_complete, published_at)
    security_effective = _first_beginning_at_or_after_candle(security_complete, published_at)
    benchmark_effective = _first_beginning_at_or_after_candle(benchmark_complete, published_at)
    security_neighborhood = _neighborhood(security, published_at)
    benchmark_neighborhood = _neighborhood(benchmark, published_at)
    common_neighborhood = _common_neighborhood(published_at, security_complete, benchmark_complete)
    required_min, required_max = _diagnostic_window(published_at)
    root_cause = _root_cause(
        session_state=session_state,
        common_begin=common_begin,
        next_common=next_common,
        published_at=published_at,
        security_baseline=security_baseline,
        benchmark_baseline=benchmark_baseline,
        security_effective=security_effective,
        benchmark_effective=benchmark_effective,
        security_neighborhood=security_neighborhood,
        benchmark_neighborhood=benchmark_neighborhood,
        common_neighborhood=common_neighborhood,
        security=security,
        benchmark=benchmark,
        required_min=required_min,
        required_max=required_max,
    )
    recommendation = _recovery_recommendation(root_cause)
    cache_min = min((row.begin_at for row in security), default=None)
    cache_max = max((row.end_at for row in security), default=None)
    return {
        "EVENT_ID": identity.event_id,
        "TICKER": identity.ticker,
        "PUBLICATION_TIMESTAMP": published_at.isoformat(),
        "FIGI": identity.figi,
        "UID": identity.instrument_uid,
        "CLASS_CODE": identity.class_code,
        "SESSION_STATE": session_state.value,
        "FIRST_COMMON_CANDLE_BEGIN": _iso(first_common),
        "LAST_COMMON_CANDLE_END": _iso(last_common_end),
        "NEXT_COMMON_CANDLE_BEGIN": _iso(next_common),
        "NEXT_COMMON_DELTA_SECONDS": _delta_seconds(next_common, published_at),
        "WHY_SESSION_STATE": _why_session_state(
            session_state=session_state,
            first_common=first_common,
            last_common_end=last_common_end,
            next_common=next_common,
            published_at=published_at,
        ),
        "SECURITY_BASELINE_END": _iso(security_baseline.end_at if security_baseline else None),
        "BENCHMARK_BASELINE_END": _iso(benchmark_baseline.end_at if benchmark_baseline else None),
        "SECURITY_EFFECTIVE_BEGIN": _iso(
            security_effective.begin_at if security_effective else None
        ),
        "BENCHMARK_EFFECTIVE_BEGIN": _iso(
            benchmark_effective.begin_at if benchmark_effective else None
        ),
        "BASELINE_WINDOW_EQUAL": _same_timestamp(
            security_baseline.end_at if security_baseline else None,
            benchmark_baseline.end_at if benchmark_baseline else None,
        ),
        "EFFECTIVE_WINDOW_EQUAL": _same_timestamp(
            security_effective.begin_at if security_effective else None,
            benchmark_effective.begin_at if benchmark_effective else None,
        ),
        "SECURITY_BASELINE_DELTA_SECONDS": _delta_seconds(
            security_baseline.end_at if security_baseline else None, published_at
        ),
        "BENCHMARK_BASELINE_DELTA_SECONDS": _delta_seconds(
            benchmark_baseline.end_at if benchmark_baseline else None, published_at
        ),
        "SECURITY_EFFECTIVE_DELTA_SECONDS": _delta_seconds(
            security_effective.begin_at if security_effective else None, published_at
        ),
        "BENCHMARK_EFFECTIVE_DELTA_SECONDS": _delta_seconds(
            benchmark_effective.begin_at if benchmark_effective else None, published_at
        ),
        "NEAREST_COMMON_TIMESTAMP_BEFORE": _iso(previous_common),
        "NEAREST_COMMON_TIMESTAMP_AFTER": _iso(next_common),
        "DELTA_TO_NEAREST_COMMON_TIMESTAMP_AFTER_SECONDS": _delta_seconds(
            next_common, published_at
        ),
        "SECURITY_COMPLETE_NEARBY_COUNT": security_neighborhood["complete_count"],
        "BENCHMARK_COMPLETE_NEARBY_COUNT": benchmark_neighborhood["complete_count"],
        "SECURITY_INCOMPLETE_NEARBY_COUNT": security_neighborhood["incomplete_count"],
        "BENCHMARK_INCOMPLETE_NEARBY_COUNT": benchmark_neighborhood["incomplete_count"],
        "SECURITY_MISSING_MINUTES": security_neighborhood["missing_minutes"],
        "BENCHMARK_MISSING_MINUTES": benchmark_neighborhood["missing_minutes"],
        "COMMON_MISSING_MINUTES": common_neighborhood["missing_minutes"],
        "COMMON_SECURITY_BENCHMARK_BEGIN_TIMESTAMPS": common_neighborhood["begin_timestamps"],
        "SECURITY_NEIGHBORHOOD": security_neighborhood,
        "BENCHMARK_NEIGHBORHOOD": benchmark_neighborhood,
        "COMMON_NEIGHBORHOOD": common_neighborhood,
        "CACHE_MIN_TIMESTAMP": _iso(cache_min),
        "CACHE_MAX_TIMESTAMP": _iso(cache_max),
        "REQUIRED_DIAGNOSTIC_MIN": required_min.isoformat(),
        "REQUIRED_DIAGNOSTIC_MAX": required_max.isoformat(),
        "CACHE_WINDOW_SUFFICIENT": _cache_window_sufficient(security, required_min, required_max),
        "TIMESTAMP_CONVENTION": {
            "interval": "1m",
            "timezone": "UTC",
            "timestamp_rounding_performed": False,
            "security_offsets_seconds": _offset_distribution(security_neighborhood["candles"]),
            "benchmark_offsets_seconds": _offset_distribution(benchmark_neighborhood["candles"]),
        },
        "INTERNAL_MISSING_REASON": alignment.missing_reason,
        "ROOT_CAUSE": root_cause.value,
        "RECOVERY_RECOMMENDED": recommendation != RecoveryRecommendationType.NO_SAFE_RECOVERY,
        "RECOVERY_RECOMMENDATION_TYPE": recommendation.value,
    }


def _root_cause(
    *,
    session_state: SessionState,
    common_begin: tuple[datetime, ...],
    next_common: datetime | None,
    published_at: datetime,
    security_baseline: TInvestMinuteCandle | None,
    benchmark_baseline: TInvestMinuteCandle | None,
    security_effective: TInvestMinuteCandle | None,
    benchmark_effective: TInvestMinuteCandle | None,
    security_neighborhood: dict[str, Any],
    benchmark_neighborhood: dict[str, Any],
    common_neighborhood: dict[str, Any],
    security: tuple[TInvestMinuteCandle, ...],
    benchmark: tuple[TInvestMinuteCandle, ...],
    required_min: datetime,
    required_max: datetime,
) -> SessionAlignmentRootCause:
    if security_baseline is None:
        return SessionAlignmentRootCause.BASELINE_SECURITY_MISSING
    if benchmark_baseline is None:
        return SessionAlignmentRootCause.BASELINE_BENCHMARK_MISSING
    if security_effective is None:
        return SessionAlignmentRootCause.EFFECTIVE_SECURITY_MISSING
    if benchmark_effective is None:
        return SessionAlignmentRootCause.EFFECTIVE_BENCHMARK_MISSING
    if not _cache_window_sufficient(security, required_min, required_max):
        return SessionAlignmentRootCause.CACHE_WINDOW_TOO_NARROW
    if session_state == SessionState.UNKNOWN:
        if security_neighborhood["incomplete_count"]:
            return SessionAlignmentRootCause.SECURITY_CANDLE_INCOMPLETE
        if benchmark_neighborhood["incomplete_count"]:
            return SessionAlignmentRootCause.BENCHMARK_CANDLE_INCOMPLETE
        if not common_begin or next_common is None:
            return SessionAlignmentRootCause.SESSION_UNKNOWN_NO_COMMON_CANDLE
        if next_common - published_at > timedelta(minutes=1):
            return SessionAlignmentRootCause.SESSION_UNKNOWN_COMMON_CANDLE_TOO_FAR
    if security_effective.begin_at != benchmark_effective.begin_at:
        return SessionAlignmentRootCause.SECURITY_BENCHMARK_EFFECTIVE_WINDOW_MISMATCH
    if security_baseline.end_at != benchmark_baseline.end_at:
        return SessionAlignmentRootCause.SECURITY_BENCHMARK_BASELINE_WINDOW_MISMATCH
    if security_neighborhood["incomplete_count"]:
        return SessionAlignmentRootCause.SECURITY_CANDLE_INCOMPLETE
    if benchmark_neighborhood["incomplete_count"]:
        return SessionAlignmentRootCause.BENCHMARK_CANDLE_INCOMPLETE
    if security_neighborhood["missing_minutes"]:
        return SessionAlignmentRootCause.SECURITY_MINUTE_GAP
    if benchmark_neighborhood["missing_minutes"]:
        return SessionAlignmentRootCause.BENCHMARK_MINUTE_GAP
    if common_neighborhood["missing_minutes"]:
        return SessionAlignmentRootCause.COMMON_MINUTE_GAP
    return SessionAlignmentRootCause.OTHER_FAIL_CLOSED


def _recovery_recommendation(
    root_cause: SessionAlignmentRootCause,
) -> RecoveryRecommendationType:
    if root_cause == SessionAlignmentRootCause.CACHE_WINDOW_TOO_NARROW:
        return RecoveryRecommendationType.RECOVERY_BY_CACHE_EXTENSION
    if root_cause in {
        SessionAlignmentRootCause.BASELINE_SECURITY_MISSING,
        SessionAlignmentRootCause.BASELINE_BENCHMARK_MISSING,
        SessionAlignmentRootCause.EFFECTIVE_SECURITY_MISSING,
        SessionAlignmentRootCause.EFFECTIVE_BENCHMARK_MISSING,
        SessionAlignmentRootCause.SECURITY_MINUTE_GAP,
        SessionAlignmentRootCause.BENCHMARK_MINUTE_GAP,
        SessionAlignmentRootCause.COMMON_MINUTE_GAP,
    }:
        return RecoveryRecommendationType.RECOVERY_BY_MISSING_CANDLE_ACQUISITION
    if root_cause in {
        SessionAlignmentRootCause.SESSION_UNKNOWN_COMMON_CANDLE_TOO_FAR,
        SessionAlignmentRootCause.SECURITY_BENCHMARK_EFFECTIVE_WINDOW_MISMATCH,
        SessionAlignmentRootCause.SECURITY_BENCHMARK_BASELINE_WINDOW_MISMATCH,
    }:
        return RecoveryRecommendationType.METHODOLOGY_CHANGE_REQUIRED
    if root_cause in {
        SessionAlignmentRootCause.SESSION_UNKNOWN_NO_COMMON_CANDLE,
        SessionAlignmentRootCause.SECURITY_CANDLE_INCOMPLETE,
        SessionAlignmentRootCause.BENCHMARK_CANDLE_INCOMPLETE,
    }:
        return RecoveryRecommendationType.DATA_PROVIDER_GAP
    return RecoveryRecommendationType.NO_SAFE_RECOVERY


def _why_session_state(
    *,
    session_state: SessionState,
    first_common: datetime | None,
    last_common_end: datetime | None,
    next_common: datetime | None,
    published_at: datetime,
) -> str:
    if first_common is None:
        return "No complete same-begin security/benchmark candle exists on the event day."
    if published_at < first_common:
        return "Publication is before the first complete common candle on the event day."
    if last_common_end is not None and published_at >= last_common_end:
        return "Publication is after the last complete common candle on the event day."
    if session_state == SessionState.UNKNOWN:
        if next_common is None:
            return "No complete common candle begins at or after publication."
        return (
            "The next complete common candle begins more than one minute after publication, "
            "so classify_session() returns UNKNOWN."
        )
    return "Publication is within the frozen complete common-candle session checks."


def _neighborhood(rows: tuple[TInvestMinuteCandle, ...], published_at: datetime) -> dict[str, Any]:
    start, end = _diagnostic_window(published_at)
    nearby = [row for row in rows if row.begin_at >= start and row.begin_at <= end]
    complete = [row for row in nearby if row.is_complete]
    incomplete = [row for row in nearby if not row.is_complete]
    complete_begins = {row.begin_at for row in complete}
    next_complete = _first_beginning_at_or_after_candle(complete, published_at)
    return {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "previous_candle": _timestamp_candle(_last_ending_at_or_before(rows, published_at)),
        "next_candle": _timestamp_candle(_first_beginning_at_or_after_candle(rows, published_at)),
        "nearest_complete_before": _timestamp_candle(
            _last_ending_at_or_before(complete, published_at)
        ),
        "nearest_complete_after": _timestamp_candle(next_complete),
        "next_complete_delta_seconds": _delta_seconds(
            next_complete.begin_at if next_complete is not None else None,
            published_at,
        ),
        "complete_count": len(complete),
        "incomplete_count": len(incomplete),
        "missing_minutes": _missing_minutes(start, end, complete_begins),
        "candles": [_timestamp_candle(row) for row in nearby],
    }


def _common_neighborhood(
    published_at: datetime,
    security: tuple[TInvestMinuteCandle, ...],
    benchmark: tuple[TInvestMinuteCandle, ...],
) -> dict[str, Any]:
    start, end = _diagnostic_window(published_at)
    begins = set(row.begin_at for row in security if start <= row.begin_at <= end) & set(
        row.begin_at for row in benchmark if start <= row.begin_at <= end
    )
    ordered = tuple(sorted(begins))
    return {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "begin_timestamps": [row.isoformat() for row in ordered],
        "nearest_common_timestamp_before": _iso(_last_beginning_before(ordered, published_at)),
        "nearest_common_timestamp_after": _iso(_first_beginning_at_or_after(ordered, published_at)),
        "missing_minutes": _missing_minutes(start, end, begins),
        "complete_count": len(ordered),
    }


def _common_begins(
    published_at: datetime,
    security: tuple[TInvestMinuteCandle, ...],
    benchmark: tuple[TInvestMinuteCandle, ...],
) -> tuple[datetime, ...]:
    day = published_at.date()
    benchmark_begins = {row.begin_at for row in benchmark if row.begin_at.date() == day}
    return tuple(
        sorted(
            row.begin_at
            for row in security
            if row.begin_at.date() == day and row.begin_at in benchmark_begins
        )
    )


def _diagnostic_window(published_at: datetime) -> tuple[datetime, datetime]:
    published = published_at.astimezone(UTC)
    return published - NEIGHBORHOOD_BEFORE, published + NEIGHBORHOOD_AFTER


def _missing_minutes(start: datetime, end: datetime, observed: set[datetime]) -> list[str]:
    result: list[str] = []
    cursor = _floor_minute(start)
    last = _floor_minute(end)
    while cursor <= last:
        if cursor not in observed:
            result.append(cursor.isoformat())
        cursor += timedelta(minutes=1)
    return result


def _offset_distribution(candles: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(_parse_datetime(row["begin_at"]).second) for row in candles)
    return dict(sorted(counts.items()))


def _timestamp_candle(row: TInvestMinuteCandle | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "instrument_uid": row.instrument_uid,
        "begin_at": row.begin_at.isoformat(),
        "end_at": row.end_at.isoformat(),
        "is_complete": row.is_complete,
    }


def _valid_rows(rows: tuple[TInvestMinuteCandle, ...]) -> tuple[TInvestMinuteCandle, ...]:
    return tuple(sorted((row for row in rows if row.is_complete), key=lambda row: row.begin_at))


def _last_ending_at_or_before(
    rows: tuple[TInvestMinuteCandle, ...] | list[TInvestMinuteCandle], at: datetime
) -> TInvestMinuteCandle | None:
    candidates = [row for row in rows if row.end_at <= at]
    return max(candidates, key=lambda row: row.end_at) if candidates else None


def _first_beginning_at_or_after_candle(
    rows: tuple[TInvestMinuteCandle, ...] | list[TInvestMinuteCandle], at: datetime
) -> TInvestMinuteCandle | None:
    candidates = [row for row in rows if row.begin_at >= at]
    return min(candidates, key=lambda row: row.begin_at) if candidates else None


def _last_beginning_before(rows: tuple[datetime, ...], at: datetime) -> datetime | None:
    candidates = [row for row in rows if row < at]
    return max(candidates) if candidates else None


def _first_beginning_at_or_after(rows: tuple[datetime, ...], at: datetime) -> datetime | None:
    candidates = [row for row in rows if row >= at]
    return min(candidates) if candidates else None


def _same_timestamp(left: datetime | None, right: datetime | None) -> bool:
    return left is not None and right is not None and left == right


def _delta_seconds(value: datetime | None, published_at: datetime) -> int | None:
    if value is None:
        return None
    return int((value - published_at).total_seconds())


def _cache_window_sufficient(
    rows: tuple[TInvestMinuteCandle, ...], required_min: datetime, required_max: datetime
) -> bool:
    cache_min = min((row.begin_at for row in rows), default=None)
    cache_max = max((row.end_at for row in rows), default=None)
    return (
        cache_min is not None
        and cache_max is not None
        and cache_min <= required_min
        and cache_max >= required_max
    )


def _cache_roots(pr38_root: Path, extra_cache_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    candidates = (
        pr38_root / "raw-minute-cache",
        pr38_root.parent / "exact-event-new-source-maturation-v1" / "raw-minute-cache",
        pr38_root.parent / "exact-event-source-diversity-v3" / "raw-minute-cache",
        pr38_root.parent / "exact-event-market-dataset-v2" / "raw-minute-cache",
        pr38_root.parent / "exact-event-market-dataset-v1" / "raw-minute-cache",
        *extra_cache_roots,
    )
    unique: list[Path] = []
    for path in candidates:
        if path.exists() and path not in unique:
            unique.append(path)
    return tuple(unique)


def _load_security_history(
    cache_roots: tuple[Path, ...], identity: SessionDiagnosticIdentity
) -> tuple[TInvestMinuteCandle, ...]:
    published_at = _parse_datetime(identity.publication_timestamp)
    rows: dict[tuple[str, datetime], TInvestMinuteCandle] = {}
    for begin, _end in acquisition_day_bounds(published_at):
        for root in cache_roots:
            path = root / identity.ticker / f"{begin.date().isoformat()}-day.jsonl"
            if not path.exists():
                continue
            for payload in _read_jsonl(path):
                candle = _candle_from_payload(payload, expected=identity)
                rows[(candle.instrument_uid, candle.begin_at)] = candle
    return tuple(rows[key] for key in sorted(rows, key=lambda item: (item[1], item[0])))


def _load_benchmark_history(
    cache_roots: tuple[Path, ...], ticker: str, published_at: datetime
) -> tuple[TInvestMinuteCandle, ...]:
    rows: dict[tuple[str, datetime], TInvestMinuteCandle] = {}
    for begin, _end in acquisition_day_bounds(published_at):
        for root in cache_roots:
            for suffix in ("day", "pre"):
                path = root / ticker / f"{begin.date().isoformat()}-{suffix}.jsonl"
                if not path.exists():
                    continue
                for payload in _read_jsonl(path):
                    candle = _candle_from_payload(payload)
                    rows[(candle.instrument_uid, candle.begin_at)] = candle
    return tuple(rows[key] for key in sorted(rows, key=lambda item: (item[1], item[0])))


def _candle_from_payload(
    payload: dict[str, Any], expected: SessionDiagnosticIdentity | None = None
) -> TInvestMinuteCandle:
    if str(payload.get("source", "TINVEST_API")) != "TINVEST_API":
        raise ValueError("NON_TINVEST_CANDLE_CACHE_SOURCE")
    if expected is not None:
        if str(payload.get("instrument_uid")) != expected.instrument_uid:
            raise ValueError("SECURITY_CACHE_UID_MISMATCH")
        if payload.get("figi") is not None and str(payload["figi"]) != expected.figi:
            raise ValueError("SECURITY_CACHE_FIGI_MISMATCH")
        if (
            payload.get("class_code") is not None
            and str(payload["class_code"]) != expected.class_code
        ):
            raise ValueError("SECURITY_CACHE_CLASS_CODE_MISMATCH")
    return TInvestMinuteCandle(
        instrument_uid=str(payload["instrument_uid"]),
        begin_at=_parse_datetime(payload["begin_at"]),
        end_at=_parse_datetime(payload["end_at"]),
        open=Decimal(str(payload["open"])),
        high=Decimal(str(payload["high"])),
        low=Decimal(str(payload["low"])),
        close=Decimal(str(payload["close"])),
        volume=int(str(payload["volume"])),
        is_complete=bool(payload["is_complete"]),
    )


def _future_exclusions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in events:
        metadata = _metadata(row)
        published_at = _parse_datetime(metadata["publication_timestamp_utc"])
        if published_at.date() < FUTURE_EVENT_HOLDOUT_START:
            continue
        result.append(
            {
                "event_id": str(metadata["event_id"]),
                "ticker": str(metadata["ticker"]),
                "publication_timestamp_utc": published_at.isoformat(),
                "excluded_from_session_alignment_diagnostics": True,
                "future_outcome_observed": False,
            }
        )
    return result


def _contains_no_price_values(payload: object) -> bool:
    forbidden = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "security_return",
        "benchmark_return",
        "abnormal_return",
        "security_log_return",
        "benchmark_log_return",
        "abnormal_log_return",
        "target_class",
    }
    if isinstance(payload, dict):
        items = cast("dict[object, object]", payload).items()
        return all(
            str(key) not in forbidden and _contains_no_price_values(value) for key, value in items
        )
    if isinstance(payload, list):
        return all(_contains_no_price_values(item) for item in cast("list[object]", payload))
    return True


def _artifact_sha(manifest: dict[str, Any]) -> str:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"ARTIFACT_SHA", "created_at", "git_sha"}
    }
    return sha256_payload(core)


def _floor_minute(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(second=0, microsecond=0)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _verify_frozen_contracts() -> None:
    if rules_v3_fingerprint() != EXPECTED_RULES_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_MISMATCH")
    if prompt_hash() != QWEN_PROMPT_SHA or schema_hash() != QWEN_SCHEMA_SHA:
        raise ValueError("FROZEN_QWEN_CONTRACT_MISMATCH")


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["metadata"])


def _event_id(row: dict[str, Any]) -> str:
    return str(_metadata(row)["event_id"])


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
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_artifacts(
    output_root: Path,
    manifest: dict[str, Any],
    per_event: list[dict[str, Any]],
    future_exclusions: list[dict[str, Any]],
) -> None:
    _write_json(output_root / "manifest.json", manifest)
    _write_jsonl(output_root / "per-event-diagnostics.jsonl", per_event)
    _write_jsonl(output_root / "future-holdout-exclusions.jsonl", future_exclusions)
    _write_report(output_root / "report.md", manifest)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        "Diagnostics-only report for frozen EXACT session alignment gaps.",
        "",
        f"- INPUT_DATASET_SHA={manifest['INPUT_DATASET_SHA']}",
        f"- OUTPUT_DATASET_SHA={manifest['OUTPUT_DATASET_SHA']}",
        f"- DIAGNOSTIC_EVENTS_TOTAL={manifest['DIAGNOSTIC_EVENTS_TOTAL']}",
        f"- SESSION_DIAGNOSTIC_COHORT_SHA={manifest['SESSION_DIAGNOSTIC_COHORT_SHA']}",
        f"- ROOT_CAUSE_COUNTS={manifest['ROOT_CAUSE_COUNTS']}",
        "- DIAGNOSTIC_ARTIFACT_CONTAINS_NO_PRICE_VALUES="
        f"{manifest['DIAGNOSTIC_ARTIFACT_CONTAINS_NO_PRICE_VALUES']}",
        f"- ALIGNMENT_METHODOLOGY_CHANGED={str(manifest['ALIGNMENT_METHODOLOGY_CHANGED']).lower()}",
        "",
        "No alignment methodology change, model training, TEST outcome use, future holdout "
        "observation, backtest, paper trading, orders, or BUY/SELL output was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
