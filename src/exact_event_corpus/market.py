from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

from src.exact_event_corpus.domain import SessionState
from src.tinvest_market.client import TInvestMinuteCandle

HORIZONS_MINUTES = (1, 5, 15, 30, 60)


@dataclass(frozen=True, slots=True)
class ExactMarketAlignment:
    session_state: SessionState
    reaction_status: str
    effective_event_at: datetime | None
    baseline_observed_at: datetime | None
    features: dict[str, str | int | bool | None]
    horizons: dict[str, dict[str, object]]
    missing_reason: str | None


def align_exact_event(
    published_at: datetime,
    security: Sequence[TInvestMinuteCandle],
    benchmark: Sequence[TInvestMinuteCandle],
    *,
    expose_outcomes: bool,
) -> ExactMarketAlignment:
    published = published_at.astimezone(UTC)
    security_rows = _valid_rows(security)
    benchmark_rows = _valid_rows(benchmark)
    session = classify_session(published, security_rows, benchmark_rows)
    if session != SessionState.DURING_MAIN_SESSION:
        return ExactMarketAlignment(
            session_state=session,
            reaction_status=f"EXACT_{session.value}_NOT_SUPPORTED",
            effective_event_at=None,
            baseline_observed_at=None,
            features={},
            horizons={},
            missing_reason=f"{session.value}_REACTION_NOT_SUPPORTED",
        )
    baseline_security = _last_ending_at_or_before(security_rows, published)
    baseline_benchmark = _last_ending_at_or_before(benchmark_rows, published)
    effective_security = _first_beginning_at_or_after(security_rows, published)
    effective_benchmark = _first_beginning_at_or_after(benchmark_rows, published)
    if None in (baseline_security, baseline_benchmark, effective_security, effective_benchmark):
        return _missing(session, "BASELINE_OR_EFFECTIVE_CANDLE_MISSING")
    assert baseline_security is not None
    assert baseline_benchmark is not None
    assert effective_security is not None
    assert effective_benchmark is not None
    if effective_security.begin_at != effective_benchmark.begin_at:
        return _missing(session, "SECURITY_BENCHMARK_EFFECTIVE_WINDOW_MISMATCH")
    if baseline_security.end_at != baseline_benchmark.end_at:
        return _missing(session, "SECURITY_BENCHMARK_BASELINE_WINDOW_MISMATCH")
    if baseline_security.end_at > published or effective_security.begin_at < published:
        raise ValueError("EVENT_MARKET_LEAKAGE_CHECK_FAILED")
    features = _pre_event_features(published, security_rows, benchmark_rows)
    if not expose_outcomes:
        return ExactMarketAlignment(
            session_state=session,
            reaction_status="FUTURE_HOLDOUT_OUTCOMES_GUARDED",
            effective_event_at=effective_security.begin_at,
            baseline_observed_at=baseline_security.end_at,
            features=features,
            horizons={},
            missing_reason=None,
        )
    horizons: dict[str, dict[str, object]] = {}
    for horizon in HORIZONS_MINUTES:
        target_end = effective_security.begin_at + timedelta(minutes=horizon)
        security_target = _ending_exactly(security_rows, target_end)
        benchmark_target = _ending_exactly(benchmark_rows, target_end)
        key = f"{horizon}m"
        if security_target is None or benchmark_target is None:
            horizons[key] = {"available": False, "reason": "EXACT_TARGET_CANDLE_MISSING"}
            continue
        security_return = security_target.close / baseline_security.close - Decimal("1")
        benchmark_return = benchmark_target.close / baseline_benchmark.close - Decimal("1")
        with localcontext() as context:
            context.prec = 28
            security_log = (security_target.close / baseline_security.close).ln()
            benchmark_log = (benchmark_target.close / baseline_benchmark.close).ln()
        horizons[key] = {
            "available": True,
            "window_begin_at": effective_security.begin_at.isoformat(),
            "window_end_at": target_end.isoformat(),
            "security_observed_at": security_target.end_at.isoformat(),
            "benchmark_observed_at": benchmark_target.end_at.isoformat(),
            "security_return": str(security_return),
            "benchmark_return": str(benchmark_return),
            "abnormal_return": str(security_return - benchmark_return),
            "security_log_return": str(security_log),
            "benchmark_log_return": str(benchmark_log),
            "abnormal_log_return": str(security_log - benchmark_log),
        }
    complete = all(bool(value["available"]) for value in horizons.values())
    return ExactMarketAlignment(
        session_state=session,
        reaction_status="REACTION_READY" if complete else "PARTIAL_REACTION",
        effective_event_at=effective_security.begin_at,
        baseline_observed_at=baseline_security.end_at,
        features=features,
        horizons=horizons,
        missing_reason=None if complete else "ONE_OR_MORE_TARGET_CANDLES_MISSING",
    )


def classify_session(
    published_at: datetime,
    security: Sequence[TInvestMinuteCandle],
    benchmark: Sequence[TInvestMinuteCandle],
) -> SessionState:
    day = published_at.astimezone(UTC).date()
    common = [
        row
        for row in security
        if row.begin_at.date() == day and any(other.begin_at == row.begin_at for other in benchmark)
    ]
    if not common:
        return SessionState.NON_TRADING_DAY
    first = min(row.begin_at for row in common)
    last = max(row.end_at for row in common)
    if published_at < first:
        return SessionState.PRE_OPEN
    if published_at >= last:
        return SessionState.AFTER_CLOSE
    effective = _first_beginning_at_or_after(common, published_at)
    if effective is None or effective.begin_at - published_at > timedelta(minutes=1):
        return SessionState.UNKNOWN
    return SessionState.DURING_MAIN_SESSION


def _pre_event_features(
    published_at: datetime,
    security: list[TInvestMinuteCandle],
    benchmark: list[TInvestMinuteCandle],
) -> dict[str, str | int | bool | None]:
    result: dict[str, str | int | bool | None] = {
        "feature_cutoff": published_at.isoformat(),
        "post_event_values_in_features": False,
    }
    for horizon in (5, 15, 30, 60):
        end_security = _last_ending_at_or_before(security, published_at)
        end_benchmark = _last_ending_at_or_before(benchmark, published_at)
        start_at = published_at - timedelta(minutes=horizon)
        start_security = _last_ending_at_or_before(security, start_at)
        start_benchmark = _last_ending_at_or_before(benchmark, start_at)
        if None in (end_security, end_benchmark, start_security, start_benchmark):
            result[f"pre_return_{horizon}m"] = None
            result[f"imoex_pre_return_{horizon}m"] = None
            continue
        assert end_security is not None and end_benchmark is not None
        assert start_security is not None and start_benchmark is not None
        result[f"pre_return_{horizon}m"] = str(end_security.close / start_security.close - 1)
        result[f"imoex_pre_return_{horizon}m"] = str(
            end_benchmark.close / start_benchmark.close - 1
        )
    return result


def _valid_rows(rows: Sequence[TInvestMinuteCandle]) -> list[TInvestMinuteCandle]:
    return sorted(
        (row for row in rows if row.is_complete),
        key=lambda row: (row.begin_at, row.instrument_uid),
    )


def _last_ending_at_or_before(
    rows: Sequence[TInvestMinuteCandle], at: datetime
) -> TInvestMinuteCandle | None:
    candidates = [row for row in rows if row.end_at <= at]
    return max(candidates, key=lambda row: row.end_at) if candidates else None


def _first_beginning_at_or_after(
    rows: Sequence[TInvestMinuteCandle], at: datetime
) -> TInvestMinuteCandle | None:
    candidates = [row for row in rows if row.begin_at >= at]
    return min(candidates, key=lambda row: row.begin_at) if candidates else None


def _ending_exactly(
    rows: Sequence[TInvestMinuteCandle], at: datetime
) -> TInvestMinuteCandle | None:
    return next((row for row in rows if row.end_at == at), None)


def _missing(session: SessionState, reason: str) -> ExactMarketAlignment:
    return ExactMarketAlignment(session, "INSUFFICIENT_DATA", None, None, {}, {}, reason)
