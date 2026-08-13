from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import StrEnum
from itertools import pairwise
from statistics import fmean, pstdev

from src.tinvest_market.client import TInvestDailyCandle, TInvestInstrument
from src.tinvest_market.policy import PRICE_ADJUSTMENT_STATUS, source_policy

RAW_DATASET_VERSION = "tinvest-market-raw-v1"
FEATURE_DATASET_VERSION = "tinvest-market-baseline-features-v1"
SOURCE = "TINVEST_API"
SECURITY_TICKERS = ("SBER", "SBERP", "GAZP", "LKOH", "ROSN", "NVTK", "YDEX", "T", "VTBR", "GMKN")
BENCHMARK_TICKER = "IMOEX"
FEATURE_WINDOW_SESSIONS = 20
FLAT_RETURN_THRESHOLD = 0.002
CLASSIFICATION_POLICY_VERSION = "market-direction-thresholds-v1"

SECURITY_FEATURES = (
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "volume_mean_5d",
    "volume_mean_20d",
    "volume_std_5d",
    "volume_std_20d",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "price_to_sma_5d",
    "price_to_sma_10d",
    "price_to_sma_20d",
    "trade_day_of_week",
    "trade_month",
)
BENCHMARK_FEATURES = (
    "imoex_return_1d",
    "imoex_return_5d",
    "imoex_return_20d",
    "imoex_volatility_5d",
    "imoex_volatility_20d",
    "relative_return_1d",
    "relative_return_5d",
    "relative_return_20d",
    "rolling_beta_20d",
    "rolling_correlation_20d",
)


class Direction(StrEnum):
    DOWN = "DOWN"
    FLAT = "FLAT"
    UP = "UP"


class SplitName(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


@dataclass(frozen=True, slots=True)
class ResolvedInstrument:
    ticker: str
    class_code: str
    instrument_uid: str
    figi: str | None
    instrument_type: str
    first_1day_candle_date: date | None
    name: str
    exchange: str | None
    currency: str | None
    resolved_at: datetime

    def payload(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "class_code": self.class_code,
            "instrument_uid": self.instrument_uid,
            "figi": self.figi,
            "instrument_type": self.instrument_type,
            "first_1day_candle_date": self.first_1day_candle_date.isoformat()
            if self.first_1day_candle_date
            else None,
            "name": self.name,
            "exchange": self.exchange,
            "currency": self.currency,
            "resolved_at": self.resolved_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class DailyBar:
    ticker: str
    instrument_uid: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_complete: bool

    def validate(self) -> None:
        if not self.ticker or not self.instrument_uid:
            raise ValueError("ticker and instrument UID are required")
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("OHLC prices must be positive")
        if (
            self.low > self.high
            or not self.low <= self.open <= self.high
            or not self.low <= self.close <= self.high
        ):
            raise ValueError("invalid OHLC range")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "ticker": self.ticker,
            "instrument_uid": self.instrument_uid,
            "trade_date": self.trade_date.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "is_complete": self.is_complete,
            "source": SOURCE,
        }


@dataclass(frozen=True, slots=True)
class FeatureRow:
    row_id: str
    ticker: str
    trade_date: date
    feature_as_of: date
    values: dict[str, float]
    benchmark_available: bool

    def validate(self) -> None:
        if self.row_id != row_id(self.ticker, self.trade_date):
            raise ValueError("invalid feature identity")
        if not self.feature_as_of < self.trade_date:
            raise ValueError("rolling features must end at t-1")
        expected = feature_names(self.benchmark_available)
        if tuple(self.values) != expected or not all(
            math.isfinite(value) for value in self.values.values()
        ):
            raise ValueError("invalid feature schema")

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "row_id": self.row_id,
            "ticker": self.ticker,
            "trade_date": self.trade_date.isoformat(),
            "feature_as_of": self.feature_as_of.isoformat(),
            "benchmark_available": self.benchmark_available,
            "features": self.values,
        }


@dataclass(frozen=True, slots=True)
class TargetRow:
    row_id: str
    ticker: str
    trade_date: date
    baseline_trade_date: date
    next_session_return: float
    imoex_next_session_return: float | None
    next_session_abnormal_return: float | None
    direction: Direction

    def validate(self) -> None:
        if (
            self.row_id != row_id(self.ticker, self.trade_date)
            or not self.baseline_trade_date < self.trade_date
        ):
            raise ValueError("invalid target identity or chronology")
        if (self.imoex_next_session_return is None) != (self.next_session_abnormal_return is None):
            raise ValueError("benchmark and abnormal target availability must match")
        if self.imoex_next_session_return is not None and not math.isclose(
            self.next_session_abnormal_return or 0.0,
            self.next_session_return - self.imoex_next_session_return,
            abs_tol=1e-12,
        ):
            raise ValueError("invalid abnormal return")

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "row_id": self.row_id,
            "ticker": self.ticker,
            "trade_date": self.trade_date.isoformat(),
            "baseline_trade_date": self.baseline_trade_date.isoformat(),
            "next_session_return": self.next_session_return,
            "imoex_next_session_return": self.imoex_next_session_return,
            "next_session_abnormal_return": self.next_session_abnormal_return,
            "direction": self.direction.value,
            "classification_policy_version": CLASSIFICATION_POLICY_VERSION,
            "flat_return_threshold": FLAT_RETURN_THRESHOLD,
        }


@dataclass(frozen=True, slots=True)
class DatasetResult:
    features: tuple[FeatureRow, ...]
    targets: tuple[TargetRow, ...]
    raw_rows: int
    benchmark_rows: int
    ticker_distribution: dict[str, int]
    quality: dict[str, object]
    price_audit: dict[str, object]
    dataset_sha: str
    feature_schema_sha: str

    def validate(self) -> None:
        feature_ids = [item.row_id for item in self.features]
        if feature_ids != [item.row_id for item in self.targets] or len(feature_ids) != len(
            set(feature_ids)
        ):
            raise ValueError("features and targets must have unique identical identities")
        for item in self.features:
            item.validate()
        for item in self.targets:
            item.validate()


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    purge_sessions: int = 1
    embargo_sessions: int = 1


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    assignments: dict[str, SplitName]
    purged_row_ids: tuple[str, ...]
    embargoed_row_ids: tuple[str, ...]
    purged_dates: tuple[str, ...]
    embargoed_dates: tuple[str, ...]
    date_ranges: dict[str, dict[str, str]]
    split_sha: str

    def counts(self) -> dict[str, int]:
        counts = Counter(value.value for value in self.assignments.values())
        return {name.value: counts[name.value] for name in SplitName}


def resolve_instrument(
    ticker: str,
    candidates: tuple[TInvestInstrument, ...],
    *,
    resolved_at: datetime,
    expected_class_code: str | None = "TQBR",
) -> ResolvedInstrument:
    normalized = ticker.strip().upper()
    exact = [
        item
        for item in candidates
        if item.ticker == normalized
        and (expected_class_code is None or item.class_code == expected_class_code)
    ]
    unique = {item.instrument_uid: item for item in exact}
    if len(unique) != 1:
        raise ValueError(f"AMBIGUOUS_OR_MISSING_INSTRUMENT:{normalized}")
    item = next(iter(unique.values()))
    return ResolvedInstrument(
        ticker=normalized,
        class_code=item.class_code,
        instrument_uid=item.instrument_uid,
        figi=item.figi,
        instrument_type=item.instrument_type,
        first_1day_candle_date=item.first_1day_candle_date,
        name=item.name,
        exchange=item.exchange,
        currency=item.currency,
        resolved_at=resolved_at,
    )


def map_candle(ticker: str, candle: TInvestDailyCandle) -> DailyBar:
    return DailyBar(
        ticker=ticker,
        instrument_uid=candle.instrument_uid,
        trade_date=candle.trade_date,
        open=float(candle.open),
        high=float(candle.high),
        low=float(candle.low),
        close=float(candle.close),
        volume=float(candle.volume),
        is_complete=candle.is_complete,
    )


def build_dataset(
    security_bars: dict[str, tuple[DailyBar, ...]], benchmark_bars: tuple[DailyBar, ...] | None
) -> DatasetResult:
    benchmark = _unique(benchmark_bars or (), BENCHMARK_TICKER)
    benchmark_available = bool(benchmark)
    benchmark_dates = sorted(benchmark)
    features: list[FeatureRow] = []
    targets: list[TargetRow] = []
    distribution: dict[str, int] = {}
    excluded: dict[str, int] = {}
    attrition: dict[str, dict[str, object]] = {}
    for ticker in sorted(security_bars):
        bars = _unique(security_bars[ticker], ticker)
        raw_dates = sorted(bars)
        dates = (
            [item for item in raw_dates if item in benchmark] if benchmark_available else raw_dates
        )
        distribution[ticker] = len(raw_dates)
        excluded[ticker] = 0
        first_feature_index = len(features)
        for index, target_date in enumerate(dates):
            if index < FEATURE_WINDOW_SESSIONS + 1:
                excluded[ticker] += 1
                continue
            security_window = [
                bars[item] for item in dates[index - FEATURE_WINDOW_SESSIONS - 1 : index + 1]
            ]
            if len(security_window) != FEATURE_WINDOW_SESSIONS + 2:
                excluded[ticker] += 1
                continue
            baseline_date = security_window[-2].trade_date
            benchmark_window: list[DailyBar] | None = None
            if benchmark_available:
                needed = [item.trade_date for item in security_window]
                if any(item not in benchmark for item in needed):
                    excluded[ticker] += 1
                    continue
                benchmark_window = [benchmark[item] for item in needed]
            values = _features(
                security_window[:-1],
                benchmark_window[:-1] if benchmark_window else None,
                target_date,
            )
            security_return = _return(security_window[-1].close, security_window[-2].close)
            benchmark_return = (
                _return(benchmark_window[-1].close, benchmark_window[-2].close)
                if benchmark_window
                else None
            )
            feature = FeatureRow(
                row_id(ticker, target_date),
                ticker,
                target_date,
                baseline_date,
                values,
                benchmark_available,
            )
            target = TargetRow(
                feature.row_id,
                ticker,
                target_date,
                baseline_date,
                security_return,
                benchmark_return,
                security_return - benchmark_return if benchmark_return is not None else None,
                direction_for_return(security_return),
            )
            feature.validate()
            target.validate()
            features.append(feature)
            targets.append(target)
        ticker_features = features[first_feature_index:]
        feature_count = len(ticker_features)
        benchmark_loss = len(raw_dates) - len(dates)
        warmup_loss = min(len(dates), FEATURE_WINDOW_SESSIONS + 1)
        other_loss = len(raw_dates) - benchmark_loss - warmup_loss - feature_count
        attrition[ticker] = {
            "raw_first_date": raw_dates[0].isoformat() if raw_dates else None,
            "raw_last_date": raw_dates[-1].isoformat() if raw_dates else None,
            "raw_rows": len(raw_dates),
            "benchmark_aligned_rows": len(dates),
            "feature_first_date": (
                ticker_features[0].trade_date.isoformat() if ticker_features else None
            ),
            "feature_last_date": (
                ticker_features[-1].trade_date.isoformat() if ticker_features else None
            ),
            "feature_rows": feature_count,
            "rows_lost": {
                "warmup": warmup_loss,
                "missing_lag_history": 0,
                "benchmark_alignment": benchmark_loss,
                "target_tail": 0,
                "other": other_loss,
            },
            "reconciled": len(raw_dates)
            == feature_count + warmup_loss + benchmark_loss + other_loss,
        }
    paired = sorted(zip(features, targets, strict=True), key=lambda item: item[0].row_id)
    feature_rows = tuple(item[0] for item in paired)
    target_rows = tuple(item[1] for item in paired)
    semantics = dataset_semantics(benchmark_available)
    payload = {
        "dataset_version": FEATURE_DATASET_VERSION,
        "semantics": semantics,
        "features": [item.payload() for item in feature_rows],
        "targets": [item.payload() for item in target_rows],
    }
    result = DatasetResult(
        features=feature_rows,
        targets=target_rows,
        raw_rows=sum(distribution.values()),
        benchmark_rows=len(benchmark_dates),
        ticker_distribution=distribution,
        quality={
            "duplicate_ticker_date_rows": 0,
            "missing_or_incomplete_window_exclusions": excluded,
            "feature_cutoff_cause": (
                "ROLLING_WINDOWS_BUILT_BEFORE_BENCHMARK_SESSION_INTERSECTION"
                if benchmark_available
                else "NOT_APPLICABLE_NO_BENCHMARK"
            ),
            "feature_cutoff_was_bug": benchmark_available,
            "row_attrition": attrition,
            "prices_forward_filled": False,
            "synthetic_market_rows": 0,
            "target_day_present_in_features": False,
            "rolling_window_ends_at": "t-1",
            "targets_clipped": False,
            "targets_winsorized": False,
            "target_based_cleaning": False,
            "benchmark_source": SOURCE if benchmark_available else None,
            "moex_rows_used": 0,
        },
        price_audit=audit_prices(security_bars, benchmark_bars),
        dataset_sha=sha256_payload(payload),
        feature_schema_sha=sha256_payload(list(feature_names(benchmark_available))),
    )
    result.validate()
    return result


def temporal_split(
    rows: tuple[FeatureRow, ...], config: SplitConfig = SplitConfig()
) -> TemporalSplit:
    if not math.isclose(
        config.train_fraction + config.validation_fraction + config.test_fraction, 1.0
    ):
        raise ValueError("split fractions must sum to one")
    dates = sorted({item.trade_date for item in rows})
    if len(dates) < 7:
        raise ValueError("at least seven dates are required")
    train_end = max(1, int(len(dates) * config.train_fraction))
    validation_end = min(
        max(train_end + 1, int(len(dates) * (config.train_fraction + config.validation_fraction))),
        len(dates) - 1,
    )
    train_dates, validation_dates, test_dates = (
        dates[:train_end],
        dates[train_end:validation_end],
        dates[validation_end:],
    )
    purged_dates: set[date] = (
        set(train_dates[-config.purge_sessions :] + validation_dates[-config.purge_sessions :])
        if config.purge_sessions
        else set()
    )
    embargoed_dates: set[date] = (
        set(validation_dates[: config.embargo_sessions] + test_dates[: config.embargo_sessions])
        if config.embargo_sessions
        else set()
    )
    assignments: dict[str, SplitName] = {}
    purged: list[str] = []
    embargoed: list[str] = []
    train_set, validation_set = set(train_dates), set(validation_dates)
    for row in sorted(rows, key=lambda item: item.row_id):
        if row.trade_date in purged_dates:
            purged.append(row.row_id)
        elif row.trade_date in embargoed_dates:
            embargoed.append(row.row_id)
        elif row.trade_date in train_set:
            assignments[row.row_id] = SplitName.TRAIN
        elif row.trade_date in validation_set:
            assignments[row.row_id] = SplitName.VALIDATION
        else:
            assignments[row.row_id] = SplitName.TEST
    ranges = {
        name.value: _date_range(
            [row.trade_date for row in rows if assignments.get(row.row_id) == name]
        )
        for name in SplitName
    }
    if any(not value for value in ranges.values()):
        raise ValueError("split produced an empty partition")
    split_payload = {
        "config": asdict(config),
        "assignments": [(key, value.value) for key, value in sorted(assignments.items())],
        "purged": sorted(purged),
        "embargoed": sorted(embargoed),
    }
    return TemporalSplit(
        assignments,
        tuple(sorted(purged)),
        tuple(sorted(embargoed)),
        tuple(item.isoformat() for item in sorted(purged_dates)),
        tuple(item.isoformat() for item in sorted(embargoed_dates)),
        ranges,
        sha256_payload(split_payload),
    )


def readiness(feature_ready: int, ticker_count: int) -> dict[str, object]:
    if feature_ready < 1000:
        status = "NOT_READY"
    elif feature_ready < 5000:
        status = "MARKET_PILOT_READY"
    elif feature_ready < 10000:
        status = "MARKET_BASELINE_EXPERIMENT_READY"
    else:
        status = "MARKET_BASELINE_TRAINING_READY"
    return {
        "market_data_readiness": status,
        "source_usage_readiness": "PRIVATE_INTERNAL_USE_CONFIRMED",
        "private_model_training_allowed": True,
        "private_research_backtest_allowed": True,
        "public_redistribution_allowed": False,
        "real_trading_allowed": False,
        "feature_ready": feature_ready,
        "ticker_count": ticker_count,
        "warnings": [] if ticker_count >= 5 else ["LOW_TICKER_DIVERSITY"],
        "model_trained": False,
        "backtest_executed": False,
        "paper_trading_executed": False,
    }


def dataset_semantics(benchmark_available: bool) -> dict[str, object]:
    return {
        "source": SOURCE,
        "source_policy": source_policy(),
        "raw_dataset_version": RAW_DATASET_VERSION,
        "feature_dataset_version": FEATURE_DATASET_VERSION,
        "benchmark": BENCHMARK_TICKER if benchmark_available else None,
        "benchmark_source": SOURCE if benchmark_available else None,
        "abnormal_return_available": benchmark_available,
        "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
        "moex_data_used": False,
    }


def feature_names(benchmark_available: bool) -> tuple[str, ...]:
    return (
        SECURITY_FEATURES[:-2]
        + (BENCHMARK_FEATURES if benchmark_available else ())
        + SECURITY_FEATURES[-2:]
    )


def audit_prices(
    security_bars: dict[str, tuple[DailyBar, ...]],
    benchmark_bars: tuple[DailyBar, ...] | None = None,
) -> dict[str, object]:
    observations: list[tuple[str, str, float]] = []
    review_observations: list[dict[str, object]] = []
    ticker_returns: dict[str, list[float]] = {}
    benchmark = sorted(benchmark_bars or (), key=lambda item: item.trade_date)
    benchmark_dates = [item.trade_date for item in benchmark]
    benchmark_by_date = {item.trade_date: item for item in benchmark}
    for ticker, rows in sorted(security_bars.items()):
        ordered = sorted(rows, key=lambda item: item.trade_date)
        values_for_ticker: list[float] = []
        for index, (previous, current) in enumerate(pairwise(ordered)):
            value = _return(current.close, previous.close)
            observations.append((ticker, current.trade_date.isoformat(), value))
            values_for_ticker.append(value)
            if abs(value) > 0.50 or (ticker == "VTBR" and current.trade_date == date(2022, 2, 24)):
                benchmark_current = benchmark_by_date.get(current.trade_date)
                benchmark_index = bisect_left(benchmark_dates, current.trade_date)
                benchmark_previous = benchmark[benchmark_index - 1] if benchmark_index else None
                benchmark_return = (
                    _return(benchmark_current.close, benchmark_previous.close)
                    if benchmark_current is not None and benchmark_previous is not None
                    else None
                )
                classification = (
                    "MARKET_MOVE"
                    if benchmark_return is not None
                    and abs(benchmark_return) > 0.10
                    and (benchmark_return > 0) == (value > 0)
                    else "UNRESOLVED"
                )
                review_observations.append(
                    {
                        "ticker": ticker,
                        "trade_date": current.trade_date.isoformat(),
                        "previous_trade_date": previous.trade_date.isoformat(),
                        "previous_close": previous.close,
                        "current": current.payload(),
                        "raw_return": value,
                        "imoex_baseline_date": (
                            benchmark_previous.trade_date.isoformat()
                            if benchmark_previous is not None
                            else None
                        ),
                        "imoex_return_same_target_session": benchmark_return,
                        "classification": classification,
                        "neighboring_sessions": [
                            item.payload()
                            for item in ordered[max(0, index - 1) : min(len(ordered), index + 3)]
                        ],
                    }
                )
        ticker_returns[ticker] = values_for_ticker
    values = [item[2] for item in observations]
    largest_positive = max(observations, key=lambda item: item[2]) if observations else None
    largest_negative = min(observations, key=lambda item: item[2]) if observations else None
    return {
        "diagnostic_only": True,
        "rows_removed_by_audit": 0,
        "targets_clipped": False,
        "targets_winsorized": False,
        "count_abs_return_gt_10pct": sum(abs(item) > 0.10 for item in values),
        "count_abs_return_gt_20pct": sum(abs(item) > 0.20 for item in values),
        "count_abs_return_gt_50pct": sum(abs(item) > 0.50 for item in values),
        "largest_positive_return": _observation_payload(largest_positive),
        "largest_negative_return": _observation_payload(largest_negative),
        "ticker_statistics": {
            ticker: _return_statistics(items) for ticker, items in sorted(ticker_returns.items())
        },
        "review_observations": review_observations,
        "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
    }


def _return_statistics(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "max": ordered[-1] if ordered else None,
        "p0.1": _percentile(ordered, 0.001),
        "p1": _percentile(ordered, 0.01),
        "p99": _percentile(ordered, 0.99),
        "p99.9": _percentile(ordered, 0.999),
    }


def _percentile(ordered: list[float], quantile: float) -> float | None:
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _observation_payload(value: tuple[str, str, float] | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {"ticker": value[0], "trade_date": value[1], "return": value[2]}


def direction_for_return(value: float) -> Direction:
    if value > FLAT_RETURN_THRESHOLD:
        return Direction.UP
    if value < -FLAT_RETURN_THRESHOLD:
        return Direction.DOWN
    return Direction.FLAT


def row_id(ticker: str, trade_date: date) -> str:
    return f"{ticker}:{trade_date.isoformat()}"


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _unique(rows: tuple[DailyBar, ...], expected_ticker: str) -> dict[date, DailyBar]:
    result: dict[date, DailyBar] = {}
    uids: set[str] = set()
    for row in sorted(rows, key=lambda item: item.trade_date):
        row.validate()
        if row.ticker != expected_ticker:
            raise ValueError(f"unexpected ticker {row.ticker}")
        if row.trade_date in result:
            raise ValueError(f"duplicate {expected_ticker}/{row.trade_date.isoformat()}")
        if not row.is_complete:
            continue
        result[row.trade_date] = row
        uids.add(row.instrument_uid)
    if len(uids) > 1:
        raise ValueError(f"multiple historical identities for {expected_ticker}")
    return result


def _features(
    security: list[DailyBar], benchmark: list[DailyBar] | None, target_date: date
) -> dict[str, float]:
    closes = [item.close for item in security]
    returns = [_return(b, a) for a, b in pairwise(closes)]
    volumes = [item.volume for item in security]
    values: dict[str, float] = {}
    for window in (1, 2, 5, 10, 20):
        values[f"return_{window}d"] = _return(closes[-1], closes[-1 - window])
    for window in (5, 10, 20):
        values[f"volatility_{window}d"] = pstdev(returns[-window:])
    for window in (5, 20):
        sample = volumes[-window:]
        average = fmean(sample)
        values[f"volume_mean_{window}d"] = average
        values[f"volume_std_{window}d"] = pstdev(sample)
        values[f"volume_ratio_{window}d"] = 0.0 if average == 0 else volumes[-1] / average
    for window in (5, 10, 20):
        values[f"price_to_sma_{window}d"] = closes[-1] / fmean(closes[-window:]) - 1
    if benchmark is not None:
        bench_closes = [item.close for item in benchmark]
        bench_returns = [_return(b, a) for a, b in pairwise(bench_closes)]
        for window in (1, 5, 20):
            values[f"imoex_return_{window}d"] = _return(bench_closes[-1], bench_closes[-1 - window])
            values[f"relative_return_{window}d"] = (
                values[f"return_{window}d"] - values[f"imoex_return_{window}d"]
            )
        for window in (5, 20):
            values[f"imoex_volatility_{window}d"] = pstdev(bench_returns[-window:])
        beta, correlation = _beta_correlation(returns[-20:], bench_returns[-20:])
        values["rolling_beta_20d"] = beta
        values["rolling_correlation_20d"] = correlation
    values["trade_day_of_week"] = float(target_date.weekday())
    values["trade_month"] = float(target_date.month)
    return {name: values[name] for name in feature_names(benchmark is not None)}


def _beta_correlation(left: list[float], right: list[float]) -> tuple[float, float]:
    left_mean, right_mean = fmean(left), fmean(right)
    covariance = fmean((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_var = fmean((item - left_mean) ** 2 for item in left)
    right_var = fmean((item - right_mean) ** 2 for item in right)
    beta = 0.0 if right_var == 0 else covariance / right_var
    denominator = math.sqrt(left_var * right_var)
    return beta, 0.0 if denominator == 0 else covariance / denominator


def _return(current: float, previous: float) -> float:
    return current / previous - 1


def _date_range(values: list[date]) -> dict[str, str]:
    return {"from": min(values).isoformat(), "to": max(values).isoformat()} if values else {}
