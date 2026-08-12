from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum
from itertools import pairwise
from statistics import fmean, pstdev
from typing import Any

DATASET_VERSION = "market-baseline-features-v1"
SOURCE_NAME = "MOEX_ISS"
SOURCE_POLICY = "ZERO_COST_OFFICIAL_ISS_ONLY"
PRICE_ADJUSTMENT_STATUS = "UNVERIFIED_MOEX_ISS_CANDLE_PRICES"
FEATURE_WINDOW_SESSIONS = 20
CLASSIFICATION_POLICY_VERSION = "market-direction-thresholds-v1"
FLAT_RETURN_THRESHOLD = 0.002

FEATURE_NAMES = (
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
    "trade_day_of_week",
    "trade_month",
)


class Direction(StrEnum):
    DOWN = "DOWN"
    FLAT = "FLAT"
    UP = "UP"


class SplitName(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class Readiness(StrEnum):
    NOT_READY = "NOT_READY"
    MARKET_PILOT_READY = "MARKET_PILOT_READY"
    MARKET_BASELINE_EXPERIMENT_READY = "MARKET_BASELINE_EXPERIMENT_READY"
    MARKET_BASELINE_TRAINING_READY = "MARKET_BASELINE_TRAINING_READY"


@dataclass(frozen=True, slots=True)
class DailyBar:
    ticker: str
    trade_date: date
    open: float
    close: float
    high: float
    low: float
    volume: float
    value: float

    def validate(self) -> None:
        if not self.ticker.strip():
            raise ValueError("ticker must not be blank")
        if min(self.open, self.close, self.high, self.low) <= 0:
            raise ValueError("OHLC prices must be positive")
        if self.low > self.high or not self.low <= self.open <= self.high:
            raise ValueError("open must be inside low-high range")
        if not self.low <= self.close <= self.high:
            raise ValueError("close must be inside low-high range")
        if self.volume < 0 or self.value < 0:
            raise ValueError("volume and value must be non-negative")


@dataclass(frozen=True, slots=True)
class FeatureRow:
    row_id: str
    ticker: str
    trade_date: date
    feature_as_of: date
    values: dict[str, float]

    def validate(self) -> None:
        if self.row_id != row_id(self.ticker, self.trade_date):
            raise ValueError("invalid feature row identity")
        if not self.feature_as_of < self.trade_date:
            raise ValueError("features must end before the target session")
        if tuple(self.values) != FEATURE_NAMES:
            raise ValueError("feature schema does not match the frozen v1 schema")
        if not all(math.isfinite(value) for value in self.values.values()):
            raise ValueError("features must be finite")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "row_id": self.row_id,
            "ticker": self.ticker,
            "trade_date": self.trade_date.isoformat(),
            "feature_as_of": self.feature_as_of.isoformat(),
            "features": self.values,
        }


@dataclass(frozen=True, slots=True)
class TargetRow:
    row_id: str
    ticker: str
    trade_date: date
    baseline_trade_date: date
    next_session_return: float
    imoex_next_session_return: float
    next_session_abnormal_return: float
    direction: Direction

    def validate(self) -> None:
        if self.row_id != row_id(self.ticker, self.trade_date):
            raise ValueError("invalid target row identity")
        if not self.baseline_trade_date < self.trade_date:
            raise ValueError("target session must follow the baseline session")
        expected = self.next_session_return - self.imoex_next_session_return
        if not math.isclose(self.next_session_abnormal_return, expected, abs_tol=1e-12):
            raise ValueError("abnormal return does not match security minus IMOEX")

    def payload(self) -> dict[str, Any]:
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
class DatasetBuildResult:
    features: tuple[FeatureRow, ...]
    targets: tuple[TargetRow, ...]
    source_row_count: int
    benchmark_row_count: int
    source_ticker_distribution: dict[str, int]
    source_date_ranges: dict[str, dict[str, str]]
    quality: dict[str, Any]
    dataset_sha256: str
    feature_schema_sha256: str

    def validate(self) -> None:
        feature_ids = [item.row_id for item in self.features]
        target_ids = [item.row_id for item in self.targets]
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("duplicate ticker/date feature row")
        if feature_ids != target_ids:
            raise ValueError("X and y identities or ordering differ")
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

    def validate(self) -> None:
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1) > 1e-12:
            raise ValueError("split fractions must sum to one")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("split fractions must be positive")
        if self.purge_sessions < 0 or self.embargo_sessions < 0:
            raise ValueError("purge and embargo must be non-negative")


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    assignments: dict[str, SplitName]
    purged_row_ids: tuple[str, ...]
    embargoed_row_ids: tuple[str, ...]
    split_sha256: str
    date_ranges: dict[str, dict[str, str]]

    def counts(self) -> dict[str, int]:
        result = Counter(value.value for value in self.assignments.values())
        return {name.value: result[name.value] for name in SplitName}


def build_dataset(
    security_bars: dict[str, tuple[DailyBar, ...]],
    benchmark_bars: tuple[DailyBar, ...],
    *,
    provider_rejected_rows: int = 0,
) -> DatasetBuildResult:
    benchmark = _unique_bars(benchmark_bars, expected_ticker="IMOEX")
    benchmark_dates = sorted(benchmark)
    benchmark_index = {trade_date: index for index, trade_date in enumerate(benchmark_dates)}
    features: list[FeatureRow] = []
    targets: list[TargetRow] = []
    source_distribution: dict[str, int] = {}
    source_ranges: dict[str, dict[str, str]] = {}
    missing_sessions: dict[str, int] = {}
    missing_window_exclusions: dict[str, int] = {}

    for ticker in sorted(security_bars):
        bars = _unique_bars(security_bars[ticker], expected_ticker=ticker)
        dates = sorted(bars)
        source_distribution[ticker] = len(dates)
        if dates:
            source_ranges[ticker] = {
                "from": dates[0].isoformat(),
                "to": dates[-1].isoformat(),
            }
            expected = {item for item in benchmark_dates if dates[0] <= item <= dates[-1]}
            missing_sessions[ticker] = len(expected - set(dates))
        excluded = 0
        for target_date in dates:
            target_index = benchmark_index.get(target_date)
            if target_index is None or target_index <= FEATURE_WINDOW_SESSIONS:
                excluded += 1
                continue
            required_dates = benchmark_dates[
                target_index - FEATURE_WINDOW_SESSIONS - 1 : target_index + 1
            ]
            if len(required_dates) != FEATURE_WINDOW_SESSIONS + 2:
                excluded += 1
                continue
            if any(item not in bars for item in required_dates):
                excluded += 1
                continue
            baseline_date = required_dates[-2]
            security_window = [bars[item] for item in required_dates]
            benchmark_window = [benchmark[item] for item in required_dates]
            feature = FeatureRow(
                row_id=row_id(ticker, target_date),
                ticker=ticker,
                trade_date=target_date,
                feature_as_of=baseline_date,
                values=_features(security_window[:-1], benchmark_window[:-1], target_date),
            )
            security_return = _return(security_window[-1].close, security_window[-2].close)
            benchmark_return = _return(benchmark_window[-1].close, benchmark_window[-2].close)
            target = TargetRow(
                row_id=feature.row_id,
                ticker=ticker,
                trade_date=target_date,
                baseline_trade_date=baseline_date,
                next_session_return=security_return,
                imoex_next_session_return=benchmark_return,
                next_session_abnormal_return=security_return - benchmark_return,
                direction=direction_for_return(security_return),
            )
            feature.validate()
            target.validate()
            features.append(feature)
            targets.append(target)
        missing_window_exclusions[ticker] = excluded

    ordered = sorted(zip(features, targets, strict=True), key=lambda pair: pair[0].row_id)
    features_tuple = tuple(pair[0] for pair in ordered)
    targets_tuple = tuple(pair[1] for pair in ordered)
    dataset_payload = {
        "dataset_version": DATASET_VERSION,
        "features": [item.payload() for item in features_tuple],
        "targets": [item.payload() for item in targets_tuple],
    }
    quality: dict[str, Any] = {
        "duplicate_ticker_date_rows": 0,
        "provider_rejected_rows": provider_rejected_rows,
        "zero_or_invalid_price_rows_in_dataset": 0,
        "missing_sessions_by_ticker": missing_sessions,
        "missing_or_incomplete_window_exclusions": missing_window_exclusions,
        "abnormal_return_missing_imoex_rows": 0,
        "prices_forward_filled": False,
        "synthetic_market_rows": 0,
        "target_based_cleaning": False,
        "extreme_targets_removed": 0,
        "listing_boundaries_respected": True,
        "target_day_present_in_features": False,
        "rolling_window_ends_at": "t-1",
        "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
    }
    result = DatasetBuildResult(
        features=features_tuple,
        targets=targets_tuple,
        source_row_count=sum(source_distribution.values()),
        benchmark_row_count=len(benchmark_dates),
        source_ticker_distribution=source_distribution,
        source_date_ranges=source_ranges,
        quality=quality,
        dataset_sha256=sha256_payload(dataset_payload),
        feature_schema_sha256=sha256_payload(list(FEATURE_NAMES)),
    )
    result.validate()
    return result


def date_grouped_temporal_split(
    rows: tuple[FeatureRow, ...], config: SplitConfig = SplitConfig()
) -> TemporalSplit:
    config.validate()
    dates = sorted({item.trade_date for item in rows})
    if len(dates) < 7:
        raise ValueError("at least seven trade dates are required for temporal split")
    train_end = max(1, int(len(dates) * config.train_fraction))
    validation_end = max(
        train_end + 1, int(len(dates) * (config.train_fraction + config.validation_fraction))
    )
    validation_end = min(validation_end, len(dates) - 1)
    train_dates = dates[:train_end]
    validation_dates = dates[train_end:validation_end]
    test_dates = dates[validation_end:]
    purged_dates = set(train_dates[-config.purge_sessions :] if config.purge_sessions else ())
    purged_dates.update(validation_dates[-config.purge_sessions :] if config.purge_sessions else ())
    embargoed_dates = set(
        validation_dates[: config.embargo_sessions] if config.embargo_sessions else ()
    )
    embargoed_dates.update(test_dates[: config.embargo_sessions] if config.embargo_sessions else ())
    assignments: dict[str, SplitName] = {}
    purged: list[str] = []
    embargoed: list[str] = []
    train_set = set(train_dates)
    validation_set = set(validation_dates)
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
        raise ValueError("purged temporal split produced an empty partition")
    payload = {
        "config": asdict(config),
        "assignments": [(key, value.value) for key, value in sorted(assignments.items())],
        "purged_row_ids": sorted(purged),
        "embargoed_row_ids": sorted(embargoed),
    }
    return TemporalSplit(
        assignments=assignments,
        purged_row_ids=tuple(sorted(purged)),
        embargoed_row_ids=tuple(sorted(embargoed)),
        split_sha256=sha256_payload(payload),
        date_ranges=ranges,
    )


def readiness_for_rows(feature_ready: int, ticker_count: int) -> dict[str, Any]:
    if feature_ready < 1000:
        status = Readiness.NOT_READY
    elif feature_ready < 5000:
        status = Readiness.MARKET_PILOT_READY
    elif feature_ready < 10000:
        status = Readiness.MARKET_BASELINE_EXPERIMENT_READY
    else:
        status = Readiness.MARKET_BASELINE_TRAINING_READY
    warnings = [] if ticker_count >= 5 else ["LOW_TICKER_DIVERSITY"]
    return {
        "status": status.value,
        "feature_ready": feature_ready,
        "ticker_count": ticker_count,
        "warnings": warnings,
        "model_trained": False,
        "thresholds": {
            "pilot": 1000,
            "experiment": 5000,
            "training": 10000,
            "minimum_tickers": 5,
        },
    }


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
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _unique_bars(bars: tuple[DailyBar, ...], *, expected_ticker: str) -> dict[date, DailyBar]:
    result: dict[date, DailyBar] = {}
    for bar in sorted(bars, key=lambda item: item.trade_date):
        bar.validate()
        if bar.ticker != expected_ticker:
            raise ValueError(f"unexpected ticker {bar.ticker}; expected {expected_ticker}")
        if bar.trade_date in result:
            raise ValueError(f"duplicate {expected_ticker}/{bar.trade_date.isoformat()}")
        result[bar.trade_date] = bar
    return result


def _features(
    security: list[DailyBar], benchmark: list[DailyBar], target_date: date
) -> dict[str, float]:
    security_closes = [item.close for item in security]
    benchmark_closes = [item.close for item in benchmark]
    security_returns = _one_day_returns(security_closes)
    benchmark_returns = _one_day_returns(benchmark_closes)
    volumes = [item.volume for item in security]
    values: dict[str, float] = {}
    for window in (1, 2, 5, 10, 20):
        values[f"return_{window}d"] = _return(security_closes[-1], security_closes[-1 - window])
    for window in (5, 10, 20):
        values[f"volatility_{window}d"] = pstdev(security_returns[-window:])
    for window in (5, 20):
        sample = volumes[-window:]
        average = fmean(sample)
        values[f"volume_mean_{window}d"] = average
        values[f"volume_std_{window}d"] = pstdev(sample)
        values[f"volume_ratio_{window}d"] = 0.0 if average == 0 else volumes[-1] / average
    for window in (5, 10, 20):
        average = fmean(security_closes[-window:])
        values[f"price_to_sma_{window}d"] = security_closes[-1] / average - 1
    for window in (1, 5, 20):
        values[f"imoex_return_{window}d"] = _return(
            benchmark_closes[-1], benchmark_closes[-1 - window]
        )
    for window in (5, 20):
        values[f"imoex_volatility_{window}d"] = pstdev(benchmark_returns[-window:])
    for window in (1, 5, 20):
        values[f"relative_return_{window}d"] = (
            values[f"return_{window}d"] - values[f"imoex_return_{window}d"]
        )
    beta, correlation = _beta_and_correlation(security_returns[-20:], benchmark_returns[-20:])
    values["rolling_beta_20d"] = beta
    values["rolling_correlation_20d"] = correlation
    values["trade_day_of_week"] = float(target_date.weekday())
    values["trade_month"] = float(target_date.month)
    return {name: values[name] for name in FEATURE_NAMES}


def _one_day_returns(closes: list[float]) -> list[float]:
    return [_return(current, previous) for previous, current in pairwise(closes)]


def _return(current: float, previous: float) -> float:
    return current / previous - 1


def _beta_and_correlation(security: list[float], benchmark: list[float]) -> tuple[float, float]:
    security_mean = fmean(security)
    benchmark_mean = fmean(benchmark)
    covariance = fmean(
        (left - security_mean) * (right - benchmark_mean)
        for left, right in zip(security, benchmark, strict=True)
    )
    security_variance = fmean((item - security_mean) ** 2 for item in security)
    benchmark_variance = fmean((item - benchmark_mean) ** 2 for item in benchmark)
    beta = 0.0 if benchmark_variance == 0 else covariance / benchmark_variance
    denominator = math.sqrt(security_variance * benchmark_variance)
    correlation = 0.0 if denominator == 0 else covariance / denominator
    return beta, correlation


def _date_range(values: list[date]) -> dict[str, str]:
    if not values:
        return {}
    return {"from": min(values).isoformat(), "to": max(values).isoformat()}
