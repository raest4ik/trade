from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from statistics import fmean, median, pstdev

from src.market_predictive_research.domain import (
    CROSS_SECTIONAL_BASES,
    DERIVED_FEATURE_NAMES,
    DEVELOPMENT_TO,
    FUTURE_HOLDOUT_START,
    OBSERVED_TEST_START,
)
from src.tinvest_market.client import TInvestInstrument
from src.tinvest_market.domain import FeatureRow, sha256_payload

RAW_VERSION = "tinvest-market-universe-raw-v1"
FEATURE_VERSION = "tinvest-market-universe-features-v1"
REPORT_VERSION = "market-universe-expansion-v1"
MEMBERSHIP_MODE = "CURRENT_TINVEST_CATALOG_SNAPSHOT"
SURVIVORSHIP_RISK = "PRESENT"
PRICE_ADJUSTMENT_STATUS = "UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES"
SOURCE_USAGE_READINESS = "PRIVATE_INTERNAL_USE_CONFIRMED"
ORIGINAL_TICKERS = (
    "SBER",
    "SBERP",
    "GAZP",
    "LKOH",
    "ROSN",
    "NVTK",
    "YDEX",
    "T",
    "VTBR",
    "GMKN",
)


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    shares: tuple[TInvestInstrument, ...]
    eligible: tuple[TInvestInstrument, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True, slots=True)
class ExpandedFeatureRow:
    row_id: str
    ticker: str
    instrument_uid: str
    trade_date: date
    feature_as_of: date
    partition: str
    values: dict[str, float]

    def payload(self) -> dict[str, object]:
        if not self.feature_as_of < self.trade_date:
            raise ValueError("feature cutoff must be t-1")
        return {
            "row_id": self.row_id,
            "ticker": self.ticker,
            "instrument_uid": self.instrument_uid,
            "trade_date": self.trade_date.isoformat(),
            "feature_as_of": self.feature_as_of.isoformat(),
            "partition": self.partition,
            "features": self.values,
        }


def discover(shares: tuple[TInvestInstrument, ...]) -> DiscoveryResult:
    by_uid: dict[str, TInvestInstrument] = {}
    for item in sorted(shares, key=lambda value: (value.instrument_uid, value.ticker)):
        previous = by_uid.get(item.instrument_uid)
        if previous is not None and previous != item:
            raise ValueError(f"conflicting instrument UID: {item.instrument_uid}")
        by_uid[item.instrument_uid] = item
    ordered = tuple(sorted(by_uid.values(), key=lambda item: (item.ticker, item.instrument_uid)))
    eligible = tuple(item for item in ordered if structurally_eligible(item))
    ticker_counts = Counter(item.ticker for item in eligible)
    duplicate_tickers = sorted(ticker for ticker, count in ticker_counts.items() if count > 1)
    diagnostics: dict[str, object] = {
        "discovered_shares_count": len(ordered),
        "instrument_status_request": "INSTRUMENT_STATUS_ALL",
        "class_code_distribution": dict(
            sorted(Counter(item.class_code for item in ordered).items())
        ),
        "exchange_distribution": dict(
            sorted(Counter(item.exchange or "UNKNOWN" for item in ordered).items())
        ),
        "real_exchange_distribution": dict(
            sorted(Counter(item.real_exchange or "UNKNOWN" for item in ordered).items())
        ),
        "availability_distribution": dict(
            sorted(
                Counter(
                    "ACTIVE_API_AVAILABLE"
                    if item.api_trade_available is True
                    else "INACTIVE_API_UNAVAILABLE"
                    if item.api_trade_available is False
                    else "UNKNOWN"
                    for item in ordered
                ).items()
            )
        ),
        "candidate_availability_distribution": dict(
            sorted(
                Counter(
                    "ACTIVE_API_AVAILABLE"
                    if item.api_trade_available is True
                    else "INACTIVE_API_UNAVAILABLE"
                    if item.api_trade_available is False
                    else "UNKNOWN"
                    for item in eligible
                ).items()
            )
        ),
        "currency_distribution": dict(
            sorted(Counter((item.currency or "UNKNOWN").lower() for item in ordered).items())
        ),
        "tqbr_rub_candidate_count": len(eligible),
        "duplicate_eligible_tickers": duplicate_tickers,
        "excluded_count": len(ordered) - len(eligible),
        "dealer_market_included": False,
        "non_share_assets_included": False,
        "ticker_name_heuristics_used": False,
        "universe_membership_mode": MEMBERSHIP_MODE,
        "historical_membership_point_in_time_verified": False,
        "survivorship_bias_risk": SURVIVORSHIP_RISK,
    }
    return DiscoveryResult(ordered, eligible, diagnostics)


def structurally_eligible(item: TInvestInstrument) -> bool:
    return (
        item.instrument_type.upper() in {"SHARE", "INSTRUMENT_TYPE_SHARE"}
        and (item.currency or "").lower() == "rub"
        and item.class_code.upper() == "TQBR"
    )


def partition_for(trade_date: date) -> str:
    if trade_date <= DEVELOPMENT_TO:
        return "DEVELOPMENT"
    if trade_date < OBSERVED_TEST_START:
        return "PURGE_EMBARGO_GAP"
    if trade_date < FUTURE_HOLDOUT_START:
        return "OBSERVED_V1_TEST"
    return "FUTURE_BLIND_HOLDOUT"


def history_tier(rows: int) -> str:
    if rows >= 1260:
        return "HISTORY_1260_PLUS"
    if rows >= 756:
        return "HISTORY_756_PLUS"
    if rows >= 252:
        return "HISTORY_252_PLUS"
    return "HISTORY_LT_252"


def enhance_features(
    rows: tuple[FeatureRow, ...], uid_by_ticker: dict[str, str]
) -> tuple[tuple[ExpandedFeatureRow, ...], tuple[str, ...]]:
    if not rows:
        raise ValueError("feature dataset is empty")
    original_names = tuple(rows[0].values)
    cross_names = tuple(
        item
        for base in CROSS_SECTIONAL_BASES
        for item in (f"cross_sectional_rank_{base}", f"cross_sectional_zscore_{base}")
    )
    names = (*original_names, *DERIVED_FEATURE_NAMES, *cross_names)
    by_date: defaultdict[date, list[FeatureRow]] = defaultdict(list)
    for row in rows:
        if tuple(row.values) != original_names or row.ticker not in uid_by_ticker:
            raise ValueError("source feature schema or identity is not stable")
        by_date[row.trade_date].append(row)
    result: list[ExpandedFeatureRow] = []
    for trade_date in sorted(by_date):
        dated = sorted(by_date[trade_date], key=lambda item: item.ticker)
        cross_values = {
            base: _cross_section([item.values[base] for item in dated])
            for base in CROSS_SECTIONAL_BASES
        }
        for index, row in enumerate(dated):
            values = {
                **row.values,
                "momentum_acceleration_5_20": row.values["return_5d"]
                - row.values["return_20d"] / 4.0,
                "volatility_term_5_20": row.values["volatility_5d"] - row.values["volatility_20d"],
                "volume_trend_5_20": _ratio(
                    row.values["volume_mean_5d"], row.values["volume_mean_20d"]
                )
                - 1.0,
                "month_end_flag": float(_is_month_end(trade_date)),
            }
            for base in CROSS_SECTIONAL_BASES:
                rank, zscore = cross_values[base][index]
                values[f"cross_sectional_rank_{base}"] = rank
                values[f"cross_sectional_zscore_{base}"] = zscore
            if tuple(values) != names or not all(math.isfinite(value) for value in values.values()):
                raise ValueError("invalid v2 feature row")
            result.append(
                ExpandedFeatureRow(
                    row_id=row.row_id,
                    ticker=row.ticker,
                    instrument_uid=uid_by_ticker[row.ticker],
                    trade_date=row.trade_date,
                    feature_as_of=row.feature_as_of,
                    partition=partition_for(row.trade_date),
                    values=values,
                )
            )
    ordered = tuple(sorted(result, key=lambda item: (item.trade_date, item.ticker)))
    return ordered, names


def feature_summary(rows: tuple[ExpandedFeatureRow, ...]) -> dict[str, object]:
    partition_counts = Counter(item.partition for item in rows)
    future = [item for item in rows if item.partition == "FUTURE_BLIND_HOLDOUT"]
    return {
        "feature_ready_rows": len(rows),
        "features_per_ticker": dict(sorted(Counter(item.ticker for item in rows).items())),
        "partition_counts": dict(sorted(partition_counts.items())),
        "future_holdout_session_count": len({item.trade_date for item in future}),
        "future_holdout_date_from": min((item.trade_date for item in future), default=None),
        "future_holdout_date_to": max((item.trade_date for item in future), default=None),
        "future_holdout_instrument_count": len({item.ticker for item in future}),
        "future_holdout_status": "ACCUMULATING",
        "future_holdout_observed": False,
        "observed_test_used": False,
        "predictive_metrics_computed": False,
        "predictions_computed": False,
    }


def median_history(values: list[int]) -> float:
    return float(median(values)) if values else 0.0


def feature_schema_sha(names: tuple[str, ...]) -> str:
    return sha256_payload(list(names))


def _cross_section(values: list[float]) -> list[tuple[float, float]]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[cursor][0]:
            end += 1
        rank = 0.5 if len(values) == 1 else ((cursor + end) / 2.0) / (len(values) - 1)
        for _value, index in ordered[cursor : end + 1]:
            ranks[index] = rank
        cursor = end + 1
    average = fmean(values)
    deviation = pstdev(values)
    return [
        (rank, 0.0 if deviation == 0 else (value - average) / deviation)
        for rank, value in zip(ranks, values, strict=True)
    ]


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _is_month_end(value: date) -> bool:
    from datetime import timedelta

    return (value + timedelta(days=1)).month != value.month
