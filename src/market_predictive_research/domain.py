from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from src.tinvest_market.policy import PRICE_ADJUSTMENT_STATUS, SOURCE_USAGE_READINESS

RESEARCH_VERSION = "market-predictive-research-v2"
DEVELOPMENT_FROM = date(2000, 2, 4)
DEVELOPMENT_TO = date(2022, 9, 15)
OBSERVED_TEST_START = date(2022, 9, 20)
FUTURE_HOLDOUT_START = date(2026, 8, 12)
OBSERVED_V1_TEST_ACCESS_ALLOWED = False
FUTURE_HOLDOUT_STATUS = "ACCUMULATING"
FUTURE_HOLDOUT_OBSERVED = False
TARGET_HORIZONS = (1, 3, 5)
RANDOM_SEED = 0
FOLD_COUNT = 5
EMBARGO_SESSIONS = 1
DIRECTIONS = ("DOWN", "FLAT", "UP")

FROZEN_DATASET_SHA = "92e8d813b755f715da8a3323fd540c8400943773202ffe5cc1c7ac6033c35425"
FROZEN_SPLIT_SHA = "6dad767f62e69e5e55e25bc48998707d3aeafbec76f65334bbdd0ad4bec85929"
FROZEN_FEATURE_SCHEMA_SHA = "83aee83bb403d7035e1e3daabe5c05b680e8263ea6a36e3c7812d11c20f838e0"
FROZEN_BASELINE_ARTIFACT_SHA = "98c4aeff24815bcc946344a7a14d81f399023e395889fa5400394bc1c455064d"

DERIVED_FEATURE_NAMES = (
    "momentum_acceleration_5_20",
    "volatility_term_5_20",
    "volume_trend_5_20",
    "month_end_flag",
)
CROSS_SECTIONAL_BASES = (
    "return_20d",
    "volatility_20d",
    "relative_return_20d",
    "volume_ratio_20d",
    "price_to_sma_20d",
)


@dataclass(frozen=True, slots=True)
class DevelopmentFeatureRow:
    row_id: str
    ticker: str
    trade_date: date
    feature_as_of: date
    values: dict[str, float]

    def validate(self) -> None:
        if not DEVELOPMENT_FROM <= self.trade_date <= DEVELOPMENT_TO:
            raise ValueError("OBSERVED_TEST_READ_ATTEMPT")
        if not self.feature_as_of < self.trade_date:
            raise ValueError("feature availability must be strictly before target date")


@dataclass(frozen=True, slots=True)
class OneSessionTarget:
    row_id: str
    ticker: str
    trade_date: date
    security_return: float
    benchmark_return: float


@dataclass(frozen=True, slots=True)
class HorizonTarget:
    row_id: str
    ticker: str
    trade_date: date
    horizon: int
    security_return: float
    benchmark_return: float
    abnormal_return: float
    direction: str


@dataclass(frozen=True, slots=True)
class DevelopmentDataset:
    rows: tuple[DevelopmentFeatureRow, ...]
    targets: dict[int, dict[str, HorizonTarget]]
    feature_names: tuple[str, ...]
    feature_schema_sha: str
    dataset_sha: str
    split_sha: str
    price_adjustment_status: str
    source_usage_readiness: str

    def aligned_rows(self, horizon: int) -> tuple[DevelopmentFeatureRow, ...]:
        available = self.targets[horizon]
        return tuple(row for row in self.rows if row.row_id in available)


@dataclass(frozen=True, slots=True)
class RollingFold:
    fold_id: str
    horizon: int
    train_row_ids: tuple[str, ...]
    validation_row_ids: tuple[str, ...]
    purged_dates: tuple[str, ...]
    embargoed_dates: tuple[str, ...]
    train_range: dict[str, str]
    validation_range: dict[str, str]

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FoldManifest:
    folds: tuple[RollingFold, ...]
    fold_manifest_sha: str
    strategy: str = "DATE_GROUPED_EXPANDING_PURGED_EMBARGOED"
    random_split_used: bool = False
    observed_test_used: bool = False
    future_holdout_used: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "random_split_used": self.random_split_used,
            "observed_test_used": self.observed_test_used,
            "future_holdout_used": self.future_holdout_used,
            "folds": [fold.payload() for fold in self.folds],
            "fold_manifest_sha": self.fold_manifest_sha,
        }


def direction_for_return(value: float, horizon: int) -> str:
    threshold = 0.002 * horizon**0.5
    if value > threshold:
        return "UP"
    if value < -threshold:
        return "DOWN"
    return "FLAT"


def safety_flags() -> dict[str, bool]:
    return {
        "RESEARCH_ONLY": True,
        "OBSERVED_TEST_USED": False,
        "FUTURE_HOLDOUT_USED": False,
        "BACKTEST_APPROVED": False,
        "PAPER_TRADING_APPROVED": False,
        "REAL_TRADING_APPROVED": False,
        "REAL_TRADING_ALLOWED": False,
        "REAL_ORDER_SUBMISSION_ALLOWED": False,
        "REAL_STOP_ORDER_ALLOWED": False,
        "REAL_MONEY_MOVEMENT_ALLOWED": False,
        "BROKER_ACCOUNT_MUTATION_ALLOWED": False,
        "MARGIN_TRADING_ALLOWED": False,
        "LIVE_EXECUTION_ALLOWED": False,
        "PAPER_TRADING_ALLOWED": False,
        "SANDBOX_ORDER_SUBMISSION_ALLOWED": False,
        "BUY_SELL_GENERATED": False,
        "PAID_SERVICES_USED": False,
    }


def frozen_research_metadata() -> dict[str, object]:
    return {
        "development_from": DEVELOPMENT_FROM.isoformat(),
        "development_to": DEVELOPMENT_TO.isoformat(),
        "observed_test_start": OBSERVED_TEST_START.isoformat(),
        "OBSERVED_V1_TEST_ACCESS_ALLOWED": OBSERVED_V1_TEST_ACCESS_ALLOWED,
        "future_holdout_start": FUTURE_HOLDOUT_START.isoformat(),
        "FUTURE_HOLDOUT_STATUS": FUTURE_HOLDOUT_STATUS,
        "FUTURE_HOLDOUT_OBSERVED": FUTURE_HOLDOUT_OBSERVED,
        "frozen_dataset_sha": FROZEN_DATASET_SHA,
        "frozen_split_sha": FROZEN_SPLIT_SHA,
        "frozen_feature_schema_sha": FROZEN_FEATURE_SCHEMA_SHA,
        "frozen_baseline_artifact_sha": FROZEN_BASELINE_ARTIFACT_SHA,
        "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
        "source_usage_readiness": SOURCE_USAGE_READINESS,
    }


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
