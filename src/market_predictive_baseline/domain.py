from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from src.tinvest_market.domain import (
    CLASSIFICATION_POLICY_VERSION,
    FEATURE_DATASET_VERSION,
    FLAT_RETURN_THRESHOLD,
    feature_names,
)
from src.tinvest_market.policy import PRICE_ADJUSTMENT_STATUS, SOURCE_USAGE_READINESS

MODEL_VERSION = "tinvest-market-predictive-baseline-v1"
TARGET_VERSION = "tinvest-next-session-targets-v1"
EXPECTED_DATASET_SHA = "92e8d813b755f715da8a3323fd540c8400943773202ffe5cc1c7ac6033c35425"
EXPECTED_SPLIT_SHA = "6dad767f62e69e5e55e25bc48998707d3aeafbec76f65334bbdd0ad4bec85929"
EXPECTED_FEATURE_SCHEMA_SHA = "83aee83bb403d7035e1e3daabe5c05b680e8263ea6a36e3c7812d11c20f838e0"
RANDOM_SEED = 0
PRIMARY_TARGET = "next_session_abnormal_return"
SECONDARY_TARGET = "next_session_return"
DIRECTIONS = ("DOWN", "FLAT", "UP")
ASSOCIATIONAL_WARNING = "ASSOCIATIONAL_MODEL_ONLY"
PRICE_WARNING = PRICE_ADJUSTMENT_STATUS
TEST_STATUS = "OBSERVED_AFTER_BASELINE_V1"


@dataclass(frozen=True, slots=True)
class MarketFeatureRow:
    row_id: str
    ticker: str
    trade_date: date
    feature_as_of: date
    values: tuple[float, ...]

    def validate(self, expected_width: int) -> None:
        if not self.row_id or not self.ticker or len(self.values) != expected_width:
            raise ValueError("invalid market feature row")
        if not self.feature_as_of < self.trade_date:
            raise ValueError("market features must be available before their target session")


@dataclass(frozen=True, slots=True)
class MarketTargetRow:
    row_id: str
    ticker: str
    trade_date: date
    direction: str
    abnormal_return: float
    security_return: float


@dataclass(frozen=True, slots=True)
class FrozenMarketDataset:
    features: tuple[MarketFeatureRow, ...]
    assignments: dict[str, str]
    date_ranges: dict[str, dict[str, str]]
    counts: dict[str, int]
    dataset_sha: str
    split_sha: str
    feature_schema_sha: str
    feature_names: tuple[str, ...]
    dataset_version: str
    source_usage_readiness: str
    price_adjustment_status: str

    def rows_for(self, split: str) -> tuple[MarketFeatureRow, ...]:
        return tuple(row for row in self.features if self.assignments.get(row.row_id) == split)


@dataclass(frozen=True, slots=True)
class FinalModelConfig:
    model_version: str = MODEL_VERSION
    dataset_version: str = FEATURE_DATASET_VERSION
    dataset_sha: str = EXPECTED_DATASET_SHA
    split_sha: str = EXPECTED_SPLIT_SHA
    feature_schema_sha: str = EXPECTED_FEATURE_SCHEMA_SHA
    feature_names: tuple[str, ...] = feature_names(True)
    primary_target: str = PRIMARY_TARGET
    secondary_target: str = SECONDARY_TARGET
    classification_target: str = "direction(next_session_return)"
    classification_policy_version: str = CLASSIFICATION_POLICY_VERSION
    flat_return_threshold: float = FLAT_RETURN_THRESHOLD
    classifier: str = "sklearn.linear_model.LogisticRegression"
    classifier_parameters: tuple[tuple[str, object], ...] = (
        ("C", 1.0),
        ("max_iter", 2000),
        ("solver", "lbfgs"),
        ("random_state", RANDOM_SEED),
    )
    regressors: tuple[str, ...] = (PRIMARY_TARGET, SECONDARY_TARGET)
    regressor: str = "sklearn.linear_model.Ridge"
    regressor_parameters: tuple[tuple[str, object], ...] = (("alpha", 1.0),)
    preprocessing: str = "StandardScaler fit only on the stage-authorized fit partition"
    random_seed: int = RANDOM_SEED
    hyperparameter_selection: str = "FIXED_A_PRIORI_NO_SEARCH"
    train_stage: str = "TRAIN"
    validation_stage: str = "TRAIN_TO_VALIDATION"
    final_fit_stage: str = "TRAIN_PLUS_VALIDATION"
    test_config_locked: bool = True

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        payload["classifier_parameters"] = dict(self.classifier_parameters)
        payload["regressors"] = list(self.regressors)
        payload["regressor_parameters"] = dict(self.regressor_parameters)
        payload["config_sha"] = sha256_payload(payload)
        return payload


def research_safety_flags() -> dict[str, bool]:
    return {
        "RESEARCH_BASELINE_ONLY": True,
        "VALID_FOR_TRADING": False,
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


def validate_frozen_metadata(dataset: FrozenMarketDataset) -> None:
    expected = {
        "dataset_sha": EXPECTED_DATASET_SHA,
        "split_sha": EXPECTED_SPLIT_SHA,
        "feature_schema_sha": EXPECTED_FEATURE_SCHEMA_SHA,
        "dataset_version": FEATURE_DATASET_VERSION,
        "source_usage_readiness": SOURCE_USAGE_READINESS,
        "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
    }
    for name, value in expected.items():
        if getattr(dataset, name) != value:
            raise ValueError(f"frozen market dataset mismatch: {name}")
    if dataset.counts != {"TRAIN": 30156, "VALIDATION": 9583, "TEST": 9852}:
        raise ValueError("frozen market split counts changed")


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
