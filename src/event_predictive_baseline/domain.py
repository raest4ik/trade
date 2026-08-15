from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

MODEL_VERSION = "exact-event-predictive-baseline-v1"
DATASET_VERSION = "exact-event-market-dataset-v2"
EXPECTED_DATASET_SHA = "20ab67ff4d94c59d6cf714f8b2f7c048031bda120bbd92ceb6e6185a838e14c3"
EXPECTED_SOURCE_REGISTRY_SHA = "be67e1e33e7a0c07b16a34aae974796d1ae18c54d4fbc1fb53242cb0a731ced0"
EXPECTED_PROVENANCE_SHA = "bee365fdfa751c78b616172fddb50bc8e104bfc25e3e39d90137f0469bb14fc5"
EXPECTED_TIMESTAMP_SHA = "86149fa99993790c6fdaa451965f433f681849a04b1e436b9d81dbe9d9f76267"
EXPECTED_REACTION_SHA = "8594caa2a773fd6142f6f13b7ddeaa6ae51c926d69d6a72a85cca59a3ceced22"
EXPECTED_CLUSTER_SHA = "d5dcf6d81810e08534528d1e4f6faf5a51013ffa296a06e7d93903be7cc75a2a"
REACTION_FAMILY = "EXACT_INTRADAY"
PREDICTIVE_UNIT = "EVENT"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
PRIMARY_EXACT_HORIZON = "15m"
SECONDARY_EXACT_HORIZONS = ("1m", "5m", "30m", "60m")
EXACT_HORIZONS = ("1m", "5m", "15m", "30m", "60m")
FLAT_RETURN_THRESHOLD = 0.002
DIRECTIONS = ("DOWN", "FLAT", "UP")
FEATURE_FAMILIES = ("A_MARKET_ONLY", "B_EVENT_ONLY", "C_EVENT_PLUS_MARKET")
TEST_STATUS = "OBSERVED_AFTER_EXACT_BASELINE_V1"
PRICE_ADJUSTMENT_STATUS = "UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES"
MIN_GROUP_METRIC_ROWS = 10


class FutureHoldoutOutcomeReadError(RuntimeError):
    """Raised when research code tries to read future holdout outcomes."""


@dataclass(frozen=True, slots=True)
class EventFeatureRow:
    event_id: str
    event_cluster_id: str
    ticker: str
    issuer_name: str
    publication_date: date
    publication_timestamp_utc: datetime
    source_family: str
    event_features: dict[str, object]
    market_features: dict[str, float]

    def values_for(self, family: str) -> dict[str, object]:
        if family == "A_MARKET_ONLY":
            return {f"market__{name}": value for name, value in self.market_features.items()}
        if family == "B_EVENT_ONLY":
            return {f"event__{name}": value for name, value in self.event_features.items()}
        if family == "C_EVENT_PLUS_MARKET":
            return {
                **{f"event__{name}": value for name, value in self.event_features.items()},
                **{f"market__{name}": value for name, value in self.market_features.items()},
            }
        raise ValueError(f"unknown feature family: {family}")


@dataclass(frozen=True, slots=True)
class EventTargetRow:
    event_id: str
    horizon: str
    direction: str
    abnormal_return: float
    security_return: float
    benchmark_return: float
    window_begin_at: str
    window_end_at: str
    security_observed_at: str
    benchmark_observed_at: str


@dataclass(frozen=True, slots=True)
class HorizonCohort:
    horizon: str
    rows: tuple[EventFeatureRow, ...]
    targets: dict[str, EventTargetRow]
    event_feature_names: tuple[str, ...]
    market_feature_names: tuple[str, ...]
    cohort_sha: str
    event_schema_sha: str
    market_schema_sha: str
    target_schema_sha: str

    def rows_for(self, assignments: dict[str, str], split: str) -> tuple[EventFeatureRow, ...]:
        return tuple(row for row in self.rows if assignments[row.event_id] == split)


@dataclass(frozen=True, slots=True)
class FrozenModelConfig:
    model_version: str = MODEL_VERSION
    dataset_sha: str = EXPECTED_DATASET_SHA
    reaction_family: str = REACTION_FAMILY
    primary_horizon: str = PRIMARY_EXACT_HORIZON
    secondary_horizons: tuple[str, ...] = SECONDARY_EXACT_HORIZONS
    primary_regression_target: str = "exact_intraday_abnormal_return"
    classification_target: str = "UP_FLAT_DOWN"
    flat_return_threshold: float = FLAT_RETURN_THRESHOLD
    classifier: str = "sklearn.linear_model.LogisticRegression"
    classifier_parameters: tuple[tuple[str, object], ...] = (
        ("C", 1.0),
        ("max_iter", 2000),
        ("solver", "lbfgs"),
        ("random_state", 0),
    )
    regressor: str = "sklearn.linear_model.Ridge"
    regressor_parameters: tuple[tuple[str, object], ...] = (("alpha", 1.0),)
    preprocessing: str = (
        "DictVectorizer + StandardScaler inside sklearn Pipeline; fit only on TRAIN "
        "for validation and TRAIN+VALIDATION once for TEST"
    )
    hyperparameter_selection: str = "FIXED_A_PRIORI_NO_SEARCH"
    test_config_locked: bool = True
    random_seed: int = 0

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["classifier_parameters"] = dict(self.classifier_parameters)
        payload["regressor_parameters"] = dict(self.regressor_parameters)
        payload["feature_families"] = list(FEATURE_FAMILIES)
        payload["config_sha"] = sha256_payload(payload)
        return payload


def classify_abnormal_return(value: float) -> str:
    if value > FLAT_RETURN_THRESHOLD:
        return "UP"
    if value < -FLAT_RETURN_THRESHOLD:
        return "DOWN"
    return "FLAT"


def guard_future_holdout_outcome_read(publication_date: date, *, context: str) -> None:
    if publication_date >= FUTURE_EVENT_HOLDOUT_START:
        raise FutureHoldoutOutcomeReadError(f"FUTURE_EVENT_HOLDOUT_READ_ATTEMPT:{context}")


def research_safety_flags() -> dict[str, bool]:
    return {
        "RESEARCH_ONLY": True,
        "CONFIRMED_SIGNAL": False,
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


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
