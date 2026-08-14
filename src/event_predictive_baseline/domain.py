from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

MODEL_VERSION = "event-predictive-baseline-v1"
DATASET_VERSION = "event-market-predictive-dataset-v2"
EXPECTED_DATASET_SHA = "dea6f55aef3b8bcbf42299891bb48926b403286ea451175f23c2ac295f4f60f4"
EXPECTED_SOURCE_REGISTRY_SHA = "9c43f2676f99287cee7d4d443ca96cb33ed4d0140305262ab59c865878a99271"
EXPECTED_PROVENANCE_SHA = "9d532efd72ddcc84cca09dcf0b5ced7f990a94c1e09b5a3b96681408c0fd8d35"
EXPECTED_FEATURE_SCHEMA_SHA = "4be00e812ba4e23a5245c0d132057bbb5d2e4fc1c6b50ea40a85ae476cfe34cc"
REACTION_FAMILY = "DATE_SAFE_DAILY"
EXACT_REACTION_FAMILY = "EXACT_INTRADAY"
PREDICTIVE_UNIT = "EVENT"
FLAT_RETURN_THRESHOLD = 0.002
DIRECTIONS = ("DOWN", "FLAT", "UP")
FEATURE_FAMILIES = ("A_MARKET_ONLY", "B_EVENT_ONLY", "C_EVENT_PLUS_MARKET")
TRAIN_END = date(2024, 12, 31)
VALIDATION_END = date(2025, 12, 31)
TEST_STATUS = "OBSERVED_AFTER_EVENT_BASELINE_V1"
PRICE_ADJUSTMENT_STATUS = "UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES"


@dataclass(frozen=True, slots=True)
class EventFeatureRow:
    event_id: str
    ticker: str
    issuer_name: str
    publication_date: date
    source_family: str
    title_hash: str | None
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
    direction: str
    abnormal_return: float
    security_return: float


@dataclass(frozen=True, slots=True)
class ComparisonCohort:
    rows: tuple[EventFeatureRow, ...]
    event_feature_names: tuple[str, ...]
    market_feature_names: tuple[str, ...]
    cohort_sha: str
    event_schema_sha: str
    market_schema_sha: str

    def rows_for(self, assignments: dict[str, str], split: str) -> tuple[EventFeatureRow, ...]:
        return tuple(row for row in self.rows if assignments[row.event_id] == split)


@dataclass(frozen=True, slots=True)
class FrozenModelConfig:
    model_version: str = MODEL_VERSION
    dataset_sha: str = EXPECTED_DATASET_SHA
    reaction_family: str = REACTION_FAMILY
    primary_regression_target: str = "next_post_event_abnormal_return"
    secondary_regression_target: str = "raw_security_post_event_return"
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
    preprocessing: str = "DictVectorizer + StandardScaler fit only on authorized fit rows"
    hyperparameter_selection: str = "FIXED_A_PRIORI_NO_SEARCH"
    final_fit_partition: str = "TRAIN_PLUS_VALIDATION"
    test_config_locked: bool = True
    random_seed: int = 0

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["classifier_parameters"] = dict(self.classifier_parameters)
        payload["regressor_parameters"] = dict(self.regressor_parameters)
        payload["feature_families"] = list(FEATURE_FAMILIES)
        payload["config_sha"] = sha256_payload(payload)
        return payload


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
