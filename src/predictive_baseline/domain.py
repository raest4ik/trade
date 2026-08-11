from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

MODEL_VERSION = "predictive-daily-baseline-v1"
CONFIG_VERSION = "predictive-daily-baseline-config-v1"
TARGET_HORIZON = "DATE_SAFE_DAILY"
MIN_REAL_TRAINING_ROWS = 100
CALIBRATION_MIN_VALIDATION_ROWS = 100
RANDOM_SEED = 0
DEVELOPMENT_WARNING = "DEVELOPMENT_SMOKE_ONLY"
NON_TRADING_WARNING = "NOT_VALID_FOR_TRADING"
NON_PERFORMANCE_WARNING = "NOT_A_PERFORMANCE_ESTIMATE"

CATEGORICAL_FEATURE_NAMES = (
    "ticker",
    "event_type",
    "source",
    "timestamp_quality",
)
FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "abnormal_return",
        "benchmark_return",
        "security_return",
        "target_close",
        "target_security_close",
        "target_imoex_close",
        "future_price",
        "future_prices",
        "future_return",
        "future_volume",
    }
)


class TrainingGate(StrEnum):
    TRAINING_BLOCKED = "TRAINING_BLOCKED"
    PILOT_TRAINING_ALLOWED = "PILOT_TRAINING_ALLOWED"
    BASELINE_EXPERIMENT_ALLOWED = "BASELINE_EXPERIMENT_ALLOWED"
    BASELINE_TRAINING_READY = "BASELINE_TRAINING_READY"


class RunMode(StrEnum):
    REAL = "REAL"
    DEVELOPMENT_SMOKE = "DEVELOPMENT_SMOKE"


class SplitName(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class Direction(StrEnum):
    DOWN = "DOWN"
    FLAT = "FLAT"
    UP = "UP"


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    version: str = CONFIG_VERSION
    random_seed: int = RANDOM_SEED
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    flat_threshold: float = 0.002
    ridge_alpha: float = 1.0
    embargo_days: int = 1
    purge_overlapping_labels: bool = True
    calibration_min_validation_rows: int = CALIBRATION_MIN_VALIDATION_ROWS

    def validate(self) -> None:
        if self.version != CONFIG_VERSION:
            raise ValueError("unsupported baseline config version")
        if self.random_seed != RANDOM_SEED:
            raise ValueError("baseline random seed is frozen")
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1.0) > 1e-12:
            raise ValueError("temporal split fractions must sum to one")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("temporal split fractions must be positive")
        if self.flat_threshold <= 0 or self.ridge_alpha < 0 or self.embargo_days < 0:
            raise ValueError("baseline config contains an invalid numeric value")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def fingerprint(self) -> str:
        return sha256_payload(self.payload())


@dataclass(frozen=True, slots=True)
class PredictiveRow:
    news_id: UUID
    ticker: str
    source: str
    timestamp_quality: str
    event_type: str
    publication_date: date
    baseline_session_date: date
    target_session_date: date
    prediction_time: datetime
    numeric_features: dict[str, float | None]
    abnormal_return: float
    dataset_version: str

    def validate(self) -> None:
        if not self.baseline_session_date < self.publication_date < self.target_session_date:
            raise ValueError("predictive row has a non-causal daily label window")
        if self.prediction_time.tzinfo is None or self.prediction_time.utcoffset() is None:
            raise ValueError("prediction_time must be timezone-aware")
        if self.prediction_time.date() > self.baseline_session_date:
            raise ValueError("features became available after the baseline session")
        if not all((self.ticker, self.source, self.timestamp_quality, self.event_type)):
            raise ValueError("predictive categorical fields must not be blank")
        assert_feature_names_safe(self.numeric_features)

    def direction(self, threshold: float) -> Direction:
        if self.abnormal_return > threshold:
            return Direction.UP
        if self.abnormal_return < -threshold:
            return Direction.DOWN
        return Direction.FLAT

    def categorical_features(self) -> dict[str, str]:
        return {
            "ticker": self.ticker,
            "event_type": self.event_type,
            "source": self.source,
            "timestamp_quality": self.timestamp_quality,
        }


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    rows: tuple[PredictiveRow, ...]
    dataset_sha256: str
    feature_schema_sha256: str
    numeric_feature_names: tuple[str, ...]
    dataset_version: str


@dataclass(frozen=True, slots=True)
class TemporalSplitResult:
    train: tuple[PredictiveRow, ...]
    validation: tuple[PredictiveRow, ...]
    test: tuple[PredictiveRow, ...]
    purged_news_ids: tuple[UUID, ...]
    embargoed_news_ids: tuple[UUID, ...]
    split_sha256: str

    def validate(self) -> None:
        if not self.train or not self.validation or not self.test:
            raise ValueError("temporal split requires non-empty train, validation, and test")
        train_latest = max(item.publication_date for item in self.train)
        validation_earliest = min(item.publication_date for item in self.validation)
        validation_latest = max(item.publication_date for item in self.validation)
        test_earliest = min(item.publication_date for item in self.test)
        if not train_latest < validation_earliest:
            raise ValueError("TRAIN must be strictly older than VALIDATION")
        if not validation_latest < test_earliest:
            raise ValueError("VALIDATION must be strictly older than TEST")
        split_ids = [item.news_id for item in (*self.train, *self.validation, *self.test)]
        if len(split_ids) != len(set(split_ids)):
            raise ValueError("one event appears in multiple temporal splits")

    def assignments(self) -> dict[UUID, SplitName]:
        return {
            **{item.news_id: SplitName.TRAIN for item in self.train},
            **{item.news_id: SplitName.VALIDATION for item in self.validation},
            **{item.news_id: SplitName.TEST for item in self.test},
        }


@dataclass(frozen=True, slots=True)
class PredictionOutput:
    news_id: UUID
    ticker: str
    prediction_time: datetime
    model_version: str
    target_horizon: str
    predicted_direction: Direction
    prob_up: float
    prob_flat: float
    prob_down: float
    predicted_abnormal_return: float
    model_probability: float
    dataset_version: str

    def payload(self) -> dict[str, Any]:
        return {
            "news_id": str(self.news_id),
            "ticker": self.ticker,
            "prediction_time": self.prediction_time.isoformat(),
            "model_version": self.model_version,
            "target_horizon": self.target_horizon,
            "predicted_direction": self.predicted_direction.value,
            "prob_up": self.prob_up,
            "prob_flat": self.prob_flat,
            "prob_down": self.prob_down,
            "predicted_abnormal_return": self.predicted_abnormal_return,
            "model_probability": self.model_probability,
            "dataset_version": self.dataset_version,
        }


@dataclass(frozen=True, slots=True)
class TrainingResult:
    mode: RunMode
    gate: TrainingGate
    status: str
    classification_status: str
    regression_status: str
    calibration_status: str
    backtest_status: str
    split: TemporalSplitResult | None
    classification_metrics: dict[str, Any]
    regression_metrics: dict[str, Any]
    predictions: tuple[PredictionOutput, ...]
    model_binary: bytes | None
    model_binary_sha256: str | None
    test_used_for_tuning: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelArtifactManifest:
    model_version: str
    dataset_sha256: str
    feature_schema_sha256: str
    split_sha256: str
    git_sha: str
    training_config: dict[str, Any]
    training_period: dict[str, str]
    validation_period: dict[str, str]
    test_period: dict[str, str]
    metrics: dict[str, Any]
    model_binary_sha256: str | None
    created_at: str
    mode: str
    warnings: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


def training_gate(feature_ready: int) -> TrainingGate:
    if feature_ready < 100:
        return TrainingGate.TRAINING_BLOCKED
    if feature_ready < 500:
        return TrainingGate.PILOT_TRAINING_ALLOWED
    if feature_ready < 1000:
        return TrainingGate.BASELINE_EXPERIMENT_ALLOWED
    return TrainingGate.BASELINE_TRAINING_READY


def readiness_payload(
    *,
    daily_feature_ready: int,
    intraday_feature_ready: int,
    ticker_count: int,
    source_count: int,
    date_from: str | None,
    date_to: str | None,
) -> dict[str, Any]:
    gate = training_gate(daily_feature_ready)
    blocked = gate == TrainingGate.TRAINING_BLOCKED
    return {
        "data_status": gate.value,
        "classification_status": "NOT_TRAINED" if blocked else "READY_FOR_TRAINING",
        "regression_status": "NOT_TRAINED" if blocked else "READY_FOR_TRAINING",
        "calibration_status": "NOT_READY",
        "backtest_status": "NOT_READY",
        "daily_feature_ready": daily_feature_ready,
        "intraday_feature_ready": intraday_feature_ready,
        "ticker_count": ticker_count,
        "source_count": source_count,
        "date_from": date_from,
        "date_to": date_to,
        "date_range": {"from": date_from, "to": date_to},
        "rows_to_100": max(0, 100 - daily_feature_ready),
        "rows_to_500": max(0, 500 - daily_feature_ready),
        "rows_to_1000": max(0, 1000 - daily_feature_ready),
        "training_gate": gate.value,
        "zero_cost": True,
        "paid_ml_api_used": False,
    }


def assert_feature_names_safe(features: Mapping[str, object]) -> None:
    normalized = {name.strip().lower() for name in features}
    forbidden = normalized & FORBIDDEN_FEATURE_NAMES
    if forbidden:
        raise ValueError("target/future columns found in X: " + ", ".join(sorted(forbidden)))
    for name in normalized:
        if name.startswith(("future_", "target_")) or name.endswith("_target"):
            raise ValueError(f"future/target feature is forbidden: {name}")


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
