from __future__ import annotations

import math
import pickle
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.predictive_baseline.data import deterministic_purged_temporal_split
from src.predictive_baseline.domain import (
    CATEGORICAL_FEATURE_NAMES,
    DEVELOPMENT_WARNING,
    MIN_REAL_TRAINING_ROWS,
    MODEL_VERSION,
    NON_PERFORMANCE_WARNING,
    NON_TRADING_WARNING,
    TARGET_HORIZON,
    BaselineConfig,
    Direction,
    LoadedDataset,
    PredictionOutput,
    PredictiveRow,
    RunMode,
    TrainingGate,
    TrainingResult,
    training_gate,
)
from src.predictive_baseline.sklearn_adapter import (
    LogisticRegressionAdapter,
    RidgeRegressionAdapter,
)

UNKNOWN_CATEGORY = "__UNKNOWN__"
_DIRECTIONS = (Direction.DOWN.value, Direction.FLAT.value, Direction.UP.value)


@dataclass(slots=True)
class DeterministicPreprocessor:
    numeric_names: tuple[str, ...]
    numeric_medians: dict[str, float]
    numeric_means: dict[str, float]
    numeric_scales: dict[str, float]
    categories: dict[str, tuple[str, ...]]
    fitted_news_ids: tuple[str, ...]

    @classmethod
    def create(cls, numeric_names: tuple[str, ...]) -> DeterministicPreprocessor:
        return cls(
            numeric_names=numeric_names,
            numeric_medians={},
            numeric_means={},
            numeric_scales={},
            categories={},
            fitted_news_ids=(),
        )

    def fit(self, rows: tuple[PredictiveRow, ...]) -> DeterministicPreprocessor:
        if not rows:
            raise ValueError("preprocessor requires TRAIN rows")
        self.fitted_news_ids = tuple(str(row.news_id) for row in rows)
        for name in self.numeric_names:
            observed = [
                value for row in rows if (value := row.numeric_features.get(name)) is not None
            ]
            median = float(statistics.median(observed)) if observed else 0.0
            filled: list[float] = []
            for row in rows:
                raw = row.numeric_features.get(name)
                filled.append(median if raw is None else raw)
            mean = statistics.fmean(filled)
            scale = math.sqrt(statistics.fmean([(value - mean) ** 2 for value in filled]))
            self.numeric_medians[name] = median
            self.numeric_means[name] = mean
            self.numeric_scales[name] = scale if scale > 0 else 1.0
        for name in CATEGORICAL_FEATURE_NAMES:
            observed = sorted({row.categorical_features()[name] for row in rows})
            self.categories[name] = tuple([*observed, UNKNOWN_CATEGORY])
        return self

    def transform(self, rows: tuple[PredictiveRow, ...]) -> list[list[float]]:
        if not self.fitted_news_ids:
            raise ValueError("preprocessor must be fit on TRAIN before transform")
        matrix: list[list[float]] = []
        for row in rows:
            values: list[float] = []
            for name in self.numeric_names:
                raw = row.numeric_features.get(name)
                filled = self.numeric_medians[name] if raw is None else raw
                values.append((filled - self.numeric_means[name]) / self.numeric_scales[name])
            row_categories = row.categorical_features()
            for name in CATEGORICAL_FEATURE_NAMES:
                categories = self.categories[name]
                value = row_categories[name]
                resolved = value if value in categories else UNKNOWN_CATEGORY
                values.extend(1.0 if category == resolved else 0.0 for category in categories)
            matrix.append(values)
        return matrix

    def feature_names(self) -> tuple[str, ...]:
        categorical = tuple(
            f"{name}={category}"
            for name in CATEGORICAL_FEATURE_NAMES
            for category in self.categories.get(name, ())
        )
        return (*self.numeric_names, *categorical)


@dataclass(slots=True)
class BaselineModels:
    preprocessor: DeterministicPreprocessor
    classifier: LogisticRegressionAdapter
    regressor: RidgeRegressionAdapter

    @classmethod
    def create(cls, dataset: LoadedDataset, config: BaselineConfig) -> BaselineModels:
        return cls(
            preprocessor=DeterministicPreprocessor.create(dataset.numeric_feature_names),
            classifier=LogisticRegressionAdapter(random_seed=config.random_seed),
            regressor=RidgeRegressionAdapter(alpha=config.ridge_alpha),
        )

    def fit(self, rows: tuple[PredictiveRow, ...], config: BaselineConfig) -> None:
        x_train = self.preprocessor.fit(rows).transform(rows)
        y_classification = [row.direction(config.flat_threshold).value for row in rows]
        if len(set(y_classification)) < 2:
            raise ValueError("classification TRAIN requires at least two direction classes")
        self.classifier.fit(x_train, y_classification)
        self.regressor.fit(x_train, [row.abnormal_return for row in rows])

    def predict(
        self, rows: tuple[PredictiveRow, ...]
    ) -> tuple[list[str], list[list[float]], list[float]]:
        matrix = self.preprocessor.transform(rows)
        predicted_class = self.classifier.predict(matrix)
        raw_probabilities = self.classifier.predict_probabilities(matrix)
        observed_classes = self.classifier.classes()
        aligned = [
            [
                probabilities[observed_classes.index(label)] if label in observed_classes else 0.0
                for label in _DIRECTIONS
            ]
            for probabilities in raw_probabilities
        ]
        predicted_regression = self.regressor.predict(matrix)
        return predicted_class, aligned, predicted_regression

    def binary(self) -> bytes:
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)


def train_predictive_baselines(
    dataset: LoadedDataset,
    config: BaselineConfig,
    *,
    mode: RunMode,
) -> TrainingResult:
    gate = training_gate(len(dataset.rows))
    if mode == RunMode.REAL and gate == TrainingGate.TRAINING_BLOCKED:
        return TrainingResult(
            mode=mode,
            gate=gate,
            status=TrainingGate.TRAINING_BLOCKED.value,
            classification_status="NOT_TRAINED",
            regression_status="NOT_TRAINED",
            calibration_status="NOT_READY",
            backtest_status="NOT_READY",
            split=None,
            classification_metrics={},
            regression_metrics={},
            predictions=(),
            model_binary=None,
            model_binary_sha256=None,
            test_used_for_tuning=False,
            warnings=(f"daily_feature_ready < {MIN_REAL_TRAINING_ROWS}",),
        )
    split = deterministic_purged_temporal_split(dataset.rows, config)
    models = BaselineModels.create(dataset, config)
    models.fit(split.train, config)
    validation = _evaluate(models, split.validation, split.train, config)
    test = _evaluate(models, split.test, split.train, config)
    test_classes, test_probabilities, test_regression = models.predict(split.test)
    predictions = tuple(
        _prediction(row, direction, probabilities, regression)
        for row, direction, probabilities, regression in zip(
            split.test,
            test_classes,
            test_probabilities,
            test_regression,
            strict=True,
        )
    )
    smoke = mode == RunMode.DEVELOPMENT_SMOKE
    warnings = (DEVELOPMENT_WARNING, NON_TRADING_WARNING, NON_PERFORMANCE_WARNING) if smoke else ()
    binary = None if smoke else models.binary()
    binary_hash = None if binary is None else _sha256_bytes(binary)
    calibration_status = (
        "ELIGIBLE_NOT_APPLIED"
        if len(split.validation) >= config.calibration_min_validation_rows
        else "NOT_READY_INSUFFICIENT_VALIDATION"
    )
    return TrainingResult(
        mode=mode,
        gate=gate,
        status=DEVELOPMENT_WARNING if smoke else "SUCCEEDED",
        classification_status=DEVELOPMENT_WARNING if smoke else "TRAINED",
        regression_status=DEVELOPMENT_WARNING if smoke else "TRAINED",
        calibration_status=calibration_status,
        backtest_status="NOT_READY",
        split=split,
        classification_metrics={
            "validation": validation["classification"],
            "test": test["classification"],
        },
        regression_metrics={
            "validation": validation["regression"],
            "test": test["regression"],
        },
        predictions=predictions,
        model_binary=binary,
        model_binary_sha256=binary_hash,
        test_used_for_tuning=False,
        warnings=warnings,
    )


def _evaluate(
    models: BaselineModels,
    rows: tuple[PredictiveRow, ...],
    train_rows: tuple[PredictiveRow, ...],
    config: BaselineConfig,
) -> dict[str, Any]:
    predicted_class, probabilities, predicted_regression = models.predict(rows)
    actual_class = [row.direction(config.flat_threshold).value for row in rows]
    actual_regression = [row.abnormal_return for row in rows]
    majority = Counter(
        row.direction(config.flat_threshold).value for row in train_rows
    ).most_common(1)[0][0]
    train_mean = statistics.fmean(row.abnormal_return for row in train_rows)
    return {
        "classification": {
            "model": classification_metrics(actual_class, predicted_class, probabilities),
            "naive_majority": classification_metrics(
                actual_class,
                [majority] * len(rows),
                None,
            ),
        },
        "regression": {
            "model": regression_metrics(actual_regression, predicted_regression),
            "naive_train_mean": regression_metrics(actual_regression, [train_mean] * len(rows)),
            "naive_zero": regression_metrics(actual_regression, [0.0] * len(rows)),
            "train_target_mean": train_mean,
            "evaluation_target_mean": statistics.fmean(actual_regression),
            "prediction_mean": statistics.fmean(predicted_regression),
        },
    }


def classification_metrics(
    actual: list[str],
    predicted: list[str],
    probabilities: list[list[float]] | None,
) -> dict[str, Any]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("classification metrics require aligned non-empty values")
    confusion = {
        label: {predicted_label: 0 for predicted_label in _DIRECTIONS} for label in _DIRECTIONS
    }
    for expected, observed in zip(actual, predicted, strict=True):
        confusion[expected][observed] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for label in _DIRECTIONS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in _DIRECTIONS if other != label)
        false_negative = sum(confusion[label][other] for other in _DIRECTIONS if other != label)
        support = sum(confusion[label].values())
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = _ratio(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    recalls = [
        float(per_class[label]["recall"]) for label in _DIRECTIONS if per_class[label]["support"]
    ]
    result: dict[str, Any] = {
        "accuracy": sum(
            expected == observed for expected, observed in zip(actual, predicted, strict=True)
        )
        / len(actual),
        "balanced_accuracy": statistics.fmean(recalls),
        "macro_f1": statistics.fmean(float(per_class[label]["f1"]) for label in _DIRECTIONS),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
    if probabilities is not None:
        result["log_loss"] = _multiclass_log_loss(actual, probabilities)
        result["brier_score"] = _multiclass_brier(actual, probabilities)
    return result


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("regression metrics require aligned non-empty values")
    errors = [observed - expected for expected, observed in zip(actual, predicted, strict=True)]
    mae = statistics.fmean(abs(error) for error in errors)
    rmse = math.sqrt(statistics.fmean(error**2 for error in errors))
    mean_actual = statistics.fmean(actual)
    total = sum((value - mean_actual) ** 2 for value in actual)
    residual = sum(error**2 for error in errors)
    r_squared = None if total == 0 else 1.0 - residual / total
    enough = len(actual) >= 3 and len(set(actual)) > 1 and len(set(predicted)) > 1
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r_squared,
        "pearson": _pearson(actual, predicted) if enough else None,
        "spearman": _pearson(_ranks(actual), _ranks(predicted)) if enough else None,
        "mean_target": mean_actual,
        "prediction_mean": statistics.fmean(predicted),
        "count": len(actual),
    }


def _prediction(
    row: PredictiveRow,
    direction: str,
    probabilities: list[float],
    regression: float,
) -> PredictionOutput:
    probability_by_class = dict(zip(_DIRECTIONS, probabilities, strict=True))
    return PredictionOutput(
        news_id=row.news_id,
        ticker=row.ticker,
        prediction_time=row.prediction_time,
        model_version=MODEL_VERSION,
        target_horizon=TARGET_HORIZON,
        predicted_direction=Direction(direction),
        prob_up=probability_by_class[Direction.UP.value],
        prob_flat=probability_by_class[Direction.FLAT.value],
        prob_down=probability_by_class[Direction.DOWN.value],
        predicted_abnormal_return=regression,
        model_probability=max(probabilities),
        dataset_version=row.dataset_version,
    )


def _multiclass_log_loss(actual: list[str], probabilities: list[list[float]]) -> float:
    epsilon = 1e-15
    losses: list[float] = []
    for expected, values in zip(actual, probabilities, strict=True):
        probability = values[_DIRECTIONS.index(expected)]
        losses.append(-math.log(min(max(probability, epsilon), 1.0 - epsilon)))
    return statistics.fmean(losses)


def _multiclass_brier(actual: list[str], probabilities: list[list[float]]) -> float:
    scores: list[float] = []
    for expected, values in zip(actual, probabilities, strict=True):
        scores.append(
            sum(
                (probability - (1.0 if label == expected else 0.0)) ** 2
                for label, probability in zip(_DIRECTIONS, values, strict=True)
            )
        )
    return statistics.fmean(scores)


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return 0.0 if denominator == 0 else numerator / denominator


def _ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average
        index = end
    return ranks


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()
