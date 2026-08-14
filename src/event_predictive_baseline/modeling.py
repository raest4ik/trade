# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import pickle
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.event_predictive_baseline.domain import (
    DIRECTIONS,
    FEATURE_FAMILIES,
    EventFeatureRow,
    EventTargetRow,
    FrozenModelConfig,
)
from src.predictive_baseline.modeling import classification_metrics, regression_metrics


@dataclass(slots=True)
class FamilyModels:
    family: str
    classifier: Pipeline
    abnormal_regressor: Pipeline
    security_regressor: Pipeline
    fitted_event_ids: tuple[str, ...]

    @classmethod
    def create(cls, family: str, config: FrozenModelConfig) -> FamilyModels:
        if family not in FEATURE_FAMILIES:
            raise ValueError(f"unknown model family: {family}")
        return cls(
            family=family,
            classifier=_classifier(config),
            abnormal_regressor=_regressor(config),
            security_regressor=_regressor(config),
            fitted_event_ids=(),
        )

    def fit(self, rows: tuple[EventFeatureRow, ...], targets: dict[str, EventTargetRow]) -> None:
        x = [row.values_for(self.family) for row in rows]
        aligned = [targets[row.event_id] for row in rows]
        classes = {target.direction for target in aligned}
        if classes != set(DIRECTIONS):
            raise ValueError("classification fit requires all three direction classes")
        self.classifier.fit(x, [target.direction for target in aligned])
        self.abnormal_regressor.fit(x, [target.abnormal_return for target in aligned])
        self.security_regressor.fit(x, [target.security_return for target in aligned])
        self.fitted_event_ids = tuple(row.event_id for row in rows)

    def predict(self, rows: tuple[EventFeatureRow, ...]) -> dict[str, Any]:
        x = [row.values_for(self.family) for row in rows]
        estimator = self.classifier.named_steps["model"]
        observed_classes = [str(item) for item in estimator.classes_]
        probabilities = [
            [float(values[observed_classes.index(label)]) for label in DIRECTIONS]
            for values in self.classifier.predict_proba(x)
        ]
        return {
            "directions": [str(item) for item in self.classifier.predict(x)],
            "probabilities": probabilities,
            "abnormal_returns": [float(item) for item in self.abnormal_regressor.predict(x)],
            "security_returns": [float(item) for item in self.security_regressor.predict(x)],
        }


def fit_all_families(
    rows: tuple[EventFeatureRow, ...],
    targets: dict[str, EventTargetRow],
    config: FrozenModelConfig,
) -> dict[str, FamilyModels]:
    models = {family: FamilyModels.create(family, config) for family in FEATURE_FAMILIES}
    for model in models.values():
        model.fit(rows, targets)
    fitted = {model.fitted_event_ids for model in models.values()}
    if len(fitted) != 1:
        raise ValueError("A/B/C models were not fit on identical event rows")
    return models


def evaluate_all_families(
    models: dict[str, FamilyModels],
    rows: tuple[EventFeatureRow, ...],
    targets: dict[str, EventTargetRow],
    fit_targets: dict[str, EventTargetRow],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(models) != set(FEATURE_FAMILIES):
        raise ValueError("A/B/C model set is incomplete")
    predictions = {family: models[family].predict(rows) for family in FEATURE_FAMILIES}
    aligned = [targets[row.event_id] for row in rows]
    actual_class = [target.direction for target in aligned]
    actual_abnormal = [target.abnormal_return for target in aligned]
    actual_security = [target.security_return for target in aligned]
    majority, majority_probabilities = _majority(fit_targets)
    abnormal_mean = statistics.fmean(target.abnormal_return for target in fit_targets.values())
    security_mean = statistics.fmean(target.security_return for target in fit_targets.values())
    metrics = {
        "rows": len(rows),
        "class_distribution": dict(sorted(Counter(actual_class).items())),
        "classification": {
            "naive_majority": _classification(
                actual_class,
                [majority] * len(rows),
                [majority_probabilities] * len(rows),
            ),
            "models": {
                family: _classification(
                    actual_class,
                    predictions[family]["directions"],
                    predictions[family]["probabilities"],
                )
                for family in FEATURE_FAMILIES
            },
        },
        "regression": {
            "abnormal_return": {
                "naive_zero": regression_metrics(actual_abnormal, [0.0] * len(rows)),
                "naive_train_mean": regression_metrics(
                    actual_abnormal, [abnormal_mean] * len(rows)
                ),
                "models": {
                    family: regression_metrics(
                        actual_abnormal, predictions[family]["abnormal_returns"]
                    )
                    for family in FEATURE_FAMILIES
                },
            },
            "security_return": {
                "naive_zero": regression_metrics(actual_security, [0.0] * len(rows)),
                "naive_train_mean": regression_metrics(
                    actual_security, [security_mean] * len(rows)
                ),
                "models": {
                    family: regression_metrics(
                        actual_security, predictions[family]["security_returns"]
                    )
                    for family in FEATURE_FAMILIES
                },
            },
        },
    }
    records = []
    for index, (row, target) in enumerate(zip(rows, aligned, strict=True)):
        records.append(
            {
                "event_id": row.event_id,
                "ticker": row.ticker,
                "issuer_name": row.issuer_name,
                "publication_date": row.publication_date.isoformat(),
                "source_family": row.source_family,
                "actual_direction": target.direction,
                "actual_abnormal_return": target.abnormal_return,
                "actual_security_return": target.security_return,
                "models": {
                    family: {
                        "predicted_direction": predictions[family]["directions"][index],
                        "probabilities": dict(
                            zip(
                                DIRECTIONS,
                                predictions[family]["probabilities"][index],
                                strict=True,
                            )
                        ),
                        "predicted_abnormal_return": predictions[family]["abnormal_returns"][index],
                        "predicted_security_return": predictions[family]["security_returns"][index],
                    }
                    for family in FEATURE_FAMILIES
                },
            }
        )
    return metrics, records


def metrics_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    actual_class = [str(row["actual_direction"]) for row in records]
    actual_abnormal = [float(row["actual_abnormal_return"]) for row in records]
    actual_security = [float(row["actual_security_return"]) for row in records]
    return {
        "rows": len(records),
        "class_distribution": dict(sorted(Counter(actual_class).items())),
        "classification": {
            family: _classification(
                actual_class,
                [str(row["models"][family]["predicted_direction"]) for row in records],
                [
                    [float(row["models"][family]["probabilities"][label]) for label in DIRECTIONS]
                    for row in records
                ],
            )
            for family in FEATURE_FAMILIES
        },
        "regression": {
            "abnormal_return": {
                family: regression_metrics(
                    actual_abnormal,
                    [float(row["models"][family]["predicted_abnormal_return"]) for row in records],
                )
                for family in FEATURE_FAMILIES
            },
            "security_return": {
                family: regression_metrics(
                    actual_security,
                    [float(row["models"][family]["predicted_security_return"]) for row in records],
                )
                for family in FEATURE_FAMILIES
            },
        },
    }


def serialize_models(models: dict[str, FamilyModels]) -> bytes:
    return pickle.dumps(models, protocol=pickle.HIGHEST_PROTOCOL)


def _classifier(config: FrozenModelConfig) -> Pipeline:
    return Pipeline(
        [
            ("vectorize", DictVectorizer(sparse=False)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def _regressor(config: FrozenModelConfig) -> Pipeline:
    return Pipeline(
        [
            ("vectorize", DictVectorizer(sparse=False)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def _majority(targets: dict[str, EventTargetRow]) -> tuple[str, list[float]]:
    counts = Counter(target.direction for target in targets.values())
    majority = sorted(DIRECTIONS, key=lambda label: (-counts[label], label))[0]
    total = sum(counts.values())
    return majority, [counts[label] / total for label in DIRECTIONS]


def _classification(
    actual: list[str], predicted: list[str], probabilities: list[list[float]]
) -> dict[str, Any]:
    result = classification_metrics(actual, predicted, probabilities)
    result["weighted_f1"] = sum(
        float(result["per_class"][label]["f1"]) * int(result["per_class"][label]["support"])
        for label in DIRECTIONS
    ) / len(actual)
    result["actual_distribution"] = dict(sorted(Counter(actual).items()))
    result["predicted_distribution"] = dict(sorted(Counter(predicted).items()))
    return result
