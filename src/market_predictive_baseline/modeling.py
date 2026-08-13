# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.market_predictive_baseline.domain import (
    DIRECTIONS,
    FinalModelConfig,
    MarketFeatureRow,
    MarketTargetRow,
)
from src.predictive_baseline.modeling import classification_metrics, regression_metrics


@dataclass(slots=True)
class MarketModels:
    classifier: Pipeline
    abnormal_regressor: Pipeline
    return_regressor: Pipeline
    fitted_row_ids: tuple[str, ...]

    @classmethod
    def create(cls, config: FinalModelConfig) -> MarketModels:
        return cls(
            classifier=Pipeline(
                [
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
            ),
            abnormal_regressor=Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
            return_regressor=Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
            fitted_row_ids=(),
        )

    def fit(
        self,
        rows: tuple[MarketFeatureRow, ...],
        targets: dict[str, MarketTargetRow],
    ) -> None:
        x = [list(row.values) for row in rows]
        aligned = [targets[row.row_id] for row in rows]
        if len({item.direction for item in aligned}) != len(DIRECTIONS):
            raise ValueError("classification fit requires all frozen direction classes")
        self.classifier.fit(x, [item.direction for item in aligned])
        self.abnormal_regressor.fit(x, [item.abnormal_return for item in aligned])
        self.return_regressor.fit(x, [item.security_return for item in aligned])
        self.fitted_row_ids = tuple(row.row_id for row in rows)

    def predict(self, rows: tuple[MarketFeatureRow, ...]) -> dict[str, Any]:
        x = [list(row.values) for row in rows]
        classifier = self.classifier.named_steps["model"]
        classes = [str(item) for item in classifier.classes_]
        raw_probabilities = self.classifier.predict_proba(x)
        probabilities = [
            [float(values[classes.index(label)]) for label in DIRECTIONS]
            for values in raw_probabilities
        ]
        return {
            "directions": [str(item) for item in self.classifier.predict(x)],
            "probabilities": probabilities,
            "abnormal_returns": [float(item) for item in self.abnormal_regressor.predict(x)],
            "security_returns": [float(item) for item in self.return_regressor.predict(x)],
        }

    def coefficients(self, feature_names: tuple[str, ...]) -> dict[str, Any]:
        classifier = self.classifier.named_steps["model"]
        classes = [str(item) for item in classifier.classes_]
        logistic: dict[str, Any] = {}
        classifier_coefficients = cast("list[list[float]]", classifier.coef_.tolist())
        for label, values in zip(classes, classifier_coefficients, strict=True):
            pairs = sorted(zip(feature_names, values, strict=True), key=lambda item: item[1])
            logistic[label] = {
                "negative": [
                    {"feature": name, "coefficient": float(value)} for name, value in pairs[:10]
                ],
                "positive": [
                    {"feature": name, "coefficient": float(value)}
                    for name, value in reversed(pairs[-10:])
                ],
            }
        ridge: dict[str, Any] = {}
        for name, pipeline in (
            ("next_session_abnormal_return", self.abnormal_regressor),
            ("next_session_return", self.return_regressor),
        ):
            values = cast("list[float]", pipeline.named_steps["model"].coef_.tolist())
            pairs = sorted(
                zip(feature_names, values, strict=True),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            ridge[name] = [
                {"feature": feature, "coefficient": float(value)} for feature, value in pairs[:15]
            ]
        return {"warning": "ASSOCIATIONAL_MODEL_ONLY", "logistic": logistic, "ridge": ridge}


def evaluate_models(
    models: MarketModels,
    rows: tuple[MarketFeatureRow, ...],
    targets: dict[str, MarketTargetRow],
    fit_targets: dict[str, MarketTargetRow],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = models.predict(rows)
    aligned = [targets[row.row_id] for row in rows]
    actual_class = [item.direction for item in aligned]
    actual_abnormal = [item.abnormal_return for item in aligned]
    actual_return = [item.security_return for item in aligned]
    majority, naive_probabilities = _classification_naive(fit_targets)
    abnormal_mean = statistics.fmean(item.abnormal_return for item in fit_targets.values())
    return_mean = statistics.fmean(item.security_return for item in fit_targets.values())
    metrics = {
        "classification": {
            "model": _classification(
                actual_class, predictions["directions"], predictions["probabilities"]
            ),
            "naive_majority": _classification(
                actual_class,
                [majority] * len(rows),
                [naive_probabilities] * len(rows),
            ),
        },
        "regression": {
            "next_session_abnormal_return": _regression_group(
                actual_abnormal, predictions["abnormal_returns"], abnormal_mean
            ),
            "next_session_return": _regression_group(
                actual_return, predictions["security_returns"], return_mean
            ),
        },
    }
    records = [
        {
            "row_id": row.row_id,
            "ticker": row.ticker,
            "trade_date": row.trade_date.isoformat(),
            "actual_direction": target.direction,
            "predicted_direction": direction,
            "prob_down": probabilities[0],
            "prob_flat": probabilities[1],
            "prob_up": probabilities[2],
            "actual_next_session_abnormal_return": target.abnormal_return,
            "predicted_next_session_abnormal_return": abnormal,
            "actual_next_session_return": target.security_return,
            "predicted_next_session_return": security,
        }
        for row, target, direction, probabilities, abnormal, security in zip(
            rows,
            aligned,
            predictions["directions"],
            predictions["probabilities"],
            predictions["abnormal_returns"],
            predictions["security_returns"],
            strict=True,
        )
    ]
    return metrics, records


def diagnostic_views(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_ticker = {
        ticker: _record_metrics([row for row in records if row["ticker"] == ticker])
        for ticker in sorted({str(row["ticker"]) for row in records})
    }
    per_year = {
        year: _record_metrics([row for row in records if str(row["trade_date"])[:4] == year])
        for year in sorted({str(row["trade_date"])[:4] for row in records})
    }
    ticker_macro = _macro_metrics(per_ticker)
    return {
        "per_ticker": per_ticker,
        "ticker_macro": ticker_macro,
        "per_year": per_year,
        "date_equal_weighted": _date_equal_metrics(records),
    }


def model_quality_status(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    classification = metrics["classification"]
    primary = metrics["regression"]["next_session_abnormal_return"]
    class_model = classification["model"]
    class_naive = classification["naive_majority"]
    regression_model = primary["model"]
    naive_rmse = min(primary["naive_zero"]["rmse"], primary["naive_train_mean"]["rmse"])
    naive_mae = min(primary["naive_zero"]["mae"], primary["naive_train_mean"]["mae"])
    classification_advantage = (
        class_model["macro_f1"] >= class_naive["macro_f1"] + 0.01
        and class_model["balanced_accuracy"] >= class_naive["balanced_accuracy"] + 0.01
    )
    regression_advantage = (
        regression_model["rmse"] <= naive_rmse * 0.99
        and regression_model["mae"] <= naive_mae * 0.99
    )
    ticker_values = list(diagnostics["per_ticker"].values())
    ticker_advantage = sum(
        item["regression"]["model"]["rmse"] < item["regression"]["naive_zero"]["rmse"]
        for item in ticker_values
    ) >= math.ceil(len(ticker_values) * 0.6)
    year_values = list(diagnostics["per_year"].values())
    year_advantage = sum(
        item["regression"]["model"]["rmse"] < item["regression"]["naive_zero"]["rmse"]
        for item in year_values
    ) >= min(3, len(year_values))
    return (
        "BASELINE_SIGNAL_PRESENT"
        if classification_advantage and regression_advantage and ticker_advantage and year_advantage
        else "NO_PREDICTIVE_SIGNAL"
    )


def _classification(
    actual: list[str], predicted: list[str], probabilities: list[list[float]]
) -> dict[str, Any]:
    result = classification_metrics(actual, predicted, probabilities)
    supports = [int(result["per_class"][label]["support"]) for label in DIRECTIONS]
    result["weighted_f1"] = sum(
        float(result["per_class"][label]["f1"]) * support
        for label, support in zip(DIRECTIONS, supports, strict=True)
    ) / len(actual)
    result["actual_distribution"] = dict(sorted(Counter(actual).items()))
    result["predicted_distribution"] = dict(sorted(Counter(predicted).items()))
    return result


def _regression_group(
    actual: list[float], predicted: list[float], train_mean: float
) -> dict[str, Any]:
    return {
        "model": regression_metrics(actual, predicted),
        "naive_zero": regression_metrics(actual, [0.0] * len(actual)),
        "naive_train_mean": regression_metrics(actual, [train_mean] * len(actual)),
        "fit_target_mean": train_mean,
    }


def _classification_naive(targets: dict[str, MarketTargetRow]) -> tuple[str, list[float]]:
    counts = Counter(item.direction for item in targets.values())
    majority = sorted(DIRECTIONS, key=lambda item: (-counts[item], item))[0]
    total = sum(counts.values())
    return majority, [counts[label] / total for label in DIRECTIONS]


def _record_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    actual_class = [str(row["actual_direction"]) for row in records]
    predicted_class = [str(row["predicted_direction"]) for row in records]
    probabilities = [
        [float(row["prob_down"]), float(row["prob_flat"]), float(row["prob_up"])] for row in records
    ]
    actual = [float(row["actual_next_session_abnormal_return"]) for row in records]
    predicted = [float(row["predicted_next_session_abnormal_return"]) for row in records]
    return {
        "rows": len(records),
        "classification": _classification(actual_class, predicted_class, probabilities),
        "regression": {
            "model": regression_metrics(actual, predicted),
            "naive_zero": regression_metrics(actual, [0.0] * len(actual)),
        },
    }


def _macro_metrics(groups: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        "classification_accuracy": statistics.fmean(
            item["classification"]["accuracy"] for item in groups.values()
        ),
        "classification_macro_f1": statistics.fmean(
            item["classification"]["macro_f1"] for item in groups.values()
        ),
        "regression_mae": statistics.fmean(
            item["regression"]["model"]["mae"] for item in groups.values()
        ),
        "regression_rmse": statistics.fmean(
            item["regression"]["model"]["rmse"] for item in groups.values()
        ),
        "regression_pearson": statistics.fmean(
            item["regression"]["model"]["pearson"] or 0.0 for item in groups.values()
        ),
    }


def _date_equal_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(str(row["trade_date"]) for row in records)
    weights = [1.0 / counts[str(row["trade_date"])] for row in records]
    total = sum(weights)
    errors = [
        float(row["predicted_next_session_abnormal_return"])
        - float(row["actual_next_session_abnormal_return"])
        for row in records
    ]
    return {
        "classification_accuracy": sum(
            weight * (row["actual_direction"] == row["predicted_direction"])
            for row, weight in zip(records, weights, strict=True)
        )
        / total,
        "regression_mae": sum(
            weight * abs(error) for weight, error in zip(weights, errors, strict=True)
        )
        / total,
        "regression_rmse": math.sqrt(
            sum(weight * error**2 for weight, error in zip(weights, errors, strict=True)) / total
        ),
    }
