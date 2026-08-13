# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.market_predictive_research.domain import (
    DIRECTIONS,
    DevelopmentDataset,
    RollingFold,
)
from src.predictive_baseline.modeling import classification_metrics, regression_metrics

MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "Ridge": {"alpha": 1.0},
    "LogisticRegression": {
        "C": 1.0,
        "max_iter": 2000,
        "solver": "lbfgs",
        "random_state": 0,
    },
    "HistGradientBoostingRegressor": {
        "learning_rate": 0.05,
        "max_iter": 60,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "random_state": 0,
    },
    "HistGradientBoostingClassifier": {
        "learning_rate": 0.05,
        "max_iter": 60,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "random_state": 0,
    },
}


def evaluate_fold(dataset: DevelopmentDataset, fold: RollingFold) -> dict[str, Any]:
    by_id = {row.row_id: row for row in dataset.rows}
    targets = dataset.targets[fold.horizon]
    train_rows = [by_id[row_id] for row_id in fold.train_row_ids]
    validation_rows = [by_id[row_id] for row_id in fold.validation_row_ids]
    train_targets = [targets[row.row_id] for row in train_rows]
    validation_targets = [targets[row.row_id] for row in validation_rows]
    x_train = [[row.values[name] for name in dataset.feature_names] for row in train_rows]
    x_validation = [[row.values[name] for name in dataset.feature_names] for row in validation_rows]
    regression_results: dict[str, Any] = {}
    predictions: dict[str, list[float]] = {}
    for target_name in ("abnormal_return", "security_return"):
        y_train = [float(getattr(item, target_name)) for item in train_targets]
        y_validation = [float(getattr(item, target_name)) for item in validation_targets]
        mean = statistics.fmean(y_train)
        target_results: dict[str, Any] = {
            "naive_zero": regression_metrics(y_validation, [0.0] * len(y_validation)),
            "naive_train_mean": regression_metrics(y_validation, [mean] * len(y_validation)),
        }
        for model_name in ("Ridge", "HistGradientBoostingRegressor"):
            model = _regressor(model_name)
            model.fit(x_train, y_train)
            predicted = [float(item) for item in model.predict(x_validation)]
            target_results[model_name] = regression_metrics(y_validation, predicted)
            if target_name == "abnormal_return":
                predictions[model_name] = predicted
        regression_results[target_name] = target_results
    classification_results: dict[str, Any] = {}
    y_train_class = [item.direction for item in train_targets]
    y_validation_class = [item.direction for item in validation_targets]
    counts = Counter(y_train_class)
    majority = sorted(DIRECTIONS, key=lambda item: (-counts[item], item))[0]
    class_probabilities = [counts[label] / len(y_train_class) for label in DIRECTIONS]
    classification_results["naive_majority"] = _classification(
        y_validation_class,
        [majority] * len(y_validation_class),
        [class_probabilities] * len(y_validation_class),
    )
    for model_name in ("LogisticRegression", "HistGradientBoostingClassifier"):
        model = _classifier(model_name)
        model.fit(x_train, y_train_class)
        predicted = [str(item) for item in model.predict(x_validation)]
        estimator = model.named_steps["model"]
        classes = [str(item) for item in estimator.classes_]
        probabilities = [
            [float(values[classes.index(label)]) for label in DIRECTIONS]
            for values in model.predict_proba(x_validation)
        ]
        classification_results[model_name] = _classification(
            y_validation_class, predicted, probabilities
        )
    validation_records = [
        {
            "row_id": row.row_id,
            "ticker": row.ticker,
            "trade_date": row.trade_date.isoformat(),
            "actual_abnormal_return": target.abnormal_return,
            "ridge_prediction": predictions["Ridge"][index],
            "hist_prediction": predictions["HistGradientBoostingRegressor"][index],
        }
        for index, (row, target) in enumerate(zip(validation_rows, validation_targets, strict=True))
    ]
    return {
        "fold_id": fold.fold_id,
        "horizon": fold.horizon,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "regression": regression_results,
        "classification": classification_results,
        "rank_ic": {
            model: daily_rank_ic(validation_records, f"{key}_prediction")
            for model, key in (
                ("Ridge", "ridge"),
                ("HistGradientBoostingRegressor", "hist"),
            )
        },
        "validation_predictions": validation_records,
    }


def aggregate_research_results(folds: list[dict[str, Any]]) -> dict[str, Any]:
    regression: dict[str, Any] = {}
    classification: dict[str, Any] = {}
    for horizon in sorted({int(item["horizon"]) for item in folds}):
        selected = [item for item in folds if item["horizon"] == horizon]
        regression[str(horizon)] = {
            target: {
                model: _mean_metrics([item["regression"][target][model] for item in selected])
                for model in (
                    "naive_zero",
                    "naive_train_mean",
                    "Ridge",
                    "HistGradientBoostingRegressor",
                )
            }
            for target in ("abnormal_return", "security_return")
        }
        classification[str(horizon)] = {
            model: _mean_metrics([item["classification"][model] for item in selected])
            for model in (
                "naive_majority",
                "LogisticRegression",
                "HistGradientBoostingClassifier",
            )
        }
    candidates = [
        (
            float(regression[str(horizon)]["abnormal_return"][model]["rmse"]),
            horizon,
            model,
        )
        for horizon in sorted(int(item) for item in regression)
        for model in ("Ridge", "HistGradientBoostingRegressor")
    ]
    _, best_horizon, best_model = min(candidates)
    best_folds = [item for item in folds if item["horizon"] == best_horizon]
    better_folds = sum(
        item["regression"]["abnormal_return"][best_model]["rmse"]
        < min(
            item["regression"]["abnormal_return"]["naive_zero"]["rmse"],
            item["regression"]["abnormal_return"]["naive_train_mean"]["rmse"],
        )
        for item in best_folds
    )
    classification_better_folds = sum(
        item["classification"]["HistGradientBoostingClassifier"]["balanced_accuracy"]
        > item["classification"]["naive_majority"]["balanced_accuracy"]
        and item["classification"]["HistGradientBoostingClassifier"]["macro_f1"]
        > item["classification"]["naive_majority"]["macro_f1"]
        for item in best_folds
    )
    rank_ics = [item["rank_ic"][best_model]["mean_daily_rank_ic"] for item in best_folds]
    development_status = (
        "DEV_SIGNAL_CANDIDATE"
        if better_folds >= 4
        and classification_better_folds >= 4
        and sum(value > 0 for value in rank_ics) >= 4
        and statistics.fmean(rank_ics) >= 0.02
        else "NO_DEV_SIGNAL"
    )
    return {
        "regression": regression,
        "classification": classification,
        "best_development_candidate": {
            "model": best_model,
            "horizon": best_horizon,
            "target": "next_session_abnormal_return",
            "folds_beating_naive_rmse": better_folds,
            "classification_folds_beating_naive": classification_better_folds,
        },
        "DEVELOPMENT_STATUS": development_status,
        "CONFIRMED_SIGNAL": False,
    }


def stability_views(folds: list[dict[str, Any]], *, model: str, horizon: int) -> dict[str, Any]:
    records = [
        record
        for fold in folds
        if fold["horizon"] == horizon
        for record in fold["validation_predictions"]
    ]
    key = "ridge_prediction" if model == "Ridge" else "hist_prediction"
    return {
        "per_ticker": {
            ticker: _stability_metrics(
                [record for record in records if record["ticker"] == ticker], key
            )
            for ticker in sorted({str(record["ticker"]) for record in records})
        },
        "per_year": {
            year: _stability_metrics(
                [record for record in records if str(record["trade_date"])[:4] == year], key
            )
            for year in sorted({str(record["trade_date"])[:4] for record in records})
        },
        "rank_ic": daily_rank_ic(records, key),
    }


def daily_rank_ic(records: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    by_date: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_date[str(record["trade_date"])].append(record)
    values = [
        _spearman(
            [float(row[prediction_key]) for row in dated],
            [float(row["actual_abnormal_return"]) for row in dated],
        )
        for dated in by_date.values()
        if len(dated) >= 3
    ]
    return {
        "date_count": len(values),
        "mean_daily_rank_ic": statistics.fmean(values) if values else 0.0,
        "median_daily_rank_ic": statistics.median(values) if values else 0.0,
        "daily_rank_ic_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "fraction_positive_daily_rank_ic": (
            sum(value > 0 for value in values) / len(values) if values else 0.0
        ),
    }


def _regressor(name: str) -> Pipeline:
    estimator: Any
    if name == "Ridge":
        estimator = Ridge(alpha=1.0)
    else:
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=60,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=0,
        )
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def _classifier(name: str) -> Pipeline:
    estimator: Any
    if name == "LogisticRegression":
        estimator = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=0)
    else:
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=60,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=0,
        )
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


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


def _mean_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "brier_score",
        "mae",
        "rmse",
        "r2",
        "pearson",
        "spearman",
    )
    return {
        name: statistics.fmean(float(item[name] or 0.0) for item in metrics)
        for name in names
        if name in metrics[0]
    }


def _stability_metrics(records: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    actual = [float(record["actual_abnormal_return"]) for record in records]
    predicted = [float(record[prediction_key]) for record in records]
    return regression_metrics(actual, predicted)


def _spearman(left: list[float], right: list[float]) -> float:
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks, strict=True)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_ranks)
        * sum((value - right_mean) ** 2 for value in right_ranks)
    )
    return 0.0 if denominator == 0 else numerator / denominator


def _ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index, _ in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks
