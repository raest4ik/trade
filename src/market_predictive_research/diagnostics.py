from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from src.market_predictive_research.domain import (
    DevelopmentDataset,
    DevelopmentFeatureRow,
    FoldManifest,
    HorizonTarget,
)
from src.predictive_baseline.modeling import regression_metrics


def dataset_diagnostics(dataset: DevelopmentDataset) -> dict[str, Any]:
    rows = dataset.rows
    primary_targets = dataset.targets[1]
    target_rows = list(primary_targets.values())
    by_year: defaultdict[str, list[HorizonTarget]] = defaultdict(list)
    by_ticker: defaultdict[str, list[HorizonTarget]] = defaultdict(list)
    by_date: defaultdict[str, list[HorizonTarget]] = defaultdict(list)
    for target in target_rows:
        by_year[str(target.trade_date.year)].append(target)
        by_ticker[target.ticker].append(target)
        by_date[target.trade_date.isoformat()].append(target)
    missingness = {
        name: sum(row.values.get(name) is None for row in rows) for name in dataset.feature_names
    }
    feature_stability = {
        name: _feature_stability([row.values[name] for row in rows])
        for name in dataset.feature_names
    }
    univariate = {
        name: _association(
            [row.values[name] for row in rows if row.row_id in primary_targets],
            [
                primary_targets[row.row_id].abnormal_return
                for row in rows
                if row.row_id in primary_targets
            ],
        )
        for name in dataset.feature_names
    }
    associations_by_ticker = {
        ticker: _feature_associations(
            [row for row in rows if row.ticker == ticker], primary_targets, dataset.feature_names
        )
        for ticker in sorted(by_ticker)
    }
    associations_by_year = {
        year: _feature_associations(
            [row for row in rows if str(row.trade_date.year) == year],
            primary_targets,
            dataset.feature_names,
        )
        for year in sorted(by_year)
    }
    return {
        "row_count": len(rows),
        "target_distribution": {
            str(horizon): _target_summary(list(targets.values()))
            for horizon, targets in dataset.targets.items()
        },
        "class_distribution_by_year": {
            year: dict(sorted(Counter(item.direction for item in items).items()))
            for year, items in sorted(by_year.items())
        },
        "target_by_year": {
            year: _values([item.abnormal_return for item in items])
            for year, items in sorted(by_year.items())
        },
        "target_by_ticker": {
            ticker: _values([item.abnormal_return for item in items])
            for ticker, items in sorted(by_ticker.items())
        },
        "target_autocorrelation": _autocorrelation(by_ticker),
        "cross_sectional_dispersion_by_date": _values(
            [
                statistics.pstdev(item.abnormal_return for item in items)
                for items in by_date.values()
                if len(items) > 1
            ]
        ),
        "feature_missingness": missingness,
        "feature_stability": feature_stability,
        "univariate_feature_target_associations": univariate,
        "feature_associations_by_ticker": associations_by_ticker,
        "feature_associations_by_year": associations_by_year,
        "causal_claim": False,
    }


def fold_feature_associations(
    dataset: DevelopmentDataset, manifest: FoldManifest
) -> dict[str, Any]:
    by_id = {row.row_id: row for row in dataset.rows}
    return {
        fold.fold_id: _feature_associations(
            [by_id[row_id] for row_id in fold.validation_row_ids],
            dataset.targets[fold.horizon],
            dataset.feature_names,
        )
        for fold in manifest.folds
    }


def _target_summary(targets: list[HorizonTarget]) -> dict[str, Any]:
    return {
        "rows": len(targets),
        "abnormal_return": _values([item.abnormal_return for item in targets]),
        "security_return": _values([item.security_return for item in targets]),
        "classes": dict(sorted(Counter(item.direction for item in targets).items())),
    }


def _values(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else 0.0,
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def _feature_stability(values: list[float]) -> dict[str, float]:
    midpoint = len(values) // 2
    left, right = values[:midpoint], values[midpoint:]
    return {
        "overall_mean": statistics.fmean(values),
        "overall_std": statistics.pstdev(values),
        "first_half_mean": statistics.fmean(left),
        "second_half_mean": statistics.fmean(right),
        "standardized_mean_shift": (
            (statistics.fmean(right) - statistics.fmean(left))
            / max(statistics.pstdev(values), 1e-12)
        ),
    }


def _association(features: list[float], targets: list[float]) -> dict[str, float | None]:
    metrics = regression_metrics(targets, features)
    return {"pearson": metrics["pearson"], "spearman": metrics["spearman"]}


def _feature_associations(
    rows: list[DevelopmentFeatureRow],
    targets: dict[str, HorizonTarget],
    feature_names: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    typed_rows = [row for row in rows if row.row_id in targets]
    return {
        name: _association(
            [float(row.values[name]) for row in typed_rows],
            [targets[row.row_id].abnormal_return for row in typed_rows],
        )
        for name in feature_names
    }


def _autocorrelation(by_ticker: dict[str, list[HorizonTarget]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for ticker, targets in sorted(by_ticker.items()):
        ordered = sorted(targets, key=lambda item: item.trade_date)
        values = [item.abnormal_return for item in ordered]
        metrics = regression_metrics(values[1:], values[:-1])
        result[ticker] = float(metrics["pearson"] or 0.0)
    return result
