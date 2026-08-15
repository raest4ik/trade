from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from src.event_predictive_baseline.domain import FEATURE_FAMILIES, MIN_GROUP_METRIC_ROWS
from src.event_predictive_baseline.modeling import metrics_from_records


def grouped_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_ticker = _group(records, lambda row: str(row["ticker"]))
    per_year = _group(records, lambda row: str(row["publication_date"])[:4])
    per_source = _group(records, lambda row: str(row["source_family"]))
    per_event_type = _group(records, lambda row: str(row["primary_event_type"]))
    return {
        "ROW_WEIGHTED": metrics_from_records(records),
        "ISSUER_MACRO": issuer_macro(per_ticker),
        "per_ticker": per_ticker,
        "per_ticker_metric_policy": "metrics are diagnostic; correlations require N>=10",
        "per_year": per_year,
        "per_source_family": per_source,
        "per_event_type": per_event_type,
        "concentration": concentration_diagnostics(records),
    }


def issuer_macro(per_ticker: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not per_ticker:
        return {}
    result: dict[str, Any] = {}
    for family in FEATURE_FAMILIES:
        classification = {
            metric: statistics.fmean(
                float(group["classification"][family][metric]) for group in per_ticker.values()
            )
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss")
        }
        regression: dict[str, Any] = {}
        for target in ("abnormal_return", "security_return"):
            regression[target] = {
                metric: _mean_defined(
                    [group["regression"][target][family][metric] for group in per_ticker.values()]
                )
                for metric in ("mae", "rmse", "r2", "pearson", "spearman")
            }
        result[family] = {"classification": classification, "regression": regression}
    return result


def comparison_deltas(metrics: dict[str, Any], left: str, right: str) -> dict[str, Any]:
    classification = metrics["classification"]["models"]
    regression = metrics["regression"]["abnormal_return"]["models"]
    return {
        "comparison": f"{left}_MINUS_{right}",
        "classification": {
            metric: _delta(classification[left].get(metric), classification[right].get(metric))
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
                "log_loss",
                "brier_score",
            )
        },
        "regression": {
            metric: _delta(regression[left].get(metric), regression[right].get(metric))
            for metric in ("mae", "rmse", "r2", "pearson", "spearman")
        },
    }


def incremental_value_status(test_metrics: dict[str, Any], test_diagnostics: dict[str, Any]) -> str:
    models_class = test_metrics["classification"]["models"]
    models_reg = test_metrics["regression"]["abnormal_return"]["models"]
    a_class, c_class = models_class["A_MARKET_ONLY"], models_class["C_EVENT_PLUS_MARKET"]
    a_reg, c_reg = models_reg["A_MARKET_ONLY"], models_reg["C_EVENT_PLUS_MARKET"]
    class_wins = sum(
        (
            c_class["balanced_accuracy"] > a_class["balanced_accuracy"],
            c_class["macro_f1"] > a_class["macro_f1"],
            c_class["log_loss"] < a_class["log_loss"],
        )
    )
    reg_wins = sum(
        (
            c_reg["mae"] < a_reg["mae"],
            c_reg["rmse"] < a_reg["rmse"],
            _greater(c_reg["r2"], a_reg["r2"]),
            _greater(c_reg["pearson"], a_reg["pearson"]),
            _greater(c_reg["spearman"], a_reg["spearman"]),
        )
    )
    macro = test_diagnostics["ISSUER_MACRO"]
    macro_support = (
        macro["C_EVENT_PLUS_MARKET"]["classification"]["macro_f1"]
        > macro["A_MARKET_ONLY"]["classification"]["macro_f1"]
        and macro["C_EVENT_PLUS_MARKET"]["regression"]["abnormal_return"]["rmse"]
        < macro["A_MARKET_ONLY"]["regression"]["abnormal_return"]["rmse"]
    )
    non_mgnt = [
        item
        for ticker, item in test_diagnostics["per_ticker"].items()
        if ticker != "MGNT" and item["rows"] >= MIN_GROUP_METRIC_ROWS
    ]
    non_mgnt_support = bool(non_mgnt) and sum(
        item["regression"]["abnormal_return"]["C_EVENT_PLUS_MARKET"]["rmse"]
        < item["regression"]["abnormal_return"]["A_MARKET_ONLY"]["rmse"]
        for item in non_mgnt
    ) >= math.ceil(len(non_mgnt) / 2)
    return (
        "EXACT_EVENT_INCREMENTAL_SIGNAL_CANDIDATE"
        if class_wins >= 2 and reg_wins >= 3 and macro_support and non_mgnt_support
        else "NO_EXACT_EVENT_INCREMENTAL_SIGNAL"
    )


def timestamp_hypothesis_status(incremental_status: str) -> str:
    return (
        "TIMESTAMP_HYPOTHESIS_SUPPORTED_AS_CANDIDATE"
        if incremental_status == "EXACT_EVENT_INCREMENTAL_SIGNAL_CANDIDATE"
        else "TIMESTAMP_HYPOTHESIS_NOT_SUPPORTED"
    )


def concentration_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["ticker"]) for row in records)
    total = sum(counts.values())
    ordered = sorted(counts.values(), reverse=True)
    hhi = sum((count / total) ** 2 for count in counts.values()) if total else 0.0
    return {
        "rows": total,
        "counts": dict(sorted(counts.items())),
        "shares": {ticker: count / total for ticker, count in sorted(counts.items())}
        if total
        else {},
        "top1_share": ordered[0] / total if total else 0.0,
        "top3_share": sum(ordered[:3]) / total if total else 0.0,
        "hhi": hhi,
        "effective_issuer_count": 1 / hhi if hhi > 0 else 0.0,
    }


def _group(
    records: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[key(row)].append(row)
    return {name: metrics_from_records(rows) for name, rows in sorted(grouped.items())}


def _mean_defined(values: list[object]) -> float | None:
    defined = [_as_float(value) for value in values if value is not None]
    return statistics.fmean(defined) if defined else None


def _delta(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return _as_float(left) - _as_float(right)


def _greater(left: object, right: object) -> bool:
    return left is not None and right is not None and _as_float(left) > _as_float(right)


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError("metric value must be numeric")
    return float(value)
