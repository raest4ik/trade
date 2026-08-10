from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from src.events.domain.enums import EventType
from src.ml_features.domain.entities import (
    FEATURE_HORIZONS_MINUTES,
    LABEL_HORIZONS_MINUTES,
    FeatureDatasetBuildResult,
    FeatureDatasetConfig,
    FeatureDatasetRow,
    FeatureExclusion,
)

JSONL_FILENAME = "ml-feature-dataset-v1.jsonl"
CSV_FILENAME = "ml-feature-dataset-v1.csv"
MANIFEST_FILENAME = "manifest.json"
STATS_FILENAME = "stats.json"
EXCLUSIONS_FILENAME = "exclusions.jsonl"

METADATA_COLUMNS = (
    "news_id",
    "instrument_id",
    "ticker",
    "published_at",
    "timestamp_quality",
    "source",
    "source_item_id",
    "dataset_version",
    "feature_version",
    "event_analysis_version",
    "fact_extractor_version",
    "reaction_version",
    "market_context_version",
    "benchmark_code",
    "generated_at",
    "ai_analysis_available",
    "ai_analyzer_version",
    "ai_model",
)
_FACT_NAMES = (
    "revenue",
    "net_profit",
    "ebitda",
    "operating_profit",
    "fcf",
    "capex",
    "net_debt",
    "dividend_per_share",
    "total_dividend",
    "production",
    "contract_value",
    "ownership_pct",
)
_BASE_FEATURE_COLUMNS = (
    "primary_event_type",
    "event_count",
    "has_financial_results",
    "has_dividend",
    "has_guidance",
    "has_ma",
    "has_production_update",
    "has_sanctions",
    "has_regulatory_action",
    "has_other",
)
_CHANGE_FEATURE_COLUMNS = (
    "net_profit_change_pct",
    "net_profit_change_direction",
    "net_profit_change_unit",
    "net_profit_change_comparison_type",
    "revenue_change_pct",
    "revenue_change_direction",
    "revenue_change_unit",
    "revenue_change_comparison_type",
    "ebitda_change_pct",
    "ebitda_change_direction",
    "ebitda_change_unit",
    "ebitda_change_comparison_type",
    "dividend_change_pct",
    "dividend_change_direction",
    "dividend_change_unit",
    "dividend_change_comparison_type",
    "production_change_pct",
    "production_change_direction",
    "production_change_unit",
    "production_change_comparison_type",
)
_OTHER_FEATURE_COLUMNS = (
    "guidance_fact_count",
    "guidance_up_count",
    "guidance_down_count",
    "guidance_unchanged_count",
    "dividend_per_share",
    "dividend_role",
    "title_length",
    "content_length",
    "word_count",
    "number_count",
    "percentage_count",
    "currency_mention_count",
    "publication_hour_local",
    "publication_minute_local",
    "day_of_week",
    "is_weekend",
)
QUALITY_COLUMNS = (
    "missing_features",
    "market_data_complete",
    "benchmark_data_complete",
    "security_observation_end_at",
    "benchmark_observation_end_at",
    "point_in_time_cutoff",
    "classification_policy",
    "classification_threshold",
)


def feature_columns() -> tuple[str, ...]:
    event_flags = tuple(f"event_type_{item.value.lower()}" for item in EventType)
    fact_columns: list[str] = ["fact_count"]
    for name in _FACT_NAMES:
        fact_columns.extend(
            [
                f"has_{name}",
                f"{name}_value",
                f"{name}_unit",
                f"{name}_currency",
                f"{name}_scale",
                f"{name}_role",
            ]
        )
    market_columns: list[str] = []
    for horizon in FEATURE_HORIZONS_MINUTES:
        market_columns.extend(
            [
                f"pre_return_{horizon}m",
                f"pre_log_return_{horizon}m",
                f"imoex_pre_return_{horizon}m",
                f"imoex_pre_log_return_{horizon}m",
                f"pre_abnormal_return_{horizon}m",
            ]
        )
    market_columns.extend(
        [
            "realized_volatility_15m",
            "realized_volatility_30m",
            "realized_volatility_60m",
            "volume_last_1m",
            "volume_sum_5m",
            "volume_sum_15m",
            "volume_sum_60m",
            "volume_ratio_5m_vs_60m",
        ]
    )
    return (
        *_BASE_FEATURE_COLUMNS,
        *event_flags,
        *fact_columns,
        *_CHANGE_FEATURE_COLUMNS,
        *_OTHER_FEATURE_COLUMNS,
        *market_columns,
    )


def label_columns() -> tuple[str, ...]:
    names = (
        "available",
        "security_simple_return",
        "benchmark_simple_return",
        "abnormal_simple_return",
        "security_log_return",
        "benchmark_log_return",
        "abnormal_log_return",
        "classification",
    )
    return tuple(f"{horizon}m_{name}" for horizon in LABEL_HORIZONS_MINUTES for name in names)


def csv_columns() -> tuple[str, ...]:
    return (
        *(f"metadata.{name}" for name in METADATA_COLUMNS),
        *(f"features.{name}" for name in feature_columns()),
        *(f"labels.{name}" for name in label_columns()),
        *(f"quality.{name}" for name in QUALITY_COLUMNS),
    )


def write_dataset_artifacts(
    output_dir: Path,
    *,
    result: FeatureDatasetBuildResult,
    config: FeatureDatasetConfig,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "jsonl": output_dir / JSONL_FILENAME,
        "csv": output_dir / CSV_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
        "stats": output_dir / STATS_FILENAME,
        "exclusions": output_dir / EXCLUSIONS_FILENAME,
    }
    write_jsonl(paths["jsonl"], result.rows)
    write_csv(paths["csv"], result.rows)
    write_exclusions(paths["exclusions"], result.exclusions)
    stats = dataset_stats(result.rows, result.exclusions)
    _write_json(paths["stats"], stats)
    _write_json(paths["manifest"], build_manifest(result=result, config=config, stats=stats))
    return paths


def write_jsonl(path: Path, rows: Iterable[FeatureDatasetRow]) -> int:
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in materialized:
            output.write(json.dumps(_json_value(row.payload()), ensure_ascii=False, sort_keys=True))
            output.write("\n")
    return len(materialized)


def write_csv(path: Path, rows: Iterable[FeatureDatasetRow]) -> int:
    materialized = list(rows)
    columns = csv_columns()
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in materialized:
            writer.writerow(_flat_row(row))
    return len(materialized)


def write_exclusions(path: Path, exclusions: Iterable[FeatureExclusion]) -> int:
    materialized = list(exclusions)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for item in materialized:
            output.write(
                json.dumps(
                    {
                        "news_id": str(item.news_id),
                        "instrument_id": (
                            None if item.instrument_id is None else str(item.instrument_id)
                        ),
                        "reason": item.reason.value,
                        "detail": item.detail,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return len(materialized)


def build_manifest(
    *,
    result: FeatureDatasetBuildResult,
    config: FeatureDatasetConfig,
    stats: dict[str, Any],
) -> dict[str, Any]:
    normalized = config.normalized()
    return {
        "dataset_version": normalized.dataset_version,
        "feature_version": normalized.feature_version,
        "event_analysis_version": normalized.event_analysis_version,
        "fact_extractor_version": normalized.fact_extractor_version,
        "reaction_version": normalized.reaction_version,
        "market_context_version": normalized.market_context_version,
        "benchmark_code": normalized.benchmark_code,
        "config_hash": normalized.hash(),
        "git_sha": result.run.git_sha,
        "generated_at": (None if not result.rows else result.rows[0].metadata["generated_at"]),
        "date_from": normalized.date_from,
        "date_to": normalized.date_to,
        "tickers": list(normalized.tickers),
        "limit": normalized.limit,
        "require_label_horizon": normalized.require_label_horizon,
        "classification_policy": (
            None
            if normalized.classification_threshold is None
            else "RESEARCH_DEFAULT_NOT_CALIBRATED"
        ),
        "classification_threshold": normalized.classification_threshold,
        "row_count": len(result.rows),
        "exclusion_count": len(result.exclusions),
        "exclusions_by_reason": stats["exclusions_by_reason"],
        "jsonl_schema": {
            "metadata": list(METADATA_COLUMNS),
            "features": list(feature_columns()),
            "labels": list(label_columns()),
            "quality": list(QUALITY_COLUMNS),
        },
        "csv_columns": list(csv_columns()),
        "point_in_time_rule": "all market observation end_at values are <= published_at",
        "fact_selection_policy": (
            "ACTUAL > FORECAST > TARGET > PREVIOUS > CONSENSUS > CHANGE > UNKNOWN; "
            "then latest explicit period, confidence, source position, UUID"
        ),
        "training_readiness": (
            "MODEL_TRAINING_NOT_READY_INSUFFICIENT_REAL_ROWS"
            if len(result.rows) <= 10
            else "REQUIRES_SEPARATE_REVIEW"
        ),
    }


def dataset_stats(
    rows: Iterable[FeatureDatasetRow],
    exclusions: Iterable[FeatureExclusion] = (),
) -> dict[str, Any]:
    materialized = list(rows)
    exclusion_items = list(exclusions)
    by_ticker = Counter(str(row.metadata["ticker"]) for row in materialized)
    by_event = Counter(str(row.features["primary_event_type"]) for row in materialized)
    by_year: Counter[str] = Counter()
    by_month: Counter[str] = Counter()
    for row in materialized:
        published_at = cast("datetime", row.metadata["published_at"])
        by_year[str(published_at.year)] += 1
        by_month[published_at.strftime("%Y-%m")] += 1
    feature_missingness = {
        name: sum(row.features.get(name) is None for row in materialized)
        for name in feature_columns()
    }
    market_names = [
        name
        for name in feature_columns()
        if name.startswith(("pre_", "imoex_", "realized_volatility_", "volume_"))
    ]
    label_availability: dict[str, int] = {}
    distributions: dict[str, dict[str, Any]] = {}
    for horizon in LABEL_HORIZONS_MINUTES:
        key = f"{horizon}m"
        values: list[Decimal] = []
        for row in materialized:
            label_value = row.labels.get(key)
            if not isinstance(label_value, dict):
                continue
            label = cast("dict[str, Any]", label_value)
            value = label.get("abnormal_simple_return")
            if isinstance(value, Decimal):
                values.append(value)
        label_availability[key] = len(values)
        distributions[key] = _distribution(values)
    return {
        "rows_total": len(materialized),
        "by_ticker": dict(sorted(by_ticker.items())),
        "by_event_type": dict(sorted(by_event.items())),
        "by_year": dict(sorted(by_year.items())),
        "by_month": dict(sorted(by_month.items())),
        "label_availability_by_horizon": label_availability,
        "feature_missingness": feature_missingness,
        "market_data_missingness": {name: feature_missingness[name] for name in market_names},
        "abnormal_simple_return_distributions": distributions,
        "exclusions_by_reason": dict(
            sorted(Counter(item.reason.value for item in exclusion_items).items())
        ),
        "sample_interpretation": (
            "INSUFFICIENT_SAMPLE_FOR_INFERENCE"
            if len(materialized) <= 10
            else "DESCRIPTIVE_STATISTICS_ONLY"
        ),
    }


def load_jsonl_rows(path: Path) -> list[FeatureDatasetRow]:
    rows: list[FeatureDatasetRow] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload_value = cast("object", json.loads(line, parse_float=Decimal))
        if not isinstance(payload_value, dict):
            raise ValueError(f"invalid ml-feature-dataset-v1 row at line {line_number}")
        payload = cast("dict[str, object]", payload_value)
        if set(payload) != {
            "metadata",
            "features",
            "labels",
            "quality",
        }:
            raise ValueError(f"invalid ml-feature-dataset-v1 row at line {line_number}")
        metadata_value = payload["metadata"]
        features_value = payload["features"]
        labels_value = payload["labels"]
        quality_value = payload["quality"]
        if not all(
            isinstance(value, dict)
            for value in (metadata_value, features_value, labels_value, quality_value)
        ):
            raise ValueError(f"invalid ml-feature-dataset-v1 row at line {line_number}")
        metadata = cast("dict[str, Any]", metadata_value)
        features = cast("dict[str, Any]", features_value)
        labels = cast("dict[str, Any]", labels_value)
        quality = cast("dict[str, Any]", quality_value)
        _restore_row_types(metadata, labels, quality)
        rows.append(
            FeatureDatasetRow(
                metadata=metadata,
                features=features,
                labels=labels,
                quality=quality,
            )
        )
    return rows


def _flat_row(row: FeatureDatasetRow) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for name in METADATA_COLUMNS:
        flat[f"metadata.{name}"] = _csv_value(row.metadata.get(name))
    for name in feature_columns():
        flat[f"features.{name}"] = _csv_value(row.features.get(name))
    for horizon in LABEL_HORIZONS_MINUTES:
        values = row.labels.get(f"{horizon}m", {})
        label = cast("dict[str, Any]", values) if isinstance(values, dict) else {}
        for column in label_columns():
            prefix = f"{horizon}m_"
            if column.startswith(prefix):
                flat[f"labels.{column}"] = _csv_value(label.get(column[len(prefix) :]))
    for name in QUALITY_COLUMNS:
        flat[f"quality.{name}"] = _csv_value(row.quality.get(name))
    return flat


def _distribution(values: list[Decimal]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }
    ordered = sorted(values)
    count = len(ordered)
    mean = sum(ordered, Decimal("0")) / Decimal(count)
    middle = count // 2
    median = (
        ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / Decimal("2")
    )
    with localcontext() as context:
        context.prec = 28
        variance = sum(((value - mean) ** 2 for value in ordered), Decimal("0")) / Decimal(count)
        std = variance.sqrt()
    return {
        "count": count,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean,
        "median": median,
        "std": std,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_value(value: object) -> object:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        items = cast("dict[object, object]", value)
        return {str(key): _json_value(item) for key, item in items.items()}
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return [_json_value(item) for item in items]
    return value


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Decimal, UUID)):
        return str(value)
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(_json_value(cast("object", value)), ensure_ascii=False, sort_keys=True)
    return value


def _restore_row_types(
    metadata: dict[str, Any],
    labels: dict[str, Any],
    quality: dict[str, Any],
) -> None:
    for name in ("published_at", "generated_at"):
        value = metadata.get(name)
        if isinstance(value, str):
            metadata[name] = datetime.fromisoformat(value)
    for name in (
        "security_observation_end_at",
        "benchmark_observation_end_at",
        "point_in_time_cutoff",
    ):
        value = quality.get(name)
        if isinstance(value, str):
            quality[name] = datetime.fromisoformat(value)
    return_names = {
        "security_simple_return",
        "benchmark_simple_return",
        "abnormal_simple_return",
        "security_log_return",
        "benchmark_log_return",
        "abnormal_log_return",
    }
    for label_value in labels.values():
        if not isinstance(label_value, dict):
            continue
        label = cast("dict[str, Any]", label_value)
        for name in return_names:
            value = label.get(name)
            if isinstance(value, str):
                label[name] = Decimal(value)
