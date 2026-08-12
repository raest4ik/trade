from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.market_baseline.domain import (
    CLASSIFICATION_POLICY_VERSION,
    DATASET_VERSION,
    FEATURE_NAMES,
    FLAT_RETURN_THRESHOLD,
    PRICE_ADJUSTMENT_STATUS,
    SOURCE_NAME,
    SOURCE_POLICY,
    DatasetBuildResult,
    TemporalSplit,
    readiness_for_rows,
)


def write_market_baseline_artifacts(
    output_dir: Path,
    *,
    result: DatasetBuildResult,
    split: TemporalSplit,
    acquisition: dict[str, Any],
    git_sha: str,
    event_daily_feature_ready: int,
    created_at: datetime | None = None,
) -> dict[str, Path]:
    result.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = (created_at or datetime.now(UTC)).isoformat()
    ticker_distribution = Counter(item.ticker for item in result.features)
    year_distribution = Counter(str(item.trade_date.year) for item in result.features)
    dates = [item.trade_date for item in result.features]
    split_counts = split.counts()
    coverage = {
        "dataset_version": DATASET_VERSION,
        "source_rows": result.source_row_count,
        "benchmark_rows": result.benchmark_row_count,
        "feature_ready": len(result.features),
        "target_ready": len(result.targets),
        "ticker_count": len(ticker_distribution),
        "ticker_distribution": dict(sorted(ticker_distribution.items())),
        "source_ticker_distribution": result.source_ticker_distribution,
        "source_date_ranges": result.source_date_ranges,
        "date_range": {
            "from": min(dates).isoformat() if dates else None,
            "to": max(dates).isoformat() if dates else None,
        },
        "years": sorted({item.trade_date.year for item in result.features}),
        "year_distribution": dict(sorted(year_distribution.items())),
        "acquisition": acquisition,
        "event_pipeline": {
            "daily_feature_ready": event_daily_feature_ready,
            "kept_separate_from_market_rows": True,
        },
    }
    readiness = {
        "dataset_version": DATASET_VERSION,
        **readiness_for_rows(len(result.features), len(ticker_distribution)),
        "event_daily_feature_ready": event_daily_feature_ready,
        "event_and_market_counts_combined": False,
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "created_at": generated_at,
        "git_sha": git_sha,
        "source": SOURCE_NAME,
        "source_policy": SOURCE_POLICY,
        "tickers": sorted(ticker_distribution),
        "date_from": min(dates).isoformat() if dates else None,
        "date_to": max(dates).isoformat() if dates else None,
        "row_count": len(result.features),
        "ticker_distribution": dict(sorted(ticker_distribution.items())),
        "year_distribution": dict(sorted(year_distribution.items())),
        "feature_schema_sha": result.feature_schema_sha256,
        "dataset_sha": result.dataset_sha256,
        "split_sha": split.split_sha256,
        "feature_names": list(FEATURE_NAMES),
        "targets_stored_separately": True,
        "classification_policy_version": CLASSIFICATION_POLICY_VERSION,
        "flat_return_threshold": FLAT_RETURN_THRESHOLD,
        "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
        "model_trained": False,
        "paid_services": False,
        "event_features_included": False,
    }
    split_manifest = {
        "dataset_version": DATASET_VERSION,
        "strategy": "DATE_GROUPED_CHRONOLOGICAL_PURGED_EMBARGOED",
        "uses_random_split": False,
        "same_trade_date_kept_together": True,
        "counts": split_counts,
        "purged_rows": len(split.purged_row_ids),
        "embargoed_rows": len(split.embargoed_row_ids),
        "date_ranges": split.date_ranges,
        "split_sha": split.split_sha256,
        "assignments": [
            {"row_id": row_id, "split": value.value}
            for row_id, value in sorted(split.assignments.items())
        ],
        "purged_row_ids": list(split.purged_row_ids),
        "embargoed_row_ids": list(split.embargoed_row_ids),
    }
    quality = {
        "dataset_version": DATASET_VERSION,
        **result.quality,
        "features_and_targets_have_identical_keys": True,
        "features_and_targets_physically_separate": True,
        "same_trade_date_crosses_splits": False,
        "random_split_used": False,
        "market_only_features": True,
        "news_or_event_features": False,
    }
    paths = {
        "features": output_dir / "features.jsonl",
        "targets": output_dir / "targets.jsonl",
        "coverage": output_dir / "coverage.json",
        "readiness": output_dir / "readiness.json",
        "dataset_manifest": output_dir / "dataset-manifest.json",
        "split_manifest": output_dir / "split-manifest.json",
        "quality_report": output_dir / "quality-report.json",
    }
    _write_jsonl(paths["features"], [item.payload() for item in result.features])
    _write_jsonl(paths["targets"], [item.payload() for item in result.targets])
    _write_json(paths["coverage"], coverage)
    _write_json(paths["readiness"], readiness)
    _write_json(paths["dataset_manifest"], manifest)
    _write_json(paths["split_manifest"], split_manifest)
    _write_json(paths["quality_report"], quality)
    return paths


def load_market_baseline_status(output_dir: Path) -> dict[str, Any]:
    required = {
        "coverage": output_dir / "coverage.json",
        "readiness": output_dir / "readiness.json",
        "manifest": output_dir / "dataset-manifest.json",
        "split": output_dir / "split-manifest.json",
        "quality": output_dir / "quality-report.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing market baseline artifacts: " + ", ".join(missing))
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in required.items()
    }
    coverage = payloads["coverage"]
    readiness = payloads["readiness"]
    split = payloads["split"]
    manifest = payloads["manifest"]
    return {
        "rows": coverage["source_rows"],
        "feature_ready": coverage["feature_ready"],
        "ticker_count": coverage["ticker_count"],
        "date_range": coverage["date_range"],
        "years": coverage["years"],
        "split_sizes": split["counts"],
        "purged_rows": split["purged_rows"],
        "embargoed_rows": split["embargoed_rows"],
        "readiness": readiness["status"],
        "warnings": readiness["warnings"],
        "price_adjustment_status": manifest["price_adjustment_status"],
        "event_daily_feature_ready": readiness["event_daily_feature_ready"],
        "model_trained": readiness["model_trained"],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
