from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from src.tinvest_market.domain import (
    FEATURE_DATASET_VERSION,
    SOURCE,
    DatasetResult,
    TemporalSplit,
    dataset_semantics,
    readiness,
)
from src.tinvest_market.policy import PRICE_ADJUSTMENT_STATUS, execution_safety, source_policy


def write_feature_artifacts(
    output_dir: Path,
    *,
    result: DatasetResult,
    split: TemporalSplit,
    acquisition_manifest: dict[str, Any],
    git_sha: str,
    event_daily_feature_ready: int,
    moex_targets_path: Path | None = None,
) -> dict[str, Path]:
    result.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = [item.trade_date for item in result.features]
    distribution = Counter(item.ticker for item in result.features)
    benchmark_available = bool(result.benchmark_rows)
    overlap = compare_moex_targets(result, moex_targets_path)
    coverage = {
        "raw_rows": result.raw_rows,
        "benchmark_rows": result.benchmark_rows,
        "feature_ready": len(result.features),
        "target_ready": len(result.targets),
        "ticker_count": len(distribution),
        "ticker_distribution": dict(sorted(distribution.items())),
        "date_range": {"from": min(dates).isoformat(), "to": max(dates).isoformat()}
        if dates
        else {"from": None, "to": None},
        "years": sorted({item.year for item in dates}),
        "event_pipeline": {"daily_feature_ready": event_daily_feature_ready, "kept_separate": True},
    }
    manifest = {
        "dataset_version": FEATURE_DATASET_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "source": SOURCE,
        "source_policy": source_policy(),
        "raw_dataset_sha": acquisition_manifest["dataset_sha"],
        "instrument_mapping_sha": acquisition_manifest["instrument_mapping_sha"],
        "dataset_sha": result.dataset_sha,
        "feature_schema_sha": result.feature_schema_sha,
        "split_sha": split.split_sha,
        "dataset_semantics": dataset_semantics(benchmark_available),
        "feature_cutoff_audit": {
            "cause": result.quality["feature_cutoff_cause"],
            "was_bug": result.quality["feature_cutoff_was_bug"],
            "alignment_policy": "COMMON_REAL_SESSIONS_NO_FORWARD_FILL",
        },
        "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
        "targets_stored_separately": True,
        "model_trained": False,
        "backtest_executed": False,
        "paper_trading_executed": False,
        "buy_sell_generated": False,
        "paid_services": False,
    }
    readiness_payload = {
        **readiness(len(result.features), len(distribution)),
        "execution_safety": execution_safety(),
        "event_daily_feature_ready": event_daily_feature_ready,
        "event_and_market_counts_combined": False,
    }
    split_payload = {
        "strategy": "DATE_GROUPED_CHRONOLOGICAL_PURGED_EMBARGOED",
        "counts": split.counts(),
        "purged_rows": len(split.purged_row_ids),
        "embargoed_rows": len(split.embargoed_row_ids),
        "purged_dates": list(split.purged_dates),
        "embargoed_dates": list(split.embargoed_dates),
        "date_ranges": split.date_ranges,
        "split_sha": split.split_sha,
        "assignments": [
            {"row_id": key, "split": value.value}
            for key, value in sorted(split.assignments.items())
        ],
        "purged_row_ids": list(split.purged_row_ids),
        "embargoed_row_ids": list(split.embargoed_row_ids),
        "same_trade_date_crosses_splits": False,
        "random_split_used": False,
    }
    paths = {
        "features": output_dir / "features.jsonl",
        "targets": output_dir / "targets.jsonl",
        "coverage": output_dir / "coverage.json",
        "manifest": output_dir / "dataset-manifest.json",
        "readiness": output_dir / "readiness.json",
        "split": output_dir / "split-manifest.json",
        "quality": output_dir / "quality-report.json",
        "price_audit": output_dir / "price-integrity-audit.json",
        "moex_overlap": output_dir / "moex-overlap-diagnostic.json",
    }
    _jsonl(paths["features"], [item.payload() for item in result.features])
    _jsonl(paths["targets"], [item.payload() for item in result.targets])
    for key, payload in (
        ("coverage", coverage),
        ("manifest", manifest),
        ("readiness", readiness_payload),
        ("split", split_payload),
        ("quality", result.quality),
        ("price_audit", result.price_audit),
        ("moex_overlap", overlap),
    ):
        _json(paths[key], payload)
    return paths


def compare_moex_targets(result: DatasetResult, path: Path | None) -> dict[str, object]:
    base: dict[str, object] = {
        "diagnostic_only": True,
        "data_sources_combined": False,
        "t_invest_filled_from_moex": False,
        "t_invest_rows": len(result.targets),
        "moex_rows": 0,
        "overlap_rows": 0,
        "missing_in_tinvest_rows": 0,
        "missing_in_moex_rows": len(result.targets),
        "mean_absolute_return_difference": None,
        "max_absolute_return_difference": None,
        "absolute_difference_gt_1pct": 0,
        "absolute_difference_gt_5pct": 0,
        "absolute_difference_gt_10pct": 0,
    }
    if path is None or not path.exists():
        return {**base, "status": "MOEX_DIAGNOSTIC_UNAVAILABLE"}
    moex: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = cast("dict[str, object]", json.loads(line))
        moex[str(row["row_id"])] = float(str(row["next_session_return"]))
    t_invest = {item.row_id: item.next_session_return for item in result.targets}
    differences = [
        abs(value - moex[row_id]) for row_id, value in t_invest.items() if row_id in moex
    ]
    return {
        **base,
        "status": "OVERLAP_DIAGNOSTIC_COMPLETE",
        "moex_rows": len(moex),
        "overlap_rows": len(differences),
        "missing_in_tinvest_rows": len(set(moex) - set(t_invest)),
        "missing_in_moex_rows": len(set(t_invest) - set(moex)),
        "mean_absolute_return_difference": sum(differences) / len(differences)
        if differences
        else None,
        "max_absolute_return_difference": max(differences) if differences else None,
        "absolute_difference_gt_1pct": sum(item > 0.01 for item in differences),
        "absolute_difference_gt_5pct": sum(item > 0.05 for item in differences),
        "absolute_difference_gt_10pct": sum(item > 0.10 for item in differences),
    }


def load_status(output_dir: Path, *, raw_dir: Path) -> dict[str, object]:
    names = (
        "coverage.json",
        "readiness.json",
        "dataset-manifest.json",
        "split-manifest.json",
        "price-integrity-audit.json",
    )
    payload = {name: json.loads((output_dir / name).read_text(encoding="utf-8")) for name in names}
    raw_manifest = json.loads((raw_dir / "dataset-manifest.json").read_text(encoding="utf-8"))
    coverage = payload["coverage.json"]
    ready = payload["readiness.json"]
    split = payload["split-manifest.json"]
    return {
        "READONLY_AUTH": "AUTH_OK_FOR_DATASET_BUILD",
        "SANDBOX_AUTH": "CONNECTIVITY_ONLY_SEPARATE_CONTOUR",
        "source": SOURCE,
        "source_policy": source_policy(),
        "real_trading_allowed": False,
        "raw_rows": raw_manifest["row_count"],
        "feature_ready": coverage["feature_ready"],
        "ticker_count": coverage["ticker_count"],
        "ticker_distribution": coverage["ticker_distribution"],
        "date_range": coverage["date_range"],
        "years": coverage["years"],
        "split_rows": split["counts"],
        "purged": split["purged_rows"],
        "embargoed": split["embargoed_rows"],
        "IMOEX_status": "AVAILABLE"
        if raw_manifest["imoex_resolved"]
        else "UNAVAILABLE_NO_MOEX_FALLBACK",
        "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
        "extreme_returns": payload["price-integrity-audit.json"],
        "dataset_sha": payload["dataset-manifest.json"]["dataset_sha"],
        "warnings": ready["warnings"],
    }


def _json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
