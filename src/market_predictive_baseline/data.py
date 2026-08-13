from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any, cast

from src.market_predictive_baseline.domain import (
    FrozenMarketDataset,
    MarketFeatureRow,
    MarketTargetRow,
    sha256_payload,
    validate_frozen_metadata,
)
from src.tinvest_market.domain import FEATURE_DATASET_VERSION, SplitConfig, dataset_semantics


def load_frozen_market_dataset(root: Path) -> FrozenMarketDataset:
    manifest = _json(root / "dataset-manifest.json")
    split = _json(root / "split-manifest.json")
    feature_payloads = _jsonl(root / "features.jsonl")
    target_payloads = _jsonl(root / "targets.jsonl")
    benchmark_available = bool(manifest["dataset_semantics"]["abnormal_return_available"])
    recomputed_dataset_sha = sha256_payload(
        {
            "dataset_version": FEATURE_DATASET_VERSION,
            "semantics": dataset_semantics(benchmark_available),
            "features": feature_payloads,
            "targets": target_payloads,
        }
    )
    if recomputed_dataset_sha != manifest.get("dataset_sha"):
        raise ValueError("feature dataset SHA verification failed")
    names = tuple(str(name) for name in _feature_names(feature_payloads))
    recomputed_schema_sha = sha256_payload(list(names))
    if recomputed_schema_sha != manifest.get("feature_schema_sha"):
        raise ValueError("feature schema SHA verification failed")
    recomputed_split_sha = sha256_payload(
        {
            "config": {
                "train_fraction": SplitConfig().train_fraction,
                "validation_fraction": SplitConfig().validation_fraction,
                "test_fraction": SplitConfig().test_fraction,
                "purge_sessions": SplitConfig().purge_sessions,
                "embargo_sessions": SplitConfig().embargo_sessions,
            },
            "assignments": [
                (str(item["row_id"]), str(item["split"])) for item in split["assignments"]
            ],
            "purged": sorted(str(item) for item in split["purged_row_ids"]),
            "embargoed": sorted(str(item) for item in split["embargoed_row_ids"]),
        }
    )
    if recomputed_split_sha != split.get("split_sha"):
        raise ValueError("temporal split SHA verification failed")
    features = tuple(_feature(payload, names) for payload in feature_payloads)
    assignments = {str(item["row_id"]): str(item["split"]) for item in split["assignments"]}
    if len(assignments) != len(split["assignments"]):
        raise ValueError("duplicate split assignment")
    target_ids = [str(item["row_id"]) for item in target_payloads]
    feature_ids = [item.row_id for item in features]
    if feature_ids != target_ids:
        raise ValueError("features and targets are not identically aligned")
    excluded = set(str(item) for item in split["purged_row_ids"]) | set(
        str(item) for item in split["embargoed_row_ids"]
    )
    if set(feature_ids) != set(assignments) | excluded:
        raise ValueError("split does not cover the frozen dataset")
    dataset = FrozenMarketDataset(
        features=features,
        assignments=assignments,
        date_ranges=cast("dict[str, dict[str, str]]", split["date_ranges"]),
        counts={key: int(value) for key, value in cast("dict[str, Any]", split["counts"]).items()},
        dataset_sha=str(manifest["dataset_sha"]),
        split_sha=str(split["split_sha"]),
        feature_schema_sha=str(manifest["feature_schema_sha"]),
        feature_names=names,
        dataset_version=str(manifest["dataset_version"]),
        source_usage_readiness=str(manifest["source_policy"]["source_usage_readiness"]),
        price_adjustment_status=str(manifest["price_adjustment_status"]),
    )
    validate_frozen_metadata(dataset)
    _assert_required_extremes_retained(set(feature_ids))
    return dataset


def load_targets_for_splits(
    path: Path,
    assignments: dict[str, str],
    allowed_splits: frozenset[str],
) -> dict[str, MarketTargetRow]:
    result: dict[str, MarketTargetRow] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            payload = cast("dict[str, Any]", json.loads(line))
            row_id = str(payload["row_id"])
            if assignments.get(row_id) not in allowed_splits:
                continue
            abnormal = payload.get("next_session_abnormal_return")
            if abnormal is None:
                raise ValueError("primary abnormal-return target is missing")
            target = MarketTargetRow(
                row_id=row_id,
                ticker=str(payload["ticker"]),
                trade_date=_date(str(payload["trade_date"])),
                direction=str(payload["direction"]),
                abnormal_return=float(abnormal),
                security_return=float(payload["next_session_return"]),
            )
            if target.direction not in {"DOWN", "FLAT", "UP"}:
                raise ValueError("unknown target direction")
            result[row_id] = target
    expected = {row_id for row_id, split in assignments.items() if split in allowed_splits}
    if set(result) != expected:
        raise ValueError("target rows do not cover requested splits")
    return result


def _feature(payload: dict[str, Any], names: tuple[str, ...]) -> MarketFeatureRow:
    values = cast("dict[str, Any]", payload["features"])
    if set(values) != set(names):
        raise ValueError("feature row schema changed")
    numeric = tuple(float(values[name]) for name in names)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("non-finite feature value")
    row = MarketFeatureRow(
        row_id=str(payload["row_id"]),
        ticker=str(payload["ticker"]),
        trade_date=_date(str(payload["trade_date"])),
        feature_as_of=_date(str(payload["feature_as_of"])),
        values=numeric,
    )
    row.validate(len(names))
    return row


def _feature_names(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    if not rows:
        raise ValueError("empty market feature dataset")
    observed = set(cast("dict[str, Any]", rows[0]["features"]))
    from src.tinvest_market.domain import feature_names

    expected = feature_names(True)
    if observed != set(expected):
        raise ValueError("unexpected market feature schema")
    return expected


def _assert_required_extremes_retained(row_ids: set[str]) -> None:
    required = {
        "ROSN:2008-09-19",
        "SBER:2008-09-19",
        "SBERP:2008-09-19",
        "VTBR:2022-02-24",
    }
    if not required <= row_ids:
        raise ValueError("required extreme-return observations are missing")


def _json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _date(value: str) -> date:
    return date.fromisoformat(value)
