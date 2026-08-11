from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from src.daily_corpus.domain import DATASET_VERSION, FEATURE_VERSION, REACTION_VERSION
from src.predictive_baseline.domain import (
    BaselineConfig,
    LoadedDataset,
    PredictiveRow,
    SplitName,
    TemporalSplitResult,
    assert_feature_names_safe,
    sha256_payload,
)


def load_daily_predictive_dataset(
    feature_path: Path,
    reaction_path: Path,
) -> LoadedDataset:
    feature_payloads = _read_jsonl(feature_path)
    reaction_payloads = _read_jsonl(reaction_path)
    reaction_by_news = {_required_text(row, "news_id"): row for row in reaction_payloads}
    if len(reaction_by_news) != len(reaction_payloads):
        raise ValueError("daily reactions contain duplicate news_id")
    rows: list[PredictiveRow] = []
    observed_news: set[UUID] = set()
    numeric_names: set[str] = set()
    for payload in feature_payloads:
        metadata = _required_object(payload, "metadata")
        features = _required_object(payload, "features")
        labels = _required_object(payload, "labels")
        news_id = UUID(_required_text(metadata, "news_id"))
        if news_id in observed_news:
            raise ValueError("daily feature dataset contains duplicate event rows")
        observed_news.add(news_id)
        reaction = reaction_by_news.get(str(news_id))
        if reaction is None:
            raise ValueError(f"daily feature row lacks reaction window: {news_id}")
        _validate_versions(metadata, reaction)
        _validate_identity(metadata, reaction)
        assert_feature_names_safe(features)
        event_type = str(features.get("primary_event_type") or "UNKNOWN")
        numeric = _numeric_features(features)
        numeric_names.update(numeric)
        target = _required_float(labels, "abnormal_return")
        reaction_target = _required_float(reaction, "abnormal_return")
        if abs(target - reaction_target) > 1e-15:
            raise ValueError("feature label and reaction target differ")
        row = PredictiveRow(
            news_id=news_id,
            ticker=_required_text(metadata, "ticker"),
            source=_required_text(metadata, "source"),
            timestamp_quality=_required_text(metadata, "timestamp_quality"),
            event_type=event_type,
            publication_date=date.fromisoformat(_required_text(metadata, "publication_date")),
            baseline_session_date=date.fromisoformat(
                _required_text(metadata, "baseline_session_date")
            ),
            target_session_date=date.fromisoformat(_required_text(reaction, "target_session_date")),
            prediction_time=datetime.fromisoformat(
                _required_text(metadata, "feature_available_at")
            ),
            numeric_features=numeric,
            abnormal_return=target,
            dataset_version=DATASET_VERSION,
        )
        row.validate()
        rows.append(row)
    ordered = tuple(sorted(rows, key=lambda item: (item.publication_date, str(item.news_id))))
    schema = {
        "numeric": sorted(numeric_names),
        "categorical": ["ticker", "event_type", "source", "timestamp_quality"],
        "feature_version": FEATURE_VERSION,
    }
    combined_hash = hashlib.sha256(
        feature_path.read_bytes() + b"\0" + reaction_path.read_bytes()
    ).hexdigest()
    return LoadedDataset(
        rows=ordered,
        dataset_sha256=combined_hash,
        feature_schema_sha256=sha256_payload(schema),
        numeric_feature_names=tuple(sorted(numeric_names)),
        dataset_version=DATASET_VERSION,
    )


def deterministic_purged_temporal_split(
    rows: tuple[PredictiveRow, ...],
    config: BaselineConfig,
) -> TemporalSplitResult:
    config.validate()
    ordered_rows = tuple(sorted(rows, key=lambda item: (item.publication_date, str(item.news_id))))
    by_date: defaultdict[date, list[PredictiveRow]] = defaultdict(list)
    for row in ordered_rows:
        row.validate()
        by_date[row.publication_date].append(row)
    dates = sorted(by_date)
    if len(dates) < 3:
        raise ValueError("temporal split requires at least three publication dates")
    train_date_count = max(1, int(len(dates) * config.train_fraction))
    validation_date_count = max(1, int(len(dates) * config.validation_fraction))
    if train_date_count + validation_date_count >= len(dates):
        validation_date_count = 1
        train_date_count = len(dates) - 2
    train_dates = set(dates[:train_date_count])
    validation_dates = set(dates[train_date_count : train_date_count + validation_date_count])
    test_dates = set(dates[train_date_count + validation_date_count :])
    train = [row for row in ordered_rows if row.publication_date in train_dates]
    validation = [row for row in ordered_rows if row.publication_date in validation_dates]
    test = [row for row in ordered_rows if row.publication_date in test_dates]
    purged: list[UUID] = []
    embargoed: list[UUID] = []
    if config.purge_overlapping_labels:
        train, removed = _purge_left(train, validation)
        purged.extend(removed)
        validation, removed = _purge_left(validation, test)
        purged.extend(removed)
    validation, removed = _embargo_right(train, validation, config.embargo_days)
    embargoed.extend(removed)
    test, removed = _embargo_right(validation, test, config.embargo_days)
    embargoed.extend(removed)
    material = {
        "config": config.payload(),
        "assignments": [
            {"news_id": str(row.news_id), "split": split.value}
            for split, split_rows in (
                (SplitName.TRAIN, train),
                (SplitName.VALIDATION, validation),
                (SplitName.TEST, test),
            )
            for row in split_rows
        ],
        "purged": sorted(str(item) for item in purged),
        "embargoed": sorted(str(item) for item in embargoed),
    }
    result = TemporalSplitResult(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
        purged_news_ids=tuple(sorted(set(purged), key=str)),
        embargoed_news_ids=tuple(sorted(set(embargoed), key=str)),
        split_sha256=sha256_payload(material),
    )
    result.validate()
    return result


def _purge_left(
    left: list[PredictiveRow], right: list[PredictiveRow]
) -> tuple[list[PredictiveRow], list[UUID]]:
    if not left or not right:
        return left, []
    boundary = min(row.publication_date for row in right)
    kept = [row for row in left if row.target_session_date < boundary]
    removed = [row.news_id for row in left if row.target_session_date >= boundary]
    return kept, removed


def _embargo_right(
    left: list[PredictiveRow],
    right: list[PredictiveRow],
    embargo_days: int,
) -> tuple[list[PredictiveRow], list[UUID]]:
    if not left or not right or embargo_days == 0:
        return right, []
    cutoff = max(row.target_session_date for row in left) + timedelta(days=embargo_days)
    kept = [row for row in right if row.publication_date > cutoff]
    removed = [row.news_id for row in right if row.publication_date <= cutoff]
    return kept, removed


def _validate_versions(metadata: dict[str, Any], reaction: dict[str, Any]) -> None:
    if _required_text(metadata, "feature_version") != FEATURE_VERSION:
        raise ValueError("unexpected daily feature version")
    if _required_text(metadata, "reaction_version") != REACTION_VERSION:
        raise ValueError("unexpected feature reaction version")
    if _required_text(reaction, "reaction_version") != REACTION_VERSION:
        raise ValueError("unexpected daily reaction version")


def _validate_identity(metadata: dict[str, Any], reaction: dict[str, Any]) -> None:
    for name in ("ticker", "source", "timestamp_quality", "publication_date"):
        if _required_text(metadata, name) != _required_text(reaction, name):
            raise ValueError(f"feature/reaction identity mismatch: {name}")


def _numeric_features(features: dict[str, Any]) -> dict[str, float | None]:
    numeric: dict[str, float | None] = {}
    for name, value in features.items():
        if name == "primary_event_type":
            continue
        if value is None:
            numeric[name] = None
            continue
        if isinstance(value, bool):
            numeric[name] = float(value)
            continue
        if isinstance(value, (int, float, str)):
            try:
                numeric[name] = float(value)
            except ValueError as exc:
                raise ValueError(f"non-numeric daily feature: {name}") from exc
            continue
        raise ValueError(f"unsupported daily feature value: {name}")
    return numeric


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = cast("object", json.loads(line))
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append({str(key): item for key, item in cast("dict[object, Any]", value).items()})
    return rows


def _required_object(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"required object is missing: {name}")
    return {str(key): item for key, item in cast("dict[object, Any]", value).items()}


def _required_text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"required text is missing: {name}")
    return value


def _required_float(payload: dict[str, Any], name: str) -> float:
    value = payload.get(name)
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError(f"required numeric value is missing: {name}")
    return float(value)
