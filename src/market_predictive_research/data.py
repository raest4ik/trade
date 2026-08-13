from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, cast

from src.market_predictive_research.domain import (
    CROSS_SECTIONAL_BASES,
    DERIVED_FEATURE_NAMES,
    DEVELOPMENT_FROM,
    DEVELOPMENT_TO,
    FROZEN_DATASET_SHA,
    FROZEN_FEATURE_SCHEMA_SHA,
    FROZEN_SPLIT_SHA,
    FUTURE_HOLDOUT_START,
    OBSERVED_TEST_START,
    TARGET_HORIZONS,
    DevelopmentDataset,
    DevelopmentFeatureRow,
    HorizonTarget,
    OneSessionTarget,
    direction_for_return,
    sha256_payload,
)
from src.tinvest_market.domain import FEATURE_DATASET_VERSION, feature_names
from src.tinvest_market.policy import PRICE_ADJUSTMENT_STATUS, SOURCE_USAGE_READINESS

_TRADE_DATE_PATTERN = re.compile(r'"trade_date"\s*:\s*"(\d{4}-\d{2}-\d{2})"')


def load_development_dataset(
    root: Path,
    *,
    requested_from: date = DEVELOPMENT_FROM,
    requested_to: date = DEVELOPMENT_TO,
) -> DevelopmentDataset:
    if requested_from < DEVELOPMENT_FROM or requested_to >= OBSERVED_TEST_START:
        raise ValueError("OBSERVED_TEST_READ_ATTEMPT")
    if requested_from > requested_to:
        raise ValueError("invalid development date range")
    manifest = cast(
        "dict[str, Any]",
        json.loads((root / "dataset-manifest.json").read_text(encoding="utf-8")),
    )
    _validate_frozen_manifest(manifest)
    source_rows = _load_guarded_features(
        root / "features.jsonl", requested_from=requested_from, requested_to=requested_to
    )
    for row in source_rows:
        row.validate()
    enhanced, names = enhance_features(source_rows)
    one_session = _load_guarded_one_session_targets(
        root / "targets.jsonl", requested_from=requested_from, requested_to=requested_to
    )
    targets = build_horizon_targets(one_session)
    schema_sha = sha256_payload(list(names))
    return DevelopmentDataset(
        rows=enhanced,
        targets=targets,
        feature_names=names,
        feature_schema_sha=schema_sha,
        dataset_sha=FROZEN_DATASET_SHA,
        split_sha=FROZEN_SPLIT_SHA,
        price_adjustment_status=PRICE_ADJUSTMENT_STATUS,
        source_usage_readiness=SOURCE_USAGE_READINESS,
    )


def enhance_features(
    rows: tuple[DevelopmentFeatureRow, ...],
) -> tuple[tuple[DevelopmentFeatureRow, ...], tuple[str, ...]]:
    if not rows:
        raise ValueError("development dataset is empty")
    original_names = tuple(rows[0].values)
    cross_names = tuple(
        item
        for base in CROSS_SECTIONAL_BASES
        for item in (f"cross_sectional_rank_{base}", f"cross_sectional_zscore_{base}")
    )
    names = (*original_names, *DERIVED_FEATURE_NAMES, *cross_names)
    by_date: defaultdict[date, list[DevelopmentFeatureRow]] = defaultdict(list)
    for row in rows:
        if tuple(row.values) != original_names:
            raise ValueError("source feature schema is not stable")
        by_date[row.trade_date].append(row)
    result: list[DevelopmentFeatureRow] = []
    for trade_date in sorted(by_date):
        dated = sorted(by_date[trade_date], key=lambda item: item.ticker)
        cross_values = {
            base: _cross_section([item.values[base] for item in dated])
            for base in CROSS_SECTIONAL_BASES
        }
        for index, row in enumerate(dated):
            values = {
                **row.values,
                "momentum_acceleration_5_20": row.values["return_5d"]
                - row.values["return_20d"] / 4.0,
                "volatility_term_5_20": row.values["volatility_5d"] - row.values["volatility_20d"],
                "volume_trend_5_20": _ratio(
                    row.values["volume_mean_5d"], row.values["volume_mean_20d"]
                )
                - 1.0,
                "month_end_flag": float(_is_month_end(trade_date)),
            }
            for base in CROSS_SECTIONAL_BASES:
                rank, zscore = cross_values[base][index]
                values[f"cross_sectional_rank_{base}"] = rank
                values[f"cross_sectional_zscore_{base}"] = zscore
            if tuple(values) != names or not all(math.isfinite(value) for value in values.values()):
                raise ValueError("invalid v2 feature row")
            result.append(
                DevelopmentFeatureRow(
                    row_id=row.row_id,
                    ticker=row.ticker,
                    trade_date=row.trade_date,
                    feature_as_of=row.feature_as_of,
                    values=values,
                )
            )
    return tuple(sorted(result, key=lambda item: (item.trade_date, item.ticker))), names


def build_horizon_targets(
    rows: tuple[OneSessionTarget, ...],
) -> dict[int, dict[str, HorizonTarget]]:
    by_ticker: defaultdict[str, list[OneSessionTarget]] = defaultdict(list)
    for row in rows:
        by_ticker[row.ticker].append(row)
    output: dict[int, dict[str, HorizonTarget]] = {horizon: {} for horizon in TARGET_HORIZONS}
    for ticker in sorted(by_ticker):
        ordered = sorted(by_ticker[ticker], key=lambda item: item.trade_date)
        for index, row in enumerate(ordered):
            for horizon in TARGET_HORIZONS:
                window = ordered[index : index + horizon]
                if len(window) != horizon:
                    continue
                security = _compound([item.security_return for item in window])
                benchmark = _compound([item.benchmark_return for item in window])
                output[horizon][row.row_id] = HorizonTarget(
                    row_id=row.row_id,
                    ticker=row.ticker,
                    trade_date=row.trade_date,
                    horizon=horizon,
                    security_return=security,
                    benchmark_return=benchmark,
                    abnormal_return=security - benchmark,
                    direction=direction_for_return(security, horizon),
                )
    return output


def future_holdout_coverage(feature_path: Path) -> dict[str, object]:
    sessions: set[date] = set()
    tickers: set[str] = set()
    rows = 0
    with feature_path.open(encoding="utf-8") as source:
        for line in source:
            trade_date = _metadata_date(line)
            if trade_date < FUTURE_HOLDOUT_START:
                continue
            payload = cast("dict[str, Any]", json.loads(line))
            sessions.add(trade_date)
            tickers.add(str(payload["ticker"]))
            rows += 1
    return {
        "future_holdout_start": FUTURE_HOLDOUT_START.isoformat(),
        "status": "ACCUMULATING",
        "observed": False,
        "session_count": len(sessions),
        "date_range": {
            "from": min(sessions).isoformat() if sessions else None,
            "to": max(sessions).isoformat() if sessions else None,
        },
        "ticker_count": len(tickers),
        "tickers": sorted(tickers),
        "row_count": rows,
        "outcomes_loaded": False,
        "performance_metrics_computed": False,
    }


def _load_guarded_one_session_targets(
    path: Path, *, requested_from: date, requested_to: date
) -> tuple[OneSessionTarget, ...]:
    rows: list[OneSessionTarget] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            trade_date = _metadata_date(line)
            if trade_date < requested_from or trade_date > requested_to:
                continue
            payload = cast("dict[str, Any]", json.loads(line))
            benchmark = payload.get("imoex_next_session_return")
            if benchmark is None:
                raise ValueError("development target requires T-Invest IMOEX")
            rows.append(
                OneSessionTarget(
                    row_id=str(payload["row_id"]),
                    ticker=str(payload["ticker"]),
                    trade_date=trade_date,
                    security_return=float(payload["next_session_return"]),
                    benchmark_return=float(benchmark),
                )
            )
    return tuple(rows)


def _load_guarded_features(
    path: Path, *, requested_from: date, requested_to: date
) -> tuple[DevelopmentFeatureRow, ...]:
    expected_names = feature_names(True)
    rows: list[DevelopmentFeatureRow] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            trade_date = _metadata_date(line)
            if trade_date < requested_from or trade_date > requested_to:
                continue
            payload = cast("dict[str, Any]", json.loads(line))
            values = cast("dict[str, Any]", payload["features"])
            if set(values) != set(expected_names):
                raise ValueError("frozen feature schema changed")
            rows.append(
                DevelopmentFeatureRow(
                    row_id=str(payload["row_id"]),
                    ticker=str(payload["ticker"]),
                    trade_date=trade_date,
                    feature_as_of=date.fromisoformat(str(payload["feature_as_of"])),
                    values={name: float(values[name]) for name in expected_names},
                )
            )
    return tuple(rows)


def _validate_frozen_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "dataset_version": FEATURE_DATASET_VERSION,
        "dataset_sha": FROZEN_DATASET_SHA,
        "split_sha": FROZEN_SPLIT_SHA,
        "feature_schema_sha": FROZEN_FEATURE_SCHEMA_SHA,
        "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"frozen dataset manifest mismatch: {field}")


def _metadata_date(line: str) -> date:
    match = _TRADE_DATE_PATTERN.search(line)
    if match is None:
        raise ValueError("market row lacks parseable trade_date metadata")
    return date.fromisoformat(match.group(1))


def _cross_section(values: list[float]) -> list[tuple[float, float]]:
    if not values:
        return []
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        rank = ((cursor + 1 + end) / 2.0 - 1.0) / max(1, len(values) - 1)
        for index, _ in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    mean = statistics.fmean(values)
    scale = statistics.pstdev(values) if len(values) > 1 else 0.0
    return [
        (rank, 0.0 if scale == 0 else (value - mean) / scale)
        for rank, value in zip(ranks, values, strict=True)
    ]


def _compound(values: list[float]) -> float:
    result = 1.0
    for value in values:
        result *= 1.0 + value
    return result - 1.0


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _is_month_end(value: date) -> bool:
    from datetime import timedelta

    return (value + timedelta(days=1)).month != value.month
