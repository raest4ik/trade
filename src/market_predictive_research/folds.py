from __future__ import annotations

from collections import defaultdict
from datetime import date

from src.market_predictive_research.domain import (
    EMBARGO_SESSIONS,
    FOLD_COUNT,
    DevelopmentDataset,
    DevelopmentFeatureRow,
    FoldManifest,
    RollingFold,
    sha256_payload,
)


def build_rolling_folds(dataset: DevelopmentDataset) -> FoldManifest:
    folds: list[RollingFold] = []
    for horizon in sorted(dataset.targets):
        rows = dataset.aligned_rows(horizon)
        by_date: defaultdict[date, list[DevelopmentFeatureRow]] = defaultdict(list)
        for row in rows:
            by_date[row.trade_date].append(row)
        dates = sorted(by_date)
        minimum_train = len(dates) // 2
        remaining = len(dates) - minimum_train
        boundaries = [
            minimum_train + remaining * index // FOLD_COUNT for index in range(FOLD_COUNT + 1)
        ]
        for index in range(FOLD_COUNT):
            validation_dates = dates[boundaries[index] : boundaries[index + 1]]
            candidate_train_dates = dates[: boundaries[index]]
            purged_dates = candidate_train_dates[-horizon:]
            embargoed_dates = validation_dates[:EMBARGO_SESSIONS]
            train_dates = set(candidate_train_dates[:-horizon])
            usable_validation_dates = set(validation_dates[EMBARGO_SESSIONS:])
            train_rows = tuple(row.row_id for row in rows if row.trade_date in train_dates)
            validation_rows = tuple(
                row.row_id for row in rows if row.trade_date in usable_validation_dates
            )
            if not train_rows or not validation_rows:
                raise ValueError("rolling fold produced an empty partition")
            fold = RollingFold(
                fold_id=f"h{horizon}-fold-{index + 1}",
                horizon=horizon,
                train_row_ids=train_rows,
                validation_row_ids=validation_rows,
                purged_dates=tuple(item.isoformat() for item in purged_dates),
                embargoed_dates=tuple(item.isoformat() for item in embargoed_dates),
                train_range={
                    "from": min(train_dates).isoformat(),
                    "to": max(train_dates).isoformat(),
                },
                validation_range={
                    "from": min(usable_validation_dates).isoformat(),
                    "to": max(usable_validation_dates).isoformat(),
                },
            )
            _validate_fold(fold, dataset)
            folds.append(fold)
    payload = [fold.payload() for fold in folds]
    return FoldManifest(folds=tuple(folds), fold_manifest_sha=sha256_payload(payload))


def _validate_fold(fold: RollingFold, dataset: DevelopmentDataset) -> None:
    by_id = {row.row_id: row for row in dataset.rows}
    train_dates = {by_id[row_id].trade_date for row_id in fold.train_row_ids}
    validation_dates = {by_id[row_id].trade_date for row_id in fold.validation_row_ids}
    if not max(train_dates) < min(validation_dates):
        raise ValueError("rolling fold is not chronological")
    if train_dates & validation_dates:
        raise ValueError("rolling fold date leakage")
    all_rows_by_date: defaultdict[date, set[str]] = defaultdict(set)
    for row in dataset.aligned_rows(fold.horizon):
        all_rows_by_date[row.trade_date].add(row.row_id)
    selected_train = set(fold.train_row_ids)
    selected_validation = set(fold.validation_row_ids)
    for trade_date in train_dates:
        if all_rows_by_date[trade_date] - selected_train:
            raise ValueError("one trade date crosses TRAIN boundary")
    for trade_date in validation_dates:
        if all_rows_by_date[trade_date] - selected_validation:
            raise ValueError("one trade date crosses VALIDATION boundary")
