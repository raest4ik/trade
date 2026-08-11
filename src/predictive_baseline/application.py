from __future__ import annotations

from typing import Any

from src.predictive_baseline.domain import LoadedDataset, readiness_payload


def dataset_readiness(
    dataset: LoadedDataset,
    *,
    intraday_feature_ready: int,
) -> dict[str, Any]:
    dates = [row.publication_date for row in dataset.rows]
    return readiness_payload(
        daily_feature_ready=len(dataset.rows),
        intraday_feature_ready=intraday_feature_ready,
        ticker_count=len({row.ticker for row in dataset.rows}),
        source_count=len({row.source for row in dataset.rows}),
        date_from=None if not dates else min(dates).isoformat(),
        date_to=None if not dates else max(dates).isoformat(),
    )
