from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest

from src.events.domain.v3 import rules_v3_fingerprint
from src.holdout_evaluation.domain import EXPECTED_RULES_FINGERPRINT
from src.market_baseline.domain import (
    DATASET_VERSION,
    FEATURE_NAMES,
    PRICE_ADJUSTMENT_STATUS,
    SOURCE_POLICY,
    DailyBar,
    Readiness,
    SplitConfig,
    SplitName,
    build_dataset,
    date_grouped_temporal_split,
    readiness_for_rows,
)
from src.market_baseline.reporting import (
    load_market_baseline_status,
    write_market_baseline_artifacts,
)
from src.market_data.infrastructure.moex_client import MoexIssClient
from src.shared.config.settings import DEFAULT_OLLAMA_MODEL


def test_dataset_build_is_deterministic_and_has_separate_targets() -> None:
    securities, benchmark = _history()
    first = build_dataset(securities, benchmark)
    second = build_dataset(
        {ticker: tuple(reversed(rows)) for ticker, rows in reversed(securities.items())},
        tuple(reversed(benchmark)),
    )
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.feature_schema_sha256 == second.feature_schema_sha256
    assert [item.row_id for item in first.features] == [item.row_id for item in first.targets]
    assert not set(FEATURE_NAMES) & {
        "next_session_return",
        "next_session_abnormal_return",
        "direction",
    }


def test_duplicate_ticker_date_is_rejected() -> None:
    securities, benchmark = _history()
    duplicate = securities["AAA"] + (securities["AAA"][0],)
    with pytest.raises(ValueError, match="duplicate AAA"):
        build_dataset({"AAA": duplicate}, benchmark)


def test_target_day_is_not_used_by_features_and_rolling_window_ends_at_t_minus_one() -> None:
    securities, benchmark = _history()
    baseline = build_dataset(securities, benchmark)
    selected = baseline.features[5]
    rows = list(securities[selected.ticker])
    target_index = next(
        index for index, item in enumerate(rows) if item.trade_date == selected.trade_date
    )
    rows[target_index] = replace(
        rows[target_index],
        open=rows[target_index].open * 2,
        close=rows[target_index].close * 2,
        high=rows[target_index].high * 2,
        low=rows[target_index].low * 2,
        volume=rows[target_index].volume * 50,
        value=rows[target_index].value * 100,
    )
    changed = build_dataset({selected.ticker: tuple(rows)}, benchmark)
    changed_feature = next(item for item in changed.features if item.row_id == selected.row_id)
    changed_target = next(item for item in changed.targets if item.row_id == selected.row_id)
    original_target = next(item for item in baseline.targets if item.row_id == selected.row_id)
    assert changed_feature.values == selected.values
    assert changed_feature.feature_as_of < changed_feature.trade_date
    assert changed_target.next_session_return != original_target.next_session_return


def test_temporal_split_is_date_grouped_ordered_purged_and_embargoed() -> None:
    securities, benchmark = _history(tickers=("AAA", "BBB"), sessions=120)
    dataset = build_dataset(securities, benchmark)
    split = date_grouped_temporal_split(
        tuple(reversed(dataset.features)), SplitConfig(purge_sessions=2, embargo_sessions=2)
    )
    by_date: dict[date, set[SplitName]] = {}
    rows_by_id = {item.row_id: item for item in dataset.features}
    for row_id, name in split.assignments.items():
        by_date.setdefault(rows_by_id[row_id].trade_date, set()).add(name)
    assert all(len(names) == 1 for names in by_date.values())
    train_dates = [
        rows_by_id[row_id].trade_date
        for row_id, name in split.assignments.items()
        if name == SplitName.TRAIN
    ]
    validation_dates = [
        rows_by_id[row_id].trade_date
        for row_id, name in split.assignments.items()
        if name == SplitName.VALIDATION
    ]
    test_dates = [
        rows_by_id[row_id].trade_date
        for row_id, name in split.assignments.items()
        if name == SplitName.TEST
    ]
    assert max(train_dates) < min(validation_dates) < max(validation_dates) < min(test_dates)
    assert split.purged_row_ids
    assert split.embargoed_row_ids
    assert (
        split.split_sha256
        == date_grouped_temporal_split(
            dataset.features, SplitConfig(purge_sessions=2, embargo_sessions=2)
        ).split_sha256
    )


def test_abnormal_return_uses_same_imoex_target_window() -> None:
    securities, benchmark = _history()
    dataset = build_dataset(securities, benchmark)
    target = dataset.targets[0]
    ticker_by_date = {item.trade_date: item for item in securities[target.ticker]}
    benchmark_by_date = {item.trade_date: item for item in benchmark}
    expected_security = (
        ticker_by_date[target.trade_date].close / ticker_by_date[target.baseline_trade_date].close
        - 1
    )
    expected_benchmark = (
        benchmark_by_date[target.trade_date].close
        / benchmark_by_date[target.baseline_trade_date].close
        - 1
    )
    assert math.isclose(target.next_session_return, expected_security, abs_tol=1e-12)
    assert math.isclose(target.imoex_next_session_return, expected_benchmark, abs_tol=1e-12)
    assert math.isclose(
        target.next_session_abnormal_return,
        expected_security - expected_benchmark,
        abs_tol=1e-12,
    )


def test_missing_session_is_not_forward_filled_and_listing_boundary_is_respected() -> None:
    securities, benchmark = _history(sessions=90)
    missing_date = securities["AAA"][45].trade_date
    shortened = tuple(item for item in securities["AAA"] if item.trade_date != missing_date)
    listed_late = securities["AAA"][30:]
    result = build_dataset({"AAA": shortened, "LATE": _reticker(listed_late, "LATE")}, benchmark)
    assert not any(
        item.trade_date == missing_date and item.ticker == "AAA" for item in result.features
    )
    assert result.quality["prices_forward_filled"] is False
    assert result.quality["synthetic_market_rows"] == 0
    assert result.quality["missing_sessions_by_ticker"]["AAA"] == 1
    first_late = min(item.trade_date for item in result.features if item.ticker == "LATE")
    assert first_late > listed_late[20].trade_date


def test_extreme_target_is_retained_and_not_used_for_cleaning() -> None:
    securities, benchmark = _history()
    rows = list(securities["AAA"])
    item = rows[30]
    rows[30] = replace(
        item,
        open=item.open * 4,
        close=item.close * 4,
        high=item.high * 4,
        low=item.low * 4,
    )
    result = build_dataset({"AAA": tuple(rows)}, benchmark)
    row_key = f"AAA:{item.trade_date.isoformat()}"
    target = next(value for value in result.targets if value.row_id == row_key)
    assert target.next_session_return > 2
    assert result.quality["target_based_cleaning"] is False
    assert result.quality["extreme_targets_removed"] == 0


def test_readiness_thresholds_and_ticker_diversity() -> None:
    assert readiness_for_rows(999, 10)["status"] == Readiness.NOT_READY.value
    assert readiness_for_rows(1000, 10)["status"] == Readiness.MARKET_PILOT_READY.value
    assert (
        readiness_for_rows(5000, 10)["status"] == Readiness.MARKET_BASELINE_EXPERIMENT_READY.value
    )
    assert readiness_for_rows(10000, 10)["status"] == Readiness.MARKET_BASELINE_TRAINING_READY.value
    assert readiness_for_rows(10000, 4)["warnings"] == ["LOW_TICKER_DIVERSITY"]


def test_reports_preserve_market_event_separation_and_fingerprints(tmp_path: Path) -> None:
    securities, benchmark = _history(tickers=("AAA", "BBB"), sessions=100)
    dataset = build_dataset(securities, benchmark)
    split = date_grouped_temporal_split(dataset.features)
    paths = write_market_baseline_artifacts(
        tmp_path,
        result=dataset,
        split=split,
        acquisition={"provider": "MOEX_ISS", "zero_cost": True},
        git_sha="abc123",
        event_daily_feature_ready=34,
    )
    feature = json.loads(paths["features"].read_text(encoding="utf-8").splitlines()[0])
    target = json.loads(paths["targets"].read_text(encoding="utf-8").splitlines()[0])
    manifest = json.loads(paths["dataset_manifest"].read_text(encoding="utf-8"))
    quality = json.loads(paths["quality_report"].read_text(encoding="utf-8"))
    assert "next_session_return" not in feature["features"]
    assert "features" not in target
    assert manifest["dataset_version"] == DATASET_VERSION
    assert manifest["price_adjustment_status"] == PRICE_ADJUSTMENT_STATUS
    assert manifest["event_features_included"] is False
    assert quality["target_day_present_in_features"] is False
    status = load_market_baseline_status(tmp_path)
    assert status["event_daily_feature_ready"] == 34
    assert status["model_trained"] is False
    assert SOURCE_POLICY == "ZERO_COST_OFFICIAL_ISS_ONLY"
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert DEFAULT_OLLAMA_MODEL == "qwen3.5:9b"


async def test_existing_moex_client_fetches_paginated_daily_candles() -> None:
    starts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        starts.append(request.url.params["start"])
        assert request.url.params["interval"] == "24"
        rows = (
            [_daily_payload_row(date(2020, 1, 1)) for _ in range(500)] if len(starts) == 1 else []
        )
        return httpx.Response(
            200,
            json={
                "candles": {
                    "columns": ["open", "close", "high", "low", "value", "volume", "begin", "end"],
                    "data": rows,
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = MoexIssClient(
            base_url="https://iss.moex.com/iss",
            timeout_seconds=1,
            max_retries=0,
            max_pages=3,
            user_agent="tests",
            client=http_client,
        )
        result = await client.fetch_daily_candles(
            security_code="SBER",
            engine="stock",
            market="shares",
            board="TQBR",
            date_from=date(2000, 1, 1),
            date_till=date(2026, 1, 1),
        )
    assert starts == ["0", "500"]
    assert result.rows_valid == 500
    assert result.candles[0].trade_date == date(2020, 1, 1)


def test_market_baseline_does_not_depend_on_event_or_live_collector_code() -> None:
    root = Path(__file__).parents[2]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "src" / "market_baseline").glob("*.py"))
    )
    assert "src.events" not in source
    assert "live_corpus_operations" not in source
    assert "train_predictive" not in source


def _history(
    *, tickers: tuple[str, ...] = ("AAA",), sessions: int = 80
) -> tuple[dict[str, tuple[DailyBar, ...]], tuple[DailyBar, ...]]:
    start = date(2020, 1, 1)
    benchmark = tuple(
        _bar("IMOEX", start + timedelta(days=index), 1000 + index * 2, 10000 + index)
        for index in range(sessions)
    )
    securities = {
        ticker: tuple(
            _bar(
                ticker,
                start + timedelta(days=index),
                100 + index * 0.7 + ticker_index,
                1000 + index * 3,
            )
            for index in range(sessions)
        )
        for ticker_index, ticker in enumerate(tickers)
    }
    return securities, benchmark


def _bar(ticker: str, trade_date: date, close: float, volume: float) -> DailyBar:
    return DailyBar(
        ticker=ticker,
        trade_date=trade_date,
        open=close - 0.2,
        close=close,
        high=close + 0.5,
        low=close - 0.5,
        volume=volume,
        value=volume * close,
    )


def _reticker(rows: tuple[DailyBar, ...], ticker: str) -> tuple[DailyBar, ...]:
    return tuple(replace(item, ticker=ticker) for item in rows)


def _daily_payload_row(trade_date: date) -> list[object]:
    text = trade_date.isoformat()
    return [100, 101, 102, 99, 100000, 1000, f"{text} 00:00:00", f"{text} 23:59:59"]
