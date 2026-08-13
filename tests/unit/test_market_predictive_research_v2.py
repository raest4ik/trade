# pyright: reportUnknownMemberType=false
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from apps.cli.market_future_holdout_status import build_parser as future_parser
from apps.cli.run_market_predictive_research_v2 import build_parser as research_parser
from src.market_predictive_research.data import (
    build_horizon_targets,
    enhance_features,
    future_holdout_coverage,
    load_development_dataset,
)
from src.market_predictive_research.diagnostics import (
    dataset_diagnostics,
    fold_feature_associations,
)
from src.market_predictive_research.domain import (
    DEVELOPMENT_TO,
    FUTURE_HOLDOUT_START,
    OBSERVED_TEST_START,
    DevelopmentDataset,
    DevelopmentFeatureRow,
    OneSessionTarget,
    safety_flags,
    sha256_payload,
)
from src.market_predictive_research.folds import build_rolling_folds
from src.market_predictive_research.modeling import (
    aggregate_research_results,
    daily_rank_ic,
    evaluate_fold,
    stability_views,
)
from src.tinvest_market.domain import feature_names
from src.tinvest_market.policy import PRICE_ADJUSTMENT_STATUS, SOURCE_USAGE_READINESS


def test_observed_test_request_fails_before_filesystem_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="OBSERVED_TEST_READ_ATTEMPT"):
        load_development_dataset(tmp_path / "missing", requested_to=OBSERVED_TEST_START)


def test_development_cutoff_is_strictly_before_observed_test() -> None:
    assert DEVELOPMENT_TO < OBSERVED_TEST_START
    source = (
        Path(__file__).parents[2] / "src" / "market_predictive_research" / "data.py"
    ).read_text(encoding="utf-8")
    assert "load_frozen_market_dataset" not in source
    assert "if trade_date < requested_from or trade_date > requested_to" in source


def test_feature_loader_accepts_json_key_order_but_keeps_frozen_schema() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "market_predictive_research" / "data.py"
    ).read_text(encoding="utf-8")
    assert "set(values) != set(expected_names)" in source
    assert "{name: float(values[name]) for name in expected_names}" in source


def test_future_status_reads_feature_metadata_without_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "features.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"trade_date":"2022-09-20", this payload is deliberately not materialized}',
                json.dumps(
                    {
                        "trade_date": FUTURE_HOLDOUT_START.isoformat(),
                        "ticker": "SBER",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = future_holdout_coverage(path)
    assert result["session_count"] == 1
    assert result["outcomes_loaded"] is False
    assert result["performance_metrics_computed"] is False


def test_v2_features_are_t_minus_one_and_cross_sectional() -> None:
    rows = _source_rows(1, 3)
    enhanced, names = enhance_features(rows)
    assert len(names) == 43
    assert all(row.feature_as_of < row.trade_date for row in enhanced)
    ranks = [row.values["cross_sectional_rank_return_20d"] for row in enhanced]
    assert ranks == [0.0, 0.5, 1.0]
    assert all("target" not in name and "future" not in name for name in names)


def test_multisession_targets_use_compounding() -> None:
    start = date(2020, 1, 1)
    rows = tuple(
        OneSessionTarget(
            row_id=f"AAA:{(start + timedelta(days=index)).isoformat()}",
            ticker="AAA",
            trade_date=start + timedelta(days=index),
            security_return=0.01,
            benchmark_return=0.005,
        )
        for index in range(5)
    )
    targets = build_horizon_targets(rows)
    first = targets[3][rows[0].row_id]
    assert first.security_return == pytest.approx(1.01**3 - 1)
    assert first.benchmark_return == pytest.approx(1.005**3 - 1)
    assert first.abnormal_return == pytest.approx(1.01**3 - 1.005**3)


def test_rolling_folds_are_deterministic_grouped_and_horizon_purged() -> None:
    dataset = _dataset(120, 3)
    first = build_rolling_folds(dataset)
    second = build_rolling_folds(dataset)
    assert first.fold_manifest_sha == second.fold_manifest_sha
    assert len(first.folds) == 15
    by_id = {row.row_id: row for row in dataset.rows}
    for fold in first.folds:
        train_dates = {by_id[item].trade_date for item in fold.train_row_ids}
        validation_dates = {by_id[item].trade_date for item in fold.validation_row_ids}
        assert max(train_dates) < min(validation_dates)
        assert len(fold.purged_dates) == fold.horizon
        assert len(fold.embargoed_dates) == 1
        for trade_date in train_dates | validation_dates:
            expected = {
                row.row_id
                for row in dataset.aligned_rows(fold.horizon)
                if row.trade_date == trade_date
            }
            selected = set(fold.train_row_ids) | set(fold.validation_row_ids)
            assert expected <= selected


def test_fixed_candidates_run_on_one_development_fold() -> None:
    dataset = _dataset(120, 3)
    fold = build_rolling_folds(dataset).folds[0]
    result = evaluate_fold(dataset, fold)
    assert set(result["regression"]["abnormal_return"]) == {
        "naive_zero",
        "naive_train_mean",
        "Ridge",
        "HistGradientBoostingRegressor",
    }
    assert set(result["classification"]) == {
        "naive_majority",
        "LogisticRegression",
        "HistGradientBoostingClassifier",
    }
    assert result["rank_ic"]["Ridge"]["date_count"] > 0


def test_aggregate_status_is_development_only() -> None:
    dataset = _dataset(80, 3)
    fold_manifest = build_rolling_folds(dataset)
    folds = [evaluate_fold(dataset, fold) for fold in fold_manifest.folds]
    result = aggregate_research_results(folds)
    best = result["best_development_candidate"]
    stability = stability_views(folds, model=best["model"], horizon=best["horizon"])
    assert result["DEVELOPMENT_STATUS"] in {"NO_DEV_SIGNAL", "DEV_SIGNAL_CANDIDATE"}
    assert result["CONFIRMED_SIGNAL"] is False
    assert set(stability) == {"per_ticker", "per_year", "rank_ic"}


def test_diagnostics_cover_required_research_views() -> None:
    dataset = _dataset(40, 3)
    result = dataset_diagnostics(dataset)
    assert set(result["target_distribution"]) == {"1", "3", "5"}
    assert result["feature_missingness"]
    assert result["feature_stability"]
    assert result["univariate_feature_target_associations"]
    assert result["feature_associations_by_ticker"]
    assert result["feature_associations_by_year"]
    assert len(fold_feature_associations(dataset, build_rolling_folds(dataset))) == 15
    assert result["target_autocorrelation"]
    assert result["causal_claim"] is False


def test_rank_ic_is_same_date_cross_sectional() -> None:
    records = [
        {
            "trade_date": "2020-01-01",
            "actual_abnormal_return": float(index),
            "prediction": float(index),
        }
        for index in range(4)
    ]
    result = daily_rank_ic(records, "prediction")
    assert result["date_count"] == 1
    assert result["mean_daily_rank_ic"] == pytest.approx(1.0)


def test_all_execution_and_trading_surfaces_are_disabled() -> None:
    flags = safety_flags()
    assert flags["RESEARCH_ONLY"] is True
    assert all(value is False for key, value in flags.items() if key != "RESEARCH_ONLY")


def test_cli_has_no_test_or_future_evaluation_switch() -> None:
    research = research_parser()
    destinations = {action.dest for action in research._actions}
    assert "test" not in destinations
    assert "future" not in destinations
    assert (
        future_parser().parse_args([]).dataset_dir.endswith("tinvest-market-baseline-features-v1")
    )


def test_documentation_records_no_backtest_and_future_policy() -> None:
    text = (Path(__file__).parents[2] / "docs" / "market-predictive-research-v2.md").read_text(
        encoding="utf-8"
    )
    assert "OBSERVED_TEST_READ_ATTEMPT" in text
    assert "does not calculate PnL" in text
    assert "FUTURE_HOLDOUT_OBSERVED=false" in text


def _dataset(days: int, ticker_count: int) -> DevelopmentDataset:
    source = _source_rows(days, ticker_count)
    rows, names = enhance_features(source)
    one_session: list[OneSessionTarget] = []
    for index, row in enumerate(rows):
        signal = (index % 9 - 4) / 1000
        one_session.append(
            OneSessionTarget(
                row_id=row.row_id,
                ticker=row.ticker,
                trade_date=row.trade_date,
                security_return=signal + (index % 3 - 1) / 100,
                benchmark_return=signal / 2,
            )
        )
    targets = build_horizon_targets(tuple(one_session))
    return DevelopmentDataset(
        rows=rows,
        targets=targets,
        feature_names=names,
        feature_schema_sha=sha256_payload(list(names)),
        dataset_sha="d" * 64,
        split_sha="s" * 64,
        price_adjustment_status=PRICE_ADJUSTMENT_STATUS,
        source_usage_readiness=SOURCE_USAGE_READINESS,
    )


def _source_rows(days: int, ticker_count: int) -> tuple[DevelopmentFeatureRow, ...]:
    start = date(2010, 1, 4)
    names = feature_names(True)
    rows: list[DevelopmentFeatureRow] = []
    for day in range(days):
        trade_date = start + timedelta(days=day)
        for ticker_index in range(ticker_count):
            ticker = f"T{ticker_index}"
            base = float(day + ticker_index + 1)
            values = {name: base / (position + 10) for position, name in enumerate(names)}
            rows.append(
                DevelopmentFeatureRow(
                    row_id=f"{ticker}:{trade_date.isoformat()}",
                    ticker=ticker,
                    trade_date=trade_date,
                    feature_as_of=trade_date - timedelta(days=1),
                    values=values,
                )
            )
    return tuple(rows)
