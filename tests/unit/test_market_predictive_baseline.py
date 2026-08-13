from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from apps.cli.train_tinvest_market_baseline import build_parser
from src.market_predictive_baseline.application import run_frozen_market_baseline
from src.market_predictive_baseline.domain import (
    DIRECTIONS,
    EXPECTED_DATASET_SHA,
    EXPECTED_FEATURE_SCHEMA_SHA,
    EXPECTED_SPLIT_SHA,
    FinalModelConfig,
    FrozenMarketDataset,
    MarketFeatureRow,
    MarketTargetRow,
    research_safety_flags,
    validate_frozen_metadata,
)
from src.market_predictive_baseline.modeling import (
    MarketModels,
    diagnostic_views,
    evaluate_models,
    model_quality_status,
)
from src.tinvest_market.domain import FEATURE_DATASET_VERSION, feature_names
from src.tinvest_market.policy import PRICE_ADJUSTMENT_STATUS, SOURCE_USAGE_READINESS


def test_final_config_is_fixed_and_fingerprinted() -> None:
    payload = FinalModelConfig().payload()
    assert payload["hyperparameter_selection"] == "FIXED_A_PRIORI_NO_SEARCH"
    assert payload["test_config_locked"] is True
    assert len(payload["config_sha"]) == 64
    assert payload["flat_return_threshold"] == 0.002


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_sha", "0" * 64),
        ("split_sha", "0" * 64),
        ("feature_schema_sha", "0" * 64),
    ],
)
def test_frozen_dataset_mismatch_fails_closed(field: str, value: str) -> None:
    dataset = _frozen_metadata()
    with pytest.raises(ValueError, match="frozen market dataset mismatch"):
        validate_frozen_metadata(replace(dataset, **{field: value}))


def test_research_safety_disables_every_execution_surface() -> None:
    flags = research_safety_flags()
    assert flags["RESEARCH_BASELINE_ONLY"] is True
    assert all(value is False for key, value in flags.items() if key != "RESEARCH_BASELINE_ONLY")


def test_models_fit_only_on_rows_explicitly_provided() -> None:
    rows, targets = _model_rows(90)
    train = rows[:60]
    models = MarketModels.create(_small_config())
    models.fit(train, targets)
    assert models.fitted_row_ids == tuple(row.row_id for row in train)
    scaler = cast("Any", models.classifier.named_steps["scale"])
    assert int(scaler.n_samples_seen_) == 60


def test_metrics_include_naive_probability_and_diagnostic_views() -> None:
    rows, targets = _model_rows(90)
    train, observed = rows[:60], rows[60:]
    models = MarketModels.create(_small_config())
    models.fit(train, targets)
    metrics, records = evaluate_models(models, observed, targets, targets)
    diagnostics = diagnostic_views(records)
    assert "weighted_f1" in metrics["classification"]["model"]
    assert "log_loss" in metrics["classification"]["naive_majority"]
    assert set(diagnostics["per_ticker"]) == {"AAA", "BBB", "CCC"}
    assert set(diagnostics["per_year"]) == {"2024"}
    assert "date_equal_weighted" in diagnostics
    assert model_quality_status(metrics, diagnostics) in {
        "NO_PREDICTIVE_SIGNAL",
        "BASELINE_SIGNAL_PRESENT",
    }


def test_cli_defaults_to_frozen_private_artifacts() -> None:
    args = build_parser().parse_args([])
    assert args.dataset_dir == "artifacts/tinvest-market-baseline-features-v1"
    assert args.output_dir == "artifacts/tinvest-market-predictive-baseline-v1"


def test_one_time_runner_rejects_nonempty_output_before_dataset_access(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "test-evaluation-state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        run_frozen_market_baseline(tmp_path / "missing", output, git_sha="a" * 40)


def test_application_locks_config_before_loading_test_targets() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "market_predictive_baseline" / "application.py"
    ).read_text(encoding="utf-8")
    lock = source.index('output_root / "final_model_config.json"')
    state = source.index('"TEST_EVALUATION_COUNT": 1')
    test_load = source.index('frozenset({"TEST"})')
    assert lock < state < test_load


def test_application_has_no_broker_or_live_automation_import() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "market_predictive_baseline" / "application.py"
    ).read_text(encoding="utf-8")
    assert "TInvestReadOnlyClient" not in source
    assert "src.tinvest_market.client" not in source
    assert '"live_automation_connected": False' in source


def test_documentation_forbids_backtest_and_marks_observed_test() -> None:
    text = (
        Path(__file__).parents[2] / "docs" / "tinvest-market-predictive-baseline-v1.md"
    ).read_text(encoding="utf-8")
    assert "OBSERVED_AFTER_BASELINE_V1" in text
    assert "do not establish causality" in text
    assert "never computes PnL" in text


def _frozen_metadata() -> FrozenMarketDataset:
    return FrozenMarketDataset(
        features=(),
        assignments={},
        date_ranges={},
        counts={"TRAIN": 30156, "VALIDATION": 9583, "TEST": 9852},
        dataset_sha=EXPECTED_DATASET_SHA,
        split_sha=EXPECTED_SPLIT_SHA,
        feature_schema_sha=EXPECTED_FEATURE_SCHEMA_SHA,
        feature_names=feature_names(True),
        dataset_version=FEATURE_DATASET_VERSION,
        source_usage_readiness=SOURCE_USAGE_READINESS,
        price_adjustment_status=PRICE_ADJUSTMENT_STATUS,
    )


def _small_config() -> FinalModelConfig:
    return replace(FinalModelConfig(), feature_names=("first", "second"))


def _model_rows(count: int) -> tuple[tuple[MarketFeatureRow, ...], dict[str, MarketTargetRow]]:
    rows: list[MarketFeatureRow] = []
    targets: dict[str, MarketTargetRow] = {}
    start = date(2024, 1, 1)
    tickers = ("AAA", "BBB", "CCC")
    for index in range(count):
        trade_date = start + timedelta(days=index)
        ticker = tickers[index % len(tickers)]
        row_id = f"{ticker}:{trade_date.isoformat()}"
        signal = float((index % 9) - 4) / 10
        direction = DIRECTIONS[index % len(DIRECTIONS)]
        rows.append(
            MarketFeatureRow(
                row_id=row_id,
                ticker=ticker,
                trade_date=trade_date,
                feature_as_of=trade_date - timedelta(days=1),
                values=(signal, float(index % 5)),
            )
        )
        targets[row_id] = MarketTargetRow(
            row_id=row_id,
            ticker=ticker,
            trade_date=trade_date,
            direction=direction,
            abnormal_return=signal / 100,
            security_return=(signal + 0.1) / 100,
        )
    return tuple(rows), targets
