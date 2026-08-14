from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from apps.cli.train_event_predictive_baseline import build_parser
from src.event_predictive_baseline.application import run_event_predictive_baseline
from src.event_predictive_baseline.data import build_temporal_split, validate_frozen_manifest
from src.event_predictive_baseline.diagnostics import (
    comparison_deltas,
    grouped_diagnostics,
    incremental_value_status,
)
from src.event_predictive_baseline.domain import (
    DIRECTIONS,
    EXPECTED_DATASET_SHA,
    EXPECTED_FEATURE_SCHEMA_SHA,
    EXPECTED_PROVENANCE_SHA,
    EXPECTED_SOURCE_REGISTRY_SHA,
    FEATURE_FAMILIES,
    ComparisonCohort,
    EventFeatureRow,
    EventTargetRow,
    FrozenModelConfig,
    research_safety_flags,
    sha256_payload,
)
from src.event_predictive_baseline.modeling import (
    evaluate_all_families,
    fit_all_families,
)


def test_frozen_config_and_safety_are_explicit() -> None:
    payload = FrozenModelConfig().payload()
    assert payload["hyperparameter_selection"] == "FIXED_A_PRIORI_NO_SEARCH"
    assert payload["test_config_locked"] is True
    assert payload["feature_families"] == list(FEATURE_FAMILIES)
    assert payload["flat_return_threshold"] == 0.002
    flags = research_safety_flags()
    assert flags["RESEARCH_ONLY"] is True
    assert all(value is False for key, value in flags.items() if key != "RESEARCH_ONLY")


def test_frozen_dataset_manifest_fails_closed_on_any_sha_change() -> None:
    manifest = _valid_manifest()
    for name in (
        "event_market_dataset_sha",
        "source_registry_sha",
        "provenance_manifest_sha",
        "feature_schema_sha",
    ):
        changed = {**manifest, name: "0" * 64}
        with pytest.raises(ValueError, match="frozen event dataset mismatch"):
            validate_frozen_manifest(changed)


def test_temporal_split_keeps_dates_issuer_dates_and_stories_together() -> None:
    rows, _ = _rows_and_targets()
    cohort = ComparisonCohort(
        rows=rows,
        event_feature_names=("primary_event_type", "event_count", "fact_count"),
        market_feature_names=("return_1d", "volatility_5d"),
        cohort_sha="a" * 64,
        event_schema_sha="b" * 64,
        market_schema_sha="c" * 64,
    )
    split = build_temporal_split(cohort)
    assignments = {item["event_id"]: item["split"] for item in split["assignments"]}
    assert split["target_outcomes_inspected_before_lock"] is False
    assert split["leakage_check"] == "PASS"
    assert len(split["split_sha"]) == 64
    by_date: dict[date, set[str]] = {}
    for row in rows:
        by_date.setdefault(row.publication_date, set()).add(assignments[row.event_id])
    assert all(len(values) == 1 for values in by_date.values())
    assert split["counts"] == {"TRAIN": 45, "VALIDATION": 45, "TEST": 45}


def test_all_models_fit_and_evaluate_on_identical_event_rows() -> None:
    rows, targets = _rows_and_targets()
    train, observed = rows[:90], rows[90:]
    train_targets = {row.event_id: targets[row.event_id] for row in train}
    observed_targets = {row.event_id: targets[row.event_id] for row in observed}
    models = fit_all_families(train, train_targets, FrozenModelConfig())
    assert {model.fitted_event_ids for model in models.values()} == {
        tuple(row.event_id for row in train)
    }
    for model in models.values():
        vectorizer = cast("Any", model.classifier.named_steps["vectorize"])
        assert len(vectorizer.feature_names_) > 0
    metrics, records = evaluate_all_families(models, observed, observed_targets, train_targets)
    assert metrics["rows"] == len(observed)
    assert set(metrics["classification"]["models"]) == set(FEATURE_FAMILIES)
    assert all(set(row["models"]) == set(FEATURE_FAMILIES) for row in records)


def test_unknown_event_category_is_safe_at_evaluation() -> None:
    rows, targets = _rows_and_targets()
    train = rows[:90]
    unseen = EventFeatureRow(
        event_id="unseen",
        ticker="ZZZZ",
        issuer_name="Unseen",
        publication_date=date(2026, 8, 1),
        source_family="OFFICIAL",
        title_hash="z" * 64,
        event_features={"primary_event_type": "UNSEEN_CATEGORY", "event_count": 1, "fact_count": 0},
        market_features={"return_1d": 0.01, "volatility_5d": 0.02},
    )
    train_targets = {row.event_id: targets[row.event_id] for row in train}
    models = fit_all_families(train, train_targets, FrozenModelConfig())
    predictions = [model.predict((unseen,)) for model in models.values()]
    assert all(len(result["directions"]) == 1 for result in predictions)


def test_diagnostics_include_row_weighted_issuer_macro_and_deltas() -> None:
    rows, targets = _rows_and_targets()
    train, observed = rows[:90], rows[90:]
    train_targets = {row.event_id: targets[row.event_id] for row in train}
    observed_targets = {row.event_id: targets[row.event_id] for row in observed}
    models = fit_all_families(train, train_targets, FrozenModelConfig())
    metrics, records = evaluate_all_families(models, observed, observed_targets, train_targets)
    diagnostics = grouped_diagnostics(records)
    deltas = comparison_deltas(metrics, "C_EVENT_PLUS_MARKET", "A_MARKET_ONLY")
    assert set(diagnostics) == {
        "ROW_WEIGHTED",
        "ISSUER_MACRO",
        "per_ticker",
        "per_year",
        "per_source_family",
    }
    assert deltas["comparison"] == "C_EVENT_PLUS_MARKET_MINUS_A_MARKET_ONLY"
    assert incremental_value_status(metrics, diagnostics) in {
        "NO_EVENT_INCREMENTAL_SIGNAL",
        "EVENT_INCREMENTAL_SIGNAL_CANDIDATE",
    }


def test_runner_locks_configuration_before_loading_test_targets() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "event_predictive_baseline" / "application.py"
    ).read_text(encoding="utf-8")
    config_lock = source.index('output_root / "final-model-config.json"')
    count_one = source.index('"TEST_EVALUATION_COUNT": 1')
    test_load = source.index("test_targets = load_targets")
    assert config_lock < count_one < test_load


def test_cohort_and_split_loader_never_open_targets() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "event_predictive_baseline" / "data.py"
    ).read_text(encoding="utf-8")
    cohort_start = source.index("def load_comparison_cohort")
    split_start = source.index("def build_temporal_split")
    target_start = source.index("def load_targets")
    assert "targets.jsonl" not in source[cohort_start:target_start]
    assert "abnormal_return" not in source[cohort_start:target_start]
    assert cohort_start < split_start < target_start


def test_runner_rejects_nonempty_output_without_touching_dataset(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        run_event_predictive_baseline(tmp_path / "missing", output, git_sha="a" * 40)


def test_runner_has_no_broker_qwen_or_live_automation_import() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "event_predictive_baseline" / "application.py"
    ).read_text(encoding="utf-8")
    assert "TInvestReadOnlyClient" not in source
    assert "src.tinvest_market.client" not in source
    assert "Ollama" not in source
    assert '"qwen_run": False' in source
    assert '"buy_sell_generated": False' in source


def test_cli_defaults_to_frozen_event_artifacts() -> None:
    args = build_parser().parse_args([])
    assert args.dataset_dir == "artifacts/event-market-predictive-dataset-v2"
    assert args.output_dir == "artifacts/event-predictive-baseline-v1"


def test_documentation_closes_test_and_forbids_trading() -> None:
    text = (Path(__file__).parents[2] / "docs" / "event-predictive-baseline-v1.md").read_text(
        encoding="utf-8"
    )
    assert "OBSERVED_AFTER_EVENT_BASELINE_V1" in text
    assert "same event IDs" in text
    assert "No PnL" in text
    assert "new forward holdout" in text


def _valid_manifest() -> dict[str, object]:
    return {
        "dataset_version": "event-market-predictive-dataset-v2",
        "event_market_dataset_sha": EXPECTED_DATASET_SHA,
        "source_registry_sha": EXPECTED_SOURCE_REGISTRY_SHA,
        "provenance_manifest_sha": EXPECTED_PROVENANCE_SHA,
        "feature_schema_sha": EXPECTED_FEATURE_SCHEMA_SHA,
        "event_market_leakage_check": "PASS",
        "predictive_unit": "EVENT",
        "new_total_real_events": 1276,
        "event_market_feature_ready": 1260,
        "unverified_events": 0,
        "rules_changed": False,
        "qwen_changed": False,
        "model_trained": False,
    }


def _rows_and_targets() -> tuple[tuple[EventFeatureRow, ...], dict[str, EventTargetRow]]:
    rows: list[EventFeatureRow] = []
    targets: dict[str, EventTargetRow] = {}
    periods = (
        (date(2024, 1, 1), 45),
        (date(2025, 1, 1), 45),
        (date(2026, 1, 1), 45),
    )
    index = 0
    for start, count in periods:
        for offset in range(count):
            ticker = ("AAA", "BBB", "CCC")[index % 3]
            publication_date = start + timedelta(days=offset // 2)
            event_id = f"event-{index}"
            signal = float((index % 11) - 5) / 100
            rows.append(
                EventFeatureRow(
                    event_id=event_id,
                    ticker=ticker,
                    issuer_name=ticker,
                    publication_date=publication_date,
                    source_family=f"{ticker}_OFFICIAL",
                    title_hash=sha256_payload([ticker, publication_date.isoformat(), offset // 2]),
                    event_features={
                        "primary_event_type": ("OTHER", "DIVIDEND", "UNKNOWN")[index % 3],
                        "event_count": index % 2,
                        "fact_count": index % 4,
                    },
                    market_features={
                        "return_1d": signal,
                        "volatility_5d": abs(signal) + 0.01,
                    },
                )
            )
            targets[event_id] = EventTargetRow(
                event_id=event_id,
                direction=DIRECTIONS[index % 3],
                abnormal_return=signal / 10,
                security_return=(signal + 0.002) / 10,
            )
            index += 1
    return tuple(rows), targets
