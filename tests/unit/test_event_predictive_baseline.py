from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from apps.cli.train_event_predictive_baseline import build_parser
from src.event_predictive_baseline.application import run_event_predictive_baseline
from src.event_predictive_baseline.data import build_temporal_split, validate_exact_manifest
from src.event_predictive_baseline.diagnostics import (
    comparison_deltas,
    concentration_diagnostics,
    grouped_diagnostics,
    incremental_value_status,
    timestamp_hypothesis_status,
)
from src.event_predictive_baseline.domain import (
    DIRECTIONS,
    EXPECTED_CLUSTER_SHA,
    EXPECTED_DATASET_SHA,
    EXPECTED_PROVENANCE_SHA,
    EXPECTED_REACTION_SHA,
    EXPECTED_SOURCE_REGISTRY_SHA,
    EXPECTED_TIMESTAMP_SHA,
    FEATURE_FAMILIES,
    FUTURE_EVENT_HOLDOUT_START,
    PRIMARY_EXACT_HORIZON,
    EventFeatureRow,
    EventTargetRow,
    FrozenModelConfig,
    FutureHoldoutOutcomeReadError,
    HorizonCohort,
    guard_future_holdout_outcome_read,
    research_safety_flags,
    sha256_payload,
)
from src.event_predictive_baseline.modeling import (
    evaluate_all_families,
    fit_all_families,
)


def test_frozen_exact_config_and_safety_are_explicit() -> None:
    payload = FrozenModelConfig().payload()
    assert payload["primary_horizon"] == PRIMARY_EXACT_HORIZON == "15m"
    assert payload["secondary_horizons"] == ("1m", "5m", "30m", "60m")
    assert payload["hyperparameter_selection"] == "FIXED_A_PRIORI_NO_SEARCH"
    assert payload["test_config_locked"] is True
    assert payload["feature_families"] == list(FEATURE_FAMILIES)
    flags = research_safety_flags()
    assert flags["RESEARCH_ONLY"] is True
    assert flags["BACKTEST_APPROVED"] is False
    assert flags["PAPER_TRADING_APPROVED"] is False
    assert flags["REAL_TRADING_APPROVED"] is False


def test_future_holdout_outcome_guard_fails_closed() -> None:
    guard_future_holdout_outcome_read(FUTURE_EVENT_HOLDOUT_START - timedelta(days=1), context="ok")
    with pytest.raises(FutureHoldoutOutcomeReadError, match="FUTURE_EVENT_HOLDOUT_READ_ATTEMPT"):
        guard_future_holdout_outcome_read(FUTURE_EVENT_HOLDOUT_START, context="target")


def test_exact_manifest_fails_closed_on_sha_or_holdout_change() -> None:
    manifest = _valid_manifest()
    holdout = _valid_holdout()
    for name in (
        "exact_dataset_sha",
        "source_registry_sha",
        "provenance_sha",
        "timestamp_manifest_sha",
        "reaction_manifest_sha",
        "cluster_manifest_sha",
    ):
        changed = {**manifest, name: "0" * 64}
        with pytest.raises(ValueError, match="frozen exact dataset mismatch"):
            validate_exact_manifest(changed, holdout)
    with pytest.raises(ValueError, match="future holdout observed"):
        validate_exact_manifest(manifest, {**holdout, "FUTURE_EVENT_HOLDOUT_OBSERVED": True})


def test_temporal_split_is_grouped_chronological_and_cluster_atomic() -> None:
    cohort = _cohort()
    split = build_temporal_split(cohort)
    assignments = {item["event_id"]: item["split"] for item in split["assignments"]}
    assert split["target_outcomes_inspected_before_lock"] is False
    assert split["cluster_integrity"] == "PASS"
    assert split["leakage_check"] == "PASS"
    assert len(split["split_sha"]) == 64
    by_date: dict[date, set[str]] = {}
    by_cluster: dict[str, set[str]] = {}
    for row in cohort.rows:
        by_date.setdefault(row.publication_date, set()).add(assignments[row.event_id])
        by_cluster.setdefault(row.event_cluster_id, set()).add(assignments[row.event_id])
    assert all(len(values) == 1 for values in by_date.values())
    assert all(len(values) == 1 for values in by_cluster.values())
    assert set(split["counts"]) == {"TRAIN", "VALIDATION", "TEST"}


def test_all_abc_models_fit_and_evaluate_on_identical_event_ids() -> None:
    cohort = _cohort()
    rows = cohort.rows
    train, observed = rows[:90], rows[90:]
    train_targets = {row.event_id: cohort.targets[row.event_id] for row in train}
    observed_targets = {row.event_id: cohort.targets[row.event_id] for row in observed}
    models = fit_all_families(train, train_targets, FrozenModelConfig())
    assert {model.fitted_event_ids for model in models.values()} == {
        tuple(row.event_id for row in train)
    }
    metrics, records = evaluate_all_families(models, observed, observed_targets, train_targets)
    assert metrics["rows"] == len(observed)
    assert set(metrics["classification"]["models"]) == set(FEATURE_FAMILIES)
    assert all(set(row["models"]) == set(FEATURE_FAMILIES) for row in records)


def test_preprocessing_is_fit_only_on_training_rows() -> None:
    cohort = _cohort()
    train = cohort.rows[:90]
    targets = {row.event_id: cohort.targets[row.event_id] for row in train}
    models = fit_all_families(train, targets, FrozenModelConfig())
    for model in models.values():
        vectorizer = cast("Any", model.classifier.named_steps["vectorize"])
        assert tuple(row.event_id for row in train) == model.fitted_event_ids
        assert len(vectorizer.feature_names_) > 0


def test_unknown_event_category_is_handled_without_taxonomy_tuning() -> None:
    cohort = _cohort()
    train = cohort.rows[:90]
    unseen = EventFeatureRow(
        event_id="unseen",
        event_cluster_id="cluster-unseen",
        ticker="ZZZZ",
        issuer_name="Unseen",
        publication_date=date(2026, 8, 1),
        publication_timestamp_utc=datetime(2026, 8, 1, 10, tzinfo=UTC),
        source_family="OFFICIAL",
        event_features={"event_count": 1, "fact_count": 0, "primary_event_type": "UNSEEN"},
        market_features={"pre_return_5m": 0.01, "imoex_pre_return_5m": 0.0},
    )
    targets = {row.event_id: cohort.targets[row.event_id] for row in train}
    models = fit_all_families(train, targets, FrozenModelConfig())
    assert all(len(model.predict((unseen,))["directions"]) == 1 for model in models.values())


def test_diagnostics_include_concentration_issuer_macro_and_decision_status() -> None:
    cohort = _cohort()
    train, observed = cohort.rows[:90], cohort.rows[90:]
    train_targets = {row.event_id: cohort.targets[row.event_id] for row in train}
    observed_targets = {row.event_id: cohort.targets[row.event_id] for row in observed}
    models = fit_all_families(train, train_targets, FrozenModelConfig())
    metrics, records = evaluate_all_families(models, observed, observed_targets, train_targets)
    diagnostics = grouped_diagnostics(records)
    deltas = comparison_deltas(metrics, "C_EVENT_PLUS_MARKET", "A_MARKET_ONLY")
    assert "ROW_WEIGHTED" in diagnostics
    assert "ISSUER_MACRO" in diagnostics
    assert "concentration" in diagnostics
    assert concentration_diagnostics(records)["effective_issuer_count"] > 0
    assert deltas["comparison"] == "C_EVENT_PLUS_MARKET_MINUS_A_MARKET_ONLY"
    status = incremental_value_status(metrics, diagnostics)
    assert status in {
        "NO_EXACT_EVENT_INCREMENTAL_SIGNAL",
        "EXACT_EVENT_INCREMENTAL_SIGNAL_CANDIDATE",
    }
    assert timestamp_hypothesis_status(status) in {
        "TIMESTAMP_HYPOTHESIS_NOT_SUPPORTED",
        "TIMESTAMP_HYPOTHESIS_SUPPORTED_AS_CANDIDATE",
    }


def test_runner_locks_config_before_test_evaluation() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "event_predictive_baseline" / "application.py"
    ).read_text(encoding="utf-8")
    lock = source.index('"test-evaluation-state.json"')
    evaluation = source.index("result, model_binary = _evaluate_horizon")
    assert lock < evaluation


def test_runner_rejects_nonempty_output_without_touching_dataset(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        run_event_predictive_baseline(tmp_path / "missing", output, git_sha="a" * 40)


def test_cli_defaults_to_exact_artifacts() -> None:
    args = build_parser().parse_args([])
    assert args.dataset_dir == "artifacts/exact-event-market-dataset-v2"
    assert args.output_dir == "artifacts/exact-event-predictive-baseline-v1"


def test_documentation_closes_future_holdout_and_forbids_trading() -> None:
    text = (Path(__file__).parents[2] / "docs" / "event-predictive-baseline-v1.md").read_text(
        encoding="utf-8"
    )
    assert "PRIMARY_EXACT_HORIZON=15m" in text
    assert "FUTURE_EVENT_HOLDOUT_OBSERVED=false" in text
    assert "No PnL" in text
    assert "BUY/SELL" in text


def _valid_manifest() -> dict[str, object]:
    return {
        "dataset_version": "exact-event-market-dataset-v2",
        "exact_dataset_sha": EXPECTED_DATASET_SHA,
        "source_registry_sha": EXPECTED_SOURCE_REGISTRY_SHA,
        "provenance_sha": EXPECTED_PROVENANCE_SHA,
        "timestamp_manifest_sha": EXPECTED_TIMESTAMP_SHA,
        "reaction_manifest_sha": EXPECTED_REACTION_SHA,
        "cluster_manifest_sha": EXPECTED_CLUSTER_SHA,
        "EVENT_MARKET_LEAKAGE_CHECK": "PASS",
        "EXACT_V1_PRESERVED": "YES",
        "EXACT_MODEL_DATA_STATUS": "READY_FOR_EXACT_BASELINE_EXPERIMENT",
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "holdout_guard": "PASS",
        "rules_changed": False,
        "qwen_changed": False,
        "qwen_run": False,
        "NLP_FROZEN": True,
        "model_trained": False,
        "abc_evaluated": False,
        "backtest_executed": False,
        "paper_trading_executed": False,
        "orders_submitted": False,
        "buy_sell_generated": False,
        "real_trading_executed": False,
    }


def _valid_holdout() -> dict[str, object]:
    return {
        "FUTURE_EVENT_HOLDOUT_START": "2026-08-11",
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "outcome_fields_exported_for_future": 0,
    }


def _cohort() -> HorizonCohort:
    rows: list[EventFeatureRow] = []
    targets: dict[str, EventTargetRow] = {}
    periods = (
        (date(2025, 1, 1), 60),
        (date(2025, 5, 1), 45),
        (date(2026, 1, 1), 45),
    )
    index = 0
    for start, count in periods:
        for offset in range(count):
            ticker = ("MGNT", "T", "X5")[index % 3]
            publication_date = start + timedelta(days=offset // 2)
            event_id = f"event-{index}"
            signal = float((index % 9) - 4) / 1000
            rows.append(
                EventFeatureRow(
                    event_id=event_id,
                    event_cluster_id=f"cluster-{publication_date.isoformat()}",
                    ticker=ticker,
                    issuer_name=ticker,
                    publication_date=publication_date,
                    publication_timestamp_utc=datetime.combine(
                        publication_date, datetime.min.time(), UTC
                    )
                    + timedelta(hours=10, minutes=index % 5),
                    source_family=f"{ticker}_OFFICIAL",
                    event_features={
                        "event_count": index % 2,
                        "fact_count": index % 4,
                        "primary_event_type": ("UNKNOWN", "DEBT_FINANCING", "DIVIDEND")[index % 3],
                    },
                    market_features={
                        "pre_return_5m": signal,
                        "imoex_pre_return_5m": -signal / 2,
                    },
                )
            )
            targets[event_id] = EventTargetRow(
                event_id=event_id,
                horizon="15m",
                direction=DIRECTIONS[index % 3],
                abnormal_return=signal,
                security_return=signal + 0.001,
                benchmark_return=0.001,
                window_begin_at="2025-01-01T10:01:00+00:00",
                window_end_at="2025-01-01T10:16:00+00:00",
                security_observed_at="2025-01-01T10:16:00+00:00",
                benchmark_observed_at="2025-01-01T10:16:00+00:00",
            )
            index += 1
    event_schema = ("event_count", "fact_count", "primary_event_type")
    market_schema = ("imoex_pre_return_5m", "pre_return_5m")
    return HorizonCohort(
        horizon="15m",
        rows=tuple(rows),
        targets=targets,
        event_feature_names=event_schema,
        market_feature_names=market_schema,
        cohort_sha=sha256_payload([row.event_id for row in rows]),
        event_schema_sha=sha256_payload(event_schema),
        market_schema_sha=sha256_payload(market_schema),
        target_schema_sha=sha256_payload(["abnormal_return"]),
    )
