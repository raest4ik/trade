from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from apps.cli.build_exact_event_data_diagnostics import build_parser
from src.exact_event_diagnostics.application import run_exact_event_data_diagnostics
from src.exact_event_diagnostics.domain import (
    ARTIFACT_VERSION,
    EXPECTED_DATASET_SHA,
    FUTURE_EVENT_HOLDOUT_START,
    diagnostic_safety_labels,
    require_expected_exact_dataset,
    sha256_payload,
)


def test_safety_labels_forbid_modeling_test_future_and_trading() -> None:
    labels = diagnostic_safety_labels()
    assert labels["DIAGNOSTIC_ONLY"] is True
    assert labels["MODEL_TRAINING_PERFORMED"] is False
    assert labels["TEST_OUTCOME_USED"] is False
    assert labels["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert labels["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert labels["BACKTEST_APPROVED"] is False
    assert labels["PAPER_TRADING_APPROVED"] is False
    assert labels["REAL_TRADING_APPROVED"] is False
    assert labels["CONFIRMED_SIGNAL"] is False


def test_manifest_guard_fails_closed_on_dataset_or_holdout_change() -> None:
    manifest = _valid_dataset_manifest()
    holdout = _valid_holdout()
    require_expected_exact_dataset(manifest, holdout)
    with pytest.raises(ValueError, match="EXACT_DATASET_SHA_MISMATCH"):
        require_expected_exact_dataset({**manifest, "exact_dataset_sha": "0" * 64}, holdout)
    with pytest.raises(ValueError, match="FUTURE_EVENT_HOLDOUT_OBSERVED"):
        require_expected_exact_dataset(manifest, {**holdout, "FUTURE_EVENT_HOLDOUT_OBSERVED": True})
    with pytest.raises(ValueError, match="FUTURE_HOLDOUT_OUTCOMES_EXPORTED"):
        require_expected_exact_dataset(
            manifest, {**holdout, "outcome_fields_exported_for_future": 1}
        )


def test_cli_defaults_to_exact_data_diagnostics_artifact() -> None:
    args = build_parser().parse_args([])
    assert args.dataset_dir == "artifacts/exact-event-market-dataset-v2"
    assert args.baseline_dir == "artifacts/exact-event-predictive-baseline-v1"
    assert args.output_dir == "artifacts/exact-event-data-diagnostics-v1"


def test_runner_rejects_nonempty_immutable_output(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable"):
        run_exact_event_data_diagnostics(
            tmp_path / "missing-dataset",
            tmp_path / "missing-baseline",
            output,
            git_sha="a" * 40,
        )


def test_real_artifact_builder_is_diagnostic_only_and_train_val_scoped(tmp_path: Path) -> None:
    repo = Path(__file__).parents[2]
    manifest = run_exact_event_data_diagnostics(
        repo / "artifacts" / "exact-event-market-dataset-v2",
        repo / "artifacts" / "exact-event-predictive-baseline-v1",
        tmp_path / ARTIFACT_VERSION,
        git_sha="b" * 40,
        created_at=datetime(2026, 8, 18, tzinfo=UTC),
    )
    assert manifest["dataset_sha"] == EXPECTED_DATASET_SHA
    assert len(manifest["artifact_sha"]) == 64
    assert manifest["artifact_sha"] == sha256_payload({**manifest, "artifact_sha": None})
    assert manifest["safety"]["DIAGNOSTIC_ONLY"] is True
    assert manifest["safety"]["MODEL_TRAINING_PERFORMED"] is False
    assert manifest["target_policy"]["TEST_OUTCOME_USED"] is False
    assert manifest["target_policy"]["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert manifest["counts"]["train"] == 244
    assert manifest["counts"]["validation"] == 82
    assert manifest["counts"]["train_validation"] == 326
    assert manifest["counts"]["test_metadata_only"] == 82
    funnel = manifest["diagnostics"]["eligibility_funnel"]
    assert funnel["funnel_reconciliation"] == "PASS"
    targets = manifest["diagnostics"]["target_quality_train_val"]
    assert targets["scope"] == "TRAIN_VALIDATION_ONLY"
    assert targets["TEST_OUTCOME_USED"] is False
    assert targets["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert all(item["rows"] == 326 for item in targets["horizons"].values())
    pairing = manifest["diagnostics"]["exact_vs_date_pairing"]
    assert pairing["EXACT_DATE_PAIRING_STATUS"].startswith("FAIL_CLOSED")
    priority = manifest["diagnostics"]["priority_report"]
    assert priority["NEXT_DATA_PRIORITY"] == "MARKET_HISTORY_WARMUP_RECOVERY"


def test_documentation_states_hard_guards() -> None:
    text = (Path(__file__).parents[2] / "docs" / "exact-event-data-diagnostics-v1.md").read_text(
        encoding="utf-8"
    )
    assert "DIAGNOSTIC_ONLY=true" in text
    assert "MODEL_TRAINING_PERFORMED=false" in text
    assert "TEST_OUTCOME_USED=false" in text
    assert FUTURE_EVENT_HOLDOUT_START.isoformat() in text
    assert "No new model" in text


def _valid_dataset_manifest() -> dict[str, object]:
    return {
        "dataset_version": "exact-event-market-dataset-v2",
        "exact_dataset_sha": EXPECTED_DATASET_SHA,
        "EVENT_MARKET_LEAKAGE_CHECK": "PASS",
        "holdout_guard": "PASS",
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "rules_changed": False,
        "qwen_changed": False,
        "qwen_run": False,
    }


def _valid_holdout() -> dict[str, object]:
    return {
        "FUTURE_EVENT_HOLDOUT_START": "2026-08-11",
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "holdout_guard": "PASS",
        "outcome_fields_exported_for_future": 0,
    }
