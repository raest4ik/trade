from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from apps.cli.recover_exact_event_market_history import build_parser
from src.exact_event_warmup_recovery.application import run_warmup_recovery
from src.exact_event_warmup_recovery.domain import (
    ARTIFACT_VERSION,
    EXPECTED_INPUT_DATASET_SHA,
    FUTURE_EVENT_HOLDOUT_START,
    REQUIRED_LOOKBACK_MINUTES,
    acquisition_dates,
    earliest_required_timestamp,
    recovery_safety_flags,
    require_input_manifest,
    sha256_payload,
)


def test_recovery_safety_flags_forbid_model_test_future_and_trading() -> None:
    flags = recovery_safety_flags()
    assert flags["DATA_RECOVERY_ONLY"] is True
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert flags["RULES_V3_CHANGED"] is False
    assert flags["QWEN_CHANGED"] is False
    assert flags["NLP_TUNING_PERFORMED"] is False
    assert flags["CONFIRMED_SIGNAL"] is False
    assert flags["BACKTEST_APPROVED"] is False
    assert flags["PAPER_TRADING_APPROVED"] is False
    assert flags["REAL_TRADING_APPROVED"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False
    assert flags["SANDBOX_ORDER_SUBMISSION_ALLOWED"] is False


def test_required_lookback_and_bounded_acquisition_are_deterministic() -> None:
    published = datetime(2026, 7, 1, 7, 45, tzinfo=UTC)
    assert REQUIRED_LOOKBACK_MINUTES == 60
    assert earliest_required_timestamp(published) == datetime(2026, 7, 1, 6, 45, tzinfo=UTC)
    dates = acquisition_dates(published, max_history_days=7)
    assert len(dates) == 8
    assert dates[0].isoformat() == "2026-07-01"
    assert dates[-1].isoformat() == "2026-06-24"


def test_input_manifest_fails_closed_on_dataset_or_frozen_contract_change() -> None:
    manifest = _valid_manifest()
    require_input_manifest(manifest)
    with pytest.raises(ValueError, match="INPUT_DATASET_SHA_MISMATCH"):
        require_input_manifest({**manifest, "exact_dataset_sha": "0" * 64})
    with pytest.raises(ValueError, match="INPUT_FUTURE_HOLDOUT_OBSERVED"):
        require_input_manifest({**manifest, "FUTURE_EVENT_HOLDOUT_OBSERVED": True})
    with pytest.raises(ValueError, match="RULES_CHANGED_NOT_FROZEN"):
        require_input_manifest({**manifest, "rules_changed": True})


def test_cli_defaults_to_warmup_recovery_artifact() -> None:
    args = build_parser().parse_args([])
    assert args.dataset_dir == "artifacts/exact-event-market-dataset-v2"
    assert args.v1_dataset_dir == "artifacts/exact-event-market-dataset-v1"
    assert args.baseline_dir == "artifacts/exact-event-predictive-baseline-v1"
    assert args.output_dir == "artifacts/exact-event-market-history-warmup-recovery-v1"


def test_real_cache_recovery_reconciles_and_preserves_existing_rows(tmp_path: Path) -> None:
    repo = Path(__file__).parents[2]
    manifest = run_warmup_recovery(
        repo / "artifacts" / "exact-event-market-dataset-v2",
        tmp_path / ARTIFACT_VERSION,
        base_main_sha="6f56f592f4dcb306424620c4a3f12a8d9412457d",
        git_sha="c" * 40,
        baseline_root=repo / "artifacts" / "exact-event-predictive-baseline-v1",
        v1_dataset_root=repo / "artifacts" / "exact-event-market-dataset-v1",
        created_at=datetime(2026, 8, 18, tzinfo=UTC),
    )
    assert manifest["INPUT_DATASET_SHA"] == EXPECTED_INPUT_DATASET_SHA
    assert manifest["WARMUP_LOST_BEFORE"] == 157
    assert manifest["WARMUP_RECOVERED"] == 156
    assert manifest["WARMUP_REMAINING"] == 1
    assert manifest["FEATURE_READY_BEFORE"] == 408
    assert manifest["FEATURE_READY_AFTER"] == 564
    assert manifest["FEATURE_READY_DELTA"] == 156
    assert manifest["TRAIN_VAL_WARMUP_LOST_BEFORE"] == 137
    assert manifest["TRAIN_VAL_WARMUP_RECOVERED"] == 136
    assert manifest["TRAIN_VAL_WARMUP_REMAINING"] == 1
    assert manifest["WARMUP_RECONCILIATION"] == "PASS"
    assert manifest["EXISTING_FEATURE_ROWS_PRESERVED"] == "PASS"
    assert manifest["LEAKAGE_CHECK"] == "PASS"
    assert manifest["TINVEST_SOURCE_ONLY"] is True
    assert manifest["MOEX_SUBSTITUTION_USED"] is False
    assert manifest["FORWARD_FILL_USED"] is False
    assert manifest["safety"]["DATA_RECOVERY_ONLY"] is True
    assert manifest["safety"]["TEST_OUTCOME_USED"] is False
    assert manifest["safety"]["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert manifest["targets_jsonl_read_as_structured_data"] is False
    assert manifest["RECOVERED_BY_TICKER"] == {
        "MGNT": 99,
        "T": 16,
        "VKCO": 3,
        "X5": 35,
        "YDEX": 3,
    }
    assert manifest["REMAINING_BY_REASON"] == {"INSUFFICIENT_REQUIRED_LOOKBACK": 1}
    assert len(manifest["ARTIFACT_SHA"]) == 64
    assert manifest["ARTIFACT_SHA"] == sha256_payload({**manifest, "ARTIFACT_SHA": None})


def test_documentation_states_recovery_safety_boundaries() -> None:
    text = (
        Path(__file__).parents[2] / "docs" / "exact-event-market-history-warmup-recovery-v1.md"
    ).read_text(encoding="utf-8")
    assert "DATA_RECOVERY_ONLY=true" in text
    assert "MODEL_TRAINING_PERFORMED=false" in text
    assert "TEST_OUTCOME_USED=false" in text
    assert "no MOEX substitution" in text
    assert "no forward-fill" in text
    assert "does not train a" in text
    assert FUTURE_EVENT_HOLDOUT_START.isoformat()[:4] == "2026"


def _valid_manifest() -> dict[str, object]:
    return {
        "dataset_version": "exact-event-market-dataset-v2",
        "exact_dataset_sha": EXPECTED_INPUT_DATASET_SHA,
        "EVENT_MARKET_LEAKAGE_CHECK": "PASS",
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "rules_changed": False,
        "qwen_changed": False,
        "qwen_run": False,
    }
