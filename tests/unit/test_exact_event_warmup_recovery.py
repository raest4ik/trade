from __future__ import annotations

import json
import os
from collections.abc import Sequence
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


def test_synthetic_cache_recovery_reconciles_and_preserves_existing_rows(
    tmp_path: Path,
) -> None:
    dataset_root, baseline_root, v1_dataset_root = _write_warmup_fixture(tmp_path)
    manifest = run_warmup_recovery(
        dataset_root,
        tmp_path / ARTIFACT_VERSION,
        base_main_sha="6f56f592f4dcb306424620c4a3f12a8d9412457d",
        git_sha="c" * 40,
        baseline_root=baseline_root,
        v1_dataset_root=v1_dataset_root,
        created_at=datetime(2026, 8, 18, tzinfo=UTC),
        expected_accounting={"events_before": 3, "features_before": 1, "affected": 2},
    )
    assert manifest["INPUT_DATASET_SHA"] == EXPECTED_INPUT_DATASET_SHA
    assert manifest["WARMUP_LOST_BEFORE"] == 2
    assert manifest["WARMUP_RECOVERED"] == 1
    assert manifest["WARMUP_REMAINING"] == 1
    assert manifest["FEATURE_READY_BEFORE"] == 1
    assert manifest["FEATURE_READY_AFTER"] == 2
    assert manifest["FEATURE_READY_DELTA"] == 1
    assert manifest["TRAIN_VAL_WARMUP_LOST_BEFORE"] == 2
    assert manifest["TRAIN_VAL_WARMUP_RECOVERED"] == 1
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
    assert manifest["RECOVERED_BY_TICKER"] == {"RCVR": 1}
    assert manifest["REMAINING_BY_REASON"] == {"INSUFFICIENT_REQUIRED_LOOKBACK": 1}
    assert len(manifest["ARTIFACT_SHA"]) == 64
    assert manifest["ARTIFACT_SHA"] == sha256_payload({**manifest, "ARTIFACT_SHA": None})


def test_local_real_cache_recovery_scientific_result_is_unchanged(tmp_path: Path) -> None:
    if os.environ.get("RUN_LOCAL_REAL_ARTIFACT_SMOKE") != "1":
        pytest.skip("optional local production artifact smoke test")
    repo = Path(__file__).parents[2]
    dataset_root = repo / "artifacts" / "exact-event-market-dataset-v2"
    baseline_root = repo / "artifacts" / "exact-event-predictive-baseline-v1"
    v1_dataset_root = repo / "artifacts" / "exact-event-market-dataset-v1"
    if not (dataset_root / "manifest.json").exists():
        pytest.skip("optional local production artifact smoke test")
    manifest = run_warmup_recovery(
        dataset_root,
        tmp_path / ARTIFACT_VERSION,
        base_main_sha="6f56f592f4dcb306424620c4a3f12a8d9412457d",
        git_sha="c" * 40,
        baseline_root=baseline_root,
        v1_dataset_root=v1_dataset_root,
        created_at=datetime(2026, 8, 18, tzinfo=UTC),
    )
    assert manifest["OUTPUT_DATASET_SHA"] == (
        "669aa6e8b11763131f3a940d669e446537a110066da22e7710649cdb2eaba6ff"
    )
    assert len(manifest["ARTIFACT_SHA"]) == 64
    assert manifest["ARTIFACT_SHA"] == sha256_payload({**manifest, "ARTIFACT_SHA": None})
    assert manifest["WARMUP_RECOVERED"] == 156
    assert manifest["WARMUP_REMAINING"] == 1
    assert manifest["FEATURE_READY_BEFORE"] == 408
    assert manifest["FEATURE_READY_AFTER"] == 564
    assert manifest["EXISTING_FEATURE_ROWS_PRESERVED"] == "PASS"
    assert manifest["LEAKAGE_CHECK"] == "PASS"


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


def _write_warmup_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "dataset"
    baseline_root = tmp_path / "baseline"
    v1_dataset_root = tmp_path / "dataset-v1"
    dataset_root.mkdir()
    baseline_root.mkdir()
    v1_dataset_root.mkdir()
    events = [
        _event("warmup-ready", "READY", "2026-07-01T10:00:00+00:00", feature_ready=True),
        _event("warmup-recover", "RCVR", "2026-07-01T10:00:00+00:00"),
        _event("warmup-remaining", "MISS", "2026-07-01T09:30:00+00:00"),
    ]
    features = [
        {
            "event_id": "warmup-ready",
            "feature_cutoff": "2026-07-01T10:00:00+00:00",
            "event_features": events[0]["event_features"],
            "market_features": _complete_market_features("2026-07-01T10:00:00+00:00"),
        }
    ]
    assignments = [
        {"event_id": "warmup-ready", "split": "TRAIN"},
        {"event_id": "warmup-recover", "split": "VALIDATION"},
        {"event_id": "warmup-remaining", "split": "TRAIN"},
    ]
    _write_json(dataset_root / "manifest.json", _valid_manifest())
    _write_jsonl(dataset_root / "events.jsonl", events)
    _write_jsonl(dataset_root / "features.jsonl", features)
    _write_jsonl(dataset_root / "targets.jsonl", [])
    _write_json(
        baseline_root / "15m-split-manifest.json",
        {"assignments": assignments},
    )
    _write_candle_cache(dataset_root / "raw-minute-cache", "RCVR", "uid-RCVR", _complete_times())
    _write_candle_cache(dataset_root / "raw-minute-cache", "MISS", "uid-MISS", _short_times())
    _write_candle_cache(dataset_root / "raw-minute-cache", "IMOEX", "uid-IMOEX", _complete_times())
    return dataset_root, baseline_root, v1_dataset_root


def _event(
    event_id: str, ticker: str, published_at: str, *, feature_ready: bool = False
) -> dict[str, object]:
    published = datetime.fromisoformat(published_at)
    return {
        "event_features": {
            "event_count": 1,
            "fact_count": 1,
            "primary_event_type": "DIVIDEND",
        },
        "metadata": {
            "event_id": event_id,
            "event_cluster_id": f"cluster-{event_id}",
            "future_holdout": False,
            "instrument_uid": f"uid-{ticker}",
            "issuer": f"Issuer {ticker}",
            "publication_date": published.date().isoformat(),
            "publication_timestamp_utc": published.isoformat(),
            "session_state": "DURING_MAIN_SESSION",
            "source_code": f"{ticker}_SYNTHETIC_EXACT",
            "ticker": ticker,
        },
        "pre_event_market_features": (
            _complete_market_features(published.isoformat())
            if feature_ready
            else _blocked_features()
        ),
        "quality": {
            "feature_cutoff": published.isoformat(),
            "no_forward_fill": True,
            "no_interpolation": True,
            "no_source_mixing": True,
            "reaction_starts_after_or_at_publication": True,
            "security_benchmark_same_window": True,
        },
        "target_availability": {
            "feature_ready": feature_ready,
            "missing_reason": None,
            "reaction_ready": True,
            "research_outcomes_visible": False,
            "status": "REACTION_READY",
        },
    }


def _complete_market_features(cutoff: str) -> dict[str, object]:
    return {
        "feature_cutoff": cutoff,
        "post_event_values_in_features": False,
        "pre_return_5m": "0.001",
        "pre_return_15m": "0.002",
        "pre_return_30m": "0.003",
        "pre_return_60m": "0.004",
        "imoex_pre_return_5m": "0.0001",
        "imoex_pre_return_15m": "0.0002",
        "imoex_pre_return_30m": "0.0003",
        "imoex_pre_return_60m": "0.0004",
    }


def _blocked_features() -> dict[str, object]:
    return {
        "feature_cutoff": "2026-07-01T10:00:00+00:00",
        "post_event_values_in_features": False,
        "pre_return_5m": None,
        "pre_return_15m": None,
        "pre_return_30m": None,
        "pre_return_60m": None,
        "imoex_pre_return_5m": None,
        "imoex_pre_return_15m": None,
        "imoex_pre_return_30m": None,
        "imoex_pre_return_60m": None,
    }


def _complete_times() -> tuple[str, ...]:
    return (
        "2026-07-01T08:59:00+00:00",
        "2026-07-01T09:29:00+00:00",
        "2026-07-01T09:30:00+00:00",
        "2026-07-01T09:44:00+00:00",
        "2026-07-01T09:54:00+00:00",
        "2026-07-01T09:59:00+00:00",
        "2026-07-01T10:00:00+00:00",
    )


def _short_times() -> tuple[str, ...]:
    return (
        "2026-07-01T09:29:00+00:00",
        "2026-07-01T09:30:00+00:00",
    )


def _write_candle_cache(
    root: Path, ticker: str, instrument_uid: str, begin_times: tuple[str, ...]
) -> None:
    ticker_root = root / ticker
    ticker_root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index, begin_text in enumerate(begin_times, start=1):
        begin = datetime.fromisoformat(begin_text)
        end = begin.replace(minute=begin.minute + 1) if begin.minute < 59 else begin
        if begin.minute == 59:
            end = begin.replace(hour=begin.hour + 1, minute=0)
        price = 100 + index
        rows.append(
            {
                "instrument_uid": instrument_uid,
                "begin_at": begin.isoformat(),
                "end_at": end.isoformat(),
                "open": str(price),
                "high": str(price),
                "low": str(price),
                "close": str(price),
                "volume": index,
                "is_complete": True,
                "source": "TINVEST_API",
            }
        )
    _write_jsonl(ticker_root / "2026-07-01-day.jsonl", rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
