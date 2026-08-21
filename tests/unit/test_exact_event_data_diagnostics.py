from __future__ import annotations

import json
from collections.abc import Sequence
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
    dataset_root, baseline_root = _write_diagnostic_fixture(tmp_path)
    manifest = run_exact_event_data_diagnostics(
        dataset_root,
        baseline_root,
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
    assert manifest["counts"]["train"] == 2
    assert manifest["counts"]["validation"] == 1
    assert manifest["counts"]["train_validation"] == 3
    assert manifest["counts"]["test_metadata_only"] == 1
    funnel = manifest["diagnostics"]["eligibility_funnel"]
    assert funnel["funnel_reconciliation"] == "PASS"
    targets = manifest["diagnostics"]["target_quality_train_val"]
    assert targets["scope"] == "TRAIN_VALIDATION_ONLY"
    assert targets["TEST_OUTCOME_USED"] is False
    assert targets["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert all(item["rows"] == 3 for item in targets["horizons"].values())
    pairing = manifest["diagnostics"]["exact_vs_date_pairing"]
    assert pairing["EXACT_DATE_PAIRING_STATUS"].startswith("FAIL_CLOSED")
    priority = manifest["diagnostics"]["priority_report"]
    assert priority["NEXT_DATA_PRIORITY"] == "MARKET_HISTORY_WARMUP_RECOVERY"
    warmup = manifest["diagnostics"]["warmup_loss"]
    assert warmup["warmup_lost_total_metadata_only"] == 1
    assert warmup["WARMUP_RECOVERABLE_CANDIDATE"] is True


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


def _write_diagnostic_fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "dataset"
    baseline_root = tmp_path / "baseline"
    dataset_root.mkdir()
    baseline_root.mkdir()
    events = [
        _event("diag-train-ready", "AAA", "2026-07-01T10:00:00+00:00", feature_ready=True),
        _event("diag-val-ready", "BBB", "2026-07-02T10:30:00+00:00", feature_ready=True),
        _event("diag-train-warmup", "AAA", "2026-07-03T11:00:00+00:00"),
        _event("diag-test-ready", "CCC", "2026-07-04T11:30:00+00:00", feature_ready=True),
    ]
    features = [
        _feature("diag-train-ready"),
        _feature("diag-val-ready"),
        _feature("diag-test-ready"),
    ]
    targets = [_target("diag-train-ready"), _target("diag-val-ready"), _target("diag-train-warmup")]
    clusters: list[dict[str, object]] = [
        {"event_cluster_id": "cluster-1", "event_id": "diag-train-ready"},
        {"event_cluster_id": "cluster-2", "event_id": "diag-val-ready"},
        {"event_cluster_id": "cluster-3", "event_id": "diag-train-warmup"},
        {"event_cluster_id": "cluster-4", "event_id": "diag-test-ready"},
    ]
    assignments = [
        {"event_id": "diag-train-ready", "split": "TRAIN"},
        {"event_id": "diag-val-ready", "split": "VALIDATION"},
        {"event_id": "diag-train-warmup", "split": "TRAIN"},
        {"event_id": "diag-test-ready", "split": "TEST"},
    ]
    _write_json(dataset_root / "manifest.json", _valid_dataset_manifest())
    _write_json(dataset_root / "future-holdout-status.json", _valid_holdout())
    _write_jsonl(dataset_root / "events.jsonl", events)
    _write_jsonl(dataset_root / "features.jsonl", features)
    _write_jsonl(dataset_root / "targets.jsonl", targets)
    _write_jsonl(dataset_root / "clusters.jsonl", clusters)
    _write_json(
        baseline_root / "manifest.json",
        {
            "model_version": "exact-event-predictive-baseline-v1",
            "dataset_sha": EXPECTED_DATASET_SHA,
            "artifact_sha": "1" * 64,
            "FUTURE_EVENT_HOLDOUT_USED": False,
            "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
            "TEST_STATUS": "OBSERVED_AFTER_EXACT_BASELINE_V1",
        },
    )
    _write_json(
        baseline_root / "15m-split-manifest.json",
        {
            "split_sha": "2" * 64,
            "target_outcomes_inspected_before_lock": False,
            "cluster_integrity": "PASS",
            "leakage_check": "PASS",
            "assignments": assignments,
        },
    )
    return dataset_root, baseline_root


def _event(
    event_id: str, ticker: str, published_at: str, *, feature_ready: bool = False
) -> dict[str, object]:
    published = datetime.fromisoformat(published_at)
    return {
        "event_features": {
            "event_count": 1,
            "fact_count": 1,
            "primary_event_type": "DIVIDEND" if ticker != "BBB" else "UNKNOWN",
        },
        "metadata": {
            "canonical_url": f"https://example.test/{event_id}",
            "event_cluster_id": f"cluster-{event_id}",
            "event_id": event_id,
            "future_holdout": False,
            "instrument_uid": f"uid-{ticker}",
            "issuer": f"Issuer {ticker}",
            "market_alignment_version": "tinvest-exact-minute-alignment-v1",
            "provenance": "SYNTHETIC",
            "publication_date": published.date().isoformat(),
            "publication_time": published.time().isoformat(),
            "publication_timestamp_raw": published_at,
            "publication_timestamp_utc": published_at,
            "publication_timezone": "UTC",
            "reaction_family": "EXACT_INTRADAY",
            "session_state": "DURING_MAIN_SESSION",
            "source_code": f"{ticker}_SYNTHETIC_EXACT",
            "source_item_id": event_id,
            "storage_policy": "METADATA_TITLE_HASH_ONLY",
            "ticker": ticker,
            "timestamp_quality": "EXACT",
            "timestamp_source_field": "synthetic fixed timestamp",
            "title_hash": event_id,
        },
        "pre_event_market_features": _market_features() if feature_ready else _blocked_features(),
        "quality": {
            "feature_cutoff": published_at,
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


def _feature(event_id: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_features": {
            "event_count": 1,
            "fact_count": 1,
            "primary_event_type": "DIVIDEND",
        },
        "market_features": _market_features(),
    }


def _market_features() -> dict[str, object]:
    return {
        "feature_cutoff": "2026-07-01T10:00:00+00:00",
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
    features = _market_features()
    features["pre_return_60m"] = None
    features["imoex_pre_return_60m"] = None
    return features


def _target(event_id: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "horizons": {
            horizon: {
                "abnormal_return": "0.001",
                "window_begin_at": "2026-07-01T10:01:00+00:00",
                "window_end_at": "2026-07-01T10:02:00+00:00",
                "security_observed_at": "2026-07-01T10:02:00+00:00",
                "benchmark_observed_at": "2026-07-01T10:02:00+00:00",
            }
            for horizon in ("1m", "5m", "15m", "30m", "60m")
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
