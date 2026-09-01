from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import src.exact_dataset_readiness_audit.ml_v2 as ml_v2
from apps.cli.audit_ml_v2_readiness import build_parser
from src.exact_dataset_readiness_audit.domain import (
    CANONICAL_ML_V2_COHORT,
    ML_V2_ARTIFACT_VERSION,
    MlV2ReadinessDecision,
    artifact_sha,
    sha256_payload,
)
from src.exact_dataset_readiness_audit.ml_v2 import (
    build_canonical_gate_criteria,
    concentration_for,
    leakage_audit,
    readiness_decision,
    run_ml_v2_readiness_audit,
    semantic_summary,
    target_coverage,
)
from src.historical_exact_semantic_backfill.domain import artifact_sha as backfill_artifact_sha


def test_cli_defaults_to_ml_v2_readiness_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "8" * 40])

    assert args.output_dir == f"artifacts/{ML_V2_ARTIFACT_VERSION}"
    assert args.base_main_sha == "8" * 40


def test_gate_boundary_values_are_ready() -> None:
    rows = [
        _audit_row(f"event-{index}", f"T{index % 10}", unknown=index < 250) for index in range(500)
    ]
    targets = [_target_row(str(row["event_id"])) for row in rows]
    criteria = build_canonical_gate_criteria(
        issuer_feature_ready=rows,
        semantic_summary=semantic_summary(rows, rows),
        ticker_concentration={"canonical_issuer": concentration_for(rows, "ticker")},
        source_concentration={
            "canonical_issuer": {
                **_source_concentration_payload(rows),
                "source_family_hhi": "0.500000",
            }
        },
        target_coverage=target_coverage(rows, rows, targets),
        leakage={"LEAKAGE_AUDIT": "PASS", "FUTURE_OUTCOMES_READ": 0, "FUTURE_TARGETS_READ": 0},
    )

    decision, blocker, secondary = readiness_decision(criteria)

    assert decision == MlV2ReadinessDecision.READY_FOR_CONTROLLED_ML_V2
    assert blocker is None
    assert secondary == []


def test_future_holdout_targets_trigger_integrity_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _write_input_artifact(
        tmp_path / "input",
        [
            _event("issuer-1", "AAA"),
            _event("future-1", "BBB", published_at="2026-08-11T10:00:00+00:00"),
        ],
        features=["issuer-1", "future-1"],
        targets=["issuer-1", "future-1"],
    )
    _patch_expected_input(monkeypatch, input_root)
    old_root = _write_old_baseline(tmp_path / "old")

    manifest = run_ml_v2_readiness_audit(
        input_root=input_root,
        old_baseline_root=old_root,
        output_root=tmp_path / "out",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    leakage = _read_json(tmp_path / "out" / "leakage-audit.json")
    assert leakage["LEAKAGE_AUDIT"] == "FAIL"
    assert manifest["FINAL_READINESS_DECISION"] == MlV2ReadinessDecision.DATASET_INTEGRITY_FAILURE
    assert manifest["FUTURE_OUTCOMES_READ"] == 0
    assert manifest["FUTURE_TARGETS_READ"] == 0


def test_observed_old_test_protection_requires_observed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _write_input_artifact(tmp_path / "input", [_event("issuer-1", "AAA")])
    _patch_expected_input(monkeypatch, input_root)
    old_root = _write_old_baseline(tmp_path / "old", test_status="BLIND_LOCKED_NOT_EVALUATED")

    with pytest.raises(ValueError, match="OLD_BASELINE_TEST_STATUS_UNEXPECTED"):
        run_ml_v2_readiness_audit(
            input_root=input_root,
            old_baseline_root=old_root,
            output_root=tmp_path / "out",
            base_main_sha="8" * 40,
            git_sha="9" * 40,
        )


def test_leakage_detection_fails_post_event_features() -> None:
    events = [_event("issuer-1", "AAA")]
    features = [_feature("issuer-1", cutoff="2026-07-01T10:01:00+00:00")]
    targets = [_target_row("issuer-1")]
    state = {"ml_v2_policy_status": "OBSERVED_DO_NOT_TUNE_ON"}

    leakage = leakage_audit(events, features, targets, state)

    assert leakage["LEAKAGE_AUDIT"] == "FAIL"
    assert leakage["violations"][0]["violation"] == "FEATURE_TIMESTAMP_AFTER_PUBLICATION"

    clean = leakage_audit(events, [_feature("issuer-1")], targets, state)
    assert clean["LEAKAGE_AUDIT"] == "PASS"

    target_like = _feature("issuer-1")
    target_like["market_features"]["target_return_15m"] = "0.01"
    leaked = leakage_audit(events, [target_like], targets, state)
    assert leaked["LEAKAGE_AUDIT"] == "FAIL"
    assert leaked["violations"][0]["violation"] == "POST_EVENT_RETURN_FEATURE_COLUMNS"


def test_concentration_calculations() -> None:
    rows = [_audit_row("a", "AAA"), _audit_row("b", "AAA"), _audit_row("c", "BBB")]

    concentration = concentration_for(rows, "ticker")

    assert concentration["top_1_share"] == "0.666667"
    assert concentration["top_3_share"] == "1.000000"
    assert concentration["hhi"] == "0.555556"
    assert concentration["effective_count"] == "1.800000"


def test_unknown_rate_and_target_coverage() -> None:
    rows = [_audit_row("a", "AAA", unknown=True), _audit_row("b", "BBB")]
    targets = [_target_row("a", missing=("60m",)), _target_row("b")]

    semantic = semantic_summary(rows, rows)
    coverage = target_coverage(rows, rows, targets)

    assert semantic["canonical_issuer"]["unknown_rate"] == "0.500000"
    assert semantic["unknown_rate_by_ticker"] == {"AAA": "1.000000", "BBB": "0.000000"}
    assert coverage["canonical_issuer"]["horizons"]["15m"]["coverage"] == "1.000000"
    assert coverage["canonical_issuer"]["horizons"]["60m"]["coverage"] == "0.500000"


def test_artifact_sha_and_replay_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _write_input_artifact(
        tmp_path / "input",
        [_event("issuer-1", "AAA"), _event("exchange-1", "BBB", source_family="MOEX_RISK")],
    )
    _patch_expected_input(monkeypatch, input_root)
    old_root = _write_old_baseline(tmp_path / "old")

    first = run_ml_v2_readiness_audit(
        input_root=input_root,
        old_baseline_root=old_root,
        output_root=tmp_path / "first",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    second = run_ml_v2_readiness_audit(
        input_root=input_root,
        old_baseline_root=old_root,
        output_root=tmp_path / "second",
        base_main_sha="8" * 40,
        git_sha="0" * 40,
        created_at=datetime(2026, 9, 2, tzinfo=UTC),
    )

    assert first["ARTIFACT_SHA"] == artifact_sha(first)
    assert first["ARTIFACT_SHA"] == second["ARTIFACT_SHA"]
    assert first["DATASET_FUNNEL_SHA"] == second["DATASET_FUNNEL_SHA"]
    assert (
        _read_json(tmp_path / "first" / "manifest.json")["CANONICAL_COHORT"]
        == CANONICAL_ML_V2_COHORT
    )


def test_invalid_input_artifact_sha_is_rejected(tmp_path: Path) -> None:
    input_root = _write_input_artifact(tmp_path / "input", [_event("issuer-1", "AAA")])
    old_root = _write_old_baseline(tmp_path / "old")

    with pytest.raises(ValueError, match="INPUT_ARTIFACT_SHA_MISMATCH"):
        run_ml_v2_readiness_audit(
            input_root=input_root,
            old_baseline_root=old_root,
            output_root=tmp_path / "out",
            base_main_sha="8" * 40,
            git_sha="9" * 40,
        )


def test_changed_rules_fingerprint_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _write_input_artifact(tmp_path / "input", [_event("issuer-1", "AAA")])
    _patch_expected_input(monkeypatch, input_root)
    old_root = _write_old_baseline(tmp_path / "old")
    monkeypatch.setattr(ml_v2, "rules_v3_fingerprint", lambda: "changed")

    with pytest.raises(ValueError, match="RULES_V3_FINGERPRINT_CHANGED"):
        run_ml_v2_readiness_audit(
            input_root=input_root,
            old_baseline_root=old_root,
            output_root=tmp_path / "out",
            base_main_sha="8" * 40,
            git_sha="9" * 40,
        )


def test_conflicting_exchange_cohort_is_not_counted_as_issuer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _write_input_artifact(
        tmp_path / "input",
        [
            _event("issuer-1", "AAA", source_family="ROSNEFT_PRESS_RELEASES_RSS"),
            _event(
                "exchange-1", "BBB", source_family="MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"
            ),
        ],
    )
    _patch_expected_input(monkeypatch, input_root)
    old_root = _write_old_baseline(tmp_path / "old")

    manifest = run_ml_v2_readiness_audit(
        input_root=input_root,
        old_baseline_root=old_root,
        output_root=tmp_path / "out",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
    )

    assert manifest["ISSUER_ORIGINATED_FEATURE_READY_EVENTS"] == 1
    assert manifest["EXCHANGE_ORIGINATED_FEATURE_READY_EVENTS"] == 1


def test_audit_module_has_no_model_training_dependency() -> None:
    source = Path(ml_v2.__file__).read_text(encoding="utf-8")

    assert "sklearn" not in source
    assert ".fit(" not in source
    assert ".predict(" not in source
    assert "train_event_predictive_baseline" not in source


def _source_concentration_payload(rows: list[dict[str, Any]]) -> dict[str, str]:
    concentration = concentration_for(rows, "source_family")
    return {
        "top_source_family_share": concentration["top_1_share"],
        "source_family_hhi": concentration["hhi"],
    }


def _audit_row(event_id: str, ticker: str, *, unknown: bool = False) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "ticker": ticker,
        "issuer": ticker,
        "source_id": f"{ticker}_SOURCE",
        "source_family": _synthetic_source_family(event_id),
        "source_item_id": event_id,
        "published_at_utc": "2026-07-01T10:00:00+00:00",
        "event_origin": "ISSUER_ORIGINATED",
        "feature_ready": True,
        "reaction_ready": True,
        "market_eligible": True,
        "primary_event_type": "UNKNOWN" if unknown else "DIVIDEND",
        "event_count": 0 if unknown else 1,
        "fact_count": 0 if unknown else 1,
        "semantic_valid": True,
        "semantic_features_sha": sha256_payload(
            {
                "event_count": 0 if unknown else 1,
                "fact_count": 0 if unknown else 1,
                "primary_event_type": "UNKNOWN" if unknown else "DIVIDEND",
            }
        ),
    }


def _synthetic_source_family(event_id: str) -> str:
    suffix = event_id.rsplit("-", 1)[-1]
    if not suffix.isdigit():
        return "SOURCE_A"
    return "SOURCE_A" if int(suffix) % 2 == 0 else "SOURCE_B"


def _write_input_artifact(
    root: Path,
    events: list[dict[str, Any]],
    *,
    features: list[str] | None = None,
    targets: list[str] | None = None,
) -> Path:
    feature_ids = features or [
        str(cast("dict[str, Any]", event["metadata"])["event_id"]) for event in events
    ]
    target_ids = targets or feature_ids
    _write_jsonl(root / "events.jsonl", events)
    _write_jsonl(root / "features.jsonl", [_feature(event_id) for event_id in feature_ids])
    _write_jsonl(root / "targets.jsonl", [_target_row(event_id) for event_id in target_ids])
    _write_jsonl(
        root / "semantic-material-provenance.jsonl",
        [
            {
                "event_id": str(cast("dict[str, Any]", event["metadata"])["event_id"]),
                "publication_material_sha": sha256_payload(event["metadata"]),
            }
            for event in events
        ],
    )
    _write_jsonl(
        root / "semantic-extraction-results.jsonl",
        [
            {
                "event_id": str(cast("dict[str, Any]", event["metadata"])["event_id"]),
                "semantic_features_sha": sha256_payload(event["event_features"]),
            }
            for event in events
        ],
    )
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": "historical-exact-semantic-backfill-v1",
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
    }
    manifest["ARTIFACT_SHA"] = backfill_artifact_sha(manifest)
    _write_json(root / "manifest.json", manifest)
    return root


def _write_old_baseline(
    root: Path, *, test_status: str = "OBSERVED_AFTER_EXACT_BASELINE_V1"
) -> Path:
    _write_json(
        root / "test-evaluation-state.json",
        {
            "TEST_CONFIG_LOCKED": "YES",
            "TEST_EVALUATION_COUNT_PRIMARY": 1,
            "TEST_STATUS": test_status,
            "artifact_sha": "baseline-sha",
            "locked_config_sha": "config-sha",
        },
    )
    _write_json(
        root / "15m-split-manifest.json",
        {
            "horizon": "15m",
            "protocol": "TEST_PROTOCOL",
            "split_sha": "split-sha",
            "counts": {"TRAIN": 1, "VALIDATION": 1, "TEST": 1},
            "date_ranges": {
                "TRAIN": {"from": "2025-01-01", "to": "2025-12-31"},
                "VALIDATION": {"from": "2026-01-01", "to": "2026-06-30"},
                "TEST": {"from": "2026-07-01", "to": "2026-08-10"},
            },
        },
    )
    return root


def _event(
    event_id: str,
    ticker: str,
    *,
    source_family: str = "ISSUER_OFFICIAL_RSS_EXACT_LIVE_V1",
    published_at: str = "2026-07-01T10:00:00+00:00",
) -> dict[str, Any]:
    return {
        "metadata": {
            "event_id": event_id,
            "ticker": ticker,
            "issuer": f"{ticker} Issuer",
            "instrument_uid": f"{ticker}-uid",
            "source_code": source_family,
            "source_family": source_family,
            "source_item_id": f"{source_family}:{event_id}",
            "publication_timestamp_utc": published_at,
            "timestamp_quality": "EXACT",
        },
        "event_features": {
            "primary_event_type": "DIVIDEND",
            "event_count": 1,
            "fact_count": 1,
        },
        "pre_event_market_features": {
            "feature_cutoff": published_at,
            "post_event_values_in_features": False,
            "pre_return_15m": "0.01",
            "imoex_pre_return_15m": "0.001",
        },
        "target_availability": {
            "feature_ready": True,
            "reaction_ready": True,
            "research_outcomes_visible": True,
        },
    }


def _feature(event_id: str, *, cutoff: str = "2026-07-01T10:00:00+00:00") -> dict[str, Any]:
    return {
        "event_id": event_id,
        "feature_cutoff": cutoff,
        "event_features": {
            "primary_event_type": "DIVIDEND",
            "event_count": 1,
            "fact_count": 1,
        },
        "market_features": {
            "feature_cutoff": cutoff,
            "post_event_values_in_features": False,
            "pre_return_15m": "0.01",
            "imoex_pre_return_15m": "0.001",
        },
    }


def _target_row(event_id: str, *, missing: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "horizons": {
            horizon: {
                "available": horizon not in missing,
                "abnormal_return": None if horizon in missing else "0.001",
            }
            for horizon in ("1m", "5m", "15m", "30m", "60m")
        },
    }


def _patch_expected_input(monkeypatch: pytest.MonkeyPatch, input_root: Path) -> None:
    monkeypatch.setattr(
        ml_v2,
        "EXPECTED_INPUT_ARTIFACT_SHA",
        _read_json(input_root / "manifest.json")["ARTIFACT_SHA"],
    )


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
