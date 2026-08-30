from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import src.exact_dataset_readiness_audit.application as audit_app
from apps.cli.audit_exact_dataset_readiness import build_parser
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_dataset_readiness_audit.application import run_exact_dataset_readiness_audit
from src.exact_dataset_readiness_audit.domain import (
    ARTIFACT_VERSION,
    DEFAULT_INPUT_ARTIFACT_ROOT,
    EXPECTED_RULES_V3_FINGERPRINT,
    EventOrigin,
    ReadinessDecision,
    artifact_sha,
    safety_flags,
    sha256_payload,
)
from src.historical_exact_semantic_backfill.domain import artifact_sha as backfill_artifact_sha


def test_cli_defaults_to_exact_dataset_readiness_audit_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "8" * 40])

    assert args.input_dir == DEFAULT_INPUT_ARTIFACT_ROOT
    assert args.output_dir == f"artifacts/{ARTIFACT_VERSION}"
    assert args.base_main_sha == "8" * 40


def test_safety_flags_forbid_model_test_backtest_trading_and_future_reads() -> None:
    flags = safety_flags()

    assert flags["RESEARCH_ONLY"] is True
    assert flags["DATA_COST_RUB"] == 0
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["TEST_EVALUATION_PERFORMED"] is False
    assert flags["BACKTEST_PERFORMED"] is False
    assert flags["FUTURE_OUTCOMES_READ"] == 0
    assert flags["FUTURE_TARGETS_READ"] == 0
    assert flags["RULES_V3_CHANGED"] is False
    assert flags["QWEN_CHANGED"] is False
    assert flags["NLP_TUNING_PERFORMED"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False


def test_dataset_readiness_audit_separates_sources_origins_unknowns_and_cohorts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _write_input_artifact(tmp_path / "input")
    expected_sha = _read_json(input_root / "manifest.json")["ARTIFACT_SHA"]
    monkeypatch.setattr(audit_app, "EXPECTED_INPUT_ARTIFACT_SHA", expected_sha)

    manifest = run_exact_dataset_readiness_audit(
        input_root=input_root,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert manifest["ARTIFACT_SHA"] == artifact_sha(manifest)
    assert manifest["CANONICAL_EXACT_EVENTS"] == 6
    assert manifest["FEATURE_READY_EVENTS"] == 5
    assert manifest["ISSUER_ORIGINATED_FEATURE_READY"] == 3
    assert manifest["EXCHANGE_ORIGINATED_FEATURE_READY"] == 2
    assert manifest["UNKNOWN_RATE_TOTAL"] == "0.400000"
    assert manifest["MOEX_RISK_UNKNOWN_RATE"] == "1.000000"
    assert manifest["TOP_TICKER_SHARE"] == "0.400000"
    assert manifest["TOP_3_TICKER_SHARE"] == "0.800000"
    assert manifest["TOP_5_TICKER_SHARE"] == "1.000000"
    assert manifest["TICKER_HHI"] == "0.280000"
    assert manifest["EFFECTIVE_TICKER_COUNT"] == "3.571429"
    assert manifest["SOURCE_FAMILY_HHI"] == "0.520000"
    assert manifest["SOURCE_ID_HHI"] == "0.360000"
    assert manifest["EVENT_ORIGIN_HHI"] == "0.520000"
    assert manifest["LABEL_DISTRIBUTION_AUDIT_SKIPPED_FOR_METHOD_SAFETY"] is False
    assert manifest["READINESS_DECISION"] == ReadinessDecision.MORE_ISSUER_EVENT_DATA_REQUIRED
    assert manifest["RECOMMENDED_PRIMARY_COHORT"] == "NO_BASELINE_PRIMARY_COHORT_RECOMMENDED"
    assert manifest["MOEX_RISK_EVENTS_TREATMENT"] == "B_SEPARATE_EXCHANGE_ORIGINATED_EVENT_FAMILY"
    assert manifest["MODEL_TRAINING_PERFORMED"] is False
    assert manifest["TEST_EVALUATION_PERFORMED"] is False
    assert manifest["FUTURE_OUTCOMES_READ"] == 0
    assert manifest["FUTURE_TARGETS_READ"] == 0

    output_root = tmp_path / "output"
    funnel = _read_json(output_root / "dataset-funnel.json")
    assert funnel["CANONICAL_EXACT_EVENTS"] == {"count": 6, "share": "1.000000"}
    assert funnel["MARKET_ELIGIBLE"]["count"] == 5
    assert funnel["REACTION_READY"]["count"] == 6
    assert funnel["FEATURE_READY"]["count"] == 5
    assert funnel["FEATURE_READY_WITH_VALID_SEMANTICS"]["count"] == 5
    assert funnel["FEATURE_READY_WITH_UNKNOWN_SEMANTICS"]["count"] == 2
    assert funnel["FEATURE_READY_WITH_NON_UNKNOWN_SEMANTICS"]["count"] == 3

    source_families = {
        row["source_family"]: row
        for row in _read_jsonl(output_root / "source-family-summary.jsonl")
    }
    assert source_families["ISSUER_OFFICIAL_RSS_EXACT_LIVE_V1"]["feature_ready"] == 3
    assert source_families["ISSUER_OFFICIAL_RSS_EXACT_LIVE_V1"]["unknown_count"] == 0
    assert source_families["MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"]["feature_ready"] == 2
    assert (
        source_families["MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"]["unknown_rate"]
        == "1.000000"
    )

    origins = {
        row["event_origin"]: row for row in _read_jsonl(output_root / "event-origin-summary.jsonl")
    }
    assert origins[EventOrigin.ISSUER]["feature_ready_count"] == 3
    assert origins[EventOrigin.EXCHANGE]["feature_ready_count"] == 2
    assert origins[EventOrigin.EXCHANGE]["unknown_semantic_rate"] == "1.000000"

    semantic = _read_json(output_root / "semantic-summary.json")
    assert semantic["PRIMARY_EVENT_TYPE_COUNTS"] == {
        "DEBT_FINANCING": 1,
        "DIVIDEND": 2,
        "UNKNOWN": 2,
    }
    assert (
        semantic["UNKNOWN_RATE_BY_SOURCE_FAMILY"]["MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1"]
        == "1.000000"
    )
    assert semantic["UNKNOWN_RATE_BY_EVENT_ORIGIN"][EventOrigin.EXCHANGE] == "1.000000"
    assert semantic["UNKNOWN_RATE_BY_TICKER"]["AAA"] == "0.000000"
    assert semantic["ADDS_SEMANTIC_DIVERSITY"] is True

    ticker_rows = {row["ticker"]: row for row in _read_jsonl(output_root / "ticker-summary.jsonl")}
    assert ticker_rows["AAA"]["feature_ready"] == 2
    assert ticker_rows["AAA"]["share"] == "0.400000"
    assert ticker_rows["AAA"]["event_origins"] == {EventOrigin.ISSUER: 2}
    ticker_concentration = _read_json(output_root / "ticker-concentration.json")
    assert ticker_concentration["top_3_share"] == "0.800000"
    assert ticker_concentration["top_5_share"] == "1.000000"
    assert ticker_concentration["ticker_hhi"] == "0.280000"
    assert ticker_concentration["effective_ticker_count"] == "3.571429"
    assert ticker_concentration["issuer_originated"]["top_3_share"] == "1.000000"
    assert ticker_concentration["exchange_originated"]["events"] == 2

    source_concentration = _read_json(output_root / "source-concentration.json")
    assert source_concentration["source_family_hhi"] == "0.520000"
    assert source_concentration["source_id_hhi"] == "0.360000"
    assert source_concentration["event_origin_hhi"] == "0.520000"
    assert source_concentration["top_source_family_shares"][0] == {
        "share": "0.600000",
        "source_family": "ISSUER_OFFICIAL_RSS_EXACT_LIVE_V1",
    }

    temporal = _read_json(output_root / "temporal-summary.json")
    assert temporal["first_date"] == "2026-07-01"
    assert temporal["last_date"] == "2026-07-02"
    assert temporal["events_per_month"] == {"2026-07": 5}
    assert temporal["events_per_quarter"] == {"2026-Q3": 5}
    assert temporal["temporal_clustering_flag"] is True

    coverage = _read_json(output_root / "target-coverage.json")
    assert coverage["whole_corpus"]["horizons"]["1m"] == {
        "available": 5,
        "missing": 0,
        "coverage": "1.000000",
    }
    assert coverage["whole_corpus"]["horizons"]["60m"] == {
        "available": 4,
        "missing": 1,
        "coverage": "0.800000",
    }
    assert coverage["by_event_origin"][EventOrigin.EXCHANGE]["horizons"]["60m"]["available"] == 1

    labels = _read_json(output_root / "label-distribution.json")
    assert labels["LABEL_DISTRIBUTION_AUDIT_SKIPPED_FOR_METHOD_SAFETY"] is False
    assert labels["horizons"]["1m"]["count"] == 5
    assert labels["horizons"]["1m"]["median"] == "0.001000"
    assert labels["horizons"]["60m"]["count"] == 4

    duplicates = _read_json(output_root / "duplicate-summary.json")
    assert duplicates["duplicate_event_id_count"] == 0
    assert duplicates["duplicate_source_item_id_within_source_count"] == 0
    assert duplicates["duplicate_publication_material_sha_count"] == 1
    assert duplicates["duplicate_semantic_features_sha_count"] == 2
    assert duplicates["top_publication_material_sha_share"] == "0.400000"

    cohort_a = _read_jsonl(output_root / "cohort-a-issuer-event-ids.jsonl")
    cohort_b = _read_jsonl(output_root / "cohort-b-exchange-event-ids.jsonl")
    cohort_c = _read_jsonl(output_root / "cohort-c-all-event-ids.jsonl")
    assert {row["event_origin"] for row in cohort_a} == {EventOrigin.ISSUER}
    assert {row["event_origin"] for row in cohort_b} == {EventOrigin.EXCHANGE}
    assert {row["event_origin"] for row in cohort_c} == {EventOrigin.ISSUER, EventOrigin.EXCHANGE}
    assert len(cohort_a) == 3
    assert len(cohort_b) == 2
    assert len(cohort_c) == 5
    assert {row["event_id"] for row in cohort_c}.isdisjoint({"future", "reaction-only"})

    cohort_summary = _read_json(output_root / "cohort-summary.json")
    assert cohort_summary["COHORT_A"]["rows"] == 3
    assert cohort_summary["COHORT_A"]["unknown_rate"] == "0.000000"
    assert cohort_summary["COHORT_B"]["rows"] == 2
    assert cohort_summary["COHORT_B"]["unknown_rate"] == "1.000000"
    assert cohort_summary["COHORT_C"]["label_coverage"]["horizons"]["60m"]["coverage"] == "0.800000"


def test_dataset_readiness_audit_hashes_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _write_input_artifact(tmp_path / "input")
    expected_sha = _read_json(input_root / "manifest.json")["ARTIFACT_SHA"]
    monkeypatch.setattr(audit_app, "EXPECTED_INPUT_ARTIFACT_SHA", expected_sha)

    left = run_exact_dataset_readiness_audit(
        input_root=input_root,
        output_root=tmp_path / "left",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )
    right = run_exact_dataset_readiness_audit(
        input_root=input_root,
        output_root=tmp_path / "right",
        base_main_sha="8" * 40,
        git_sha="0" * 40,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    deterministic_keys = (
        "DATASET_FUNNEL_SHA",
        "SOURCE_FAMILY_SUMMARY_SHA",
        "EVENT_ORIGIN_SUMMARY_SHA",
        "SEMANTIC_SUMMARY_SHA",
        "TICKER_SUMMARY_SHA",
        "TICKER_CONCENTRATION_SHA",
        "SOURCE_CONCENTRATION_SHA",
        "TEMPORAL_SUMMARY_SHA",
        "TARGET_COVERAGE_SHA",
        "LABEL_DISTRIBUTION_SHA",
        "DUPLICATE_SUMMARY_SHA",
        "COHORT_A_SHA",
        "COHORT_B_SHA",
        "COHORT_C_SHA",
        "ARTIFACT_SHA",
    )
    for key in deterministic_keys:
        assert left[key] == right[key]


def test_audit_module_has_no_model_training_dependency() -> None:
    source = Path(audit_app.__file__).read_text(encoding="utf-8")

    assert "sklearn" not in source
    assert ".fit(" not in source
    assert "predict(" not in source
    assert "accuracy" not in source.lower()
    assert "auc" not in source.lower()
    assert "rmse" not in source.lower()


def test_rules_v3_fingerprint_is_frozen() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_V3_FINGERPRINT


def _write_input_artifact(root: Path) -> Path:
    events = [
        _event("issuer-1", "AAA", "issuer-a", "issuer-a-1", "DIVIDEND", 1, 1),
        _event("issuer-2", "AAA", "issuer-a", "issuer-a-2", "DIVIDEND", 1, 1),
        _event("issuer-3", "BBB", "issuer-b", "issuer-b-1", "DEBT_FINANCING", 1, 0),
        _event(
            "exchange-1",
            "CCC",
            "moex-risk",
            "MOEX:https://moex.example/1",
            "UNKNOWN",
            0,
            0,
            source_family="MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
        ),
        _event(
            "exchange-2",
            "DDD",
            "moex-risk",
            "MOEX:https://moex.example/2",
            "UNKNOWN",
            0,
            0,
            source_family="MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
        ),
        _event(
            "reaction-only",
            "EEE",
            "issuer-c",
            "issuer-c-1",
            "UNKNOWN",
            0,
            0,
            feature_ready=False,
            reaction_ready=True,
            market_complete=False,
        ),
        _event(
            "future",
            "FFF",
            "issuer-future",
            "issuer-future-1",
            "DIVIDEND",
            1,
            1,
            published_at="2026-08-12T10:00:00+00:00",
        ),
    ]
    feature_ids = ["issuer-1", "issuer-2", "issuer-3", "exchange-1", "exchange-2", "future"]
    _write_jsonl(root / "events.jsonl", events)
    _write_jsonl(root / "features.jsonl", [{"event_id": event_id} for event_id in feature_ids])
    _write_jsonl(root / "targets.jsonl", [_target_row(event_id) for event_id in feature_ids])
    _write_jsonl(
        root / "semantic-material-provenance.jsonl",
        [
            _material("issuer-1", "same-dividend-material"),
            _material("issuer-2", "same-dividend-material"),
            _material("issuer-3", "debt-material"),
            _material("exchange-1", "exchange-template-1"),
            _material("exchange-2", "exchange-template-2"),
            _material("future", "future-material"),
        ],
    )
    _write_jsonl(
        root / "semantic-extraction-results.jsonl",
        [
            _semantic("issuer-1", "DIVIDEND", 1, 1),
            _semantic("issuer-2", "DIVIDEND", 1, 1),
            _semantic("issuer-3", "DEBT_FINANCING", 1, 0),
            _semantic("exchange-1", "UNKNOWN", 0, 0),
            _semantic("exchange-2", "UNKNOWN", 0, 0),
            _semantic("future", "DIVIDEND", 1, 1),
        ],
    )
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": "historical-exact-semantic-backfill-v1",
        "FEATURE_READY_AFTER": 6,
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


def _event(
    event_id: str,
    ticker: str,
    source_id: str,
    source_item_id: str,
    primary_event_type: str,
    event_count: int,
    fact_count: int,
    *,
    source_family: str = "ISSUER_OFFICIAL_RSS_EXACT_LIVE_V1",
    feature_ready: bool = True,
    reaction_ready: bool = True,
    market_complete: bool = True,
    published_at: str = "2026-07-01T10:00:00+00:00",
) -> dict[str, object]:
    return {
        "metadata": {
            "event_id": event_id,
            "ticker": ticker,
            "source_id": source_id,
            "source_family": source_family,
            "source_item_id": source_item_id,
            "publication_timestamp_utc": published_at
            if event_id not in {"issuer-3", "exchange-2"}
            else "2026-07-02T10:00:00+00:00",
        },
        "event_features": {
            "primary_event_type": primary_event_type,
            "event_count": event_count,
            "fact_count": fact_count,
        },
        "pre_event_market_features": _market_features(market_complete),
        "target_availability": {
            "reaction_ready": reaction_ready,
            "feature_ready": feature_ready,
            "status": "REACTION_READY" if reaction_ready else "METADATA_ONLY",
        },
    }


def _market_features(complete: bool) -> dict[str, object]:
    return {
        "feature_cutoff": "2026-07-01T10:00:00+00:00",
        "post_event_values_in_features": False,
        "pre_return_5m": "0.01",
        "pre_return_15m": "0.01",
        "pre_return_30m": "0.01",
        "pre_return_60m": "0.01" if complete else None,
        "imoex_pre_return_5m": "0.001",
        "imoex_pre_return_15m": "0.001",
        "imoex_pre_return_30m": "0.001",
        "imoex_pre_return_60m": "0.001" if complete else None,
    }


def _target_row(event_id: str) -> dict[str, object]:
    horizons: dict[str, dict[str, object]] = {}
    for index, horizon in enumerate(("1m", "5m", "15m", "30m", "60m"), start=1):
        available = not (event_id == "exchange-2" and horizon == "60m")
        horizons[horizon] = {
            "available": available,
            "abnormal_return": None if not available else str(index / 1000),
        }
    return {"event_id": event_id, "horizons": horizons}


def _material(event_id: str, material_id: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "publication_material_sha": sha256_payload(material_id),
        "publication_material_available": True,
    }


def _semantic(
    event_id: str, primary_event_type: str, event_count: int, fact_count: int
) -> dict[str, object]:
    features = {
        "primary_event_type": primary_event_type,
        "event_count": event_count,
        "fact_count": fact_count,
    }
    return {
        "event_id": event_id,
        "semantic_features_sha": sha256_payload(features),
        "semantic_features": features,
        "semantic_ready": True,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
