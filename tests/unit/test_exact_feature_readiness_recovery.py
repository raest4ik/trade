from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

import src.exact_feature_readiness_recovery.application as recovery_app
from apps.cli.recover_exact_feature_readiness import build_parser
from src.consolidated_active_exact_historical_maturation.domain import (
    artifact_sha as input_artifact_sha,
)
from src.exact_feature_readiness_recovery.application import (
    run_exact_feature_readiness_recovery,
)
from src.exact_feature_readiness_recovery.domain import (
    ARTIFACT_VERSION,
    DEFAULT_INPUT_ARTIFACT_ROOT,
    FeatureRecoveryBlocker,
    artifact_sha,
    safety_flags,
)


def test_cli_defaults_to_feature_readiness_recovery_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "8" * 40])
    assert args.input_dir == DEFAULT_INPUT_ARTIFACT_ROOT
    assert args.output_dir == f"artifacts/{ARTIFACT_VERSION}"
    assert args.base_main_sha == "8" * 40


def test_safety_flags_forbid_future_targets_model_test_and_trading() -> None:
    flags = safety_flags()
    assert flags["RESEARCH_ONLY"] is True
    assert flags["DATA_COST_RUB"] == 0
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["TEST_EVALUATION_PERFORMED"] is False
    assert flags["FUTURE_PRICE_LOOKUPS"] == 0
    assert flags["FUTURE_REACTIONS_COMPUTED"] == 0
    assert flags["FUTURE_TARGETS_COMPUTED"] == 0
    assert flags["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert flags["FEATURE_DEFINITION_CHANGED"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False
    assert flags["SANDBOX_ORDER_SUBMISSION_ALLOWED"] is False


def test_feature_recovery_diagnoses_recovers_and_preserves_reactions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _write_input_artifact(tmp_path / "input")
    expected_sha = _read_json(input_root / "manifest.json")["ARTIFACT_SHA"]
    monkeypatch.setattr(recovery_app, "EXPECTED_INPUT_MATURATION_ARTIFACT_SHA", expected_sha)

    manifest = run_exact_feature_readiness_recovery(
        input_root=input_root,
        output_root=tmp_path / "output",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
    )

    assert manifest["ARTIFACT_SHA"] == artifact_sha(manifest)
    assert manifest["TARGET_REACTION_READY_FEATURE_BLOCKED"] == 10
    assert manifest["FUTURE_EVENTS_IN_TARGET_COHORT"] == 0
    assert manifest["FEATURE_READY_RECOVERED"] == 3
    assert manifest["FEATURE_READY_STILL_BLOCKED"] == 7
    assert manifest["FEATURE_READY_BEFORE"] == 1
    assert manifest["FEATURE_READY_AFTER"] == 4
    assert manifest["MARKET_FEATURES_COMPLETE"] == 7
    assert manifest["SEMANTIC_EVENT_FEATURES_PRESENT"] == 1
    assert manifest["SEMANTIC_EVENT_FEATURES_RECONSTRUCTED"] == 2
    assert manifest["SEMANTIC_EVENT_FEATURES_MISSING"] == 7
    assert manifest["REACTION_ROWS_CHANGED"] == 0
    assert manifest["REACTION_ROWS_SHA_BEFORE"] == manifest["REACTION_ROWS_SHA_AFTER"]
    assert manifest["FINAL_DECISION"] == "SEMANTIC_EVENT_FEATURES_MISSING"

    output_root = tmp_path / "output"
    cohort = _read_jsonl(output_root / "input-target-cohort.jsonl")
    assert [row["event_id"] for row in cohort] == [
        "bmiss-blocked",
        "empty-blocked",
        "incomplete-blocked",
        "none-blocked",
        "reconstruct-dividend",
        "reconstruct-unknown",
        "state-propagate",
        "short-warmup",
        "session-blocked",
        "ambiguous-blocked",
    ]
    assert "future-holdout" not in {row["event_id"] for row in cohort}
    assert "reaction-missing" not in {row["event_id"] for row in cohort}

    blockers = {row["event_id"]: row for row in _read_jsonl(output_root / "feature-blockers.jsonl")}
    assert (
        blockers["none-blocked"]["primary_blocker"]
        == FeatureRecoveryBlocker.SEMANTIC_EVENT_FEATURES_MISSING
    )
    assert (
        blockers["empty-blocked"]["primary_blocker"]
        == FeatureRecoveryBlocker.SEMANTIC_EVENT_FEATURES_MISSING
    )
    assert (
        blockers["incomplete-blocked"]["primary_blocker"]
        == FeatureRecoveryBlocker.SEMANTIC_EVENT_FEATURES_MISSING
    )
    assert (
        blockers["state-propagate"]["primary_blocker"]
        == FeatureRecoveryBlocker.FEATURE_STATE_NOT_PROPAGATED
    )
    assert (
        blockers["reconstruct-dividend"]["semantic_event_feature_source"]
        == "FROZEN_EVENT_ANALYZER_V3"
    )
    assert blockers["reconstruct-dividend"]["semantic_event_features_reconstructed"] is True
    assert blockers["reconstruct-unknown"]["semantic_event_features_reconstructed"] is True
    assert blockers["none-blocked"]["semantic_event_pipeline_reconstructable"] is False
    assert blockers["empty-blocked"]["semantic_event_pipeline_reconstructable"] is False
    assert blockers["incomplete-blocked"]["semantic_event_pipeline_reconstructable"] is False
    assert (
        blockers["short-warmup"]["primary_blocker"]
        == FeatureRecoveryBlocker.PRE_EVENT_WARMUP_INSUFFICIENT
    )
    assert (
        blockers["bmiss-blocked"]["primary_blocker"]
        == FeatureRecoveryBlocker.BENCHMARK_HISTORY_MISSING
    )
    assert (
        blockers["ambiguous-blocked"]["primary_blocker"]
        == FeatureRecoveryBlocker.INSTRUMENT_IDENTITY_AMBIGUOUS
    )
    assert blockers["session-blocked"]["primary_blocker"] == FeatureRecoveryBlocker.PRE_OPEN
    assert blockers["none-blocked"]["market_feature_pipeline_invoked_during_recovery"] is True
    assert blockers["short-warmup"]["market_feature_pipeline_invoked_during_recovery"] is True

    provenance = {
        row["event_id"]: row
        for row in _read_jsonl(output_root / "market-recovery-provenance.jsonl")
    }
    assert provenance["none-blocked"]["network_fetch_performed"] is False
    assert provenance["none-blocked"]["token_value_read"] is False
    assert provenance["none-blocked"]["security_candles_read"] == len(_complete_times("2026-07-20"))
    assert provenance["none-blocked"]["benchmark_candles_read"] == len(
        _complete_times("2026-07-20")
    )
    assert "future-holdout" not in provenance

    results = {
        row["event_id"]: row for row in _read_jsonl(output_root / "feature-recovery-results.jsonl")
    }
    assert results["none-blocked"]["feature_ready_after"] is False
    assert results["empty-blocked"]["feature_ready_after"] is False
    assert results["incomplete-blocked"]["feature_ready_after"] is False
    assert results["reconstruct-dividend"]["feature_ready_after"] is True
    assert results["reconstruct-unknown"]["feature_ready_after"] is True
    assert results["state-propagate"]["feature_ready_after"] is True
    assert results["short-warmup"]["feature_ready_after"] is False
    assert all(row["reaction_changed"] is False for row in results.values())
    assert all(row["post_event_market_input_used"] is False for row in results.values())

    events = {row["metadata"]["event_id"]: row for row in _read_jsonl(output_root / "events.jsonl")}
    assert events["none-blocked"]["target_availability"]["feature_ready"] is False
    assert events["none-blocked"]["event_features"] is None
    assert events["empty-blocked"]["target_availability"]["feature_ready"] is False
    assert events["empty-blocked"]["event_features"] == {}
    assert events["incomplete-blocked"]["target_availability"]["feature_ready"] is False
    assert events["incomplete-blocked"]["event_features"] == {"primary_event_type": "DIVIDEND"}
    assert events["reconstruct-dividend"]["target_availability"]["feature_ready"] is True
    assert events["reconstruct-dividend"]["event_features"]["primary_event_type"] == "DIVIDEND"
    assert events["reconstruct-unknown"]["target_availability"]["feature_ready"] is True
    assert events["reconstruct-unknown"]["event_features"] == {
        "primary_event_type": "UNKNOWN",
        "event_count": 0,
        "fact_count": 0,
    }
    assert events["state-propagate"]["event_features"] == {
        "primary_event_type": "BUYBACK",
        "event_count": 1,
        "fact_count": 2,
    }
    assert events["future-holdout"]["target_availability"]["feature_ready"] is False

    features = _read_jsonl(output_root / "features.jsonl")
    assert [row["event_id"] for row in features] == [
        "old-ready",
        "reconstruct-dividend",
        "reconstruct-unknown",
        "state-propagate",
    ]
    assert len({row["event_id"] for row in features}) == len(features)
    assert _read_jsonl(output_root / "targets.jsonl") == _read_jsonl(input_root / "targets.jsonl")


def test_feature_recovery_hashes_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _write_input_artifact(tmp_path / "input")
    expected_sha = _read_json(input_root / "manifest.json")["ARTIFACT_SHA"]
    monkeypatch.setattr(recovery_app, "EXPECTED_INPUT_MATURATION_ARTIFACT_SHA", expected_sha)

    left = run_exact_feature_readiness_recovery(
        input_root=input_root,
        output_root=tmp_path / "left",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    right = run_exact_feature_readiness_recovery(
        input_root=input_root,
        output_root=tmp_path / "right",
        base_main_sha="8" * 40,
        git_sha="0" * 40,
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert left["TARGET_COHORT_SHA"] == right["TARGET_COHORT_SHA"]
    assert left["FEATURE_BLOCKERS_SHA"] == right["FEATURE_BLOCKERS_SHA"]
    assert left["MARKET_RECOVERY_PROVENANCE_SHA"] == right["MARKET_RECOVERY_PROVENANCE_SHA"]
    assert left["FEATURE_RECOVERY_RESULT_SHA"] == right["FEATURE_RECOVERY_RESULT_SHA"]
    assert left["OUTPUT_DATASET_SHA"] == right["OUTPUT_DATASET_SHA"]
    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]


def _write_input_artifact(root: Path) -> Path:
    events = [
        _event("old-ready", "OLD", "2026-07-20T10:00:30+00:00", feature_ready=True),
        _event("bmiss-blocked", "BMISS", "2026-06-01T10:00:30+00:00"),
        _event("none-blocked", "NONE", "2026-07-20T10:00:30+00:00"),
        _event("empty-blocked", "EMPTY", "2026-07-20T10:00:30+00:00", event_features={}),
        _event(
            "incomplete-blocked",
            "INCOMP",
            "2026-07-20T10:00:30+00:00",
            event_features={"primary_event_type": "DIVIDEND"},
        ),
        _event("reconstruct-dividend", "REDIV", "2026-07-20T10:00:30+00:00"),
        _event("reconstruct-unknown", "REUNK", "2026-07-20T10:00:30+00:00"),
        _event(
            "state-propagate",
            "STATE",
            "2026-07-20T10:00:30+00:00",
            event_features={"primary_event_type": "BUYBACK", "event_count": 1, "fact_count": 2},
        ),
        _event("short-warmup", "SHORT", "2026-07-22T10:00:30+00:00"),
        _event("session-blocked", "SESSION", "2026-07-23T08:00:00+00:00"),
        _event("ambiguous-blocked", "AMBIG", "2026-07-24T10:00:30+00:00"),
        _event("future-holdout", "FUTR", "2026-08-12T10:00:30+00:00"),
        _event("reaction-missing", "NORXN", "2026-07-25T10:00:30+00:00"),
    ]
    results = [
        _maturation_result("bmiss-blocked", "BMISS", "2026-06-01T10:00:30+00:00"),
        _maturation_result("none-blocked", "NONE", "2026-07-20T10:00:30+00:00"),
        _maturation_result("empty-blocked", "EMPTY", "2026-07-20T10:00:30+00:00"),
        _maturation_result("incomplete-blocked", "INCOMP", "2026-07-20T10:00:30+00:00"),
        _maturation_result("reconstruct-dividend", "REDIV", "2026-07-20T10:00:30+00:00"),
        _maturation_result("reconstruct-unknown", "REUNK", "2026-07-20T10:00:30+00:00"),
        _maturation_result("state-propagate", "STATE", "2026-07-20T10:00:30+00:00"),
        _maturation_result("short-warmup", "SHORT", "2026-07-22T10:00:30+00:00"),
        _maturation_result("session-blocked", "SESSION", "2026-07-23T08:00:00+00:00"),
        _maturation_result("ambiguous-blocked", "AMBIG", "2026-07-24T10:00:30+00:00"),
        _maturation_result("future-holdout", "FUTR", "2026-08-12T10:00:30+00:00"),
        _maturation_result(
            "reaction-missing",
            "NORXN",
            "2026-07-25T10:00:30+00:00",
            reaction_ready=False,
        ),
    ]
    _write_jsonl(root / "maturation-cohort.jsonl", results)
    _write_jsonl(root / "maturation-results.jsonl", results)
    _write_jsonl(root / "events.jsonl", events)
    _write_jsonl(
        root / "input-v1-events.jsonl",
        [
            _source_material(
                "reconstruct-dividend",
                "Board recommends dividends of 12 rub per share for 2025.",
            ),
            _source_material("reconstruct-unknown", "Regular exchange bulletin."),
        ],
    )
    _write_jsonl(root / "input-v2-events.jsonl", [])
    _write_jsonl(root / "features.jsonl", [{"event_id": "old-ready", "marker": "preserved"}])
    _write_jsonl(root / "targets.jsonl", [_target_row(row["event_id"]) for row in results])
    _write_jsonl(
        root / "instrument-identities.jsonl",
        [
            _identity("BMISS", "uid-BMISS"),
            _identity("NONE", "uid-NONE"),
            _identity("EMPTY", "uid-EMPTY"),
            _identity("INCOMP", "uid-INCOMP"),
            _identity("REDIV", "uid-REDIV"),
            _identity("REUNK", "uid-REUNK"),
            _identity("STATE", "uid-STATE"),
            _identity("SHORT", "uid-SHORT"),
            _identity("SESSION", "uid-SESSION"),
            _identity("AMBIG", "uid-AMBIG", ambiguous=True),
            _identity("FUTR", "uid-FUTR"),
            _identity("NORXN", "uid-NORXN"),
        ],
    )
    _write_jsonl(
        root / "market-acquisition-provenance.jsonl",
        [
            _acquisition("bmiss-blocked", security="PASS", benchmark="MISSING"),
            _acquisition("none-blocked"),
            _acquisition("empty-blocked"),
            _acquisition("incomplete-blocked"),
            _acquisition("reconstruct-dividend"),
            _acquisition("reconstruct-unknown"),
            _acquisition("state-propagate"),
            _acquisition("short-warmup"),
            _acquisition("session-blocked"),
            _acquisition("ambiguous-blocked"),
            _acquisition("future-holdout"),
            _acquisition("reaction-missing"),
        ],
    )
    _write_cache(root, "BMISS", "uid-BMISS", "2026-06-01", _complete_times("2026-06-01"))
    for ticker in ("NONE", "EMPTY", "INCOMP", "REDIV", "REUNK", "STATE"):
        _write_cache(root, ticker, f"uid-{ticker}", "2026-07-20", _complete_times("2026-07-20"))
    _write_cache(root, "IMOEX", "uid-IMOEX", "2026-07-20", _complete_times("2026-07-20"))
    _write_cache(root, "NONE", "uid-NONE", "2026-07-21", _complete_times("2026-07-21"))
    _write_cache(root, "IMOEX", "uid-IMOEX", "2026-07-21", _complete_times("2026-07-21"))
    _write_cache(root, "SHORT", "uid-SHORT", "2026-07-22", _short_times("2026-07-22"))
    _write_cache(root, "IMOEX", "uid-IMOEX", "2026-07-22", _short_times("2026-07-22"))
    _write_cache(root, "SESSION", "uid-SESSION", "2026-07-23", _pre_open_times("2026-07-23"))
    _write_cache(root, "IMOEX", "uid-IMOEX", "2026-07-23", _pre_open_times("2026-07-23"))
    _write_cache(root, "AMBIG", "uid-AMBIG", "2026-07-24", _complete_times("2026-07-24"))
    _write_cache(root, "IMOEX", "uid-IMOEX", "2026-07-24", _complete_times("2026-07-24"))

    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": "consolidated-active-exact-historical-maturation-v1",
        "FEATURE_READY_AFTER": 1,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
    }
    manifest["ARTIFACT_SHA"] = input_artifact_sha(manifest)
    _write_json(root / "manifest.json", manifest)
    return root


def _event(
    event_id: str,
    ticker: str,
    published_at: str,
    *,
    event_features: dict[str, object] | None = None,
    feature_ready: bool = False,
) -> dict[str, object]:
    published = datetime.fromisoformat(published_at)
    return {
        "metadata": {
            "event_id": event_id,
            "source_id": f"source-{ticker.lower()}",
            "source_family": "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "source_item_id": event_id,
            "ticker": ticker,
            "issuer": f"{ticker} Issuer",
            "instrument_uid": f"uid-{ticker}",
            "publication_timestamp_utc": published.isoformat(),
            "publication_date": published.date().isoformat(),
            "publication_time": published.time().isoformat(),
            "timestamp_quality": "EXACT",
            "session_state": "DURING_MAIN_SESSION",
            "title_hash": event_id,
        },
        "event_features": event_features,
        "pre_event_market_features": {},
        "target_availability": {
            "research_outcomes_visible": not feature_ready,
            "reaction_ready": True,
            "feature_ready": feature_ready,
            "status": "REACTION_READY" if feature_ready else "METADATA_ONLY",
            "missing_reason": None if feature_ready else "EVENT_FEATURES_MISSING",
        },
        "quality": {
            "feature_cutoff": published.isoformat(),
            "reaction_starts_after_or_at_publication": True,
            "security_benchmark_same_window": True,
            "no_forward_fill": True,
            "no_interpolation": True,
            "no_source_mixing": True,
        },
    }


def _maturation_result(
    event_id: str,
    ticker: str,
    published_at: str,
    *,
    reaction_ready: bool = True,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "ticker": ticker,
        "source_id": f"source-{ticker.lower()}",
        "source_family": "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
        "published_at_utc": published_at,
        "reaction_ready": reaction_ready,
        "feature_ready": False,
        "ready_1m": reaction_ready,
        "ready_5m": reaction_ready,
        "ready_15m": reaction_ready,
        "ready_30m": reaction_ready,
        "ready_60m": reaction_ready,
        "primary_blocker": "EVENT_FEATURES_MISSING",
    }


def _identity(ticker: str, uid: str, *, ambiguous: bool = False) -> dict[str, object]:
    return {
        "ticker": ticker,
        "instrument_uid": uid,
        "identity_provenance": "AMBIGUOUS" if ambiguous else "LIVE_SOURCE_REGISTRY",
    }


def _acquisition(
    event_id: str, *, security: str = "PASS", benchmark: str = "PASS"
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "security_status": security,
        "benchmark_status": benchmark,
    }


def _source_material(event_id: str, title: str) -> dict[str, object]:
    return {
        "metadata": {
            "event_id": event_id,
            "publication_timestamp_utc": "2026-07-20T10:00:30+00:00",
            "ticker": event_id.upper(),
        },
        "title": title,
        "event_features": None,
    }


def _target_row(event_id: object) -> dict[str, object]:
    return {
        "event_id": str(event_id),
        "horizon": "1m",
        "abnormal_return": "0.01",
    }


def _complete_times(day: str) -> tuple[str, ...]:
    return (
        f"{day}T08:59:00+00:00",
        f"{day}T09:29:00+00:00",
        f"{day}T09:44:00+00:00",
        f"{day}T09:54:00+00:00",
        f"{day}T09:59:00+00:00",
        f"{day}T10:01:00+00:00",
    )


def _short_times(day: str) -> tuple[str, ...]:
    return (f"{day}T09:59:00+00:00", f"{day}T10:01:00+00:00")


def _pre_open_times(day: str) -> tuple[str, ...]:
    return (f"{day}T09:00:00+00:00", f"{day}T09:01:00+00:00")


def _write_cache(root: Path, ticker: str, uid: str, day: str, begin_times: Sequence[str]) -> None:
    rows: list[dict[str, object]] = []
    for index, begin_text in enumerate(begin_times, start=1):
        begin = datetime.fromisoformat(begin_text)
        end = begin + timedelta(minutes=1)
        price = Decimal(100 + index)
        rows.append(
            {
                "instrument_uid": uid,
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
    _write_jsonl(root / "raw-minute-cache" / ticker / f"{day}-day.jsonl", rows)


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


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
