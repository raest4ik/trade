from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from apps.cli.mature_new_exact_source_events import build_parser
from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_new_source_maturation.application import run_new_source_maturation
from src.exact_event_new_source_maturation.domain import (
    ARTIFACT_VERSION,
    FUTURE_EVENT_HOLDOUT_START,
    INPUT_DATASET_SHA,
    PR35_ARTIFACT_SHA,
    PREVIOUS_DATASET_SHA,
    maturation_safety_flags,
    sha256_payload,
)


def test_cli_defaults_to_new_source_maturation_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "3" * 40])
    assert args.previous_dir == "artifacts/exact-event-market-history-warmup-recovery-v1"
    assert args.current_dir == "artifacts/exact-event-source-diversity-v3"
    assert args.output_dir == "artifacts/exact-event-new-source-maturation-v1"


def test_safety_flags_forbid_model_test_future_and_trading() -> None:
    flags = maturation_safety_flags()
    assert flags["DATA_MATURATION_ONLY"] is True
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
    assert flags["MOEX_SUBSTITUTION_USED"] is False
    assert flags["FORWARD_FILL_USED"] is False


def test_new_source_maturation_recovers_only_safe_historical_events(
    tmp_path: Path,
) -> None:
    previous_root, current_root = _write_fixture(tmp_path)
    output_root = tmp_path / ARTIFACT_VERSION
    manifest = run_new_source_maturation(
        previous_root=previous_root,
        current_root=current_root,
        output_root=output_root,
        base_main_sha="392b04ddb12c83c8adff94fc1ec627d814316b61",
        git_sha="4" * 40,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )

    assert manifest["INPUT_DATASET_SHA"] == INPUT_DATASET_SHA
    assert manifest["PREVIOUS_DATASET_SHA"] == PREVIOUS_DATASET_SHA
    assert manifest["NEW_EVENT_IDS"] == ["future-new", "missing-new", "recover-new"]
    assert manifest["INPUT_NEW_EVENT_COHORT_SHA"] == sha256_payload(
        ["future-new", "missing-new", "recover-new"]
    )
    assert manifest["NEW_EVENTS_TOTAL"] == 3
    assert manifest["NEW_EVENTS_HISTORICAL"] == 2
    assert manifest["NEW_EVENTS_FUTURE_METADATA_ONLY"] == 1
    assert manifest["EXACT_TOTAL_BEFORE"] == manifest["EXACT_TOTAL_AFTER"] == 4
    assert manifest["REACTION_READY_DELTA"] == 1
    assert manifest["FEATURE_READY_DELTA"] == 1
    assert manifest["NEW_EVENTS_REACTION_READY_BEFORE"] == 0
    assert manifest["NEW_EVENTS_REACTION_READY_AFTER"] == 1
    assert manifest["NEW_EVENTS_FEATURE_READY_BEFORE"] == 0
    assert manifest["NEW_EVENTS_FEATURE_READY_AFTER"] == 1
    assert manifest["RECOVERED_EVENT_IDS"] == ["recover-new"]
    assert manifest["BLOCKED_EVENT_IDS"] == ["future-new", "missing-new"]
    assert manifest["PER_HORIZON_READY_COUNTS"] == {
        "1m": 1,
        "5m": 1,
        "15m": 1,
        "30m": 1,
        "60m": 1,
    }
    assert manifest["EXACT_V3_PRESERVED"] == "YES"
    assert manifest["EXISTING_EVENT_ROWS_PRESERVED"] == "PASS"
    assert manifest["EXISTING_FEATURE_ROWS_PRESERVED"] == "PASS"
    assert manifest["LEAKAGE_CHECK"] == "PASS"
    assert manifest["ARTIFACT_SHA"] == sha256_payload({**manifest, "ARTIFACT_SHA": None})

    per_event = {
        row["event_id"]: row for row in cast("list[dict[str, Any]]", manifest["PER_EVENT_STATUS"])
    }
    assert per_event["recover-new"]["primary_readiness_blocker"] is None
    assert per_event["recover-new"]["final_feature_ready"] is True
    assert per_event["recover-new"]["strict_feature_timestamp_before_publication"] is True
    assert per_event["missing-new"]["primary_readiness_blocker"] == "MARKET_HISTORY_MISSING"
    assert per_event["future-new"]["historical_or_future"] == "FUTURE_METADATA_ONLY"
    assert per_event["future-new"]["market_context_acquisition_status"] == "SKIPPED_FUTURE_HOLDOUT"

    events_after = {_event_id(row): row for row in _read_jsonl(output_root / "events.jsonl")}
    assert events_after["old-ready"] == _old_ready_event()
    assert (
        cast("dict[str, Any]", events_after["future-new"]["target_availability"])[
            "research_outcomes_visible"
        ]
        is False
    )
    assert (
        cast("dict[str, Any]", events_after["future-new"]["target_availability"])["feature_ready"]
        is False
    )


def test_future_events_do_not_read_market_or_target_cache(tmp_path: Path) -> None:
    previous_root, current_root = _write_fixture(tmp_path, include_historical_cache=False)
    _write_candle_cache(
        current_root / "raw-minute-cache",
        "FUTR",
        "uid-FUTR",
        _complete_times(),
        source="MOEX_SUBSTITUTE",
    )
    manifest = run_new_source_maturation(
        previous_root=previous_root,
        current_root=current_root,
        output_root=tmp_path / "future-skip",
        base_main_sha="392b04ddb12c83c8adff94fc1ec627d814316b61",
        git_sha="4" * 40,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    per_event = {
        row["event_id"]: row for row in cast("list[dict[str, Any]]", manifest["PER_EVENT_STATUS"])
    }
    assert per_event["future-new"]["security_history_available"] is None
    assert manifest["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert manifest["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False


def test_non_tinvest_cache_fails_closed_without_moex_substitution(tmp_path: Path) -> None:
    previous_root, current_root = _write_fixture(tmp_path, include_historical_cache=False)
    _write_candle_cache(
        current_root / "raw-minute-cache",
        "RCVR",
        "uid-RCVR",
        _complete_times(),
        source="MOEX_SUBSTITUTE",
    )
    _write_candle_cache(current_root / "raw-minute-cache", "IMOEX", "uid-IMOEX", _complete_times())
    with pytest.raises(ValueError, match="NON_TINVEST_CANDLE_CACHE_SOURCE"):
        run_new_source_maturation(
            previous_root=previous_root,
            current_root=current_root,
            output_root=tmp_path / "moex-block",
            base_main_sha="392b04ddb12c83c8adff94fc1ec627d814316b61",
            git_sha="4" * 40,
            created_at=datetime(2026, 8, 21, tzinfo=UTC),
        )


def test_deterministic_reproduction_hashes(tmp_path: Path) -> None:
    previous_root, current_root = _write_fixture(tmp_path)
    left = run_new_source_maturation(
        previous_root=previous_root,
        current_root=current_root,
        output_root=tmp_path / "left",
        base_main_sha="392b04ddb12c83c8adff94fc1ec627d814316b61",
        git_sha="4" * 40,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    right = run_new_source_maturation(
        previous_root=previous_root,
        current_root=current_root,
        output_root=tmp_path / "right",
        base_main_sha="392b04ddb12c83c8adff94fc1ec627d814316b61",
        git_sha="4" * 40,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    assert left["INPUT_NEW_EVENT_COHORT_SHA"] == right["INPUT_NEW_EVENT_COHORT_SHA"]
    assert left["OUTPUT_DATASET_SHA"] == right["OUTPUT_DATASET_SHA"]
    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]


def test_frozen_rules_v3_and_qwen_contracts_are_unchanged() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert prompt_hash() == QWEN_PROMPT_SHA
    assert schema_hash() == QWEN_SCHEMA_SHA
    assert FUTURE_EVENT_HOLDOUT_START.isoformat() == "2026-08-11"


def test_documentation_states_safety_boundaries() -> None:
    text = (
        Path(__file__).parents[2] / "docs" / "exact-event-new-source-maturation-v1.md"
    ).read_text(encoding="utf-8")
    assert "DATA_MATURATION_ONLY=true" in text
    assert "MODEL_TRAINING_PERFORMED=false" in text
    assert "TEST_OUTCOME_USED=false" in text
    assert "FUTURE_EVENT_HOLDOUT_OBSERVED=false" in text
    assert "no MOEX substitution" in text
    assert "no forward-fill" in text
    assert "no source expansion" in text


def _write_fixture(tmp_path: Path, *, include_historical_cache: bool = True) -> tuple[Path, Path]:
    previous_root = tmp_path / "warmup"
    current_root = tmp_path / "source-v3"
    previous_root.mkdir()
    current_root.mkdir()
    old_event = _old_ready_event()
    current_events = [
        old_event,
        _new_event("recover-new", "RCVR", "Recover Issuer", "2026-07-20T10:00:30+00:00"),
        _new_event("missing-new", "MISS", "Missing Issuer", "2026-07-20T10:00:30+00:00"),
        _new_event(
            "future-new",
            "FUTR",
            "Future Issuer",
            "2026-08-13T07:00:00+00:00",
            future=True,
        ),
    ]
    _write_json(
        previous_root / "manifest.json",
        {"OUTPUT_DATASET_SHA": PREVIOUS_DATASET_SHA},
    )
    _write_json(
        current_root / "manifest.json",
        {
            "OUTPUT_DATASET_SHA": INPUT_DATASET_SHA,
            "ARTIFACT_SHA": PR35_ARTIFACT_SHA,
            "EXACT_V2_PRESERVED": "YES",
            "FUTURE_EVENT_HOLDOUT_USED": False,
            "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
            "TEST_OUTCOME_USED": False,
        },
    )
    _write_jsonl(previous_root / "events.jsonl", [old_event])
    _write_jsonl(current_root / "events.jsonl", current_events)
    _write_jsonl(
        current_root / "features.jsonl",
        [
            {
                "event_id": "old-ready",
                "feature_cutoff": "2026-07-20T10:00:30+00:00",
                "event_features": old_event["event_features"],
                "market_features": _complete_market_features("2026-07-20T10:00:30+00:00"),
            }
        ],
    )
    _write_jsonl(
        current_root / "targets.jsonl",
        [{"event_id": "old-ready", "reaction_family": "EXACT_INTRADAY", "horizons": {}}],
    )
    if include_historical_cache:
        _write_candle_cache(
            current_root / "raw-minute-cache", "RCVR", "uid-RCVR", _complete_times()
        )
        _write_candle_cache(
            current_root / "raw-minute-cache", "IMOEX", "uid-IMOEX", _complete_times()
        )
    return previous_root, current_root


def _old_ready_event() -> dict[str, object]:
    return _event(
        "old-ready",
        "OLD",
        "Old Issuer",
        "2026-07-20T10:00:30+00:00",
        feature_ready=True,
        reaction_ready=True,
        research_outcomes_visible=True,
        status="REACTION_READY",
        missing_reason=None,
        market_features=_complete_market_features("2026-07-20T10:00:30+00:00"),
    )


def _new_event(
    event_id: str, ticker: str, issuer: str, published_at: str, *, future: bool = False
) -> dict[str, object]:
    return _event(
        event_id,
        ticker,
        issuer,
        published_at,
        feature_ready=False,
        reaction_ready=False,
        research_outcomes_visible=False,
        status="FUTURE_HOLDOUT_METADATA_ONLY"
        if future
        else "TINVEST_HISTORY_NOT_ACQUIRED_IN_V3_CACHE_ONLY",
        missing_reason="FUTURE_HOLDOUT_OUTCOMES_GUARDED"
        if future
        else "TINVEST_HISTORY_UNAVAILABLE_CACHE_ONLY",
        market_features={},
        future=future,
    )


def _event(
    event_id: str,
    ticker: str,
    issuer: str,
    published_at: str,
    *,
    feature_ready: bool,
    reaction_ready: bool,
    research_outcomes_visible: bool,
    status: str,
    missing_reason: str | None,
    market_features: dict[str, object],
    future: bool = False,
) -> dict[str, object]:
    published = datetime.fromisoformat(published_at)
    return {
        "metadata": {
            "event_id": event_id,
            "source_code": f"{ticker}_OFFICIAL_RSS",
            "source_item_id": event_id,
            "canonical_url": f"https://issuer.example/{event_id}",
            "ticker": ticker,
            "issuer": issuer,
            "instrument_uid": f"uid-{ticker}",
            "publication_timestamp_utc": published.isoformat(),
            "publication_timestamp_raw": published.isoformat(),
            "publication_date": published.date().isoformat(),
            "publication_time": published.time().isoformat(),
            "publication_timezone": "UTC",
            "timestamp_source_field": "synthetic exact timestamp",
            "timestamp_quality": "EXACT",
            "future_holdout": future,
            "session_state": "FUTURE_METADATA_ONLY" if future else "MARKET_CONTEXT_NOT_BUILT",
            "title_hash": event_id,
        },
        "event_features": {"primary_event_type": "DIVIDEND", "event_count": 1, "fact_count": 0},
        "pre_event_market_features": market_features,
        "target_availability": {
            "research_outcomes_visible": research_outcomes_visible,
            "reaction_ready": reaction_ready,
            "feature_ready": feature_ready,
            "status": status,
            "missing_reason": missing_reason,
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


def _complete_times() -> tuple[str, ...]:
    return (
        "2026-07-20T08:59:00+00:00",
        "2026-07-20T09:29:00+00:00",
        "2026-07-20T09:44:00+00:00",
        "2026-07-20T09:54:00+00:00",
        "2026-07-20T09:59:00+00:00",
        "2026-07-20T10:01:00+00:00",
        "2026-07-20T10:05:00+00:00",
        "2026-07-20T10:15:00+00:00",
        "2026-07-20T10:30:00+00:00",
        "2026-07-20T11:00:00+00:00",
    )


def _write_candle_cache(
    root: Path,
    ticker: str,
    instrument_uid: str,
    begin_times: tuple[str, ...],
    *,
    source: str = "TINVEST_API",
) -> None:
    ticker_root = root / ticker
    ticker_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, begin_text in enumerate(begin_times, start=1):
        begin = datetime.fromisoformat(begin_text)
        end = begin + timedelta(minutes=1)
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
                "source": source,
            }
        )
    _write_jsonl(ticker_root / "2026-07-20-day.jsonl", rows)


def _event_id(row: dict[str, object]) -> str:
    return str(cast("dict[str, object]", row["metadata"])["event_id"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
