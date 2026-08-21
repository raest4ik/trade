from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from apps.cli.recover_exact_event_security_history import build_parser
from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_security_history_recovery.application import (
    run_security_history_recovery,
)
from src.exact_event_security_history_recovery.domain import (
    ARTIFACT_VERSION,
    FUTURE_EVENT_HOLDOUT_START,
    PR36_ARTIFACT_SHA,
    PR36_OUTPUT_DATASET_SHA,
    PR37_ARTIFACT_SHA,
    PR37_DIAGNOSTIC_COHORT_SHA,
    recovery_safety_flags,
    sha256_payload,
)
from src.tinvest_market.client import TInvestMinuteCandle, TInvestMinuteCandleBatch


def test_cli_defaults_to_security_history_recovery_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "7" * 40])
    assert args.pr36_dir == "artifacts/exact-event-new-source-maturation-v1"
    assert args.diagnostics_dir == "artifacts/exact-event-security-history-diagnostics-v1"
    assert args.output_dir == "artifacts/exact-event-security-history-recovery-v1"
    assert args.live_readonly is False


def test_safety_flags_forbid_model_test_future_and_trading() -> None:
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
    assert flags["MOEX_SUBSTITUTION_USED"] is False
    assert flags["FORWARD_FILL_USED"] is False
    assert flags["SYNTHETIC_MARKET_DATA_USED"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False
    assert flags["SANDBOX_ORDER_SUBMISSION_ALLOWED"] is False


async def test_security_history_recovery_acquires_cache_and_recovers_event(
    tmp_path: Path,
) -> None:
    pr36_root, diagnostics_root = _write_fixture(tmp_path)
    client = _FakeRecoveryClient("uid-RCVR")
    output_root = tmp_path / ARTIFACT_VERSION
    manifest = await run_security_history_recovery(
        pr36_root=pr36_root,
        diagnostics_root=diagnostics_root,
        output_root=output_root,
        base_main_sha="7d39a7e96eb9f55ed20151a2bcab283b5b83c909",
        git_sha="8" * 40,
        client=client,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )

    assert manifest["INPUT_DATASET_SHA"] == PR36_OUTPUT_DATASET_SHA
    assert manifest["PR37_ARTIFACT_SHA"] == PR37_ARTIFACT_SHA
    assert manifest["RECOVERY_COHORT_SHA"] == sha256_payload(["recover-new"])
    assert manifest["RECOVERY_COHORT_TOTAL"] == 1
    assert manifest["EXACT_TOTAL_BEFORE"] == manifest["EXACT_TOTAL_AFTER"] == 3
    assert manifest["REACTION_READY_DELTA"] == 1
    assert manifest["FEATURE_READY_DELTA"] == 1
    assert manifest["RECOVERY_SUCCESS_COUNT"] == 1
    assert manifest["RECOVERY_BLOCKED_COUNT"] == 0
    assert manifest["RECOVERED_EVENT_IDS"] == ["recover-new"]
    assert manifest["CACHE_ACQUISITION_STATUS"] == "PASS"
    assert manifest["CACHE_CANDLES_ACQUIRED_BY_TICKER"] == {"RCVR": 10}
    assert manifest["CACHE_DEDUPE"] == "PASS"
    assert manifest["LEAKAGE_CHECK"] == "PASS"
    assert manifest["EXISTING_EVENT_ROWS_PRESERVED"] == "PASS"
    assert manifest["EXISTING_FEATURE_ROWS_PRESERVED"] == "PASS"
    assert manifest["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert manifest["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert len(str(manifest["ARTIFACT_SHA"])) == 64

    per_event = cast("list[dict[str, Any]]", manifest["PER_EVENT_RECOVERY"])
    recovered = per_event[0]
    assert recovered["EVENT_ID"] == "recover-new"
    assert recovered["UID"] == "uid-RCVR"
    assert recovered["FIGI"] == "figi-RCVR"
    assert recovered["CLASS_CODE"] == "TQBR"
    assert recovered["SECURITY_HISTORY_READY"] is True
    assert recovered["BENCHMARK_HISTORY_READY"] is True
    assert recovered["PRE_EVENT_CONTEXT_READY"] is True
    assert recovered["MAX_FEATURE_TIMESTAMP"] < recovered["PUBLICATION_TIMESTAMP"]
    assert recovered["REACTION_1M_READY"] is True
    assert recovered["REACTION_5M_READY"] is True
    assert recovered["REACTION_15M_READY"] is True
    assert recovered["REACTION_30M_READY"] is True
    assert recovered["REACTION_60M_READY"] is True
    assert recovered["FEATURE_READY_BEFORE"] is False
    assert recovered["FEATURE_READY_AFTER"] is True
    assert recovered["RECOVERY_STATUS"] == "RECOVERED"
    assert recovered["FINAL_BLOCKER"] is None

    cache_rows = _read_jsonl(output_root / "raw-minute-cache" / "RCVR" / "2026-07-20-day.jsonl")
    assert {row["figi"] for row in cache_rows} == {"figi-RCVR"}
    assert {row["class_code"] for row in cache_rows} == {"TQBR"}
    assert {row["instrument_uid"] for row in cache_rows} == {"uid-RCVR"}
    assert client.requested_uids == ["uid-RCVR"] * 8


async def test_cache_only_replay_is_deterministic(tmp_path: Path) -> None:
    pr36_root, diagnostics_root = _write_fixture(tmp_path)
    first = await run_security_history_recovery(
        pr36_root=pr36_root,
        diagnostics_root=diagnostics_root,
        output_root=tmp_path / "first",
        base_main_sha="7d39a7e96eb9f55ed20151a2bcab283b5b83c909",
        git_sha="8" * 40,
        client=_FakeRecoveryClient("uid-RCVR"),
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    replay = await run_security_history_recovery(
        pr36_root=pr36_root,
        diagnostics_root=diagnostics_root,
        output_root=tmp_path / "replay",
        base_main_sha="7d39a7e96eb9f55ed20151a2bcab283b5b83c909",
        git_sha="9" * 40,
        client=None,
        created_at=datetime(2026, 8, 22, tzinfo=UTC),
        extra_cache_roots=(tmp_path / "first" / "raw-minute-cache",),
    )

    assert replay["CACHE_ACQUISITION_STATUS"] == "PASS"
    assert replay["CACHE_CANDLES_ACQUIRED_BY_TICKER"] == {"RCVR": 0}
    assert first["OUTPUT_DATASET_SHA"] == replay["OUTPUT_DATASET_SHA"]
    assert first["ARTIFACT_SHA"] == replay["ARTIFACT_SHA"]


async def test_duplicate_acquisition_dedupes_by_uid_and_timestamp(tmp_path: Path) -> None:
    pr36_root, diagnostics_root = _write_fixture(tmp_path)
    client = _FakeRecoveryClient("uid-RCVR", duplicate_event_day=True)
    manifest = await run_security_history_recovery(
        pr36_root=pr36_root,
        diagnostics_root=diagnostics_root,
        output_root=tmp_path / "dedupe",
        base_main_sha="7d39a7e96eb9f55ed20151a2bcab283b5b83c909",
        git_sha="8" * 40,
        client=client,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    acquisition = _read_jsonl(tmp_path / "dedupe" / "cache-acquisition.jsonl")[0]
    assert acquisition["DUPLICATES_REMOVED"] == 1
    assert acquisition["UNIQUE_TIMESTAMP_IDENTITY"] == "PASS"
    assert manifest["CACHE_DEDUPE"] == "PASS"


async def test_future_diagnostic_row_fails_closed_before_cache_access(tmp_path: Path) -> None:
    pr36_root, diagnostics_root = _write_fixture(tmp_path, include_future_diagnostic=True)
    client = _FakeRecoveryClient("uid-RCVR")
    with pytest.raises(ValueError, match="FUTURE_EVENT_ENTERED_SECURITY_HISTORY_RECOVERY"):
        await run_security_history_recovery(
            pr36_root=pr36_root,
            diagnostics_root=diagnostics_root,
            output_root=tmp_path / "future-fail",
            base_main_sha="7d39a7e96eb9f55ed20151a2bcab283b5b83c909",
            git_sha="8" * 40,
            client=client,
            created_at=datetime(2026, 8, 21, tzinfo=UTC),
        )
    assert client.requested_uids == []


async def test_blocked_event_remains_fail_closed_without_security_history(
    tmp_path: Path,
) -> None:
    pr36_root, diagnostics_root = _write_fixture(tmp_path)
    manifest = await run_security_history_recovery(
        pr36_root=pr36_root,
        diagnostics_root=diagnostics_root,
        output_root=tmp_path / "blocked",
        base_main_sha="7d39a7e96eb9f55ed20151a2bcab283b5b83c909",
        git_sha="8" * 40,
        client=None,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    event = cast("list[dict[str, Any]]", manifest["PER_EVENT_RECOVERY"])[0]
    assert event["RECOVERY_STATUS"] == "BLOCKED"
    assert event["FINAL_BLOCKER"] == "SECURITY_HISTORY_INSUFFICIENT"
    assert event["FEATURE_READY_AFTER"] is False
    assert manifest["RECOVERY_SUCCESS_COUNT"] == 0
    assert manifest["RECOVERY_BLOCKED_COUNT"] == 1


def test_frozen_rules_qwen_and_documentation_contracts() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert prompt_hash() == QWEN_PROMPT_SHA
    assert schema_hash() == QWEN_SCHEMA_SHA
    assert FUTURE_EVENT_HOLDOUT_START.isoformat() == "2026-08-11"
    text = (
        Path(__file__).parents[2] / "docs" / "exact-event-security-history-recovery-v1.md"
    ).read_text(encoding="utf-8")
    assert "T-Invest read-only" in text
    assert "no MOEX substitution" in text
    assert "no forward-fill" in text
    assert "no synthetic market data" in text
    assert "MODEL_TRAINING_PERFORMED=false" in text
    assert "TEST_OUTCOME_USED=false" in text
    assert "FUTURE_EVENT_HOLDOUT_OBSERVED=false" in text


class _FakeRecoveryClient:
    def __init__(self, uid: str, *, duplicate_event_day: bool = False) -> None:
        self.uid = uid
        self.duplicate_event_day = duplicate_event_day
        self.requested_uids: list[str] = []

    async def fetch_minute_candles_audited(
        self, *, instrument_uid: str, date_from: datetime, date_to: datetime
    ) -> TInvestMinuteCandleBatch:
        self.requested_uids.append(instrument_uid)
        if instrument_uid != self.uid:
            return TInvestMinuteCandleBatch((), ("UNEXPECTED_UID",))
        if date_from.date().isoformat() != "2026-07-20":
            return TInvestMinuteCandleBatch((), ())
        candles = list(_candles(self.uid, _complete_times()))
        if self.duplicate_event_day:
            candles.append(candles[0])
        return TInvestMinuteCandleBatch(tuple(candles), ())


def _write_fixture(tmp_path: Path, *, include_future_diagnostic: bool = False) -> tuple[Path, Path]:
    pr36_root = tmp_path / "pr36"
    diagnostics_root = tmp_path / "pr37"
    pr36_root.mkdir()
    diagnostics_root.mkdir()
    events = [
        _event(
            "old-ready",
            "OLD",
            "Old Issuer",
            "uid-OLD",
            "2026-07-20T10:00:30+00:00",
            ready=True,
        ),
        _event(
            "recover-new",
            "RCVR",
            "Recover Issuer",
            "uid-RCVR",
            "2026-07-20T10:00:30+00:00",
        ),
        _event(
            "future-new",
            "FUTR",
            "Future Issuer",
            "uid-FUTR",
            "2026-08-13T07:00:00+00:00",
            future=True,
        ),
    ]
    _write_json(
        pr36_root / "manifest.json",
        {
            "OUTPUT_DATASET_SHA": PR36_OUTPUT_DATASET_SHA,
            "ARTIFACT_SHA": PR36_ARTIFACT_SHA,
            "FUTURE_EVENT_HOLDOUT_USED": False,
            "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
            "TEST_OUTCOME_USED": False,
        },
    )
    _write_jsonl(pr36_root / "events.jsonl", events)
    _write_jsonl(
        pr36_root / "features.jsonl",
        [
            {
                "event_id": "old-ready",
                "feature_cutoff": "2026-07-20T10:00:30+00:00",
                "event_features": events[0]["event_features"],
                "market_features": _complete_market_features(),
            }
        ],
    )
    _write_jsonl(
        pr36_root / "targets.jsonl",
        [{"event_id": "old-ready", "reaction_family": "EXACT_INTRADAY", "horizons": {}}],
    )
    _write_candle_cache(pr36_root / "raw-minute-cache", "IMOEX", "uid-IMOEX")
    _write_json(
        diagnostics_root / "manifest.json",
        {
            "ARTIFACT_SHA": PR37_ARTIFACT_SHA,
            "INPUT_DATASET_SHA": PR36_OUTPUT_DATASET_SHA,
            "OUTPUT_DATASET_SHA": PR36_OUTPUT_DATASET_SHA,
            "DIAGNOSTIC_COHORT_SHA": PR37_DIAGNOSTIC_COHORT_SHA,
            "FUTURE_EVENT_HOLDOUT_USED": False,
            "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
            "TEST_OUTCOME_USED": False,
        },
    )
    rows = [_diagnostic("recover-new", "RCVR", "Recover Issuer", "uid-RCVR")]
    if include_future_diagnostic:
        rows.append(
            _diagnostic(
                "future-new",
                "FUTR",
                "Future Issuer",
                "uid-FUTR",
                published_at="2026-08-13T07:00:00+00:00",
            )
        )
    _write_jsonl(diagnostics_root / "per-event-diagnostics.jsonl", rows)
    return pr36_root, diagnostics_root


def _event(
    event_id: str,
    ticker: str,
    issuer: str,
    uid: str,
    published_at: str,
    *,
    future: bool = False,
    ready: bool = False,
) -> dict[str, object]:
    published = datetime.fromisoformat(published_at)
    return {
        "metadata": {
            "event_id": event_id,
            "ticker": ticker,
            "issuer": issuer,
            "instrument_uid": uid,
            "source_code": "SYNTHETIC_EXACT",
            "publication_timestamp_utc": published.isoformat(),
            "publication_date": published.date().isoformat(),
            "timestamp_source_field": "synthetic exact timestamp",
            "future_holdout": future,
            "session_state": "FUTURE_METADATA_ONLY" if future else "MARKET_CONTEXT_NOT_BUILT",
        },
        "event_features": {"primary_event_type": "DIVIDEND", "event_count": 1, "fact_count": 0},
        "pre_event_market_features": _complete_market_features() if ready else {},
        "target_availability": {
            "reaction_ready": ready,
            "feature_ready": ready,
            "research_outcomes_visible": ready,
            "status": "REACTION_READY" if ready else "BLOCKED",
            "missing_reason": None if ready else "MARKET_HISTORY_MISSING",
        },
        "quality": {
            "feature_cutoff": published.isoformat(),
            "no_forward_fill": True,
            "no_interpolation": True,
            "no_source_mixing": True,
        },
    }


def _diagnostic(
    event_id: str,
    ticker: str,
    issuer: str,
    uid: str,
    *,
    published_at: str = "2026-07-20T10:00:30+00:00",
) -> dict[str, object]:
    return {
        "EVENT_ID": event_id,
        "TICKER": ticker,
        "ISSUER": issuer,
        "PUBLICATION_TIMESTAMP": published_at,
        "CURRENT_FIGI": f"figi-{ticker}",
        "CURRENT_UID": uid,
        "CURRENT_CLASS_CODE": "TQBR",
        "ROOT_CAUSE": "CURRENT_IDENTITY_HAS_HISTORY",
        "RECOVERY_POSSIBLE": True,
        "RECOVERY_PERFORMED": False,
    }


def _complete_market_features() -> dict[str, object]:
    return {
        "feature_cutoff": "2026-07-20T10:00:30+00:00",
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


def _candles(uid: str, begin_times: tuple[str, ...]) -> tuple[TInvestMinuteCandle, ...]:
    rows: list[TInvestMinuteCandle] = []
    for index, begin_text in enumerate(begin_times, start=1):
        begin = datetime.fromisoformat(begin_text)
        price = Decimal(100 + index)
        rows.append(
            TInvestMinuteCandle(
                instrument_uid=uid,
                begin_at=begin,
                end_at=begin + timedelta(minutes=1),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=index,
                is_complete=True,
            )
        )
    return tuple(rows)


def _write_candle_cache(root: Path, ticker: str, instrument_uid: str) -> None:
    rows: list[dict[str, object]] = []
    for candle in _candles(instrument_uid, _complete_times()):
        rows.append(
            {
                "instrument_uid": candle.instrument_uid,
                "begin_at": candle.begin_at.isoformat(),
                "end_at": candle.end_at.isoformat(),
                "open": str(candle.open),
                "high": str(candle.high),
                "low": str(candle.low),
                "close": str(candle.close),
                "volume": candle.volume,
                "is_complete": True,
                "source": "TINVEST_API",
            }
        )
    ticker_root = root / ticker
    ticker_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(ticker_root / "2026-07-20-day.jsonl", rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
