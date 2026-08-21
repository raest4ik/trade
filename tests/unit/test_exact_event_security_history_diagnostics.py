from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from apps.cli.diagnose_exact_event_security_history import build_parser
from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_security_history_diagnostics.application import (
    run_security_history_diagnostics,
)
from src.exact_event_security_history_diagnostics.domain import (
    ARTIFACT_VERSION,
    FUTURE_EVENT_HOLDOUT_START,
    PR36_ARTIFACT_SHA,
    PR36_COHORT_SHA,
    PR36_INPUT_DATASET_SHA,
    PR36_OUTPUT_DATASET_SHA,
    RootCauseStatus,
    security_history_safety_flags,
    sha256_payload,
)
from src.tinvest_market.client import (
    TInvestCandleBatch,
    TInvestDailyCandle,
    TInvestInstrument,
    TInvestMinuteCandle,
    TInvestMinuteCandleBatch,
)


def test_cli_defaults_to_security_history_diagnostics_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "a" * 40])
    assert args.pr36_dir == "artifacts/exact-event-new-source-maturation-v1"
    assert args.universe == "artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json"
    assert args.output_dir == "artifacts/exact-event-security-history-diagnostics-v1"
    assert args.live_readonly is False


def test_safety_flags_forbid_model_test_future_and_trading() -> None:
    flags = security_history_safety_flags()
    assert flags["DIAGNOSTICS_ONLY"] is True
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert flags["RULES_V3_CHANGED"] is False
    assert flags["QWEN_CHANGED"] is False
    assert flags["NLP_TUNING_PERFORMED"] is False
    assert flags["MOEX_SUBSTITUTION_USED"] is False
    assert flags["FORWARD_FILL_USED"] is False
    assert flags["SYNTHETIC_MARKET_DATA_USED"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False
    assert flags["SANDBOX_ORDER_SUBMISSION_ALLOWED"] is False


async def test_security_history_diagnostics_filter_cohort_and_exclude_future(
    tmp_path: Path,
) -> None:
    pr36_root, universe_path = _write_fixture(tmp_path)
    client = _FakeSecurityHistoryClient(
        instruments={
            "uid-CUR": _instrument("CURH", "uid-CUR", first=date(2020, 1, 1)),
            "uid-MISS": _instrument("MISS", "uid-MISS", first=date(2020, 1, 1)),
            "uid-LATE": _instrument("LATE", "uid-LATE", first=date(2026, 8, 1)),
            "uid-FUT": _instrument("FUTR", "uid-FUT", first=date(2020, 1, 1)),
        },
        query_results={
            "CURH": ("uid-CUR",),
            "Current Issuer": ("uid-CUR",),
            "MISS": ("uid-MISS",),
            "Missing Issuer": ("uid-MISS",),
            "LATE": ("uid-LATE",),
            "Late Issuer": ("uid-LATE",),
        },
        daily_history={"uid-CUR": True},
        minute_history={"uid-CUR": True},
    )
    manifest = await run_security_history_diagnostics(
        pr36_root=pr36_root,
        universe_path=universe_path,
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="aaf5de2e1f2692105c03e126c73f9d0f6aa87d7b",
        git_sha="b" * 40,
        client=client,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )

    assert manifest["INPUT_DATASET_SHA"] == PR36_OUTPUT_DATASET_SHA
    assert manifest["OUTPUT_DATASET_SHA"] == PR36_OUTPUT_DATASET_SHA
    assert manifest["DIAGNOSTIC_EVENTS_TOTAL"] == 3
    assert manifest["DIAGNOSTIC_COHORT_SHA"] == sha256_payload(
        ["current-history", "late-history", "missing-history"]
    )
    assert manifest["CURRENT_IDENTITY_HAS_HISTORY_COUNT"] == 1
    assert manifest["NOT_TRADING_AT_EVENT_TIME_COUNT"] == 1
    assert manifest["TINVEST_HISTORY_UNAVAILABLE_COUNT"] == 1
    assert manifest["RECOVERY_POSSIBLE_COUNT"] == 1
    assert manifest["RECOVERY_PERFORMED_COUNT"] == 0
    assert manifest["RECOVERED_EVENT_IDS"] == []
    assert manifest["BLOCKED_EVENT_IDS"] == [
        "current-history",
        "late-history",
        "missing-history",
    ]
    assert manifest["REACTION_READY_BEFORE"] == manifest["REACTION_READY_AFTER"] == 565
    assert manifest["FEATURE_READY_BEFORE"] == manifest["FEATURE_READY_AFTER"] == 564
    assert manifest["EXACT_V3_PRESERVED"] == "YES"
    assert manifest["PR36_DATASET_PRESERVED"] == "YES"
    assert manifest["EXISTING_EVENT_ROWS_PRESERVED"] == "PASS"
    assert manifest["EXISTING_FEATURE_ROWS_PRESERVED"] == "PASS"
    assert manifest["LEAKAGE_CHECK"] == "PASS"
    assert manifest["ARTIFACT_SHA"] == sha256_payload({**manifest, "ARTIFACT_SHA": None})

    per_event = {
        row["EVENT_ID"]: row
        for row in cast("list[dict[str, Any]]", manifest["PER_EVENT_DIAGNOSTICS"])
    }
    assert (
        per_event["current-history"]["ROOT_CAUSE"] == RootCauseStatus.CURRENT_IDENTITY_HAS_HISTORY
    )
    assert per_event["current-history"]["CURRENT_HISTORY_AVAILABLE"] is True
    assert per_event["late-history"]["ROOT_CAUSE"] == (
        RootCauseStatus.INSTRUMENT_NOT_TRADING_AT_EVENT_TIME
    )
    assert per_event["missing-history"]["ROOT_CAUSE"] == RootCauseStatus.TINVEST_HISTORY_UNAVAILABLE
    assert "uid-FUT" not in client.probed_uids
    assert manifest["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert manifest["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False


async def test_historical_identity_found_when_current_uid_has_no_history(tmp_path: Path) -> None:
    pr36_root, universe_path = _write_fixture(tmp_path)
    client = _FakeSecurityHistoryClient(
        instruments={
            "uid-CUR": _instrument("CURH", "uid-CUR", first=date(2020, 1, 1)),
            "uid-CUR-OLD": _instrument("CURH", "uid-CUR-OLD", first=date(2019, 1, 1)),
            "uid-MISS": _instrument("MISS", "uid-MISS", first=date(2020, 1, 1)),
            "uid-LATE": _instrument("LATE", "uid-LATE", first=date(2020, 1, 1)),
        },
        query_results={
            "CURH": ("uid-CUR", "uid-CUR-OLD"),
            "Current Issuer": ("uid-CUR", "uid-CUR-OLD"),
            "MISS": ("uid-MISS",),
            "Missing Issuer": ("uid-MISS",),
            "LATE": ("uid-LATE",),
            "Late Issuer": ("uid-LATE",),
        },
        daily_history={"uid-CUR-OLD": True},
        minute_history={},
    )
    manifest = await run_security_history_diagnostics(
        pr36_root=pr36_root,
        universe_path=universe_path,
        output_root=tmp_path / "historical",
        base_main_sha="aaf5de2e1f2692105c03e126c73f9d0f6aa87d7b",
        git_sha="b" * 40,
        client=client,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    per_event = {
        row["EVENT_ID"]: row
        for row in cast("list[dict[str, Any]]", manifest["PER_EVENT_DIAGNOSTICS"])
    }
    assert per_event["current-history"]["ROOT_CAUSE"] == RootCauseStatus.HISTORICAL_IDENTITY_FOUND
    assert per_event["current-history"]["HISTORICAL_UID"] == "uid-CUR-OLD"
    assert manifest["HISTORICAL_IDENTITY_FOUND_COUNT"] == 1


async def test_ambiguous_identity_fails_closed_when_uid_is_absent(tmp_path: Path) -> None:
    pr36_root, universe_path = _write_fixture(tmp_path, include_current_in_universe=False)
    client = _FakeSecurityHistoryClient(
        instruments={
            "uid-A": _instrument("CURH", "uid-A", first=date(2020, 1, 1)),
            "uid-B": _instrument("CURH", "uid-B", first=date(2020, 1, 1)),
            "uid-MISS": _instrument("MISS", "uid-MISS", first=date(2020, 1, 1)),
            "uid-LATE": _instrument("LATE", "uid-LATE", first=date(2020, 1, 1)),
        },
        query_results={
            "CURH": ("uid-A", "uid-B"),
            "Current Issuer": ("uid-A", "uid-B"),
            "MISS": ("uid-MISS",),
            "Missing Issuer": ("uid-MISS",),
            "LATE": ("uid-LATE",),
            "Late Issuer": ("uid-LATE",),
        },
        daily_history={},
        minute_history={},
    )
    manifest = await run_security_history_diagnostics(
        pr36_root=pr36_root,
        universe_path=universe_path,
        output_root=tmp_path / "ambiguous",
        base_main_sha="aaf5de2e1f2692105c03e126c73f9d0f6aa87d7b",
        git_sha="b" * 40,
        client=client,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    per_event = {
        row["EVENT_ID"]: row
        for row in cast("list[dict[str, Any]]", manifest["PER_EVENT_DIAGNOSTICS"])
    }
    assert per_event["current-history"]["ROOT_CAUSE"] == RootCauseStatus.IDENTITY_AMBIGUOUS
    assert manifest["IDENTITY_AMBIGUOUS_COUNT"] == 1


async def test_deterministic_diagnostic_hash(tmp_path: Path) -> None:
    pr36_root, universe_path = _write_fixture(tmp_path)
    first = await run_security_history_diagnostics(
        pr36_root=pr36_root,
        universe_path=universe_path,
        output_root=tmp_path / "left",
        base_main_sha="aaf5de2e1f2692105c03e126c73f9d0f6aa87d7b",
        git_sha="b" * 40,
        client=None,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    second = await run_security_history_diagnostics(
        pr36_root=pr36_root,
        universe_path=universe_path,
        output_root=tmp_path / "right",
        base_main_sha="aaf5de2e1f2692105c03e126c73f9d0f6aa87d7b",
        git_sha="b" * 40,
        client=None,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    assert first["DIAGNOSTIC_COHORT_SHA"] == second["DIAGNOSTIC_COHORT_SHA"]
    assert first["ARTIFACT_SHA"] == second["ARTIFACT_SHA"]


def test_frozen_rules_v3_and_qwen_contracts_are_unchanged() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert prompt_hash() == QWEN_PROMPT_SHA
    assert schema_hash() == QWEN_SCHEMA_SHA
    assert FUTURE_EVENT_HOLDOUT_START.isoformat() == "2026-08-11"


def test_documentation_states_safety_boundaries() -> None:
    text = (
        Path(__file__).parents[2] / "docs" / "exact-event-security-history-diagnostics-v1.md"
    ).read_text(encoding="utf-8")
    assert "DIAGNOSTICS_ONLY=true" in text
    assert "MODEL_TRAINING_PERFORMED=false" in text
    assert "TEST_OUTCOME_USED=false" in text
    assert "FUTURE_EVENT_HOLDOUT_OBSERVED=false" in text
    assert "MOEX_SUBSTITUTION_USED=false" in text
    assert "FORWARD_FILL_USED=false" in text
    assert "SYNTHETIC_MARKET_DATA_USED=false" in text


class _FakeSecurityHistoryClient:
    def __init__(
        self,
        *,
        instruments: dict[str, TInvestInstrument],
        query_results: dict[str, tuple[str, ...]],
        daily_history: dict[str, bool],
        minute_history: dict[str, bool],
    ) -> None:
        self._instruments = instruments
        self._query_results = query_results
        self._daily_history = daily_history
        self._minute_history = minute_history
        self.probed_uids: set[str] = set()

    async def find_instruments(
        self, query: str, *, instrument_kind: str
    ) -> tuple[TInvestInstrument, ...]:
        assert instrument_kind == "INSTRUMENT_TYPE_SHARE"
        return tuple(self._instruments[uid] for uid in self._query_results.get(query, ()))

    async def get_instrument_by_uid(self, instrument_uid: str) -> TInvestInstrument:
        return self._instruments[instrument_uid]

    async def fetch_daily_candles_audited(
        self, *, instrument_uid: str, date_from: date, date_to: date
    ) -> TInvestCandleBatch:
        self.probed_uids.add(instrument_uid)
        candles = (
            (
                TInvestDailyCandle(
                    instrument_uid=instrument_uid,
                    trade_date=date_from,
                    open=Decimal("100"),
                    high=Decimal("101"),
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=100,
                    is_complete=True,
                ),
            )
            if self._daily_history.get(instrument_uid, False)
            else ()
        )
        return TInvestCandleBatch(candles, ())

    async def fetch_minute_candles_audited(
        self, *, instrument_uid: str, date_from: datetime, date_to: datetime
    ) -> TInvestMinuteCandleBatch:
        self.probed_uids.add(instrument_uid)
        candles = (
            (
                TInvestMinuteCandle(
                    instrument_uid=instrument_uid,
                    begin_at=date_from,
                    end_at=date_from + timedelta(minutes=1),
                    open=Decimal("100"),
                    high=Decimal("101"),
                    low=Decimal("99"),
                    close=Decimal("100"),
                    volume=100,
                    is_complete=True,
                ),
            )
            if self._minute_history.get(instrument_uid, False)
            else ()
        )
        return TInvestMinuteCandleBatch(candles, ())


def _write_fixture(
    tmp_path: Path, *, include_current_in_universe: bool = True
) -> tuple[Path, Path]:
    pr36_root = tmp_path / "pr36"
    pr36_root.mkdir()
    events = [
        _event("current-history", "CURH", "Current Issuer", "uid-CUR", "2026-07-15T10:00:00+00:00"),
        _event(
            "missing-history", "MISS", "Missing Issuer", "uid-MISS", "2026-07-16T10:00:00+00:00"
        ),
        _event("late-history", "LATE", "Late Issuer", "uid-LATE", "2026-07-17T10:00:00+00:00"),
        _event(
            "future-one",
            "FUTR",
            "Future Issuer",
            "uid-FUT",
            "2026-08-13T07:00:00+00:00",
            future=True,
        ),
        _event(
            "old-ready", "OLD", "Old Issuer", "uid-OLD", "2026-07-10T10:00:00+00:00", ready=True
        ),
    ]
    _write_json(
        pr36_root / "manifest.json",
        {
            "INPUT_DATASET_SHA": PR36_INPUT_DATASET_SHA,
            "OUTPUT_DATASET_SHA": PR36_OUTPUT_DATASET_SHA,
            "ARTIFACT_SHA": PR36_ARTIFACT_SHA,
            "INPUT_NEW_EVENT_COHORT_SHA": PR36_COHORT_SHA,
            "NEW_EVENT_IDS": [
                "current-history",
                "missing-history",
                "late-history",
                "future-one",
            ],
            "EXACT_V3_PRESERVED": "YES",
            "EXISTING_EVENT_ROWS_PRESERVED": "PASS",
            "EXISTING_FEATURE_ROWS_PRESERVED": "PASS",
            "FUTURE_EVENT_HOLDOUT_USED": False,
            "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
            "TEST_OUTCOME_USED": False,
            "REACTION_READY_AFTER": 565,
            "FEATURE_READY_AFTER": 564,
        },
    )
    _write_jsonl(pr36_root / "events.jsonl", events)
    _write_jsonl(
        pr36_root / "features.jsonl",
        [{"event_id": "old-ready", "event_features": {}, "market_features": {}}],
    )
    _write_jsonl(pr36_root / "per-event-status.jsonl", [_status(event) for event in events[:4]])
    universe_rows = [
        _universe("MISS", "Missing Issuer", "uid-MISS", first="2020-01-01"),
        _universe("LATE", "Late Issuer", "uid-LATE", first="2026-08-01"),
    ]
    if include_current_in_universe:
        universe_rows.append(_universe("CURH", "Current Issuer", "uid-CUR", first="2020-01-01"))
    universe_path = tmp_path / "instrument-mapping.json"
    _write_json(universe_path, {"instruments": universe_rows})
    return pr36_root, universe_path


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
        },
        "event_features": {},
        "pre_event_market_features": {},
        "target_availability": {
            "reaction_ready": ready,
            "feature_ready": ready,
            "research_outcomes_visible": ready,
            "status": "REACTION_READY" if ready else "BLOCKED",
            "missing_reason": None if ready else "MARKET_HISTORY_MISSING",
        },
        "quality": {},
    }


def _status(event: dict[str, object]) -> dict[str, object]:
    metadata = cast("dict[str, object]", event["metadata"])
    future = bool(metadata["future_holdout"])
    return {
        "event_id": metadata["event_id"],
        "ticker": metadata["ticker"],
        "issuer": metadata["issuer"],
        "publication_timestamp_utc": metadata["publication_timestamp_utc"],
        "timestamp_provenance": metadata["timestamp_source_field"],
        "future_holdout": future,
        "historical_or_future": "FUTURE_METADATA_ONLY" if future else "HISTORICAL",
        "primary_readiness_blocker": "FUTURE_METADATA_ONLY" if future else "MARKET_HISTORY_MISSING",
    }


def _universe(ticker: str, name: str, uid: str, *, first: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "name": name,
        "instrument_uid": uid,
        "figi": f"figi-{uid}",
        "class_code": "TQBR",
        "instrument_type": "INSTRUMENT_TYPE_SHARE",
        "first_1day_candle_date": first,
        "last_1day_candle_date": None,
        "exchange": "moex_morning_weekend",
        "currency": "rub",
    }


def _instrument(ticker: str, uid: str, *, first: date) -> TInvestInstrument:
    return TInvestInstrument(
        ticker=ticker,
        class_code="TQBR",
        instrument_uid=uid,
        figi=f"figi-{uid}",
        instrument_type="INSTRUMENT_TYPE_SHARE",
        first_1day_candle_date=first,
        name={
            "CURH": "Current Issuer",
            "MISS": "Missing Issuer",
            "LATE": "Late Issuer",
        }.get(ticker, ticker),
        exchange="moex_morning_weekend",
        currency="rub",
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
