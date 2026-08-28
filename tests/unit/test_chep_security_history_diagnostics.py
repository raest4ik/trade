from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from apps.cli.diagnose_chep_security_history import build_parser
from src.chep_security_history_diagnostics.application import (
    run_chep_security_history_diagnostics,
)
from src.chep_security_history_diagnostics.domain import (
    ARTIFACT_VERSION,
    EXPECTED_CHEP_FIGI,
    EXPECTED_CHEP_UID,
    CandidateClassification,
    FutureHoldoutProbeError,
    PrimaryRootCause,
    build_probe_windows,
    choose_primary_root_cause,
    classify_candidate,
    diagnostics_safety_flags,
    guard_no_future_probe,
    sha256_payload,
)
from src.tinvest_market.client import (
    TInvestCandleBatch,
    TInvestDailyCandle,
    TInvestInstrument,
    TInvestMinuteCandle,
    TInvestMinuteCandleBatch,
)


def test_cli_defaults_to_chep_diagnostics_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "a" * 40])
    assert args.input_dir == "artifacts/chep-historical-exact-maturation-v1"
    assert args.output_dir == "artifacts/chep-security-history-diagnostics-v1"
    assert args.live_readonly is False
    assert args.moex is False


def test_safety_flags_forbid_future_training_and_trading() -> None:
    flags = diagnostics_safety_flags()
    assert flags["DIAGNOSTICS_ONLY"] is True
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["TEST_EVALUATION_PERFORMED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert flags["FUTURE_CHEP_PRICE_LOOKUPS"] == 0
    assert flags["MOEX_SUBSTITUTION_USED"] is False
    assert flags["SYNTHETIC_MARKET_DATA_USED"] is False
    assert flags["LOCAL_ACQUISITION_LOGIC_ROOT_CAUSE"] is False


def test_instrument_candidate_classification() -> None:
    assert (
        classify_candidate(_candidate(uid=EXPECTED_CHEP_UID, figi=EXPECTED_CHEP_FIGI))
        == CandidateClassification.CURRENT_CONFIRMED
    )
    assert (
        classify_candidate(_candidate(uid="uid-old", figi=EXPECTED_CHEP_FIGI))
        == CandidateClassification.HISTORICAL_CONFIRMED
    )
    assert (
        classify_candidate(_candidate(uid="uid-old", figi=None))
        == CandidateClassification.LEGACY_POSSIBLE
    )
    assert (
        classify_candidate(_candidate(uid=EXPECTED_CHEP_UID, figi=None, ticker="OTHER"))
        == CandidateClassification.AMBIGUOUS
    )
    assert (
        classify_candidate(_candidate(uid="other", figi="other", ticker="GAZP"))
        == CandidateClassification.UNRELATED
    )


def test_bounded_probe_construction_and_future_rejection() -> None:
    rows = [
        _cohort_row("early", "2026-07-01T10:00:00+00:00"),
        _cohort_row("middle", "2026-07-10T10:00:00+00:00"),
        _cohort_row("late", "2026-08-10T10:00:00+00:00"),
    ]
    windows = build_probe_windows(rows)
    assert [window.event_id for window in windows] == ["early", "middle", "late"]
    assert windows[0].minute_from == datetime(2026, 7, 1, 9, 50, tzinfo=UTC)
    assert windows[0].minute_to == datetime(2026, 7, 1, 11, 10, tzinfo=UTC)
    assert windows[0].daily_from == date(2026, 6, 29)
    assert windows[0].daily_to == date(2026, 7, 3)
    try:
        guard_no_future_probe(datetime(2026, 8, 11, 0, 0, tzinfo=UTC))
    except FutureHoldoutProbeError as exc:
        assert str(exc) == "FUTURE_EVENT_HOLDOUT_READ_ATTEMPT"
    else:
        raise AssertionError("future holdout guard did not fire")


async def test_zero_candle_result_classifies_historical_security_not_supported(
    tmp_path: Path,
) -> None:
    input_root = _write_input_artifact(tmp_path)
    client = _FakeChepDiagnosticsClient(
        instruments={
            EXPECTED_CHEP_UID: _instrument(
                uid=EXPECTED_CHEP_UID,
                first=date(2009, 1, 30),
                last=date(2021, 9, 21),
                trade_available=False,
            )
        },
        query_results={"CHEP": (EXPECTED_CHEP_UID,), EXPECTED_CHEP_FIGI: (EXPECTED_CHEP_UID,)},
        daily_history={},
        minute_history={},
    )
    manifest = await run_chep_security_history_diagnostics(
        input_root=input_root,
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="a" * 40,
        git_sha="b" * 40,
        client=client,
        moex_client=None,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
    )
    assert manifest["PRIMARY_ROOT_CAUSE"] == PrimaryRootCause.HISTORICAL_SECURITY_NOT_SUPPORTED
    assert manifest["MINUTE_PROBES_WITH_DATA"] == 0
    assert manifest["DAILY_PROBES_WITH_DATA"] == 0
    report = json.loads((tmp_path / ARTIFACT_VERSION / "diagnostic-report.json").read_text())
    assert report["local_implementation_audit"]["LOCAL_ACQUISITION_LOGIC_ROOT_CAUSE"] is False


async def test_daily_present_minute_absent_classification(tmp_path: Path) -> None:
    input_root = _write_input_artifact(tmp_path)
    client = _FakeChepDiagnosticsClient(
        instruments={EXPECTED_CHEP_UID: _instrument(uid=EXPECTED_CHEP_UID)},
        query_results={"CHEP": (EXPECTED_CHEP_UID,), EXPECTED_CHEP_FIGI: (EXPECTED_CHEP_UID,)},
        daily_history={EXPECTED_CHEP_UID: True},
        minute_history={},
    )
    manifest = await run_chep_security_history_diagnostics(
        input_root=input_root,
        output_root=tmp_path / "daily-present",
        base_main_sha="a" * 40,
        git_sha="b" * 40,
        client=client,
        moex_client=None,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
    )
    assert manifest["PRIMARY_ROOT_CAUSE"] == PrimaryRootCause.TINVEST_MINUTE_HISTORY_UNAVAILABLE
    assert manifest["DAILY_PROBES_WITH_DATA"] > 0
    assert manifest["MINUTE_PROBES_WITH_DATA"] == 0


def test_ambiguous_identity_fails_closed() -> None:
    root = choose_primary_root_cause(
        candidate_rows=[
            {
                **_candidate(uid="uid-current", figi=None, ticker="OTHER"),
                "classification": CandidateClassification.AMBIGUOUS,
            }
        ],
        candle_probes=[],
        local_acquisition_logic_root_cause=False,
        moex_event_date_trading_confirmed=None,
    )
    assert root == PrimaryRootCause.IDENTITY_AMBIGUOUS


async def test_no_silent_legacy_substitution(tmp_path: Path) -> None:
    input_root = _write_input_artifact(tmp_path)
    client = _FakeChepDiagnosticsClient(
        instruments={
            EXPECTED_CHEP_UID: _instrument(uid=EXPECTED_CHEP_UID),
            "uid-legacy": _instrument(uid="uid-legacy", figi=EXPECTED_CHEP_FIGI),
        },
        query_results={
            "CHEP": (EXPECTED_CHEP_UID, "uid-legacy"),
            EXPECTED_CHEP_FIGI: ("uid-legacy",),
        },
        daily_history={},
        minute_history={"uid-legacy": True},
    )
    await run_chep_security_history_diagnostics(
        input_root=input_root,
        output_root=tmp_path / "legacy",
        base_main_sha="a" * 40,
        git_sha="b" * 40,
        client=client,
        moex_client=None,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
    )
    rows = _read_jsonl(tmp_path / "legacy" / "instrument-candidates.jsonl")
    legacy = next(row for row in rows if row["instrument_uid"] == "uid-legacy")
    assert legacy["classification"] == CandidateClassification.HISTORICAL_CONFIRMED
    assert legacy["canonical_substitution_allowed"] is False


async def test_deterministic_report_hash_generation(tmp_path: Path) -> None:
    input_root = _write_input_artifact(tmp_path)
    first = await run_chep_security_history_diagnostics(
        input_root=input_root,
        output_root=tmp_path / "first",
        base_main_sha="a" * 40,
        git_sha="b" * 40,
        client=None,
        moex_client=None,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
    )
    second = await run_chep_security_history_diagnostics(
        input_root=input_root,
        output_root=tmp_path / "second",
        base_main_sha="a" * 40,
        git_sha="b" * 40,
        client=None,
        moex_client=None,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
    )
    assert first["DIAGNOSTIC_REPORT_SHA"] == second["DIAGNOSTIC_REPORT_SHA"]
    assert first["ARTIFACT_SHA"] == sha256_payload({**first, "ARTIFACT_SHA": None})


class _FakeChepDiagnosticsClient:
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

    async def find_instruments(
        self, query: str, *, instrument_kind: str
    ) -> tuple[TInvestInstrument, ...]:
        assert instrument_kind == "INSTRUMENT_TYPE_SHARE"
        return tuple(self._instruments[uid] for uid in self._query_results.get(query, ()))

    async def get_instrument_by_uid(self, instrument_uid: str) -> TInvestInstrument:
        return self._instruments[instrument_uid]

    async def list_shares(self) -> tuple[TInvestInstrument, ...]:
        return tuple(self._instruments.values())

    async def fetch_daily_candles_audited(
        self, *, instrument_uid: str, date_from: date, date_to: date
    ) -> TInvestCandleBatch:
        candles = (
            (
                TInvestDailyCandle(
                    instrument_uid=instrument_uid,
                    trade_date=date_from,
                    open=Decimal("10"),
                    high=Decimal("11"),
                    low=Decimal("9"),
                    close=Decimal("10"),
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
        candles = (
            (
                TInvestMinuteCandle(
                    instrument_uid=instrument_uid,
                    begin_at=date_from,
                    end_at=date_from + timedelta(minutes=1),
                    open=Decimal("10"),
                    high=Decimal("11"),
                    low=Decimal("9"),
                    close=Decimal("10"),
                    volume=100,
                    is_complete=True,
                ),
            )
            if self._minute_history.get(instrument_uid, False)
            else ()
        )
        return TInvestMinuteCandleBatch(candles, ())


def _write_input_artifact(tmp_path: Path) -> Path:
    root = tmp_path / "input"
    root.mkdir()
    start = datetime(2026, 6, 28, 10, 0, tzinfo=UTC)
    historical = [
        _cohort_row(f"event-{index:02d}", (start + timedelta(days=index)).isoformat())
        for index in range(44)
    ]
    future = [
        _cohort_row(f"future-{index}", f"2026-08-{11 + index:02d}T10:00:00+00:00")
        for index in range(6)
    ]
    identity = {
        "ticker": "CHEP",
        "figi": EXPECTED_CHEP_FIGI,
        "instrument_uid": EXPECTED_CHEP_UID,
        "class_code": "TQBR",
        "issuer": "CHEP issuer",
        "exchange": "moex",
        "currency": "rub",
        "first_1day_candle_date": "2009-01-30",
        "last_1day_candle_date": "2021-09-21",
    }
    _write_json(
        root / "manifest.json",
        {
            "ARTIFACT_SHA": "236ab1579cafda265eceeefc148b359d3ab2e5c54538d1d434bc789fc5775305",
            "HISTORICAL_COHORT_SHA": (
                "b7cff6dd3a94df7560da4565a0c06cca42dae5c1dd2ca0bc0cf736b54c10092e"
            ),
            "FUTURE_METADATA_COHORT_SHA": (
                "ae1568923e517968020bf108ebf59b0371720f4567c21ee2fef0ea59e7be35c3"
            ),
            "INSTRUMENT_IDENTITY_SHA": (
                "5f4945a81aca2d55916ccf379e03b8496d765da7229984d0014410af3c5d6c56"
            ),
            "CHEP_HISTORICAL_EVENTS_TOTAL": 44,
            "CHEP_REACTION_READY": 0,
            "CHEP_FEATURE_READY": 0,
            "BLOCKER_COUNTS": {"FUTURE_METADATA_ONLY": 6, "SECURITY_HISTORY_MISSING": 44},
            "FUTURE_EVENT_HOLDOUT_USED": False,
            "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
            "MODEL_TRAINING_PERFORMED": False,
            "TEST_OUTCOME_USED": False,
            "TEST_EVALUATION_PERFORMED": False,
        },
    )
    _write_jsonl(root / "historical-cohort.jsonl", historical)
    _write_jsonl(root / "future-metadata-cohort.jsonl", future)
    _write_jsonl(root / "instrument-identity.jsonl", [identity])
    return root


def _cohort_row(event_id: str, published_at: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "ticker": "CHEP",
        "publication_timestamp_utc": published_at,
        "source_item_id": f"https://example.test/{event_id}",
    }


def _candidate(
    *,
    uid: str,
    figi: str | None,
    ticker: str = "CHEP",
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "figi": figi,
        "instrument_uid": uid,
        "class_code": "TQBR",
        "instrument_type": "INSTRUMENT_TYPE_SHARE",
        "name": ticker,
    }


def _instrument(
    *,
    uid: str,
    figi: str = EXPECTED_CHEP_FIGI,
    first: date = date(2009, 1, 30),
    last: date | None = None,
    trade_available: bool = True,
) -> TInvestInstrument:
    return TInvestInstrument(
        ticker="CHEP",
        class_code="TQBR",
        instrument_uid=uid,
        figi=figi,
        instrument_type="INSTRUMENT_TYPE_SHARE",
        first_1day_candle_date=first,
        name="CHEP",
        exchange="moex",
        currency="rub",
        trading_status=(
            "SECURITY_TRADING_STATUS_NORMAL_TRADING"
            if trade_available
            else "SECURITY_TRADING_STATUS_NOT_AVAILABLE_FOR_TRADING"
        ),
        api_trade_available=trade_available,
        buy_available=trade_available,
        sell_available=trade_available,
        last_1day_candle_date=last,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
