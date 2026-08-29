from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from apps.cli.mature_consolidated_active_exact_historical import build_parser
from src.consolidated_active_exact_historical_maturation.application import (
    run_consolidated_active_exact_historical_maturation,
)
from src.consolidated_active_exact_historical_maturation.domain import (
    ARTIFACT_VERSION,
    DEFAULT_BASE_DATASET_ROOT,
    DEFAULT_LIVE_REGISTRY_PATH,
    DEFAULT_UNIVERSE_ROOT,
    DEFAULT_V1_ARTIFACT_ROOT,
    DEFAULT_V2_ARTIFACT_ROOT,
    EXPECTED_V1_ARTIFACT_SHA,
    HORIZONS,
    artifact_sha,
    safety_flags,
)
from src.exact_event_live_source_breadth_expansion_v2.domain import (
    artifact_sha as v2_artifact_sha,
)
from src.tinvest_market.client import (
    TInvestInstrument,
    TInvestMinuteCandle,
    TInvestMinuteCandleBatch,
)


def test_cli_defaults_to_consolidated_active_maturation_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "8" * 40])
    assert args.v1_dir == DEFAULT_V1_ARTIFACT_ROOT
    assert args.v2_dir == DEFAULT_V2_ARTIFACT_ROOT
    assert args.base_dataset_dir == DEFAULT_BASE_DATASET_ROOT
    assert args.live_registry == DEFAULT_LIVE_REGISTRY_PATH
    assert args.universe_dir == DEFAULT_UNIVERSE_ROOT
    assert args.output_dir == f"artifacts/{ARTIFACT_VERSION}"
    assert args.live_readonly is False


def test_safety_flags_forbid_future_targets_model_test_and_trading() -> None:
    flags = safety_flags()
    assert flags["DATA_MATURATION_ONLY"] is True
    assert flags["DATA_COST_RUB"] == 0
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["TEST_EVALUATION_PERFORMED"] is False
    assert flags["FUTURE_PRICE_LOOKUPS"] == 0
    assert flags["FUTURE_REACTIONS_COMPUTED"] == 0
    assert flags["FUTURE_TARGETS_COMPUTED"] == 0
    assert flags["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False
    assert flags["SANDBOX_ORDER_SUBMISSION_ALLOWED"] is False


@pytest.mark.asyncio
async def test_consolidated_active_maturation_guards_and_matures_fixture(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    client = _FakeActiveClient()

    manifest = await run_consolidated_active_exact_historical_maturation(
        v1_root=fixture["v1_root"],
        v2_root=fixture["v2_root"],
        base_dataset_root=fixture["base_root"],
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        live_registry_path=fixture["live_registry"],
        universe_root=fixture["universe_root"],
        client=client,
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
    )

    assert manifest["INPUT_V1_ARTIFACT_SHA"] == EXPECTED_V1_ARTIFACT_SHA
    assert manifest["V1_HISTORICAL_INPUT"] == 3
    assert manifest["V2_HISTORICAL_INPUT"] == 5
    assert manifest["COMBINED_HISTORICAL_INPUT"] == 8
    assert manifest["DEDUPED_HISTORICAL_INPUT"] == 7
    assert manifest["MARKET_ELIGIBLE_INPUT"] == 5
    assert manifest["MARKET_INELIGIBLE_INPUT"] == 2
    assert manifest["CANONICAL_EXACT_EVENTS_BEFORE"] == 10
    assert manifest["CANONICAL_EXACT_EVENTS_AFTER"] == 10
    assert manifest["MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_BEFORE"] == 2
    assert manifest["MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_AFTER"] == 7
    assert manifest["NEW_REACTION_READY"] == 2
    assert manifest["NEW_FEATURE_READY"] == 2
    assert all(manifest[f"NEW_{horizon.upper()}_READY"] == 2 for horizon in HORIZONS)
    assert manifest["FUTURE_PRICE_LOOKUPS"] == 0
    assert manifest["FUTURE_REACTIONS_COMPUTED"] == 0
    assert manifest["FUTURE_TARGETS_COMPUTED"] == 0
    assert manifest["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert manifest["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert manifest["EXISTING_CANONICAL_ROWS_PRESERVED"] == "PASS"
    assert manifest["LEAKAGE_CHECK"] == "PASS"
    assert manifest["ARTIFACT_SHA"] == artifact_sha(manifest)

    output_root = tmp_path / ARTIFACT_VERSION
    cohort = _read_jsonl(output_root / "maturation-cohort.jsonl")
    assert [row["event_id"] for row in cohort] == [
        "bmiss-new",
        "amb-new",
        "bad-new",
        "dup-new",
        "good-new",
        "miss-new",
        "pre-new",
    ]
    assert "chep-historical" not in {row["event_id"] for row in cohort}
    assert len([row for row in cohort if row["event_id"] == "dup-new"]) == 1

    future = _read_jsonl(output_root / "future-excluded-cohort.jsonl")
    assert [row["event_id"] for row in future] == ["future-new"]

    identities = {
        row["ticker"]: row for row in _read_jsonl(output_root / "instrument-identities.jsonl")
    }
    assert identities["AMB"]["identity_provenance"] == "AMBIGUOUS"
    assert identities["GOOD"]["identity_provenance"] == "LIVE_SOURCE_REGISTRY_WITH_TINVEST_UNIVERSE"

    results = {
        row["event_id"]: row for row in _read_jsonl(output_root / "maturation-results.jsonl")
    }
    assert results["good-new"]["reaction_ready"] is True
    assert results["dup-new"]["reaction_ready"] is True
    assert results["bad-new"]["primary_blocker"] == "TRADING_STATUS_UNVERIFIABLE"
    assert results["amb-new"]["primary_blocker"] == "INSTRUMENT_IDENTITY_AMBIGUOUS"
    assert results["miss-new"]["primary_blocker"] == "SECURITY_HISTORY_MISSING"
    assert results["bmiss-new"]["primary_blocker"] == "BENCHMARK_HISTORY_MISSING"
    assert results["pre-new"]["primary_blocker"] == "PRE_OPEN"
    assert (
        results["good-new"]["max_feature_timestamp_utc"] < results["good-new"]["published_at_utc"]
    )
    assert results["good-new"]["post_event_feature_access"] is False

    events = {row["metadata"]["event_id"]: row for row in _read_jsonl(output_root / "events.jsonl")}
    assert events["old-ready"] == _old_ready_event()
    assert events["future-new"]["target_availability"]["reaction_ready"] is False
    assert events["future-new"]["target_availability"]["feature_ready"] is False
    assert events["future-new"]["target_availability"]["research_outcomes_visible"] is False

    targets = {row["event_id"] for row in _read_jsonl(output_root / "targets.jsonl")}
    assert "future-new" not in targets
    assert {"good-new", "dup-new"} <= targets
    assert "uid-FUTR" not in client.requested_uids
    assert "uid-CHEP" not in client.requested_uids
    assert "uid-BAD" not in client.requested_uids
    assert "uid-AMB-a" not in client.requested_uids
    assert "uid-GOOD" in client.requested_uids


@pytest.mark.asyncio
async def test_consolidated_active_maturation_hashes_are_deterministic(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    left = await run_consolidated_active_exact_historical_maturation(
        v1_root=fixture["v1_root"],
        v2_root=fixture["v2_root"],
        base_dataset_root=fixture["base_root"],
        output_root=tmp_path / "left",
        base_main_sha="8" * 40,
        git_sha="9" * 40,
        live_registry_path=fixture["live_registry"],
        universe_root=fixture["universe_root"],
        client=_FakeActiveClient(),
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    right = await run_consolidated_active_exact_historical_maturation(
        v1_root=fixture["v1_root"],
        v2_root=fixture["v2_root"],
        base_dataset_root=fixture["base_root"],
        output_root=tmp_path / "right",
        base_main_sha="8" * 40,
        git_sha="0" * 40,
        live_registry_path=fixture["live_registry"],
        universe_root=fixture["universe_root"],
        client=_FakeActiveClient(),
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    assert left["MATURATION_COHORT_SHA"] == right["MATURATION_COHORT_SHA"]
    assert left["FUTURE_EXCLUDED_COHORT_SHA"] == right["FUTURE_EXCLUDED_COHORT_SHA"]
    assert left["OUTPUT_DATASET_SHA"] == right["OUTPUT_DATASET_SHA"]
    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    v1_root = tmp_path / "v1"
    v2_root = tmp_path / "v2"
    base_root = tmp_path / "base"
    universe_root = tmp_path / "universe"
    live_registry = tmp_path / "live-registry.json"
    for path in (
        v1_root / "live-collection",
        v2_root / "live-collection",
        base_root,
        universe_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    v1_rows = [
        _event("good-new", "GOOD", "2026-07-20T10:00:30+00:00"),
        _event("dup-new", "DUP", "2026-07-20T10:00:30+00:00"),
        _event("dup-new", "DUP", "2026-07-20T10:00:30+00:00"),
        _event("future-new", "FUTR", "2026-08-13T07:00:00+00:00", future=True),
        _event("chep-historical", "CHEP", "2026-07-20T10:00:30+00:00"),
    ]
    v2_rows = [
        _event("bmiss-new", "BMISS", "2026-06-01T10:00:30+00:00"),
        _event("bad-new", "BAD", "2026-07-20T10:00:30+00:00"),
        _event("amb-new", "AMB", "2026-07-20T10:00:30+00:00"),
        _event("miss-new", "MISS", "2026-07-21T10:00:30+00:00"),
        _event("pre-new", "PRE", "2026-07-23T08:00:00+00:00"),
    ]
    _write_jsonl(v1_root / "live-collection" / "collected-event-metadata.jsonl", v1_rows)
    _write_jsonl(v2_root / "live-collection" / "collected-event-metadata.jsonl", v2_rows)

    _write_json(v1_root / "manifest.json", _v1_manifest())
    _write_json(v2_root / "manifest.json", _v2_manifest())

    source_rows = [
        _source("GOOD", "uid-GOOD", "source-good"),
        _source("DUP", "uid-DUP", "source-dup"),
        _source("FUTR", "uid-FUTR", "source-futr"),
        _source("CHEP", "uid-CHEP", "source-chep"),
        _source("BMISS", "uid-BMISS", "source-bmiss"),
        _source("BAD", "uid-BAD", "source-bad"),
        _source("AMB", "uid-AMB-a", "source-amb-a"),
        _source("AMB", "uid-AMB-b", "source-amb-b"),
        _source("MISS", "uid-MISS", "source-miss"),
        _source("PRE", "uid-PRE", "source-pre"),
    ]
    _write_json(v1_root / "collection-source-registry.json", {"sources": source_rows[:4]})
    _write_json(v2_root / "collection-source-registry.json", {"sources": source_rows[4:]})
    _write_json(live_registry, {"sources": source_rows})

    _write_jsonl(base_root / "events.jsonl", [_old_ready_event()])
    _write_jsonl(
        base_root / "features.jsonl",
        [{"event_id": "old-ready", "event_features": {"primary_event_type": "DIVIDEND"}}],
    )
    _write_jsonl(base_root / "targets.jsonl", [{"event_id": "old-ready", "horizon": "1m"}])
    _write_candle_cache(
        base_root / "raw-minute-cache",
        "IMOEX",
        "uid-IMOEX",
        "2026-01-01",
        ("2026-01-01T10:00:00+00:00",),
    )

    universe_rows = [
        _universe("GOOD", "uid-GOOD"),
        _universe("DUP", "uid-DUP"),
        _universe("BMISS", "uid-BMISS"),
        _universe("BAD", "uid-BAD", currency="usd"),
        _universe("AMB", "uid-AMB-a"),
        _universe("MISS", "uid-MISS"),
        _universe("PRE", "uid-PRE"),
    ]
    _write_jsonl(universe_root / "history-coverage.jsonl", universe_rows)
    _write_jsonl(universe_root / "discovery-shares.jsonl", universe_rows)
    return {
        "v1_root": v1_root,
        "v2_root": v2_root,
        "base_root": base_root,
        "universe_root": universe_root,
        "live_registry": live_registry,
    }


class _FakeActiveClient:
    def __init__(self) -> None:
        self.requested_uids: list[str] = []

    async def get_instrument_by_uid(self, instrument_uid: str) -> TInvestInstrument:
        return TInvestInstrument(
            ticker=instrument_uid.removeprefix("uid-"),
            class_code="TQBR",
            instrument_uid=instrument_uid,
            figi=f"figi-{instrument_uid}",
            instrument_type="INSTRUMENT_TYPE_SHARE",
            first_1day_candle_date=None,
            name=instrument_uid,
        )

    async def fetch_minute_candles_audited(
        self, *, instrument_uid: str, date_from: datetime, date_to: datetime
    ) -> TInvestMinuteCandleBatch:
        self.requested_uids.append(instrument_uid)
        day = date_from.date().isoformat()
        if instrument_uid == "uid-MISS":
            return TInvestMinuteCandleBatch((), ())
        if instrument_uid == "uid-IMOEX" and day.startswith("2026-05"):
            return TInvestMinuteCandleBatch((), ())
        if instrument_uid == "uid-IMOEX" and day == "2026-06-01":
            return TInvestMinuteCandleBatch((), ())
        if instrument_uid == "uid-PRE" and day == "2026-07-23":
            return TInvestMinuteCandleBatch(_candles(instrument_uid, _pre_open_times(day)), ())
        if instrument_uid == "uid-IMOEX" and day == "2026-07-23":
            return TInvestMinuteCandleBatch(_candles(instrument_uid, _pre_open_times(day)), ())
        if day in {"2026-07-20", "2026-07-21"} or (
            instrument_uid == "uid-BMISS" and day == "2026-06-01"
        ):
            return TInvestMinuteCandleBatch(_candles(instrument_uid, _complete_times(day)), ())
        return TInvestMinuteCandleBatch((), ())


def _v1_manifest() -> dict[str, Any]:
    return {
        "ARTIFACT_SHA": EXPECTED_V1_ARTIFACT_SHA,
        "NEW_EXACT_LIVE_SOURCES": 5,
        "NEW_CANONICAL_EXACT_EVENTS": 56,
        "NEW_HISTORICAL_EXACT_EVENTS": 45,
        "NEW_FUTURE_METADATA_ONLY_EVENTS": 11,
        "REPLAY_ITEMS_NEW": 0,
        "REPLAY_ITEMS_DUPLICATE": 56,
        "FINAL_DECISION": "SOURCE_BREADTH_GAINED",
        "NEW_TICKERS_WITH_EXACT_SOURCE": ["AFKS", "ASTR", "ELMT", "OZON", "RUAL"],
        "REACTION_READY_EVENTS_AFTER": 1,
        "FEATURE_READY_EVENTS_AFTER": 1,
    }


def _v2_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": "exact-event-live-source-breadth-expansion-v2",
        "NEW_EXACT_LIVE_SOURCES": 5,
        "REPLAY_ITEMS_NEW": 0,
        "FINAL_DECISION": "SOURCE_BREADTH_GAINED",
        "NEW_TICKERS_WITH_EXACT_SOURCE": ["GOOD", "DUP", "BMISS", "BAD", "PRE"],
        "CANONICAL_EXACT_EVENTS_AFTER": 10,
        "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_AFTER": 2,
    }
    manifest["ARTIFACT_SHA"] = v2_artifact_sha(manifest)
    return manifest


def _event(
    event_id: str, ticker: str, published_at: str, *, future: bool = False
) -> dict[str, object]:
    published = datetime.fromisoformat(published_at)
    return {
        "metadata": {
            "event_id": event_id,
            "source_code": "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "source_id": f"source-{ticker.lower()}",
            "source_item_id": event_id,
            "canonical_url": f"https://issuer.example/{event_id}",
            "ticker": ticker,
            "issuer": f"{ticker} Issuer",
            "instrument_uid": f"uid-{ticker}",
            "publication_timestamp_utc": published.isoformat(),
            "publication_timestamp_raw": published.isoformat(),
            "publication_date": published.date().isoformat(),
            "publication_time": published.time().isoformat(),
            "publication_timezone": "UTC",
            "timestamp_source_field": "synthetic exact timestamp",
            "timestamp_quality": "EXACT",
            "future_holdout": future,
            "future_holdout_metadata_only": future,
            "session_state": "FUTURE_METADATA_ONLY" if future else "MARKET_CONTEXT_NOT_BUILT",
            "title_hash": event_id,
        },
        "event_features": {"primary_event_type": "DIVIDEND", "event_count": 1, "fact_count": 1},
        "pre_event_market_features": {},
        "target_availability": {
            "research_outcomes_visible": False,
            "reaction_ready": False,
            "feature_ready": False,
            "status": "FUTURE_METADATA_ONLY" if future else "MARKET_CONTEXT_NOT_BUILT",
            "missing_reason": "FUTURE_HOLDOUT_OUTCOMES_GUARDED"
            if future
            else "ACTIVE_MATURATION_PENDING",
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


def _old_ready_event() -> dict[str, object]:
    event = _event("old-ready", "OLD", "2026-07-20T10:00:30+00:00")
    cast("dict[str, object]", event["target_availability"]).update(
        {
            "research_outcomes_visible": True,
            "reaction_ready": True,
            "feature_ready": True,
            "status": "REACTION_READY",
            "missing_reason": None,
        }
    )
    return event


def _source(ticker: str, uid: str, source_id: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "issuer": f"{ticker} Issuer",
        "instrument_uid": uid,
        "source_id": source_id,
        "source_family": "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
        "source_url": f"https://www.moex.com/{source_id}.rss",
        "official_domain": "moex.com",
    }


def _universe(ticker: str, uid: str, *, currency: str = "rub") -> dict[str, object]:
    return {
        "ticker": ticker,
        "instrument_uid": uid,
        "figi": f"figi-{ticker}",
        "class_code": "TQBR",
        "exchange": "MOEX",
        "currency": currency,
    }


def _complete_times(day: str) -> tuple[str, ...]:
    return (
        f"{day}T08:59:00+00:00",
        f"{day}T09:29:00+00:00",
        f"{day}T09:44:00+00:00",
        f"{day}T09:54:00+00:00",
        f"{day}T09:59:00+00:00",
        f"{day}T10:01:00+00:00",
        f"{day}T10:05:00+00:00",
        f"{day}T10:15:00+00:00",
        f"{day}T10:30:00+00:00",
        f"{day}T11:00:00+00:00",
    )


def _pre_open_times(day: str) -> tuple[str, ...]:
    return (f"{day}T09:00:00+00:00", f"{day}T09:01:00+00:00")


def _candles(uid: str, begin_times: Sequence[str]) -> tuple[TInvestMinuteCandle, ...]:
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


def _write_candle_cache(
    root: Path, ticker: str, uid: str, day: str, begin_times: Sequence[str]
) -> None:
    rows: list[dict[str, object]] = []
    for candle in _candles(uid, begin_times):
        rows.append(
            {
                "ticker": ticker,
                "figi": f"figi-{ticker}",
                "instrument_uid": uid,
                "class_code": "TQBR",
                "interval": "1m",
                "begin_at": candle.begin_at.isoformat(),
                "end_at": candle.end_at.isoformat(),
                "open": str(candle.open),
                "high": str(candle.high),
                "low": str(candle.low),
                "close": str(candle.close),
                "volume": candle.volume,
                "is_complete": candle.is_complete,
                "source": "TINVEST_API",
                "provenance": "TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES",
            }
        )
    _write_jsonl(root / ticker / f"{day}-day.jsonl", rows)


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
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
