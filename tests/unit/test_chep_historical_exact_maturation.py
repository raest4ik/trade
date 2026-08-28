from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from apps.cli.mature_chep_historical_exact import build_parser
from src.chep_historical_exact_maturation.application import (
    run_chep_historical_exact_maturation,
    split_chep_collector_rows,
)
from src.chep_historical_exact_maturation.domain import (
    ARTIFACT_VERSION,
    EXPECTED_COLLECTOR_ARTIFACT_SHA,
    EXPECTED_COLLECTOR_DEDUPE_STATE_SHA,
    EXPECTED_COLLECTOR_EVENT_METADATA_SHA,
    EXPECTED_COLLECTOR_NETWORK_PROVENANCE_SHA,
    EXPECTED_COLLECTOR_RAW_SNAPSHOT_SHA,
    EXPECTED_COLLECTOR_SOURCE_REGISTRY_SHA,
    FutureHoldoutReadAttemptError,
    guard_future_market_access,
    maturation_safety_flags,
)


def test_cli_defaults_to_chep_maturation_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "a" * 40])
    assert args.collector_dir == "artifacts/exact-event-live-official-collection-v1"
    assert args.base_dataset_dir == "artifacts/exact-event-new-source-maturation-v1"
    assert args.output_dir == "artifacts/chep-historical-exact-maturation-v1"


def test_safety_flags_keep_model_test_future_and_trading_frozen() -> None:
    flags = maturation_safety_flags()
    assert flags["DATA_MATURATION_ONLY"] is True
    assert flags["DATA_COST_RUB"] == 0
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["TEST_EVALUATION_PERFORMED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False
    assert flags["SANDBOX_ORDER_SUBMISSION_ALLOWED"] is False


def test_split_collector_rows_has_expected_44_6_boundary() -> None:
    rows = _collector_rows()
    historical, future = split_chep_collector_rows(rows)
    assert len(historical) == 44
    assert len(future) == 6
    assert _metadata(historical[-1])["publication_timestamp_utc"] < "2026-08-11T00:00:00+00:00"
    assert _metadata(future[0])["publication_timestamp_utc"] >= "2026-08-11T00:00:00+00:00"


def test_future_market_access_fails_closed() -> None:
    guard_future_market_access(datetime(2026, 8, 10, 23, 59, 59, tzinfo=UTC))
    with pytest.raises(FutureHoldoutReadAttemptError, match="FUTURE_EVENT_HOLDOUT_READ_ATTEMPT"):
        guard_future_market_access(datetime(2026, 8, 11, tzinfo=UTC))


@pytest.mark.asyncio
async def test_cache_only_run_preserves_future_and_reports_missing_security(tmp_path: Path) -> None:
    collector_root = _write_collector_artifact(tmp_path)
    base_root = _write_base_dataset(tmp_path)
    output_root = tmp_path / ARTIFACT_VERSION

    manifest = await run_chep_historical_exact_maturation(
        collector_root=collector_root,
        base_dataset_root=base_root,
        output_root=output_root,
        base_main_sha="b" * 40,
        git_sha="c" * 40,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
        universe_root=_write_universe(tmp_path),
    )

    assert manifest["INPUT_COLLECTOR_ARTIFACT_SHA"] == EXPECTED_COLLECTOR_ARTIFACT_SHA
    assert manifest["CHEP_HISTORICAL_EVENTS_TOTAL"] == 44
    assert manifest["FUTURE_CHEP_EVENTS"] == 6
    assert manifest["CHEP_REACTION_READY"] == 0
    assert manifest["CHEP_FEATURE_READY"] == 0
    assert manifest["BLOCKER_COUNTS"] == {
        "FUTURE_METADATA_ONLY": 6,
        "SECURITY_HISTORY_MISSING": 44,
    }
    assert manifest["FUTURE_CHEP_PRICE_LOOKUPS"] == 0
    assert manifest["FUTURE_CHEP_REACTIONS_COMPUTED"] == 0
    assert manifest["FUTURE_CHEP_TARGETS_COMPUTED"] == 0
    assert manifest["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert manifest["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert manifest["EXISTING_CANONICAL_ROWS_PRESERVED"] == "PASS"
    assert manifest["FINAL_DECISION"] == "CHEP_MARKET_HISTORY_RECOVERY_NEXT"

    rows = _read_jsonl(output_root / "per-event-maturation.jsonl")
    future = [row for row in rows if row["historical_or_future"] == "FUTURE_METADATA_ONLY"]
    assert len(future) == 6
    assert {row["market_price_lookup_performed"] for row in future} == {False}


@pytest.mark.asyncio
async def test_historical_event_can_become_reaction_ready_with_complete_cache(
    tmp_path: Path,
) -> None:
    collector_root = _write_collector_artifact(tmp_path)
    base_root = _write_base_dataset(tmp_path)
    cache_root = tmp_path / "cache"
    _write_candle_cache(cache_root, "CHEP", "b1f4f4fc-dac5-4e29-ae56-95fe441416ee")
    _write_candle_cache(cache_root, "IMOEX", "uid-IMOEX")

    manifest = await run_chep_historical_exact_maturation(
        collector_root=collector_root,
        base_dataset_root=base_root,
        output_root=tmp_path / "ready",
        base_main_sha="b" * 40,
        git_sha="c" * 40,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
        extra_cache_roots=(cache_root,),
        universe_root=_write_universe(tmp_path),
    )

    assert manifest["CHEP_REACTION_READY"] == 1
    assert manifest["CHEP_FEATURE_READY"] == 0
    assert manifest["CHEP_1M_READY"] == 1
    assert manifest["CHEP_5M_READY"] == 1
    assert manifest["CHEP_15M_READY"] == 1
    assert manifest["CHEP_30M_READY"] == 1
    assert manifest["CHEP_60M_READY"] == 1
    assert manifest["BLOCKER_COUNTS"]["EVENT_FEATURES_MISSING"] == 1


@pytest.mark.asyncio
async def test_missing_benchmark_blocker_is_preserved(tmp_path: Path) -> None:
    collector_root = _write_collector_artifact(tmp_path)
    base_root = _write_base_dataset(tmp_path)
    cache_root = tmp_path / "cache"
    _write_candle_cache(cache_root, "CHEP", "b1f4f4fc-dac5-4e29-ae56-95fe441416ee")

    manifest = await run_chep_historical_exact_maturation(
        collector_root=collector_root,
        base_dataset_root=base_root,
        output_root=tmp_path / "missing-benchmark",
        base_main_sha="b" * 40,
        git_sha="c" * 40,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
        extra_cache_roots=(cache_root,),
        universe_root=_write_universe(tmp_path),
    )

    assert manifest["BLOCKER_COUNTS"]["BENCHMARK_HISTORY_MISSING"] == 44
    assert manifest["CHEP_REACTION_READY"] == 0


@pytest.mark.asyncio
async def test_strict_session_failure_is_preserved(tmp_path: Path) -> None:
    collector_root = _write_collector_artifact(tmp_path)
    base_root = _write_base_dataset(tmp_path)
    cache_root = tmp_path / "cache"
    _write_candle_cache(
        cache_root,
        "CHEP",
        "b1f4f4fc-dac5-4e29-ae56-95fe441416ee",
        begin_times=("2026-07-06T12:00:00+00:00", "2026-07-06T12:01:00+00:00"),
    )
    _write_candle_cache(
        cache_root,
        "IMOEX",
        "uid-IMOEX",
        begin_times=("2026-07-06T12:00:00+00:00", "2026-07-06T12:01:00+00:00"),
    )

    manifest = await run_chep_historical_exact_maturation(
        collector_root=collector_root,
        base_dataset_root=base_root,
        output_root=tmp_path / "session",
        base_main_sha="b" * 40,
        git_sha="c" * 40,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
        extra_cache_roots=(cache_root,),
        universe_root=_write_universe(tmp_path),
    )

    assert manifest["BLOCKER_COUNTS"]["PRE_OPEN"] == 1
    assert manifest["CHEP_REACTION_READY"] == 0


@pytest.mark.asyncio
async def test_feature_leakage_guard_rejects_equal_cutoff_cache(tmp_path: Path) -> None:
    collector_root = _write_collector_artifact(
        tmp_path, first_historical_at="2026-07-06T09:00:00+00:00"
    )
    base_root = _write_base_dataset(tmp_path)
    cache_root = tmp_path / "cache"
    _write_candle_cache(
        cache_root,
        "CHEP",
        "b1f4f4fc-dac5-4e29-ae56-95fe441416ee",
        begin_times=(
            "2026-07-06T08:59:00+00:00",
            "2026-07-06T09:00:00+00:00",
            "2026-07-06T09:01:00+00:00",
            "2026-07-06T09:05:00+00:00",
            "2026-07-06T09:15:00+00:00",
            "2026-07-06T09:30:00+00:00",
            "2026-07-06T10:00:00+00:00",
        ),
    )
    _write_candle_cache(
        cache_root,
        "IMOEX",
        "uid-IMOEX",
        begin_times=(
            "2026-07-06T08:59:00+00:00",
            "2026-07-06T09:00:00+00:00",
            "2026-07-06T09:01:00+00:00",
            "2026-07-06T09:05:00+00:00",
            "2026-07-06T09:15:00+00:00",
            "2026-07-06T09:30:00+00:00",
            "2026-07-06T10:00:00+00:00",
        ),
    )

    with pytest.raises(ValueError, match="CHEP_FEATURE_LEAKAGE_CHECK_FAILED"):
        await run_chep_historical_exact_maturation(
            collector_root=collector_root,
            base_dataset_root=base_root,
            output_root=tmp_path / "leak",
            base_main_sha="b" * 40,
            git_sha="c" * 40,
            created_at=datetime(2026, 8, 28, tzinfo=UTC),
            extra_cache_roots=(cache_root,),
            universe_root=_write_universe(tmp_path),
        )


@pytest.mark.asyncio
async def test_existing_canonical_row_change_fails_closed(tmp_path: Path) -> None:
    collector_root = _write_collector_artifact(tmp_path)
    base_root = _write_base_dataset(tmp_path)
    row = _read_jsonl(base_root / "events.jsonl")[0]
    cast("dict[str, Any]", row["metadata"])["ticker"] = "MUTATED"
    _write_jsonl(base_root / "events.jsonl", [row])

    manifest = await run_chep_historical_exact_maturation(
        collector_root=collector_root,
        base_dataset_root=base_root,
        output_root=tmp_path / "preserved",
        base_main_sha="b" * 40,
        git_sha="c" * 40,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
        universe_root=_write_universe(tmp_path),
    )

    assert manifest["EXISTING_CANONICAL_ROWS_PRESERVED"] == "PASS"
    assert _metadata(_read_jsonl(tmp_path / "preserved" / "events.jsonl")[0])["ticker"] == "MUTATED"


@pytest.mark.asyncio
async def test_deterministic_replay_hashes(tmp_path: Path) -> None:
    collector_root = _write_collector_artifact(tmp_path)
    base_root = _write_base_dataset(tmp_path)
    universe_root = _write_universe(tmp_path)
    left = await run_chep_historical_exact_maturation(
        collector_root=collector_root,
        base_dataset_root=base_root,
        output_root=tmp_path / "left",
        base_main_sha="b" * 40,
        git_sha="c" * 40,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
        universe_root=universe_root,
    )
    right = await run_chep_historical_exact_maturation(
        collector_root=collector_root,
        base_dataset_root=base_root,
        output_root=tmp_path / "right",
        base_main_sha="b" * 40,
        git_sha="c" * 40,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
        universe_root=universe_root,
    )
    assert left["HISTORICAL_COHORT_SHA"] == right["HISTORICAL_COHORT_SHA"]
    assert left["FUTURE_METADATA_COHORT_SHA"] == right["FUTURE_METADATA_COHORT_SHA"]
    assert left["OUTPUT_DATASET_SHA"] == right["OUTPUT_DATASET_SHA"]
    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]


def test_documentation_states_safety_boundaries() -> None:
    text = (
        Path(__file__).parents[2] / "docs" / "chep-historical-exact-maturation-v1.md"
    ).read_text(encoding="utf-8")
    assert "DATA_MATURATION_ONLY=true" in text
    assert "MODEL_TRAINING_PERFORMED=false" in text
    assert "TEST_OUTCOME_USED=false" in text
    assert "FUTURE_EVENT_HOLDOUT_OBSERVED=false" in text
    assert "REAL_ORDER_SUBMISSION_ALLOWED=false" in text


def _write_collector_artifact(
    tmp_path: Path, *, first_historical_at: str = "2026-07-06T09:04:40+00:00"
) -> Path:
    root = tmp_path / "collector"
    root.mkdir()
    _write_json(
        root / "manifest.json",
        {
            "ARTIFACT_SHA": EXPECTED_COLLECTOR_ARTIFACT_SHA,
            "SOURCE_REGISTRY_SHA": EXPECTED_COLLECTOR_SOURCE_REGISTRY_SHA,
            "NETWORK_PROVENANCE_SHA": EXPECTED_COLLECTOR_NETWORK_PROVENANCE_SHA,
            "RAW_SNAPSHOT_SHA": EXPECTED_COLLECTOR_RAW_SNAPSHOT_SHA,
            "COLLECTED_EVENT_METADATA_SHA": EXPECTED_COLLECTOR_EVENT_METADATA_SHA,
            "DEDUPE_STATE_SHA": EXPECTED_COLLECTOR_DEDUPE_STATE_SHA,
            "ITEMS_FETCHED": 50,
            "NEW_HISTORICAL_EXACT_EVENTS": 44,
            "NEW_FUTURE_METADATA_ONLY_EVENTS": 6,
            "FUTURE_EVENT_HOLDOUT_USED": False,
            "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
            "TEST_OUTCOME_USED": False,
            "MODEL_TRAINING_PERFORMED": False,
            "TEST_EVALUATION_PERFORMED": False,
        },
    )
    rows = _collector_rows(first_historical_at=first_historical_at)
    _write_jsonl(root / "collected-event-metadata.jsonl", rows)
    return root


def _collector_rows(
    *, first_historical_at: str = "2026-07-06T09:04:40+00:00"
) -> list[dict[str, object]]:
    historical_times = [first_historical_at] + [
        (datetime(2026, 7, 7, 9, 0, tzinfo=UTC) + timedelta(hours=index)).isoformat()
        for index in range(43)
    ]
    future_times = [
        (datetime(2026, 8, 11, 9, 0, tzinfo=UTC) + timedelta(days=index)).isoformat()
        for index in range(6)
    ]
    rows: list[dict[str, object]] = []
    for index, published_at in enumerate([*historical_times, *future_times]):
        published = datetime.fromisoformat(published_at)
        future = published.date().isoformat() >= "2026-08-11"
        event_id = f"chep-{index:02d}"
        rows.append(
            {
                "metadata": {
                    "event_id": event_id,
                    "source_code": "CHEP_OFFICIAL_RSS_EXACT_LIVE_V1",
                    "source_item_id": f"https://example.test/{event_id}",
                    "canonical_url": f"https://example.test/{event_id}",
                    "ticker": "CHEP",
                    "issuer": "ЧТПЗ",
                    "instrument_uid": "b1f4f4fc-dac5-4e29-ae56-95fe441416ee",
                    "publication_timestamp_utc": published.isoformat(),
                    "publication_timestamp_raw": published.isoformat(),
                    "publication_date": published.date().isoformat(),
                    "publication_time": published.time().isoformat(),
                    "publication_timezone": "UTC",
                    "timestamp_quality": "EXACT",
                    "timestamp_source_field": "RSS item pubDate",
                    "future_holdout": future,
                    "future_holdout_metadata_only": future,
                    "title_hash": event_id,
                },
                "event_features": None,
                "pre_event_market_features": None,
                "target_availability": {
                    "reaction_ready": False,
                    "feature_ready": False,
                    "research_outcomes_visible": False,
                    "status": "METADATA_ONLY",
                },
                "quality": {"metadata_only": True},
            }
        )
    return rows


def _write_base_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "base"
    root.mkdir()
    event = {
        "metadata": {
            "event_id": "base-event",
            "ticker": "BASE",
            "issuer": "Base Issuer",
            "publication_timestamp_utc": "2026-07-01T09:00:00+00:00",
            "future_holdout": False,
        },
        "event_features": {"primary_event_type": "UNKNOWN", "event_count": 0, "fact_count": 0},
        "pre_event_market_features": {},
        "target_availability": {
            "reaction_ready": True,
            "feature_ready": True,
            "research_outcomes_visible": True,
            "status": "REACTION_READY",
        },
        "quality": {},
    }
    _write_json(root / "manifest.json", {"OUTPUT_DATASET_SHA": "base-sha"})
    _write_jsonl(root / "events.jsonl", [event])
    _write_jsonl(
        root / "features.jsonl",
        [
            {
                "event_id": "base-event",
                "feature_cutoff": "2026-07-01T09:00:00+00:00",
                "event_features": event["event_features"],
                "market_features": {},
            }
        ],
    )
    _write_jsonl(
        root / "targets.jsonl",
        [{"event_id": "base-event", "reaction_family": "EXACT_INTRADAY", "horizons": {}}],
    )
    return root


def _write_universe(tmp_path: Path) -> Path:
    root = tmp_path / "universe"
    root.mkdir()
    _write_jsonl(
        root / "history-coverage.jsonl",
        [
            {
                "ticker": "CHEP",
                "issuer": "ЧТПЗ",
                "instrument_uid": "b1f4f4fc-dac5-4e29-ae56-95fe441416ee",
                "figi": "BBG000Q49F45",
                "class_code": "TQBR",
                "exchange": "unknown",
                "currency": "rub",
                "historical_candle_available": True,
                "first_1day_candle_date": "2009-01-30",
                "last_1day_candle_date": "2021-09-21",
            }
        ],
    )
    return root


def _write_candle_cache(
    root: Path,
    ticker: str,
    instrument_uid: str,
    begin_times: Sequence[str] | None = None,
) -> None:
    times = begin_times or (
        "2026-07-06T08:00:00+00:00",
        "2026-07-06T08:04:00+00:00",
        "2026-07-06T08:34:00+00:00",
        "2026-07-06T08:49:00+00:00",
        "2026-07-06T08:59:00+00:00",
        "2026-07-06T09:04:00+00:00",
        "2026-07-06T09:05:00+00:00",
        "2026-07-06T09:09:00+00:00",
        "2026-07-06T09:19:00+00:00",
        "2026-07-06T09:34:00+00:00",
        "2026-07-06T10:04:00+00:00",
    )
    rows: list[dict[str, object]] = []
    for index, text in enumerate(times, start=1):
        begin = datetime.fromisoformat(text)
        rows.append(
            {
                "ticker": ticker,
                "figi": "BBG000Q49F45" if ticker == "CHEP" else None,
                "class_code": "TQBR" if ticker == "CHEP" else None,
                "instrument_uid": instrument_uid,
                "begin_at": begin.isoformat(),
                "end_at": (begin + timedelta(minutes=1)).isoformat(),
                "open": str(100 + index),
                "high": str(100 + index),
                "low": str(100 + index),
                "close": str(100 + index),
                "volume": index,
                "is_complete": True,
                "source": "TINVEST_API",
            }
        )
    _write_jsonl(root / ticker / "2026-07-06-day.jsonl", rows)


def _metadata(row: dict[str, object]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["metadata"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
