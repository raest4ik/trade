from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_corpus.application import build_exact_source_registry
from src.exact_event_corpus.domain import (
    FUTURE_EVENT_HOLDOUT_START,
    ExactEvent,
    SessionState,
    TimestampCapability,
    deterministic_clusters,
)
from src.exact_event_corpus.holdout import FutureEventHoldoutReadError, guard_outcome_read
from src.exact_event_corpus.market import align_exact_event
from src.exact_event_corpus.sources import (
    ExactAppStateProfile,
    acquire_exact_json_pages,
    parse_exact_app_state,
)
from src.tinvest_market.client import TInvestMinuteCandle


def test_exact_requires_real_aware_source_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _event(published_at=datetime(2026, 8, 10, 10))
    with pytest.raises(ValueError, match="raw timestamp"):
        _event(raw="")


def test_date_only_registry_cannot_become_exact(tmp_path: Path) -> None:
    mapping, previous = _registry_files(tmp_path, 315)
    registry = build_exact_source_registry(mapping, previous)
    assert len(registry) == 315
    unknown = next(item for item in registry if item.ticker == "T000")
    assert unknown.timestamp_capability == TimestampCapability.DATE_ONLY
    assert unknown.timestamp_field_source is None


def test_unix_timestamp_timezone_conversion_is_deterministic() -> None:
    profile = ExactAppStateProfile(
        source_code="OFFICIAL_APP_EXACT",
        ticker="GMKN",
        issuer="Issuer",
        instrument_uid="uid",
        base_url="https://issuer.example/",
        timestamp_field="activeFrom",
        title_field="name",
        url_field="detailPageUrl",
    )
    payload = (
        'App = {"items":[{"name":"Release","detailPageUrl":"/release/","activeFrom":1785507000}]};'
    )
    first = parse_exact_app_state(payload, profile=profile)[0]
    second = parse_exact_app_state(payload, profile=profile)[0]
    assert first.publication_timestamp_utc == datetime(2026, 7, 31, 14, 10, tzinfo=UTC)
    assert first == second


def test_source_local_midnight_placeholder_is_not_accepted_as_exact() -> None:
    profile = ExactAppStateProfile(
        source_code="OFFICIAL_APP_EXACT",
        ticker="GMKN",
        issuer="Issuer",
        instrument_uid="uid",
        base_url="https://issuer.example/",
        timestamp_field="activeFrom",
        title_field="name",
        url_field="detailPageUrl",
        reject_source_local_midnight=True,
    )
    payload = (
        'App = {"items":[{"name":"Date placeholder","detailPageUrl":"/release/",'
        '"activeFrom":1782766800}]};'
    )
    assert parse_exact_app_state(payload, profile=profile) == []


@pytest.mark.asyncio
async def test_official_json_acquisition_is_bounded_and_uses_no_auth(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        page = int(request.url.params["page"])
        return httpx.Response(
            200,
            request=request,
            json={
                "items": [
                    {
                        "date": 1786358400 - page * 60,
                        "name": f"Release {page}",
                        "link": f"/ru/media/press-releases/{page}/",
                    }
                ],
                "nav": {"current": page, "total": 99},
            },
        )

    profile = ExactAppStateProfile(
        source_code="MAGNIT_OFFICIAL_JSON_EXACT",
        ticker="MGNT",
        issuer="Magnit",
        instrument_uid="uid",
        base_url="https://www.magnit.com/",
        timestamp_field="date",
        title_field="name",
        url_field="link",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = await acquire_exact_json_pages(
            profile=profile,
            url_template="https://www.magnit.com/ru/api/news?page={page}",
            date_from=datetime(2026, 8, 1, tzinfo=UTC).date(),
            date_to=datetime(2026, 8, 14, tzinfo=UTC).date(),
            page_limit=2,
            item_limit=2,
            cache_dir=tmp_path,
            client=client,
        )
    assert len(events) == 2
    assert len(calls) == 2
    assert all(request.url.host == "www.magnit.com" for request in calls)
    assert all("authorization" not in request.headers for request in calls)


def test_inside_minute_starts_at_next_full_candle_without_leakage() -> None:
    published = datetime(2026, 8, 10, 10, 0, 17, tzinfo=UTC)
    security = _candles(datetime(2026, 8, 10, 9, 0, tzinfo=UTC), 123, "security")
    benchmark = _candles(datetime(2026, 8, 10, 9, 0, tzinfo=UTC), 123, "benchmark")
    result = align_exact_event(published, security, benchmark, expose_outcomes=True)
    assert result.effective_event_at == datetime(2026, 8, 10, 10, 1, tzinfo=UTC)
    assert result.baseline_observed_at == datetime(2026, 8, 10, 10, 0, tzinfo=UTC)
    assert result.reaction_status == "REACTION_READY"
    assert all(
        datetime.fromisoformat(str(row["window_begin_at"])) >= published
        for row in result.horizons.values()
    )


def test_security_and_imoex_use_identical_actual_windows() -> None:
    published = datetime(2026, 8, 10, 10, tzinfo=UTC)
    security = _candles(datetime(2026, 8, 10, 9, tzinfo=UTC), 123, "security")
    benchmark = _candles(datetime(2026, 8, 10, 9, tzinfo=UTC), 123, "benchmark")
    result = align_exact_event(published, security, benchmark, expose_outcomes=True)
    assert all(
        row["security_observed_at"] == row["benchmark_observed_at"]
        for row in result.horizons.values()
    )


@pytest.mark.parametrize(
    ("published", "expected"),
    [
        (datetime(2026, 8, 10, 8, 59, tzinfo=UTC), SessionState.PRE_OPEN),
        (datetime(2026, 8, 10, 11, 1, tzinfo=UTC), SessionState.AFTER_CLOSE),
    ],
)
def test_unsupported_preopen_and_afterclose_fail_closed(
    published: datetime, expected: SessionState
) -> None:
    rows = _candles(datetime(2026, 8, 10, 9, tzinfo=UTC), 121, "uid")
    result = align_exact_event(published, rows, rows, expose_outcomes=True)
    assert result.session_state == expected
    assert result.horizons == {}
    assert result.reaction_status.endswith("NOT_SUPPORTED")


def test_non_trading_day_fails_closed_without_synthetic_candles() -> None:
    result = align_exact_event(datetime(2026, 8, 9, 10, tzinfo=UTC), (), (), expose_outcomes=True)
    assert result.session_state == SessionState.NON_TRADING_DAY
    assert result.horizons == {}
    assert result.features == {}


def test_missing_candle_is_not_forward_filled_or_interpolated() -> None:
    published = datetime(2026, 8, 10, 10, tzinfo=UTC)
    security = _candles(datetime(2026, 8, 10, 9, tzinfo=UTC), 123, "security")
    benchmark = list(_candles(datetime(2026, 8, 10, 9, tzinfo=UTC), 123, "benchmark"))
    benchmark = [row for row in benchmark if row.end_at != published + timedelta(minutes=5)]
    result = align_exact_event(published, security, benchmark, expose_outcomes=True)
    assert result.horizons["5m"] == {
        "available": False,
        "reason": "EXACT_TARGET_CANDLE_MISSING",
    }


def test_event_cluster_is_deterministic_and_keeps_distinct_events() -> None:
    first = _event(source_item_id="one", url="https://issuer.example/one")
    second = _event(source_item_id="two", url="https://issuer.example/two", title="Other")
    assert deterministic_clusters([first, second]) == deterministic_clusters([second, first])
    assert len(deterministic_clusters([first, second])) == 2


def test_future_holdout_hard_guard() -> None:
    guard_outcome_read(FUTURE_EVENT_HOLDOUT_START - timedelta(days=1))
    with pytest.raises(FutureEventHoldoutReadError, match="FUTURE_EVENT_HOLDOUT_READ_ATTEMPT"):
        guard_outcome_read(FUTURE_EVENT_HOLDOUT_START)


def test_future_holdout_alignment_exports_no_outcomes() -> None:
    published = datetime(2026, 8, 11, 10, tzinfo=UTC)
    rows = _candles(datetime(2026, 8, 11, 9, tzinfo=UTC), 123, "uid")
    result = align_exact_event(published, rows, rows, expose_outcomes=False)
    assert result.horizons == {}
    assert result.reaction_status == "FUTURE_HOLDOUT_OUTCOMES_GUARDED"


def test_nlp_contract_is_frozen() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert prompt_hash() == QWEN_PROMPT_SHA
    assert schema_hash() == QWEN_SCHEMA_SHA


def test_exact_package_has_no_model_evaluation_or_trading_capability() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in Path("src/exact_event_corpus").glob("*.py")
    ).lower()
    forbidden = (
        "logisticregression",
        "histgradientboosting",
        "ridgeclassifier",
        "fit(",
        "predict(",
        "place_order",
        "postorder",
        "buy_order",
        "sell_order",
    )
    assert not any(value in text for value in forbidden)
    assert '"backtest_executed": false' in text
    assert '"paper_trading_executed": false' in text


def _event(
    *,
    published_at: datetime = datetime(2026, 8, 10, 10, tzinfo=UTC),
    raw: str = "1786356000",
    source_item_id: str = "one",
    url: str = "https://issuer.example/one",
    title: str = "Release",
) -> ExactEvent:
    return ExactEvent.create(
        source_code="OFFICIAL_EXACT",
        source_item_id=source_item_id,
        canonical_url=url,
        ticker="TEST",
        issuer="Issuer",
        instrument_uid="uid",
        title=title,
        publication_timestamp_raw=raw,
        publication_timestamp_utc=published_at,
        timestamp_source_field="source.timestamp",
    )


def _candles(begin: datetime, count: int, uid: str) -> tuple[TInvestMinuteCandle, ...]:
    return tuple(
        TInvestMinuteCandle(
            instrument_uid=uid,
            begin_at=begin + timedelta(minutes=index),
            end_at=begin + timedelta(minutes=index + 1),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100") + Decimal(index) / Decimal("100"),
            volume=100,
            is_complete=True,
        )
        for index in range(count)
    )


def _registry_files(tmp_path: Path, count: int) -> tuple[Path, Path]:
    mapping = tmp_path / "mapping.json"
    previous = tmp_path / "previous.jsonl"
    instruments = [
        {"ticker": f"T{index:03d}", "name": f"Issuer {index}", "instrument_uid": f"uid-{index}"}
        for index in range(count)
    ]
    mapping.write_text(json.dumps({"instruments": instruments}), encoding="utf-8")
    previous.write_text(
        "".join(
            json.dumps(
                {
                    "ticker": item["ticker"],
                    "date_only_available": True,
                    "collector_status": "SOURCE_READY",
                    "reason": "Official source only exposes a date",
                }
            )
            + "\n"
            for item in instruments
        ),
        encoding="utf-8",
    )
    return mapping, previous
