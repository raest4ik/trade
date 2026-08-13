from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

import httpx
import pytest

from src.tinvest_market.client import TInvestContour, TInvestInstrument, TInvestReadOnlyClient
from src.tinvest_market.domain import DailyBar, build_dataset
from src.tinvest_market_universe.application import ExpansionResult, corpus_lock, expand_universe
from src.tinvest_market_universe.domain import (
    MEMBERSHIP_MODE,
    ORIGINAL_TICKERS,
    SURVIVORSHIP_RISK,
    discover,
    enhance_features,
    feature_schema_sha,
    partition_for,
)


async def test_shares_discovery_uses_instrument_status_all_and_parses_metadata() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"instruments": [_instrument_payload("SBER")]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TInvestReadOnlyClient(
            token="secret", contour=TInvestContour.READONLY_PRODUCTION, client=http_client
        )
        shares = await client.list_shares()
    body = json.loads(requests[0].content)
    assert body == {"instrumentStatus": "INSTRUMENT_STATUS_ALL"}
    assert requests[0].url.path.endswith("InstrumentsService/Shares")
    assert shares[0].real_exchange == "REAL_EXCHANGE_MOEX"
    assert shares[0].api_trade_available is True
    assert shares[0].instrument_type == "INSTRUMENT_TYPE_SHARE"


async def test_universe_audited_candles_reject_invalid_rows_without_fabrication() -> None:
    invalid = _candle_payload(date(2020, 1, 1))
    invalid["close"] = {"units": 0, "nano": 0}

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candles": [invalid, _candle_payload(date(2020, 1, 2))]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TInvestReadOnlyClient(
            token="secret", contour=TInvestContour.READONLY_PRODUCTION, client=http_client
        )
        batch = await client.fetch_daily_candles_audited(
            instrument_uid="uid-SBER",
            date_from=date(2020, 1, 1),
            date_to=date(2020, 1, 2),
        )
    assert len(batch.candles) == 1
    assert batch.rejected_reasons == ("TINVEST_CANDLE_INVALID_PRICE",)


def test_structural_filter_is_deterministic_and_excludes_dealer_and_non_rub() -> None:
    base = _instrument("SBER")
    values = (
        base,
        replace(base, ticker="USD", instrument_uid="uid-USD", currency="usd"),
        replace(base, ticker="DEAL", instrument_uid="uid-DEAL", class_code="SPBRU"),
        replace(
            base,
            ticker="BOND",
            instrument_uid="uid-BOND",
            instrument_type="INSTRUMENT_TYPE_BOND",
        ),
    )
    forward = discover(values)
    reverse = discover(tuple(reversed(values)))
    assert [item.instrument_uid for item in forward.eligible] == ["uid-SBER"]
    assert forward.diagnostics == reverse.diagnostics
    assert forward.diagnostics["dealer_market_included"] is False
    assert forward.diagnostics["non_share_assets_included"] is False
    assert forward.diagnostics["universe_membership_mode"] == MEMBERSHIP_MODE
    assert forward.diagnostics["survivorship_bias_risk"] == SURVIVORSHIP_RISK
    assert forward.diagnostics["candidate_availability_distribution"] == {"ACTIVE_API_AVAILABLE": 1}


def test_features_are_t_minus_one_and_partitions_are_fail_closed() -> None:
    securities, benchmark = _history()
    base = build_dataset(securities, benchmark)
    rows, names = enhance_features(
        base.features, {ticker: f"uid-{ticker}" for ticker in securities}
    )
    assert len(names) == 43
    assert (
        feature_schema_sha(names)
        == "f7a60ecf55d7d0f7d455035810312224a30ee637a3bab2dfede231ca9dc0bb45"
    )
    assert all(item.feature_as_of < item.trade_date for item in rows)
    assert partition_for(date(2022, 9, 15)) == "DEVELOPMENT"
    assert partition_for(date(2022, 9, 16)) == "PURGE_EMBARGO_GAP"
    assert partition_for(date(2022, 9, 20)) == "OBSERVED_V1_TEST"
    assert partition_for(date(2026, 8, 12)) == "FUTURE_BLIND_HOLDOUT"


async def test_expansion_is_checkpointed_idempotent_preserves_originals_and_has_no_secrets(
    tmp_path: Path,
) -> None:
    candle_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal candle_calls
        body = json.loads(request.content)
        if request.url.path.endswith("InstrumentsService/Shares"):
            return httpx.Response(
                200,
                json={"instruments": [_instrument_payload(ticker) for ticker in ORIGINAL_TICKERS]},
            )
        if request.url.path.endswith("InstrumentsService/Indicatives"):
            return httpx.Response(
                200,
                json={
                    "instruments": [
                        _instrument_payload(
                            "IMOEX", kind="INSTRUMENT_TYPE_INDEX", class_code="INDX"
                        )
                    ]
                },
            )
        if request.url.path.endswith("InstrumentsService/GetInstrumentBy"):
            ticker = str(body["id"]).removeprefix("uid-")
            return httpx.Response(
                200,
                json={
                    "instrument": _instrument_payload(
                        ticker,
                        kind="INSTRUMENT_TYPE_INDEX"
                        if ticker == "IMOEX"
                        else "INSTRUMENT_TYPE_SHARE",
                        class_code="INDX" if ticker == "IMOEX" else "TQBR",
                    )
                },
            )
        if request.url.path.endswith("MarketDataService/GetCandles"):
            candle_calls += 1
            return httpx.Response(
                200,
                json={
                    "candles": [
                        _candle_payload(date(2020, 1, 1) + timedelta(days=i)) for i in range(30)
                    ]
                },
            )
        raise AssertionError(request.url.path)

    baseline = tmp_path / "baseline"
    _write_baseline(baseline)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TInvestReadOnlyClient(
            token="top-secret", contour=TInvestContour.READONLY_PRODUCTION, client=http_client
        )
        first = await _test_expansion(client, tmp_path, baseline)
        first_calls = candle_calls
        second = await _test_expansion(client, tmp_path, baseline)
    assert candle_calls == first_calls
    assert first.raw_manifest["dataset_sha"] == second.raw_manifest["dataset_sha"]
    preservation = cast("dict[str, object]", first.raw_manifest["original_ticker_preservation"])
    assert preservation["all_original_10_preserved"] is True
    assert first.feature_manifest["target_values_persisted"] is False
    assert first.feature_manifest["observed_test_used"] is False
    assert first.feature_manifest["future_holdout_evaluated"] is False
    source = await _read_tree(tmp_path)
    assert "top-secret" not in source


def test_concurrent_collection_lock_fails_closed(tmp_path: Path) -> None:
    lock = tmp_path / "corpus.lock"
    with corpus_lock(lock), pytest.raises(RuntimeError, match="ALREADY_RUNNING"):
        with corpus_lock(lock):
            pass
    assert not lock.exists()


def test_no_order_model_source_mixing_or_future_evaluation_surface() -> None:
    source = "".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/tinvest_market_universe").rglob("*.py")
    ) + Path("apps/cli/tinvest_market_universe_build.py").read_text(encoding="utf-8")
    for forbidden in (
        "OrdersService",
        "MOEX ISS",
        "train_predictive",
        "predict_proba",
        "submit_order",
        "BUY_SIGNAL",
        "SELL_SIGNAL",
    ):
        assert forbidden not in source


def test_original_ten_are_declared_without_ticker_selection_heuristics() -> None:
    assert ORIGINAL_TICKERS == (
        "SBER",
        "SBERP",
        "GAZP",
        "LKOH",
        "ROSN",
        "NVTK",
        "YDEX",
        "T",
        "VTBR",
        "GMKN",
    )
    assert "ticker_name_heuristics_used" in Path("src/tinvest_market_universe/domain.py").read_text(
        encoding="utf-8"
    )


def _instrument(ticker: str) -> TInvestInstrument:
    return TInvestInstrument(
        ticker=ticker,
        class_code="TQBR",
        instrument_uid=f"uid-{ticker}",
        figi=f"figi-{ticker}",
        instrument_type="INSTRUMENT_TYPE_SHARE",
        first_1day_candle_date=date(2020, 1, 1),
        name=ticker,
        exchange="MOEX",
        currency="rub",
        real_exchange="REAL_EXCHANGE_MOEX",
        trading_status="SECURITY_TRADING_STATUS_NORMAL_TRADING",
        api_trade_available=True,
        buy_available=True,
        sell_available=True,
    )


async def _test_expansion(
    client: TInvestReadOnlyClient, tmp_path: Path, baseline: Path
) -> ExpansionResult:
    return await expand_universe(
        client,
        raw_dir=tmp_path / "raw",
        feature_dir=tmp_path / "features",
        baseline_raw_dir=baseline,
        date_from=date(2020, 1, 1),
        date_to=date(2020, 2, 1),
        git_sha="abc",
        discovered_at=datetime(2026, 8, 13, tzinfo=UTC),
    )


def _instrument_payload(
    ticker: str, *, kind: str = "INSTRUMENT_TYPE_SHARE", class_code: str = "TQBR"
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "classCode": class_code,
        "uid": f"uid-{ticker}",
        "figi": f"figi-{ticker}",
        "instrumentType": kind,
        "first1dayCandleDate": "2020-01-01T00:00:00Z",
        "name": ticker,
        "exchange": "MOEX",
        "currency": "rub",
        "realExchange": "REAL_EXCHANGE_MOEX",
        "tradingStatus": "SECURITY_TRADING_STATUS_NORMAL_TRADING",
        "apiTradeAvailableFlag": True,
        "buyAvailableFlag": True,
        "sellAvailableFlag": True,
    }


def _candle_payload(day: date) -> dict[str, object]:
    value = 100 + (day - date(2020, 1, 1)).days
    return {
        "open": {"units": value, "nano": 0},
        "high": {"units": value + 1, "nano": 0},
        "low": {"units": value - 1, "nano": 0},
        "close": {"units": value, "nano": 0},
        "volume": "1000",
        "time": f"{day.isoformat()}T00:00:00Z",
        "isComplete": True,
    }


def _history() -> tuple[dict[str, tuple[DailyBar, ...]], tuple[DailyBar, ...]]:
    start = date(2022, 8, 1)
    benchmark = tuple(_bar("IMOEX", start + timedelta(days=i), 1000 + i) for i in range(60))
    securities = {
        ticker: tuple(_bar(ticker, start + timedelta(days=i), 100 + i) for i in range(60))
        for ticker in ("SBER", "GAZP")
    }
    return securities, benchmark


def _bar(ticker: str, day: date, close: float) -> DailyBar:
    return DailyBar(ticker, f"uid-{ticker}", day, close, close + 1, close - 1, close, 1000, True)


def _write_baseline(root: Path) -> None:
    (root / "series").mkdir(parents=True)
    for ticker in ORIGINAL_TICKERS:
        rows = [
            _bar(ticker, date(2020, 1, 1) + timedelta(days=i), 100 + i).payload() for i in range(30)
        ]
        (root / "series" / f"{ticker}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )


async def _read_tree(root: Path) -> str:
    import asyncio

    return await asyncio.to_thread(
        lambda: "".join(path.read_text(encoding="utf-8") for path in root.rglob("*.*"))
    )
