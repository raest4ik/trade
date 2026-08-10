from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

import httpx
import pytest

from src.market_data.application.exceptions import (
    MarketDataPartialProviderError,
    MarketDataProviderContractError,
    MarketDataValidationError,
)
from src.market_data.domain.entities import MarketBenchmark
from src.market_data.infrastructure.moex_client import MoexIssClient, retry_delay_seconds


def moex_payload(
    rows: Sequence[Sequence[object]],
    columns: list[str] | None = None,
) -> dict[str, object]:
    return {
        "candles": {
            "columns": columns
            or ["begin", "end", "open", "high", "low", "close", "volume", "value"],
            "data": rows,
        }
    }


async def test_moex_client_parses_columns_by_name_and_converts_moscow_to_utc() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["interval"] == "1"
        return httpx.Response(
            200,
            json=moex_payload(
                [
                    [
                        "2026-07-01 10:00:00",
                        "2026-07-01 10:00:59",
                        "100.10",
                        "101.20",
                        "99.90",
                        "100.50",
                        "1000",
                        "100500.25",
                    ]
                ]
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = MoexIssClient(
            base_url="https://iss.moex.com/iss",
            timeout_seconds=1,
            max_retries=0,
            max_pages=10,
            user_agent="tests",
            client=http_client,
        )
        result = await client.fetch_candles_with_rejections(
            instrument_id=uuid4(),
            ticker="SBER",
            board="TQBR",
            date_from=date(2026, 7, 1),
            date_till=date(2026, 7, 1),
            interval_minutes=1,
        )

    assert result.pages_received == 1
    assert result.rows_received == 1
    assert result.rows_rejected == 0
    candle = result.candles[0]
    assert candle.begin_at == datetime(2026, 7, 1, 7, 0, tzinfo=UTC)
    assert candle.end_at == datetime(2026, 7, 1, 7, 0, 59, tzinfo=UTC)
    assert candle.open == Decimal("100.10")
    assert candle.value == Decimal("100500.25")


async def test_moex_client_counts_rejected_bad_ohlc_rows() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=moex_payload(
                [
                    [
                        "2026-07-01 10:00:00",
                        "2026-07-01 10:00:59",
                        "105",
                        "101",
                        "99",
                        "100",
                        "1",
                        "1",
                    ]
                ]
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = MoexIssClient(
            base_url="https://iss.moex.com/iss",
            timeout_seconds=1,
            max_retries=0,
            max_pages=10,
            user_agent="tests",
            client=http_client,
        )
        result = await client.fetch_candles_with_rejections(
            instrument_id=uuid4(),
            ticker="SBER",
            board="TQBR",
            date_from=date(2026, 7, 1),
            date_till=date(2026, 7, 1),
            interval_minutes=1,
        )

    assert result.candles == []
    assert result.rows_rejected == 1
    assert result.rejected_rows[0].reason == "open and close must be inside low-high range"


async def test_moex_client_retries_http_500_and_respects_retry_after() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json=moex_payload([]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = MoexIssClient(
            base_url="https://iss.moex.com/iss",
            timeout_seconds=1,
            max_retries=1,
            max_pages=10,
            user_agent="tests",
            client=http_client,
            sleep=False,
        )
        result = await client.fetch_candles_with_rejections(
            instrument_id=uuid4(),
            ticker="SBER",
            board="TQBR",
            date_from=date(2026, 7, 1),
            date_till=date(2026, 7, 1),
            interval_minutes=1,
        )

    assert calls == 2
    assert result.rows_received == 0


async def test_moex_client_raises_partial_error_after_later_page_failure() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            rows = [
                [
                    "2026-07-01 10:00:00",
                    "2026-07-01 10:00:59",
                    "100",
                    "100",
                    "100",
                    "100",
                    "1",
                    "100",
                ]
                for _ in range(500)
            ]
            return httpx.Response(200, json=moex_payload(rows))
        return httpx.Response(500, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = MoexIssClient(
            base_url="https://iss.moex.com/iss",
            timeout_seconds=1,
            max_retries=0,
            max_pages=10,
            user_agent="tests",
            client=http_client,
            sleep=False,
        )
        with pytest.raises(MarketDataPartialProviderError) as exc_info:
            await client.fetch_candles_with_rejections(
                instrument_id=uuid4(),
                ticker="SBER",
                board="TQBR",
                date_from=date(2026, 7, 1),
                date_till=date(2026, 7, 1),
                interval_minutes=1,
            )

    assert calls == 2
    assert len(exc_info.value.candles) == 500
    assert exc_info.value.pages_received == 1
    assert exc_info.value.rows_valid == 500


async def test_moex_client_rejects_bad_path_values_and_large_ranges() -> None:
    client = MoexIssClient(
        base_url="https://iss.moex.com/iss",
        timeout_seconds=1,
        max_retries=0,
        max_pages=10,
        user_agent="tests",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200))
        ),
    )
    with pytest.raises(MarketDataValidationError):
        await client.fetch_candles_with_rejections(
            instrument_id=uuid4(),
            ticker="../SBER",
            board="TQBR",
            date_from=date(2026, 7, 1),
            date_till=date(2026, 7, 1),
            interval_minutes=1,
        )
    with pytest.raises(MarketDataValidationError):
        await client.fetch_candles_with_rejections(
            instrument_id=uuid4(),
            ticker="SBER",
            board="TQBR",
            date_from=date(2026, 1, 1),
            date_till=date(2026, 3, 1),
            interval_minutes=1,
        )
    await client.aclose()


async def test_moex_client_rejects_missing_required_columns() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=moex_payload([], columns=["begin", "end"]))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = MoexIssClient(
            base_url="https://iss.moex.com/iss",
            timeout_seconds=1,
            max_retries=0,
            max_pages=10,
            user_agent="tests",
            client=http_client,
        )
        with pytest.raises(MarketDataProviderContractError):
            await client.fetch_candles_with_rejections(
                instrument_id=uuid4(),
                ticker="SBER",
                board="TQBR",
                date_from=date(2026, 7, 1),
                date_till=date(2026, 7, 1),
                interval_minutes=1,
            )


async def test_moex_client_fetches_imoex_with_index_endpoint_utc_and_decimal() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert "/markets/index/boards/SNDX/securities/IMOEX/candles.json" in str(request.url)
        return httpx.Response(
            200,
            json=moex_payload(
                [
                    [
                        "2026-07-01 10:00:00",
                        "2026-07-01 10:00:59",
                        "2800.10",
                        "2801.20",
                        "2799.90",
                        "2800.50",
                        "1000",
                        "100500.25",
                    ]
                ]
            ),
        )

    benchmark = MarketBenchmark.create(code="IMOEX", name="MOEX Russia Index", board="SNDX")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = MoexIssClient(
            base_url="https://iss.moex.com/iss",
            timeout_seconds=1,
            max_retries=0,
            max_pages=10,
            user_agent="tests",
            client=http_client,
        )
        result = await client.fetch_benchmark_candles_with_rejections(
            benchmark=benchmark,
            date_from=date(2026, 7, 1),
            date_till=date(2026, 7, 1),
            interval_minutes=1,
        )

    candle = result.candles[0]
    assert candle.begin_at == datetime(2026, 7, 1, 7, 0, tzinfo=UTC)
    assert candle.close == Decimal("2800.50")


async def test_moex_benchmark_pagination_uses_start_offset() -> None:
    starts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        starts.append(request.url.params["start"])
        if len(starts) == 1:
            rows = [
                [
                    "2026-07-01 10:00:00",
                    "2026-07-01 10:00:59",
                    "100",
                    "100",
                    "100",
                    "100",
                    "1",
                    "100",
                ]
                for _ in range(500)
            ]
            return httpx.Response(200, json=moex_payload(rows))
        return httpx.Response(200, json=moex_payload([]))

    benchmark = MarketBenchmark.create(code="IMOEX", name="MOEX Russia Index", board="SNDX")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = MoexIssClient(
            base_url="https://iss.moex.com/iss",
            timeout_seconds=1,
            max_retries=0,
            max_pages=10,
            user_agent="tests",
            client=http_client,
        )
        result = await client.fetch_benchmark_candles_with_rejections(
            benchmark=benchmark,
            date_from=date(2026, 7, 1),
            date_till=date(2026, 7, 1),
            interval_minutes=1,
        )
    assert starts == ["0", "500"]
    assert result.pages_received == 2
    assert result.rows_valid == 500


async def test_moex_benchmark_retries_temporary_failure() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json=moex_payload([]))

    benchmark = MarketBenchmark.create(code="IMOEX", name="MOEX Russia Index", board="SNDX")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = MoexIssClient(
            base_url="https://iss.moex.com/iss",
            timeout_seconds=1,
            max_retries=1,
            max_pages=10,
            user_agent="tests",
            client=http_client,
            sleep=False,
        )
        result = await client.fetch_benchmark_candles_with_rejections(
            benchmark=benchmark,
            date_from=date(2026, 7, 1),
            date_till=date(2026, 7, 1),
            interval_minutes=1,
        )
    assert calls == 2
    assert result.rows_received == 0


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [("2.5", 2.5), ("60", 30.0), ("invalid", 1.0), (None, 0.25)],
)
def test_retry_after_policy_is_bounded(retry_after: str | None, expected: float) -> None:
    assert retry_delay_seconds(0, retry_after) == expected
