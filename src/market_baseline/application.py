from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.instruments.infrastructure.seed import SEED_INSTRUMENTS
from src.market_baseline.domain import DailyBar
from src.market_data.infrastructure.moex_client import MoexDailyFetchResult, MoexIssClient

BENCHMARK_CODE = "IMOEX"
BENCHMARK_BOARD = "SNDX"
DEFAULT_TICKERS = tuple(item.ticker for item in SEED_INSTRUMENTS)


@dataclass(frozen=True, slots=True)
class AcquiredMarketHistory:
    security_bars: dict[str, tuple[DailyBar, ...]]
    benchmark_bars: tuple[DailyBar, ...]
    acquisition: dict[str, Any]


async def acquire_market_history(
    client: MoexIssClient,
    *,
    tickers: tuple[str, ...],
    date_from: date,
    date_till: date,
    max_concurrency: int = 3,
) -> AcquiredMarketHistory:
    seeded_boards = {item.ticker: item.primary_board for item in SEED_INSTRUMENTS}
    normalized = tuple(sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()}))
    unknown = sorted(set(normalized) - set(seeded_boards))
    if unknown:
        raise ValueError("tickers are not in the seeded universe: " + ", ".join(unknown))
    if not normalized:
        raise ValueError("at least one ticker is required")
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def fetch_security(ticker: str) -> tuple[str, MoexDailyFetchResult]:
        async with semaphore:
            result = await client.fetch_daily_candles(
                security_code=ticker,
                engine="stock",
                market="shares",
                board=seeded_boards[ticker],
                date_from=date_from,
                date_till=date_till,
            )
        return ticker, result

    async def fetch_benchmark() -> MoexDailyFetchResult:
        async with semaphore:
            return await client.fetch_daily_candles(
                security_code=BENCHMARK_CODE,
                engine="stock",
                market="index",
                board=BENCHMARK_BOARD,
                date_from=date_from,
                date_till=date_till,
            )

    security_results, benchmark_result = await asyncio.gather(
        asyncio.gather(*(fetch_security(ticker) for ticker in normalized)),
        fetch_benchmark(),
    )
    bars: dict[str, tuple[DailyBar, ...]] = {}
    series_stats: dict[str, dict[str, Any]] = {}
    for ticker, result in security_results:
        mapped = tuple(_map_bar(item) for item in result.candles)
        bars[ticker] = mapped
        series_stats[ticker] = _stats(result, mapped)
    benchmark_mapped = tuple(_map_bar(item) for item in benchmark_result.candles)
    series_stats[BENCHMARK_CODE] = _stats(benchmark_result, benchmark_mapped)
    return AcquiredMarketHistory(
        security_bars=bars,
        benchmark_bars=benchmark_mapped,
        acquisition={
            "requested_from": date_from.isoformat(),
            "requested_to": date_till.isoformat(),
            "tickers": list(normalized),
            "benchmark": BENCHMARK_CODE,
            "provider": "MOEX_ISS",
            "endpoint_kind": "official_daily_candles_interval_24",
            "zero_cost": True,
            "paid_services": False,
            "max_concurrency": max(1, max_concurrency),
            "series": series_stats,
        },
    )


def _map_bar(item: object) -> DailyBar:
    from src.market_data.infrastructure.moex_client import MoexDailyCandle

    if not isinstance(item, MoexDailyCandle):
        raise TypeError("unexpected MOEX daily candle type")
    return DailyBar(
        ticker=item.security_code,
        trade_date=item.trade_date,
        open=float(item.open),
        close=float(item.close),
        high=float(item.high),
        low=float(item.low),
        volume=float(item.volume),
        value=float(item.value),
    )


def _stats(result: MoexDailyFetchResult, bars: tuple[DailyBar, ...]) -> dict[str, Any]:
    dates = [item.trade_date for item in bars]
    rejected_reasons = Counter(item.reason for item in result.rejected_rows)
    return {
        "rows_received": result.rows_received,
        "rows_valid": result.rows_valid,
        "rows_rejected": result.rows_rejected,
        "rejected_reason_distribution": dict(sorted(rejected_reasons.items())),
        "pages_received": result.pages_received,
        "date_from": min(dates).isoformat() if dates else None,
        "date_to": max(dates).isoformat() if dates else None,
    }
