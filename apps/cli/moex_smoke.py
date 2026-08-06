from __future__ import annotations

import asyncio
from datetime import date
from uuid import uuid4

from src.market_data.infrastructure.moex_client import MoexIssClient
from src.shared.config.settings import get_settings


async def run() -> int:
    settings = get_settings()
    async with MoexIssClient(
        base_url=settings.moex_iss_base_url,
        timeout_seconds=settings.moex_http_timeout_seconds,
        max_retries=settings.moex_http_max_retries,
        max_pages=10,
        user_agent=settings.moex_http_user_agent,
    ) as client:
        result = await client.fetch_candles_with_rejections(
            instrument_id=uuid4(),
            ticker="SBER",
            board="TQBR",
            date_from=date(2026, 7, 1),
            date_till=date(2026, 7, 2),
            interval_minutes=1,
        )
    print(
        f"candles={len(result.candles)} rows_received={result.rows_received} "
        f"rows_rejected={result.rows_rejected}"
    )
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(run()))
    except Exception as exc:
        print(f"MOEX smoke failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
