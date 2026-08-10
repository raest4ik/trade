from __future__ import annotations

import argparse
import asyncio
from datetime import date

from src.market_data.application.exceptions import MarketDataApplicationError
from src.market_data.application.use_cases import (
    BackfillBenchmarkCandles,
    BackfillBenchmarkCandlesCommand,
)
from src.market_data.infrastructure.moex_client import MoexIssClient
from src.market_data.infrastructure.repositories import SqlAlchemyMarketDataRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            async with MoexIssClient(
                base_url=settings.moex_iss_base_url,
                timeout_seconds=settings.moex_http_timeout_seconds,
                max_retries=settings.moex_http_max_retries,
                max_pages=settings.moex_http_max_pages,
                user_agent=settings.moex_http_user_agent,
            ) as client:
                result = await BackfillBenchmarkCandles(
                    market_data_repository=SqlAlchemyMarketDataRepository(session),
                    provider=client,
                ).execute(
                    BackfillBenchmarkCandlesCommand(
                        benchmark_code=args.code,
                        date_from=date.fromisoformat(args.date_from),
                        date_till=date.fromisoformat(args.date_till),
                    )
                )
    except (MarketDataApplicationError, ValueError) as exc:
        print(str(exc))
        return 1
    finally:
        await engine.dispose()
    item = result.import_record
    print(
        " ".join(
            [
                f"import_id={item.id}",
                f"dataset_type={item.dataset_type.value}",
                f"benchmark={result.benchmark.code}",
                f"status={item.status.value}",
                f"rows_inserted={item.rows_inserted}",
                f"rows_existing={item.rows_existing}",
                f"rows_rejected={item.rows_rejected}",
            ]
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill MOEX benchmark minute candles.")
    parser.add_argument("code", choices=("IMOEX",))
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--till", dest="date_till", required=True)
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
