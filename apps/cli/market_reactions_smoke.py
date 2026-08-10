from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.instruments.domain.entities import Instrument, InstrumentMatch, NewsInstrumentMatch
from src.instruments.domain.enums import AliasType, InstrumentType, MatchType
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.market_data.domain.entities import BenchmarkCandle, MarketBenchmark, MarketCandle
from src.market_data.infrastructure.repositories import SqlAlchemyMarketDataRepository
from src.news.domain.entities import NewsItem
from src.news.domain.enums import PublicationTimestampQuality
from src.news.infrastructure.models import NewsItemRecord
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.reactions.application.use_cases import CalculateNewsMarketReactions
from src.reactions.infrastructure.models import NewsMarketReactionRecord
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run() -> int:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        instruments = SqlAlchemyInstrumentRepository(session)
        market = SqlAlchemyMarketDataRepository(session)
        news_repository = SqlAlchemyNewsRepository(session)
        instrument = (
            await instruments.save_instrument(
                Instrument.create(
                    ticker="SMOKE",
                    figi=None,
                    isin=None,
                    short_name="Synthetic Smoke Security",
                    full_name="Synthetic Smoke Security",
                    issuer_name="Synthetic Smoke Security",
                    exchange="MOEX",
                    currency="RUB",
                    instrument_type=InstrumentType.COMMON_STOCK,
                    primary_board="TQBR",
                )
            )
        ).instrument
        published_at = _at(0) + timedelta(seconds=30)
        news = (
            await news_repository.save(
                NewsItem.create(
                    source_id=f"reaction-smoke-{uuid4()}",
                    source_name="synthetic-reaction-smoke",
                    source_url="https://example.invalid/reaction-smoke",
                    title="Synthetic exact-timestamp reaction smoke",
                    raw_content="Synthetic fixture; never production history.",
                    language="en",
                    published_at=published_at,
                    received_at=published_at + timedelta(seconds=1),
                    publication_timestamp_quality=PublicationTimestampQuality.EXACT,
                )
            )
        ).item
        await instruments.replace_news_matches(
            news_id=news.id,
            matcher_version="synthetic-smoke-v1",
            matches=[
                NewsInstrumentMatch.create(
                    news_id=news.id,
                    match=InstrumentMatch(
                        instrument_id=instrument.id,
                        ticker=instrument.ticker,
                        issuer_name=instrument.issuer_name,
                        matched_alias=instrument.ticker,
                        alias_type=AliasType.TICKER,
                        match_type=MatchType.EXACT_TICKER,
                        confidence=1.0,
                        start_position=0,
                        end_position=5,
                        is_ambiguous=False,
                    ),
                )
            ],
        )
        target_minutes = (2, 6, 16, 31, 61)
        await market.save_candles(
            [_security_candle(instrument.id, _at(-1), Decimal("100"))]
            + [_security_candle(instrument.id, _at(1), Decimal("100"))]
            + [
                _security_candle(instrument.id, _at(minute), Decimal("101.5"))
                for minute in target_minutes
            ]
        )
        benchmark = await market.save_benchmark(
            MarketBenchmark.create(
                code="IMOEX",
                name="MOEX Russia Index",
                board="SNDX",
            )
        )
        await market.save_benchmark_candles(
            [_benchmark_candle(benchmark.id, _at(-1), Decimal("1000"))]
            + [
                _benchmark_candle(benchmark.id, _at(minute), Decimal("1009"))
                for minute in target_minutes
            ]
        )
        result = await CalculateNewsMarketReactions(
            news_repository=news_repository,
            instrument_repository=instruments,
            market_data_repository=market,
            reaction_repository=SqlAlchemyReactionRepository(session),
        ).execute(news.id)
        point = next(item for item in result.reactions[0].points if item.horizon_minutes == 15)
        adjustment = point.benchmark_adjustment
        if adjustment is None:
            raise RuntimeError("15m benchmark adjustment was not created")
        date_only_reactions = await _reaction_count_for_quality(
            session, PublicationTimestampQuality.DATE_ONLY
        )
        batch_001_reactions = await session.scalar(
            select(func.count(NewsMarketReactionRecord.id))
            .join(NewsItemRecord, NewsItemRecord.id == NewsMarketReactionRecord.news_id)
            .where(NewsItemRecord.source_name == "seed-dataset")
        )
    await engine.dispose()
    print(f"ticker={instrument.ticker} published_at={published_at.isoformat()}")
    print(
        " ".join(
            [
                "horizon=15m",
                f"security_return={point.simple_return}",
                f"IMOEX_return={adjustment.simple_return}",
                f"abnormal_return={adjustment.abnormal_simple_return}",
            ]
        )
    )
    print(
        f"market_reaction_rows_created={len(result.reactions)} "
        f"date_only_reactions={date_only_reactions} "
        f"batch_001_reactions={batch_001_reactions}"
    )
    return 0


async def _reaction_count_for_quality(
    session: AsyncSession, quality: PublicationTimestampQuality
) -> int:
    count = await session.scalar(
        select(func.count(NewsMarketReactionRecord.id))
        .join(NewsItemRecord, NewsItemRecord.id == NewsMarketReactionRecord.news_id)
        .where(NewsItemRecord.publication_timestamp_quality == quality.value)
    )
    return 0 if count is None else count


def _at(minutes: int) -> datetime:
    return datetime(2026, 7, 1, 7, 0, tzinfo=UTC) + timedelta(minutes=minutes)


def _security_candle(instrument_id: UUID, begin_at: datetime, price: Decimal) -> MarketCandle:
    return MarketCandle.create(
        instrument_id=instrument_id,
        board="TQBR",
        ticker_snapshot="SMOKE",
        interval_minutes=1,
        begin_at=begin_at,
        end_at=begin_at + timedelta(seconds=59),
        open_price=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1"),
        value=price,
    )


def _benchmark_candle(benchmark_id: UUID, begin_at: datetime, price: Decimal) -> BenchmarkCandle:
    return BenchmarkCandle.create(
        benchmark_id=benchmark_id,
        interval_minutes=1,
        begin_at=begin_at,
        end_at=begin_at + timedelta(seconds=59),
        open_price=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1"),
        value=price,
    )


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
