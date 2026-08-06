from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.instruments.domain.entities import Instrument, InstrumentMatch, NewsInstrumentMatch
from src.instruments.domain.enums import AliasType, InstrumentType, MatchType
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.market_data.domain.entities import MarketCandle
from src.market_data.infrastructure.repositories import SqlAlchemyMarketDataRepository
from src.news.domain.entities import NewsItem
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.reactions.application.use_cases import CalculateNewsMarketReactions
from src.reactions.domain.enums import ReactionPointStatus, ReactionStatus
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository
from src.shared.database.base import Base


async def test_market_candle_repository_is_idempotent_and_queries_ranges(
    tmp_path: Path,
) -> None:
    session_factory = await _session_factory(tmp_path)
    async with session_factory() as session:
        repository = SqlAlchemyMarketDataRepository(session)
        instrument_id = uuid4()
        candle = _candle(instrument_id, datetime(2026, 7, 1, 7, 0, tzinfo=UTC), "100")

        first = await repository.save_candles([candle])
        second = await repository.save_candles([candle])
        listed = await repository.list_candles(
            instrument_id=instrument_id,
            interval_minutes=1,
            from_at=datetime(2026, 7, 1, 6, 59, tzinfo=UTC),
            till_at=datetime(2026, 7, 1, 7, 2, tzinfo=UTC),
            limit=10,
            offset=0,
        )
        baseline = await repository.get_last_candle_ending_at_or_before(
            instrument_id=instrument_id,
            interval_minutes=1,
            at=datetime(2026, 7, 1, 7, 1, tzinfo=UTC),
        )

    assert first.inserted == 1
    assert second.existing == 1
    assert [item.id for item in listed] == [candle.id]
    assert baseline is not None
    assert baseline.close == Decimal("100")


async def test_reaction_calculation_uses_previous_complete_candle_for_baseline(
    tmp_path: Path,
) -> None:
    session_factory = await _session_factory(tmp_path)
    async with session_factory() as session:
        news_repository = SqlAlchemyNewsRepository(session)
        instrument_repository = SqlAlchemyInstrumentRepository(session)
        market_repository = SqlAlchemyMarketDataRepository(session)
        reaction_repository = SqlAlchemyReactionRepository(session)
        instrument = (
            await instrument_repository.save_instrument(
                Instrument.create(
                    ticker="SBER",
                    figi=None,
                    isin=None,
                    short_name="Sber",
                    full_name="Sber",
                    issuer_name="Sber",
                    exchange="MOEX",
                    currency="RUB",
                    instrument_type=InstrumentType.COMMON_STOCK,
                    primary_board="TQBR",
                )
            )
        ).instrument
        news = (
            await news_repository.save(
                NewsItem.create(
                    source_id="source",
                    source_name="Source",
                    source_url="https://example.com/1",
                    title="SBER",
                    raw_content="SBER news",
                    language="en",
                    published_at=datetime(2026, 7, 1, 7, 0, 30, tzinfo=UTC),
                    received_at=datetime(2026, 7, 1, 7, 0, 31, tzinfo=UTC),
                )
            )
        ).item
        await instrument_repository.replace_news_matches(
            news_id=news.id,
            matcher_version="deterministic-v1",
            matches=[
                NewsInstrumentMatch.create(
                    news_id=news.id,
                    match=InstrumentMatch(
                        instrument_id=instrument.id,
                        ticker="SBER",
                        issuer_name="Sber",
                        matched_alias="SBER",
                        alias_type=AliasType.TICKER,
                        match_type=MatchType.EXACT_TICKER,
                        confidence=1.0,
                        start_position=0,
                        end_position=4,
                        is_ambiguous=False,
                    ),
                )
            ],
        )
        await market_repository.save_candles(
            [
                _candle(instrument.id, datetime(2026, 7, 1, 6, 59, tzinfo=UTC), "100"),
                _candle(instrument.id, datetime(2026, 7, 1, 7, 0, tzinfo=UTC), "999"),
                _candle(instrument.id, datetime(2026, 7, 1, 7, 1, tzinfo=UTC), "101"),
                _candle(instrument.id, datetime(2026, 7, 1, 7, 2, tzinfo=UTC), "101"),
                _candle(instrument.id, datetime(2026, 7, 1, 7, 5, tzinfo=UTC), "105"),
                _candle(instrument.id, datetime(2026, 7, 1, 8, 0, tzinfo=UTC), "160"),
            ]
        )

        result = await CalculateNewsMarketReactions(
            news_repository=news_repository,
            instrument_repository=instrument_repository,
            market_data_repository=market_repository,
            reaction_repository=reaction_repository,
        ).execute(news.id)

    reaction = result.reactions[0]
    assert reaction.status == ReactionStatus.PARTIAL
    assert reaction.baseline_price == Decimal("100")
    assert reaction.baseline_observed_at == datetime(2026, 7, 1, 6, 59, 59, tzinfo=UTC)
    assert reaction.effective_event_at == datetime(2026, 7, 1, 7, 1, tzinfo=UTC)
    first_point = reaction.points[0]
    assert first_point.status == ReactionPointStatus.AVAILABLE
    assert first_point.simple_return == Decimal("0.01")
    assert reaction.publication_to_receipt_ms == 1000
    assert reaction.publication_to_effective_event_ms == 30000


def _candle(instrument_id: UUID, begin_at: datetime, close: str) -> MarketCandle:
    price = Decimal(close)
    return MarketCandle.create(
        instrument_id=instrument_id,
        board="TQBR",
        ticker_snapshot="SBER",
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


async def _session_factory(tmp_path: Path) -> async_sessionmaker[AsyncSession]:
    database_path = tmp_path / "test.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}", pool_pre_ping=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
