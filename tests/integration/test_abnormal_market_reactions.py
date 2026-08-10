from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.instruments.domain.entities import Instrument, InstrumentMatch, NewsInstrumentMatch
from src.instruments.domain.enums import AliasType, InstrumentType, MatchType
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.market_data.application.use_cases import (
    BackfillBenchmarkCandles,
    BackfillBenchmarkCandlesCommand,
)
from src.market_data.domain.entities import (
    MOEX_INDEX_BOARD,
    BenchmarkCandle,
    MarketBenchmark,
    MarketCandle,
)
from src.market_data.domain.enums import MarketDataSetType
from src.market_data.infrastructure.models import BenchmarkCandleRecord, MarketDataImportRecord
from src.market_data.infrastructure.repositories import SqlAlchemyMarketDataRepository
from src.news.domain.entities import NewsItem
from src.news.domain.enums import PublicationTimestampQuality
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.reactions.application.exceptions import ReactionTimestampIneligibleError
from src.reactions.application.use_cases import CalculateNewsMarketReactions
from src.reactions.domain.entities import DEFAULT_REACTION_HORIZONS_MINUTES
from src.reactions.domain.enums import BenchmarkAdjustmentStatus, ReactionStatus
from src.reactions.infrastructure.models import (
    NewsMarketReactionRecord,
    ReactionBenchmarkAdjustmentRecord,
)
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository
from src.shared.database.base import Base


async def test_benchmark_repository_is_idempotent(tmp_path: Path) -> None:
    factory = await _session_factory(tmp_path)
    async with factory() as session:
        repository = SqlAlchemyMarketDataRepository(session)
        first = await repository.save_benchmark(_benchmark())
        second = await repository.save_benchmark(_benchmark())
    assert first.id == second.id
    assert first.code == "IMOEX"


async def test_benchmark_candle_import_is_idempotent(tmp_path: Path) -> None:
    factory = await _session_factory(tmp_path)
    async with factory() as session:
        repository = SqlAlchemyMarketDataRepository(session)
        benchmark = await repository.save_benchmark(_benchmark())
        candle = _benchmark_candle(benchmark.id, _at(-1), "100")
        first = await repository.save_benchmark_candles([candle])
        second = await repository.save_benchmark_candles([candle])
    assert first.inserted == 1
    assert second.existing == 1


async def test_benchmark_candle_import_deduplicates_one_batch(tmp_path: Path) -> None:
    factory = await _session_factory(tmp_path)
    async with factory() as session:
        repository = SqlAlchemyMarketDataRepository(session)
        benchmark = await repository.save_benchmark(_benchmark())
        candle = _benchmark_candle(benchmark.id, _at(-1), "100")
        result = await repository.save_benchmark_candles([candle, candle])
    assert result.inserted == 1
    assert result.existing == 1


async def test_benchmark_lookup_uses_completed_saved_candles_across_gaps(tmp_path: Path) -> None:
    factory = await _session_factory(tmp_path)
    async with factory() as session:
        repository = SqlAlchemyMarketDataRepository(session)
        benchmark = await repository.save_benchmark(_benchmark())
        await repository.save_benchmark_candles(
            [
                _benchmark_candle(benchmark.id, _at(-5), "99"),
                _benchmark_candle(benchmark.id, _at(5), "101"),
            ]
        )
        baseline = await repository.get_last_benchmark_candle_ending_at_or_before(
            benchmark_id=benchmark.id,
            interval_minutes=1,
            at=_at(0),
        )
        target = await repository.get_first_benchmark_candle_ending_at_or_after(
            benchmark_id=benchmark.id,
            interval_minutes=1,
            at=_at(1),
        )
    assert baseline is not None and baseline.close == Decimal("99")
    assert target is not None and target.close == Decimal("101")


async def test_benchmark_backfill_records_auditable_import(tmp_path: Path) -> None:
    factory = await _session_factory(tmp_path)
    async with factory() as session:
        repository = SqlAlchemyMarketDataRepository(session)
        result = await BackfillBenchmarkCandles(
            market_data_repository=repository,
            provider=_BenchmarkProvider(),
        ).execute(
            BackfillBenchmarkCandlesCommand(
                benchmark_code="IMOEX",
                date_from=date(2026, 7, 1),
                date_till=date(2026, 7, 1),
            )
        )
        record = (
            await session.execute(
                select(MarketDataImportRecord).where(
                    MarketDataImportRecord.id == result.import_record.id
                )
            )
        ).scalar_one()
    assert record.dataset_type == MarketDataSetType.BENCHMARK.value
    assert record.instrument_id is None
    assert record.benchmark_id == result.benchmark.id
    assert record.ticker == "IMOEX"
    assert record.rows_inserted == 1


async def test_exact_timestamp_computes_aligned_abnormal_returns(tmp_path: Path) -> None:
    factory = await _session_factory(tmp_path)
    async with factory() as session:
        result = await _calculate(session, PublicationTimestampQuality.EXACT, benchmark=True)
    reaction = result.reactions[0]
    assert reaction.status == ReactionStatus.COMPLETE
    assert reaction.baseline_observed_at == _at(-1) + timedelta(seconds=59)
    assert reaction.effective_event_at == _at(1)
    assert [point.horizon_minutes for point in reaction.points] == list(
        DEFAULT_REACTION_HORIZONS_MINUTES
    )
    for point in reaction.points:
        adjustment = point.benchmark_adjustment
        assert adjustment is not None
        assert adjustment.status == BenchmarkAdjustmentStatus.AVAILABLE
        assert adjustment.baseline_observed_at == reaction.baseline_observed_at
        assert adjustment.target_observed_at == point.observed_at
        assert point.simple_return == Decimal("0.02")
        assert adjustment.simple_return == Decimal("0.01")
        assert adjustment.abnormal_simple_return == Decimal("0.01")
        assert point.log_return is not None
        assert adjustment.log_return is not None
        assert adjustment.abnormal_log_return == point.log_return - adjustment.log_return


async def test_missing_benchmark_keeps_abnormal_returns_null(tmp_path: Path) -> None:
    factory = await _session_factory(tmp_path)
    async with factory() as session:
        result = await _calculate(session, PublicationTimestampQuality.EXACT, benchmark=False)
    for point in result.reactions[0].points:
        adjustment = point.benchmark_adjustment
        assert adjustment is not None
        assert adjustment.status == BenchmarkAdjustmentStatus.MISSING
        assert adjustment.simple_return is None
        assert adjustment.abnormal_simple_return is None
        assert adjustment.missing_reason == "benchmark_baseline_and_target_candle_missing"


@pytest.mark.parametrize(
    "quality",
    [PublicationTimestampQuality.DATE_ONLY, PublicationTimestampQuality.UNKNOWN],
)
async def test_ineligible_timestamp_creates_no_reaction_rows(
    tmp_path: Path,
    quality: PublicationTimestampQuality,
) -> None:
    factory = await _session_factory(tmp_path)
    async with factory() as session:
        with pytest.raises(ReactionTimestampIneligibleError):
            await _calculate(session, quality, benchmark=True)
        reaction_count = await session.scalar(select(func.count(NewsMarketReactionRecord.id)))
        adjustment_count = await session.scalar(
            select(func.count(ReactionBenchmarkAdjustmentRecord.id))
        )
    assert reaction_count == 0
    assert adjustment_count == 0


async def test_batch_001_marker_date_only_creates_zero_reactions(tmp_path: Path) -> None:
    factory = await _session_factory(tmp_path)
    async with factory() as session:
        with pytest.raises(ReactionTimestampIneligibleError):
            await _calculate(
                session,
                PublicationTimestampQuality.DATE_ONLY,
                benchmark=True,
                source_name="seed-dataset",
                title_suffix="DATE_ONLY / DO_NOT_USE_FOR_REACTION",
            )
        reaction_count = await session.scalar(select(func.count(NewsMarketReactionRecord.id)))
    assert reaction_count == 0


async def test_recomputation_is_idempotent_without_duplicate_adjustments(tmp_path: Path) -> None:
    factory = await _session_factory(tmp_path)
    async with factory() as session:
        first = await _calculate(session, PublicationTimestampQuality.EXACT, benchmark=True)
        news_id = first.news_id
        calculator = CalculateNewsMarketReactions(
            news_repository=SqlAlchemyNewsRepository(session),
            instrument_repository=SqlAlchemyInstrumentRepository(session),
            market_data_repository=SqlAlchemyMarketDataRepository(session),
            reaction_repository=SqlAlchemyReactionRepository(session),
        )
        second = await calculator.execute(news_id)
        reaction_count = await session.scalar(select(func.count(NewsMarketReactionRecord.id)))
        adjustment_count = await session.scalar(
            select(func.count(ReactionBenchmarkAdjustmentRecord.id))
        )
    assert len(second.reactions) == 1
    assert reaction_count == 1
    assert adjustment_count == len(DEFAULT_REACTION_HORIZONS_MINUTES)


async def test_benchmark_candle_storage_uses_decimal_columns(tmp_path: Path) -> None:
    factory = await _session_factory(tmp_path)
    async with factory() as session:
        repository = SqlAlchemyMarketDataRepository(session)
        benchmark = await repository.save_benchmark(_benchmark())
        await repository.save_benchmark_candles(
            [_benchmark_candle(benchmark.id, _at(0), "123.456789")]
        )
        record = (await session.execute(select(BenchmarkCandleRecord))).scalar_one()
    assert isinstance(record.close, Decimal)
    assert record.close == Decimal("123.4567890000")


async def _calculate(
    session: AsyncSession,
    quality: PublicationTimestampQuality,
    *,
    benchmark: bool,
    source_name: str = "integration-test",
    title_suffix: str = "",
):
    news_repository = SqlAlchemyNewsRepository(session)
    instrument_repository = SqlAlchemyInstrumentRepository(session)
    market_repository = SqlAlchemyMarketDataRepository(session)
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
                source_id=str(uuid4()),
                source_name=source_name,
                source_url="https://example.com/exact",
                title=f"SBER {title_suffix}".strip(),
                raw_content="SBER publishes exact timestamp news",
                language="en",
                published_at=_at(0) + timedelta(seconds=30),
                received_at=_at(0) + timedelta(seconds=31),
                publication_timestamp_quality=quality,
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
    target_minutes = [1 + horizon for horizon in DEFAULT_REACTION_HORIZONS_MINUTES]
    await market_repository.save_candles(
        [_security_candle(instrument.id, _at(-1), "100")]
        + [_security_candle(instrument.id, _at(1), "100")]
        + [_security_candle(instrument.id, _at(minutes), "102") for minutes in target_minutes]
    )
    if benchmark:
        benchmark_entity = await market_repository.save_benchmark(_benchmark())
        await market_repository.save_benchmark_candles(
            [_benchmark_candle(benchmark_entity.id, _at(-1), "100")]
            + [
                _benchmark_candle(benchmark_entity.id, _at(minutes), "101")
                for minutes in target_minutes
            ]
        )
    return await CalculateNewsMarketReactions(
        news_repository=news_repository,
        instrument_repository=instrument_repository,
        market_data_repository=market_repository,
        reaction_repository=SqlAlchemyReactionRepository(session),
    ).execute(news.id)


def _benchmark() -> MarketBenchmark:
    return MarketBenchmark.create(
        code="IMOEX",
        name="MOEX Russia Index",
        board=MOEX_INDEX_BOARD,
    )


def _at(minutes: int) -> datetime:
    return datetime(2026, 7, 1, 7, 0, tzinfo=UTC) + timedelta(minutes=minutes)


def _benchmark_candle(benchmark_id: UUID, begin_at: datetime, close: str) -> BenchmarkCandle:
    price = Decimal(close)
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


def _security_candle(instrument_id: UUID, begin_at: datetime, close: str) -> MarketCandle:
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
    database_path = tmp_path / f"{uuid4()}.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}", pool_pre_ping=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class _BenchmarkProvider:
    async def fetch_benchmark_candles(
        self,
        *,
        benchmark: MarketBenchmark,
        date_from: date,
        date_till: date,
        interval_minutes: int,
    ) -> tuple[list[BenchmarkCandle], int, int, int, int]:
        del date_from, date_till
        candle = _benchmark_candle(benchmark.id, _at(0), "100")
        return [candle], 1, 1, 1, 0
