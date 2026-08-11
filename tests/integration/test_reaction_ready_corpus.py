from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from src.events.infrastructure.repositories import SqlAlchemyEventAnalysisRepository
from src.historical_news.application.use_cases import (
    IngestHistoricalNews,
    IngestHistoricalNewsCommand,
)
from src.historical_news.domain.entities import HistoricalNewsSource
from src.historical_news.domain.enums import ContentStoragePolicy, HistoricalNewsSourceKind
from src.historical_news.infrastructure.local_archive import LocalArchiveNewsSource
from src.historical_news.infrastructure.repositories import SqlAlchemyHistoricalNewsRepository
from src.instruments.domain.entities import Instrument, IssuerAlias
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.instruments.infrastructure.seed import SEED_INSTRUMENTS
from src.market_data.domain.entities import BenchmarkCandle, MarketBenchmark, MarketCandle
from src.market_data.infrastructure.repositories import SqlAlchemyMarketDataRepository
from src.ml_features.application.feature_builder import BuildMlFeatureDataset
from src.ml_features.domain.entities import FeatureDatasetConfig
from src.ml_features.infrastructure.repositories import SqlAlchemyMlFeatureRepository
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.reaction_ready_corpus.application import (
    PrepareCorpusCommand,
    PrepareReactionReadyCorpus,
)
from src.reaction_ready_corpus.reporting import (
    batch_001_reaction_count,
    build_and_write_reports,
    load_acquisition_run,
    load_candidate_snapshots,
)
from src.reactions.application.use_cases import CalculateNewsMarketReactions
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository
from src.shared.database.base import Base

PUBLISHED_AT = datetime(2026, 6, 5, 15, 0, tzinfo=UTC)
SOURCE_CODE = "ROSNEFT_PRESS_RELEASES_RSS"


@pytest.fixture
async def corpus_session(tmp_path: Path) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'corpus.sqlite3'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
    await engine.dispose()


async def test_real_source_reaches_reaction_and_feature_ready_without_live_http(
    corpus_session: AsyncSession, tmp_path: Path
) -> None:
    instrument_id = await _seed_rosneft(corpus_session)
    source_file = tmp_path / "issuer-rss-fixture.jsonl"
    source_file.write_text(json.dumps(_source_row()) + "\n", encoding="utf-8")
    ingester = IngestHistoricalNews(
        repository=SqlAlchemyHistoricalNewsRepository(corpus_session),
        news_repository=SqlAlchemyNewsRepository(corpus_session),
        instrument_repository=SqlAlchemyInstrumentRepository(corpus_session),
        source_client=LocalArchiveNewsSource(source_file, max_items=10),
    )
    source = HistoricalNewsSource.create(
        source_code=SOURCE_CODE,
        source_kind=HistoricalNewsSourceKind.ISSUER_RSS,
        content_storage_policy=ContentStoragePolicy.EXCERPT_ALLOWED,
        source_timezone="Europe/Moscow",
        feed_url="https://www.rosneft.com/press/releases/rss/",
    )
    command = IngestHistoricalNewsCommand(
        date_from=PUBLISHED_AT - timedelta(days=1),
        date_to=PUBLISHED_AT + timedelta(days=1),
        limit=10,
        max_pages=1,
        match_instruments=True,
    )
    first = await ingester.execute(source=source, command=command)
    rerun = await ingester.execute(source=source, command=command)
    assert first.run_id is not None
    assert (first.discovered_count, first.validated_count, first.imported_count) == (1, 1, 1)
    assert rerun.imported_count == 0
    assert rerun.duplicate_count == 1

    prepared = await PrepareReactionReadyCorpus(corpus_session).execute(
        PrepareCorpusCommand(
            date_from=PUBLISHED_AT - timedelta(days=1),
            date_to=PUBLISHED_AT + timedelta(days=1),
            source_codes=(SOURCE_CODE,),
            tickers=("ROSN",),
            limit=10,
        )
    )
    assert (prepared.candidate_count, prepared.matched_count, prepared.analyzed_count) == (1, 1, 1)
    assert prepared.ambiguous_count == 0
    assert prepared.windows[0].ticker == "ROSN"

    snapshots = await load_candidate_snapshots(
        corpus_session,
        date_from=PUBLISHED_AT - timedelta(days=1),
        date_to=PUBLISHED_AT + timedelta(days=1),
    )
    news_id = snapshots[0].news_id
    assert news_id is not None
    await _save_market_data(corpus_session, instrument_id)
    reaction_result = await CalculateNewsMarketReactions(
        news_repository=SqlAlchemyNewsRepository(corpus_session),
        instrument_repository=SqlAlchemyInstrumentRepository(corpus_session),
        market_data_repository=SqlAlchemyMarketDataRepository(corpus_session),
        reaction_repository=SqlAlchemyReactionRepository(corpus_session),
    ).execute(news_id)
    assert len(reaction_result.reactions) == 1
    abnormal = reaction_result.reactions[0].points[0].benchmark_adjustment
    assert abnormal is not None
    assert abnormal.abnormal_simple_return is not None

    config = FeatureDatasetConfig(
        date_from=PUBLISHED_AT - timedelta(days=1),
        date_to=PUBLISHED_AT + timedelta(days=1),
        tickers=("ROSN",),
        limit=10,
    )
    feature_result = await BuildMlFeatureDataset(
        repository=SqlAlchemyMlFeatureRepository(corpus_session),
        event_repository=SqlAlchemyEventAnalysisRepository(corpus_session),
        reaction_repository=SqlAlchemyReactionRepository(corpus_session),
    ).execute(config=config, git_sha="integration", dry_run=False)
    assert len(feature_result.rows) == 1
    assert feature_result.rows[0].metadata["ticker"] == "ROSN"
    assert feature_result.rows[0].labels["1m"]["available"] is True

    snapshots = await load_candidate_snapshots(
        corpus_session,
        date_from=PUBLISHED_AT - timedelta(days=1),
        date_to=PUBLISHED_AT + timedelta(days=1),
    )
    paths = build_and_write_reports(
        tmp_path / "artifacts",
        snapshots=snapshots,
        feature_result=feature_result,
        acquisition=await load_acquisition_run(corpus_session, first.run_id),
        date_from=config.date_from,
        date_to=config.date_to,
        git_sha="integration",
        batch_reactions=await batch_001_reaction_count(corpus_session),
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["real_reaction_ready_rows"] == 1
    assert manifest["reaction_rows"] == 1
    assert manifest["feature_rows"] == 1
    assert manifest["correction_count"] == 0
    assert manifest["batch_001_reaction_count"] == 0
    assert len(paths["corpus"].read_text(encoding="utf-8").splitlines()) == 1


async def test_large_market_batches_use_bounded_existing_key_queries(
    corpus_session: AsyncSession,
) -> None:
    instrument_id = await _seed_rosneft(corpus_session)
    repository = SqlAlchemyMarketDataRepository(corpus_session)
    security = [
        _security_candle(
            instrument_id,
            PUBLISHED_AT + timedelta(minutes=minute),
            Decimal("100"),
        )
        for minute in range(1200)
    ]
    first_security = await repository.save_candles(security)
    second_security = await repository.save_candles(security)
    assert (first_security.inserted, second_security.existing) == (1200, 1200)

    benchmark = await repository.save_benchmark(
        MarketBenchmark.create(code="IMOEX", name="Synthetic IMOEX", board="SNDX")
    )
    benchmark_rows = [
        _benchmark_candle(
            benchmark.id,
            PUBLISHED_AT + timedelta(minutes=minute),
            Decimal("1000"),
        )
        for minute in range(1200)
    ]
    first_benchmark = await repository.save_benchmark_candles(benchmark_rows)
    second_benchmark = await repository.save_benchmark_candles(benchmark_rows)
    assert (first_benchmark.inserted, second_benchmark.existing) == (1200, 1200)


def _source_row() -> dict[str, object]:
    return {
        "schema_version": "historical-news-source-v1",
        "source_item_id": "rosneft-release-1",
        "source_url": "https://www.rosneft.com/press/releases/item/1/",
        "title": "Rosneft announces operating update",
        "published_at": PUBLISHED_AT.isoformat(),
        "source_timezone": "Europe/Moscow",
        "content": "Rosneft announces a production update for the quarter.",
        "content_storage_policy": "EXCERPT_ALLOWED",
        "content_is_excerpt": True,
    }


async def _seed_rosneft(session: AsyncSession) -> UUID:
    seed = next(item for item in SEED_INSTRUMENTS if item.ticker == "ROSN")
    repository = SqlAlchemyInstrumentRepository(session)
    instrument = (
        await repository.save_instrument(
            Instrument.create(
                ticker=seed.ticker,
                figi=None,
                isin=None,
                short_name=seed.short_name,
                full_name=seed.full_name,
                issuer_name=seed.issuer_name,
                exchange="MOEX",
                currency="RUB",
                instrument_type=seed.instrument_type,
                primary_board=seed.primary_board,
            )
        )
    ).instrument
    for alias in seed.aliases:
        await repository.save_alias(
            IssuerAlias.create(
                instrument_id=instrument.id,
                alias=alias.alias,
                alias_type=alias.alias_type,
                priority=alias.priority,
            )
        )
    return instrument.id


async def _save_market_data(session: AsyncSession, instrument_id: UUID) -> None:
    repository = SqlAlchemyMarketDataRepository(session)
    security_candles = [
        _security_candle(instrument_id, PUBLISHED_AT - timedelta(minutes=offset), Decimal("100"))
        for offset in (120, 60, 30, 15, 5, 1)
    ]
    security_candles.extend(
        _security_candle(
            instrument_id,
            PUBLISHED_AT + timedelta(minutes=minute),
            Decimal("100") + Decimal(minute) / Decimal("100"),
        )
        for minute in range(0, 61)
    )
    await repository.save_candles(security_candles)
    benchmark = await repository.save_benchmark(
        MarketBenchmark.create(code="IMOEX", name="Synthetic IMOEX", board="SNDX")
    )
    benchmark_candles = [
        _benchmark_candle(benchmark.id, PUBLISHED_AT - timedelta(minutes=offset), Decimal("1000"))
        for offset in (120, 60, 30, 15, 5, 1)
    ]
    benchmark_candles.extend(
        _benchmark_candle(
            benchmark.id,
            PUBLISHED_AT + timedelta(minutes=minute),
            Decimal("1000") + Decimal(minute) / Decimal("10"),
        )
        for minute in range(0, 61)
    )
    await repository.save_benchmark_candles(benchmark_candles)


def _security_candle(instrument_id: UUID, begin_at: datetime, price: Decimal) -> MarketCandle:
    return MarketCandle.create(
        instrument_id=instrument_id,
        board="TQBR",
        ticker_snapshot="ROSN",
        interval_minutes=1,
        begin_at=begin_at,
        end_at=begin_at + timedelta(seconds=59),
        open_price=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("10"),
        value=price * Decimal("10"),
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
