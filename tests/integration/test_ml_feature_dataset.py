from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from src.events.domain.analyzer import EventAnalyzer
from src.events.infrastructure.repositories import SqlAlchemyEventAnalysisRepository
from src.historical_news.application.use_cases import (
    IngestHistoricalNews,
    IngestHistoricalNewsCommand,
)
from src.historical_news.domain.entities import HistoricalNewsSource
from src.historical_news.domain.enums import ContentStoragePolicy, HistoricalNewsSourceKind
from src.historical_news.infrastructure.local_archive import LocalArchiveNewsSource
from src.historical_news.infrastructure.models import HistoricalNewsCandidateRecord
from src.historical_news.infrastructure.repositories import SqlAlchemyHistoricalNewsRepository
from src.instruments.domain.entities import Instrument, IssuerAlias
from src.instruments.domain.enums import AliasType, InstrumentType
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.market_data.domain.entities import BenchmarkCandle, MarketBenchmark, MarketCandle
from src.market_data.infrastructure.repositories import SqlAlchemyMarketDataRepository
from src.ml_features.application.feature_builder import BuildMlFeatureDataset
from src.ml_features.domain.entities import FeatureDatasetConfig
from src.ml_features.domain.enums import FeatureExclusionReason
from src.ml_features.infrastructure.export import write_dataset_artifacts
from src.ml_features.infrastructure.models import MlFeatureDatasetRunRecord
from src.ml_features.infrastructure.repositories import SqlAlchemyMlFeatureRepository
from src.news.domain.entities import NewsItem
from src.news.domain.enums import PublicationTimestampQuality
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.reactions.application.use_cases import CalculateNewsMarketReactions
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository
from src.shared.database.base import Base

PUBLISHED_AT = datetime(2026, 7, 1, 7, 0, tzinfo=UTC)


@pytest.fixture
async def ml_session(tmp_path: Path) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ml-features.sqlite3'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
    await engine.dispose()


async def test_synthetic_exact_news_builds_leakage_safe_ml_row(
    ml_session: AsyncSession,
    tmp_path: Path,
) -> None:
    instrument_id = await _save_sber(ml_session)
    news_id = await _ingest_exact_news(ml_session, tmp_path, match=True)
    news = await SqlAlchemyNewsRepository(ml_session).get_by_id(news_id)
    assert news is not None
    analysis = EventAnalyzer().analyze(news_id=news.id, raw_content=news.raw_content)
    await SqlAlchemyEventAnalysisRepository(ml_session).replace_analysis(analysis)
    await _save_market_fixture(ml_session, instrument_id)
    reaction_result = await CalculateNewsMarketReactions(
        news_repository=SqlAlchemyNewsRepository(ml_session),
        instrument_repository=SqlAlchemyInstrumentRepository(ml_session),
        market_data_repository=SqlAlchemyMarketDataRepository(ml_session),
        reaction_repository=SqlAlchemyReactionRepository(ml_session),
        horizons_minutes=(15,),
    ).execute(news_id)
    adjustment = reaction_result.reactions[0].points[0].benchmark_adjustment
    assert adjustment is not None
    assert adjustment.abnormal_simple_return == Decimal("0.006")

    config = _config(require_label_horizon=15)
    build = await BuildMlFeatureDataset(
        repository=SqlAlchemyMlFeatureRepository(ml_session),
        event_repository=SqlAlchemyEventAnalysisRepository(ml_session),
        reaction_repository=SqlAlchemyReactionRepository(ml_session),
    ).execute(
        config=config,
        git_sha="synthetic-e2e",
        dry_run=False,
        generated_at=PUBLISHED_AT,
    )
    assert build.run.built_count == 1
    row = build.rows[0]
    assert row.features["primary_event_type"] == "FINANCIAL_RESULTS"
    assert row.features["net_profit_change_pct"] == Decimal("18")
    assert row.features["pre_return_15m"] == Decimal("0.002")
    assert row.features["imoex_pre_return_15m"] == Decimal("0.001")
    assert row.features["pre_abnormal_return_15m"] == Decimal("0.001")
    assert row.labels["15m"]["abnormal_simple_return"] == Decimal("0.006")
    assert Decimal("0.006") not in row.features.values()
    assert "abnormal_simple_return" not in row.features
    run_record = (await ml_session.execute(select(MlFeatureDatasetRunRecord))).scalar_one()
    assert run_record.config_hash == config.hash()
    assert run_record.built_count == 1

    paths = write_dataset_artifacts(tmp_path / "output", result=build, config=config)
    assert paths["jsonl"].exists()
    assert paths["csv"].exists()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["row_count"] == 1
    assert manifest["training_readiness"] == ("MODEL_TRAINING_NOT_READY_INSUFFICIENT_REAL_ROWS")


async def test_sql_point_in_time_query_excludes_future_candle(
    ml_session: AsyncSession,
) -> None:
    instrument_id = await _save_sber(ml_session)
    market = SqlAlchemyMarketDataRepository(ml_session)
    await market.save_candles(
        [
            _security_candle(instrument_id, PUBLISHED_AT, Decimal("100")),
            _security_candle(
                instrument_id,
                PUBLISHED_AT + timedelta(seconds=1),
                Decimal("999"),
            ),
        ]
    )
    candles = await SqlAlchemyMlFeatureRepository(ml_session).list_security_candles_as_of(
        instrument_id=instrument_id,
        as_of=PUBLISHED_AT,
        lookback_minutes=60,
    )
    assert [item.close for item in candles] == [Decimal("100")]
    assert all(item.end_at <= PUBLISHED_AT for item in candles)


async def test_historical_candidates_have_deterministic_order(
    ml_session: AsyncSession,
    tmp_path: Path,
) -> None:
    await _save_sber(ml_session)
    path = tmp_path / "ordered.jsonl"
    rows = [
        _source_row("later", "2026-07-02T10:00:00Z"),
        _source_row("earlier", "2026-07-01T10:00:00Z"),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    await _ingest_path(ml_session, path, source_code="ORDERED", match=True)
    candidates = await SqlAlchemyMlFeatureRepository(ml_session).list_candidates(_config())
    assert [item.source_item_id for item in candidates] == ["earlier", "later"]


async def test_date_only_candidate_is_excluded_and_never_built(
    ml_session: AsyncSession,
    tmp_path: Path,
) -> None:
    await _save_sber(ml_session)
    path = tmp_path / "date-only.jsonl"
    path.write_text(
        json.dumps(_source_row("date-only", "2026-07-01")) + "\n",
        encoding="utf-8",
    )
    await _ingest_path(ml_session, path, source_code="DATE_ONLY_SOURCE", match=True)
    build = await BuildMlFeatureDataset(
        repository=SqlAlchemyMlFeatureRepository(ml_session),
        event_repository=SqlAlchemyEventAnalysisRepository(ml_session),
        reaction_repository=SqlAlchemyReactionRepository(ml_session),
    ).execute(config=_config(), git_sha="test", dry_run=True)
    assert build.rows == []
    assert build.exclusions[0].reason == FeatureExclusionReason.TIMESTAMP_NOT_EXACT


async def test_unknown_candidate_and_batch_001_are_not_in_training_candidates(
    ml_session: AsyncSession,
    tmp_path: Path,
) -> None:
    path = tmp_path / "unknown.jsonl"
    path.write_text(
        json.dumps(_source_row("unknown", "2026-07-01T10:00:00", timezone=None)) + "\n",
        encoding="utf-8",
    )
    await _ingest_path(ml_session, path, source_code="UNKNOWN_SOURCE", match=False)
    await SqlAlchemyNewsRepository(ml_session).save(
        NewsItem.create(
            source_id="batch-001",
            source_name="seed-dataset",
            source_url="https://example.invalid/batch-001",
            title="Batch 001",
            raw_content="SBER batch fixture",
            language="en",
            published_at=PUBLISHED_AT,
            received_at=PUBLISHED_AT,
            publication_timestamp_quality=PublicationTimestampQuality.DATE_ONLY,
        )
    )
    candidates = await SqlAlchemyMlFeatureRepository(ml_session).list_candidates(_config())
    assert candidates == []
    assert await ml_session.scalar(select(func.count(HistoricalNewsCandidateRecord.id))) == 1


async def test_dry_run_does_not_persist_dataset_run(
    ml_session: AsyncSession,
) -> None:
    await BuildMlFeatureDataset(
        repository=SqlAlchemyMlFeatureRepository(ml_session),
        event_repository=SqlAlchemyEventAnalysisRepository(ml_session),
        reaction_repository=SqlAlchemyReactionRepository(ml_session),
    ).execute(config=_config(), git_sha="dry", dry_run=True)
    assert await ml_session.scalar(select(func.count(MlFeatureDatasetRunRecord.id))) == 0


def _config(*, require_label_horizon: int | None = None) -> FeatureDatasetConfig:
    return FeatureDatasetConfig(
        date_from=datetime(2026, 1, 1, tzinfo=UTC),
        date_to=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        require_label_horizon=require_label_horizon,
    )


async def _save_sber(session: AsyncSession) -> UUID:
    repository = SqlAlchemyInstrumentRepository(session)
    instrument = (
        await repository.save_instrument(
            Instrument.create(
                ticker="SBER",
                figi=None,
                isin=None,
                short_name="Sber",
                full_name="Sber synthetic",
                issuer_name="Sber",
                exchange="MOEX",
                currency="RUB",
                instrument_type=InstrumentType.COMMON_STOCK,
                primary_board="TQBR",
            )
        )
    ).instrument
    await repository.save_alias(
        IssuerAlias.create(
            instrument_id=instrument.id,
            alias="SBER",
            alias_type=AliasType.TICKER,
            priority=1,
        )
    )
    return instrument.id


async def _ingest_exact_news(session: AsyncSession, tmp_path: Path, *, match: bool) -> UUID:
    path = tmp_path / "exact.jsonl"
    path.write_text(
        json.dumps(_source_row("exact", "2026-07-01T07:00:00Z")) + "\n",
        encoding="utf-8",
    )
    await _ingest_path(session, path, source_code="ML_SYNTHETIC", match=match)
    candidate = (await session.execute(select(HistoricalNewsCandidateRecord))).scalar_one()
    assert candidate.imported_news_id is not None
    return candidate.imported_news_id


async def _ingest_path(
    session: AsyncSession,
    path: Path,
    *,
    source_code: str,
    match: bool,
) -> None:
    await IngestHistoricalNews(
        repository=SqlAlchemyHistoricalNewsRepository(session),
        news_repository=SqlAlchemyNewsRepository(session),
        instrument_repository=SqlAlchemyInstrumentRepository(session),
        source_client=LocalArchiveNewsSource(path),
    ).execute(
        source=HistoricalNewsSource.create(
            source_code=source_code,
            source_kind=HistoricalNewsSourceKind.LOCAL_ARCHIVE,
            content_storage_policy=ContentStoragePolicy.FULL_TEXT_ALLOWED,
        ),
        command=IngestHistoricalNewsCommand(
            date_from=datetime(2026, 1, 1, tzinfo=UTC),
            date_to=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
            limit=100,
            max_pages=10,
            match_instruments=match,
        ),
    )


def _source_row(
    source_item_id: str,
    published_at: str,
    *,
    timezone: str | None = "Europe/Moscow",
) -> dict[str, object]:
    return {
        "schema_version": "historical-news-source-v1",
        "source_item_id": source_item_id,
        "source_url": f"https://example.invalid/{source_item_id}",
        "title": "SBER financial results",
        "published_at": published_at,
        "source_timezone": timezone,
        "content": (
            "SBER published financial results. Net profit increased by 18% to RUB 118 billion."
        ),
        "content_storage_policy": "FULL_TEXT_ALLOWED",
    }


async def _save_market_fixture(session: AsyncSession, instrument_id: UUID) -> None:
    market = SqlAlchemyMarketDataRepository(session)
    await market.save_candles(
        [
            _security_candle(
                instrument_id,
                PUBLISHED_AT - timedelta(minutes=15),
                Decimal("100"),
            ),
            _security_candle(instrument_id, PUBLISHED_AT, Decimal("100.2")),
            _security_candle(
                instrument_id,
                PUBLISHED_AT + timedelta(minutes=1),
                Decimal("100.2"),
                begin_after_cutoff=True,
            ),
            _security_candle(
                instrument_id,
                PUBLISHED_AT + timedelta(minutes=16),
                Decimal("101.202"),
                begin_after_cutoff=True,
            ),
        ]
    )
    benchmark = await market.save_benchmark(
        MarketBenchmark.create(code="IMOEX", name="Synthetic IMOEX", board="SNDX")
    )
    await market.save_benchmark_candles(
        [
            _benchmark_candle(
                benchmark.id,
                PUBLISHED_AT - timedelta(minutes=15),
                Decimal("1000"),
            ),
            _benchmark_candle(benchmark.id, PUBLISHED_AT, Decimal("1001")),
            _benchmark_candle(
                benchmark.id,
                PUBLISHED_AT + timedelta(minutes=16),
                Decimal("1005.004"),
            ),
        ]
    )


def _security_candle(
    instrument_id: UUID,
    end_at: datetime,
    price: Decimal,
    *,
    begin_after_cutoff: bool = False,
) -> MarketCandle:
    begin_at = (
        PUBLISHED_AT + timedelta(seconds=1)
        if begin_after_cutoff and end_at == PUBLISHED_AT + timedelta(minutes=1)
        else end_at - timedelta(seconds=59)
    )
    return MarketCandle.create(
        instrument_id=instrument_id,
        board="TQBR",
        ticker_snapshot="SBER",
        interval_minutes=1,
        begin_at=begin_at,
        end_at=end_at,
        open_price=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("10"),
        value=price * Decimal("10"),
    )


def _benchmark_candle(benchmark_id: UUID, end_at: datetime, price: Decimal) -> BenchmarkCandle:
    return BenchmarkCandle.create(
        benchmark_id=benchmark_id,
        interval_minutes=1,
        begin_at=end_at - timedelta(seconds=59),
        end_at=end_at,
        open_price=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1"),
        value=price,
    )
