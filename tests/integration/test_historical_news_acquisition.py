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

from src.historical_news.application.exceptions import HistoricalNewsIngestionError
from src.historical_news.application.use_cases import (
    HistoricalNewsIngestionResult,
    IngestHistoricalNews,
    IngestHistoricalNewsCommand,
)
from src.historical_news.domain.entities import HistoricalNewsSource
from src.historical_news.domain.enums import (
    ContentStoragePolicy,
    HistoricalNewsCandidateStatus,
    HistoricalNewsImportStatus,
    HistoricalNewsSourceKind,
)
from src.historical_news.infrastructure.local_archive import LocalArchiveNewsSource
from src.historical_news.infrastructure.models import (
    HistoricalNewsCandidateRecord,
    HistoricalNewsImportRunRecord,
    HistoricalNewsSourceRecord,
)
from src.historical_news.infrastructure.reporting import (
    corpus_stats,
    load_corpus_rows,
    write_corpus,
)
from src.historical_news.infrastructure.repositories import SqlAlchemyHistoricalNewsRepository
from src.instruments.domain.entities import Instrument, IssuerAlias
from src.instruments.domain.enums import AliasType, InstrumentType
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.market_data.domain.entities import BenchmarkCandle, MarketBenchmark, MarketCandle
from src.market_data.infrastructure.repositories import SqlAlchemyMarketDataRepository
from src.news.domain.enums import PublicationTimestampQuality
from src.news.infrastructure.models import NewsItemRecord
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.reactions.application.exceptions import ReactionTimestampIneligibleError
from src.reactions.application.use_cases import CalculateNewsMarketReactions
from src.reactions.infrastructure.models import NewsMarketReactionRecord
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository
from src.shared.database.base import Base

DATE_FROM = datetime(2026, 1, 1, tzinfo=UTC)
DATE_TO = datetime(2026, 12, 31, 23, 59, tzinfo=UTC)


@pytest.fixture
async def historical_session(tmp_path: Path) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'historical.sqlite3'}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
    await engine.dispose()


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "historical-news-source-v1",
        "source_item_id": "exact-1",
        "source_url": "https://example.invalid/exact-1",
        "title": "SBER historical test",
        "published_at": "2026-07-01T10:00:00+03:00",
        "source_timezone": "Europe/Moscow",
        "content": "SBER publishes a synthetic historical update.",
        "content_storage_policy": "FULL_TEXT_ALLOWED",
    }
    row.update(overrides)
    return row


async def _ingest(
    session: AsyncSession,
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    source_code: str = "TEST_ARCHIVE",
    source_policy: ContentStoragePolicy = ContentStoragePolicy.FULL_TEXT_ALLOWED,
    source_timezone: str | None = "Europe/Moscow",
    match: bool = False,
    dry_run: bool = False,
) -> HistoricalNewsIngestionResult:
    path = tmp_path / f"{source_code}.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return await IngestHistoricalNews(
        repository=SqlAlchemyHistoricalNewsRepository(session),
        news_repository=SqlAlchemyNewsRepository(session),
        instrument_repository=SqlAlchemyInstrumentRepository(session),
        source_client=LocalArchiveNewsSource(path, max_items=100),
    ).execute(
        source=HistoricalNewsSource.create(
            source_code=source_code,
            source_kind=HistoricalNewsSourceKind.LOCAL_ARCHIVE,
            content_storage_policy=source_policy,
            source_timezone=source_timezone,
        ),
        command=IngestHistoricalNewsCommand(
            date_from=DATE_FROM,
            date_to=DATE_TO,
            limit=100,
            max_pages=10,
            dry_run=dry_run,
            match_instruments=match,
        ),
    )


async def test_candidate_is_staged_promoted_and_run_is_audited(
    historical_session: AsyncSession, tmp_path: Path
) -> None:
    result = await _ingest(historical_session, tmp_path, [_row()])
    candidate = (
        await historical_session.execute(select(HistoricalNewsCandidateRecord))
    ).scalar_one()
    run = (await historical_session.execute(select(HistoricalNewsImportRunRecord))).scalar_one()
    assert result.imported_count == 1
    assert candidate.status == HistoricalNewsCandidateStatus.IMPORTED.value
    assert candidate.imported_news_id is not None
    assert run.status == HistoricalNewsImportStatus.SUCCEEDED.value
    assert (run.discovered_count, run.validated_count, run.imported_count) == (1, 1, 1)


async def test_idempotent_rerun_does_not_duplicate_news(
    historical_session: AsyncSession, tmp_path: Path
) -> None:
    await _ingest(historical_session, tmp_path, [_row()])
    rerun = await _ingest(historical_session, tmp_path, [_row()])
    assert rerun.duplicate_count == 1
    assert await historical_session.scalar(select(func.count(NewsItemRecord.id))) == 1
    assert (
        await historical_session.scalar(select(func.count(HistoricalNewsCandidateRecord.id))) == 1
    )


@pytest.mark.parametrize(
    ("source_policy", "item_overrides", "expected_status"),
    [
        (ContentStoragePolicy.METADATA_ONLY, {}, "METADATA_ONLY"),
        (ContentStoragePolicy.EXCERPT_ALLOWED, {}, "METADATA_ONLY"),
        (
            ContentStoragePolicy.EXCERPT_ALLOWED,
            {"content_storage_policy": "EXCERPT_ALLOWED", "content_is_excerpt": True},
            "IMPORTED",
        ),
    ],
)
async def test_storage_policy_controls_promotion(
    historical_session: AsyncSession,
    tmp_path: Path,
    source_policy: ContentStoragePolicy,
    item_overrides: dict[str, object],
    expected_status: str,
) -> None:
    await _ingest(
        historical_session,
        tmp_path,
        [_row(**item_overrides)],
        source_policy=source_policy,
    )
    candidate = (
        await historical_session.execute(select(HistoricalNewsCandidateRecord))
    ).scalar_one()
    assert candidate.status == expected_status
    if expected_status == "METADATA_ONLY":
        assert candidate.content is None
        assert await historical_session.scalar(select(func.count(NewsItemRecord.id))) == 0


async def test_required_full_text_missing_is_rejected(
    historical_session: AsyncSession, tmp_path: Path
) -> None:
    result = await _ingest(historical_session, tmp_path, [_row(content=None)])
    candidate = (
        await historical_session.execute(select(HistoricalNewsCandidateRecord))
    ).scalar_one()
    assert result.rejected_count == 1
    assert candidate.rejection_reason == "required_content_missing"


async def test_unknown_timezone_is_staged_but_not_promoted(
    historical_session: AsyncSession, tmp_path: Path
) -> None:
    result = await _ingest(
        historical_session,
        tmp_path,
        [_row(published_at="2026-07-01T10:00:00", source_timezone=None)],
        source_timezone=None,
    )
    candidate = (
        await historical_session.execute(select(HistoricalNewsCandidateRecord))
    ).scalar_one()
    assert result.rejected_count == 1
    assert candidate.publication_timestamp_quality == PublicationTimestampQuality.UNKNOWN.value
    assert candidate.imported_news_id is None


async def test_date_only_is_promoted_but_reaction_ineligible(
    historical_session: AsyncSession, tmp_path: Path
) -> None:
    await _ingest(historical_session, tmp_path, [_row(published_at="2026-07-01")])
    candidate = (
        await historical_session.execute(select(HistoricalNewsCandidateRecord))
    ).scalar_one()
    assert candidate.publication_timestamp_quality == PublicationTimestampQuality.DATE_ONLY.value
    assert candidate.imported_news_id is not None
    with pytest.raises(ReactionTimestampIneligibleError):
        await CalculateNewsMarketReactions(
            news_repository=SqlAlchemyNewsRepository(historical_session),
            instrument_repository=SqlAlchemyInstrumentRepository(historical_session),
            market_data_repository=SqlAlchemyMarketDataRepository(historical_session),
            reaction_repository=SqlAlchemyReactionRepository(historical_session),
        ).execute(candidate.imported_news_id)


async def test_cross_source_exact_duplicate_is_flagged_not_deleted(
    historical_session: AsyncSession, tmp_path: Path
) -> None:
    await _ingest(historical_session, tmp_path, [_row()], source_code="SOURCE_A")
    await _ingest(
        historical_session,
        tmp_path,
        [_row(source_item_id="other-id", source_url="https://example.invalid/other")],
        source_code="SOURCE_B",
    )
    candidates = (
        (
            await historical_session.execute(
                select(HistoricalNewsCandidateRecord).order_by(
                    HistoricalNewsCandidateRecord.created_at
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(candidates) == 2
    assert candidates[0].exact_content_duplicate is False
    assert candidates[1].exact_content_duplicate is True


async def test_explicit_correction_links_without_overwriting_original(
    historical_session: AsyncSession, tmp_path: Path
) -> None:
    await _ingest(
        historical_session,
        tmp_path,
        [
            _row(),
            _row(
                source_item_id="correction-2",
                source_url="https://example.invalid/correction-2",
                content="Corrected synthetic content",
                corrects_source_item_id="exact-1",
            ),
        ],
    )
    candidates = (
        (
            await historical_session.execute(
                select(HistoricalNewsCandidateRecord).order_by(
                    HistoricalNewsCandidateRecord.source_item_id
                )
            )
        )
        .scalars()
        .all()
    )
    correction = next(item for item in candidates if item.source_item_id == "correction-2")
    original = next(item for item in candidates if item.source_item_id == "exact-1")
    assert correction.supersedes_candidate_id == original.id
    assert original.content == "SBER publishes a synthetic historical update."


async def test_dry_run_writes_nothing(historical_session: AsyncSession, tmp_path: Path) -> None:
    result = await _ingest(historical_session, tmp_path, [_row()], dry_run=True)
    assert result.run_id is None
    assert result.validated_count == 1
    assert await historical_session.scalar(select(func.count(HistoricalNewsSourceRecord.id))) == 0
    assert (
        await historical_session.scalar(select(func.count(HistoricalNewsCandidateRecord.id))) == 0
    )


async def test_failed_source_is_recorded_in_import_run(
    historical_session: AsyncSession,
) -> None:
    class FailingSource:
        async def fetch_items(self, **kwargs: object) -> object:
            del kwargs
            raise RuntimeError("synthetic source failure")

    use_case = IngestHistoricalNews(
        repository=SqlAlchemyHistoricalNewsRepository(historical_session),
        news_repository=SqlAlchemyNewsRepository(historical_session),
        instrument_repository=SqlAlchemyInstrumentRepository(historical_session),
        source_client=FailingSource(),  # type: ignore[arg-type]
    )
    with pytest.raises(HistoricalNewsIngestionError):
        await use_case.execute(
            source=HistoricalNewsSource.create(
                source_code="FAIL_SOURCE",
                source_kind=HistoricalNewsSourceKind.LOCAL_ARCHIVE,
                content_storage_policy=ContentStoragePolicy.FULL_TEXT_ALLOWED,
            ),
            command=IngestHistoricalNewsCommand(DATE_FROM, DATE_TO, 10, 2),
        )
    run = (await historical_session.execute(select(HistoricalNewsImportRunRecord))).scalar_one()
    assert run.status == HistoricalNewsImportStatus.FAILED.value
    assert run.error == "synthetic source failure"


async def test_matcher_export_stats_and_reaction_ready_filter(
    historical_session: AsyncSession, tmp_path: Path
) -> None:
    await _save_sber(historical_session)
    result = await _ingest(historical_session, tmp_path, [_row()], match=True)
    rows = await load_corpus_rows(historical_session)
    stats = corpus_stats(rows)
    output = tmp_path / "reaction-ready.jsonl"
    written = write_corpus(
        output,
        rows,
        reaction_ready_only=True,
        include_content=False,
    )
    assert result.matched_news_count == 1
    assert rows[0]["reaction_ready"] is True
    assert stats["matched_count"] == 1
    assert stats["reaction_ready_count"] == 1
    assert stats["by_ticker"] == {"SBER": 1}
    assert written == 1
    assert "content" not in json.loads(output.read_text(encoding="utf-8"))


async def test_exact_historical_candidate_reaches_abnormal_reaction(
    historical_session: AsyncSession, tmp_path: Path
) -> None:
    instrument_id = await _save_sber(historical_session)
    await _ingest(historical_session, tmp_path, [_row()], match=True)
    candidate = (
        await historical_session.execute(select(HistoricalNewsCandidateRecord))
    ).scalar_one()
    assert candidate.imported_news_id is not None
    market = SqlAlchemyMarketDataRepository(historical_session)
    published_at = datetime(2026, 7, 1, 7, 0, tzinfo=UTC)
    await market.save_candles(
        [
            _security_candle(instrument_id, published_at - timedelta(minutes=1), Decimal("100")),
            _security_candle(instrument_id, published_at, Decimal("100")),
            _security_candle(instrument_id, published_at + timedelta(minutes=1), Decimal("102")),
        ]
    )
    benchmark = await market.save_benchmark(
        MarketBenchmark.create(code="IMOEX", name="Synthetic IMOEX", board="SNDX")
    )
    await market.save_benchmark_candles(
        [
            _benchmark_candle(benchmark.id, published_at - timedelta(minutes=1), Decimal("1000")),
            _benchmark_candle(benchmark.id, published_at + timedelta(minutes=1), Decimal("1010")),
        ]
    )
    result = await CalculateNewsMarketReactions(
        news_repository=SqlAlchemyNewsRepository(historical_session),
        instrument_repository=SqlAlchemyInstrumentRepository(historical_session),
        market_data_repository=market,
        reaction_repository=SqlAlchemyReactionRepository(historical_session),
        horizons_minutes=(1,),
    ).execute(candidate.imported_news_id)
    adjustment = result.reactions[0].points[0].benchmark_adjustment
    assert adjustment is not None
    assert adjustment.abnormal_simple_return == Decimal("0.01")
    assert await historical_session.scalar(select(func.count(NewsMarketReactionRecord.id))) == 1
    assert (
        await historical_session.scalar(
            select(func.count(NewsMarketReactionRecord.id))
            .join(NewsItemRecord, NewsItemRecord.id == NewsMarketReactionRecord.news_id)
            .where(NewsItemRecord.source_name == "seed-dataset")
        )
        == 0
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


def _security_candle(instrument_id: UUID, begin: datetime, price: Decimal) -> MarketCandle:
    return MarketCandle.create(
        instrument_id=instrument_id,
        board="TQBR",
        ticker_snapshot="SBER",
        interval_minutes=1,
        begin_at=begin,
        end_at=begin + timedelta(seconds=59),
        open_price=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1"),
        value=price,
    )


def _benchmark_candle(benchmark_id: UUID, begin: datetime, price: Decimal) -> BenchmarkCandle:
    return BenchmarkCandle.create(
        benchmark_id=benchmark_id,
        interval_minutes=1,
        begin_at=begin,
        end_at=begin + timedelta(seconds=59),
        open_price=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1"),
        value=price,
    )
