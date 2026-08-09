from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.evaluation.application.seed_curation import (
    SeedEventRecord,
    SeedSource,
    process_seed_batch,
)
from src.evaluation.domain.entities import GoldEvent, GoldFinancialFact
from src.evaluation.domain.enums import ReviewStatus
from src.evaluation.domain.serialization import annotation_from_json
from src.events.domain.enums import (
    ChangeDirection,
    ComparisonType,
    Currency,
    EventType,
    FactRole,
    FactUnit,
    FinancialMetric,
    PeriodType,
    ValueScale,
)
from src.events.infrastructure.repositories import SqlAlchemyEventAnalysisRepository
from src.instruments.domain.entities import Instrument, IssuerAlias
from src.instruments.domain.enums import AliasType, InstrumentType
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.news.infrastructure.models import NewsItemRecord
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.reactions.infrastructure.models import NewsMarketReactionRecord
from src.shared.database.base import Base


async def test_seed_processing_is_idempotent_and_keeps_review_artifacts_draft(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "seed.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}", pool_pre_ping=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    output_dir = tmp_path / "artifacts" / "seed"
    records = [_seed_record()]

    async with session_factory() as session:
        news_repository = SqlAlchemyNewsRepository(session)
        instrument_repository = SqlAlchemyInstrumentRepository(session)
        event_repository = SqlAlchemyEventAnalysisRepository(session)
        await _seed_sber(instrument_repository)
        first = await process_seed_batch(
            records=records,
            news_repository=news_repository,
            instrument_repository=instrument_repository,
            event_repository=event_repository,
            output_dir=output_dir,
        )
        second = await process_seed_batch(
            records=records,
            news_repository=news_repository,
            instrument_repository=instrument_repository,
            event_repository=event_repository,
            output_dir=output_dir,
        )
        news_count = await session.scalar(select(func.count()).select_from(NewsItemRecord))
        reaction_count = await session.scalar(
            select(func.count()).select_from(NewsMarketReactionRecord)
        )

    review_line = first.review_jsonl_path.read_text(encoding="utf-8").splitlines()[0]
    review_example = annotation_from_json(json.loads(review_line))

    assert first.stats.created == 1
    assert second.stats.created == 0
    assert second.stats.already_exists == 1
    assert news_count == 1
    assert reaction_count == 0
    assert first.mapping_path.exists()
    assert (first.comparison_dir / "metrics.json").exists()
    assert (first.comparison_dir / "errors.jsonl").exists()
    assert first.review_queue_path.exists()
    assert review_example.review_status == ReviewStatus.DRAFT
    assert review_example.predicted_events
    assert review_example.gold_events
    assert "DATE_ONLY / DO_NOT_USE_FOR_REACTION" in str(review_example.notes)
    assert first.stats.instrument_matches_total >= 1
    await engine.dispose()


async def _seed_sber(repository: SqlAlchemyInstrumentRepository) -> None:
    result = await repository.save_instrument(
        Instrument.create(
            ticker="SBER",
            figi=None,
            isin=None,
            short_name="SBER",
            full_name="SBER",
            issuer_name="SBER",
            exchange="MOEX",
            currency="RUB",
            instrument_type=InstrumentType.COMMON_STOCK,
            primary_board="TQBR",
        )
    )
    await repository.save_alias(
        IssuerAlias.create(
            instrument_id=result.instrument.id,
            alias="SBER",
            alias_type=AliasType.TICKER,
            priority=10,
        )
    )


def _seed_record() -> SeedEventRecord:
    text = "SBER published financial results. Revenue for FY2025 reached 100 million rub."
    return SeedEventRecord(
        schema_version="event-seed-v1",
        target_schema="event-gold-v1",
        batch_id="integration-seed",
        record_id="integration-record-001",
        source_published_date="2026-08-06",
        tickers=["SBER"],
        company="SBER",
        quota_category="FINANCIAL_RESULTS",
        annotation_text=text,
        text_origin="SOURCE_BACKED_PARAPHRASE",
        raw_content_hash="not-used-here",
        review_status="SEED_REVIEW_REQUIRED",
        notes=None,
        gold_events=[
            GoldEvent(
                event_type=EventType.FINANCIAL_RESULTS,
                evidence_text="financial results",
                start_position=text.index("financial"),
                end_position=text.index("financial") + len("financial results"),
                is_primary=True,
            )
        ],
        gold_financial_facts=[
            GoldFinancialFact(
                metric=FinancialMetric.REVENUE,
                raw_value=Decimal("100"),
                normalized_value=Decimal("100000000"),
                unit=FactUnit.MONEY,
                currency=Currency.RUB,
                scale=ValueScale.MILLION,
                period_type=PeriodType.YEAR,
                period_year=2025,
                period_quarter=None,
                period_month=None,
                raw_period="FY2025",
                fact_role=FactRole.ACTUAL,
                comparison_type=ComparisonType.NONE,
                change_direction=ChangeDirection.UNCHANGED,
                change_value=None,
                change_unit=None,
                evidence_text="100 million rub",
                start_position=text.index("100"),
                end_position=text.index("100") + len("100 million rub"),
            )
        ],
        source=SeedSource(
            title="Integration source",
            url="https://example.com/source/1",
            tier="PRIMARY",
            support_url=None,
        ),
    )
