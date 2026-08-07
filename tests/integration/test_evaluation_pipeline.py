from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from apps.api.main import create_app
from src.evaluation.domain.entities import (
    GOLD_SCHEMA_VERSION,
    AnnotationExample,
    GoldEvent,
    GoldFinancialFact,
)
from src.evaluation.domain.enums import DatasetSplit, ReviewStatus
from src.evaluation.infrastructure.repositories import (
    SqlAlchemyEvaluationRepository,
    dataset_from_examples,
)
from src.events.domain.analyzer import EventAnalyzer
from src.news.domain.hash import calculate_raw_content_hash
from src.shared.config.settings import Settings
from src.shared.database.base import Base


@pytest.fixture
async def evaluation_client(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, async_sessionmaker[AsyncSession]]]:
    database_path = tmp_path / "evaluation.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}", pool_pre_ping=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    app = create_app(
        Settings(database_url=f"sqlite+aiosqlite:///{database_path}"),
        session_factory=session_factory,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client, session_factory
    await engine.dispose()


async def test_evaluation_dataset_api_and_run(
    evaluation_client: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
    tmp_path: Path,
) -> None:
    client, session_factory = evaluation_client
    raw_content = (
        "The company published financial results. Revenue for FY2025 reached 100 million rub."
    )
    created = await client.post(
        "/api/v1/news",
        json={
            "source_id": "evaluation-source-001",
            "source_name": "Evaluation News",
            "source_url": "https://example.com/evaluation/1",
            "title": "Financial results",
            "raw_content": raw_content,
            "language": "en",
            "published_at": "2026-08-06T08:00:00Z",
            "received_at": "2026-08-06T08:00:01Z",
        },
    )
    news_id = str(created.json()["id"])
    analysis = EventAnalyzer().analyze(news_id=UUID(news_id), raw_content=raw_content)
    example = AnnotationExample(
        schema_version=GOLD_SCHEMA_VERSION,
        news_id=analysis.news_id,
        published_at=datetime.fromisoformat("2026-08-06T03:00:00+00:00"),
        raw_content_hash=calculate_raw_content_hash("evaluation-source-001", raw_content),
        split=DatasetSplit.TEST,
        review_status=ReviewStatus.REVIEWED,
        annotator="qa",
        notes=None,
        predicted_events=[],
        predicted_financial_facts=[],
        gold_events=[
            GoldEvent(
                event_type=event.event_type,
                evidence_text=event.evidence_text,
                start_position=event.start_position,
                end_position=event.end_position,
                is_primary=True,
            )
            for event in analysis.events[:1]
        ],
        gold_financial_facts=[
            GoldFinancialFact(
                metric=fact.metric,
                raw_value=fact.raw_value,
                normalized_value=fact.normalized_value,
                unit=fact.unit,
                currency=fact.currency,
                scale=fact.scale,
                period_type=fact.period_type,
                period_year=fact.year,
                period_quarter=fact.quarter,
                period_month=fact.month,
                raw_period=fact.raw_period,
                fact_role=fact.fact_role,
                comparison_type=fact.comparison_type,
                change_direction=fact.change_direction,
                change_value=fact.change_value,
                change_unit=fact.change_unit,
                evidence_text=fact.evidence_text,
                start_position=fact.start_position,
                end_position=fact.end_position,
            )
            for fact in analysis.financial_facts[:1]
        ],
    )
    dataset = dataset_from_examples(
        name="test-dataset",
        source_file_hash="e" * 64,
        examples=[example],
        description="integration",
    )
    async with session_factory() as session:
        repository = SqlAlchemyEvaluationRepository(session)
        imported = await repository.import_dataset(dataset=dataset, examples=[example])
    dataset_id = str(imported.dataset.id)

    listed = await client.get("/api/v1/evaluation/datasets")
    fetched = await client.get(f"/api/v1/evaluation/datasets/{dataset_id}")
    run = await client.post(
        f"/api/v1/evaluation/datasets/{dataset_id}/runs",
        json={"split": "TEST", "output_dir": str(tmp_path / "reports")},
    )
    fetched_run = await client.get(f"/api/v1/evaluation/runs/{run.json()['id']}")

    assert news_id == str(example.news_id)
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == dataset_id
    assert fetched.status_code == 200
    assert fetched.json()["example_count"] == 1
    assert run.status_code == 200
    assert run.json()["status"] == "SUCCEEDED"
    assert run.json()["metrics_json"]["events"]["micro"]["f1"] == 1.0
    assert fetched_run.status_code == 200
    assert fetched_run.json()["id"] == run.json()["id"]
