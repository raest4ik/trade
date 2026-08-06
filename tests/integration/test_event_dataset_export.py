from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from pytest import MonkeyPatch
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from apps.cli.export_event_dataset import run
from src.events.application.use_cases import AnalyzeNewsEvent
from src.events.infrastructure.repositories import SqlAlchemyEventAnalysisRepository
from src.news.domain.entities import NewsItem
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.shared.config.settings import get_settings
from src.shared.database.base import Base


async def test_export_event_dataset_writes_jsonl_without_raw_content_by_default(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    database_path = tmp_path / "dataset.sqlite3"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    engine = create_async_engine(database_url, pool_pre_ping=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        news_repository = SqlAlchemyNewsRepository(session)
        event_repository = SqlAlchemyEventAnalysisRepository(session)
        news = (
            await news_repository.save(
                NewsItem.create(
                    source_id="dataset-source",
                    source_name="Dataset Source",
                    source_url="https://example.com/dataset",
                    title="Dataset event",
                    raw_content="Выручка за 2025 год составила 1 млрд руб.",
                    language="ru",
                    published_at=datetime(2026, 8, 6, 3, 0, tzinfo=UTC),
                )
            )
        ).item
        await AnalyzeNewsEvent(
            news_repository=news_repository,
            event_repository=event_repository,
        ).execute(news.id)
    await engine.dispose()

    monkeypatch.setenv("DATABASE_URL", database_url)
    get_settings.cache_clear()
    output = tmp_path / "event-dataset.jsonl"

    exit_code = await run(argparse.Namespace(output=str(output), include_raw_content=False))

    assert exit_code == 0
    [line] = output.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert "raw_content" not in record["news"]
    assert record["news"]["raw_content_hash"]
    assert record["event_analysis"]["primary_event_type"] == "FINANCIAL_RESULTS"
    assert record["event_analysis"]["financial_facts"][0]["metric"] == "REVENUE"
