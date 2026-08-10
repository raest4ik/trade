from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from src.evaluation.infrastructure.models import (
    EvaluationDatasetRecord,
    EvaluationExampleRecord,
    EvaluationRunRecord,
    GoldEventRecord,
    GoldFinancialFactRecord,
)
from src.events.infrastructure.models import (
    DetectedEventRecord,
    ExtractedFinancialFactRecord,
    NewsEventAnalysisRecord,
)
from src.historical_news.infrastructure.models import (
    HistoricalNewsCandidateRecord,
    HistoricalNewsImportRunRecord,
    HistoricalNewsSourceRecord,
)
from src.instruments.infrastructure.models import (
    InstrumentRecord,
    IssuerAliasRecord,
    NewsInstrumentMatchRecord,
)
from src.market_data.infrastructure.models import (
    BenchmarkCandleRecord,
    MarketBenchmarkRecord,
    MarketCandleRecord,
    MarketDataImportRecord,
)
from src.news.infrastructure.models import NewsItemRecord
from src.reactions.infrastructure.models import (
    NewsMarketReactionRecord,
    ReactionBenchmarkAdjustmentRecord,
    ReactionPointRecord,
)
from src.shared.config.settings import get_settings
from src.shared.database.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    return get_settings().sync_database_url


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

_ = (
    InstrumentRecord,
    IssuerAliasRecord,
    NewsInstrumentMatchRecord,
    MarketCandleRecord,
    MarketDataImportRecord,
    MarketBenchmarkRecord,
    BenchmarkCandleRecord,
    NewsItemRecord,
    NewsMarketReactionRecord,
    ReactionPointRecord,
    ReactionBenchmarkAdjustmentRecord,
    NewsEventAnalysisRecord,
    DetectedEventRecord,
    ExtractedFinancialFactRecord,
    EvaluationDatasetRecord,
    EvaluationExampleRecord,
    GoldEventRecord,
    GoldFinancialFactRecord,
    EvaluationRunRecord,
    HistoricalNewsSourceRecord,
    HistoricalNewsImportRunRecord,
    HistoricalNewsCandidateRecord,
)
