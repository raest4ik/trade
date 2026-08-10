from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.events.domain.analyzer import EventAnalyzer
from src.events.infrastructure.repositories import SqlAlchemyEventAnalysisRepository
from src.historical_news.application.use_cases import (
    IngestHistoricalNews,
    IngestHistoricalNewsCommand,
)
from src.historical_news.domain.entities import (
    HistoricalNewsPage,
    HistoricalNewsSource,
    HistoricalSourceItem,
)
from src.historical_news.domain.enums import ContentStoragePolicy, HistoricalNewsSourceKind
from src.historical_news.infrastructure.repositories import SqlAlchemyHistoricalNewsRepository
from src.instruments.application.use_cases import MatchNewsInstruments
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.market_data.domain.entities import BenchmarkCandle, MarketBenchmark, MarketCandle
from src.market_data.infrastructure.repositories import SqlAlchemyMarketDataRepository
from src.ml_features.application.feature_builder import BuildMlFeatureDataset
from src.ml_features.domain.entities import FeatureDatasetConfig
from src.ml_features.infrastructure.export import write_dataset_artifacts
from src.ml_features.infrastructure.repositories import SqlAlchemyMlFeatureRepository
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.reactions.application.use_cases import CalculateNewsMarketReactions
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory

PUBLISHED_AT = datetime(2026, 7, 1, 7, 0, tzinfo=UTC)


class SyntheticHistoricalSource:
    def __init__(self, source_item_id: str) -> None:
        self._source_item_id = source_item_id

    async def fetch_items(
        self,
        *,
        from_datetime: datetime,
        to_datetime: datetime,
        cursor: str | None,
        limit: int,
    ) -> HistoricalNewsPage:
        del from_datetime, to_datetime, limit
        if cursor is not None:
            return HistoricalNewsPage(items=[], next_cursor=None)
        return HistoricalNewsPage(
            items=[
                HistoricalSourceItem(
                    source_item_id=self._source_item_id,
                    source_url=f"https://example.invalid/{self._source_item_id}",
                    title="SBER synthetic financial results",
                    published_at_text=PUBLISHED_AT.isoformat(),
                    source_timezone="UTC",
                    content=(
                        "SBER published financial results. "
                        "Net profit increased by 18% to RUB 118 billion."
                    ),
                    content_storage_policy=ContentStoragePolicy.FULL_TEXT_ALLOWED,
                    content_is_excerpt=False,
                    original_timestamp_text=PUBLISHED_AT.isoformat(),
                    corrects_source_item_id=None,
                    fetched_at=PUBLISHED_AT + timedelta(seconds=1),
                )
            ],
            next_cursor=None,
        )


async def run(output: Path) -> int:
    engine = create_engine(get_settings().database_url)
    try:
        async with create_session_factory(engine)() as session:
            instruments = SqlAlchemyInstrumentRepository(session)
            instrument = await instruments.get_instrument_by_ticker("SBER")
            if instrument is None:
                raise RuntimeError("SBER is missing; run apps.api.seed_instruments first")
            source_item_id = f"ml-feature-smoke-{uuid4()}"
            news_repository = SqlAlchemyNewsRepository(session)
            ingestion = await IngestHistoricalNews(
                repository=SqlAlchemyHistoricalNewsRepository(session),
                news_repository=news_repository,
                instrument_repository=instruments,
                source_client=SyntheticHistoricalSource(source_item_id),
            ).execute(
                source=HistoricalNewsSource.create(
                    source_code="ML_FEATURE_SYNTHETIC_SMOKE",
                    source_kind=HistoricalNewsSourceKind.LOCAL_ARCHIVE,
                    content_storage_policy=ContentStoragePolicy.FULL_TEXT_ALLOWED,
                    source_timezone="UTC",
                ),
                command=IngestHistoricalNewsCommand(
                    date_from=PUBLISHED_AT - timedelta(days=1),
                    date_to=PUBLISHED_AT + timedelta(days=1),
                    limit=1,
                    max_pages=1,
                ),
            )
            if ingestion.run_id is None:
                raise RuntimeError("synthetic historical ingestion did not create a run")
            news = await news_repository.get_by_source("ML_FEATURE_SYNTHETIC_SMOKE", source_item_id)
            if news is None:
                raise RuntimeError("synthetic historical news was not promoted")
            await MatchNewsInstruments(news_repository, instruments).execute(news.id)
            await SqlAlchemyEventAnalysisRepository(session).replace_analysis(
                EventAnalyzer().analyze(news_id=news.id, raw_content=news.raw_content)
            )
            await _save_market_fixture(session, instrument.id)
            await CalculateNewsMarketReactions(
                news_repository=news_repository,
                instrument_repository=instruments,
                market_data_repository=SqlAlchemyMarketDataRepository(session),
                reaction_repository=SqlAlchemyReactionRepository(session),
                horizons_minutes=(15,),
            ).execute(news.id)
            config = FeatureDatasetConfig(
                date_from=PUBLISHED_AT - timedelta(minutes=1),
                date_to=PUBLISHED_AT + timedelta(minutes=1),
                tickers=("SBER",),
                limit=1,
                require_label_horizon=15,
            )
            result = await BuildMlFeatureDataset(
                repository=SqlAlchemyMlFeatureRepository(session),
                event_repository=SqlAlchemyEventAnalysisRepository(session),
                reaction_repository=SqlAlchemyReactionRepository(session),
            ).execute(config=config, git_sha="SYNTHETIC_SMOKE", dry_run=False)
        if len(result.rows) != 1:
            raise RuntimeError("synthetic smoke did not build exactly one row")
        paths = write_dataset_artifacts(output, result=result, config=config)
    finally:
        await engine.dispose()
    row = result.rows[0]
    label = row.labels["15m"]["abnormal_simple_return"]
    if label in row.features.values() or "abnormal_simple_return" in row.features:
        raise RuntimeError("post-event label leaked into features")
    print(f"row_count=1 output={paths['jsonl']}")
    print(
        " ".join(
            [
                f"sber_pre_return_15m={row.features['pre_return_15m']}",
                f"imoex_pre_return_15m={row.features['imoex_pre_return_15m']}",
                f"pre_abnormal_return_15m={row.features['pre_abnormal_return_15m']}",
                f"post_abnormal_label_15m={label}",
                "label_absent_from_features=true",
            ]
        )
    )
    return 0


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
                effective=True,
            ),
            _security_candle(
                instrument_id,
                PUBLISHED_AT + timedelta(minutes=16),
                Decimal("101.202"),
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
    effective: bool = False,
) -> MarketCandle:
    begin_at = PUBLISHED_AT + timedelta(seconds=1) if effective else end_at - timedelta(seconds=59)
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


def _benchmark_candle(
    benchmark_id: UUID,
    end_at: datetime,
    price: Decimal,
) -> BenchmarkCandle:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic PostgreSQL ML feature dataset smoke fixture."
    )
    parser.add_argument("--output", default="artifacts/ml-feature-dataset-v1")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(Path(args.output))))


if __name__ == "__main__":
    main()
