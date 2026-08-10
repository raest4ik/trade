from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from src.instruments.domain.entities import Instrument, NewsInstrumentMatch
from src.market_data.domain.entities import BenchmarkCandle, MarketCandle
from src.ml_features.domain.entities import FeatureDatasetConfig, MlFeatureDatasetRun
from src.news.domain.entities import NewsItem


@dataclass(frozen=True, slots=True)
class CandidateInstrumentMatch:
    match: NewsInstrumentMatch
    instrument: Instrument


@dataclass(frozen=True, slots=True)
class FeatureCandidate:
    news: NewsItem
    source_code: str
    source_item_id: str
    matches: list[CandidateInstrumentMatch]


class MlFeatureRepository(Protocol):
    async def list_candidates(self, config: FeatureDatasetConfig) -> list[FeatureCandidate]: ...

    async def list_security_candles_as_of(
        self,
        *,
        instrument_id: UUID,
        as_of: datetime,
        lookback_minutes: int,
    ) -> list[MarketCandle]: ...

    async def list_benchmark_candles_as_of(
        self,
        *,
        benchmark_code: str,
        as_of: datetime,
        lookback_minutes: int,
    ) -> list[BenchmarkCandle] | None: ...

    async def create_run(self, run: MlFeatureDatasetRun) -> MlFeatureDatasetRun: ...

    async def finish_run(self, run: MlFeatureDatasetRun) -> MlFeatureDatasetRun: ...
