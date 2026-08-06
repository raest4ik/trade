from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from src.events.application.exceptions import EventAnalysisNewsNotFoundError
from src.events.application.ports import EventAnalysisRepository
from src.events.domain.analyzer import EventAnalyzer
from src.events.domain.entities import EVENT_ANALYSIS_VERSION, NewsEventAnalysis
from src.news.application.ports import NewsRepository


@dataclass(frozen=True, slots=True)
class AnalyzeNewsEventResult:
    analysis: NewsEventAnalysis


class AnalyzeNewsEvent:
    def __init__(
        self,
        *,
        news_repository: NewsRepository,
        event_repository: EventAnalysisRepository,
        analyzer: EventAnalyzer | None = None,
    ) -> None:
        self._news_repository = news_repository
        self._event_repository = event_repository
        self._analyzer = analyzer or EventAnalyzer()

    async def execute(self, news_id: UUID) -> AnalyzeNewsEventResult:
        news = await self._news_repository.get_by_id(news_id)
        if news is None:
            raise EventAnalysisNewsNotFoundError("news item not found")
        analysis = self._analyzer.analyze(news_id=news.id, raw_content=news.raw_content)
        return AnalyzeNewsEventResult(
            analysis=await self._event_repository.replace_analysis(analysis)
        )


class GetNewsEventAnalysis:
    def __init__(self, repository: EventAnalysisRepository) -> None:
        self._repository = repository

    async def execute(self, news_id: UUID) -> NewsEventAnalysis | None:
        return await self._repository.get_by_news_id(
            news_id=news_id,
            analysis_version=EVENT_ANALYSIS_VERSION,
        )
