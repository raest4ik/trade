from __future__ import annotations

from typing import Protocol
from uuid import UUID

from src.events.domain.entities import NewsEventAnalysis


class EventAnalysisRepository(Protocol):
    async def replace_analysis(self, analysis: NewsEventAnalysis) -> NewsEventAnalysis: ...

    async def get_by_news_id(
        self,
        *,
        news_id: UUID,
        analysis_version: str | None = None,
    ) -> NewsEventAnalysis | None: ...
