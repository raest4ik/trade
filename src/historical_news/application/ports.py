from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from src.historical_news.domain.entities import (
    HistoricalNewsCandidate,
    HistoricalNewsImportRun,
    HistoricalNewsPage,
    HistoricalNewsSource,
)


class HistoricalNewsSourceClient(Protocol):
    async def fetch_items(
        self,
        *,
        from_datetime: datetime,
        to_datetime: datetime,
        cursor: str | None,
        limit: int,
    ) -> HistoricalNewsPage: ...


class HistoricalNewsRepository(Protocol):
    async def save_source(self, source: HistoricalNewsSource) -> HistoricalNewsSource: ...

    async def get_source_by_code(self, source_code: str) -> HistoricalNewsSource | None: ...

    async def create_import_run(self, run: HistoricalNewsImportRun) -> HistoricalNewsImportRun: ...

    async def finish_import_run(self, run: HistoricalNewsImportRun) -> HistoricalNewsImportRun: ...

    async def get_candidate(
        self,
        *,
        source_id: UUID,
        source_item_id: str,
    ) -> HistoricalNewsCandidate | None: ...

    async def find_content_duplicate(
        self,
        *,
        content_hash: str,
        excluding_source_id: UUID,
    ) -> HistoricalNewsCandidate | None: ...

    async def save_candidate(
        self, candidate: HistoricalNewsCandidate
    ) -> tuple[HistoricalNewsCandidate, bool]: ...

    async def update_candidate(
        self, candidate: HistoricalNewsCandidate
    ) -> HistoricalNewsCandidate: ...
