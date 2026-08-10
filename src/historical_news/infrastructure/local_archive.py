from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from src.historical_news.domain.entities import HistoricalNewsPage, HistoricalSourceItem
from src.historical_news.domain.time import parse_publication_timestamp
from src.historical_news.infrastructure.schemas import HistoricalNewsSourceItemV1


class HistoricalSourceContractError(ValueError):
    """Raised when a configured source returns an invalid payload."""


class LocalArchiveNewsSource:
    def __init__(
        self,
        path: Path,
        *,
        max_items: int = 10_000,
        max_file_bytes: int = 50_000_000,
    ) -> None:
        if not 1 <= max_items <= 100_000:
            raise ValueError("max_items must be between 1 and 100000")
        if not 1 <= max_file_bytes <= 500_000_000:
            raise ValueError("max_file_bytes must be between 1 and 500000000")
        self._path = path
        self._max_items = max_items
        self._max_file_bytes = max_file_bytes
        self._cached_items: list[HistoricalSourceItem] | None = None

    async def fetch_items(
        self,
        *,
        from_datetime: datetime,
        to_datetime: datetime,
        cursor: str | None,
        limit: int,
    ) -> HistoricalNewsPage:
        items = await self._load_items()
        bounded_limit = max(1, min(limit, self._max_items))
        try:
            offset = 0 if cursor is None else int(cursor)
        except ValueError as exc:
            raise HistoricalSourceContractError("local archive cursor must be an integer") from exc
        if offset < 0:
            raise HistoricalSourceContractError("local archive cursor must not be negative")
        filtered = [
            item
            for item in items
            if _within_range(item, from_datetime=from_datetime, to_datetime=to_datetime)
        ][: self._max_items]
        page_items = filtered[offset : offset + bounded_limit]
        next_offset = offset + len(page_items)
        return HistoricalNewsPage(
            items=page_items,
            next_cursor=str(next_offset) if next_offset < len(filtered) else None,
        )

    async def _load_items(self) -> list[HistoricalSourceItem]:
        if self._cached_items is not None:
            return self._cached_items
        if self._path.suffix.lower() != ".jsonl":
            raise HistoricalSourceContractError("local archive must be a JSONL file")
        try:
            if self._path.stat().st_size > self._max_file_bytes:
                raise HistoricalSourceContractError(
                    "local archive exceeds configured max_file_bytes"
                )
            text = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
        except HistoricalSourceContractError:
            raise
        except OSError as exc:
            raise HistoricalSourceContractError("local archive could not be read") from exc
        parsed: list[HistoricalSourceItem] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed.append(HistoricalNewsSourceItemV1.model_validate_json(line).to_source_item())
            except ValidationError as exc:
                raise HistoricalSourceContractError(
                    f"invalid historical-news-source-v1 row at line {line_number}"
                ) from exc
            if len(parsed) > self._max_items:
                raise HistoricalSourceContractError("local archive exceeds configured max_items")
        self._cached_items = parsed
        return parsed


def _within_range(
    item: HistoricalSourceItem,
    *,
    from_datetime: datetime,
    to_datetime: datetime,
) -> bool:
    parsed = parse_publication_timestamp(
        item.published_at_text,
        source_timezone=item.source_timezone,
    )
    if parsed.published_at is None:
        return True
    return from_datetime <= parsed.published_at <= to_datetime
