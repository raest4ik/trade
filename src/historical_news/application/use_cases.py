from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse
from uuid import UUID

from src.historical_news.application.exceptions import HistoricalNewsIngestionError
from src.historical_news.application.ports import (
    HistoricalNewsRepository,
    HistoricalNewsSourceClient,
)
from src.historical_news.domain.entities import (
    HistoricalNewsCandidate,
    HistoricalNewsImportRun,
    HistoricalNewsSource,
    HistoricalSourceItem,
)
from src.historical_news.domain.enums import (
    ContentStoragePolicy,
    HistoricalNewsCandidateStatus,
    HistoricalNewsImportStatus,
)
from src.historical_news.domain.time import parse_publication_timestamp
from src.instruments.application.use_cases import MatchNewsInstruments
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.news.application.use_cases import CreateNewsItem, CreateNewsItemCommand
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository


@dataclass(frozen=True, slots=True)
class IngestHistoricalNewsCommand:
    date_from: datetime
    date_to: datetime
    limit: int
    max_pages: int
    dry_run: bool = False
    match_instruments: bool = False


@dataclass(frozen=True, slots=True)
class HistoricalNewsIngestionResult:
    run_id: UUID | None
    source_code: str
    status: HistoricalNewsImportStatus
    discovered_count: int
    validated_count: int
    imported_count: int
    duplicate_count: int
    rejected_count: int
    metadata_only_count: int
    matched_news_count: int


@dataclass(slots=True)
class _Counters:
    discovered: int = 0
    validated: int = 0
    imported: int = 0
    duplicate: int = 0
    rejected: int = 0
    metadata_only: int = 0
    matched_news: int = 0


class IngestHistoricalNews:
    def __init__(
        self,
        *,
        repository: HistoricalNewsRepository,
        news_repository: SqlAlchemyNewsRepository,
        instrument_repository: SqlAlchemyInstrumentRepository,
        source_client: HistoricalNewsSourceClient,
    ) -> None:
        self._repository = repository
        self._news_repository = news_repository
        self._instrument_repository = instrument_repository
        self._source_client = source_client

    async def execute(
        self,
        *,
        source: HistoricalNewsSource,
        command: IngestHistoricalNewsCommand,
    ) -> HistoricalNewsIngestionResult:
        if not 1 <= command.limit <= 100_000:
            raise HistoricalNewsIngestionError("limit must be between 1 and 100000")
        if not 1 <= command.max_pages <= 1_000:
            raise HistoricalNewsIngestionError("max_pages must be between 1 and 1000")
        if command.date_to < command.date_from:
            raise HistoricalNewsIngestionError("date_to must not be before date_from")
        saved_source = source
        run = HistoricalNewsImportRun.start(
            source_id=source.id,
            date_from=command.date_from,
            date_to=command.date_to,
        )
        if not command.dry_run:
            saved_source = await self._repository.save_source(source)
            run = await self._repository.create_import_run(
                HistoricalNewsImportRun.start(
                    source_id=saved_source.id,
                    date_from=command.date_from,
                    date_to=command.date_to,
                )
            )
        counters = _Counters()
        cursor: str | None = None
        pages = 0
        try:
            while counters.discovered < command.limit:
                if pages >= command.max_pages:
                    raise HistoricalNewsIngestionError("source pagination exceeded max_pages")
                page = await self._source_client.fetch_items(
                    from_datetime=command.date_from,
                    to_datetime=command.date_to,
                    cursor=cursor,
                    limit=min(500, command.limit - counters.discovered),
                )
                pages += 1
                for item in page.items:
                    await self._process_item(
                        source=saved_source,
                        run_id=run.id,
                        item=item,
                        command=command,
                        counters=counters,
                    )
                    if counters.discovered >= command.limit:
                        break
                if page.next_cursor is None or page.not_modified:
                    break
                cursor = page.next_cursor
        except Exception as exc:
            if not command.dry_run:
                failed = run.finish(
                    status=HistoricalNewsImportStatus.FAILED,
                    discovered_count=counters.discovered,
                    validated_count=counters.validated,
                    imported_count=counters.imported,
                    duplicate_count=counters.duplicate,
                    rejected_count=counters.rejected,
                    metadata_only_count=counters.metadata_only,
                    error=str(exc)[:2000],
                )
                await self._repository.finish_import_run(failed)
            if isinstance(exc, HistoricalNewsIngestionError):
                raise
            raise HistoricalNewsIngestionError("historical ingestion failed") from exc
        status = (
            HistoricalNewsImportStatus.PARTIAL
            if counters.rejected
            else HistoricalNewsImportStatus.SUCCEEDED
        )
        if not command.dry_run:
            finished = run.finish(
                status=status,
                discovered_count=counters.discovered,
                validated_count=counters.validated,
                imported_count=counters.imported,
                duplicate_count=counters.duplicate,
                rejected_count=counters.rejected,
                metadata_only_count=counters.metadata_only,
            )
            await self._repository.finish_import_run(finished)
        return HistoricalNewsIngestionResult(
            run_id=None if command.dry_run else run.id,
            source_code=saved_source.source_code,
            status=status,
            discovered_count=counters.discovered,
            validated_count=counters.validated,
            imported_count=counters.imported,
            duplicate_count=counters.duplicate,
            rejected_count=counters.rejected,
            metadata_only_count=counters.metadata_only,
            matched_news_count=counters.matched_news,
        )

    async def _process_item(
        self,
        *,
        source: HistoricalNewsSource,
        run_id: UUID,
        item: HistoricalSourceItem,
        command: IngestHistoricalNewsCommand,
        counters: _Counters,
    ) -> None:
        counters.discovered += 1
        if not command.dry_run:
            existing = await self._repository.get_candidate(
                source_id=source.id,
                source_item_id=item.source_item_id,
            )
            if existing is not None:
                counters.duplicate += 1
                return
        prepared = await self._prepare_candidate(source=source, run_id=run_id, item=item)
        if prepared.status == HistoricalNewsCandidateStatus.REJECTED:
            counters.rejected += 1
        elif prepared.status == HistoricalNewsCandidateStatus.METADATA_ONLY:
            counters.metadata_only += 1
        else:
            counters.validated += 1
        if command.dry_run:
            return
        candidate, created = await self._repository.save_candidate(prepared)
        if not created:
            counters.duplicate += 1
            return
        if candidate.status != HistoricalNewsCandidateStatus.VALIDATED:
            return
        if candidate.source_published_at is None or candidate.content is None:
            return
        existing_news = await self._news_repository.get_by_source(
            source.source_code,
            candidate.source_item_id,
        )
        if existing_news is None:
            saved = await CreateNewsItem(self._news_repository).execute(
                CreateNewsItemCommand(
                    source_id=candidate.source_item_id,
                    source_name=source.source_code,
                    source_url=candidate.source_url,
                    title=candidate.title,
                    raw_content=candidate.content,
                    language="ru",
                    published_at=candidate.source_published_at,
                    received_at=candidate.fetched_at,
                    publication_timestamp_quality=candidate.publication_timestamp_quality,
                )
            )
            news = saved.item
            duplicate = not saved.created
        else:
            news = existing_news
            duplicate = True
        await self._repository.update_candidate(
            candidate.mark_imported(news.id, duplicate=duplicate)
        )
        if duplicate:
            counters.duplicate += 1
        else:
            counters.imported += 1
        if command.match_instruments:
            matches = await MatchNewsInstruments(
                news_repository=self._news_repository,
                instrument_repository=self._instrument_repository,
            ).execute(news.id)
            if matches.matches:
                counters.matched_news += 1

    async def _prepare_candidate(
        self,
        *,
        source: HistoricalNewsSource,
        run_id: UUID,
        item: HistoricalSourceItem,
    ) -> HistoricalNewsCandidate:
        parsed = parse_publication_timestamp(
            item.published_at_text,
            source_timezone=item.source_timezone or source.source_timezone,
        )
        reason = _identity_error(item)
        effective_policy = effective_storage_policy(
            source.content_storage_policy,
            item.content_storage_policy,
        )
        stored_content = allowed_content(
            item.content,
            policy=effective_policy,
            content_is_excerpt=item.content_is_excerpt,
        )
        content_hash = (
            None
            if stored_content is None
            else hashlib.sha256(stored_content.encode("utf-8")).hexdigest()
        )
        if reason is None and parsed.error is not None:
            reason = parsed.error
        if (
            reason is None
            and stored_content is None
            and effective_policy == ContentStoragePolicy.FULL_TEXT_ALLOWED
        ):
            reason = "required_content_missing"
        if reason is not None:
            status = HistoricalNewsCandidateStatus.REJECTED
        elif stored_content is None:
            status = HistoricalNewsCandidateStatus.METADATA_ONLY
        else:
            status = HistoricalNewsCandidateStatus.VALIDATED
        exact_duplicate = False
        if content_hash is not None:
            exact_duplicate = (
                await self._repository.find_content_duplicate(
                    content_hash=content_hash,
                    excluding_source_id=source.id,
                )
                is not None
            )
        supersedes_id = None
        if item.corrects_source_item_id:
            corrected = await self._repository.get_candidate(
                source_id=source.id,
                source_item_id=item.corrects_source_item_id,
            )
            supersedes_id = None if corrected is None else corrected.id
        return HistoricalNewsCandidate.create(
            source_id=source.id,
            ingestion_run_id=run_id,
            source_item_id=item.source_item_id,
            source_url=item.source_url,
            title=item.title,
            source_published_at=parsed.published_at,
            source_timezone=parsed.source_timezone,
            publication_timestamp_quality=parsed.quality,
            original_timestamp_text=item.original_timestamp_text,
            fetched_at=item.fetched_at,
            content=stored_content,
            content_hash=content_hash,
            content_storage_policy=effective_policy,
            content_is_excerpt=item.content_is_excerpt,
            exact_content_duplicate=exact_duplicate,
            corrects_source_item_id=item.corrects_source_item_id,
            supersedes_candidate_id=supersedes_id,
            status=status,
            rejection_reason=reason,
        )


def _identity_error(item: HistoricalSourceItem) -> str | None:
    if not item.source_item_id.strip():
        return "source_item_id_missing"
    if not item.title.strip():
        return "title_missing"
    parsed_url = urlparse(item.source_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return "source_url_invalid"
    return None


def effective_storage_policy(
    source_policy: ContentStoragePolicy,
    item_policy: ContentStoragePolicy,
) -> ContentStoragePolicy:
    if ContentStoragePolicy.UNKNOWN in {source_policy, item_policy}:
        return ContentStoragePolicy.UNKNOWN
    if ContentStoragePolicy.METADATA_ONLY in {source_policy, item_policy}:
        return ContentStoragePolicy.METADATA_ONLY
    if ContentStoragePolicy.EXCERPT_ALLOWED in {source_policy, item_policy}:
        return ContentStoragePolicy.EXCERPT_ALLOWED
    return ContentStoragePolicy.FULL_TEXT_ALLOWED


def allowed_content(
    content: str | None,
    *,
    policy: ContentStoragePolicy,
    content_is_excerpt: bool,
) -> str | None:
    if content is None or not content.strip():
        return None
    if policy == ContentStoragePolicy.FULL_TEXT_ALLOWED:
        return content
    if policy == ContentStoragePolicy.EXCERPT_ALLOWED and content_is_excerpt:
        return content
    return None
