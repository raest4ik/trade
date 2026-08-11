from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.corpus_quality.domain import PublicationTimeRecord, SourceAcceptanceEvidence
from src.events.domain.entities import EVENT_ANALYSIS_VERSION
from src.events.infrastructure.models import NewsEventAnalysisRecord
from src.historical_news.infrastructure.models import (
    HistoricalNewsCandidateRecord,
    HistoricalNewsSourceRecord,
)
from src.news.infrastructure.models import NewsItemRecord
from src.reaction_ready_corpus.domain import REAL_SOURCE_CODES, SourceAuditStatus
from src.reaction_ready_corpus.reporting import load_candidate_snapshots
from src.reaction_ready_corpus.source_audit import audited_sources


async def load_publication_time_records(
    session: AsyncSession,
    *,
    date_from: datetime,
    date_to: datetime,
    feature_news_ids: set[UUID] | None = None,
    limit: int = 100,
) -> list[PublicationTimeRecord]:
    if not 1 <= limit <= 100:
        raise ValueError("quality corpus load limit must be between 1 and 100")
    snapshots = await load_candidate_snapshots(
        session, date_from=date_from, date_to=date_to, limit=limit
    )
    snapshot_by_news = {item.news_id: item for item in snapshots if item.news_id is not None}
    result = await session.execute(
        select(
            HistoricalNewsCandidateRecord,
            HistoricalNewsSourceRecord,
            NewsItemRecord,
            NewsEventAnalysisRecord,
        )
        .join(
            HistoricalNewsSourceRecord,
            HistoricalNewsSourceRecord.id == HistoricalNewsCandidateRecord.source_id,
        )
        .join(NewsItemRecord, NewsItemRecord.id == HistoricalNewsCandidateRecord.imported_news_id)
        .outerjoin(
            NewsEventAnalysisRecord,
            and_(
                NewsEventAnalysisRecord.news_id == NewsItemRecord.id,
                NewsEventAnalysisRecord.analysis_version == EVENT_ANALYSIS_VERSION,
            ),
        )
        .where(
            HistoricalNewsSourceRecord.source_code.in_(REAL_SOURCE_CODES),
            HistoricalNewsCandidateRecord.source_published_at >= date_from,
            HistoricalNewsCandidateRecord.source_published_at <= date_to,
        )
        .order_by(
            HistoricalNewsCandidateRecord.source_published_at,
            HistoricalNewsCandidateRecord.source_item_id,
        )
        .limit(limit)
        .options(
            selectinload(NewsEventAnalysisRecord.events),
            selectinload(NewsEventAnalysisRecord.financial_facts),
        )
    )
    records: list[PublicationTimeRecord] = []
    for candidate, source, news, analysis in result.all():
        snapshot = snapshot_by_news.get(news.id)
        if snapshot is None:
            continue
        if not snapshot.matches:
            ticker = "UNMATCHED"
        else:
            ticker = snapshot.matches[0].ticker
        records.append(
            PublicationTimeRecord(
                news_id=news.id,
                ticker=ticker,
                source_code=source.source_code,
                source_item_id=candidate.source_item_id,
                source_url=candidate.source_url,
                title=news.title,
                content=news.raw_content,
                published_at=news.published_at,
                timestamp_quality=snapshot.timestamp_quality,
                storage_policy=candidate.content_storage_policy,
                content_is_excerpt=candidate.content_is_excerpt,
                rules_primary_event="UNKNOWN" if analysis is None else analysis.primary_event_type,
                rules_event_count=0 if analysis is None else len(analysis.events),
                rules_fact_count=0 if analysis is None else len(analysis.financial_facts),
                analysis_status="" if analysis is None else analysis.status,
                analysis_warnings=(),
                matched=snapshot.match_status.value == "MATCHED",
                market_data_ready=snapshot.market_data_status.value == "MARKET_DATA_READY",
                reaction_ready=snapshot.reaction_ready,
                feature_ready=news.id in (feature_news_ids or set()),
                valid_label_horizons=snapshot.valid_label_horizons,
            )
        )
    return records


def source_acceptance_evidence() -> list[SourceAcceptanceEvidence]:
    evidence: list[SourceAcceptanceEvidence] = []
    for entry in audited_sources():
        compliant = entry.status == SourceAuditStatus.COMPLIANT
        code = entry.source_code or f"AUDIT_{entry.tickers[0]}"
        evidence.append(
            SourceAcceptanceEvidence(
                source_code=code,
                tickers=entry.tickers,
                source_url=entry.source_url,
                issuer=entry.issuer,
                source_owner=entry.source_owner,
                publication_timestamp_semantics=(
                    f"{entry.timestamp_precision}; {entry.timezone_semantics}"
                ),
                storage_policy=entry.storage_policy.value,
                issuer_owned=entry.source_owner == entry.issuer,
                exact_publication_timestamp=compliant,
                timezone_semantics_confirmed=compliant,
                stable_item_identity=compliant,
                storage_policy_confirmed=compliant,
                https=entry.https,
                bounded_acquisition=compliant,
                blocker=None if compliant else entry.blocker,
            )
        )
    return evidence
