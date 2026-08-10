from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.events.domain.entities import EVENT_ANALYSIS_VERSION
from src.events.infrastructure.models import NewsEventAnalysisRecord
from src.historical_news.infrastructure.models import (
    HistoricalNewsCandidateRecord,
    HistoricalNewsImportRunRecord,
    HistoricalNewsSourceRecord,
)
from src.instruments.infrastructure.models import InstrumentRecord, NewsInstrumentMatchRecord
from src.ml_features.domain.entities import LABEL_HORIZONS_MINUTES, FeatureDatasetBuildResult
from src.news.domain.enums import PublicationTimestampQuality
from src.news.infrastructure.models import NewsItemRecord
from src.reaction_ready_corpus.domain import (
    CORPUS_VERSION,
    VERSION_SUMMARY,
    CorpusProvenance,
    ExclusionReason,
    MarketDataStatus,
    MatchStatus,
    classify_provenance,
    match_status,
    readiness_status,
    timestamp_exclusion,
)
from src.reactions.domain.entities import REACTION_VERSION
from src.reactions.domain.enums import (
    BenchmarkAdjustmentStatus,
    ReactionPointStatus,
    ReactionStatus,
)
from src.reactions.infrastructure.models import NewsMarketReactionRecord, ReactionPointRecord


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    instrument_id: UUID
    ticker: str
    is_ambiguous: bool


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    candidate_id: UUID
    news_id: UUID | None
    source_code: str
    source_name: str | None
    source_item_id: str
    source_url: str
    published_at: datetime | None
    timestamp_quality: PublicationTimestampQuality
    storage_policy: str
    imported: bool
    is_correction: bool
    matches: tuple[CandidateMatch, ...]
    event_type: str | None
    market_data_status: MarketDataStatus
    valid_label_horizons: tuple[int, ...]

    @property
    def provenance(self) -> CorpusProvenance:
        return classify_provenance(self.source_code, self.source_name)

    @property
    def match_status(self) -> MatchStatus:
        return match_status(len(self.matches), any(item.is_ambiguous for item in self.matches))

    @property
    def reaction_ready(self) -> bool:
        return bool(self.valid_label_horizons)


@dataclass(frozen=True, slots=True)
class AcquisitionRunSummary:
    discovered: int
    validated: int
    imported: int
    duplicates: int
    rejected: int


async def load_candidate_snapshots(
    session: AsyncSession,
    *,
    date_from: datetime,
    date_to: datetime,
    limit: int = 100_000,
) -> list[CandidateSnapshot]:
    result = await session.execute(
        select(
            HistoricalNewsCandidateRecord, HistoricalNewsSourceRecord, NewsItemRecord.source_name
        )
        .join(
            HistoricalNewsSourceRecord,
            HistoricalNewsSourceRecord.id == HistoricalNewsCandidateRecord.source_id,
        )
        .outerjoin(
            NewsItemRecord, NewsItemRecord.id == HistoricalNewsCandidateRecord.imported_news_id
        )
        .where(
            HistoricalNewsCandidateRecord.source_published_at >= date_from,
            HistoricalNewsCandidateRecord.source_published_at <= date_to,
        )
        .order_by(
            HistoricalNewsCandidateRecord.source_published_at,
            HistoricalNewsCandidateRecord.source_item_id,
        )
        .limit(limit)
    )
    candidate_rows = result.all()
    news_ids = [
        candidate.imported_news_id
        for candidate, _, _ in candidate_rows
        if candidate.imported_news_id
    ]
    matches_by_news: dict[UUID, list[CandidateMatch]] = defaultdict(list)
    event_by_news: dict[UUID, str] = {}
    reaction_state_by_news: dict[UUID, tuple[MarketDataStatus, tuple[int, ...]]] = {}
    if news_ids:
        matches = await session.execute(
            select(NewsInstrumentMatchRecord, InstrumentRecord)
            .join(InstrumentRecord, InstrumentRecord.id == NewsInstrumentMatchRecord.instrument_id)
            .where(NewsInstrumentMatchRecord.news_id.in_(news_ids))
            .order_by(NewsInstrumentMatchRecord.news_id, InstrumentRecord.ticker)
        )
        for match, instrument in matches.all():
            matches_by_news[match.news_id].append(
                CandidateMatch(instrument.id, instrument.ticker, match.is_ambiguous)
            )
        analyses = await session.execute(
            select(NewsEventAnalysisRecord).where(
                NewsEventAnalysisRecord.news_id.in_(news_ids),
                NewsEventAnalysisRecord.analysis_version == EVENT_ANALYSIS_VERSION,
            )
        )
        event_by_news = {item.news_id: item.primary_event_type for item in analyses.scalars()}
        reactions = await session.execute(
            select(NewsMarketReactionRecord)
            .where(
                NewsMarketReactionRecord.news_id.in_(news_ids),
                NewsMarketReactionRecord.reaction_version == REACTION_VERSION,
            )
            .options(
                selectinload(NewsMarketReactionRecord.points).selectinload(
                    ReactionPointRecord.benchmark_adjustment
                )
            )
        )
        grouped_reactions: dict[UUID, list[NewsMarketReactionRecord]] = defaultdict(list)
        for reaction in reactions.scalars():
            grouped_reactions[reaction.news_id].append(reaction)
        reaction_state_by_news = {
            news_id: _reaction_state(items) for news_id, items in grouped_reactions.items()
        }
    snapshots: list[CandidateSnapshot] = []
    for candidate, source, source_name in candidate_rows:
        news_id = candidate.imported_news_id
        market_status, horizons = reaction_state_by_news.get(news_id, (MarketDataStatus.OTHER, ()))
        snapshots.append(
            CandidateSnapshot(
                candidate_id=candidate.id,
                news_id=news_id,
                source_code=source.source_code,
                source_name=source_name,
                source_item_id=candidate.source_item_id,
                source_url=candidate.source_url,
                published_at=candidate.source_published_at,
                timestamp_quality=PublicationTimestampQuality(
                    candidate.publication_timestamp_quality
                ),
                storage_policy=candidate.content_storage_policy,
                imported=news_id is not None,
                is_correction=candidate.corrects_source_item_id is not None,
                matches=tuple(matches_by_news.get(news_id, [])),
                event_type=event_by_news.get(news_id),
                market_data_status=market_status,
                valid_label_horizons=horizons,
            )
        )
    return snapshots


async def load_acquisition_run(session: AsyncSession, run_id: UUID) -> AcquisitionRunSummary:
    result = await session.execute(
        select(HistoricalNewsImportRunRecord).where(HistoricalNewsImportRunRecord.id == run_id)
    )
    run = result.scalar_one()
    return AcquisitionRunSummary(
        discovered=run.discovered_count,
        validated=run.validated_count,
        imported=run.imported_count,
        duplicates=run.duplicate_count,
        rejected=run.rejected_count,
    )


async def batch_001_reaction_count(session: AsyncSession) -> int:
    count = await session.scalar(
        select(func.count(NewsMarketReactionRecord.id))
        .join(NewsItemRecord, NewsItemRecord.id == NewsMarketReactionRecord.news_id)
        .where(NewsItemRecord.source_name == "seed-dataset")
    )
    return int(count or 0)


def build_and_write_reports(
    output_dir: Path,
    *,
    snapshots: list[CandidateSnapshot],
    feature_result: FeatureDatasetBuildResult,
    acquisition: AcquisitionRunSummary,
    date_from: datetime,
    date_to: datetime,
    git_sha: str,
    batch_reactions: int,
    selected_source_codes: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = None if selected_source_codes is None else set(selected_source_codes)
    real_snapshots = [
        item
        for item in snapshots
        if item.provenance == CorpusProvenance.REAL
        and (selected is None or item.source_code in selected)
    ]
    real_rows = [
        row
        for row in feature_result.rows
        if classify_provenance(str(row.metadata.get("source", ""))) == CorpusProvenance.REAL
        and (selected is None or str(row.metadata.get("source", "")) in selected)
    ]
    feature_news_ids = {cast("UUID", row.metadata["news_id"]) for row in real_rows}
    coverage = _coverage(real_snapshots, real_rows)
    funnel = _funnel(real_snapshots, feature_news_ids, acquisition)
    exclusions = _exclusions(real_snapshots, feature_news_ids)
    warnings = _diversity_warnings(coverage, len(real_rows))
    source_policy_summary = dict(
        sorted(Counter(item.storage_policy for item in real_snapshots).items())
    )
    provenance_counts = Counter(
        classify_provenance(str(row.metadata.get("source", ""))).value
        for row in feature_result.rows
    )
    manifest: dict[str, Any] = {
        "schema_version": CORPUS_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "source_codes": sorted({item.source_code for item in real_snapshots}),
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "real_candidates": len(real_snapshots),
        "real_exact": sum(
            item.timestamp_quality == PublicationTimestampQuality.EXACT for item in real_snapshots
        ),
        "real_matched": sum(item.match_status == MatchStatus.MATCHED for item in real_snapshots),
        "real_ambiguous": sum(
            item.match_status == MatchStatus.AMBIGUOUS for item in real_snapshots
        ),
        "real_unmatched": sum(
            item.match_status == MatchStatus.UNMATCHED for item in real_snapshots
        ),
        "market_data_ready": sum(
            item.market_data_status == MarketDataStatus.MARKET_DATA_READY for item in real_snapshots
        ),
        "reaction_rows": sum(item.reaction_ready for item in real_snapshots),
        "feature_rows": len(real_rows),
        "real_reaction_ready_rows": len(real_rows),
        **VERSION_SUMMARY,
        "source_policy_summary": source_policy_summary,
        "correction_count": sum(item.is_correction for item in real_snapshots),
        "funnel": funnel,
        "exclusions_by_reason": exclusions,
        "provenance_feature_rows": dict(sorted(provenance_counts.items())),
        "synthetic_rows_counted_as_real": 0,
        "batch_001_reaction_count": batch_reactions,
        "readiness_status": readiness_status(len(real_rows)).value,
        "readiness_policy": {
            "under_100": "NOT_READY",
            "100_to_499": "PILOT_ONLY",
            "500_to_999": "BASELINE_EXPERIMENT_READY",
            "1000_or_more": "BASELINE_TRAINING_READY",
        },
        "diversity_warnings": warnings,
        "selection_policy": "source + publication date + configured universe only",
        "label_leakage_into_acquisition": False,
    }
    paths = {
        "manifest": output_dir / "manifest.json",
        "coverage": output_dir / "coverage.json",
        "corpus": output_dir / "corpus.jsonl",
        "exclusions": output_dir / "exclusions.jsonl",
    }
    _write_json(paths["manifest"], manifest)
    _write_json(paths["coverage"], coverage)
    _write_corpus(paths["corpus"], real_rows)
    _write_exclusions(paths["exclusions"], real_snapshots, feature_news_ids)
    return paths


def _reaction_state(
    reactions: list[NewsMarketReactionRecord],
) -> tuple[MarketDataStatus, tuple[int, ...]]:
    if any(item.status == ReactionStatus.OUTSIDE_SESSION.value for item in reactions):
        return MarketDataStatus.NON_TRADING_EVENT, ()
    points = [point for reaction in reactions for point in reaction.points]
    security_available = any(
        point.status == ReactionPointStatus.AVAILABLE.value for point in points
    )
    benchmark_available = any(
        point.benchmark_adjustment is not None
        and point.benchmark_adjustment.status == BenchmarkAdjustmentStatus.AVAILABLE.value
        for point in points
    )
    valid = tuple(
        sorted(
            {
                point.horizon_minutes
                for point in points
                if point.status == ReactionPointStatus.AVAILABLE.value
                and point.benchmark_adjustment is not None
                and point.benchmark_adjustment.status == BenchmarkAdjustmentStatus.AVAILABLE.value
                and point.benchmark_adjustment.abnormal_simple_return is not None
            }
        )
    )
    if valid:
        return MarketDataStatus.MARKET_DATA_READY, valid
    if not security_available:
        return MarketDataStatus.MARKET_DATA_MISSING_SECURITY, ()
    if not benchmark_available:
        return MarketDataStatus.MARKET_DATA_MISSING_BENCHMARK, ()
    return MarketDataStatus.OTHER, ()


def _coverage(snapshots: list[CandidateSnapshot], rows: list[Any]) -> dict[str, Any]:
    by_source = Counter(item.source_code for item in snapshots)
    by_ticker = Counter(str(row.metadata["ticker"]) for row in rows)
    by_event = Counter(str(row.features["primary_event_type"]) for row in rows)
    by_month = Counter(
        cast("datetime", row.metadata["published_at"]).strftime("%Y-%m") for row in rows
    )
    by_year = Counter(str(cast("datetime", row.metadata["published_at"]).year) for row in rows)
    timestamp_quality = Counter(item.timestamp_quality.value for item in snapshots)
    matching = Counter(item.match_status.value for item in snapshots)
    label_availability = {
        f"{horizon}m": sum(
            bool(cast("dict[str, Any]", row.labels.get(f"{horizon}m", {})).get("available"))
            for row in rows
        )
        for horizon in LABEL_HORIZONS_MINUTES
    }
    return {
        "schema_version": CORPUS_VERSION,
        "by_source": dict(sorted(by_source.items())),
        "by_ticker": dict(sorted(by_ticker.items())),
        "by_event_type": dict(sorted(by_event.items())),
        "by_month": dict(sorted(by_month.items())),
        "by_year": dict(sorted(by_year.items())),
        "timestamp_quality": dict(sorted(timestamp_quality.items())),
        "matching": dict(sorted(matching.items())),
        "reaction_ready": sum(item.reaction_ready for item in snapshots),
        "feature_ready": len(rows),
        "label_availability": label_availability,
    }


def _funnel(
    snapshots: list[CandidateSnapshot],
    feature_news_ids: set[UUID],
    acquisition: AcquisitionRunSummary,
) -> list[dict[str, Any]]:
    exact = [
        item for item in snapshots if item.timestamp_quality == PublicationTimestampQuality.EXACT
    ]
    matched = [item for item in exact if item.match_status == MatchStatus.MATCHED]
    market_ready = [
        item for item in matched if item.market_data_status == MarketDataStatus.MARKET_DATA_READY
    ]
    reaction_ready = [item for item in market_ready if item.reaction_ready]
    values = (
        ("discovered", acquisition.discovered),
        ("validated", acquisition.validated),
        ("imported", acquisition.imported),
        ("EXACT", len(exact)),
        ("matched", len(matched)),
        ("market-data-ready", len(market_ready)),
        ("reaction-ready", len(reaction_ready)),
        ("feature-ready", len(feature_news_ids)),
    )
    stages: list[dict[str, Any]] = []
    previous: int | None = None
    for name, count in values:
        if previous is None or previous == 0:
            loss = None
        else:
            loss = round((previous - count) * 100 / previous, 2)
        stages.append({"stage": name, "count": count, "loss_from_previous_pct": loss})
        previous = count
    return stages


def _candidate_exclusion(
    item: CandidateSnapshot, feature_news_ids: set[UUID]
) -> ExclusionReason | None:
    if item.provenance != CorpusProvenance.REAL:
        return ExclusionReason.NON_REAL_PROVENANCE
    if not item.imported or item.storage_policy in {"METADATA_ONLY", "UNKNOWN"}:
        return ExclusionReason.STORAGE_POLICY
    timestamp_reason = timestamp_exclusion(item.timestamp_quality)
    if timestamp_reason is not None:
        return timestamp_reason
    if item.match_status == MatchStatus.UNMATCHED:
        return ExclusionReason.UNMATCHED
    if item.match_status == MatchStatus.AMBIGUOUS:
        return ExclusionReason.AMBIGUOUS
    if item.event_type is None:
        return ExclusionReason.NO_EVENT_ANALYSIS
    if item.market_data_status == MarketDataStatus.MARKET_DATA_MISSING_SECURITY:
        return ExclusionReason.SECURITY_MARKET_DATA_MISSING
    if item.market_data_status == MarketDataStatus.MARKET_DATA_MISSING_BENCHMARK:
        return ExclusionReason.IMOEX_DATA_MISSING
    if not item.reaction_ready:
        return ExclusionReason.NO_VALID_REACTION
    if item.news_id not in feature_news_ids:
        return ExclusionReason.NO_VALID_REACTION
    return None


def _exclusions(snapshots: list[CandidateSnapshot], feature_news_ids: set[UUID]) -> dict[str, int]:
    reasons = [
        reason.value
        for item in snapshots
        if (reason := _candidate_exclusion(item, feature_news_ids)) is not None
    ]
    return dict(sorted(Counter(reasons).items()))


def _diversity_warnings(coverage: dict[str, Any], row_count: int) -> list[str]:
    if row_count == 0:
        return ["NO_REAL_FEATURE_ROWS", "LOW_DIVERSITY"]
    dimensions = (
        cast("dict[str, int]", coverage["by_ticker"]),
        cast("dict[str, int]", coverage["by_event_type"]),
        cast("dict[str, int]", coverage["by_month"]),
    )
    return ["LOW_DIVERSITY"] if any(len(values) <= 1 for values in dimensions) else []


def _write_corpus(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            metadata = row.metadata
            event = {
                "analysis_version": metadata["event_analysis_version"],
                "fact_extractor_version": metadata["fact_extractor_version"],
                "primary_event_type": row.features["primary_event_type"],
                "event_count": row.features["event_count"],
                "fact_count": row.features["fact_count"],
            }
            payload = {
                "metadata": {
                    "news_id": metadata["news_id"],
                    "ticker": metadata["ticker"],
                    "published_at": metadata["published_at"],
                    "source": metadata["source"],
                    "source_item_id": metadata["source_item_id"],
                    "dataset_version": metadata["dataset_version"],
                    "feature_version": metadata["feature_version"],
                    "reaction_version": metadata["reaction_version"],
                    "provenance": CorpusProvenance.REAL.value,
                },
                "event": event,
                "features_available_at_publication": row.features,
                "labels": row.labels,
            }
            output.write(
                json.dumps(_json_value(payload), ensure_ascii=False, sort_keys=True) + "\n"
            )


def _write_exclusions(
    path: Path, snapshots: list[CandidateSnapshot], feature_news_ids: set[UUID]
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for item in snapshots:
            reason = _candidate_exclusion(item, feature_news_ids)
            if reason is None:
                continue
            output.write(
                json.dumps(
                    {
                        "candidate_id": str(item.candidate_id),
                        "news_id": None if item.news_id is None else str(item.news_id),
                        "source_code": item.source_code,
                        "source_item_id": item.source_item_id,
                        "reason": reason.value,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_value(value: object) -> object:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        items = cast("dict[object, object]", value)
        return {str(key): _json_value(item) for key, item in items.items()}
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return [_json_value(item) for item in items]
    return value
