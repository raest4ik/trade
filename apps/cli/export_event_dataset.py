from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.events.domain.entities import EVENT_ANALYSIS_VERSION
from src.events.infrastructure.models import NewsEventAnalysisRecord
from src.instruments.infrastructure.models import NewsInstrumentMatchRecord
from src.news.infrastructure.models import NewsItemRecord
from src.reactions.infrastructure.models import NewsMarketReactionRecord
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    rows_written = 0
    lines: list[str] = []
    async with session_factory() as session:
        result = await session.execute(select(NewsItemRecord).order_by(NewsItemRecord.published_at))
        for news in result.scalars():
            analysis = (
                await session.execute(
                    select(NewsEventAnalysisRecord)
                    .where(
                        NewsEventAnalysisRecord.news_id == news.id,
                        NewsEventAnalysisRecord.analysis_version == EVENT_ANALYSIS_VERSION,
                    )
                    .options(
                        selectinload(NewsEventAnalysisRecord.events),
                        selectinload(NewsEventAnalysisRecord.financial_facts),
                    )
                )
            ).scalar_one_or_none()
            matches = (
                await session.execute(
                    select(NewsInstrumentMatchRecord)
                    .where(NewsInstrumentMatchRecord.news_id == news.id)
                    .order_by(NewsInstrumentMatchRecord.start_position)
                )
            ).scalars()
            reactions = (
                await session.execute(
                    select(NewsMarketReactionRecord)
                    .where(NewsMarketReactionRecord.news_id == news.id)
                    .options(selectinload(NewsMarketReactionRecord.points))
                    .order_by(NewsMarketReactionRecord.created_at)
                )
            ).scalars()
            payload = {
                "news": _news_payload(news, include_raw_content=args.include_raw_content),
                "event_analysis": None if analysis is None else _analysis_payload(analysis),
                "instrument_matches": [_match_payload(match) for match in matches],
                "market_reactions": [_reaction_payload(reaction) for reaction in reactions],
            }
            lines.append(json.dumps(_json_value(payload), ensure_ascii=False, sort_keys=True))
            rows_written += 1
    await engine.dispose()
    await asyncio.to_thread(_write_lines, output, lines)
    print(f"rows_written={rows_written} output={output}")
    return 0


def _news_payload(news: NewsItemRecord, *, include_raw_content: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": news.id,
        "source_id": news.source_id,
        "source_name": news.source_name,
        "source_url": news.source_url,
        "title": news.title,
        "raw_content_hash": news.raw_content_hash,
        "language": news.language,
        "published_at": news.published_at,
        "received_at": news.received_at,
        "created_at": news.created_at,
    }
    if include_raw_content:
        payload["raw_content"] = news.raw_content
    return payload


def _analysis_payload(analysis: NewsEventAnalysisRecord) -> dict[str, object]:
    return {
        "id": analysis.id,
        "analysis_version": analysis.analysis_version,
        "status": analysis.status,
        "primary_event_type": analysis.primary_event_type,
        "created_at": analysis.created_at,
        "analyzed_at": analysis.analyzed_at,
        "events": [
            {
                "id": event.id,
                "event_type": event.event_type,
                "confidence": event.confidence,
                "matched_rule": event.matched_rule,
                "evidence_text": event.evidence_text,
                "start_position": event.start_position,
                "end_position": event.end_position,
            }
            for event in sorted(analysis.events, key=lambda item: item.start_position)
        ],
        "financial_facts": [
            {
                "id": fact.id,
                "metric": fact.metric,
                "raw_value": fact.raw_value,
                "normalized_value": fact.normalized_value,
                "unit": fact.unit,
                "currency": fact.currency,
                "scale": fact.scale,
                "period_type": fact.period_type,
                "year": fact.year,
                "quarter": fact.quarter,
                "month": fact.month,
                "date_from": fact.date_from,
                "date_to": fact.date_to,
                "raw_period": fact.raw_period,
                "comparison_type": fact.comparison_type,
                "fact_role": fact.fact_role,
                "change_direction": fact.change_direction,
                "change_value": fact.change_value,
                "change_unit": fact.change_unit,
                "confidence": fact.confidence,
                "evidence_text": fact.evidence_text,
                "start_position": fact.start_position,
                "end_position": fact.end_position,
                "extractor_version": fact.extractor_version,
                "matched_rule": fact.matched_rule,
            }
            for fact in sorted(analysis.financial_facts, key=lambda item: item.start_position)
        ],
    }


def _match_payload(match: NewsInstrumentMatchRecord) -> dict[str, object]:
    return {
        "id": match.id,
        "instrument_id": match.instrument_id,
        "matched_alias": match.matched_alias,
        "alias_type": match.alias_type,
        "match_type": match.match_type,
        "confidence": match.confidence,
        "start_position": match.start_position,
        "end_position": match.end_position,
        "is_ambiguous": match.is_ambiguous,
        "matcher_version": match.matcher_version,
    }


def _reaction_payload(reaction: NewsMarketReactionRecord) -> dict[str, object]:
    return {
        "id": reaction.id,
        "instrument_id": reaction.instrument_id,
        "reaction_version": reaction.reaction_version,
        "published_at": reaction.published_at,
        "received_at": reaction.received_at,
        "effective_event_at": reaction.effective_event_at,
        "baseline_observed_at": reaction.baseline_observed_at,
        "baseline_price": reaction.baseline_price,
        "publication_to_receipt_ms": reaction.publication_to_receipt_ms,
        "publication_to_effective_event_ms": reaction.publication_to_effective_event_ms,
        "status": reaction.status,
        "is_ambiguous_instrument": reaction.is_ambiguous_instrument,
        "points": [
            {
                "horizon_minutes": point.horizon_minutes,
                "target_at": point.target_at,
                "observed_at": point.observed_at,
                "price": point.price,
                "simple_return": point.simple_return,
                "log_return": point.log_return,
                "status": point.status,
            }
            for point in sorted(reaction.points, key=lambda item: item.horizon_minutes)
        ],
    }


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        items = cast("dict[object, object]", value)
        return {str(key): _json_value(item) for key, item in items.items()}
    if isinstance(value, list):
        items = cast("list[object]", value)
        return [_json_value(item) for item in items]
    return value


def _write_lines(output: Path, lines: list[str]) -> None:
    output.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export news event analysis dataset as JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-raw-content", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
