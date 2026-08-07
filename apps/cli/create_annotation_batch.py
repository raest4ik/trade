from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import exists, select

from src.evaluation.domain.entities import (
    GOLD_SCHEMA_VERSION,
    AnnotationExample,
)
from src.evaluation.domain.enums import DatasetSplit, ReviewStatus
from src.evaluation.domain.serialization import annotation_to_json
from src.events.domain.analyzer import EventAnalyzer
from src.events.domain.entities import DetectedEvent, ExtractedFinancialFact
from src.events.domain.enums import EventType
from src.instruments.infrastructure.models import InstrumentRecord, NewsInstrumentMatchRecord
from src.news.infrastructure.models import NewsItemRecord
from src.reactions.infrastructure.models import NewsMarketReactionRecord
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output_exists = await asyncio.to_thread(output.exists)
    if output_exists and not args.overwrite:
        print(f"output_exists={output} use --overwrite")
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    analyzer = EventAnalyzer()
    lines: list[str] = []
    async with session_factory() as session:
        query = select(NewsItemRecord).order_by(NewsItemRecord.published_at, NewsItemRecord.id)
        if args.date_from is not None:
            query = query.where(NewsItemRecord.published_at >= _parse_date(args.date_from))
        if args.date_till is not None:
            query = query.where(NewsItemRecord.published_at < _parse_date(args.date_till))
        if args.ticker is not None:
            query = (
                query.join(
                    NewsInstrumentMatchRecord,
                    NewsInstrumentMatchRecord.news_id == NewsItemRecord.id,
                )
                .join(
                    InstrumentRecord, InstrumentRecord.id == NewsInstrumentMatchRecord.instrument_id
                )
                .where(InstrumentRecord.ticker == args.ticker.upper())
            )
        if args.only_with_reactions:
            query = query.where(
                exists().where(NewsMarketReactionRecord.news_id == NewsItemRecord.id)
            )
        query = query.limit(args.limit).offset(args.offset)
        result = await session.execute(query)
        for news in result.scalars().unique():
            analysis = analyzer.analyze(news_id=news.id, raw_content=news.raw_content)
            if args.event_type is not None:
                selected = EventType(args.event_type)
                if selected not in {event.event_type for event in analysis.events}:
                    continue
            example = AnnotationExample(
                schema_version=GOLD_SCHEMA_VERSION,
                news_id=news.id,
                published_at=news.published_at,
                raw_content_hash=news.raw_content_hash,
                split=DatasetSplit.UNASSIGNED,
                review_status=ReviewStatus.DRAFT,
                annotator=args.annotator,
                notes=None,
                predicted_events=[_event_json(event) for event in analysis.events],
                predicted_financial_facts=[_fact_json(fact) for fact in analysis.financial_facts],
                gold_events=[],
                gold_financial_facts=[],
                raw_content=news.raw_content if args.include_raw_content else None,
            )
            lines.append(
                json.dumps(annotation_to_json(example), ensure_ascii=False, sort_keys=True)
            )
    await engine.dispose()
    await asyncio.to_thread(_write_text, output, "".join(f"{line}\n" for line in lines))
    print(f"rows_written={len(lines)} output={output}")
    return 0


def _event_json(event: DetectedEvent) -> dict[str, object]:
    return {
        "event_type": event.event_type.value,
        "confidence": str(event.confidence),
        "rule_id": event.rule_id,
        "matched_rule": event.matched_rule,
        "evidence_text": event.evidence_text,
        "start_position": event.start_position,
        "end_position": event.end_position,
    }


def _fact_json(fact: ExtractedFinancialFact) -> dict[str, object]:
    return {
        "metric": fact.metric.value,
        "raw_value": str(fact.raw_value),
        "normalized_value": str(fact.normalized_value),
        "unit": fact.unit.value,
        "currency": fact.currency.value,
        "scale": fact.scale.value,
        "period_type": fact.period_type.value,
        "period_year": fact.year,
        "period_quarter": fact.quarter,
        "period_month": fact.month,
        "raw_period": fact.raw_period,
        "comparison_type": fact.comparison_type.value,
        "fact_role": fact.fact_role.value,
        "change_direction": fact.change_direction.value,
        "change_value": None if fact.change_value is None else str(fact.change_value),
        "change_unit": None if fact.change_unit is None else fact.change_unit.value,
        "confidence": str(fact.confidence),
        "rule_id": fact.rule_id,
        "evidence_text": fact.evidence_text,
        "start_position": fact.start_position,
        "end_position": fact.end_position,
        "extractor_version": fact.extractor_version,
        "matched_rule": fact.matched_rule,
    }


def _parse_date(value: str) -> datetime:
    parsed = date.fromisoformat(value)
    return datetime(parsed.year, parsed.month, parsed.day)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an event-gold-v1 annotation batch.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--date-from")
    parser.add_argument("--date-till")
    parser.add_argument("--ticker")
    parser.add_argument("--event-type", choices=[item.value for item in EventType])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--annotator", default="manual")
    parser.add_argument("--include-raw-content", action="store_true")
    parser.add_argument("--only-with-reactions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
