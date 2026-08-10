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

from src.events.infrastructure.models import NewsEventAnalysisRecord
from src.instruments.infrastructure.models import InstrumentRecord
from src.news.infrastructure.models import NewsItemRecord
from src.reactions.infrastructure.models import (
    NewsMarketReactionRecord,
    ReactionPointRecord,
)
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory

DATASET_SCHEMA_VERSION = "market-reaction-dataset-v2"


async def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    lines: list[str] = []
    try:
        async with session_factory() as session:
            analysis_version = (
                select(NewsEventAnalysisRecord.analysis_version)
                .where(NewsEventAnalysisRecord.news_id == NewsItemRecord.id)
                .order_by(NewsEventAnalysisRecord.analyzed_at.desc())
                .limit(1)
                .scalar_subquery()
            )
            statement = (
                select(
                    ReactionPointRecord,
                    NewsMarketReactionRecord,
                    NewsItemRecord,
                    InstrumentRecord,
                    analysis_version.label("analysis_version"),
                )
                .join(
                    NewsMarketReactionRecord,
                    NewsMarketReactionRecord.id == ReactionPointRecord.reaction_id,
                )
                .join(NewsItemRecord, NewsItemRecord.id == NewsMarketReactionRecord.news_id)
                .join(
                    InstrumentRecord,
                    InstrumentRecord.id == NewsMarketReactionRecord.instrument_id,
                )
                .options(selectinload(ReactionPointRecord.benchmark_adjustment))
                .order_by(
                    NewsItemRecord.published_at,
                    NewsMarketReactionRecord.instrument_id,
                    ReactionPointRecord.horizon_minutes,
                )
                .limit(args.limit)
            )
            result = await session.execute(statement)
            for point, reaction, news, instrument, version in result.all():
                payload = market_reaction_row_payload(
                    point=point,
                    reaction=reaction,
                    news=news,
                    instrument=instrument,
                    analysis_version=version,
                )
                lines.append(json.dumps(_json_value(payload), ensure_ascii=False, sort_keys=True))
    finally:
        await engine.dispose()
    await asyncio.to_thread(_write_lines, output, lines)
    print(f"schema_version={DATASET_SCHEMA_VERSION} rows_written={len(lines)} output={output}")
    return 0


def market_reaction_row_payload(
    *,
    point: ReactionPointRecord,
    reaction: NewsMarketReactionRecord,
    news: NewsItemRecord,
    instrument: InstrumentRecord,
    analysis_version: str | None,
) -> dict[str, object]:
    adjustment = point.benchmark_adjustment
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "news_id": news.id,
        "published_at": news.published_at,
        "timestamp_quality": news.publication_timestamp_quality,
        "instrument_id": instrument.id,
        "ticker": instrument.ticker,
        "event_analysis_version": analysis_version,
        "reaction_version": reaction.reaction_version,
        "horizon_minutes": point.horizon_minutes,
        "security": {
            "baseline_value": reaction.baseline_price,
            "baseline_observed_at": reaction.baseline_observed_at,
            "target_value": point.price,
            "target_observed_at": point.observed_at,
            "simple_return": point.simple_return,
            "log_return": point.log_return,
            "status": point.status,
        },
        "benchmark": None
        if adjustment is None
        else {
            "code": adjustment.benchmark_code,
            "baseline_value": adjustment.baseline_value,
            "baseline_observed_at": adjustment.baseline_observed_at,
            "target_value": adjustment.target_value,
            "target_observed_at": adjustment.target_observed_at,
            "simple_return": adjustment.simple_return,
            "log_return": adjustment.log_return,
            "status": adjustment.status,
            "missing_reason": adjustment.missing_reason,
        },
        "labels": {
            "abnormal_simple_return": None
            if adjustment is None
            else adjustment.abnormal_simple_return,
            "abnormal_log_return": None if adjustment is None else adjustment.abnormal_log_return,
        },
    }


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
    if isinstance(value, list):
        items = cast("list[object]", value)
        return [_json_value(item) for item in items]
    return value


def _write_lines(output: Path, lines: list[str]) -> None:
    output.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export raw benchmark-adjusted reaction labels.")
    parser.add_argument("--output", default="artifacts/market-reaction-dataset-v2.jsonl")
    parser.add_argument("--limit", type=_bounded_limit, default=10000)
    return parser


def _bounded_limit(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100000:
        raise argparse.ArgumentTypeError("limit must be between 1 and 100000")
    return parsed


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
