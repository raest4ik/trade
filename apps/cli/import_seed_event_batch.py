from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.evaluation.application.seed_curation import (
    SEED_SOURCE_NAME,
    SeedProcessingResult,
    dry_run_counts,
    process_seed_batch,
    validate_seed_file,
)
from src.events.infrastructure.repositories import SqlAlchemyEventAnalysisRepository
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    validation = await asyncio.to_thread(validate_seed_file, input_path)
    if not validation.ok:
        for error in validation.errors:
            print(f"error={error}")
        print(
            f"records_total={len(validation.records)} would_create=0 already_exists=0 "
            f"invalid={len(validation.errors)}"
        )
        return 1
    settings = get_settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        news_repository = SqlAlchemyNewsRepository(session)
        instrument_repository = SqlAlchemyInstrumentRepository(session)
        event_repository = SqlAlchemyEventAnalysisRepository(session)
        if args.dry_run:
            existing: set[str] = set()
            for record in validation.records:
                if (
                    await news_repository.get_by_source(SEED_SOURCE_NAME, record.record_id)
                    is not None
                ):
                    existing.add(record.record_id)
            counts = dry_run_counts(validation.records, existing)
            print(
                f"records_total={counts['records_total']} "
                f"would_create={counts['would_create']} "
                f"already_exists={counts['already_exists']} invalid={counts['invalid']}"
            )
            await engine.dispose()
            return 0
        result = await process_seed_batch(
            records=validation.records,
            news_repository=news_repository,
            instrument_repository=instrument_repository,
            event_repository=event_repository,
            output_dir=output_dir,
            dry_run=False,
        )
    await engine.dispose()
    print(json.dumps(_result_payload(result), ensure_ascii=False, sort_keys=True))
    return 0


def _result_payload(result: SeedProcessingResult) -> dict[str, object]:
    stats = result.stats
    return {
        "records_total": stats.records_total,
        "created": stats.created,
        "already_exists": stats.already_exists,
        "invalid": stats.invalid,
        "instrument_matches_total": stats.instrument_matches_total,
        "records_with_instrument_matches": stats.records_with_instrument_matches,
        "ambiguous_instrument_matches": stats.ambiguous_instrument_matches,
        "event_status_counts": stats.event_status_counts,
        "primary_event_counts": stats.primary_event_counts,
        "predicted_fact_count": stats.predicted_fact_count,
        "records_with_predicted_facts": stats.records_with_predicted_facts,
        "category_counts": stats.category_counts,
        "source_review_required": stats.source_review_required,
        "review_jsonl_path": str(result.review_jsonl_path),
        "mapping_path": str(result.mapping_path),
        "comparison_dir": str(result.comparison_dir),
        "review_queue_path": str(result.review_queue_path),
        "baseline_metrics": stats.baseline_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import event-seed-v1 records as research NewsItem rows and build review artifacts."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="artifacts/seed")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
