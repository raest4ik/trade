from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from uuid import UUID

from apps.cli.historical_news_common import bounded_limit, parse_range_datetime
from src.events.domain.entities import EVENT_ANALYSIS_VERSION, FINANCIAL_FACTS_VERSION
from src.events.infrastructure.repositories import SqlAlchemyEventAnalysisRepository
from src.ml_features.application.feature_builder import BuildMlFeatureDataset
from src.ml_features.domain.entities import FeatureDatasetConfig
from src.ml_features.infrastructure.export import write_dataset_artifacts
from src.ml_features.infrastructure.repositories import SqlAlchemyMlFeatureRepository
from src.reaction_ready_corpus.application import (
    PrepareCorpusCommand,
    PrepareReactionReadyCorpus,
)
from src.reaction_ready_corpus.domain import REAL_SOURCE_CODES, UNIVERSE
from src.reaction_ready_corpus.reporting import (
    batch_001_reaction_count,
    build_and_write_reports,
    load_acquisition_run,
    load_candidate_snapshots,
)
from src.reaction_ready_corpus.source_audit import write_source_audit
from src.reactions.domain.entities import REACTION_VERSION
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    date_from = parse_range_datetime(args.date_from, end_of_day=False)
    date_to = parse_range_datetime(args.date_to, end_of_day=True)
    source_codes = _csv(args.source_codes)
    tickers = _csv(args.tickers)
    git_sha = args.git_sha or _git_sha()
    config = FeatureDatasetConfig(
        date_from=date_from,
        date_to=date_to,
        tickers=tickers,
        limit=args.limit,
        event_analysis_version=EVENT_ANALYSIS_VERSION,
        fact_extractor_version=FINANCIAL_FACTS_VERSION,
        reaction_version=REACTION_VERSION,
    ).normalized()
    engine = create_engine(get_settings().database_url)
    try:
        async with create_session_factory(engine)() as session:
            prepared = await PrepareReactionReadyCorpus(session).execute(
                PrepareCorpusCommand(
                    date_from=date_from,
                    date_to=date_to,
                    source_codes=source_codes,
                    tickers=tickers,
                    limit=args.limit,
                    dry_run=args.dry_run,
                )
            )
            feature_result = await BuildMlFeatureDataset(
                repository=SqlAlchemyMlFeatureRepository(session),
                event_repository=SqlAlchemyEventAnalysisRepository(session),
                reaction_repository=SqlAlchemyReactionRepository(session),
            ).execute(config=config, git_sha=git_sha, dry_run=args.dry_run)
            if args.dry_run:
                paths: dict[str, Path] = {}
            else:
                if args.ingestion_run_id is None:
                    raise ValueError("--ingestion-run-id is required unless --dry-run is used")
                acquisition = await load_acquisition_run(session, UUID(args.ingestion_run_id))
                snapshots = await load_candidate_snapshots(
                    session,
                    date_from=date_from,
                    date_to=date_to,
                    limit=args.limit,
                )
                batch_reactions = await batch_001_reaction_count(session)
                output_dir = Path(args.output)
                paths = build_and_write_reports(
                    output_dir,
                    snapshots=snapshots,
                    feature_result=feature_result,
                    acquisition=acquisition,
                    date_from=date_from,
                    date_to=date_to,
                    git_sha=git_sha,
                    batch_reactions=batch_reactions,
                    selected_source_codes=source_codes,
                )
                ml_paths = write_dataset_artifacts(
                    output_dir / "ml-feature-dataset-v1",
                    result=feature_result,
                    config=config,
                )
                paths.update({f"ml_{name}": path for name, path in ml_paths.items()})
                write_source_audit(Path(args.source_audit_output))
    finally:
        await engine.dispose()
    print(
        json.dumps(
            {
                "prepared_candidates": prepared.candidate_count,
                "analyzed": prepared.analyzed_count,
                "matched": prepared.matched_count,
                "ambiguous": prepared.ambiguous_count,
                "unmatched": prepared.unmatched_count,
                "market_backfill_windows": [window.payload() for window in prepared.windows],
                "feature_rows": len(feature_result.rows),
                "feature_exclusions": len(feature_result.exclusions),
                "dry_run": args.dry_run,
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build canonical REAL reaction-ready corpus artifacts from existing labels."
    )
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--limit", type=bounded_limit, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-codes", default=",".join(sorted(REAL_SOURCE_CODES)))
    parser.add_argument("--tickers", default=",".join(UNIVERSE))
    parser.add_argument("--ingestion-run-id")
    parser.add_argument("--git-sha")
    parser.add_argument("--output", default="artifacts/reaction-ready-corpus-v1")
    parser.add_argument(
        "--source-audit-output",
        default="artifacts/historical-news-v1/source-audit.json",
    )
    return parser


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip().upper() for item in value.split(",") if item.strip())


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
