from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from uuid import UUID

from apps.cli.historical_news_common import parse_range_datetime
from src.corpus_quality.application import load_publication_time_records
from src.events.domain.entities import EVENT_ANALYSIS_VERSION, FINANCIAL_FACTS_VERSION
from src.events.infrastructure.repositories import SqlAlchemyEventAnalysisRepository
from src.ml_features.application.feature_builder import BuildMlFeatureDataset
from src.ml_features.domain.entities import FeatureDatasetConfig
from src.ml_features.infrastructure.export import write_dataset_artifacts
from src.ml_features.infrastructure.repositories import SqlAlchemyMlFeatureRepository
from src.official_sources.registry import official_source_configs, reaction_ready_configs
from src.official_sources.reporting import write_official_source_corpus
from src.reaction_ready_corpus.application import PrepareCorpusCommand, PrepareReactionReadyCorpus
from src.reaction_ready_corpus.reporting import batch_001_reaction_count
from src.reactions.domain.entities import REACTION_VERSION
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    date_from = parse_range_datetime(args.date_from, end_of_day=False)
    date_to = parse_range_datetime(args.date_to, end_of_day=True)
    approved = reaction_ready_configs()
    source_codes = tuple(item.source_code for item in approved)
    tickers = tuple(ticker for item in approved for ticker in item.tickers)
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
            ).execute(config=config, git_sha=args.git_sha or _git_sha(), dry_run=args.dry_run)
            feature_ids = {UUID(str(row.metadata["news_id"])) for row in feature_result.rows}
            records = await load_publication_time_records(
                session,
                date_from=date_from,
                date_to=date_to,
                feature_news_ids=feature_ids,
                limit=args.limit,
            )
            batch_reactions = await batch_001_reaction_count(session)
    finally:
        await engine.dispose()
    paths: dict[str, Path] = {}
    if not args.dry_run:
        output = Path(args.output)
        paths = write_official_source_corpus(
            output,
            records=records,
            feature_rows=feature_result.rows,
            source_configs=official_source_configs(),
            git_sha=args.git_sha or _git_sha(),
            batch_001_reactions=batch_reactions,
            batch_002_path=Path(args.batch_002),
            annotation_output=Path(args.annotation_output),
        )
        ml_paths = write_dataset_artifacts(
            output / "ml-feature-dataset-v1",
            result=feature_result,
            config=config,
        )
        paths.update({f"ml_{name}": path for name, path in ml_paths.items()})
    print(
        json.dumps(
            {
                "prepared": {
                    "candidates": prepared.candidate_count,
                    "analyzed": prepared.analyzed_count,
                    "matched": prepared.matched_count,
                    "ambiguous": prepared.ambiguous_count,
                    "unmatched": prepared.unmatched_count,
                    "windows": [item.payload() for item in prepared.windows],
                },
                "feature_rows": len(feature_result.rows),
                "records": len(records),
                "outputs": {name: str(path) for name, path in paths.items()},
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the cumulative official-source reaction-ready corpus v3."
    )
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--git-sha")
    parser.add_argument("--output", default="artifacts/reaction-ready-corpus-v3")
    parser.add_argument(
        "--annotation-output",
        default="artifacts/corpus-quality-v1/annotation-batch-003.jsonl",
    )
    parser.add_argument(
        "--batch-002",
        default="artifacts/corpus-quality-v1/annotation-batch-002.jsonl",
    )
    return parser


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
