from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import UTC, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.events.domain.entities import EVENT_ANALYSIS_VERSION, FINANCIAL_FACTS_VERSION
from src.events.infrastructure.repositories import SqlAlchemyEventAnalysisRepository
from src.ml_features.application.feature_builder import BuildMlFeatureDataset
from src.ml_features.domain.entities import (
    DATASET_VERSION,
    FEATURE_VERSION,
    LABEL_HORIZONS_MINUTES,
    MARKET_CONTEXT_VERSION,
    FeatureDatasetConfig,
    FeatureExclusion,
)
from src.ml_features.infrastructure.export import write_dataset_artifacts
from src.ml_features.infrastructure.repositories import SqlAlchemyMlFeatureRepository
from src.reactions.domain.entities import REACTION_VERSION
from src.reactions.infrastructure.repositories import SqlAlchemyReactionRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    config = FeatureDatasetConfig(
        date_from=_range_datetime(args.date_from, end_of_day=False),
        date_to=_range_datetime(args.date_to, end_of_day=True),
        tickers=_tickers(args.tickers),
        limit=args.limit,
        require_label_horizon=args.require_label_horizon,
        classification_threshold=args.classification_threshold,
        event_analysis_version=args.event_analysis_version,
        fact_extractor_version=args.fact_extractor_version,
        reaction_version=args.reaction_version,
    ).normalized()
    engine = create_engine(get_settings().database_url)
    try:
        async with create_session_factory(engine)() as session:
            result = await BuildMlFeatureDataset(
                repository=SqlAlchemyMlFeatureRepository(session),
                event_repository=SqlAlchemyEventAnalysisRepository(session),
                reaction_repository=SqlAlchemyReactionRepository(session),
            ).execute(
                config=config,
                git_sha=args.git_sha or _git_sha(),
                dry_run=args.dry_run,
            )
    finally:
        await engine.dispose()
    paths: dict[str, Path] = {}
    if not args.dry_run:
        paths = write_dataset_artifacts(Path(args.output), result=result, config=config)
    summary = {
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "market_context_version": MARKET_CONTEXT_VERSION,
        "run_id": None if args.dry_run else str(result.run.id),
        "status": result.run.status.value,
        "candidate_count": result.run.candidate_count,
        "eligible_count": result.run.eligible_count,
        "built_count": result.run.built_count,
        "excluded_count": result.run.excluded_count,
        "failed_count": result.run.failed_count,
        "config_hash": result.run.config_hash,
        "exclusions_by_reason": _exclusion_counts(result.exclusions),
        "dry_run": args.dry_run,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe ml-feature-dataset-v1 artifacts."
    )
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--tickers", help="Comma-separated MOEX tickers.")
    parser.add_argument("--limit", type=_bounded_limit, default=10_000)
    parser.add_argument(
        "--require-label-horizon",
        type=int,
        choices=LABEL_HORIZONS_MINUTES,
    )
    parser.add_argument("--classification-threshold", type=_nonnegative_decimal)
    parser.add_argument("--output", default="artifacts/ml-feature-dataset-v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--git-sha")
    parser.add_argument("--event-analysis-version", default=EVENT_ANALYSIS_VERSION)
    parser.add_argument("--fact-extractor-version", default=FINANCIAL_FACTS_VERSION)
    parser.add_argument("--reaction-version", default=REACTION_VERSION)
    return parser


def _range_datetime(value: str, *, end_of_day: bool) -> datetime:
    normalized = value.strip()
    try:
        if len(normalized) == 10:
            parsed_date = datetime.fromisoformat(normalized).date()
            return datetime.combine(
                parsed_date,
                time.max if end_of_day else time.min,
                tzinfo=UTC,
            )
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("datetime must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("datetime range must include timezone")
    return parsed.astimezone(UTC)


def _tickers(value: str | None) -> tuple[str, ...]:
    return () if value is None else tuple(item for item in value.split(",") if item.strip())


def _bounded_limit(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100_000:
        raise argparse.ArgumentTypeError("limit must be between 1 and 100000")
    return parsed


def _nonnegative_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("threshold must be a decimal") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("threshold must not be negative")
    return parsed


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _exclusion_counts(exclusions: list[FeatureExclusion]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in exclusions:
        counts[item.reason.value] = counts.get(item.reason.value, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
