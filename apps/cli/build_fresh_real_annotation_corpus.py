from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from apps.cli.historical_news_common import parse_range_datetime
from src.fresh_real_corpus.application import (
    load_bounded_records,
    refresh_instrument_matches,
)
from src.fresh_real_corpus.domain import (
    APPROVED_SOURCE_TICKERS,
    SelectionPolicy,
    freeze_temporal_split,
    load_exclusion_index,
    select_fresh_records,
)
from src.fresh_real_corpus.reporting import write_fresh_corpus_artifacts
from src.official_sources.registry import official_source_configs
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory

DEFAULT_EXCLUSIONS = [
    "artifacts/seed/batch-001-gold-v1-reviewed-only.jsonl",
    "artifacts/corpus-quality-v1/annotation-batch-002.jsonl",
    "artifacts/corpus-quality-v1/annotation-batch-003.jsonl",
    "artifacts/corpus-quality-v1/batch-003-human-review-v1.jsonl",
    "artifacts/real-gold-benchmark-v2/gold/dataset.jsonl",
]


async def run(args: argparse.Namespace) -> int:
    policy = SelectionPolicy(
        source_codes=tuple(args.source),
        date_from=parse_range_datetime(args.date_from, end_of_day=False),
        date_to=parse_range_datetime(args.date_to, end_of_day=True),
        limit=args.limit,
    ).normalized()
    engine = create_engine(get_settings().database_url)
    try:
        async with create_session_factory(engine)() as session:
            matched_candidates = await refresh_instrument_matches(session, policy=policy)
            available = await load_bounded_records(session, policy=policy)
    finally:
        await engine.dispose()
    exclusion_paths = tuple(Path(value) for value in args.exclude)
    missing = [str(path) for path in exclusion_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required prior-batch exclusions are missing: {missing}")
    exclusions = load_exclusion_index(exclusion_paths)
    selected = select_fresh_records(available, policy=policy, exclusions=exclusions)
    split = freeze_temporal_split(selected.records)
    paths = write_fresh_corpus_artifacts(
        Path(args.output),
        annotation_copy=Path(args.annotation_output),
        result=selected,
        split=split,
        policy=policy,
        source_configs=official_source_configs(),
        batch_001_path=Path(args.batch_001_gold),
        git_sha=args.git_sha or _git_sha(),
    )
    print(
        json.dumps(
            {
                "available": len(available),
                "matched_candidates_refreshed": matched_candidates,
                "excluded_previous_batch_overlaps": selected.excluded_overlap_count,
                "selected": len(selected.records),
                "development": sum(
                    split.split_for(item.news_id).value == "DEVELOPMENT"
                    for item in selected.records
                ),
                "fresh_holdout": sum(
                    split.split_for(item.news_id).value == "FRESH_HOLDOUT"
                    for item in selected.records
                ),
                "split_sha256": split.split_sha256,
                "outputs": {key: str(value) for key, value in paths.items()},
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a fresh REAL EXACT development/holdout annotation corpus."
    )
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--limit", type=int, choices=range(2, 101), required=True)
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(APPROVED_SOURCE_TICKERS),
        required=True,
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=DEFAULT_EXCLUSIONS,
        help="Prior annotation/gold JSONL whose news and source identities must be excluded.",
    )
    parser.add_argument(
        "--batch-001-gold",
        default="artifacts/seed/batch-001-gold-v1-reviewed-only.jsonl",
    )
    parser.add_argument("--output", default="artifacts/fresh-real-corpus-v1")
    parser.add_argument(
        "--annotation-output",
        default="artifacts/corpus-quality-v1/annotation-batch-004.jsonl",
    )
    parser.add_argument("--git-sha")
    return parser


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
