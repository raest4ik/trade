from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.live_corpus_operations.domain import DEFAULT_LOOKBACK_DAYS, LiveRunConfig, RunStatus
from src.live_corpus_operations.local_backend import LocalLiveCorpusBackend
from src.live_corpus_operations.runner import LiveCorpusRunner
from src.live_corpus_operations.state import OperationsStateStore


def run(args: argparse.Namespace) -> int:
    repo_root = Path(os.path.abspath(args.repo_root))
    artifact_root = Path(os.path.abspath(repo_root / args.artifact_root))
    return asyncio.run(_run(args, repo_root=repo_root, artifact_root=artifact_root))


async def _run(args: argparse.Namespace, *, repo_root: Path, artifact_root: Path) -> int:
    now = datetime.now(UTC)
    date_from = (
        _datetime(args.date_from) if args.date_from else now - timedelta(days=args.lookback_days)
    )
    date_to = _datetime(args.date_to) if args.date_to else now
    config = LiveRunConfig(
        date_from=date_from,
        date_to=date_to,
        limit=args.limit,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
    )
    report = await LiveCorpusRunner(
        backend=LocalLiveCorpusBackend(repo_root=repo_root),
        state_store=OperationsStateStore(artifact_root, log_retention=args.log_retention),
        lock_path=artifact_root / "live-corpus.lock",
    ).execute(config)
    print(json.dumps(report.payload(), ensure_ascii=False, sort_keys=True))
    return 1 if report.status == RunStatus.FAILED else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one bounded, zero-cost live corpus collection and maturation cycle."
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--artifact-root",
        default="artifacts/live-corpus-operations-v1",
    )
    parser.add_argument("--from", dest="date_from")
    parser.add_argument("--to", dest="date_to")
    parser.add_argument("--lookback-days", type=_lookback, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--limit", type=_limit, default=100)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, choices=range(0, 6), default=2)
    parser.add_argument("--log-retention", type=_positive, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("datetime must include timezone")
    return parsed.astimezone(UTC)


def _limit(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("limit must be between 1 and 100")
    return parsed


def _lookback(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 90:
        raise argparse.ArgumentTypeError("lookback-days must be between 1 and 90")
    return parsed


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
