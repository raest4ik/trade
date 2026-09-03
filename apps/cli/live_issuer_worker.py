from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from src.free_live_issuer_accumulation.operation import (
    DEFAULT_OPERATION_ARTIFACT_ROOT,
    FreeLiveResearchOperation,
    OperationConfig,
)


def run(args: argparse.Namespace) -> int:
    config = OperationConfig(
        artifact_root=Path(args.artifact_root),
        registry_path=Path(args.source_registry),
        historical_ticker_summary_path=Path(args.historical_ticker_summary),
        default_interval_minutes=args.poll_interval_minutes,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        failure_threshold=args.failure_threshold,
        cooldown_minutes=args.cooldown_minutes,
    )
    report = FreeLiveResearchOperation(config).run_worker(
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        max_iterations=args.max_iterations,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["LIVE_RESEARCH_OPERATION_STATUS"] in {"READY", "DEGRADED"} else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the free live issuer research worker.")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--artifact-root", default=str(DEFAULT_OPERATION_ARTIFACT_ROOT))
    parser.add_argument("--source-registry", default="config/live_issuer_sources_v1.json")
    parser.add_argument(
        "--historical-ticker-summary",
        default="artifacts/ml-v2-readiness-audit-v1/ticker-summary.jsonl",
    )
    parser.add_argument("--poll-interval-minutes", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--failure-threshold", type=int, default=3)
    parser.add_argument("--cooldown-minutes", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=None)
    return parser


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
