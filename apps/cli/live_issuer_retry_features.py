from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.free_live_issuer_accumulation.operation import (
    DEFAULT_OPERATION_ARTIFACT_ROOT,
    FreeLiveResearchOperation,
    OperationConfig,
)


def run(args: argparse.Namespace) -> int:
    published_at = _datetime(args.published_at) if args.published_at else None
    payload = FreeLiveResearchOperation(
        OperationConfig(
            artifact_root=Path(args.artifact_root),
            registry_path=Path(args.source_registry),
            historical_ticker_summary_path=Path(args.historical_ticker_summary),
        )
    ).retry_features(published_at=published_at)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retry bounded pre-event feature blockers.")
    parser.add_argument("--artifact-root", default=str(DEFAULT_OPERATION_ARTIFACT_ROOT))
    parser.add_argument("--source-registry", default="config/live_issuer_sources_v1.json")
    parser.add_argument(
        "--historical-ticker-summary",
        default="artifacts/ml-v2-readiness-audit-v1/ticker-summary.jsonl",
    )
    parser.add_argument("--published-at", default=None)
    return parser


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("published-at must include timezone")
    return parsed.astimezone(UTC)


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
