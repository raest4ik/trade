from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.free_live_issuer_accumulation.operation import (
    DEFAULT_OPERATION_ARTIFACT_ROOT,
    build_operation_status,
)


def run(args: argparse.Namespace) -> int:
    payload = build_operation_status(
        Path(args.artifact_root),
        registry_path=Path(args.source_registry),
        historical_ticker_summary_path=Path(args.historical_ticker_summary),
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show free live issuer research operation status.")
    parser.add_argument("--artifact-root", default=str(DEFAULT_OPERATION_ARTIFACT_ROOT))
    parser.add_argument("--source-registry", default="config/live_issuer_sources_v1.json")
    parser.add_argument(
        "--historical-ticker-summary",
        default="artifacts/ml-v2-readiness-audit-v1/ticker-summary.jsonl",
    )
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
