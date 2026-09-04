from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.free_live_operational_burnin_and_onboarding_v3.application import (
    ARTIFACT_VERSION,
    run_free_live_operational_burnin_and_onboarding_v3,
)


def run(args: argparse.Namespace) -> int:
    created_at = _created_at(args.created_at) if args.created_at else None
    manifest = run_free_live_operational_burnin_and_onboarding_v3(
        output_root=Path(args.output_root),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        operation_root=Path(args.operation_root),
        source_registry_path=Path(args.source_registry),
        historical_ticker_summary_path=Path(args.historical_ticker_summary),
        instrument_mapping_path=Path(args.instrument_mapping),
        network_check=not args.no_network,
        created_at=created_at,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 1 if manifest["OPERATIONAL_BURN_IN"] == "FAIL" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=ARTIFACT_VERSION)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument(
        "--operation-root",
        default="artifacts/free-live-research-operation-v1",
    )
    parser.add_argument("--source-registry", default="config/live_issuer_sources_v1.json")
    parser.add_argument(
        "--historical-ticker-summary",
        default="artifacts/ml-v2-readiness-audit-v1/ticker-summary.jsonl",
    )
    parser.add_argument(
        "--instrument-mapping",
        default="artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json",
    )
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--created-at", default=None, help=argparse.SUPPRESS)
    return parser


def _created_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("created-at must include timezone")
    return parsed.astimezone(UTC)


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
