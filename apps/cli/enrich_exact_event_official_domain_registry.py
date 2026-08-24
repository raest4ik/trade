from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_event_official_domain_registry.application import (
    build_official_domain_registry_artifact,
)


def run(args: argparse.Namespace) -> int:
    manifest = build_official_domain_registry_artifact(
        input_root=Path(args.input_dir),
        source_registry_path=Path(args.source_registry),
        universe_path=Path(args.universe),
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        live_source_report_path=(
            Path(args.live_source_report) if args.live_source_report is not None else None
        ),
        candidate_domains_path=(
            Path(args.candidate_domains) if args.candidate_domains is not None else None
        ),
        created_at=(
            datetime.fromisoformat(args.created_at) if args.created_at is not None else None
        ),
    )
    print(
        json.dumps(
            {
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "INPUT_DATASET_SHA": manifest["INPUT_DATASET_SHA"],
                "LIVE_DOMAIN_ENRICHMENT_EXECUTED": manifest["LIVE_DOMAIN_ENRICHMENT_EXECUTED"],
                "LIVE_DOMAIN_ENRICHMENT_BLOCKER": manifest["LIVE_DOMAIN_ENRICHMENT_BLOCKER"],
                "DOMAIN_TICKERS_TARGETED": manifest["DOMAIN_TICKERS_TARGETED"],
                "DOMAIN_CONFIRMED_COUNT": manifest["DOMAIN_CONFIRMED_COUNT"],
                "NEWLY_DOMAIN_ENABLED_TICKERS": manifest["NEWLY_DOMAIN_ENABLED_TICKERS"],
                "SECOND_LIVE_RUN_EXECUTED": manifest["SECOND_LIVE_RUN_EXECUTED"],
                "DOWNSTREAM_NEW_EXACT_CAPABLE_SOURCES": manifest[
                    "DOWNSTREAM_NEW_EXACT_CAPABLE_SOURCES"
                ],
                "DOWNSTREAM_NEW_EXACT_EVENTS": manifest["DOWNSTREAM_NEW_EXACT_EVENTS"],
                "LIVE_SOURCE_DISCOVERY_CONCLUSION": manifest["LIVE_SOURCE_DISCOVERY_CONCLUSION"],
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich exact-event official domain registry v1.")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument(
        "--input-dir",
        default="artifacts/exact-event-official-source-discovery-v5",
    )
    parser.add_argument(
        "--source-registry",
        default="artifacts/exact-event-official-source-discovery-v5/source-registry.jsonl",
    )
    parser.add_argument(
        "--universe",
        default="artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json",
    )
    parser.add_argument(
        "--live-source-report",
        default="artifacts/exact-event-live-official-source-snapshot-v1/source-report.jsonl",
    )
    parser.add_argument("--candidate-domains", default=None)
    parser.add_argument(
        "--output-dir",
        default="artifacts/exact-event-official-domain-registry-enrichment-v1",
    )
    parser.add_argument("--created-at", default=None)
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
