from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.free_live_issuer_accumulation.application import audit_live_issuer_sources
from src.free_live_issuer_accumulation.domain import DEFAULT_SOURCE_REGISTRY_PATH


def run(args: argparse.Namespace) -> int:
    manifest = audit_live_issuer_sources(
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        registry_path=Path(args.source_registry),
        created_at=datetime.fromisoformat(args.created_at) if args.created_at else None,
    )
    print(json.dumps(_summary(manifest), ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit free official live issuer sources.")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--source-registry", default=DEFAULT_SOURCE_REGISTRY_PATH)
    parser.add_argument("--output-dir", default="artifacts/free-live-issuer-accumulation-v1-audit")
    parser.add_argument("--created-at", default=None)
    return parser


def _summary(manifest: dict[str, object]) -> dict[str, object]:
    keys = (
        "ARTIFACT_SHA",
        "STRICT_ANSWER",
        "FREE_OFFICIAL_SOURCES_AUDITED",
        "PAID_SOURCES_USED",
        "LIVE_STRICT_EXACT_READY_SOURCES",
        "UNIQUE_ISSUER_TICKERS_COVERED",
        "NEW_TICKERS_RELATIVE_TO_HISTORICAL_7",
        "SOURCES_WITH_EXPLICIT_TIMEZONE",
        "SOURCES_REJECTED_FOR_TIMEZONE",
        "LIVE_DIVERSITY_STATUS",
        "FREE_BLOCKER",
    )
    return {key: manifest[key] for key in keys}


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
