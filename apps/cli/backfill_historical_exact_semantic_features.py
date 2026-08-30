from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.historical_exact_semantic_backfill.application import (
    run_historical_exact_semantic_backfill,
)
from src.historical_exact_semantic_backfill.domain import (
    ARTIFACT_VERSION,
    DEFAULT_DIAGNOSIS_ARTIFACT_ROOT,
    DEFAULT_MARKET_ARTIFACT_ROOT,
    DEFAULT_SNAPSHOT_ROOTS,
)


def run(args: argparse.Namespace) -> int:
    manifest = run_historical_exact_semantic_backfill(
        diagnosis_root=Path(args.diagnosis_dir),
        market_root=Path(args.market_dir),
        snapshot_roots=tuple(Path(root) for root in args.snapshot_roots),
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        created_at=(
            datetime.fromisoformat(args.created_at) if args.created_at is not None else None
        ),
    )
    print(
        json.dumps(
            {
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "TARGET_EVENTS": manifest["TARGET_EVENTS"],
                "SNAPSHOT_MATCHED_EXACT": manifest["SNAPSHOT_MATCHED_EXACT"],
                "SNAPSHOT_IDENTITY_UNRESOLVED": manifest["SNAPSHOT_IDENTITY_UNRESOLVED"],
                "PUBLICATION_MATERIAL_AVAILABLE": manifest["PUBLICATION_MATERIAL_AVAILABLE"],
                "SEMANTIC_EXTRACTION_SUCCEEDED": manifest["SEMANTIC_EXTRACTION_SUCCEEDED"],
                "ANALYZER_PRODUCED_UNKNOWN": manifest["ANALYZER_PRODUCED_UNKNOWN"],
                "FEATURE_READY_RECOVERED": manifest["FEATURE_READY_RECOVERED"],
                "FEATURE_READY_STILL_BLOCKED": manifest["FEATURE_READY_STILL_BLOCKED"],
                "FEATURE_READY_BEFORE": manifest["FEATURE_READY_BEFORE"],
                "FEATURE_READY_AFTER": manifest["FEATURE_READY_AFTER"],
                "REACTION_ROWS_CHANGED": manifest["REACTION_ROWS_CHANGED"],
                "NETWORK_MARKET_FETCHES": manifest["NETWORK_MARKET_FETCHES"],
                "DECISION": manifest["DECISION"],
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill semantic features for historical strict-EXACT reaction-ready rows."
    )
    parser.add_argument("--diagnosis-dir", default=DEFAULT_DIAGNOSIS_ARTIFACT_ROOT)
    parser.add_argument("--market-dir", default=DEFAULT_MARKET_ARTIFACT_ROOT)
    parser.add_argument("--snapshot-roots", nargs="+", default=list(DEFAULT_SNAPSHOT_ROOTS))
    parser.add_argument("--output-dir", default=f"artifacts/{ARTIFACT_VERSION}")
    parser.add_argument("--base-main-sha", required=True)
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
