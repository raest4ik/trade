from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from src.issuer_historical_diversity_recovery.application import (
    run_historical_issuer_diversity_recovery_audit,
)
from src.issuer_historical_diversity_recovery.domain import (
    ARTIFACT_VERSION,
    DEFAULT_BACKFILL_ROOT,
    DEFAULT_CHEP_MATURATION_ROOT,
    DEFAULT_CONSOLIDATED_MATURATION_ROOT,
    DEFAULT_ISSUER_DIVERSITY_ROOT,
    DEFAULT_ML_V2_READINESS_ROOT,
    DEFAULT_TZ_DISCOVERY_ROOT,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit safe historical issuer diversity recovery paths."
    )
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--output-dir", default=f"artifacts/{ARTIFACT_VERSION}")
    parser.add_argument("--backfill-dir", default=DEFAULT_BACKFILL_ROOT)
    parser.add_argument("--readiness-dir", default=DEFAULT_ML_V2_READINESS_ROOT)
    parser.add_argument("--tz-discovery-dir", default=DEFAULT_TZ_DISCOVERY_ROOT)
    parser.add_argument("--issuer-diversity-dir", default=DEFAULT_ISSUER_DIVERSITY_ROOT)
    parser.add_argument("--consolidated-dir", default=DEFAULT_CONSOLIDATED_MATURATION_ROOT)
    parser.add_argument("--chep-dir", default=DEFAULT_CHEP_MATURATION_ROOT)
    parser.add_argument("--created-at")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    created_at = datetime.fromisoformat(args.created_at) if args.created_at else None
    manifest = run_historical_issuer_diversity_recovery_audit(
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        backfill_root=Path(args.backfill_dir),
        readiness_root=Path(args.readiness_dir),
        tz_discovery_root=Path(args.tz_discovery_dir),
        issuer_diversity_root=Path(args.issuer_diversity_dir),
        consolidated_root=Path(args.consolidated_dir),
        chep_root=Path(args.chep_dir),
        created_at=created_at,
        env_names=sorted(os.environ),
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "STRICT_ANSWER": manifest["STRICT_ANSWER"],
                "FINAL_DECISION": manifest["FINAL_DECISION"],
                "CURRENT_ISSUER_ROWS": manifest["CURRENT_ISSUER_ROWS"],
                "CURRENT_ISSUER_TICKERS": manifest["CURRENT_ISSUER_TICKERS"],
                "DOMINANT_TICKER": manifest["DOMINANT_TICKER"],
                "ROWS_REQUIRED_TOP1_LE_50": manifest["ROWS_REQUIRED_TOP1_LE_50"],
                "NON_UNKNOWN_ROWS_REQUIRED_UNKNOWN_LE_50": manifest[
                    "NON_UNKNOWN_ROWS_REQUIRED_UNKNOWN_LE_50"
                ],
                "EXHAUSTED_HISTORICAL_SOURCE_CANDIDATES": manifest[
                    "EXHAUSTED_HISTORICAL_SOURCE_CANDIDATES"
                ],
                "NEW_MECHANISMS_EVALUATED": manifest["NEW_MECHANISMS_EVALUATED"],
                "PAID_AUTHENTICATED_VIABLE_SOURCES_FOUND": manifest[
                    "PAID_AUTHENTICATED_VIABLE_SOURCES_FOUND"
                ],
                "FUTURE_OUTCOMES_READ": manifest["FUTURE_OUTCOMES_READ"],
                "FUTURE_TARGETS_READ": manifest["FUTURE_TARGETS_READ"],
                "FUTURE_PRICE_LOOKUPS": manifest["FUTURE_PRICE_LOOKUPS"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    main()
