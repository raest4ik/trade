from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from src.exact_feature_readiness_recovery.application import (
    run_exact_feature_readiness_recovery,
)
from src.exact_feature_readiness_recovery.domain import (
    ARTIFACT_VERSION,
    DEFAULT_INPUT_ARTIFACT_ROOT,
)


def run(args: argparse.Namespace) -> int:
    manifest = run_exact_feature_readiness_recovery(
        input_root=Path(args.input_dir),
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
                "INPUT_MATURATION_ARTIFACT_SHA": manifest["INPUT_MATURATION_ARTIFACT_SHA"],
                "TARGET_REACTION_READY_FEATURE_BLOCKED": manifest[
                    "TARGET_REACTION_READY_FEATURE_BLOCKED"
                ],
                "FEATURE_READY_RECOVERED": manifest["FEATURE_READY_RECOVERED"],
                "FEATURE_READY_STILL_BLOCKED": manifest["FEATURE_READY_STILL_BLOCKED"],
                "FEATURE_READY_BEFORE": manifest["FEATURE_READY_BEFORE"],
                "FEATURE_READY_AFTER": manifest["FEATURE_READY_AFTER"],
                "MARKET_FEATURES_COMPLETE": manifest["MARKET_FEATURES_COMPLETE"],
                "SEMANTIC_EVENT_FEATURES_PRESENT": manifest["SEMANTIC_EVENT_FEATURES_PRESENT"],
                "SEMANTIC_EVENT_FEATURES_RECONSTRUCTED": manifest[
                    "SEMANTIC_EVENT_FEATURES_RECONSTRUCTED"
                ],
                "SEMANTIC_EVENT_FEATURES_MISSING": manifest["SEMANTIC_EVENT_FEATURES_MISSING"],
                "FINAL_DECISION": manifest["FINAL_DECISION"],
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose and recover strict-EXACT reaction-ready feature readiness."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_ARTIFACT_ROOT)
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
