from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tinvest_market.config import token_presence
from src.tinvest_market.policy import execution_safety, source_policy
from src.tinvest_market.reporting import load_status


def run(args: argparse.Namespace) -> int:
    feature_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    if feature_dir.exists() and raw_dir.exists():
        status = load_status(feature_dir, raw_dir=raw_dir)
    else:
        status = {
            "READONLY_AUTH": "NOT_CHECKED",
            "SANDBOX_AUTH": "NOT_CHECKED",
            "raw_rows": 0,
            "feature_ready": 0,
            "warnings": ["LOCAL_DATASET_NOT_BUILT"],
        }
    print(
        json.dumps(
            {
                **status,
                "token_presence": token_presence(),
                "source_policy": source_policy(),
                "execution_safety": execution_safety(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show private read-only T-Invest market dataset status."
    )
    parser.add_argument("--raw-dir", default="artifacts/tinvest-market-raw-v1")
    parser.add_argument("--output-dir", default="artifacts/tinvest-market-baseline-features-v1")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
