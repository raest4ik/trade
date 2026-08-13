from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.market_predictive_research.application import future_status


def run(args: argparse.Namespace) -> int:
    print(json.dumps(future_status(Path(args.dataset_dir)), sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report future market holdout coverage without loading outcomes."
    )
    parser.add_argument("--dataset-dir", default="artifacts/tinvest-market-baseline-features-v1")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
