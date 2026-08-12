from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.market_baseline.reporting import load_market_baseline_status


def run(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            load_market_baseline_status(Path(args.output_dir)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show market baseline dataset readiness.")
    parser.add_argument("--output-dir", default="artifacts/market-baseline-v1")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
