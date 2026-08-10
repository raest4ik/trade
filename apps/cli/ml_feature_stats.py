from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ml_features.infrastructure.export import dataset_stats, load_jsonl_rows


def run(args: argparse.Namespace) -> int:
    rows = load_jsonl_rows(Path(args.input))
    stats = dataset_stats(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(stats, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute ml-feature-dataset-v1 stats.")
    parser.add_argument(
        "--input",
        default="artifacts/ml-feature-dataset-v1/ml-feature-dataset-v1.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/ml-feature-dataset-v1/stats.json",
    )
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
