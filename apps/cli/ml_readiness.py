from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from src.predictive_baseline.application import dataset_readiness
from src.predictive_baseline.data import load_daily_predictive_dataset
from src.predictive_baseline.reporting import write_readiness


def run(args: argparse.Namespace) -> int:
    dataset = load_daily_predictive_dataset(Path(args.features), Path(args.reactions))
    coverage = _read_object(Path(args.coverage))
    intraday = coverage.get("intraday")
    intraday_payload = cast("dict[str, Any]", intraday) if isinstance(intraday, dict) else {}
    payload = dataset_readiness(
        dataset,
        intraday_feature_ready=int(intraday_payload.get("feature_ready", 0)),
    )
    write_readiness(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report predictive baseline data readiness.")
    parser.add_argument(
        "--features",
        default="artifacts/free-daily-historical-v1/daily-feature-dataset.jsonl",
    )
    parser.add_argument(
        "--reactions",
        default="artifacts/free-daily-historical-v1/daily-reactions.jsonl",
    )
    parser.add_argument(
        "--coverage",
        default="artifacts/free-daily-historical-v1/coverage.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/predictive-baseline-v1/readiness.json",
    )
    return parser


def _read_object(path: Path) -> dict[str, Any]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return {str(key): item for key, item in cast("dict[object, Any]", value).items()}


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
