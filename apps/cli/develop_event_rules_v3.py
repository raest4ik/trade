from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.events.domain.analyzer import EventAnalyzer
from src.events.domain.v3 import EventAnalyzerV3, rules_v3_fingerprint
from src.real_dev_rules.application import (
    evaluate_deterministic,
    evaluate_frozen_qwen,
    write_baseline_artifacts,
    write_candidate_artifacts,
)
from src.real_dev_rules.domain import freeze_development_gold


async def run(args: argparse.Namespace) -> int:
    source_path = Path(args.human_review)
    if "development-human-review" not in source_path.name:
        raise ValueError("input must be the DEVELOPMENT-only human review export")
    output = Path(args.output_dir)
    dataset = freeze_development_gold(
        source_review_path=source_path,
        split_manifest_path=Path(args.split_manifest),
        output_directory=output / "development-gold",
    )
    rules_v2 = evaluate_deterministic(dataset, EventAnalyzer(), label="event-rules-v2")
    qwen = await evaluate_frozen_qwen(
        dataset,
        cache_directory=output / "qwen-cache",
    )
    write_baseline_artifacts(
        output_directory=output / "baseline",
        dataset=dataset,
        rules_v2=rules_v2,
        qwen=qwen,
    )
    rules_v3 = evaluate_deterministic(dataset, EventAnalyzerV3(), label="event-rules-v3")
    write_candidate_artifacts(
        output_directory=output / "candidate",
        dataset=dataset,
        rules_v2=rules_v2,
        rules_v3=rules_v3,
        qwen=qwen,
    )
    print(
        f"development={len(dataset.records)} holdout_metadata={dataset.holdout_count} "
        f"qwen_successful={qwen.successful} qwen_failed={qwen.failed} "
        f"rules_fingerprint={rules_v3_fingerprint()}"
    )
    return 0 if qwen.failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and evaluate event-rules-v3 on Batch 004 DEVELOPMENT only."
    )
    parser.add_argument(
        "--human-review",
        default=("artifacts/fresh-real-corpus-v1/batch-004-development-human-review-v1.jsonl"),
    )
    parser.add_argument(
        "--split-manifest",
        default="artifacts/fresh-real-corpus-v1/split-manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/event-rules-v3-real-dev",
    )
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
