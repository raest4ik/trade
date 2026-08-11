from __future__ import annotations

import argparse
from pathlib import Path

from src.events.domain.v3 import EventAnalyzerV3
from src.holdout_evaluation.application import (
    claim_single_run,
    run_single_holdout_evaluation,
    write_holdout_artifacts,
)
from src.holdout_evaluation.domain import freeze_holdout_gold, verify_frozen_candidate


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    candidate = verify_frozen_candidate(Path(args.candidate_manifest))
    dataset = freeze_holdout_gold(
        source_review_path=Path(args.holdout_review),
        split_manifest_path=Path(args.split_manifest),
        output_directory=output / "holdout-gold",
    )
    claim_single_run(output)
    result = run_single_holdout_evaluation(dataset, EventAnalyzerV3())
    write_holdout_artifacts(
        output_directory=output,
        dataset=dataset,
        candidate=candidate,
        result=result,
    )
    event_micro = result.metrics["micro"]
    print(
        f"records={len(dataset.records)} primary_accuracy={result.metrics['primary_accuracy']} "
        f"micro_f1={event_micro['f1']} status=OBSERVED_HOLDOUT"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the single blind HOLDOUT evaluation of frozen event-rules-v3."
    )
    parser.add_argument(
        "--holdout-review",
        default="artifacts/fresh-real-corpus-v1/batch-004-holdout-human-review-v1.jsonl",
    )
    parser.add_argument(
        "--split-manifest",
        default="artifacts/fresh-real-corpus-v1/split-manifest.json",
    )
    parser.add_argument(
        "--candidate-manifest",
        default="artifacts/event-rules-v3-real-dev/candidate/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/event-rules-v3-holdout-eval",
    )
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
