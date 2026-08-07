from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from uuid import UUID

from src.evaluation.application.exceptions import EvaluationThresholdError
from src.evaluation.application.use_cases import run_evaluation
from src.evaluation.domain.enums import DatasetSplit
from src.evaluation.infrastructure.repositories import SqlAlchemyEvaluationRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            repository = SqlAlchemyEvaluationRepository(session)
            result = await run_evaluation(
                repository=repository,
                dataset_id=UUID(args.dataset_id),
                split=DatasetSplit(args.split),
                thresholds_path=Path(args.thresholds),
                output_dir=Path(args.output_dir),
                fail_below_thresholds=args.fail_below_thresholds,
            )
    except EvaluationThresholdError as exc:
        await engine.dispose()
        print(f"threshold_failed={exc}")
        return 1
    await engine.dispose()
    print(
        f"run_id={result.run.id} status={result.run.status.value} "
        f"examples={result.run.example_count} report_dir={result.report_directory}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate deterministic event extraction.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--split",
        default=DatasetSplit.TEST.value,
        choices=[item.value for item in DatasetSplit],
    )
    parser.add_argument("--thresholds", default="config/evaluation_thresholds.toml")
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    parser.add_argument("--fail-below-thresholds", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
