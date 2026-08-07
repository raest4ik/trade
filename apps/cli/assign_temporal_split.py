from __future__ import annotations

import argparse
import asyncio
from datetime import date
from uuid import UUID

from src.evaluation.domain.enums import DatasetSplit
from src.evaluation.infrastructure.repositories import SqlAlchemyEvaluationRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    train_until = date.fromisoformat(args.train_until)
    validation_until = date.fromisoformat(args.validation_until)
    dataset_id = UUID(args.dataset_id)
    settings = get_settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        repository = SqlAlchemyEvaluationRepository(session)
        if args.dry_run:
            rows = await repository.list_examples_with_news(dataset_id=dataset_id)
            counts = {split: 0 for split in DatasetSplit}
            for row in rows:
                published = row.news.published_at.date()
                if published <= train_until:
                    counts[DatasetSplit.TRAIN] += 1
                elif published <= validation_until:
                    counts[DatasetSplit.VALIDATION] += 1
                else:
                    counts[DatasetSplit.TEST] += 1
        else:
            counts = await repository.assign_temporal_split(
                dataset_id=dataset_id,
                train_until=train_until,
                validation_until=validation_until,
            )
    await engine.dispose()
    print(
        f"train={counts[DatasetSplit.TRAIN]} "
        f"validation={counts[DatasetSplit.VALIDATION]} test={counts[DatasetSplit.TEST]}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assign temporal train/validation/test splits.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--train-until", required=True)
    parser.add_argument("--validation-until", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
