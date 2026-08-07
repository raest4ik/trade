from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.evaluation.application.use_cases import import_annotation_dataset
from src.evaluation.domain.serialization import annotation_from_json
from src.evaluation.infrastructure.repositories import (
    SqlAlchemyEvaluationRepository,
    dataset_from_examples,
)
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    path = Path(args.input)
    settings = get_settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        repository = SqlAlchemyEvaluationRepository(session)
        if args.dry_run:
            lines = await asyncio.to_thread(_read_lines, path)
            examples = [annotation_from_json(json.loads(line)) for line in lines if line.strip()]
            dataset = dataset_from_examples(
                name=args.name,
                source_file_hash="dry-run",
                examples=examples,
                description=args.description,
            )
            print(
                "dry_run=true "
                f"examples={dataset.example_count} reviewed={dataset.reviewed_count} "
                f"train={dataset.train_count} validation={dataset.validation_count} "
                f"test={dataset.test_count}"
            )
            return 0
        result = await import_annotation_dataset(
            repository=repository,
            path=path,
            name=args.name,
            description=args.description,
            allow_missing_news=args.allow_missing_news,
        )
    await engine.dispose()
    print(f"dataset_id={result.dataset.id} created={str(result.created).lower()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a validated event-gold-v1 dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace-draft", action="store_true")
    parser.add_argument("--allow-missing-news", action="store_true")
    return parser


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
