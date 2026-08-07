from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import UUID

from src.evaluation.domain.validation import ValidationIssue, validate_jsonl_payloads
from src.evaluation.infrastructure.repositories import SqlAlchemyEvaluationRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    path = Path(args.input)
    lines = await asyncio.to_thread(_read_lines, path)
    news_ids = _news_ids(lines)
    settings = get_settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        repository = SqlAlchemyEvaluationRepository(session)
        content_by_id, hash_by_id = await repository.raw_news_maps(news_ids)
    await engine.dispose()
    result = validate_jsonl_payloads(
        lines,
        raw_content_by_news_id=content_by_id,
        raw_hash_by_news_id=hash_by_id,
        strict=args.strict,
        allow_missing_news=args.allow_missing_news,
    )
    for issue in result.errors:
        _print_issue("error", issue)
    if args.include_warnings:
        for issue in result.warnings:
            _print_issue("warning", issue)
    print(
        f"ok={str(result.ok).lower()} errors={len(result.errors)} warnings={len(result.warnings)}"
    )
    return 0 if result.ok else 1


def _news_ids(lines: list[str]) -> list[UUID]:
    ids: list[UUID] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            ids.append(UUID(str(payload["news_id"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return ids


def _print_issue(level: str, issue: ValidationIssue) -> None:
    print(
        f"{level} line={issue.line_number} code={issue.code} "
        f"news_id={issue.news_id} {issue.message}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an event-gold-v1 JSONL dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-missing-news", action="store_true")
    parser.add_argument("--include-warnings", action="store_true")
    return parser


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
