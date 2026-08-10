from __future__ import annotations

import argparse
import asyncio
import json
from uuid import UUID

from src.ai_events.application.serialization import analysis_result_to_json, failure_to_json
from src.ai_events.application.use_cases import AnalyzeAIEventCommand, sanitize_failure
from src.ai_events.domain.enums import AIProvider
from src.ai_events.infrastructure.factory import (
    create_ai_event_analyzer,
    resolve_ai_provider_config,
)
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    news_id = None if args.news_id is None else UUID(args.news_id)
    raw_content = args.text
    if news_id is not None:
        engine = create_engine(settings.database_url)
        try:
            async with create_session_factory(engine)() as session:
                news = await SqlAlchemyNewsRepository(session).get_by_id(news_id)
        finally:
            await engine.dispose()
        if news is None:
            print(json.dumps({"error": "news item not found"}, sort_keys=True))
            return 1
        raw_content = news.raw_content
    assert raw_content is not None
    try:
        provider = resolve_ai_provider_config(settings, args.provider, args.model)
        analyzer = create_ai_event_analyzer(
            settings,
            provider_override=args.provider,
            model_override=args.model,
        )
        result = await analyzer.execute(
            AnalyzeAIEventCommand(
                provider=provider.provider.value,
                raw_content=raw_content,
                news_id=news_id,
                record_id=args.record_id,
                requested_model=provider.requested_model,
                reasoning_effort=provider.reasoning_effort,
                max_output_tokens=settings.ai_max_output_tokens,
                think=provider.think,
                random_seed=provider.random_seed,
                context_length=provider.context_length,
                force_refresh=args.force_refresh,
            )
        )
    except Exception as exc:
        failure = sanitize_failure(exc, record_id=args.record_id, news_id=news_id)
        print(json.dumps(failure_to_json(failure), ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(analysis_result_to_json(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze corporate events with a configured AI.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--news-id")
    parser.add_argument("--record-id")
    parser.add_argument("--provider", choices=[item.value for item in AIProvider])
    parser.add_argument("--model")
    parser.add_argument("--force-refresh", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
