from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.ai_events.application.serialization import analysis_result_to_json, failure_to_json
from src.ai_events.application.use_cases import AnalyzeAIEventCommand, sanitize_failure
from src.ai_events.infrastructure.factory import (
    create_ai_event_analyzer,
    resolve_ai_provider_config,
)
from src.events.domain.entities import EVENT_ANALYSIS_VERSION
from src.events.infrastructure.models import NewsEventAnalysisRecord
from src.news.infrastructure.models import NewsItemRecord
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    baseline_ids = _baseline_ids(Path(args.baseline))
    settings = get_settings()
    provider = resolve_ai_provider_config(settings, "ollama", args.model)
    if provider.requested_model != "qwen3.5:9b" or provider.think or provider.random_seed != 0:
        raise ValueError("shadow run requires qwen3.5:9b, think=false, and random seed 0")
    engine = create_engine(settings.database_url)
    try:
        before_signature = await _deterministic_signature(engine, baseline_ids)
        feature_before = _file_sha256(Path(args.feature_corpus))
        news_items = await _load_news(engine, baseline_ids)
        analyzer = create_ai_event_analyzer(
            settings,
            cache_directory=output_dir / "cache",
            provider_override="ollama",
            model_override=args.model,
        )
        predictions: list[dict[str, Any]] = []
        for news in news_items:
            try:
                result = await analyzer.execute(
                    AnalyzeAIEventCommand(
                        provider=provider.provider.value,
                        raw_content=news.raw_content,
                        news_id=news.id,
                        record_id=str(news.id),
                        requested_model=provider.requested_model,
                        reasoning_effort=provider.reasoning_effort,
                        max_output_tokens=settings.ai_max_output_tokens,
                        think=provider.think,
                        random_seed=provider.random_seed,
                        context_length=provider.context_length,
                        force_refresh=args.force_refresh,
                    )
                )
                payload = analysis_result_to_json(result)
                payload["news_id"] = str(news.id)
            except Exception as exc:
                failure = sanitize_failure(exc, record_id=str(news.id), news_id=news.id)
                payload = failure_to_json(failure)
                payload["news_id"] = str(news.id)
            predictions.append(payload)
        after_signature = await _deterministic_signature(engine, baseline_ids)
        feature_after = _file_sha256(Path(args.feature_corpus))
    finally:
        await engine.dispose()
    prediction_path = output_dir / "predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    successful = [item for item in predictions if item.get("status") != "FAILED"]
    summary = {
        "items": len(predictions),
        "successful": len(successful),
        "failed": len(predictions) - len(successful),
        "event_detected_count": sum(bool(item.get("events")) for item in successful),
        "primary_UNKNOWN_count": sum(
            item.get("primary_event_type") == "UNKNOWN" for item in successful
        ),
        "fact_count": sum(len(item.get("financial_facts", [])) for item in successful),
        "coverage_only_not_accuracy": True,
    }
    metadata_value = successful[0].get("metadata") if successful else None
    first_metadata = (
        cast("dict[str, object]", metadata_value) if isinstance(metadata_value, dict) else {}
    )
    manifest: dict[str, object] = {
        "provider": "ollama",
        "model": provider.requested_model,
        "think": provider.think,
        "random_seed": provider.random_seed,
        "prompt_version": first_metadata.get("prompt_version"),
        "prompt_sha256": first_metadata.get("prompt_sha256"),
        "schema_version": first_metadata.get("schema_version"),
        "schema_sha256": first_metadata.get("schema_sha256"),
        "items_requested": len(baseline_ids),
        "deterministic_analysis_signature_before": before_signature,
        "deterministic_analysis_signature_after": after_signature,
        "deterministic_analysis_unchanged": before_signature == after_signature,
        "ml_feature_corpus_sha256_before": feature_before,
        "ml_feature_corpus_sha256_after": feature_after,
        "ml_features_unchanged": feature_before == feature_after,
        "writes_gold": False,
        "writes_reaction_labels": False,
        "writes_deterministic_features": False,
        "hybrid_enabled": False,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({**summary, "output": str(output_dir)}, sort_keys=True))
    return 0 if len(successful) == len(predictions) else 1


async def _load_news(engine: Any, news_ids: set[UUID]) -> list[NewsItemRecord]:
    async with create_session_factory(engine)() as session:
        result = await session.execute(
            select(NewsItemRecord)
            .where(NewsItemRecord.id.in_(news_ids))
            .order_by(NewsItemRecord.published_at, NewsItemRecord.id)
        )
        items = list(result.scalars())
    if {item.id for item in items} != news_ids:
        raise ValueError("baseline news IDs are not all present in the configured database")
    return items


async def _deterministic_signature(engine: Any, news_ids: set[UUID]) -> str:
    async with create_session_factory(engine)() as session:
        result = await session.execute(
            select(NewsEventAnalysisRecord)
            .where(
                NewsEventAnalysisRecord.news_id.in_(news_ids),
                NewsEventAnalysisRecord.analysis_version == EVENT_ANALYSIS_VERSION,
            )
            .order_by(NewsEventAnalysisRecord.news_id)
            .options(
                selectinload(NewsEventAnalysisRecord.events),
                selectinload(NewsEventAnalysisRecord.financial_facts),
            )
        )
        rows = [
            {
                "news_id": str(item.news_id),
                "analysis_version": item.analysis_version,
                "status": item.status,
                "primary_event_type": item.primary_event_type,
                "events": sorted(
                    (event.event_type, event.rule_id, event.start_position, event.end_position)
                    for event in item.events
                ),
                "facts": sorted(
                    (fact.metric, fact.rule_id, fact.start_position, fact.end_position)
                    for fact in item.financial_facts
                ),
            }
            for item in result.scalars()
        ]
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen Qwen coverage diagnostics in shadow mode."
    )
    parser.add_argument("--baseline", default="artifacts/corpus-quality-v1/rosn-baseline.json")
    parser.add_argument("--output", default="artifacts/corpus-quality-v1/qwen-shadow")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument(
        "--feature-corpus",
        default="artifacts/reaction-ready-corpus-v1/corpus.jsonl",
    )
    parser.add_argument("--force-refresh", action="store_true")
    return parser


def _baseline_ids(path: Path) -> set[UUID]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    identifiers = {UUID(value) for value in payload["news_ids"]}
    if len(identifiers) != 10:
        raise ValueError("Qwen shadow run is restricted to the frozen 10-row baseline")
    return identifiers


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
