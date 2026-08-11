from __future__ import annotations

import asyncio
import json
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from src.ai_events.application.serialization import analysis_result_to_json
from src.ai_events.application.use_cases import (
    AIEventAnalysisResult,
    AnalyzeAIEvent,
    AnalyzeAIEventCommand,
    sanitize_failure,
)
from src.ai_events.domain.enums import AIProvider
from src.ai_events.domain.prompt import (
    ANALYSIS_VERSION,
    FACT_EXTRACTOR_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    prompt_hash,
    schema_hash,
)
from src.ai_events.infrastructure.factory import (
    AIProviderConfig,
    create_ai_event_analyzer,
    resolve_ai_provider_config,
)
from src.ai_events.infrastructure.ollama_client import (
    OllamaModelIdentity,
    fetch_ollama_model_identity,
)
from src.events.domain.analyzer import EventAnalyzer
from src.events.domain.entities import (
    EVENT_ANALYSIS_VERSION,
    FINANCIAL_FACTS_VERSION,
    NewsEventAnalysis,
)
from src.events.domain.enums import EventAnalysisStatus, EventType
from src.real_gold_benchmark.domain import (
    BenchmarkExample,
    BenchmarkPrediction,
    BenchmarkValidationError,
    FrozenDataset,
    PredictionEvaluation,
    analyzer_input,
    evaluate_prediction_set,
    prediction_payload,
    write_json,
    write_jsonl,
)
from src.shared.config.settings import Settings

QWEN_PROVIDER = "ollama"
QWEN_MODEL = "qwen3.5:9b"
QWEN_THINK = False
QWEN_RANDOM_SEED = 0
QWEN_CONTEXT_LENGTH = 4096


def run_rules(frozen: FrozenDataset) -> tuple[BenchmarkPrediction, ...]:
    _verify_frozen(frozen)
    analyzer = EventAnalyzer()
    predictions: list[BenchmarkPrediction] = []
    for example in frozen.examples:
        payload = analyzer_input(example)
        started = time.perf_counter()
        analysis = analyzer.analyze(
            news_id=example.annotation.news_id,
            raw_content=cast("str", payload["raw_content"]),
        )
        predictions.append(
            BenchmarkPrediction(
                record_id=example.record_id,
                news_id=example.annotation.news_id,
                analysis=analysis,
                runtime={"latency_ms": round((time.perf_counter() - started) * 1000)},
            )
        )
    return tuple(predictions)


async def prepare_qwen_frozen_config(
    frozen: FrozenDataset,
    *,
    settings: Settings,
    config_path: Path,
) -> tuple[AIProviderConfig, OllamaModelIdentity, dict[str, object]]:
    _verify_frozen(frozen)
    provider = resolve_ai_provider_config(settings, QWEN_PROVIDER, QWEN_MODEL)
    _validate_qwen_provider(provider, settings)
    identity = await fetch_ollama_model_identity(
        base_url=settings.ollama_base_url,
        model_tag=QWEN_MODEL,
        timeout_seconds=settings.ai_request_timeout_seconds,
    )
    config = _qwen_config(frozen, provider, identity, settings)
    if await asyncio.to_thread(config_path.exists):
        existing = await asyncio.to_thread(_read_object, config_path)
        mismatches = [key for key, value in config.items() if existing.get(key) != value]
        if mismatches:
            raise BenchmarkValidationError(
                "frozen Qwen config mismatch: " + ", ".join(sorted(mismatches))
            )
    else:
        await asyncio.to_thread(write_json, config_path, config)
    return provider, identity, config


async def run_qwen(
    frozen: FrozenDataset,
    *,
    settings: Settings,
    output_directory: Path,
    frozen_config_path: Path,
) -> tuple[tuple[BenchmarkPrediction, ...], dict[str, object]]:
    provider, _, expected_config = await prepare_qwen_frozen_config(
        frozen,
        settings=settings,
        config_path=frozen_config_path,
    )
    verify_qwen_config(frozen_config_path, expected_config)
    analyzer = create_ai_event_analyzer(
        settings,
        cache_directory=output_directory / "cache",
        provider_override=QWEN_PROVIDER,
        model_override=QWEN_MODEL,
    )
    results = await asyncio.gather(
        *[
            _run_qwen_example(analyzer, example, settings=settings, provider=provider)
            for example in frozen.examples
        ]
    )
    return tuple(results), expected_config


def write_prediction_artifacts(
    output_directory: Path,
    *,
    frozen: FrozenDataset,
    predictions: tuple[BenchmarkPrediction, ...],
    evaluation: PredictionEvaluation,
    manifest: dict[str, object],
) -> None:
    by_id = {item.news_id: item for item in predictions}
    write_jsonl(
        output_directory / "predictions.jsonl",
        [prediction_payload(item, by_id[item.annotation.news_id]) for item in frozen.examples],
    )
    write_json(output_directory / "metrics.json", evaluation.metrics)
    write_jsonl(output_directory / "errors.jsonl", [dict(item) for item in evaluation.errors])
    write_json(
        output_directory / "manifest.json",
        {
            "created_at": _utc_now(),
            "git_commit_sha": _git_sha(),
            "dataset_name": frozen.manifest["name"],
            "dataset_sha256": frozen.dataset_sha256,
            "records": len(frozen.examples),
            "input_policy": "frozen raw_content only",
            "gold_fields_in_analyzer_input": False,
            "future_market_fields_in_analyzer_input": False,
            "model_tuning_performed": False,
            **manifest,
        },
    )


def rules_manifest() -> dict[str, object]:
    return {
        "system": "rules-v2",
        "analysis_version": EVENT_ANALYSIS_VERSION,
        "fact_extractor_version": FINANCIAL_FACTS_VERSION,
        "rules_changed": False,
    }


def qwen_manifest(
    config: dict[str, object],
    predictions: tuple[BenchmarkPrediction, ...],
) -> dict[str, object]:
    actual_models = Counter(
        str(item.runtime.get("actual_model"))
        for item in predictions
        if item.succeeded and item.runtime.get("actual_model")
    )
    return {
        "system": QWEN_MODEL,
        **config,
        "actual_response_models": dict(sorted(actual_models.items())),
        "successful": sum(item.succeeded for item in predictions),
        "failed": sum(not item.succeeded for item in predictions),
        "semantic_rerun_performed": False,
        "prompt_changed": False,
        "schema_changed": False,
        "model_config_changed": False,
    }


def evaluate(
    frozen: FrozenDataset,
    predictions: tuple[BenchmarkPrediction, ...],
    *,
    system_name: str,
) -> PredictionEvaluation:
    return evaluate_prediction_set(frozen, predictions, system_name=system_name)


async def _run_qwen_example(
    analyzer: AnalyzeAIEvent,
    example: BenchmarkExample,
    *,
    settings: Settings,
    provider: AIProviderConfig,
) -> BenchmarkPrediction:
    payload = analyzer_input(example)
    try:
        result = await analyzer.execute(
            AnalyzeAIEventCommand(
                provider=provider.provider.value,
                raw_content=cast("str", payload["raw_content"]),
                requested_model=provider.requested_model,
                reasoning_effort=provider.reasoning_effort,
                max_output_tokens=settings.ai_max_output_tokens,
                think=provider.think,
                random_seed=provider.random_seed,
                context_length=provider.context_length,
                news_id=example.annotation.news_id,
                record_id=example.record_id,
                force_refresh=False,
            )
        )
    except Exception as exc:
        failure = sanitize_failure(
            exc,
            record_id=example.record_id,
            news_id=example.annotation.news_id,
        )
        return BenchmarkPrediction(
            record_id=example.record_id,
            news_id=example.annotation.news_id,
            analysis=_failed_analysis(example),
            runtime={},
            failure={
                "error_code": failure.error_code,
                "message": failure.message,
            },
        )
    return _qwen_prediction(example, result)


def _qwen_prediction(
    example: BenchmarkExample,
    result: AIEventAnalysisResult,
) -> BenchmarkPrediction:
    serialized = analysis_result_to_json(result)
    metadata = cast("dict[str, object]", serialized["metadata"])
    return BenchmarkPrediction(
        record_id=example.record_id,
        news_id=example.annotation.news_id,
        analysis=result.analysis,
        runtime={
            "latency_ms": result.metadata.latency_ms,
            "input_tokens": result.metadata.input_tokens,
            "output_tokens": result.metadata.output_tokens,
            "total_tokens": result.metadata.total_tokens,
            "cached": result.metadata.cached,
            "actual_model": result.metadata.actual_model,
            "prompt_sha256": metadata["prompt_sha256"],
            "schema_sha256": metadata["schema_sha256"],
        },
    )


def _failed_analysis(example: BenchmarkExample) -> NewsEventAnalysis:
    return NewsEventAnalysis.create(
        news_id=example.annotation.news_id,
        status=EventAnalysisStatus.FAILED,
        primary_event_type=EventType.UNKNOWN,
        events=[],
        financial_facts=[],
        analysis_version=ANALYSIS_VERSION,
    )


def _validate_qwen_provider(provider: AIProviderConfig, settings: Settings) -> None:
    expected = {
        "provider": AIProvider.OLLAMA,
        "requested_model": QWEN_MODEL,
        "think": QWEN_THINK,
        "random_seed": QWEN_RANDOM_SEED,
        "context_length": QWEN_CONTEXT_LENGTH,
    }
    actual = {
        "provider": provider.provider,
        "requested_model": provider.requested_model,
        "think": provider.think,
        "random_seed": provider.random_seed,
        "context_length": provider.context_length,
    }
    mismatches = [key for key, value in expected.items() if actual[key] != value]
    if settings.ai_max_output_tokens != 4096:
        mismatches.append("max_output_tokens")
    if mismatches:
        raise BenchmarkValidationError(
            "Qwen frozen settings mismatch: " + ", ".join(sorted(mismatches))
        )


def _qwen_config(
    frozen: FrozenDataset,
    provider: AIProviderConfig,
    identity: OllamaModelIdentity,
    settings: Settings,
) -> dict[str, object]:
    return {
        "dataset_sha256": frozen.dataset_sha256,
        "provider": provider.provider.value,
        "requested_model": provider.requested_model,
        "model_tag": identity.model_tag,
        "model_digest": identity.model_digest,
        "parameter_size": identity.parameter_size,
        "quantization_level": identity.quantization_level,
        "ollama_version": identity.ollama_version,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash(),
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": schema_hash(),
        "analyzer_version": ANALYSIS_VERSION,
        "fact_extractor_version": FACT_EXTRACTOR_VERSION,
        "think": provider.think,
        "random_seed": provider.random_seed,
        "context_length": provider.context_length,
        "max_output_tokens": settings.ai_max_output_tokens,
        "temperature": 0,
    }


def verify_qwen_config(path: Path, expected: dict[str, object]) -> None:
    current = _read_object(path)
    mismatches = [key for key, value in expected.items() if current.get(key) != value]
    if mismatches:
        raise BenchmarkValidationError(
            "Qwen prompt/schema/model fingerprint mismatch: " + ", ".join(sorted(mismatches))
        )


def _verify_frozen(frozen: FrozenDataset) -> None:
    if frozen.manifest.get("freeze_state") != "FROZEN_BEFORE_PREDICTIONS":
        raise BenchmarkValidationError("predictions require a pre-frozen dataset")
    if frozen.manifest.get("dataset_sha256") != frozen.dataset_sha256:
        raise BenchmarkValidationError("frozen dataset fingerprint changed")


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"expected JSON object: {path}")
    return {str(key): item for key, item in cast("dict[object, object]", value).items()}


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
