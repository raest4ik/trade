from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID, uuid4

from src.ai_events.application.ports import (
    AIEventCache,
    AIEventModelClient,
    AIModelCompletion,
    AIModelRequest,
)
from src.ai_events.domain.evidence import resolve_exact_evidence
from src.ai_events.domain.exceptions import AIConfigurationError, AIEventError
from src.ai_events.domain.prompt import (
    ANALYSIS_VERSION,
    FACT_EXTRACTOR_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    canonical_json,
    prompt_hash,
    schema_hash,
    sha256_text,
)
from src.events.domain.entities import (
    DetectedEvent,
    ExtractedFinancialFact,
    NewsEventAnalysis,
)
from src.events.domain.enums import EventAnalysisStatus, EventType
from src.news.domain.time import utc_now


@dataclass(frozen=True, slots=True)
class AnalyzeAIEventCommand:
    raw_content: str
    requested_model: str
    reasoning_effort: str | None
    max_output_tokens: int
    news_id: UUID | None = None
    record_id: str | None = None
    force_refresh: bool = False


@dataclass(frozen=True, slots=True)
class AIAnalysisMetadata:
    record_id: str | None
    news_id: UUID | None
    raw_content_hash: str
    requested_model: str
    actual_model: str
    prompt_version: str
    prompt_hash: str
    schema_version: str
    schema_hash: str
    analyzer_version: str
    fact_extractor_version: str
    reasoning_effort: str | None
    response_id: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached: bool
    cache_key: str


@dataclass(frozen=True, slots=True)
class AIEventAnalysisResult:
    analysis: NewsEventAnalysis
    warnings: list[str]
    metadata: AIAnalysisMetadata


@dataclass(frozen=True, slots=True)
class AIItemFailure:
    record_id: str | None
    news_id: UUID | None
    error_code: str
    message: str


class AnalyzeAIEvent:
    def __init__(self, client: AIEventModelClient, cache: AIEventCache) -> None:
        self._client = client
        self._cache = cache

    async def execute(self, command: AnalyzeAIEventCommand) -> AIEventAnalysisResult:
        if not command.raw_content.strip():
            raise AIConfigurationError("raw_content must not be empty")
        if not command.requested_model.strip():
            raise AIConfigurationError("requested_model must not be empty")
        if command.max_output_tokens <= 0:
            raise AIConfigurationError("max_output_tokens must be positive")

        request = build_model_request(command)
        key = cache_key(request)
        completion = None if command.force_refresh else await self._cache.get(key)
        cached = completion is not None
        if completion is None:
            completion = await self._client.analyze(request)
            await self._cache.put(key, completion)
        return _build_result(command, completion, key=key, cached=cached)


def build_model_request(command: AnalyzeAIEventCommand) -> AIModelRequest:
    return AIModelRequest(
        raw_content=command.raw_content,
        requested_model=command.requested_model,
        instructions=SYSTEM_PROMPT,
        prompt_version=PROMPT_VERSION,
        prompt_hash=prompt_hash(),
        schema_version=SCHEMA_VERSION,
        schema_hash=schema_hash(),
        analyzer_version=ANALYSIS_VERSION,
        reasoning_effort=command.reasoning_effort,
        max_output_tokens=command.max_output_tokens,
    )


def cache_key(request: AIModelRequest) -> str:
    payload = {
        "analyzer_version": request.analyzer_version,
        "prompt_hash": request.prompt_hash,
        "raw_content_hash": sha256_text(request.raw_content),
        "reasoning_effort": request.reasoning_effort,
        "requested_model": request.requested_model,
        "schema_hash": request.schema_hash,
        "max_output_tokens": request.max_output_tokens,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sanitize_failure(
    exc: Exception,
    *,
    record_id: str | None = None,
    news_id: UUID | None = None,
) -> AIItemFailure:
    if isinstance(exc, AIEventError):
        code = type(exc).__name__
    else:
        code = "AIEventUnexpectedError"
    return AIItemFailure(
        record_id=record_id,
        news_id=news_id,
        error_code=code,
        message="AI event analysis failed",
    )


def _build_result(
    command: AnalyzeAIEventCommand,
    completion: AIModelCompletion,
    *,
    key: str,
    cached: bool,
) -> AIEventAnalysisResult:
    output = completion.output
    analysis_id = uuid4()
    warnings = list(output.warnings)
    events: list[DetectedEvent] = []
    for prediction in output.events:
        span = resolve_exact_evidence(command.raw_content, prediction.evidence_text)
        if span.warning:
            warnings.append(f"event {prediction.event_type.value}: {span.warning}")
        events.append(
            DetectedEvent(
                id=uuid4(),
                analysis_id=analysis_id,
                event_type=prediction.event_type,
                confidence=Decimal(str(prediction.confidence)),
                rule_id=PROMPT_VERSION,
                matched_rule="openai-responses-structured-output",
                evidence_text=prediction.evidence_text,
                start_position=span.start,
                end_position=span.end,
            )
        )
    facts: list[ExtractedFinancialFact] = []
    for prediction in output.financial_facts:
        span = resolve_exact_evidence(command.raw_content, prediction.evidence_text)
        if span.warning:
            warnings.append(f"fact {prediction.metric.value}: {span.warning}")
        normalized_value = prediction.decimal_value()
        facts.append(
            ExtractedFinancialFact(
                id=uuid4(),
                analysis_id=analysis_id,
                metric=prediction.metric,
                raw_value=normalized_value,
                normalized_value=normalized_value,
                unit=prediction.unit,
                currency=prediction.currency,
                scale=prediction.scale,
                period_type=prediction.period_type,
                year=prediction.period_year,
                quarter=prediction.period_quarter,
                month=None,
                date_from=None,
                date_to=None,
                raw_period=None,
                comparison_type=prediction.comparison_type,
                fact_role=prediction.fact_role,
                change_direction=prediction.change_direction,
                change_value=prediction.decimal_change_value(),
                change_unit=prediction.change_unit,
                confidence=Decimal(str(prediction.confidence)),
                rule_id=PROMPT_VERSION,
                evidence_text=prediction.evidence_text,
                start_position=span.start,
                end_position=span.end,
                extractor_version=FACT_EXTRACTOR_VERSION,
                matched_rule=prediction.metric_name or "openai-responses-structured-output",
            )
        )
    primary = next(
        (item.event_type for item in output.events if item.is_primary),
        EventType.UNKNOWN,
    )
    status = _status(output.events, facts, primary)
    now = utc_now()
    analysis = NewsEventAnalysis(
        id=analysis_id,
        news_id=command.news_id or uuid4(),
        analysis_version=ANALYSIS_VERSION,
        status=status,
        primary_event_type=primary,
        created_at=now,
        analyzed_at=now,
        events=events,
        financial_facts=facts,
    )
    metadata = AIAnalysisMetadata(
        record_id=command.record_id,
        news_id=command.news_id,
        raw_content_hash=sha256_text(command.raw_content),
        requested_model=command.requested_model,
        actual_model=completion.actual_model,
        prompt_version=PROMPT_VERSION,
        prompt_hash=prompt_hash(),
        schema_version=SCHEMA_VERSION,
        schema_hash=schema_hash(),
        analyzer_version=ANALYSIS_VERSION,
        fact_extractor_version=FACT_EXTRACTOR_VERSION,
        reasoning_effort=command.reasoning_effort,
        response_id=completion.response_id,
        latency_ms=completion.latency_ms,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        total_tokens=completion.total_tokens,
        cached=cached,
        cache_key=key,
    )
    return AIEventAnalysisResult(analysis=analysis, warnings=warnings, metadata=metadata)


def _status(
    events: Sequence[object],
    facts: list[ExtractedFinancialFact],
    primary: EventType,
) -> EventAnalysisStatus:
    if not events and not facts:
        return EventAnalysisStatus.NO_EVENT_FOUND
    if not events or primary == EventType.UNKNOWN:
        return EventAnalysisStatus.PARTIAL
    return EventAnalysisStatus.COMPLETE
