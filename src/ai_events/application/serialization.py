from __future__ import annotations

from src.ai_events.application.use_cases import AIEventAnalysisResult, AIItemFailure


def analysis_result_to_json(result: AIEventAnalysisResult) -> dict[str, object]:
    analysis = result.analysis
    return {
        "analysis_version": analysis.analysis_version,
        "status": analysis.status.value,
        "primary_event_type": analysis.primary_event_type.value,
        "events": [
            {
                "event_type": event.event_type.value,
                "is_primary": event.event_type == analysis.primary_event_type,
                "confidence": str(event.confidence),
                "evidence_text": event.evidence_text,
                "evidence_valid": event.start_position >= 0 and event.end_position >= 0,
                "start_position": None if event.start_position < 0 else event.start_position,
                "end_position": None if event.end_position < 0 else event.end_position,
                "rule_id": event.rule_id,
            }
            for event in analysis.events
        ],
        "financial_facts": [
            {
                "metric": fact.metric.value,
                "metric_name": None
                if fact.matched_rule == "ai-structured-output"
                else fact.matched_rule,
                "normalized_value": str(fact.normalized_value),
                "unit": fact.unit.value,
                "currency": fact.currency.value,
                "scale": fact.scale.value,
                "fact_role": fact.fact_role.value,
                "period_type": fact.period_type.value,
                "period_year": fact.year,
                "period_quarter": fact.quarter,
                "comparison_type": fact.comparison_type.value,
                "change_direction": fact.change_direction.value,
                "change_value": None if fact.change_value is None else str(fact.change_value),
                "change_unit": None if fact.change_unit is None else fact.change_unit.value,
                "evidence_text": fact.evidence_text,
                "evidence_valid": fact.start_position >= 0 and fact.end_position >= 0,
                "start_position": None if fact.start_position < 0 else fact.start_position,
                "end_position": None if fact.end_position < 0 else fact.end_position,
                "confidence": str(fact.confidence),
                "extractor_version": fact.extractor_version,
            }
            for fact in analysis.financial_facts
        ],
        "warnings": result.warnings,
        "metadata": {
            "record_id": result.metadata.record_id,
            "news_id": None if result.metadata.news_id is None else str(result.metadata.news_id),
            "raw_content_hash": result.metadata.raw_content_hash,
            "provider": result.metadata.provider,
            "requested_model": result.metadata.requested_model,
            "actual_model": result.metadata.actual_model,
            "actual_response_model": result.metadata.actual_model,
            "prompt_version": result.metadata.prompt_version,
            "prompt_sha256": result.metadata.prompt_hash,
            "schema_version": result.metadata.schema_version,
            "schema_sha256": result.metadata.schema_hash,
            "analyzer_version": result.metadata.analyzer_version,
            "fact_extractor_version": result.metadata.fact_extractor_version,
            "reasoning_effort": result.metadata.reasoning_effort,
            "response_id": result.metadata.response_id,
            "latency_ms": result.metadata.latency_ms,
            "input_tokens": result.metadata.input_tokens,
            "output_tokens": result.metadata.output_tokens,
            "total_tokens": result.metadata.total_tokens,
            "cached": result.metadata.cached,
            "cache_key": result.metadata.cache_key,
            "provider_metadata": result.metadata.provider_metadata,
            "cloud_cost_usd": result.metadata.cloud_cost_usd,
        },
    }


def failure_to_json(failure: AIItemFailure) -> dict[str, object]:
    return {
        "status": "FAILED",
        "record_id": failure.record_id,
        "news_id": None if failure.news_id is None else str(failure.news_id),
        "error_type": failure.error_code,
        "error_message": failure.message,
    }
