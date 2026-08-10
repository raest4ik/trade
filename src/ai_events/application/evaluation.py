from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.ai_events.application.use_cases import AIEventAnalysisResult
from src.evaluation.domain.entities import GoldEvent, GoldFinancialFact
from src.evaluation.domain.metrics import (
    EvaluationMetricResult,
    EventEvaluationInput,
    FactEvaluationInput,
    evaluate_event_predictions,
    evaluate_fact_predictions,
)


@dataclass(frozen=True, slots=True)
class AIEvaluationMetricViews:
    successful_events: EvaluationMetricResult
    successful_facts: EvaluationMetricResult
    end_to_end_events: EvaluationMetricResult
    end_to_end_facts: EvaluationMetricResult
    requested_count: int
    successful_count: int
    failed_count: int
    item_success_rate: float


def evaluate_ai_metric_views(
    *,
    successful_event_inputs: Sequence[EventEvaluationInput],
    successful_fact_inputs: Sequence[FactEvaluationInput],
    failed_event_inputs: Sequence[EventEvaluationInput],
    failed_fact_inputs: Sequence[FactEvaluationInput],
) -> AIEvaluationMetricViews:
    if len(successful_event_inputs) != len(successful_fact_inputs):
        raise ValueError("successful event/fact input counts must match")
    if len(failed_event_inputs) != len(failed_fact_inputs):
        raise ValueError("failed event/fact input counts must match")
    successful_count = len(successful_event_inputs)
    failed_count = len(failed_event_inputs)
    requested_count = successful_count + failed_count
    return AIEvaluationMetricViews(
        successful_events=evaluate_event_predictions(successful_event_inputs),
        successful_facts=evaluate_fact_predictions(successful_fact_inputs),
        end_to_end_events=evaluate_event_predictions(
            [*successful_event_inputs, *failed_event_inputs]
        ),
        end_to_end_facts=evaluate_fact_predictions([*successful_fact_inputs, *failed_fact_inputs]),
        requested_count=requested_count,
        successful_count=successful_count,
        failed_count=failed_count,
        item_success_rate=(
            1.0 if requested_count == 0 else round(successful_count / requested_count, 6)
        ),
    )


def to_evaluation_inputs(
    *,
    gold_events: Sequence[GoldEvent],
    gold_facts: Sequence[GoldFinancialFact],
    prediction: AIEventAnalysisResult,
) -> tuple[EventEvaluationInput, FactEvaluationInput]:
    gold_primary = next(
        (event.event_type for event in gold_events if event.is_primary),
        next((event.event_type for event in gold_events), None),
    )
    analysis = prediction.analysis
    return (
        EventEvaluationInput(
            gold_events=gold_events,
            predicted_events=analysis.events,
            gold_primary_event_type=gold_primary,
            predicted_primary_event_type=analysis.primary_event_type,
            prediction_status=analysis.status.value,
        ),
        FactEvaluationInput(
            gold_facts=gold_facts,
            predicted_facts=analysis.financial_facts,
        ),
    )
