from __future__ import annotations

from collections.abc import Sequence

from src.ai_events.application.use_cases import AIEventAnalysisResult
from src.evaluation.domain.entities import GoldEvent, GoldFinancialFact
from src.evaluation.domain.metrics import EventEvaluationInput, FactEvaluationInput


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
