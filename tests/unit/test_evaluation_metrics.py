from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

from src.evaluation.domain.entities import GoldEvent, GoldFinancialFact
from src.evaluation.domain.metrics import (
    EventEvaluationInput,
    FactEvaluationInput,
    evaluate_event_predictions,
    evaluate_fact_predictions,
)
from src.events.domain.entities import DetectedEvent, ExtractedFinancialFact
from src.events.domain.enums import (
    ChangeDirection,
    ComparisonType,
    Currency,
    EventType,
    FactRole,
    FactUnit,
    FinancialMetric,
    PeriodType,
    ValueScale,
)


def test_event_metrics_include_micro_macro_primary_and_confusion() -> None:
    inputs = [
        EventEvaluationInput(
            gold_events=[_gold_event(EventType.FINANCIAL_RESULTS)],
            predicted_events=[_detected_event(EventType.FINANCIAL_RESULTS)],
            gold_primary_event_type=EventType.FINANCIAL_RESULTS,
            predicted_primary_event_type=EventType.FINANCIAL_RESULTS,
            prediction_status="COMPLETE",
        ),
        EventEvaluationInput(
            gold_events=[_gold_event(EventType.DIVIDEND)],
            predicted_events=[_detected_event(EventType.GUIDANCE)],
            gold_primary_event_type=EventType.DIVIDEND,
            predicted_primary_event_type=EventType.GUIDANCE,
            prediction_status="AMBIGUOUS",
        ),
    ]

    result = evaluate_event_predictions(inputs)

    assert result.metrics["primary_accuracy"] == 0.5
    assert result.metrics["ambiguous_rate"] == 0.5
    assert result.metrics["micro"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert result.metrics["confusion_matrix"] == [
        {"gold": "DIVIDEND", "predicted": "GUIDANCE", "count": 1}
    ]
    assert {error["type"] for error in result.errors} >= {
        "MISSED_EVENT",
        "EXTRA_EVENT",
        "WRONG_PRIMARY_EVENT",
    }


def test_fact_metrics_are_strict_exact_and_report_field_errors() -> None:
    gold = _gold_fact(
        metric=FinancialMetric.REVENUE,
        value=Decimal("100"),
        currency=Currency.RUB,
        scale=ValueScale.MILLION,
        year=2025,
        role=FactRole.ACTUAL,
        start=10,
        end=13,
    )
    predicted = _predicted_fact(
        metric=FinancialMetric.REVENUE,
        value=Decimal("100"),
        currency=Currency.RUB,
        scale=ValueScale.MILLION,
        year=2025,
        role=FactRole.ACTUAL,
        start=10,
        end=13,
    )

    result = evaluate_fact_predictions([FactEvaluationInput([gold], [predicted])])

    assert result.metrics["strict"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result.metrics["semantic_strict"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result.metrics["semantic_strict_f1"] == 1.0
    assert result.metrics["value"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result.metrics["metric"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result.metrics["evidence_span_accuracy"] == 1.0
    assert result.errors == []


def test_fact_metrics_use_deterministic_one_to_one_matching() -> None:
    first_gold = _gold_fact(metric=FinancialMetric.REVENUE, value=Decimal("100"), start=5, end=8)
    second_gold = _gold_fact(metric=FinancialMetric.EBITDA, value=Decimal("20"), start=20, end=22)
    first_predicted = _predicted_fact(
        metric=FinancialMetric.EBITDA, value=Decimal("20"), start=20, end=22
    )
    second_predicted = _predicted_fact(
        metric=FinancialMetric.REVENUE, value=Decimal("100"), start=5, end=8
    )

    result = evaluate_fact_predictions(
        [FactEvaluationInput([first_gold, second_gold], [first_predicted, second_predicted])]
    )

    assert result.metrics["matched_pair_count"] == 2
    strict_metrics = cast("dict[str, float]", result.metrics["strict"])
    semantic_strict_metrics = cast("dict[str, float]", result.metrics["semantic_strict"])
    assert strict_metrics["f1"] == 1.0
    assert semantic_strict_metrics["f1"] == 1.0


def test_fact_metrics_distinguish_period_role_currency_scale_and_span_errors() -> None:
    gold = _gold_fact(
        metric=FinancialMetric.NET_PROFIT,
        value=Decimal("5"),
        currency=Currency.USD,
        scale=ValueScale.BILLION,
        year=2025,
        role=FactRole.FORECAST,
        start=30,
        end=31,
    )
    predicted = _predicted_fact(
        metric=FinancialMetric.NET_PROFIT,
        value=Decimal("5"),
        currency=Currency.RUB,
        scale=ValueScale.MILLION,
        year=2024,
        role=FactRole.ACTUAL,
        start=29,
        end=31,
    )

    result = evaluate_fact_predictions([FactEvaluationInput([gold], [predicted])])

    strict_metrics = cast("dict[str, float]", result.metrics["strict"])
    value_metrics = cast("dict[str, float]", result.metrics["value"])
    metric_metrics = cast("dict[str, float]", result.metrics["metric"])
    assert strict_metrics["f1"] == 0.0
    assert value_metrics["f1"] == 1.0
    assert metric_metrics["f1"] == 1.0
    assert {error["type"] for error in result.errors} >= {
        "WRONG_CURRENCY",
        "WRONG_SCALE",
        "WRONG_PERIOD",
        "WRONG_ROLE",
        "WRONG_EVIDENCE_SPAN",
    }


def test_fact_metrics_treat_empty_fact_sets_as_perfect_match() -> None:
    result = evaluate_fact_predictions([FactEvaluationInput([], [])])

    assert result.metrics["strict"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result.metrics["semantic_strict"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result.metrics["semantic_strict_f1"] == 1.0
    assert result.metrics["evidence_span_accuracy"] == 1.0
    assert result.errors == []


def test_fact_metrics_count_semantic_strict_without_exact_evidence_span() -> None:
    gold = _gold_fact(
        metric=FinancialMetric.REVENUE,
        value=Decimal("100"),
        start=10,
        end=13,
    )
    predicted = _predicted_fact(
        metric=FinancialMetric.REVENUE,
        value=Decimal("100"),
        start=11,
        end=14,
    )

    result = evaluate_fact_predictions([FactEvaluationInput([gold], [predicted])])

    strict_metrics = cast("dict[str, float]", result.metrics["strict"])
    semantic_strict_metrics = cast("dict[str, float]", result.metrics["semantic_strict"])
    field_accuracy = cast("dict[str, float]", result.metrics["field_accuracy"])
    assert strict_metrics["f1"] == 0.0
    assert semantic_strict_metrics["f1"] == 1.0
    assert result.metrics["semantic_strict_f1"] == 1.0
    assert result.metrics["evidence_span_accuracy"] == 0.0
    assert "evidence_span" not in field_accuracy
    assert {error["type"] for error in result.errors} == {"WRONG_EVIDENCE_SPAN"}


def test_fact_metrics_include_change_value_and_unit() -> None:
    gold = _gold_fact(
        metric=FinancialMetric.REVENUE,
        value=Decimal("15"),
        change_value=Decimal("15"),
        change_unit=FactUnit.PERCENT,
    )
    predicted = _predicted_fact(
        metric=FinancialMetric.REVENUE,
        value=Decimal("15"),
        change_value=Decimal("12"),
        change_unit=FactUnit.MONEY,
    )

    result = evaluate_fact_predictions([FactEvaluationInput([gold], [predicted])])

    semantic_strict_metrics = cast("dict[str, float]", result.metrics["semantic_strict"])
    field_accuracy = cast("dict[str, float]", result.metrics["field_accuracy"])
    assert semantic_strict_metrics["f1"] == 0.0
    assert field_accuracy["change_value"] == 0.0
    assert field_accuracy["change_unit"] == 0.0
    assert {error["type"] for error in result.errors} >= {
        "WRONG_CHANGE_VALUE",
        "WRONG_CHANGE_UNIT",
    }


def _gold_event(event_type: EventType) -> GoldEvent:
    return GoldEvent(
        event_type=event_type,
        evidence_text="evidence",
        start_position=0,
        end_position=8,
        is_primary=True,
    )


def _detected_event(event_type: EventType) -> DetectedEvent:
    return DetectedEvent(
        id=uuid4(),
        analysis_id=UUID(int=0),
        event_type=event_type,
        confidence=Decimal("0.90"),
        rule_id="rule",
        matched_rule="rule",
        evidence_text="evidence",
        start_position=0,
        end_position=8,
    )


def _gold_fact(
    *,
    metric: FinancialMetric,
    value: Decimal,
    currency: Currency = Currency.RUB,
    scale: ValueScale = ValueScale.MILLION,
    year: int | None = 2025,
    role: FactRole = FactRole.ACTUAL,
    start: int = 10,
    end: int = 13,
    change_value: Decimal | None = None,
    change_unit: FactUnit | None = None,
) -> GoldFinancialFact:
    return GoldFinancialFact(
        metric=metric,
        raw_value=value,
        normalized_value=value,
        unit=FactUnit.MONEY,
        currency=currency,
        scale=scale,
        period_type=PeriodType.YEAR,
        period_year=year,
        period_quarter=None,
        period_month=None,
        raw_period=None if year is None else str(year),
        fact_role=role,
        comparison_type=ComparisonType.NONE,
        change_direction=ChangeDirection.UNCHANGED if change_value is None else ChangeDirection.UP,
        change_value=change_value,
        change_unit=change_unit,
        evidence_text=str(value),
        start_position=start,
        end_position=end,
    )


def _predicted_fact(
    *,
    metric: FinancialMetric,
    value: Decimal,
    currency: Currency = Currency.RUB,
    scale: ValueScale = ValueScale.MILLION,
    year: int | None = 2025,
    role: FactRole = FactRole.ACTUAL,
    start: int = 10,
    end: int = 13,
    change_value: Decimal | None = None,
    change_unit: FactUnit | None = None,
) -> ExtractedFinancialFact:
    return ExtractedFinancialFact(
        id=uuid4(),
        analysis_id=UUID(int=0),
        metric=metric,
        raw_value=value,
        normalized_value=value,
        unit=FactUnit.MONEY,
        currency=currency,
        scale=scale,
        period_type=PeriodType.YEAR,
        year=year,
        quarter=None,
        month=None,
        date_from=date(2025, 1, 1) if year == 2025 else None,
        date_to=date(2025, 12, 31) if year == 2025 else None,
        raw_period=None if year is None else str(year),
        comparison_type=ComparisonType.NONE,
        fact_role=role,
        change_direction=ChangeDirection.UNCHANGED if change_value is None else ChangeDirection.UP,
        change_value=change_value,
        change_unit=change_unit,
        confidence=Decimal("0.91"),
        rule_id="metric.test",
        evidence_text=str(value),
        start_position=start,
        end_position=end,
        extractor_version="financial-facts-v1",
        matched_rule="metric.test",
    )
