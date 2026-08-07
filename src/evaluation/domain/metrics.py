from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from src.evaluation.domain.entities import GoldEvent, GoldFinancialFact
from src.evaluation.domain.enums import EvaluationErrorType
from src.events.domain.entities import DetectedEvent, ExtractedFinancialFact
from src.events.domain.enums import EventType


@dataclass(frozen=True, slots=True)
class EventEvaluationInput:
    gold_events: Sequence[GoldEvent]
    predicted_events: Sequence[DetectedEvent]
    gold_primary_event_type: EventType | None
    predicted_primary_event_type: EventType | None
    prediction_status: str


@dataclass(frozen=True, slots=True)
class FactEvaluationInput:
    gold_facts: Sequence[GoldFinancialFact]
    predicted_facts: Sequence[ExtractedFinancialFact]


@dataclass(frozen=True, slots=True)
class EvaluationMetricResult:
    metrics: dict[str, object]
    errors: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class NormalizedFact:
    metric: str
    normalized_value: Decimal
    unit: str
    currency: str
    scale: str
    period_type: str
    period_year: int | None
    period_quarter: int | None
    period_month: int | None
    fact_role: str
    comparison_type: str
    change_direction: str
    evidence_text: str
    start_position: int
    end_position: int


@dataclass(frozen=True, slots=True)
class FactMatch:
    gold_index: int
    prediction_index: int
    strict: bool
    value: bool
    metric: bool
    evidence_overlap: Decimal


def evaluate_event_predictions(
    examples: Sequence[EventEvaluationInput],
) -> EvaluationMetricResult:
    labels = sorted(
        {
            event_type.value
            for example in examples
            for event_type in (
                list(_gold_event_types(example.gold_events))
                + list(_predicted_event_types(example.predicted_events))
            )
        }
    )
    true_positive = Counter[str]()
    false_positive = Counter[str]()
    false_negative = Counter[str]()
    confusion = Counter[tuple[str, str]]()
    primary_correct = 0
    known_count = 0
    ambiguous_count = 0
    errors: list[dict[str, object]] = []

    for index, example in enumerate(examples):
        gold_types = _gold_event_types(example.gold_events)
        predicted_types = _predicted_event_types(example.predicted_events)
        for event_type in gold_types & predicted_types:
            true_positive[event_type.value] += 1
        for event_type in predicted_types - gold_types:
            false_positive[event_type.value] += 1
            errors.append(
                {
                    "example_index": index,
                    "type": EvaluationErrorType.EXTRA_EVENT.value,
                    "predicted_event_type": event_type.value,
                }
            )
        for event_type in gold_types - predicted_types:
            false_negative[event_type.value] += 1
            errors.append(
                {
                    "example_index": index,
                    "type": EvaluationErrorType.MISSED_EVENT.value,
                    "gold_event_type": event_type.value,
                }
            )
        gold_primary = example.gold_primary_event_type
        predicted_primary = example.predicted_primary_event_type
        if gold_primary is not None:
            known_count += 1
            if gold_primary == predicted_primary:
                primary_correct += 1
            else:
                errors.append(
                    {
                        "example_index": index,
                        "type": EvaluationErrorType.WRONG_PRIMARY_EVENT.value,
                        "gold_event_type": gold_primary.value,
                        "predicted_event_type": None
                        if predicted_primary is None
                        else predicted_primary.value,
                    }
                )
                if predicted_primary is not None:
                    confusion[(gold_primary.value, predicted_primary.value)] += 1
        if "AMBIGUOUS" in example.prediction_status:
            ambiguous_count += 1

    total_tp = sum(true_positive.values())
    total_fp = sum(false_positive.values())
    total_fn = sum(false_negative.values())
    per_class = {
        label: _prf(
            true_positive[label],
            false_positive[label],
            false_negative[label],
        )
        | {"support": true_positive[label] + false_negative[label]}
        for label in labels
    }
    macro_f1 = _mean(
        [float(values["f1"]) for values in per_class.values() if values["support"] > 0]
    )
    metrics: dict[str, object] = {
        "example_count": len(examples),
        "micro": _prf(total_tp, total_fp, total_fn),
        "macro_f1": macro_f1,
        "per_class": per_class,
        "primary_accuracy": _safe_ratio(primary_correct, known_count),
        "coverage": _safe_ratio(
            sum(1 for example in examples if example.predicted_events),
            len(examples),
        ),
        "unknown_rate": _safe_ratio(
            sum(
                1
                for example in examples
                if example.predicted_primary_event_type == EventType.UNKNOWN
            ),
            len(examples),
        ),
        "ambiguous_rate": _safe_ratio(ambiguous_count, len(examples)),
        "confusion_matrix": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in sorted(confusion.items())
        ],
    }
    return EvaluationMetricResult(metrics=metrics, errors=errors)


def evaluate_fact_predictions(
    examples: Sequence[FactEvaluationInput],
) -> EvaluationMetricResult:
    strict_tp = 0
    value_tp = 0
    metric_tp = 0
    gold_total = 0
    prediction_total = 0
    matched_pairs = 0
    field_totals = Counter[str]()
    field_matches = Counter[str]()
    errors: list[dict[str, object]] = []

    for example_index, example in enumerate(examples):
        gold = [_gold_fact(item) for item in example.gold_facts]
        predicted = [_predicted_fact(item) for item in example.predicted_facts]
        gold_total += len(gold)
        prediction_total += len(predicted)
        matches = _match_facts(gold, predicted)
        matched_pairs += len(matches)
        matched_gold = {match.gold_index for match in matches}
        matched_predictions = {match.prediction_index for match in matches}
        strict_tp += sum(1 for match in matches if match.strict)
        value_tp += sum(1 for match in matches if match.value)
        metric_tp += sum(1 for match in matches if match.metric)
        for match in matches:
            gold_fact = gold[match.gold_index]
            predicted_fact = predicted[match.prediction_index]
            _count_fact_fields(field_totals, field_matches, gold_fact, predicted_fact)
            errors.extend(_fact_errors(example_index, gold_fact, predicted_fact, match))
        for gold_index in sorted(set(range(len(gold))) - matched_gold):
            errors.append(
                {
                    "example_index": example_index,
                    "type": EvaluationErrorType.MISSED_FACT.value,
                    "gold_index": gold_index,
                    "metric": gold[gold_index].metric,
                }
            )
        for prediction_index in sorted(set(range(len(predicted))) - matched_predictions):
            errors.append(
                {
                    "example_index": example_index,
                    "type": EvaluationErrorType.EXTRA_FACT.value,
                    "prediction_index": prediction_index,
                    "metric": predicted[prediction_index].metric,
                }
            )

    metrics: dict[str, object] = {
        "example_count": len(examples),
        "gold_fact_count": gold_total,
        "predicted_fact_count": prediction_total,
        "matched_pair_count": matched_pairs,
        "strict": _binary_metrics(strict_tp, prediction_total, gold_total),
        "value": _binary_metrics(value_tp, prediction_total, gold_total),
        "metric": _binary_metrics(metric_tp, prediction_total, gold_total),
        "field_accuracy": {
            field: _safe_ratio(field_matches[field], total)
            for field, total in sorted(field_totals.items())
        },
    }
    return EvaluationMetricResult(metrics=metrics, errors=errors)


def _gold_event_types(events: Sequence[GoldEvent]) -> set[EventType]:
    return {event.event_type for event in events}


def _predicted_event_types(events: Sequence[DetectedEvent]) -> set[EventType]:
    return {event.event_type for event in events}


def _gold_fact(fact: GoldFinancialFact) -> NormalizedFact:
    return NormalizedFact(
        metric=fact.metric.value,
        normalized_value=fact.normalized_value,
        unit=fact.unit.value,
        currency=fact.currency.value,
        scale=fact.scale.value,
        period_type=fact.period_type.value,
        period_year=fact.period_year,
        period_quarter=fact.period_quarter,
        period_month=fact.period_month,
        fact_role=fact.fact_role.value,
        comparison_type=fact.comparison_type.value,
        change_direction=fact.change_direction.value,
        evidence_text=fact.evidence_text,
        start_position=fact.start_position,
        end_position=fact.end_position,
    )


def _predicted_fact(fact: ExtractedFinancialFact) -> NormalizedFact:
    return NormalizedFact(
        metric=fact.metric.value,
        normalized_value=fact.normalized_value,
        unit=fact.unit.value,
        currency=fact.currency.value,
        scale=fact.scale.value,
        period_type=fact.period_type.value,
        period_year=fact.year,
        period_quarter=fact.quarter,
        period_month=fact.month,
        fact_role=fact.fact_role.value,
        comparison_type=fact.comparison_type.value,
        change_direction=fact.change_direction.value,
        evidence_text=fact.evidence_text,
        start_position=fact.start_position,
        end_position=fact.end_position,
    )


def _match_facts(
    gold: Sequence[NormalizedFact],
    predicted: Sequence[NormalizedFact],
) -> list[FactMatch]:
    candidates = {
        (gold_index, prediction_index): _fact_score(gold_fact, predicted_fact)
        for gold_index, gold_fact in enumerate(gold)
        for prediction_index, predicted_fact in enumerate(predicted)
    }
    cache: dict[tuple[int, int], tuple[int, tuple[tuple[int, int], ...]]] = {}

    def solve(gold_index: int, used_mask: int) -> tuple[int, tuple[tuple[int, int], ...]]:
        key = (gold_index, used_mask)
        if key in cache:
            return cache[key]
        if gold_index >= len(gold):
            return 0, ()
        best_score, best_pairs = solve(gold_index + 1, used_mask)
        for prediction_index in range(len(predicted)):
            bit = 1 << prediction_index
            if used_mask & bit:
                continue
            score = candidates[(gold_index, prediction_index)]
            if score <= 0:
                continue
            next_score, next_pairs = solve(gold_index + 1, used_mask | bit)
            candidate_score = score + next_score
            candidate_pairs = ((gold_index, prediction_index), *next_pairs)
            if _better_match(candidate_score, candidate_pairs, best_score, best_pairs):
                best_score = candidate_score
                best_pairs = candidate_pairs
        cache[key] = best_score, best_pairs
        return cache[key]

    _, pairs = solve(0, 0)
    matches: list[FactMatch] = []
    for gold_index, prediction_index in pairs:
        gold_fact = gold[gold_index]
        predicted_fact = predicted[prediction_index]
        matches.append(
            FactMatch(
                gold_index=gold_index,
                prediction_index=prediction_index,
                strict=_strict_fact_match(gold_fact, predicted_fact),
                value=_value_match(gold_fact, predicted_fact),
                metric=gold_fact.metric == predicted_fact.metric,
                evidence_overlap=_span_overlap(gold_fact, predicted_fact),
            )
        )
    return sorted(matches, key=lambda item: (item.gold_index, item.prediction_index))


def _better_match(
    candidate_score: int,
    candidate_pairs: tuple[tuple[int, int], ...],
    best_score: int,
    best_pairs: tuple[tuple[int, int], ...],
) -> bool:
    if candidate_score != best_score:
        return candidate_score > best_score
    if len(candidate_pairs) != len(best_pairs):
        return len(candidate_pairs) > len(best_pairs)
    return candidate_pairs < best_pairs


def _fact_score(gold: NormalizedFact, predicted: NormalizedFact) -> int:
    overlap = int(_span_overlap(gold, predicted) * Decimal("100"))
    score = 0
    if gold.metric == predicted.metric:
        score += 1000
    if _value_match(gold, predicted):
        score += 700
    if gold.unit == predicted.unit:
        score += 150
    if gold.currency == predicted.currency:
        score += 120
    if gold.scale == predicted.scale:
        score += 80
    score += overlap
    return score if score >= 700 or overlap > 0 else 0


def _strict_fact_match(gold: NormalizedFact, predicted: NormalizedFact) -> bool:
    return (
        gold.metric == predicted.metric
        and _value_match(gold, predicted)
        and gold.unit == predicted.unit
        and gold.currency == predicted.currency
        and gold.scale == predicted.scale
        and gold.period_type == predicted.period_type
        and gold.period_year == predicted.period_year
        and gold.period_quarter == predicted.period_quarter
        and gold.period_month == predicted.period_month
        and gold.fact_role == predicted.fact_role
        and gold.comparison_type == predicted.comparison_type
        and gold.change_direction == predicted.change_direction
        and gold.start_position == predicted.start_position
        and gold.end_position == predicted.end_position
    )


def _value_match(gold: NormalizedFact, predicted: NormalizedFact) -> bool:
    return gold.normalized_value == predicted.normalized_value


def _span_overlap(gold: NormalizedFact, predicted: NormalizedFact) -> Decimal:
    left = max(gold.start_position, predicted.start_position)
    right = min(gold.end_position, predicted.end_position)
    overlap = max(0, right - left)
    denominator = max(
        gold.end_position - gold.start_position, predicted.end_position - predicted.start_position
    )
    if denominator <= 0:
        return Decimal("0")
    return Decimal(overlap) / Decimal(denominator)


def _count_fact_fields(
    totals: Counter[str],
    matches: Counter[str],
    gold: NormalizedFact,
    predicted: NormalizedFact,
) -> None:
    fields = (
        "metric",
        "normalized_value",
        "unit",
        "currency",
        "scale",
        "period_type",
        "period_year",
        "period_quarter",
        "period_month",
        "fact_role",
        "comparison_type",
        "change_direction",
    )
    for field in fields:
        totals[field] += 1
        if getattr(gold, field) == getattr(predicted, field):
            matches[field] += 1
    totals["evidence_span"] += 1
    if (
        gold.start_position == predicted.start_position
        and gold.end_position == predicted.end_position
    ):
        matches["evidence_span"] += 1


def _fact_errors(
    example_index: int,
    gold: NormalizedFact,
    predicted: NormalizedFact,
    match: FactMatch,
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    checks: tuple[tuple[str, bool], ...] = (
        (EvaluationErrorType.WRONG_METRIC.value, gold.metric == predicted.metric),
        (EvaluationErrorType.WRONG_VALUE.value, _value_match(gold, predicted)),
        (EvaluationErrorType.WRONG_CURRENCY.value, gold.currency == predicted.currency),
        (EvaluationErrorType.WRONG_SCALE.value, gold.scale == predicted.scale),
        (EvaluationErrorType.WRONG_PERIOD.value, _period_match(gold, predicted)),
        (EvaluationErrorType.WRONG_ROLE.value, gold.fact_role == predicted.fact_role),
        (
            EvaluationErrorType.WRONG_CHANGE_DIRECTION.value,
            gold.change_direction == predicted.change_direction,
        ),
        (
            EvaluationErrorType.WRONG_EVIDENCE_SPAN.value,
            gold.start_position == predicted.start_position
            and gold.end_position == predicted.end_position,
        ),
    )
    for error_type, ok in checks:
        if not ok:
            errors.append(
                {
                    "example_index": example_index,
                    "type": error_type,
                    "gold_metric": gold.metric,
                    "predicted_metric": predicted.metric,
                    "evidence_overlap": str(match.evidence_overlap),
                }
            )
    return errors


def _period_match(gold: NormalizedFact, predicted: NormalizedFact) -> bool:
    return (
        gold.period_type == predicted.period_type
        and gold.period_year == predicted.period_year
        and gold.period_quarter == predicted.period_quarter
        and gold.period_month == predicted.period_month
    )


def _binary_metrics(true_positive: int, predicted_total: int, gold_total: int) -> dict[str, float]:
    precision = _safe_ratio(true_positive, predicted_total)
    recall = _safe_ratio(true_positive, gold_total)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 6)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)
