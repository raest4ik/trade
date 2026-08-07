from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import cast
from uuid import UUID

from src.evaluation.domain.entities import AnnotationExample, GoldEvent, GoldFinancialFact
from src.evaluation.domain.enums import DatasetSplit, ReviewStatus
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
from src.news.domain.time import ensure_aware_utc


def annotation_to_json(example: AnnotationExample) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": example.schema_version,
        "news_id": str(example.news_id),
        "published_at": example.published_at.isoformat().replace("+00:00", "Z"),
        "raw_content_hash": example.raw_content_hash,
        "split": example.split.value,
        "review_status": example.review_status.value,
        "annotator": example.annotator,
        "notes": example.notes,
        "predicted_events": example.predicted_events,
        "predicted_financial_facts": example.predicted_financial_facts,
        "gold_events": [gold_event_to_json(item) for item in example.gold_events],
        "gold_financial_facts": [gold_fact_to_json(item) for item in example.gold_financial_facts],
    }
    if example.raw_content is not None:
        payload["raw_content"] = example.raw_content
    return payload


def gold_event_to_json(event: GoldEvent) -> dict[str, object]:
    return {
        "event_type": event.event_type.value,
        "evidence_text": event.evidence_text,
        "start_position": event.start_position,
        "end_position": event.end_position,
        "is_primary": event.is_primary,
        "notes": event.notes,
    }


def gold_fact_to_json(fact: GoldFinancialFact) -> dict[str, object]:
    return {
        "metric": fact.metric.value,
        "raw_value": str(fact.raw_value),
        "normalized_value": str(fact.normalized_value),
        "unit": fact.unit.value,
        "currency": fact.currency.value,
        "scale": fact.scale.value,
        "period_type": fact.period_type.value,
        "period_year": fact.period_year,
        "period_quarter": fact.period_quarter,
        "period_month": fact.period_month,
        "raw_period": fact.raw_period,
        "fact_role": fact.fact_role.value,
        "comparison_type": fact.comparison_type.value,
        "change_direction": fact.change_direction.value,
        "change_value": None if fact.change_value is None else str(fact.change_value),
        "change_unit": None if fact.change_unit is None else fact.change_unit.value,
        "evidence_text": fact.evidence_text,
        "start_position": fact.start_position,
        "end_position": fact.end_position,
        "notes": fact.notes,
    }


def annotation_from_json(payload: dict[str, object]) -> AnnotationExample:
    return AnnotationExample(
        schema_version=str(payload["schema_version"]),
        news_id=UUID(str(payload["news_id"])),
        published_at=ensure_aware_utc(
            _parse_datetime(str(payload["published_at"])), "published_at"
        ),
        raw_content_hash=str(payload["raw_content_hash"]),
        split=DatasetSplit(str(payload["split"])),
        review_status=ReviewStatus(str(payload["review_status"])),
        annotator=str(payload["annotator"]),
        notes=None if payload.get("notes") is None else str(payload.get("notes")),
        predicted_events=_list_of_dicts(payload.get("predicted_events", [])),
        predicted_financial_facts=_list_of_dicts(payload.get("predicted_financial_facts", [])),
        gold_events=[
            gold_event_from_json(item) for item in _list_of_dicts(payload.get("gold_events", []))
        ],
        gold_financial_facts=[
            gold_fact_from_json(item)
            for item in _list_of_dicts(payload.get("gold_financial_facts", []))
        ],
        raw_content=None if payload.get("raw_content") is None else str(payload.get("raw_content")),
    )


def gold_event_from_json(payload: dict[str, object]) -> GoldEvent:
    return GoldEvent(
        event_type=EventType(str(payload["event_type"])),
        evidence_text=str(payload["evidence_text"]),
        start_position=_required_int(payload, "start_position"),
        end_position=_required_int(payload, "end_position"),
        is_primary=bool(payload.get("is_primary", False)),
        notes=None if payload.get("notes") is None else str(payload.get("notes")),
    )


def gold_fact_from_json(payload: dict[str, object]) -> GoldFinancialFact:
    return GoldFinancialFact(
        metric=FinancialMetric(str(payload["metric"])),
        raw_value=Decimal(str(payload["raw_value"])),
        normalized_value=Decimal(str(payload["normalized_value"])),
        unit=FactUnit(str(payload["unit"])),
        currency=Currency(str(payload["currency"])),
        scale=ValueScale(str(payload["scale"])),
        period_type=PeriodType(str(payload["period_type"])),
        period_year=_optional_int(payload.get("period_year")),
        period_quarter=_optional_int(payload.get("period_quarter")),
        period_month=_optional_int(payload.get("period_month")),
        raw_period=None if payload.get("raw_period") is None else str(payload.get("raw_period")),
        fact_role=FactRole(str(payload["fact_role"])),
        comparison_type=ComparisonType(str(payload["comparison_type"])),
        change_direction=ChangeDirection(str(payload["change_direction"])),
        change_value=None
        if payload.get("change_value") is None
        else Decimal(str(payload["change_value"])),
        change_unit=None
        if payload.get("change_unit") is None
        else FactUnit(str(payload["change_unit"])),
        evidence_text=str(payload["evidence_text"]),
        start_position=_required_int(payload, "start_position"),
        end_position=_required_int(payload, "end_position"),
        notes=None if payload.get("notes") is None else str(payload.get("notes")),
    )


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _optional_int(value: object) -> int | None:
    return None if value is None else int(str(value))


def _required_int(payload: dict[str, object], key: str) -> int:
    return int(str(payload[key]))


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    result: list[dict[str, object]] = []
    for item in cast("list[object]", value):
        if not isinstance(item, dict):
            raise ValueError("expected object item")
        typed = cast("dict[object, object]", item)
        result.append({str(key): val for key, val in typed.items()})
    return result
