from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from src.evaluation.domain.entities import GoldEvent, GoldFinancialFact
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

DEVELOPMENT_DATASET_NAME = "ru-corporate-events-real-batch-004-development-gold-v1"
DEVELOPMENT_SCHEMA_VERSION = "real-development-gold-v1"
EXPECTED_SPLIT_SHA256 = "a32956626d194158eb69869f6bdca510456ded47ac5810ca91fe90b86aa45dea"
EXPECTED_EVENT_DISTRIBUTION = {
    "DIVIDEND": 1,
    "FINANCIAL_RESULTS": 4,
    "MANAGEMENT_CHANGE": 1,
    "OTHER": 3,
    "SANCTIONS": 1,
}
FUTURE_OR_PREDICTION_FIELDS = frozenset(
    {
        "abnormal_return",
        "future_price",
        "future_return",
        "future_volume",
        "market_reaction",
        "qwen_prediction",
        "rules_prediction",
    }
)


@dataclass(frozen=True, slots=True)
class DevelopmentGoldRecord:
    news_id: UUID
    annotation_text: str
    raw_content_hash: str
    primary_event: EventType
    events: tuple[GoldEvent, ...]
    facts: tuple[GoldFinancialFact, ...]
    source_payload: dict[str, Any]

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "annotation_text": self.annotation_text,
            "dataset_name": DEVELOPMENT_DATASET_NAME,
            "gold_events": [
                {
                    "end_position": item.end_position,
                    "event_type": item.event_type.value,
                    "evidence_text": item.evidence_text,
                    "is_primary": item.is_primary,
                    "notes": item.notes,
                    "start_position": item.start_position,
                }
                for item in self.events
            ],
            "gold_financial_facts": [_fact_payload(item) for item in self.facts],
            "gold_primary_event": self.primary_event.value,
            "news_id": str(self.news_id),
            "provenance": "REAL",
            "published_at": self.source_payload["published_at"],
            "purpose": "DEVELOPMENT",
            "raw_content_hash": self.raw_content_hash,
            "review_basis": "EXCERPT_ONLY",
            "schema_version": DEVELOPMENT_SCHEMA_VERSION,
            "source": self.source_payload["source"],
            "source_item_id": self.source_payload["source_item_id"],
            "ticker": self.source_payload["ticker"],
        }


@dataclass(frozen=True, slots=True)
class DevelopmentGoldDataset:
    records: tuple[DevelopmentGoldRecord, ...]
    source_review_sha256: str
    dataset_sha256: str
    split_sha256: str
    holdout_count: int

    @property
    def event_distribution(self) -> dict[str, int]:
        return dict(sorted(Counter(item.primary_event.value for item in self.records).items()))


def freeze_development_gold(
    *, source_review_path: Path, split_manifest_path: Path, output_directory: Path
) -> DevelopmentGoldDataset:
    split_sha, development_ids, holdout_count = load_split_metadata(split_manifest_path)
    source_bytes = source_review_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    records = _load_review_records(source_review_path)
    record_ids = {item.news_id for item in records}
    if record_ids != development_ids:
        raise ValueError("human review records do not exactly match frozen DEVELOPMENT ids")
    canonical_rows = [item.canonical_payload() for item in records]
    canonical_text = "".join(stable_json(item) + "\n" for item in canonical_rows)
    dataset_sha = hashlib.sha256(canonical_text.encode()).hexdigest()
    dataset = DevelopmentGoldDataset(
        records=records,
        source_review_sha256=source_sha,
        dataset_sha256=dataset_sha,
        split_sha256=split_sha,
        holdout_count=holdout_count,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "dataset.jsonl").write_text(canonical_text, encoding="utf-8")
    manifest = {
        "canonical_dataset_sha256": dataset_sha,
        "dataset_name": DEVELOPMENT_DATASET_NAME,
        "event_distribution": dataset.event_distribution,
        "frozen_before_model_tuning": True,
        "holdout_count_metadata_only": holdout_count,
        "provenance": "REAL",
        "purpose": "DEVELOPMENT",
        "records": len(records),
        "review_basis": "EXCERPT_ONLY",
        "schema_version": DEVELOPMENT_SCHEMA_VERSION,
        "source_review_sha256": source_sha,
        "split_sha256": split_sha,
    }
    _write_json(output_directory / "manifest.json", manifest)
    return dataset


def load_split_metadata(path: Path) -> tuple[str, set[UUID], int]:
    payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    split_sha = str(payload.get("split_sha256", ""))
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise ValueError("frozen split SHA does not match Batch 004")
    assignments = cast("list[dict[str, Any]]", payload.get("assignments", []))
    development_ids = {
        UUID(str(item["news_id"])) for item in assignments if item.get("split") == "DEVELOPMENT"
    }
    holdout_count = sum(item.get("split") == "FRESH_HOLDOUT" for item in assignments)
    if len(development_ids) != 10 or holdout_count != 4:
        raise ValueError("frozen split metadata must contain DEVELOPMENT=10 and FRESH_HOLDOUT=4")
    return split_sha, development_ids, holdout_count


def stable_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _load_review_records(path: Path) -> tuple[DevelopmentGoldRecord, ...]:
    payloads = [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(payloads) != 10:
        raise ValueError("DEVELOPMENT human review must contain exactly 10 records")
    records = tuple(_review_record(item) for item in payloads)
    distribution = dict(sorted(Counter(item.primary_event.value for item in records).items()))
    if distribution != EXPECTED_EVENT_DISTRIBUTION:
        raise ValueError("DEVELOPMENT event distribution does not match reviewed batch")
    if sum(len(item.facts) for item in records) != 1:
        raise ValueError("DEVELOPMENT review must contain exactly one explicit fact")
    return records


def _review_record(payload: dict[str, Any]) -> DevelopmentGoldRecord:
    leaked = FUTURE_OR_PREDICTION_FIELDS.intersection(payload)
    if leaked:
        raise ValueError(f"DEVELOPMENT review contains prohibited fields: {sorted(leaked)}")
    if payload.get("human_review_status") != "REVIEWED":
        raise ValueError("every DEVELOPMENT record must be REVIEWED")
    if payload.get("human_review_basis") != "annotation_text_excerpt_only":
        raise ValueError("every DEVELOPMENT record must use excerpt-only review")
    if payload.get("is_gold") is not False:
        raise ValueError("source human review must remain non-gold")
    text = str(payload["annotation_text"])
    expected_hash = hashlib.sha256(text.encode()).hexdigest()
    if payload.get("raw_content_hash") != expected_hash:
        raise ValueError("review annotation text hash mismatch")
    primary = EventType(str(payload["human_primary_event"]))
    event_types = tuple(EventType(str(item)) for item in cast("list[str]", payload["human_events"]))
    if not event_types or primary not in event_types:
        raise ValueError("review requires a primary event present in human_events")
    events = tuple(
        GoldEvent(
            event_type=item,
            evidence_text=text,
            start_position=0,
            end_position=len(text),
            is_primary=item == primary,
            notes=str(payload.get("human_review_notes") or "") or None,
        )
        for item in event_types
    )
    facts = tuple(
        _gold_fact(text, item)
        for item in cast("list[dict[str, Any]]", payload["human_financial_facts"])
    )
    return DevelopmentGoldRecord(
        news_id=UUID(str(payload["news_id"])),
        annotation_text=text,
        raw_content_hash=expected_hash,
        primary_event=primary,
        events=events,
        facts=facts,
        source_payload=payload,
    )


def _gold_fact(text: str, payload: dict[str, Any]) -> GoldFinancialFact:
    if payload.get("metric") != "DIVIDEND_PER_SHARE":
        raise ValueError("unexpected DEVELOPMENT financial fact metric")
    value = Decimal(str(payload["value"]))
    needle = re_escape_amount(str(payload["value"]))
    match = re.search(rf"{needle}\s+(?:roubles?|rub)\s+per\s+share\b", text, re.IGNORECASE)
    if match is None:
        raise ValueError("reviewed dividend fact lacks exact excerpt evidence")
    period_text = str(payload.get("period_text") or "")
    year = int(period_text) if period_text.isdigit() else None
    return GoldFinancialFact(
        metric=FinancialMetric.DIVIDEND_PER_SHARE,
        raw_value=value,
        normalized_value=value,
        unit=FactUnit(str(payload["unit"])),
        currency=Currency(str(payload["currency"])),
        scale=ValueScale(str(payload["scale"])),
        period_type=PeriodType.YEAR if year is not None else PeriodType.UNKNOWN,
        period_year=year,
        period_quarter=None,
        period_month=None,
        raw_period=period_text or None,
        fact_role=FactRole(str(payload["role"])),
        comparison_type=ComparisonType(str(payload["comparison_type"])),
        change_direction=ChangeDirection(str(payload["change_direction"])),
        change_value=None,
        change_unit=None,
        evidence_text=match.group(0),
        start_position=match.start(),
        end_position=match.end(),
        notes=str(payload.get("notes") or "") or None,
    )


def re_escape_amount(value: str) -> str:
    return re.escape(value).replace(r"\.", r"[.,]")


def _fact_payload(item: GoldFinancialFact) -> dict[str, Any]:
    return {
        "change_direction": item.change_direction.value,
        "change_unit": None if item.change_unit is None else item.change_unit.value,
        "change_value": None if item.change_value is None else str(item.change_value),
        "comparison_type": item.comparison_type.value,
        "currency": item.currency.value,
        "end_position": item.end_position,
        "evidence_text": item.evidence_text,
        "fact_role": item.fact_role.value,
        "metric": item.metric.value,
        "normalized_value": str(item.normalized_value),
        "notes": item.notes,
        "period_month": item.period_month,
        "period_quarter": item.period_quarter,
        "period_type": item.period_type.value,
        "period_year": item.period_year,
        "raw_period": item.raw_period,
        "raw_value": str(item.raw_value),
        "scale": item.scale.value,
        "start_position": item.start_position,
        "unit": item.unit.value,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
