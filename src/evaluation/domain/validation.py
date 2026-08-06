from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import cast
from uuid import UUID

from src.evaluation.domain.entities import GOLD_SCHEMA_VERSION
from src.evaluation.domain.enums import DatasetSplit, ReviewStatus
from src.evaluation.domain.serialization import annotation_from_json

ALLOWED_FIELDS = {
    "schema_version",
    "news_id",
    "published_at",
    "raw_content_hash",
    "split",
    "review_status",
    "annotator",
    "notes",
    "predicted_events",
    "predicted_financial_facts",
    "gold_events",
    "gold_financial_facts",
    "raw_content",
}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    line_number: int
    code: str
    message: str
    news_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_jsonl_payloads(
    lines: list[str],
    *,
    raw_content_by_news_id: dict[UUID, str] | None = None,
    raw_hash_by_news_id: dict[UUID, str] | None = None,
    strict: bool = False,
    allow_missing_news: bool = False,
) -> ValidationResult:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    seen: dict[UUID, DatasetSplit] = {}
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(ValidationIssue(index, "MALFORMED_JSON", str(exc)))
            continue
        if not isinstance(payload, dict):
            errors.append(ValidationIssue(index, "INVALID_RECORD", "line must be JSON object"))
            continue
        raw_record = cast("dict[object, object]", payload)
        record: dict[str, object] = {str(key): value for key, value in raw_record.items()}
        if strict:
            unknown = sorted(set(record) - ALLOWED_FIELDS)
            if unknown:
                errors.append(ValidationIssue(index, "UNKNOWN_FIELDS", ", ".join(unknown)))
        if record.get("schema_version") != GOLD_SCHEMA_VERSION:
            errors.append(ValidationIssue(index, "INVALID_SCHEMA_VERSION", "unsupported schema"))
        news_id = _parse_uuid(record.get("news_id"))
        if news_id is None:
            errors.append(ValidationIssue(index, "INVALID_UUID", "news_id must be UUID"))
            continue
        if news_id in seen and seen[news_id] != _safe_split(record.get("split")):
            errors.append(
                ValidationIssue(
                    index,
                    "SPLIT_OVERLAP",
                    "one news_id cannot appear in multiple splits",
                    str(news_id),
                )
            )
        elif news_id in seen:
            errors.append(
                ValidationIssue(index, "DUPLICATE_NEWS_ID", "duplicate news_id", str(news_id))
            )
        split = _safe_split(record.get("split"))
        if split is None:
            errors.append(ValidationIssue(index, "INVALID_SPLIT", "invalid split", str(news_id)))
        else:
            seen[news_id] = split
        if _safe_review_status(record.get("review_status")) is None:
            errors.append(
                ValidationIssue(
                    index, "INVALID_REVIEW_STATUS", "invalid review_status", str(news_id)
                )
            )
        try:
            example = annotation_from_json(record)
        except (KeyError, ValueError, TypeError, InvalidOperation) as exc:
            errors.append(ValidationIssue(index, "INVALID_RECORD", str(exc), str(news_id)))
            continue
        if example.published_at.tzinfo is None or example.published_at.utcoffset() is None:
            errors.append(
                ValidationIssue(
                    index, "NAIVE_DATETIME", "published_at must be timezone-aware", str(news_id)
                )
            )
        raw_hash = None if raw_hash_by_news_id is None else raw_hash_by_news_id.get(news_id)
        raw_content = (
            None if raw_content_by_news_id is None else raw_content_by_news_id.get(news_id)
        )
        if raw_hash_by_news_id is not None and raw_hash is None and not allow_missing_news:
            errors.append(ValidationIssue(index, "MISSING_NEWS", "news_id not found", str(news_id)))
        if raw_hash is not None and raw_hash != example.raw_content_hash:
            errors.append(
                ValidationIssue(index, "HASH_MISMATCH", "raw_content_hash differs", str(news_id))
            )
        for event in example.gold_events:
            _validate_span(
                errors,
                index,
                str(news_id),
                raw_content,
                event.start_position,
                event.end_position,
                event.evidence_text,
            )
        for fact in example.gold_financial_facts:
            _validate_span(
                errors,
                index,
                str(news_id),
                raw_content,
                fact.start_position,
                fact.end_position,
                fact.evidence_text,
            )
            _validate_decimal_string(errors, index, str(news_id), str(fact.raw_value))
            _validate_decimal_string(errors, index, str(news_id), str(fact.normalized_value))
        if example.review_status == ReviewStatus.REVIEWED:
            if not example.gold_events and not example.gold_financial_facts:
                warnings.append(
                    ValidationIssue(
                        index,
                        "EMPTY_REVIEWED_LABELS",
                        "reviewed record has no gold labels",
                        str(news_id),
                    )
                )
    return ValidationResult(errors=errors, warnings=warnings)


def _parse_uuid(value: object) -> UUID | None:
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def _safe_split(value: object) -> DatasetSplit | None:
    try:
        return DatasetSplit(str(value))
    except ValueError:
        return None


def _safe_review_status(value: object) -> ReviewStatus | None:
    try:
        return ReviewStatus(str(value))
    except ValueError:
        return None


def _validate_span(
    errors: list[ValidationIssue],
    line_number: int,
    news_id: str,
    raw_content: str | None,
    start: int,
    end: int,
    evidence_text: str,
) -> None:
    if start < 0 or end <= start:
        errors.append(
            ValidationIssue(
                line_number, "INVALID_SPAN", "start_position < end_position required", news_id
            )
        )
        return
    if raw_content is not None and raw_content[start:end] != evidence_text:
        errors.append(
            ValidationIssue(line_number, "EVIDENCE_MISMATCH", "evidence span mismatch", news_id)
        )


def _validate_decimal_string(
    errors: list[ValidationIssue],
    line_number: int,
    news_id: str,
    value: str,
) -> None:
    try:
        Decimal(value)
    except InvalidOperation:
        errors.append(
            ValidationIssue(line_number, "INVALID_DECIMAL", "invalid Decimal string", news_id)
        )
