from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from uuid import UUID

from src.news.domain.enums import PublicationTimestampQuality

CORPUS_VERSION = "fresh-real-corpus-v1"
ANNOTATION_BATCH_VERSION = "annotation-batch-004"
BATCH_003_DATASET = "ru-corporate-events-real-batch-003-gold-v1"
BATCH_003_STATUS = "OBSERVED_EVALUATION_SET"
APPROVED_SOURCE_TICKERS: dict[str, str] = {
    "ROSNEFT_PRESS_RELEASES_RSS": "ROSN",
    "YANDEX_IR_PRESS_RELEASES_RSS": "YDEX",
}
PREDICTION_FIELDS = frozenset(
    {
        "events",
        "financial_facts",
        "gold_primary_event",
        "human_events",
        "human_primary_event",
        "primary_event",
        "qwen_primary_event",
        "rules_primary_event",
    }
)
FUTURE_MARKET_FIELDS = frozenset(
    {
        "abnormal_return",
        "future_return",
        "future_volume",
        "labels",
        "market_reaction",
        "post_event_price",
        "return",
        "volume",
    }
)


class CorpusSplit(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    FRESH_HOLDOUT = "FRESH_HOLDOUT"


class MatchStatus(StrEnum):
    MATCHED = "MATCHED"
    AMBIGUOUS = "AMBIGUOUS"
    UNMATCHED = "UNMATCHED"


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    source_codes: tuple[str, ...]
    date_from: datetime
    date_to: datetime
    limit: int
    source_order: str = "published_at, source_code, source_item_id"

    def normalized(self) -> SelectionPolicy:
        source_codes = tuple(sorted({item.strip().upper() for item in self.source_codes}))
        if not source_codes or any(item not in APPROVED_SOURCE_TICKERS for item in source_codes):
            raise ValueError("selection requires explicitly approved issuer sources")
        if self.date_from.tzinfo is None or self.date_to.tzinfo is None:
            raise ValueError("selection range must be timezone-aware")
        if self.date_to < self.date_from:
            raise ValueError("selection range is invalid")
        if not 1 <= self.limit <= 100:
            raise ValueError("fresh corpus limit must be between 1 and 100")
        return SelectionPolicy(
            source_codes=source_codes,
            date_from=self.date_from.astimezone(UTC),
            date_to=self.date_to.astimezone(UTC),
            limit=self.limit,
            source_order=self.source_order,
        )


@dataclass(frozen=True, slots=True)
class FreshCorpusRecord:
    news_id: UUID
    source_code: str
    source_item_id: str
    source_url: str
    ticker: str
    published_at: datetime
    original_timestamp_text: str
    source_timezone: str | None
    timestamp_quality: PublicationTimestampQuality
    title: str
    annotation_text: str
    content_hash: str
    storage_policy: str
    content_is_excerpt: bool
    match_status: MatchStatus
    reaction_ready: bool = False
    feature_ready: bool = False

    @property
    def logical_key(self) -> tuple[str, str]:
        return self.source_code, self.source_item_id

    def validate(self) -> None:
        if self.source_code not in APPROVED_SOURCE_TICKERS:
            raise ValueError("record is not from an approved REAL source")
        if self.timestamp_quality != PublicationTimestampQuality.EXACT:
            raise ValueError("Batch 004 accepts EXACT timestamps only")
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")
        required = (
            self.source_item_id,
            self.source_url,
            self.ticker,
            self.original_timestamp_text,
            self.title,
            self.annotation_text,
            self.content_hash,
        )
        if any(not item.strip() for item in required):
            raise ValueError("fresh corpus record misses required provenance or content")
        parsed_url = urlparse(self.source_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("fresh corpus source URL must be issuer-owned HTTPS")
        expected_hash = hashlib.sha256(self.annotation_text.encode()).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError("annotation text does not match the stored content hash")
        if self.storage_policy == "EXCERPT_ALLOWED" and not self.content_is_excerpt:
            raise ValueError("EXCERPT_ALLOWED content must be explicitly marked as excerpt")
        if self.storage_policy not in {"EXCERPT_ALLOWED", "FULL_TEXT_ALLOWED"}:
            raise ValueError("storage policy does not permit annotation text")

    def annotation_payload(self, split: CorpusSplit) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "schema_version": ANNOTATION_BATCH_VERSION,
            "record_id": f"batch-004-{self.news_id}",
            "news_id": str(self.news_id),
            "source": self.source_code,
            "source_item_id": self.source_item_id,
            "source_url": self.source_url,
            "ticker": self.ticker,
            "published_at": _utc_text(self.published_at),
            "original_timestamp_text": self.original_timestamp_text,
            "source_timezone": self.source_timezone,
            "timestamp_quality": self.timestamp_quality.value,
            "title": self.title,
            "annotation_text": self.annotation_text,
            "raw_content_hash": self.content_hash,
            "content_storage_policy": self.storage_policy,
            "content_is_excerpt": self.content_is_excerpt,
            "provenance": "REAL",
            "match_status": self.match_status.value,
            "split": split.value,
            "annotation_status": "DRAFT",
            "assignment_status": "UNASSIGNED",
            "is_gold": False,
        }
        assert_safe_annotation_payload(payload)
        return payload


@dataclass(frozen=True, slots=True)
class ExclusionIndex:
    news_ids: frozenset[UUID] = frozenset()
    logical_keys: frozenset[tuple[str, str]] = frozenset()
    content_hashes: frozenset[str] = frozenset()

    def contains(self, record: FreshCorpusRecord) -> bool:
        return (
            record.news_id in self.news_ids
            or record.logical_key in self.logical_keys
            or record.content_hash in self.content_hashes
        )


@dataclass(frozen=True, slots=True)
class SelectionResult:
    records: tuple[FreshCorpusRecord, ...]
    excluded_overlap_count: int


@dataclass(frozen=True, slots=True)
class FrozenSplit:
    assignments: tuple[tuple[UUID, CorpusSplit], ...]
    split_sha256: str

    def split_for(self, news_id: UUID) -> CorpusSplit:
        return next(split for candidate, split in self.assignments if candidate == news_id)


def selection_field_names() -> set[str]:
    return {item.name for item in fields(SelectionPolicy)}


def load_exclusion_index(paths: tuple[Path, ...]) -> ExclusionIndex:
    news_ids: set[UUID] = set()
    logical_keys: set[tuple[str, str]] = set()
    content_hashes: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = cast("dict[str, Any]", json.loads(line))
            if payload.get("news_id"):
                news_ids.add(UUID(str(payload["news_id"])))
            source = str(payload.get("source_code") or payload.get("source") or "")
            source_item_id = str(payload.get("source_item_id") or "")
            if source and source_item_id:
                logical_keys.add((source, source_item_id))
            content_hash = str(payload.get("raw_content_hash") or payload.get("content_hash") or "")
            if content_hash:
                content_hashes.add(content_hash)
    return ExclusionIndex(
        news_ids=frozenset(news_ids),
        logical_keys=frozenset(logical_keys),
        content_hashes=frozenset(content_hashes),
    )


def select_fresh_records(
    records: list[FreshCorpusRecord],
    *,
    policy: SelectionPolicy,
    exclusions: ExclusionIndex,
) -> SelectionResult:
    normalized = policy.normalized()
    selected: list[FreshCorpusRecord] = []
    seen: set[tuple[str, str]] = set()
    excluded_overlap_count = 0
    ordered = sorted(records, key=_record_order)
    for record in ordered:
        record.validate()
        if record.source_code not in normalized.source_codes:
            continue
        published = record.published_at.astimezone(UTC)
        if not normalized.date_from <= published <= normalized.date_to:
            continue
        if record.logical_key in seen:
            continue
        seen.add(record.logical_key)
        if exclusions.contains(record):
            excluded_overlap_count += 1
            continue
        selected.append(record)
        if len(selected) == normalized.limit:
            break
    return SelectionResult(tuple(selected), excluded_overlap_count)


def freeze_temporal_split(
    records: tuple[FreshCorpusRecord, ...], *, development_ratio: float = 0.70
) -> FrozenSplit:
    if not records:
        raise ValueError("cannot freeze an empty corpus")
    if len(records) < 2:
        raise ValueError("fresh corpus requires at least two records for a holdout")
    if not 0.5 <= development_ratio <= 0.9:
        raise ValueError("development_ratio must be between 0.5 and 0.9")
    ordered = tuple(sorted(records, key=_record_order))
    development_count = min(len(ordered) - 1, max(1, int(len(ordered) * development_ratio + 0.5)))
    assignments = tuple(
        (
            record.news_id,
            CorpusSplit.DEVELOPMENT if index < development_count else CorpusSplit.FRESH_HOLDOUT,
        )
        for index, record in enumerate(ordered)
    )
    material = [
        {
            "news_id": str(record.news_id),
            "published_at": _utc_text(record.published_at),
            "source": record.source_code,
            "source_item_id": record.source_item_id,
            "split": split.value,
        }
        for record, (_, split) in zip(ordered, assignments, strict=True)
    ]
    digest = hashlib.sha256((stable_json(material) + "\n").encode()).hexdigest()
    return FrozenSplit(assignments=assignments, split_sha256=digest)


def coverage_payload(records: tuple[FreshCorpusRecord, ...]) -> dict[str, Any]:
    ordered = tuple(sorted(records, key=_record_order))
    lengths = sorted(len(record.annotation_text) for record in ordered)
    return {
        "schema_version": CORPUS_VERSION,
        "records": len(ordered),
        "timestamp_quality": {"EXACT": len(ordered)},
        "ticker_distribution": dict(sorted(Counter(item.ticker for item in ordered).items())),
        "source_distribution": dict(sorted(Counter(item.source_code for item in ordered).items())),
        "month_distribution": dict(
            sorted(Counter(item.published_at.strftime("%Y-%m") for item in ordered).items())
        ),
        "year_distribution": dict(
            sorted(Counter(item.published_at.strftime("%Y") for item in ordered).items())
        ),
        "match_distribution": dict(
            sorted(Counter(item.match_status.value for item in ordered).items())
        ),
        "text_length": {
            "min": lengths[0] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
            "median": lengths[len(lengths) // 2] if lengths else 0,
        },
        "date_range": _date_range(ordered),
        "human_event_distribution": "UNAVAILABLE_BEFORE_REVIEW",
        "model_diagnostic_event_distribution": "NOT_COMPUTED",
    }


def assert_safe_annotation_payload(payload: dict[str, Any]) -> None:
    forbidden = PREDICTION_FIELDS | FUTURE_MARKET_FIELDS
    leaked = forbidden.intersection(payload)
    if leaked:
        raise ValueError(f"annotation payload contains prohibited fields: {sorted(leaked)}")
    if payload.get("annotation_status") != "DRAFT":
        raise ValueError("Batch 004 records must remain DRAFT")
    if payload.get("assignment_status") != "UNASSIGNED":
        raise ValueError("Batch 004 records must remain UNASSIGNED")
    if payload.get("is_gold") is not False:
        raise ValueError("Batch 004 must not be marked as gold")


def stable_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _record_order(record: FreshCorpusRecord) -> tuple[datetime, str, str]:
    return record.published_at.astimezone(UTC), record.source_code, record.source_item_id


def _date_range(records: tuple[FreshCorpusRecord, ...]) -> dict[str, str | None]:
    if not records:
        return {"from": None, "to": None}
    dates = sorted(item.published_at for item in records)
    return {"from": _utc_text(dates[0]), "to": _utc_text(dates[-1])}


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
