from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from src.news.domain.enums import PublicationTimestampQuality

CORPUS_QUALITY_VERSION = "corpus-quality-expansion-v1"
CORPUS_VERSION = "reaction-ready-corpus-v2"
ANNOTATION_BATCH_VERSION = "annotation-batch-002"
FROZEN_BATCH_001_GOLD_SHA256 = "4934b37b1c036eedb6191dae5ece2fa49e710d00455576cee3de081cc9e7c196"


class UnknownDiagnosticCategory(StrEnum):
    TRUE_NO_SUPPORTED_EVENT = "TRUE_NO_SUPPORTED_EVENT"
    CONTENT_TOO_THIN = "CONTENT_TOO_THIN"
    SOURCE_PARSE_OR_TRUNCATION = "SOURCE_PARSE_OR_TRUNCATION"
    RULE_MISS_CANDIDATE = "RULE_MISS_CANDIDATE"
    UNCERTAIN = "UNCERTAIN"


class AnnotationStatus(StrEnum):
    DRAFT = "DRAFT"


class AssignmentStatus(StrEnum):
    UNASSIGNED = "UNASSIGNED"


@dataclass(frozen=True, slots=True)
class PublicationTimeRecord:
    news_id: UUID
    ticker: str
    source_code: str
    source_item_id: str
    source_url: str
    title: str
    content: str
    published_at: datetime
    timestamp_quality: PublicationTimestampQuality
    storage_policy: str
    content_is_excerpt: bool
    rules_primary_event: str
    rules_event_count: int
    rules_fact_count: int
    analysis_status: str
    analysis_warnings: tuple[str, ...]
    matched: bool
    market_data_ready: bool
    reaction_ready: bool
    feature_ready: bool
    valid_label_horizons: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        parsed = urlparse(self.source_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("quality records require an issuer HTTPS source URL")
        if not self.title.strip() or not self.source_item_id.strip():
            raise ValueError("quality record identity and title must not be empty")
        if min(self.rules_event_count, self.rules_fact_count) < 0:
            raise ValueError("analysis counts must not be negative")
        if self.timestamp_quality != PublicationTimestampQuality.EXACT and (
            self.market_data_ready or self.reaction_ready or self.feature_ready
        ):
            raise ValueError("non-EXACT publication cannot be reaction-ready")


DIAGNOSIS_INPUT_FIELDS = frozenset(
    {
        "news_id",
        "ticker",
        "source_code",
        "source_item_id",
        "source_url",
        "title",
        "content",
        "published_at",
        "timestamp_quality",
        "storage_policy",
        "content_is_excerpt",
        "rules_primary_event",
        "rules_event_count",
        "rules_fact_count",
        "analysis_status",
        "analysis_warnings",
    }
)


@dataclass(frozen=True, slots=True)
class UnknownDiagnosis:
    record: PublicationTimeRecord
    category: UnknownDiagnosticCategory
    rationale: str

    def payload(self) -> dict[str, Any]:
        item = self.record
        return {
            "schema_version": CORPUS_QUALITY_VERSION,
            "news_id": str(item.news_id),
            "ticker": item.ticker,
            "source_code": item.source_code,
            "source_item_id": item.source_item_id,
            "source_url": item.source_url,
            "published_at": item.published_at.isoformat(),
            "timestamp_quality": item.timestamp_quality.value,
            "title": item.title,
            "stored_content": item.content,
            "content_length": len(item.content),
            "content_is_excerpt": item.content_is_excerpt,
            "storage_policy": item.storage_policy,
            "rules_primary_event": item.rules_primary_event,
            "rules_event_count": item.rules_event_count,
            "rules_fact_count": item.rules_fact_count,
            "analysis_status": item.analysis_status,
            "analysis_warnings": list(item.analysis_warnings),
            "diagnostic_category": self.category.value,
            "rationale": self.rationale,
            "is_gold_label": False,
            "uses_post_event_data": False,
        }


_CLEAR_NON_SUPPORTED = re.compile(
    r"(?:personnel training|university|education|domestic tourism|car tourism|tourism potential)",
    re.IGNORECASE,
)
_SUPPORTED_EVENT_SIGNAL = re.compile(
    r"(?:financial results|dividend|guidance|forecast|major contract|acquisition|merger|"
    r"buyback|credit rating|sanction|production (?:rose|fell|increased|decreased)|net profit|"
    r"revenue|ebitda)",
    re.IGNORECASE,
)
_TRUNCATION_SIGNAL = re.compile(r"(?:\.\.\.|\u2026|\[truncated\])\s*$", re.IGNORECASE)


def diagnose_unknown(record: PublicationTimeRecord) -> UnknownDiagnosis:
    if record.rules_primary_event != "UNKNOWN":
        raise ValueError("UNKNOWN diagnosis accepts only deterministic UNKNOWN records")
    visible = _visible_text(record.content)
    combined = f"{record.title}\n{visible}"
    if not visible or len(visible) < 60:
        category = UnknownDiagnosticCategory.CONTENT_TOO_THIN
        rationale = "Stored publication-time content is absent or too short for event semantics."
    elif _TRUNCATION_SIGNAL.search(visible) or _looks_malformed(record.content):
        category = UnknownDiagnosticCategory.SOURCE_PARSE_OR_TRUNCATION
        rationale = "Stored source payload has an explicit truncation or parse signal."
    elif _CLEAR_NON_SUPPORTED.search(combined):
        category = UnknownDiagnosticCategory.TRUE_NO_SUPPORTED_EVENT
        rationale = (
            "Available text describes education or tourism cooperation without a supported "
            "material corporate-event signal."
        )
    elif _SUPPORTED_EVENT_SIGNAL.search(combined):
        category = UnknownDiagnosticCategory.RULE_MISS_CANDIDATE
        rationale = "Available text contains a supported-event signal that rules did not detect."
    elif record.content_is_excerpt and len(visible) < 350:
        category = UnknownDiagnosticCategory.CONTENT_TOO_THIN
        rationale = (
            "Issuer RSS stores only a short excerpt; material terms may exist only on the linked "
            "release page."
        )
    else:
        category = UnknownDiagnosticCategory.UNCERTAIN
        rationale = "Publication-time evidence is insufficient for a stronger diagnostic category."
    return UnknownDiagnosis(record=record, category=category, rationale=rationale)


@dataclass(frozen=True, slots=True)
class SourceAcceptanceEvidence:
    source_code: str
    tickers: tuple[str, ...]
    source_url: str
    issuer: str
    source_owner: str
    publication_timestamp_semantics: str
    storage_policy: str
    issuer_owned: bool
    exact_publication_timestamp: bool
    timezone_semantics_confirmed: bool
    stable_item_identity: bool
    storage_policy_confirmed: bool
    https: bool
    bounded_acquisition: bool
    blocker: str | None = None

    @property
    def compliant(self) -> bool:
        required = (
            self.issuer_owned,
            self.exact_publication_timestamp,
            self.timezone_semantics_confirmed,
            self.stable_item_identity,
            self.storage_policy_confirmed,
            self.https,
            self.bounded_acquisition,
        )
        return all(required)

    def validate(self) -> None:
        parsed = urlparse(self.source_url)
        if not self.source_code.strip() or not self.tickers:
            raise ValueError("source acceptance requires identity and tickers")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source acceptance URL must be absolute HTTP(S)")
        if not all(
            value.strip()
            for value in (
                self.issuer,
                self.source_owner,
                self.publication_timestamp_semantics,
                self.storage_policy,
            )
        ):
            raise ValueError("source provenance and timestamp fields must not be empty")
        if not self.compliant and not (self.blocker or "").strip():
            raise ValueError("non-compliant source acceptance requires a blocker")

    def payload(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["tickers"] = list(self.tickers)
        payload["compliant"] = self.compliant
        return payload


@dataclass(frozen=True, slots=True)
class ExpansionSelection:
    source_code: str
    date_from: datetime
    date_to: datetime
    limit: int
    tickers: tuple[str, ...]

    def validate(self) -> None:
        if not self.source_code.strip() or not self.tickers:
            raise ValueError("source and issuer universe are required")
        if self.date_to < self.date_from:
            raise ValueError("explicit date range is invalid")
        if not 1 <= self.limit <= 100:
            raise ValueError("live source limit must be between 1 and 100")


@dataclass(frozen=True, slots=True)
class RequiredMarketWindow:
    ticker: str
    date_from: datetime
    date_to: datetime
    pre_context_minutes: int = 60
    post_label_minutes: int = 60

    def __post_init__(self) -> None:
        if self.date_to <= self.date_from:
            raise ValueError("market window must have positive duration")
        if self.date_to - self.date_from > timedelta(days=3):
            raise ValueError("market window must remain bounded")


def required_market_windows(
    records: list[PublicationTimeRecord], *, safety_margin: timedelta = timedelta(days=1)
) -> list[RequiredMarketWindow]:
    if safety_margin < timedelta() or safety_margin > timedelta(days=1):
        raise ValueError("market safety margin must be between zero and one day")
    return [
        RequiredMarketWindow(
            ticker=item.ticker,
            date_from=item.published_at - safety_margin - timedelta(minutes=60),
            date_to=item.published_at + safety_margin + timedelta(minutes=60),
        )
        for item in sorted(records, key=lambda value: (value.ticker, value.published_at))
        if item.timestamp_quality == PublicationTimestampQuality.EXACT and item.matched
    ]


@dataclass(frozen=True, slots=True)
class ShadowPrediction:
    news_id: UUID
    primary_event: str
    event_count: int
    fact_count: int
    successful: bool


def rules_vs_shadow(
    records: list[PublicationTimeRecord], predictions: list[ShadowPrediction]
) -> list[dict[str, Any]]:
    by_news = {item.news_id: item for item in predictions if item.successful}
    comparisons: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (item.published_at, str(item.news_id))):
        prediction = by_news.get(record.news_id)
        if prediction is None:
            continue
        comparisons.append(
            {
                "news_id": str(record.news_id),
                "ticker": record.ticker,
                "rules_primary_event": record.rules_primary_event,
                "rules_event_count": record.rules_event_count,
                "rules_fact_count": record.rules_fact_count,
                "ai_primary_event": prediction.primary_event,
                "ai_event_count": prediction.event_count,
                "ai_fact_count": prediction.fact_count,
                "event_agreement": record.rules_primary_event == prediction.primary_event,
            }
        )
    return comparisons


def select_annotation_batch(
    records: list[PublicationTimeRecord], *, limit: int = 50
) -> list[dict[str, Any]]:
    if not 1 <= limit <= 50:
        raise ValueError("annotation batch limit must be between 1 and 50")
    eligible = [
        item
        for item in records
        if item.timestamp_quality == PublicationTimestampQuality.EXACT
        and item.matched
        and item.storage_policy not in {"METADATA_ONLY", "UNKNOWN"}
    ]
    groups: dict[tuple[str, str], deque[PublicationTimeRecord]] = defaultdict(deque)
    for item in sorted(eligible, key=lambda value: (value.published_at, value.source_item_id)):
        groups[(item.ticker, item.source_code)].append(item)
    selected: list[PublicationTimeRecord] = []
    keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].popleft())
    return [_annotation_payload(item) for item in selected]


def build_baseline(
    records: list[PublicationTimeRecord], *, expected_count: int = 10
) -> dict[str, Any]:
    if len(records) != expected_count:
        raise ValueError(f"baseline requires exactly {expected_count} records")
    tickers = sorted({item.ticker for item in records})
    if tickers != ["ROSN"]:
        raise ValueError("diagnostic baseline must contain ROSN only")
    return {
        "schema_version": CORPUS_QUALITY_VERSION,
        "count": len(records),
        "ticker": "ROSN",
        "EXACT": sum(
            item.timestamp_quality == PublicationTimestampQuality.EXACT for item in records
        ),
        "matched": sum(item.matched for item in records),
        "reaction_ready": sum(item.reaction_ready for item in records),
        "feature_ready": sum(item.feature_ready for item in records),
        "deterministic_primary_UNKNOWN": sum(
            item.rules_primary_event == "UNKNOWN" for item in records
        ),
        "news_ids": [str(item.news_id) for item in records],
        "uses_post_event_data_for_diagnosis": False,
    }


def diversity_warnings(
    ticker_distribution: dict[str, int], event_distribution: dict[str, int]
) -> list[str]:
    total = sum(event_distribution.values())
    if total == 0:
        return ["HIGH_UNKNOWN_EVENT_RATE", "LOW_TICKER_DIVERSITY", "LOW_EVENT_DIVERSITY"]
    warnings: list[str] = []
    if event_distribution.get("UNKNOWN", 0) / total > 0.5:
        warnings.append("HIGH_UNKNOWN_EVENT_RATE")
    if ticker_distribution and max(ticker_distribution.values()) / total > 0.7:
        warnings.append("LOW_TICKER_DIVERSITY")
    if event_distribution and max(event_distribution.values()) / total > 0.7:
        warnings.append("LOW_EVENT_DIVERSITY")
    return warnings


def readiness_report(
    *, reaction_rows: int, annotation_rows: int, tickers: int, unknown_rate: float
) -> dict[str, str]:
    if reaction_rows < 100:
        reaction = "NOT_READY"
    elif reaction_rows < 500:
        reaction = "PILOT_ONLY"
    elif reaction_rows < 1000:
        reaction = "BASELINE_EXPERIMENT_READY"
    else:
        reaction = "BASELINE_TRAINING_READY"
    annotation = "REVIEW_SAMPLE_READY" if annotation_rows >= 20 else "NOT_READY"
    event_quality = (
        "EVENT_FEATURE_QUALITY_BLOCKER" if unknown_rate > 0.5 else "EVENT_FEATURE_AUDIT_READY"
    )
    model = reaction
    if event_quality == "EVENT_FEATURE_QUALITY_BLOCKER" or tickers < 3:
        model = "NOT_READY"
    return {
        "REACTION_DATA_READINESS": reaction,
        "EVENT_ANNOTATION_READINESS": annotation,
        "EVENT_FEATURE_QUALITY": event_quality,
        "MODEL_TRAINING_READINESS": model,
    }


def cumulative_funnel(records: list[PublicationTimeRecord]) -> list[dict[str, int | str]]:
    values = (
        ("discovered", len(records)),
        ("validated", len(records)),
        ("imported", len(records)),
        (
            "EXACT",
            sum(item.timestamp_quality == PublicationTimestampQuality.EXACT for item in records),
        ),
        ("matched", sum(item.matched for item in records)),
        ("event-analyzed", sum(bool(item.analysis_status) for item in records)),
        ("market-data-ready", sum(item.market_data_ready for item in records)),
        ("reaction-ready", sum(item.reaction_ready for item in records)),
        ("feature-ready", sum(item.feature_ready for item in records)),
    )
    return [{"stage": name, "count": count} for name, count in values]


def distributions(records: list[PublicationTimeRecord]) -> tuple[dict[str, int], dict[str, int]]:
    tickers = Counter(item.ticker for item in records)
    events = Counter(item.rules_primary_event for item in records)
    return dict(sorted(tickers.items())), dict(sorted(events.items()))


def _annotation_payload(item: PublicationTimeRecord) -> dict[str, Any]:
    return {
        "schema_version": ANNOTATION_BATCH_VERSION,
        "record_id": f"batch-002-{item.news_id}",
        "news_id": str(item.news_id),
        "ticker": item.ticker,
        "source_code": item.source_code,
        "source_item_id": item.source_item_id,
        "source_url": item.source_url,
        "published_at": item.published_at.isoformat(),
        "timestamp_quality": item.timestamp_quality.value,
        "storage_policy": item.storage_policy,
        "annotation_text": item.content,
        "rules_primary_event": item.rules_primary_event,
        "annotation_status": AnnotationStatus.DRAFT.value,
        "assignment_status": AssignmentStatus.UNASSIGNED.value,
        "is_gold": False,
        "selection_policy": "deterministic ticker/source strata then publication order",
    }


def _visible_text(content: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", content)).strip()


def _looks_malformed(content: str) -> bool:
    return content.count("<") != content.count(">")
