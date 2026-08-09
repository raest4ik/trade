from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from src.evaluation.domain.entities import (
    GOLD_SCHEMA_VERSION,
    AnnotationExample,
    GoldEvent,
    GoldFinancialFact,
)
from src.evaluation.domain.enums import DatasetSplit, ReviewStatus
from src.evaluation.domain.metrics import (
    EventEvaluationInput,
    FactEvaluationInput,
    evaluate_event_predictions,
    evaluate_fact_predictions,
)
from src.evaluation.domain.serialization import (
    annotation_to_json,
    gold_event_from_json,
    gold_fact_from_json,
)
from src.events.application.use_cases import AnalyzeNewsEvent
from src.events.domain.entities import DetectedEvent, ExtractedFinancialFact, NewsEventAnalysis
from src.events.domain.enums import EventType
from src.events.infrastructure.repositories import SqlAlchemyEventAnalysisRepository
from src.instruments.application.use_cases import MatchNewsInstruments
from src.instruments.domain.entities import NewsInstrumentMatch
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.news.application.use_cases import CreateNewsItem, CreateNewsItemCommand
from src.news.domain.entities import NewsItem
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository

SEED_SCHEMA_VERSION = "event-seed-v1"
SEED_TARGET_SCHEMA = "event-gold-v1"
SEED_SOURCE_NAME = "seed-dataset"
SEED_ANNOTATOR = "seed-curation"
DATE_ONLY_MARKER = "DATE_ONLY / DO_NOT_USE_FOR_REACTION"
EXPECTED_SEED_QUOTAS = {
    "FINANCIAL_RESULTS": 20,
    "DIVIDEND": 10,
    "GUIDANCE": 8,
    "M&A_CONTRACT": 5,
    "PRODUCTION_UPDATE": 4,
    "SANCTIONS_REGULATORY": 3,
}
SECONDARY_SOURCE_TIERS = {"SECONDARY"}


@dataclass(frozen=True, slots=True)
class SeedSource:
    title: str
    url: str
    tier: str
    support_url: str | None


@dataclass(frozen=True, slots=True)
class SeedEventRecord:
    schema_version: str
    target_schema: str
    batch_id: str
    record_id: str
    source_published_date: str
    tickers: list[str]
    company: str
    quota_category: str
    annotation_text: str
    text_origin: str
    raw_content_hash: str
    review_status: str
    notes: str | None
    gold_events: list[GoldEvent]
    gold_financial_facts: list[GoldFinancialFact]
    source: SeedSource


@dataclass(frozen=True, slots=True)
class SeedValidationResult:
    records: list[SeedEventRecord]
    errors: list[str]
    quotas: dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class SeedProcessingStats:
    records_total: int
    created: int
    already_exists: int
    invalid: int
    instrument_matches_total: int
    records_with_instrument_matches: int
    ambiguous_instrument_matches: int
    event_status_counts: dict[str, int]
    primary_event_counts: dict[str, int]
    predicted_fact_count: int
    records_with_predicted_facts: int
    category_counts: dict[str, int]
    source_review_required: list[str]
    baseline_metrics: dict[str, object]
    validation_errors: list[str]


@dataclass(frozen=True, slots=True)
class SeedProcessingResult:
    stats: SeedProcessingStats
    review_jsonl_path: Path
    mapping_path: Path
    comparison_dir: Path
    review_queue_path: Path


@dataclass(frozen=True, slots=True)
class ProcessedSeedRecord:
    seed: SeedEventRecord
    news: NewsItem
    created: bool
    matches: list[NewsInstrumentMatch]
    analysis: NewsEventAnalysis
    category: str
    review_reasons: list[str]
    source_review_required: bool


def validate_seed_file(path: Path) -> SeedValidationResult:
    errors: list[str] = []
    records: list[SeedEventRecord] = []
    seen_record_ids: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"line {line_number}: empty line")
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: malformed JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"line {line_number}: record must be an object")
            continue
        record_payload = cast("dict[str, object]", payload)
        record = _seed_record_from_payload(record_payload, line_number, errors)
        if record is None:
            continue
        if record.record_id in seen_record_ids:
            errors.append(f"line {line_number}: duplicate record_id {record.record_id}")
        seen_record_ids.add(record.record_id)
        _validate_seed_record(record, line_number, errors)
        records.append(record)
    quotas = dict(sorted(Counter(record.quota_category for record in records).items()))
    if len(records) != 50:
        errors.append(f"records_total={len(records)} expected=50")
    for quota, expected in EXPECTED_SEED_QUOTAS.items():
        actual = quotas.get(quota, 0)
        if actual != expected:
            errors.append(f"quota {quota}={actual} expected={expected}")
    extra_quotas = sorted(set(quotas) - set(EXPECTED_SEED_QUOTAS))
    if extra_quotas:
        errors.append(f"unexpected quota categories: {', '.join(extra_quotas)}")
    return SeedValidationResult(records=records, errors=errors, quotas=quotas)


async def process_seed_batch(
    *,
    records: list[SeedEventRecord],
    news_repository: SqlAlchemyNewsRepository,
    instrument_repository: SqlAlchemyInstrumentRepository,
    event_repository: SqlAlchemyEventAnalysisRepository,
    output_dir: Path,
    dry_run: bool = False,
) -> SeedProcessingResult:
    processed: list[ProcessedSeedRecord] = []
    created = 0
    already_exists = 0
    for record in records:
        existing = await news_repository.get_by_source(SEED_SOURCE_NAME, record.record_id)
        if dry_run:
            if existing is None:
                created += 1
            else:
                already_exists += 1
            continue
        command = CreateNewsItemCommand(
            source_id=record.record_id,
            source_name=SEED_SOURCE_NAME,
            source_url=record.source.url,
            title=_seed_title(record),
            raw_content=record.annotation_text,
            language="ru",
            published_at=_technical_published_at(record.source_published_date),
            received_at=_technical_published_at(record.source_published_date),
        )
        save_result = await CreateNewsItem(news_repository).execute(command)
        created += 1 if save_result.created else 0
        already_exists += 0 if save_result.created else 1
        match_result = await MatchNewsInstruments(
            news_repository=news_repository,
            instrument_repository=instrument_repository,
        ).execute(save_result.item.id)
        analysis = (
            await AnalyzeNewsEvent(
                news_repository=news_repository,
                event_repository=event_repository,
            ).execute(save_result.item.id)
        ).analysis
        source_review_required, source_reasons = _source_review(record)
        category, review_reasons = _review_category(
            record=record,
            analysis=analysis,
            source_review_required=source_review_required,
            source_reasons=source_reasons,
        )
        processed.append(
            ProcessedSeedRecord(
                seed=record,
                news=save_result.item,
                created=save_result.created,
                matches=match_result.matches,
                analysis=analysis,
                category=category,
                review_reasons=review_reasons,
                source_review_required=source_review_required,
            )
        )
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    review_jsonl_path = output_dir / "batch-001-review.jsonl"
    mapping_path = output_dir / "batch-001-mapping.json"
    comparison_dir = output_dir / "batch-001-comparison"
    review_queue_path = output_dir / "batch-001-review-queue.md"
    if dry_run:
        stats = SeedProcessingStats(
            records_total=len(records),
            created=0,
            already_exists=already_exists,
            invalid=0,
            instrument_matches_total=0,
            records_with_instrument_matches=0,
            ambiguous_instrument_matches=0,
            event_status_counts={},
            primary_event_counts={},
            predicted_fact_count=0,
            records_with_predicted_facts=0,
            category_counts={},
            source_review_required=[],
            baseline_metrics={},
            validation_errors=[],
        )
        return SeedProcessingResult(
            stats=stats,
            review_jsonl_path=review_jsonl_path,
            mapping_path=mapping_path,
            comparison_dir=comparison_dir,
            review_queue_path=review_queue_path,
        )
    _write_review_jsonl(review_jsonl_path, processed)
    _write_mapping(mapping_path, processed)
    metrics, errors = _comparison_metrics(processed)
    _write_comparison_report(comparison_dir, processed, metrics, errors)
    _write_review_queue(review_queue_path, processed)
    stats = _stats(
        processed=processed,
        records_total=len(records),
        created=created,
        already_exists=already_exists,
        baseline_metrics=metrics,
        validation_errors=[],
    )
    return SeedProcessingResult(
        stats=stats,
        review_jsonl_path=review_jsonl_path,
        mapping_path=mapping_path,
        comparison_dir=comparison_dir,
        review_queue_path=review_queue_path,
    )


def dry_run_counts(
    records: list[SeedEventRecord],
    existing_source_ids: set[str],
) -> dict[str, int]:
    would_create = sum(1 for record in records if record.record_id not in existing_source_ids)
    already_exists = len(records) - would_create
    return {
        "records_total": len(records),
        "would_create": would_create,
        "already_exists": already_exists,
        "invalid": 0,
    }


def _seed_record_from_payload(
    payload: dict[str, object],
    line_number: int,
    errors: list[str],
) -> SeedEventRecord | None:
    try:
        source_payload = _required_dict(payload, "source")
        gold_events = [
            gold_event_from_json(item) for item in _required_list_of_dicts(payload, "gold_events")
        ]
        gold_facts = [
            gold_fact_from_json(item)
            for item in _required_list_of_dicts(payload, "gold_financial_facts")
        ]
        return SeedEventRecord(
            schema_version=str(payload["schema_version"]),
            target_schema=str(payload["target_schema"]),
            batch_id=str(payload["batch_id"]),
            record_id=str(payload["record_id"]),
            source_published_date=str(payload["source_published_date"]),
            tickers=[str(item) for item in _required_list(payload, "tickers")],
            company=str(payload["company"]),
            quota_category=str(payload["quota_category"]),
            annotation_text=str(payload["annotation_text"]),
            text_origin=str(payload["text_origin"]),
            raw_content_hash=str(payload["raw_content_hash"]),
            review_status=str(payload["review_status"]),
            notes=None if payload.get("notes") is None else str(payload.get("notes")),
            gold_events=gold_events,
            gold_financial_facts=gold_facts,
            source=SeedSource(
                title=str(source_payload.get("title", "")),
                url=str(source_payload["url"]),
                tier=str(source_payload.get("tier", "")),
                support_url=None
                if source_payload.get("support_url") is None
                else str(source_payload.get("support_url")),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"line {line_number}: invalid seed record: {exc}")
        return None


def _validate_seed_record(
    record: SeedEventRecord,
    line_number: int,
    errors: list[str],
) -> None:
    if record.schema_version != SEED_SCHEMA_VERSION:
        errors.append(f"line {line_number}: schema_version={record.schema_version}")
    if record.target_schema != SEED_TARGET_SCHEMA:
        errors.append(f"line {line_number}: target_schema={record.target_schema}")
    if record.review_status != "SEED_REVIEW_REQUIRED":
        errors.append(f"line {line_number}: review_status={record.review_status}")
    if not record.annotation_text.strip():
        errors.append(f"line {line_number}: annotation_text is empty")
    expected_hash = hashlib.sha256(record.annotation_text.encode("utf-8")).hexdigest()
    if record.raw_content_hash != expected_hash:
        errors.append(f"line {line_number}: raw_content_hash mismatch")
    if not record.source.url.strip():
        errors.append(f"line {line_number}: source.url is empty")
    if not record.tickers:
        errors.append(f"line {line_number}: tickers are empty")
    for item in [*record.gold_events, *record.gold_financial_facts]:
        if item.start_position < 0 or item.end_position <= item.start_position:
            errors.append(f"line {line_number}: invalid evidence span")
            continue
        if item.end_position > len(record.annotation_text):
            errors.append(f"line {line_number}: evidence span outside annotation_text")
            continue
        if record.annotation_text[item.start_position : item.end_position] != item.evidence_text:
            errors.append(f"line {line_number}: evidence span text mismatch")


def _technical_published_at(source_published_date: str) -> datetime:
    parsed = datetime.strptime(source_published_date, "%Y-%m-%d").date()
    return datetime.combine(parsed, time.min, tzinfo=UTC)


def _seed_title(record: SeedEventRecord) -> str:
    return (
        f"{record.company} {record.quota_category} {record.source_published_date} "
        f"[{DATE_ONLY_MARKER}]"
    )


def _source_review(record: SeedEventRecord) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    parsed = urlparse(record.source.url)
    if record.source.tier in SECONDARY_SOURCE_TIERS:
        reasons.append("SOURCE_REVIEW_REQUIRED: secondary source")
    if not parsed.netloc or parsed.path in {"", "/"}:
        reasons.append("SOURCE_REVIEW_REQUIRED: general or non-specific URL")
    if record.notes and any(
        marker in record.notes.lower()
        for marker in ("source", "confirm", "confirmation", "conflict", "incomplete")
    ):
        reasons.append("SOURCE_REVIEW_REQUIRED: seed notes require source/label confirmation")
    if any(fact.notes for fact in record.gold_financial_facts):
        reasons.append("SOURCE_REVIEW_REQUIRED: financial fact notes require review")
    return bool(reasons), reasons


def _review_category(
    *,
    record: SeedEventRecord,
    analysis: NewsEventAnalysis,
    source_review_required: bool,
    source_reasons: list[str],
) -> tuple[str, list[str]]:
    reasons: list[str] = list(source_reasons)
    gold_events = {event.event_type for event in record.gold_events}
    predicted_events = {event.event_type for event in analysis.events}
    event_match = (
        gold_events == predicted_events
        and _primary_gold_event(record) == analysis.primary_event_type
    )
    fact_result = evaluate_fact_predictions(
        [FactEvaluationInput(record.gold_financial_facts, analysis.financial_facts)]
    )
    strict_metrics = cast("dict[str, float]", fact_result.metrics["strict"])
    perfect_fact_match = strict_metrics["f1"] == 1.0
    if source_review_required:
        return "E", reasons
    if len(predicted_events) > 1 or analysis.status.value == "AMBIGUOUS" or len(gold_events) > 1:
        reasons.append("multiple or ambiguous events")
        return "D", reasons
    if not event_match:
        reasons.append("event mismatch")
        return "C", reasons
    if not perfect_fact_match:
        reasons.append("minor fact mismatch")
        return "B", reasons
    return "A", ["extractor and preliminary seed labels agree"]


def _primary_gold_event(record: SeedEventRecord) -> EventType | None:
    for event in record.gold_events:
        if event.is_primary:
            return event.event_type
    return record.gold_events[0].event_type if record.gold_events else None


def _write_review_jsonl(path: Path, processed: list[ProcessedSeedRecord]) -> None:
    lines: list[str] = []
    for item in processed:
        notes = (
            f"{DATE_ONLY_MARKER}; seed_record_id={item.seed.record_id}; "
            f"source_published_date={item.seed.source_published_date}; "
            f"source_tier={item.seed.source.tier}; source_url={item.seed.source.url}"
        )
        example = AnnotationExample(
            schema_version=GOLD_SCHEMA_VERSION,
            news_id=item.news.id,
            published_at=item.news.published_at,
            raw_content_hash=item.news.raw_content_hash,
            split=DatasetSplit.UNASSIGNED,
            review_status=ReviewStatus.DRAFT,
            annotator=SEED_ANNOTATOR,
            notes=notes,
            predicted_events=[_event_payload(event) for event in item.analysis.events],
            predicted_financial_facts=[
                _fact_payload(fact) for fact in item.analysis.financial_facts
            ],
            gold_events=item.seed.gold_events,
            gold_financial_facts=item.seed.gold_financial_facts,
        )
        lines.append(json.dumps(annotation_to_json(example), ensure_ascii=False, sort_keys=True))
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _write_mapping(path: Path, processed: list[ProcessedSeedRecord]) -> None:
    mapping = {
        item.seed.record_id: {
            "news_id": str(item.news.id),
            "source_item_id": item.seed.record_id,
            "raw_content_hash": item.news.raw_content_hash,
            "instrument_matches": [_match_payload(match) for match in item.matches],
        }
        for item in processed
    }
    path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _comparison_metrics(
    processed: list[ProcessedSeedRecord],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    event_inputs = [
        EventEvaluationInput(
            gold_events=item.seed.gold_events,
            predicted_events=item.analysis.events,
            gold_primary_event_type=_primary_gold_event(item.seed),
            predicted_primary_event_type=item.analysis.primary_event_type,
            prediction_status=item.analysis.status.value,
        )
        for item in processed
    ]
    fact_inputs = [
        FactEvaluationInput(item.seed.gold_financial_facts, item.analysis.financial_facts)
        for item in processed
    ]
    event_result = evaluate_event_predictions(event_inputs)
    fact_result = evaluate_fact_predictions(fact_inputs)
    per_record = [_record_summary(item) for item in processed]
    metrics: dict[str, object] = {
        "records_total": len(processed),
        "events": event_result.metrics,
        "facts": fact_result.metrics,
        "records": {
            "perfect_event_match": sum(1 for item in per_record if item["perfect_event_match"]),
            "event_mismatch": sum(1 for item in per_record if item["event_mismatch"]),
            "perfect_fact_match": sum(1 for item in per_record if item["perfect_fact_match"]),
            "requiring_human_review": sum(1 for item in processed if item.category != "A"),
        },
        "categories": dict(sorted(Counter(item.category for item in processed).items())),
    }
    errors = [
        {
            "record_id": processed[_error_example_index(error)].seed.record_id,
            **error,
        }
        for error in event_result.errors + fact_result.errors
    ]
    return metrics, errors


def _error_example_index(error: dict[str, object]) -> int:
    return int(str(error["example_index"]))


def _write_comparison_report(
    directory: Path,
    processed: list[ProcessedSeedRecord],
    metrics: dict[str, object],
    errors: list[dict[str, object]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (directory / "errors.jsonl").write_text(
        "".join(json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n" for error in errors),
        encoding="utf-8",
    )
    lines = [
        "# Batch 001 Seed Comparison",
        "",
        f"- records: {len(processed)}",
        f"- categories: {dict(sorted(Counter(item.category for item in processed).items()))}",
        f"- errors: {len(errors)}",
        "- market reactions: not calculated",
        "- review status: DRAFT only",
        "",
        (
            "This is an exploratory baseline against preliminary seed labels, "
            "not a final TEST benchmark."
        ),
    ]
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_review_queue(path: Path, processed: list[ProcessedSeedRecord]) -> None:
    grouped: dict[str, list[ProcessedSeedRecord]] = {
        category: [] for category in ("A", "B", "C", "D", "E")
    }
    for item in processed:
        grouped[item.category].append(item)
    lines = [
        "# Batch 001 Human Review Queue",
        "",
        f"Market reactions: forbidden/not calculated for {DATE_ONLY_MARKER} seed records.",
        "",
    ]
    headings = {
        "A": "A - extractor and seed gold fully agree",
        "B": "B - minor fact mismatch",
        "C": "C - event mismatch",
        "D": "D - multiple/ambiguous events",
        "E": "E - source/date/label requires manual verification",
    }
    for category in ("A", "B", "C", "D", "E"):
        lines.extend([f"## {headings[category]}", "", f"Count: {len(grouped[category])}", ""])
        for item in grouped[category]:
            if category == "A":
                lines.append(
                    f"- `{item.seed.record_id}` {item.seed.company} "
                    f"{item.seed.source_published_date}"
                )
                continue
            lines.extend(_review_record_lines(item))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _review_record_lines(item: ProcessedSeedRecord) -> list[str]:
    predicted_events = [event.event_type.value for event in item.analysis.events]
    gold_events = [event.event_type.value for event in item.seed.gold_events]
    return [
        f"### {item.seed.record_id}",
        "",
        f"- company: {item.seed.company}",
        f"- date: {item.seed.source_published_date}",
        f"- annotation_text: {item.seed.annotation_text}",
        f"- tickers: {', '.join(item.seed.tickers)}",
        f"- prediction event: {', '.join(predicted_events) or 'NONE'}",
        f"- gold event: {', '.join(gold_events) or 'NONE'}",
        f"- prediction facts: {_facts_text(item.analysis.financial_facts)}",
        f"- gold facts: {_gold_facts_text(item.seed.gold_financial_facts)}",
        f"- source URL: {item.seed.source.url}",
        f"- reason for review: {'; '.join(item.review_reasons)}",
        "",
    ]


def _stats(
    *,
    processed: list[ProcessedSeedRecord],
    records_total: int,
    created: int,
    already_exists: int,
    baseline_metrics: dict[str, object],
    validation_errors: list[str],
) -> SeedProcessingStats:
    source_review_required = [
        item.seed.record_id for item in processed if item.source_review_required
    ]
    return SeedProcessingStats(
        records_total=records_total,
        created=created,
        already_exists=already_exists,
        invalid=0,
        instrument_matches_total=sum(len(item.matches) for item in processed),
        records_with_instrument_matches=sum(1 for item in processed if item.matches),
        ambiguous_instrument_matches=sum(
            1 for item in processed for match in item.matches if match.is_ambiguous
        ),
        event_status_counts=dict(
            sorted(Counter(item.analysis.status.value for item in processed).items())
        ),
        primary_event_counts=dict(
            sorted(Counter(item.analysis.primary_event_type.value for item in processed).items())
        ),
        predicted_fact_count=sum(len(item.analysis.financial_facts) for item in processed),
        records_with_predicted_facts=sum(1 for item in processed if item.analysis.financial_facts),
        category_counts=dict(sorted(Counter(item.category for item in processed).items())),
        source_review_required=source_review_required,
        baseline_metrics=baseline_metrics,
        validation_errors=validation_errors,
    )


def _record_summary(item: ProcessedSeedRecord) -> dict[str, bool]:
    gold_events = {event.event_type for event in item.seed.gold_events}
    predicted_events = {event.event_type for event in item.analysis.events}
    perfect_event = (
        gold_events == predicted_events
        and _primary_gold_event(item.seed) == item.analysis.primary_event_type
    )
    fact_result = evaluate_fact_predictions(
        [FactEvaluationInput(item.seed.gold_financial_facts, item.analysis.financial_facts)]
    )
    strict_metrics = cast("dict[str, float]", fact_result.metrics["strict"])
    perfect_fact = strict_metrics["f1"] == 1.0
    return {
        "perfect_event_match": perfect_event,
        "event_mismatch": not perfect_event,
        "perfect_fact_match": perfect_fact,
    }


def _event_payload(event: DetectedEvent) -> dict[str, object]:
    return {
        "event_type": event.event_type.value,
        "confidence": str(event.confidence),
        "rule_id": event.rule_id,
        "matched_rule": event.matched_rule,
        "evidence_text": event.evidence_text,
        "start_position": event.start_position,
        "end_position": event.end_position,
    }


def _fact_payload(fact: ExtractedFinancialFact) -> dict[str, object]:
    return {
        "metric": fact.metric.value,
        "raw_value": str(fact.raw_value),
        "normalized_value": str(fact.normalized_value),
        "unit": fact.unit.value,
        "currency": fact.currency.value,
        "scale": fact.scale.value,
        "period_type": fact.period_type.value,
        "period_year": fact.year,
        "period_quarter": fact.quarter,
        "period_month": fact.month,
        "raw_period": fact.raw_period,
        "comparison_type": fact.comparison_type.value,
        "fact_role": fact.fact_role.value,
        "change_direction": fact.change_direction.value,
        "change_value": None if fact.change_value is None else str(fact.change_value),
        "change_unit": None if fact.change_unit is None else fact.change_unit.value,
        "confidence": str(fact.confidence),
        "rule_id": fact.rule_id,
        "evidence_text": fact.evidence_text,
        "start_position": fact.start_position,
        "end_position": fact.end_position,
        "extractor_version": fact.extractor_version,
        "matched_rule": fact.matched_rule,
    }


def _match_payload(match: NewsInstrumentMatch) -> dict[str, object]:
    return {
        "instrument_id": str(match.instrument_id),
        "matched_alias": match.matched_alias,
        "alias_type": match.alias_type.value,
        "match_type": match.match_type.value,
        "confidence": match.confidence,
        "start_position": match.start_position,
        "end_position": match.end_position,
        "is_ambiguous": match.is_ambiguous,
        "matcher_version": match.matcher_version,
    }


def _facts_text(facts: list[ExtractedFinancialFact]) -> str:
    if not facts:
        return "NONE"
    return "; ".join(
        f"{fact.metric.value}={fact.normalized_value} {fact.currency.value} "
        f"{fact.period_type.value}/{fact.year} role={fact.fact_role.value}"
        for fact in facts
    )


def _gold_facts_text(facts: list[GoldFinancialFact]) -> str:
    if not facts:
        return "NONE"
    return "; ".join(
        f"{fact.metric.value}={fact.normalized_value} {fact.currency.value} "
        f"{fact.period_type.value}/{fact.period_year} role={fact.fact_role.value}"
        for fact in facts
    )


def _required_dict(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be object")
    return cast("dict[str, object]", value)


def _required_list(payload: dict[str, object], key: str) -> list[object]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be list")
    return cast("list[object]", value)


def _required_list_of_dicts(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for item in _required_list(payload, key):
        if not isinstance(item, dict):
            raise ValueError(f"{key} items must be objects")
        result.append(cast("dict[str, object]", item))
    return result
