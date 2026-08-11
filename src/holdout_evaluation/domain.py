from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from src.evaluation.domain.entities import GoldEvent
from src.events.domain.enums import EventType
from src.events.domain.v3 import rules_v3_fingerprint

HOLDOUT_DATASET_NAME = "ru-corporate-events-real-batch-004-holdout-gold-v1"
HOLDOUT_SCHEMA_VERSION = "real-holdout-gold-v1"
EXPECTED_CANDIDATE_NAME = "event-rules-v3-real-dev-candidate"
EXPECTED_RULES_FINGERPRINT = "3510511d1f7b3ce02a4efa245816b9422e6014088f1595b0339dcfd5be9e7f06"
EXPECTED_SPLIT_SHA256 = "a32956626d194158eb69869f6bdca510456ded47ac5810ca91fe90b86aa45dea"


@dataclass(frozen=True, slots=True)
class FrozenCandidate:
    name: str
    rules_fingerprint: str
    development_gold_sha256: str
    git_sha: str
    split_sha256: str


@dataclass(frozen=True, slots=True)
class HoldoutGoldRecord:
    news_id: UUID
    annotation_text: str
    raw_content_hash: str
    primary_event: EventType
    events: tuple[GoldEvent, ...]
    source_payload: dict[str, Any]

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "annotation_text": self.annotation_text,
            "dataset_name": HOLDOUT_DATASET_NAME,
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
            "gold_financial_facts": [],
            "gold_primary_event": self.primary_event.value,
            "news_id": str(self.news_id),
            "provenance": "REAL",
            "published_at": self.source_payload["published_at"],
            "purpose": "FRESH_HOLDOUT",
            "raw_content_hash": self.raw_content_hash,
            "review_basis": "EXCERPT_ONLY",
            "schema_version": HOLDOUT_SCHEMA_VERSION,
            "source": self.source_payload["source"],
            "source_item_id": self.source_payload["source_item_id"],
            "ticker": self.source_payload["ticker"],
        }


@dataclass(frozen=True, slots=True)
class HoldoutGoldDataset:
    records: tuple[HoldoutGoldRecord, ...]
    source_review_sha256: str
    dataset_sha256: str
    split_sha256: str

    @property
    def event_distribution(self) -> dict[str, int]:
        return dict(sorted(Counter(item.primary_event.value for item in self.records).items()))


def verify_frozen_candidate(path: Path) -> FrozenCandidate:
    payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    candidate = FrozenCandidate(
        name=str(payload.get("candidate_name", "")),
        rules_fingerprint=str(payload.get("rules_fingerprint_sha256", "")),
        development_gold_sha256=str(payload.get("development_gold_sha256", "")),
        git_sha=str(payload.get("git_sha", "")),
        split_sha256=str(payload.get("split_sha256", "")),
    )
    if payload.get("frozen") is not True or candidate.name != EXPECTED_CANDIDATE_NAME:
        raise ValueError("event-rules-v3 candidate is not the expected frozen candidate")
    if candidate.rules_fingerprint != EXPECTED_RULES_FINGERPRINT:
        raise ValueError("frozen candidate rules fingerprint mismatch")
    if rules_v3_fingerprint() != EXPECTED_RULES_FINGERPRINT:
        raise ValueError("current event-rules-v3 fingerprint differs from frozen candidate")
    if candidate.split_sha256 != EXPECTED_SPLIT_SHA256:
        raise ValueError("frozen candidate split SHA mismatch")
    return candidate


def freeze_holdout_gold(
    *, source_review_path: Path, split_manifest_path: Path, output_directory: Path
) -> HoldoutGoldDataset:
    split_sha, holdout_ids = _load_holdout_split_metadata(split_manifest_path)
    source_bytes = source_review_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    records = _load_review_records(source_review_path)
    if {item.news_id for item in records} != holdout_ids:
        raise ValueError("human review records do not exactly match frozen FRESH_HOLDOUT ids")
    canonical_rows = [item.canonical_payload() for item in records]
    canonical_text = "".join(_stable_json(item) + "\n" for item in canonical_rows)
    dataset_sha = hashlib.sha256(canonical_text.encode()).hexdigest()
    dataset = HoldoutGoldDataset(
        records=records,
        source_review_sha256=source_sha,
        dataset_sha256=dataset_sha,
        split_sha256=split_sha,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "dataset.jsonl").write_text(canonical_text, encoding="utf-8")
    _write_json(
        output_directory / "manifest.json",
        {
            "canonical_dataset_sha256": dataset_sha,
            "dataset_name": HOLDOUT_DATASET_NAME,
            "event_distribution": dataset.event_distribution,
            "financial_fact_count": 0,
            "observed": False,
            "provenance": "REAL",
            "purpose": "FRESH_HOLDOUT",
            "records": len(records),
            "review_basis": "EXCERPT_ONLY",
            "schema_version": HOLDOUT_SCHEMA_VERSION,
            "source_review_sha256": source_sha,
            "split_sha256": split_sha,
        },
    )
    return dataset


def _load_holdout_split_metadata(path: Path) -> tuple[str, set[UUID]]:
    payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    split_sha = str(payload.get("split_sha256", ""))
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise ValueError("frozen split SHA mismatch")
    assignments = cast("list[dict[str, Any]]", payload.get("assignments", []))
    holdout_ids = {
        UUID(str(item["news_id"])) for item in assignments if item.get("split") == "FRESH_HOLDOUT"
    }
    if len(holdout_ids) != 4:
        raise ValueError("frozen split must contain exactly four FRESH_HOLDOUT ids")
    return split_sha, holdout_ids


def _load_review_records(path: Path) -> tuple[HoldoutGoldRecord, ...]:
    payloads = [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(payloads) != 4:
        raise ValueError("HOLDOUT human review must contain exactly four records")
    records = tuple(_review_record(item) for item in payloads)
    if Counter(item.primary_event.value for item in records) != {"OTHER": 4}:
        raise ValueError("HOLDOUT gold distribution must be OTHER=4")
    return records


def _review_record(payload: dict[str, Any]) -> HoldoutGoldRecord:
    if payload.get("provenance") != "REAL":
        raise ValueError("HOLDOUT review requires REAL provenance")
    if payload.get("human_review_status") != "REVIEWED":
        raise ValueError("every HOLDOUT record must be REVIEWED")
    if payload.get("human_review_basis") != "annotation_text_excerpt_only":
        raise ValueError("every HOLDOUT record must use excerpt-only review")
    if payload.get("is_gold") is not False:
        raise ValueError("source HOLDOUT review must remain non-gold")
    if cast("list[dict[str, Any]]", payload.get("human_financial_facts", [])):
        raise ValueError("HOLDOUT review must contain zero financial facts")
    text = str(payload["annotation_text"])
    expected_hash = hashlib.sha256(text.encode()).hexdigest()
    if payload.get("raw_content_hash") != expected_hash:
        raise ValueError("HOLDOUT annotation text hash mismatch")
    primary = EventType(str(payload["human_primary_event"]))
    event_types = tuple(EventType(str(item)) for item in cast("list[str]", payload["human_events"]))
    if primary != EventType.OTHER or event_types != (EventType.OTHER,):
        raise ValueError("HOLDOUT gold event must be exactly OTHER")
    event = GoldEvent(
        event_type=EventType.OTHER,
        evidence_text=text,
        start_position=0,
        end_position=len(text),
        is_primary=True,
        notes=str(payload.get("human_review_notes") or "") or None,
    )
    return HoldoutGoldRecord(
        news_id=UUID(str(payload["news_id"])),
        annotation_text=text,
        raw_content_hash=expected_hash,
        primary_event=primary,
        events=(event,),
        source_payload=payload,
    )


def _stable_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
