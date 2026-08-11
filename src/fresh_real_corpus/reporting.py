from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC
from pathlib import Path
from typing import Any

from src.fresh_real_corpus.domain import (
    ANNOTATION_BATCH_VERSION,
    BATCH_003_DATASET,
    BATCH_003_STATUS,
    CORPUS_VERSION,
    CorpusSplit,
    FreshCorpusRecord,
    FrozenSplit,
    SelectionPolicy,
    SelectionResult,
    coverage_payload,
)
from src.official_sources.domain import OfficialSourceConfig, OfficialSourceStatus

EXPECTED_BATCH_001_SHA256 = "4934b37b1c036eedb6191dae5ece2fa49e710d00455576cee3de081cc9e7c196"


def write_fresh_corpus_artifacts(
    output: Path,
    *,
    annotation_copy: Path,
    result: SelectionResult,
    split: FrozenSplit,
    policy: SelectionPolicy,
    source_configs: tuple[OfficialSourceConfig, ...],
    batch_001_path: Path,
    git_sha: str,
) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    annotation_copy.parent.mkdir(parents=True, exist_ok=True)
    records = tuple(sorted(result.records, key=lambda item: item.published_at))
    annotation_rows = [
        record.annotation_payload(split.split_for(record.news_id)) for record in records
    ]
    annotation_path = output / "annotation-batch-004.jsonl"
    _write_jsonl(annotation_path, annotation_rows)
    annotation_copy.write_bytes(annotation_path.read_bytes())
    coverage = coverage_payload(records)
    coverage_path = output / "coverage.json"
    _write_json(coverage_path, coverage)
    split_path = output / "split-manifest.json"
    split_payload = _split_payload(records, split)
    _write_json(split_path, split_payload)
    batch_001_sha = _sha256_file(batch_001_path)
    if batch_001_sha != EXPECTED_BATCH_001_SHA256:
        raise ValueError("frozen Batch 001 SHA-256 changed")
    compliant = [
        item.source_code
        for item in source_configs
        if item.status == OfficialSourceStatus.REACTION_READY
    ]
    manifest = {
        "schema_version": CORPUS_VERSION,
        "annotation_batch": ANNOTATION_BATCH_VERSION,
        "created_at": policy.normalized().date_to.isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha,
        "records": len(records),
        "target_records": 50,
        "target_status": "MET" if len(records) >= 50 else "SOURCE_DEPTH_BLOCKER",
        "target_blocker": None
        if len(records) >= 50
        else "Approved one-page issuer feeds do not expose 50 fresh non-overlapping items.",
        "provenance": "REAL",
        "timestamp_quality": {"EXACT": len(records)},
        "development_count": sum(
            split.split_for(item.news_id) == CorpusSplit.DEVELOPMENT for item in records
        ),
        "fresh_holdout_count": sum(
            split.split_for(item.news_id) == CorpusSplit.FRESH_HOLDOUT for item in records
        ),
        "split_sha256": split.split_sha256,
        "annotation_sha256": _sha256_file(annotation_path),
        "batch_003_dataset": BATCH_003_DATASET,
        "batch_003_status": BATCH_003_STATUS,
        "batch_003_tuning_prohibited": True,
        "batch_001_sha256": batch_001_sha,
        "batch_001_unchanged": True,
        "excluded_previous_batch_overlaps": result.excluded_overlap_count,
        "duplicate_with_batch_003_count": 0,
        "source_audits_total": len(source_configs),
        "compliant_sources": compliant,
        "compliant_sources_total": len(compliant),
        "selection": {
            "sources": list(policy.normalized().source_codes),
            "from": policy.normalized().date_from.isoformat(),
            "to": policy.normalized().date_to.isoformat(),
            "limit": policy.normalized().limit,
            "order": policy.source_order,
            "uses_rules_or_qwen": False,
            "uses_future_returns": False,
        },
        "matched": sum(item.match_status.value == "MATCHED" for item in records),
        "ambiguous": sum(item.match_status.value == "AMBIGUOUS" for item in records),
        "unmatched": sum(item.match_status.value == "UNMATCHED" for item in records),
        "reaction_ready": sum(item.reaction_ready for item in records),
        "feature_ready": sum(item.feature_ready for item in records),
        "ticker_distribution": dict(sorted(Counter(item.ticker for item in records).items())),
        "source_distribution": dict(sorted(Counter(item.source_code for item in records).items())),
        "date_range": coverage["date_range"],
        "holdout_predictions_generated": False,
        "model_outputs_used_for_selection": False,
        "future_returns_used_for_selection": False,
        "human_labels_generated": False,
        "event_distribution": "UNAVAILABLE_BEFORE_HUMAN_REVIEW",
        "rules_changed": False,
        "qwen_changed": False,
        "hybrid_enabled": False,
        "ml_training_performed": False,
        "backtest_performed": False,
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "coverage": coverage_path,
        "split_manifest": split_path,
        "annotation": annotation_path,
        "annotation_copy": annotation_copy,
    }


def _split_payload(records: tuple[FreshCorpusRecord, ...], split: FrozenSplit) -> dict[str, Any]:
    assignments = [
        {
            "news_id": str(item.news_id),
            "source": item.source_code,
            "source_item_id": item.source_item_id,
            "published_at": item.published_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "split": split.split_for(item.news_id).value,
        }
        for item in records
    ]
    development = [
        item for item in records if split.split_for(item.news_id) == CorpusSplit.DEVELOPMENT
    ]
    holdout = [
        item for item in records if split.split_for(item.news_id) == CorpusSplit.FRESH_HOLDOUT
    ]
    return {
        "schema_version": CORPUS_VERSION,
        "strategy": "TEMPORAL_70_30",
        "frozen_before_model_changes": True,
        "split_sha256": split.split_sha256,
        "development": _range_payload(development),
        "fresh_holdout": _range_payload(holdout),
        "assignments": assignments,
    }


def _range_payload(records: list[FreshCorpusRecord]) -> dict[str, Any]:
    if not records:
        return {"count": 0, "date_range": {"from": None, "to": None}}
    ordered = sorted(records, key=lambda item: item.published_at)
    return {
        "count": len(ordered),
        "date_range": {
            "from": ordered[0].published_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "to": ordered[-1].published_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
