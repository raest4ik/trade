from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from src.evaluation.domain.metrics import EventEvaluationInput, evaluate_event_predictions
from src.events.domain.entities import NewsEventAnalysis
from src.holdout_evaluation.domain import (
    EXPECTED_RULES_FINGERPRINT,
    FrozenCandidate,
    HoldoutGoldDataset,
    HoldoutGoldRecord,
)


class HoldoutAnalyzer(Protocol):
    def analyze(self, *, news_id: Any, raw_content: str) -> NewsEventAnalysis: ...


@dataclass(frozen=True, slots=True)
class HoldoutEvaluationResult:
    metrics: dict[str, Any]
    predictions: tuple[dict[str, Any], ...]
    record_results: tuple[dict[str, Any], ...]


def claim_single_run(output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    marker = output_directory / "single-run.marker"
    if marker.exists():
        raise ValueError("blind HOLDOUT evaluation has already been claimed")
    marker.write_text(
        json.dumps(
            {
                "claimed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "rules_fingerprint": EXPECTED_RULES_FINGERPRINT,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker


def run_single_holdout_evaluation(
    dataset: HoldoutGoldDataset,
    analyzer: HoldoutAnalyzer,
) -> HoldoutEvaluationResult:
    analyses = tuple(
        analyzer.analyze(news_id=record.news_id, raw_content=record.annotation_text)
        for record in dataset.records
    )
    inputs = [
        EventEvaluationInput(
            gold_events=record.events,
            predicted_events=analysis.events,
            gold_primary_event_type=record.primary_event,
            predicted_primary_event_type=analysis.primary_event_type,
            prediction_status=analysis.status.value,
        )
        for record, analysis in zip(dataset.records, analyses, strict=True)
    ]
    evaluated = evaluate_event_predictions(inputs)
    record_results = tuple(
        {
            "correct": record.primary_event == analysis.primary_event_type,
            "gold": record.primary_event.value,
            "news_id": str(record.news_id),
            "prediction": analysis.primary_event_type.value,
        }
        for record, analysis in zip(dataset.records, analyses, strict=True)
    )
    predictions = tuple(
        _prediction_payload(record, analysis)
        for record, analysis in zip(dataset.records, analyses, strict=True)
    )
    return HoldoutEvaluationResult(
        metrics={
            **evaluated.metrics,
            "dataset_sha256": dataset.dataset_sha256,
            "holdout_status": "OBSERVED_HOLDOUT",
            "statistical_uncertainty": "VERY_HIGH_N_EQUALS_4",
        },
        predictions=predictions,
        record_results=record_results,
    )


def write_holdout_artifacts(
    *,
    output_directory: Path,
    dataset: HoldoutGoldDataset,
    candidate: FrozenCandidate,
    result: HoldoutEvaluationResult,
) -> None:
    _write_jsonl(output_directory / "predictions.jsonl", result.predictions)
    _write_jsonl(output_directory / "record-results.jsonl", result.record_results)
    _write_json(output_directory / "metrics.json", result.metrics)
    _write_json(
        output_directory / "run-manifest.json",
        {
            "candidate": candidate.name,
            "candidate_git_sha": candidate.git_sha,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "dataset_sha256": dataset.dataset_sha256,
            "evaluation_runs": 1,
            "git_sha": _git_sha(),
            "holdout_status": "OBSERVED_HOLDOUT",
            "hybrid": False,
            "next_project_priority": "HISTORICAL_DATA_ACQUISITION",
            "nlp_development_cycle": "CLOSED",
            "post_holdout_tuning_allowed": False,
            "rules_changed_after_freeze": False,
            "rules_fingerprint_sha256": candidate.rules_fingerprint,
            "split_sha256": dataset.split_sha256,
        },
    )
    gold_manifest_path = output_directory / "holdout-gold" / "manifest.json"
    gold_manifest = json.loads(gold_manifest_path.read_text(encoding="utf-8"))
    gold_manifest["observed"] = True
    gold_manifest["status"] = "OBSERVED_HOLDOUT"
    _write_json(gold_manifest_path, gold_manifest)


def _prediction_payload(record: HoldoutGoldRecord, analysis: NewsEventAnalysis) -> dict[str, Any]:
    return {
        "analysis_version": analysis.analysis_version,
        "events": [
            {
                "event_type": item.event_type.value,
                "evidence_text": item.evidence_text,
                "rule_id": item.rule_id,
            }
            for item in analysis.events
        ],
        "financial_fact_count": len(analysis.financial_facts),
        "news_id": str(record.news_id),
        "primary_event": analysis.primary_event_type.value,
        "rules_fingerprint_sha256": EXPECTED_RULES_FINGERPRINT,
        "status": analysis.status.value,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, payloads: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            for item in payloads
        ),
        encoding="utf-8",
    )


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
