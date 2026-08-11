from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.corpus_quality.domain import (
    CORPUS_QUALITY_VERSION,
    CORPUS_VERSION,
    FROZEN_BATCH_001_GOLD_SHA256,
    PublicationTimeRecord,
    ShadowPrediction,
    SourceAcceptanceEvidence,
    UnknownDiagnosis,
    build_baseline,
    cumulative_funnel,
    distributions,
    diversity_warnings,
    readiness_report,
    rules_vs_shadow,
    select_annotation_batch,
)


def write_quality_artifacts(
    output_dir: Path,
    v2_output_dir: Path,
    *,
    baseline_records: list[PublicationTimeRecord],
    cumulative_records: list[PublicationTimeRecord],
    source_evidence: list[SourceAcceptanceEvidence],
    batch_001_gold_path: Path,
    shadow_predictions: list[ShadowPrediction] | None = None,
    rss_audit: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    v2_output_dir.mkdir(parents=True, exist_ok=True)
    baseline = build_baseline(baseline_records)
    diagnoses = [diagnose.payload() for diagnose in _diagnoses(baseline_records)]
    batch = select_annotation_batch(cumulative_records)
    comparisons = rules_vs_shadow(baseline_records, shadow_predictions or [])
    tickers, events = distributions(cumulative_records)
    total = len(cumulative_records)
    unknown_count = events.get("UNKNOWN", 0)
    unknown_rate = 0.0 if total == 0 else unknown_count / total
    warnings = diversity_warnings(tickers, events)
    readiness = readiness_report(
        reaction_rows=sum(item.reaction_ready for item in cumulative_records),
        annotation_rows=len(batch),
        tickers=len(tickers),
        unknown_rate=unknown_rate,
    )
    accepted_sources = [item for item in source_evidence if item.compliant]
    batch_001_hash = _sha256_file(batch_001_gold_path)
    if batch_001_hash != FROZEN_BATCH_001_GOLD_SHA256:
        raise ValueError("frozen Batch 001 gold checksum changed")
    manifest = {
        "schema_version": CORPUS_VERSION,
        "corpus_quality_version": CORPUS_QUALITY_VERSION,
        "funnel": cumulative_funnel(cumulative_records),
        "deterministic_event_known": total - unknown_count,
        "deterministic_event_unknown": unknown_count,
        "AI_shadow_items": len(shadow_predictions or []),
        "AI_shadow_event_known": sum(
            item.successful and item.primary_event != "UNKNOWN"
            for item in (shadow_predictions or [])
        ),
        "event_distribution": events,
        "ticker_distribution": tickers,
        "unknown_rate": unknown_rate,
        "label_availability": {
            f"{horizon}m": sum(horizon in item.valid_label_horizons for item in cumulative_records)
            for horizon in (1, 5, 15, 30, 60)
        },
        "source_audits_total": len(source_evidence),
        "compliant_live_sources_total": len(accepted_sources),
        "approved_source_codes": [item.source_code for item in accepted_sources],
        "annotation_batch_002_rows": len(batch),
        "readiness": readiness,
        "diversity_warnings": warnings,
        "selection_uses_future_returns": False,
        "batch_001_reaction_count": 0,
        "batch_001_gold_sha256": batch_001_hash,
        "deterministic_rules_changed": False,
        "qwen_configuration_changed": False,
        "hybrid_enabled": False,
        "ml_training_performed": False,
    }
    coverage = {
        "schema_version": CORPUS_VERSION,
        "PRIMARY_EVENT": events,
        "ticker": tickers,
        "unknown_count": unknown_count,
        "unknown_rate": unknown_rate,
        "warnings": warnings,
        "label_availability": manifest["label_availability"],
    }
    source_payload = {
        "schema_version": CORPUS_QUALITY_VERSION,
        "sources": [item.payload() for item in source_evidence],
        "compliant_live_sources_total": len(accepted_sources),
    }
    paths = {
        "baseline": output_dir / "rosn-baseline.json",
        "diagnosis_jsonl": output_dir / "unknown-diagnosis.jsonl",
        "diagnosis_markdown": output_dir / "unknown-diagnosis.md",
        "source_audit": output_dir / "source-expansion-audit.json",
        "rss_audit": output_dir / "rosneft-rss-content-audit.json",
        "annotation_batch": output_dir / "annotation-batch-002.jsonl",
        "disagreements": output_dir / "rules-vs-qwen-shadow.jsonl",
        "manifest": v2_output_dir / "manifest.json",
        "coverage": v2_output_dir / "coverage.json",
    }
    _write_json(paths["baseline"], baseline)
    _write_jsonl(paths["diagnosis_jsonl"], diagnoses)
    _write_diagnosis_markdown(paths["diagnosis_markdown"], diagnoses)
    _write_json(paths["source_audit"], source_payload)
    _write_json(paths["rss_audit"], rss_audit or {})
    _write_jsonl(paths["annotation_batch"], batch)
    _write_jsonl(paths["disagreements"], comparisons)
    _write_json(paths["manifest"], manifest)
    _write_json(paths["coverage"], coverage)
    return paths


def diagnosis_counts(diagnoses: list[UnknownDiagnosis]) -> dict[str, int]:
    return dict(sorted(Counter(item.category.value for item in diagnoses).items()))


def _diagnoses(records: list[PublicationTimeRecord]) -> list[UnknownDiagnosis]:
    from src.corpus_quality.domain import diagnose_unknown

    return [diagnose_unknown(item) for item in records]


def _write_diagnosis_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(item["diagnostic_category"]) for item in rows)
    lines = [
        "# ROSN UNKNOWN diagnosis",
        "",
        "This is a publication-time diagnostic, not gold annotation. No returns are used.",
        "",
        "## Counts",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in sorted(counts.items()))
    lines.extend(["", "## Records", ""])
    for item in rows:
        lines.extend(
            [
                f"### {item['news_id']}",
                "",
                f"- Category: {item['diagnostic_category']}",
                f"- Title: {item['title']}",
                f"- Content length: {item['content_length']}",
                f"- Rationale: {item['rationale']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
