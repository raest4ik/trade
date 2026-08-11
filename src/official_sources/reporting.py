from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from src.corpus_quality.domain import (
    PublicationTimeRecord,
    cumulative_funnel,
    distributions,
    diversity_warnings,
    readiness_report,
    select_annotation_batch,
)
from src.ml_features.domain.entities import FeatureDatasetRow
from src.official_sources.domain import (
    ANNOTATION_BATCH_VERSION,
    CORPUS_VERSION,
    OfficialSourceConfig,
    audit_payload,
)
from src.reaction_ready_corpus.domain import CorpusProvenance, classify_provenance

FROZEN_BATCH_002_SHA256 = "358ea17184a6328283147e4c423db6d825147dfeed3add789a9bc2aef86c3159"


def write_official_source_corpus(
    output_dir: Path,
    *,
    records: list[PublicationTimeRecord],
    feature_rows: list[FeatureDatasetRow],
    source_configs: tuple[OfficialSourceConfig, ...],
    git_sha: str,
    batch_001_reactions: int,
    batch_002_path: Path,
    annotation_output: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_output.parent.mkdir(parents=True, exist_ok=True)
    batch_002_hash = _sha256_file(batch_002_path)
    if batch_002_hash != FROZEN_BATCH_002_SHA256:
        raise ValueError("Batch 002 changed after it was frozen for human review")
    real_rows = [
        row
        for row in feature_rows
        if classify_provenance(str(row.metadata.get("source", ""))) == CorpusProvenance.REAL
    ]
    _, events = distributions(records)
    tickers = represented_ticker_distribution(records)
    sources = dict(sorted(Counter(item.source_code for item in records).items()))
    months = dict(sorted(Counter(item.published_at.strftime("%Y-%m") for item in records).items()))
    total = len(records)
    unknown_count = events.get("UNKNOWN", 0)
    unknown_rate = 0.0 if total == 0 else unknown_count / total
    warnings = diversity_warnings(tickers, events)
    batch = (
        select_annotation_batch(
            records,
            limit=50,
            batch_version=ANNOTATION_BATCH_VERSION,
            record_prefix="batch-003",
        )
        if total >= 20
        else []
    )
    labels = {
        f"{horizon}m": sum(horizon in item.valid_label_horizons for item in records)
        for horizon in (1, 5, 15, 30, 60)
    }
    readiness = readiness_report(
        reaction_rows=sum(item.reaction_ready for item in records),
        annotation_rows=len(batch),
        tickers=len(tickers),
        unknown_rate=unknown_rate,
    )
    funnel = _funnel_payload(records)
    manifest = {
        "schema_version": CORPUS_VERSION,
        "git_sha": git_sha,
        "funnel": funnel["overall"],
        "REAL_discovered": total,
        "REAL_EXACT": sum(item.timestamp_quality.value == "EXACT" for item in records),
        "matched": sum(item.matched for item in records),
        "ambiguous": 0,
        "unmatched": sum(not item.matched for item in records),
        "market_data_ready": sum(item.market_data_ready for item in records),
        "reaction_ready": sum(item.reaction_ready for item in records),
        "feature_ready": len(real_rows),
        "deterministic_event_known": total - unknown_count,
        "deterministic_event_unknown": unknown_count,
        "unknown_rate": unknown_rate,
        "event_distribution": events,
        "ticker_distribution": tickers,
        "source_distribution": sources,
        "month_distribution": months,
        "label_availability": labels,
        "annotation_batch_003_rows": len(batch),
        "diversity_warnings": warnings,
        "readiness": readiness,
        "batch_001_reaction_count": batch_001_reactions,
        "batch_002_sha256": batch_002_hash,
        "batch_002_unchanged": True,
        "synthetic_rows_counted_as_real": 0,
        "seed_rows_counted_as_real": 0,
        "selection_uses_rules": False,
        "selection_uses_ai": False,
        "selection_uses_future_returns": False,
        "deterministic_rules_changed": False,
        "qwen_configuration_changed": False,
        "hybrid_enabled": False,
        "ml_training_performed": False,
    }
    coverage = {
        "schema_version": CORPUS_VERSION,
        "ticker_distribution": tickers,
        "source_distribution": sources,
        "event_distribution": events,
        "month_distribution": months,
        "unknown_count": unknown_count,
        "unknown_rate": unknown_rate,
        "label_availability": labels,
        "diversity_warnings": warnings,
    }
    paths = {
        "manifest": output_dir / "manifest.json",
        "coverage": output_dir / "coverage.json",
        "funnel": output_dir / "funnel.json",
        "source_audit": output_dir / "source-audit.json",
        "corpus": output_dir / "corpus.jsonl",
        "annotation_batch": annotation_output,
    }
    _write_json(paths["manifest"], manifest)
    _write_json(paths["coverage"], coverage)
    _write_json(paths["funnel"], funnel)
    _write_json(paths["source_audit"], audit_payload(source_configs))
    _write_jsonl(paths["corpus"], [_feature_payload(row) for row in real_rows])
    _write_jsonl(paths["annotation_batch"], batch)
    return paths


def _funnel_payload(records: list[PublicationTimeRecord]) -> dict[str, Any]:
    tickers = sorted(represented_ticker_distribution(records))
    sources = sorted({item.source_code for item in records})
    return {
        "schema_version": CORPUS_VERSION,
        "overall": cumulative_funnel(records),
        "by_ticker": {
            ticker: cumulative_funnel([item for item in records if item.ticker == ticker])
            for ticker in tickers
        },
        "by_source": {
            source: cumulative_funnel([item for item in records if item.source_code == source])
            for source in sources
        },
    }


def represented_ticker_distribution(
    records: list[PublicationTimeRecord],
) -> dict[str, int]:
    return dict(sorted(Counter(item.ticker for item in records if item.matched).items()))


def _feature_payload(row: FeatureDatasetRow) -> dict[str, Any]:
    metadata = row.metadata
    return _json_value(
        {
            "schema_version": CORPUS_VERSION,
            "metadata": {
                "news_id": metadata["news_id"],
                "ticker": metadata["ticker"],
                "published_at": metadata["published_at"],
                "source": metadata["source"],
                "source_item_id": metadata["source_item_id"],
                "provenance": CorpusProvenance.REAL.value,
                "feature_version": metadata["feature_version"],
                "reaction_version": metadata["reaction_version"],
            },
            "features_available_at_publication": row.features,
            "labels": row.labels,
            "quality": row.quality,
        }
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: object) -> Any:
    if isinstance(value, (date, datetime, Decimal, UUID)):
        return str(value)
    if isinstance(value, dict):
        items = cast("dict[object, object]", value)
        return {str(key): _json_value(item) for key, item in items.items()}
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return [_json_value(item) for item in items]
    return value
