from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.daily_corpus.application import DailyCorpusBuildResult
from src.daily_corpus.domain import (
    DATASET_VERSION,
    FEATURE_VERSION,
    LABEL_FAMILY,
    MAX_HISTORICAL_IMPORT,
    REACTION_VERSION,
    SourceAcceptanceStatus,
    SourceVerification,
    daily_readiness,
    deterministic_temporal_split,
)


def write_daily_corpus_reports(
    output_dir: Path,
    *,
    result: DailyCorpusBuildResult,
    verifications: tuple[SourceVerification, ...],
    intraday: dict[str, int],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    for verification in verifications:
        verification.validate()
    verification_payload = {
        "schema_version": DATASET_VERSION,
        "generated_at": generated_at,
        "sample_limit_per_source": 20,
        "sampling_uses_model_predictions": False,
        "sampling_uses_market_returns": False,
        "estimated_items": sum(item.estimated_items for item in verifications),
        "verified_accessible_items": sum(item.verified_accessible_items for item in verifications),
        "verified_date_items": sum(item.verified_date_items for item in verifications),
        "verified_exact_items": sum(item.verified_exact_items for item in verifications),
        "verified_daily_eligible_items": sum(
            item.verified_daily_eligible_items for item in verifications
        ),
        "new_compliant_date_safe_daily_sources": [
            item.source_code
            for item in verifications
            if item.status == SourceAcceptanceStatus.COMPLIANT_DATE_SAFE_DAILY
        ],
        "new_compliant_exact_sources": [
            item.source_code
            for item in verifications
            if item.status == SourceAcceptanceStatus.COMPLIANT_EXACT
        ],
        "sources": [item.payload() for item in verifications],
        "paid_services_used": False,
        "purchases_required": False,
        "access_restrictions_bypassed": False,
    }
    imported: list[dict[str, Any]] = []
    import_payload = {
        "schema_version": DATASET_VERSION,
        "generated_at": generated_at,
        "max_new_real": MAX_HISTORICAL_IMPORT,
        "new_real": len(imported),
        "new_date_only": 0,
        "new_exact": 0,
        "selection_order": "ticker, source, publication_date, source_item_id",
        "selection_uses_model_predictions": False,
        "selection_uses_event_type": False,
        "selection_uses_market_returns": False,
        "outcome": "NO_NEW_SOURCE_PASSED_POLICY_VERIFICATION",
        "records": imported,
    }
    reactions_payload = [item.payload() for item in result.reactions]
    features_payload = [item.payload() for item in result.features]
    exact = sum(item.timestamp_quality.value == "EXACT" for item in result.candidates)
    date_only = sum(item.timestamp_quality.value == "DATE_ONLY" for item in result.candidates)
    unmatched = sum(item.match_count == 0 for item in result.candidates)
    ambiguous = (
        sum(item.match_count != 1 or item.ambiguous_match for item in result.candidates) - unmatched
    )
    matched = len(result.candidates) - unmatched - ambiguous
    ticker_counts = Counter(item.ticker for item in result.eligible if item.ticker is not None)
    reaction_ticker_counts = Counter(item.ticker for item in result.reactions)
    feature_ticker_counts = Counter(item.ticker for item in result.features)
    source_counts = Counter(item.source_code for item in result.candidates)
    feature_source_counts = Counter(item.source for item in result.features)
    year_counts = Counter(
        str(item.publication_date.year)
        for item in result.eligible
        if item.publication_date is not None
    )
    reaction_year_counts = Counter(str(item.publication_date.year) for item in result.reactions)
    reaction_timestamp_counts = Counter(item.timestamp_quality.value for item in result.reactions)
    month_counts = Counter(
        item.publication_date.strftime("%Y-%m")
        for item in result.eligible
        if item.publication_date is not None
    )
    feature_month_counts = Counter(
        item.publication_date.strftime("%Y-%m") for item in result.features
    )
    text_lengths = [item.text_length for item in result.candidates]
    exclusion_counts = Counter(reason.value for reason in result.exclusions.values())
    coverage = {
        "schema_version": DATASET_VERSION,
        "generated_at": generated_at,
        "label_family": LABEL_FAMILY,
        "reaction_version": REACTION_VERSION,
        "feature_version": FEATURE_VERSION,
        "total": len(result.candidates),
        "provenance_real": sum(item.provenance == "REAL" for item in result.candidates),
        "timestamp_quality": {"EXACT": exact, "DATE_ONLY": date_only},
        "matched": matched,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "daily_eligible": len(result.eligible),
        "daily_reaction_ready": len(result.reactions),
        "daily_feature_ready": len(result.features),
        "per_ticker": dict(sorted(ticker_counts.items())),
        "daily_reaction_per_ticker": dict(sorted(reaction_ticker_counts.items())),
        "daily_feature_per_ticker": dict(sorted(feature_ticker_counts.items())),
        "per_source": dict(sorted(source_counts.items())),
        "daily_feature_per_source": dict(sorted(feature_source_counts.items())),
        "per_year": dict(sorted(year_counts.items())),
        "daily_reaction_per_year": dict(sorted(reaction_year_counts.items())),
        "daily_reaction_timestamp_quality": dict(sorted(reaction_timestamp_counts.items())),
        "per_month": dict(sorted(month_counts.items())),
        "daily_feature_per_month": dict(sorted(feature_month_counts.items())),
        "median_text_length": statistics.median(text_lengths) if text_lengths else 0,
        "missing_market_data": sum(
            count
            for reason, count in exclusion_counts.items()
            if "MARKET_DATA" in reason or "SESSION_WINDOW" in reason
        ),
        "duplicate_count": sum(item.duplicate for item in result.candidates),
        "exclusions": dict(sorted(exclusion_counts.items())),
        "intraday": intraday,
    }
    readiness = {
        "schema_version": DATASET_VERSION,
        "generated_at": generated_at,
        **daily_readiness(
            len(result.features),
            ticker_count=len(ticker_counts),
            source_count=len(feature_source_counts),
            month_count=len(feature_month_counts),
        ),
        "daily_eligible": len(result.eligible),
        "daily_reaction_ready": len(result.reactions),
        "path_to_100": max(0, 100 - len(result.features)),
        "path_to_500": max(0, 500 - len(result.features)),
        "path_to_1000": max(0, 1000 - len(result.features)),
        "remaining_blocker": (
            "NO_NEW_ARCHIVE_HAS_VERIFIED_AUTOMATION_AND_STORAGE_PERMISSION"
            if not verification_payload["new_compliant_date_safe_daily_sources"]
            else None
        ),
    }
    assignments = deterministic_temporal_split(result.features)
    split_counts = Counter(split.value for split in assignments.values())
    split_payload = {
        "schema_version": DATASET_VERSION,
        "generated_at": generated_at,
        "method": "deterministic chronological 70/15/15",
        "uses_random_split": False,
        "uses_future_returns": False,
        "counts": dict(sorted(split_counts.items())),
        "assignments": [
            {
                "news_id": str(row.news_id),
                "publication_date": row.publication_date.isoformat(),
                "split": assignments[row.news_id].value,
            }
            for row in sorted(
                result.features, key=lambda item: (item.publication_date, str(item.news_id))
            )
        ],
    }
    paths = {
        "source_verification": output_dir / "source-verification.json",
        "import_manifest": output_dir / "import-manifest.json",
        "daily_reactions": output_dir / "daily-reactions.jsonl",
        "daily_features": output_dir / "daily-feature-dataset.jsonl",
        "coverage": output_dir / "coverage.json",
        "readiness": output_dir / "readiness.json",
        "split_readiness": output_dir / "split-readiness.json",
    }
    _write_json(paths["source_verification"], verification_payload)
    _write_json(paths["import_manifest"], import_payload)
    _write_jsonl(paths["daily_reactions"], reactions_payload)
    _write_jsonl(paths["daily_features"], features_payload)
    _write_json(paths["coverage"], coverage)
    _write_json(paths["readiness"], readiness)
    _write_json(paths["split_readiness"], split_payload)
    return paths


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
