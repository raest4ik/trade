from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from src.evaluation.application.seed_curation import (
    EXPECTED_SEED_QUOTAS,
    SeedEventRecord,
    dry_run_counts,
    validate_seed_file,
)


def test_valid_50_record_seed_structure(tmp_path: Path) -> None:
    path = tmp_path / "seed.jsonl"
    records = [
        _seed_payload(category, index)
        for category, count in EXPECTED_SEED_QUOTAS.items()
        for index in range(count)
    ]
    _write_jsonl(path, records)

    result = validate_seed_file(path)

    assert result.ok
    assert len(result.records) == 50
    assert result.quotas == EXPECTED_SEED_QUOTAS


def test_seed_validation_reports_invalid_json_duplicate_hash_evidence_and_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "seed-invalid.jsonl"
    first = _seed_payload("FINANCIAL_RESULTS", 1)
    duplicate = _seed_payload("FINANCIAL_RESULTS", 2) | {"record_id": first["record_id"]}
    bad_hash = _seed_payload("DIVIDEND", 1) | {"raw_content_hash": "bad"}
    bad_evidence = _seed_payload("GUIDANCE", 1)
    gold_events = cast("list[dict[str, object]]", bad_evidence["gold_events"])
    gold_events[0]["evidence_text"] = "not in text"
    missing_source = _seed_payload("PRODUCTION_UPDATE", 1)
    source = cast("dict[str, object]", missing_source["source"])
    source["url"] = ""
    path.write_text(
        "{bad json\n"
        + "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in [first, duplicate, bad_hash, bad_evidence, missing_source]
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_seed_file(path)

    errors = "\n".join(result.errors)
    assert not result.ok
    assert "malformed JSON" in errors
    assert "duplicate record_id" in errors
    assert "raw_content_hash mismatch" in errors
    assert "evidence span text mismatch" in errors
    assert "source.url is empty" in errors


def test_dry_run_counts_are_idempotent(tmp_path: Path) -> None:
    records = [
        _seed_payload("FINANCIAL_RESULTS", 1),
        _seed_payload("DIVIDEND", 1),
    ]
    path_records = _records_from_payloads(tmp_path, records)

    counts = dry_run_counts(path_records, {str(records[0]["record_id"])})

    assert counts == {
        "records_total": 2,
        "would_create": 1,
        "already_exists": 1,
        "invalid": 0,
    }


def _records_from_payloads(
    tmp_path: Path,
    payloads: list[dict[str, object]],
) -> list[SeedEventRecord]:
    tmp = tmp_path / "unused-seed.jsonl"
    _write_jsonl(tmp, payloads)
    try:
        return list(validate_seed_file(tmp).records)
    finally:
        tmp.unlink(missing_ok=True)


def _seed_payload(category: str, index: int) -> dict[str, object]:
    text = (
        f"SBER published financial results {category} {index}. "
        "Revenue for FY2025 reached 100 million rub."
    )
    return {
        "schema_version": "event-seed-v1",
        "target_schema": "event-gold-v1",
        "batch_id": "unit-seed",
        "record_id": f"{category.lower().replace('&', 'and')}-{index}",
        "news_id": None,
        "source_published_date": "2026-08-06",
        "published_at": None,
        "tickers": ["SBER"],
        "company": "SBER",
        "quota_category": category,
        "annotation_text": text,
        "text_origin": "SOURCE_BACKED_PARAPHRASE",
        "raw_content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "review_status": "SEED_REVIEW_REQUIRED",
        "annotator": None,
        "notes": None,
        "predicted_events": [],
        "predicted_financial_facts": [],
        "gold_events": [
            {
                "event_type": "FINANCIAL_RESULTS",
                "evidence_text": text,
                "start_position": 0,
                "end_position": len(text),
                "is_primary": True,
                "notes": None,
            }
        ],
        "gold_financial_facts": [
            {
                "metric": "REVENUE",
                "raw_value": "100",
                "normalized_value": "100000000",
                "unit": "MONEY",
                "currency": "RUB",
                "scale": "MILLION",
                "period_type": "YEAR",
                "period_year": 2025,
                "period_quarter": None,
                "period_month": None,
                "raw_period": "FY2025",
                "fact_role": "ACTUAL",
                "comparison_type": "NONE",
                "change_direction": "UNCHANGED",
                "change_value": None,
                "change_unit": None,
                "notes": None,
                "evidence_text": "100 million rub",
                "start_position": text.index("100"),
                "end_position": len(text) - 1,
            }
        ],
        "source": {
            "title": "Unit source",
            "url": f"https://example.com/source/{category}/{index}",
            "tier": "PRIMARY",
            "support_url": None,
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
