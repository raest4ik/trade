from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from src.fresh_real_corpus.domain import (
    ANNOTATION_BATCH_VERSION,
    BATCH_003_STATUS,
    FUTURE_MARKET_FIELDS,
    PREDICTION_FIELDS,
    CorpusSplit,
    ExclusionIndex,
    FreshCorpusRecord,
    MatchStatus,
    SelectionPolicy,
    assert_safe_annotation_payload,
    freeze_temporal_split,
    load_exclusion_index,
    select_fresh_records,
    selection_field_names,
)
from src.fresh_real_corpus.reporting import write_fresh_corpus_artifacts
from src.news.domain.enums import PublicationTimestampQuality
from src.official_sources.registry import official_source_configs

PUBLISHED = datetime(2025, 1, 1, 8, 0, tzinfo=UTC)


def test_batch_003_is_marked_observed() -> None:
    assert BATCH_003_STATUS == "OBSERVED_EVALUATION_SET"


def test_batch_003_news_id_is_excluded_from_batch_004() -> None:
    record = _record(1)
    result = select_fresh_records(
        [record], policy=_policy(), exclusions=ExclusionIndex(news_ids=frozenset({record.news_id}))
    )
    assert result.records == ()
    assert result.excluded_overlap_count == 1


def test_batch_003_logical_record_is_excluded_even_with_new_uuid() -> None:
    record = _record(1)
    exclusions = ExclusionIndex(logical_keys=frozenset({record.logical_key}))
    assert select_fresh_records([record], policy=_policy(), exclusions=exclusions).records == ()


def test_batch_004_requires_real_approved_provenance() -> None:
    with pytest.raises(ValueError, match="approved REAL"):
        replace(_record(1), source_code="SYNTHETIC_TEST").validate()


def test_batch_004_accepts_exact_timestamps_only() -> None:
    with pytest.raises(ValueError, match="EXACT"):
        replace(_record(1), timestamp_quality=PublicationTimestampQuality.DATE_ONLY).validate()


def test_selection_is_deterministic_by_source_date_identity() -> None:
    records = [_record(index) for index in range(8)]
    first = select_fresh_records(records, policy=_policy(), exclusions=ExclusionIndex())
    second = select_fresh_records(
        list(reversed(records)), policy=_policy(), exclusions=ExclusionIndex()
    )
    assert first.records == second.records


def test_selection_policy_has_no_prediction_fields() -> None:
    assert selection_field_names() == {
        "source_codes",
        "date_from",
        "date_to",
        "limit",
        "source_order",
    }


def test_selection_policy_has_no_future_return_fields() -> None:
    assert not selection_field_names().intersection(FUTURE_MARKET_FIELDS)


def test_annotation_has_no_qwen_or_rules_labels() -> None:
    payload = _record(1).annotation_payload(CorpusSplit.DEVELOPMENT)
    assert not set(payload).intersection(PREDICTION_FIELDS)


def test_annotation_has_no_future_market_data() -> None:
    payload = _record(1).annotation_payload(CorpusSplit.DEVELOPMENT)
    assert not set(payload).intersection(FUTURE_MARKET_FIELDS)


def test_development_and_holdout_are_disjoint() -> None:
    split = freeze_temporal_split(tuple(_record(index) for index in range(10)))
    development = {
        news_id for news_id, name in split.assignments if name == CorpusSplit.DEVELOPMENT
    }
    holdout = {news_id for news_id, name in split.assignments if name == CorpusSplit.FRESH_HOLDOUT}
    assert development.isdisjoint(holdout)
    assert len(development | holdout) == 10


def test_temporal_split_places_newest_records_in_holdout() -> None:
    records = tuple(_record(index) for index in range(10))
    split = freeze_temporal_split(tuple(reversed(records)))
    development_dates = [
        item.published_at
        for item in records
        if split.split_for(item.news_id) == CorpusSplit.DEVELOPMENT
    ]
    holdout_dates = [
        item.published_at
        for item in records
        if split.split_for(item.news_id) == CorpusSplit.FRESH_HOLDOUT
    ]
    assert max(development_dates) < min(holdout_dates)


def test_frozen_split_sha_is_reproducible() -> None:
    records = tuple(_record(index) for index in range(10))
    assert (
        freeze_temporal_split(records).split_sha256
        == freeze_temporal_split(tuple(reversed(records))).split_sha256
    )


def test_holdout_payload_contains_no_predictions() -> None:
    payload = _record(1).annotation_payload(CorpusSplit.FRESH_HOLDOUT)
    assert payload["split"] == "FRESH_HOLDOUT"
    assert not set(payload).intersection(PREDICTION_FIELDS)


def test_original_timezone_provenance_is_preserved() -> None:
    payload = _record(1).annotation_payload(CorpusSplit.DEVELOPMENT)
    assert payload["original_timestamp_text"] == "Wed, 01 Jan 2025 11:00:00 +0300"
    assert payload["source_timezone"] == "UTC+03:00"
    assert str(payload["published_at"]).endswith("Z")


def test_storage_policy_must_allow_annotation_text() -> None:
    with pytest.raises(ValueError, match="storage policy"):
        replace(_record(1), storage_policy="METADATA_ONLY").validate()


def test_excerpt_policy_requires_excerpt_flag() -> None:
    with pytest.raises(ValueError, match="explicitly marked"):
        replace(_record(1), content_is_excerpt=False).validate()


def test_annotation_text_hash_must_match() -> None:
    with pytest.raises(ValueError, match="content hash"):
        replace(_record(1), content_hash="0" * 64).validate()


def test_seed_source_is_excluded() -> None:
    with pytest.raises(ValueError, match="approved REAL"):
        replace(_record(1), source_code="BATCH_001").validate()


def test_bounded_import_limit_rejects_more_than_100() -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        replace(_policy(), limit=101).normalized()


def test_selection_is_idempotent() -> None:
    records = [_record(index) for index in range(5)]
    first = select_fresh_records(records, policy=_policy(), exclusions=ExclusionIndex())
    second = select_fresh_records(records, policy=_policy(), exclusions=ExclusionIndex())
    assert first == second


def test_exclusion_index_loads_news_logical_and_content_ids(tmp_path: Path) -> None:
    path = tmp_path / "prior.jsonl"
    path.write_text(
        json.dumps(
            {
                "news_id": str(UUID(int=1)),
                "source": "ROSNEFT_PRESS_RELEASES_RSS",
                "source_item_id": "item-1",
                "raw_content_hash": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = load_exclusion_index((path,))
    assert UUID(int=1) in index.news_ids
    assert ("ROSNEFT_PRESS_RELEASES_RSS", "item-1") in index.logical_keys
    assert "a" * 64 in index.content_hashes


def test_annotation_state_is_draft_unassigned_non_gold() -> None:
    payload = _record(1).annotation_payload(CorpusSplit.DEVELOPMENT)
    assert payload["schema_version"] == ANNOTATION_BATCH_VERSION
    assert payload["annotation_status"] == "DRAFT"
    assert payload["assignment_status"] == "UNASSIGNED"
    assert payload["is_gold"] is False


def test_safe_payload_rejects_model_output() -> None:
    payload = _record(1).annotation_payload(CorpusSplit.DEVELOPMENT)
    payload["qwen_primary_event"] = "OTHER"
    with pytest.raises(ValueError, match="prohibited"):
        assert_safe_annotation_payload(payload)


def test_generated_artifacts_are_gitignored() -> None:
    ignore = (Path(__file__).parents[2] / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/" in ignore


def test_artifact_writer_emits_identical_human_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.fresh_real_corpus.reporting as reporting

    records = tuple(_record(index) for index in range(4))
    result = select_fresh_records(list(records), policy=_policy(), exclusions=ExclusionIndex())
    split = freeze_temporal_split(result.records)
    old_gold = tmp_path / "old.jsonl"
    old_gold.write_text("frozen\n", encoding="utf-8")
    monkeypatch.setattr(
        reporting, "EXPECTED_BATCH_001_SHA256", hashlib.sha256(old_gold.read_bytes()).hexdigest()
    )
    paths = write_fresh_corpus_artifacts(
        tmp_path / "corpus",
        annotation_copy=tmp_path / "copy.jsonl",
        result=result,
        split=split,
        policy=_policy(),
        source_configs=official_source_configs(),
        batch_001_path=old_gold,
        git_sha="test-sha",
    )
    assert paths["annotation"].read_bytes() == paths["annotation_copy"].read_bytes()
    rows = _jsonl(paths["annotation"])
    assert all("qwen_primary_event" not in row for row in rows)


def test_unit_tests_do_not_perform_live_http() -> None:
    root = Path(__file__).parents[2]
    application = (root / "src" / "fresh_real_corpus" / "application.py").read_text(
        encoding="utf-8"
    )
    cli = (root / "apps" / "cli" / "build_fresh_real_annotation_corpus.py").read_text(
        encoding="utf-8"
    )
    assert "import httpx" not in application + cli
    assert "requests.get(" not in application + cli


def _policy() -> SelectionPolicy:
    return SelectionPolicy(
        source_codes=("ROSNEFT_PRESS_RELEASES_RSS", "YANDEX_IR_PRESS_RELEASES_RSS"),
        date_from=PUBLISHED - timedelta(days=1),
        date_to=PUBLISHED + timedelta(days=30),
        limit=100,
    )


def _record(index: int) -> FreshCorpusRecord:
    source = "ROSNEFT_PRESS_RELEASES_RSS" if index % 2 else "YANDEX_IR_PRESS_RELEASES_RSS"
    ticker = "ROSN" if source.startswith("ROSNEFT") else "YDEX"
    text = f"Issuer-owned permitted excerpt {index}."
    return FreshCorpusRecord(
        news_id=UUID(int=index + 1),
        source_code=source,
        source_item_id=f"item-{index}",
        source_url=f"https://issuer.example/releases/{index}",
        ticker=ticker,
        published_at=PUBLISHED + timedelta(days=index),
        original_timestamp_text="Wed, 01 Jan 2025 11:00:00 +0300",
        source_timezone="UTC+03:00",
        timestamp_quality=PublicationTimestampQuality.EXACT,
        title=f"Issuer release {index}",
        annotation_text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        storage_policy="EXCERPT_ALLOWED",
        content_is_excerpt=True,
        match_status=MatchStatus.MATCHED,
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
