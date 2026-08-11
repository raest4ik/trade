from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from src.events.domain.entities import NewsEventAnalysis
from src.events.domain.enums import EventAnalysisStatus, EventType
from src.real_gold_benchmark.domain import (
    EXPECTED_EVENT_DISTRIBUTION,
    OLD_BATCH_001_SHA256,
    BenchmarkPrediction,
    BenchmarkValidationError,
    analyzer_input,
    canonicalize_human_review,
    compare_prediction_sets,
    freeze_canonical_dataset,
    taxonomy_summary,
)
from src.real_gold_benchmark.runner import (
    QWEN_CONTEXT_LENGTH,
    QWEN_MODEL,
    QWEN_RANDOM_SEED,
    QWEN_THINK,
    rules_manifest,
    verify_qwen_config,
)


def test_human_review_file_validation_accepts_expected_fixture(tmp_path: Path) -> None:
    canonical = canonicalize_human_review(_human_file(tmp_path))
    assert len(canonical.examples) == 26


def test_human_review_requires_exactly_twenty_six_records(tmp_path: Path) -> None:
    path = _human_file(tmp_path, rows=_human_rows()[:-1])
    with pytest.raises(BenchmarkValidationError, match="expected 26 records"):
        canonicalize_human_review(path)


def test_human_review_requires_real_provenance(tmp_path: Path) -> None:
    rows = _human_rows()
    rows[0]["source_code"] = "SYNTHETIC_TEST"
    with pytest.raises(BenchmarkValidationError, match="REAL provenance"):
        canonicalize_human_review(_human_file(tmp_path, rows=rows))


def test_human_review_requires_exact_timestamps(tmp_path: Path) -> None:
    rows = _human_rows()
    rows[0]["timestamp_quality"] = "DATE_ONLY"
    with pytest.raises(BenchmarkValidationError, match="timestamp must be EXACT"):
        canonicalize_human_review(_human_file(tmp_path, rows=rows))


def test_human_review_requires_excerpt_only_basis(tmp_path: Path) -> None:
    rows = _human_rows()
    rows[0]["human_review_basis"] = "FULL_TEXT"
    with pytest.raises(BenchmarkValidationError, match="excerpt-only"):
        canonicalize_human_review(_human_file(tmp_path, rows=rows))


def test_human_event_distribution_is_frozen(tmp_path: Path) -> None:
    canonical = canonicalize_human_review(_human_file(tmp_path))
    distribution: dict[str, int] = {}
    for item in canonical.examples:
        name = item.gold_primary_event.value
        distribution[name] = distribution.get(name, 0) + 1
    assert distribution == EXPECTED_EVENT_DISTRIBUTION


def test_canonical_serialization_is_stable(tmp_path: Path) -> None:
    path = _human_file(tmp_path)
    first = canonicalize_human_review(path)
    second = canonicalize_human_review(path)
    assert first.canonical_bytes == second.canonical_bytes


def test_dataset_sha_is_reproducible(tmp_path: Path) -> None:
    canonical = canonicalize_human_review(_human_file(tmp_path))
    assert canonical.dataset_sha256 == hashlib.sha256(canonical.canonical_bytes).hexdigest()


def test_batch_001_sha_constant_is_unchanged() -> None:
    assert OLD_BATCH_001_SHA256 == (
        "4934b37b1c036eedb6191dae5ece2fa49e710d00455576cee3de081cc9e7c196"
    )


def test_gold_is_frozen_before_predictions(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path)
    assert frozen.manifest["freeze_state"] == "FROZEN_BEFORE_PREDICTIONS"
    assert frozen.dataset_path.is_file()


def test_rules_evaluator_input_reads_no_gold_fields(tmp_path: Path) -> None:
    payload = analyzer_input(_frozen(tmp_path).examples[0])
    assert set(payload) == {"news_id", "record_id", "raw_content"}


def test_ai_evaluator_input_reads_no_gold_fields(tmp_path: Path) -> None:
    payload = analyzer_input(_frozen(tmp_path).examples[1])
    assert "human_primary_event" not in payload
    assert "rules_primary_event" not in payload


def test_analyzer_input_has_no_future_market_fields(tmp_path: Path) -> None:
    payload = analyzer_input(_frozen(tmp_path).examples[0])
    assert not set(payload).intersection(
        {"abnormal_return", "post_event_price", "future_volume", "reaction_labels"}
    )


def test_four_way_comparison_counts_all_outcomes(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path)
    rules = tuple(
        _prediction(item.annotation.news_id, item.gold_primary_event) for item in frozen.examples
    )
    qwen_values = list(rules)
    qwen_values[0] = _prediction(frozen.examples[0].annotation.news_id, EventType.UNKNOWN)
    _, summary = compare_prediction_sets(frozen, rules, tuple(qwen_values))
    four_way = _dict(summary["four_way"])
    assert _dict(four_way["BOTH_CORRECT"])["count"] == 25
    assert _dict(four_way["RULES_ONLY_CORRECT"])["count"] == 1


def test_existing_metrics_supply_per_class_values(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path)
    rules = tuple(
        _prediction(item.annotation.news_id, item.gold_primary_event) for item in frozen.examples
    )
    from src.real_gold_benchmark.domain import evaluate_prediction_set

    metrics = _dict(evaluate_prediction_set(frozen, rules, system_name="test").metrics["events"])
    assert set(_dict(metrics["per_class"])) == {"FINANCIAL_RESULTS", "GUIDANCE", "OTHER"}


def test_existing_metrics_supply_macro_f1(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path)
    predictions = tuple(
        _prediction(item.annotation.news_id, EventType.UNKNOWN) for item in frozen.examples
    )
    from src.real_gold_benchmark.domain import evaluate_prediction_set

    metrics = _dict(
        evaluate_prediction_set(frozen, predictions, system_name="test").metrics["events"]
    )
    assert "macro_f1" in metrics


def test_existing_metrics_supply_confusion_matrix(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path)
    predictions = tuple(
        _prediction(item.annotation.news_id, EventType.UNKNOWN) for item in frozen.examples
    )
    from src.real_gold_benchmark.domain import evaluate_prediction_set

    metrics = _dict(
        evaluate_prediction_set(frozen, predictions, system_name="test").metrics["events"]
    )
    assert metrics["confusion_matrix"]


def test_error_taxonomy_is_research_only() -> None:
    _, summary = taxonomy_summary(
        ({"system": "rules-v2", "category": "MISSED_EVENT"},),
        ({"system": QWEN_MODEL, "category": "FALSE_SPECIFIC_EVENT"},),
    )
    assert summary["research_only"] is True
    assert summary["models_unchanged"] is True


def test_oracle_upper_bound_is_marked_diagnostic(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path)
    predictions = tuple(
        _prediction(item.annotation.news_id, item.gold_primary_event) for item in frozen.examples
    )
    _, summary = compare_prediction_sets(frozen, predictions, predictions)
    oracle = _dict(summary["ORACLE_UPPER_BOUND"])
    assert oracle == {"diagnostic_only": True, "primary_accuracy": 1.0}


def test_no_hybrid_predictions_are_emitted(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path)
    predictions = tuple(
        _prediction(item.annotation.news_id, item.gold_primary_event) for item in frozen.examples
    )
    _, summary = compare_prediction_sets(frozen, predictions, predictions)
    assert summary["hybrid_predictions_emitted"] is False


def test_generated_artifact_directory_is_gitignored() -> None:
    root = Path(__file__).parents[2]
    assert "artifacts/" in (root / ".gitignore").read_text(encoding="utf-8")


def test_qwen_frozen_config_signature_verification(tmp_path: Path) -> None:
    path = tmp_path / "frozen-config.json"
    expected: dict[str, object] = {
        "model": QWEN_MODEL,
        "think": QWEN_THINK,
        "seed": QWEN_RANDOM_SEED,
        "context": QWEN_CONTEXT_LENGTH,
    }
    path.write_text(json.dumps(expected), encoding="utf-8")
    verify_qwen_config(path, expected)
    with pytest.raises(BenchmarkValidationError, match="fingerprint mismatch"):
        verify_qwen_config(path, expected | {"seed": 99})


def test_deterministic_rules_versions_are_unchanged() -> None:
    assert rules_manifest() == {
        "system": "rules-v2",
        "analysis_version": "event-rules-v2",
        "fact_extractor_version": "financial-facts-v2",
        "rules_changed": False,
    }


def test_guidance_facts_preserve_percent_not_percentage_points(tmp_path: Path) -> None:
    canonical = canonicalize_human_review(_human_file(tmp_path))
    guidance = next(
        item for item in canonical.examples if item.gold_primary_event == EventType.GUIDANCE
    )
    assert len(guidance.annotation.gold_financial_facts) == 3
    assert {item.unit.value for item in guidance.annotation.gold_financial_facts} == {"PERCENT"}


def test_unit_tests_do_not_use_live_http() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "http" + "x." not in source
    assert "localhost" + ":11434" not in source


def _human_file(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]] | None = None,
) -> Path:
    path = tmp_path / "human.jsonl"
    materialized = rows or _human_rows()
    path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in materialized
        ),
        encoding="utf-8",
    )
    return path


def _human_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(26):
        primary = "OTHER"
        text = f"Issuer update {index}."
        facts: list[dict[str, object]] = []
        if index == 24:
            primary = "FINANCIAL_RESULTS"
            text = "The issuer published financial results."
        elif index == 25:
            primary = "GUIDANCE"
            text = (
                "By end of 2026: не&nbsp;менее 75% разработчиков, "
                "не&nbsp;менее 75% изменений и не&nbsp;менее 75% кода."
            )
            facts = [
                _human_fact("developers_regularly_using_ai_share", "developers"),
                _human_fact("company_changes_with_ai_participation_share", "changes"),
                _human_fact("ai_generated_code_share_within_ai_assisted_changes", "code"),
            ]
        source = "ROSNEFT_PRESS_RELEASES_RSS" if index < 10 else "YANDEX_IR_PRESS_RELEASES_RSS"
        ticker = "ROSN" if index < 10 else "YDEX"
        news_id = UUID(int=index + 1)
        rows.append(
            {
                "annotation_text": text,
                "human_events": [primary],
                "human_financial_facts": facts,
                "human_primary_event": primary,
                "human_review_basis": "annotation_text_excerpt_only",
                "human_review_notes": "independent review",
                "human_review_status": "REVIEWED",
                "news_id": str(news_id),
                "published_at": f"2026-07-{index + 1:02d}T08:00:00+00:00",
                "record_id": f"batch-003-{news_id}",
                "source_code": source,
                "source_item_id": f"item-{index}",
                "source_url": f"https://issuer.example/item-{index}",
                "storage_policy": "EXCERPT_ALLOWED",
                "ticker": ticker,
                "timestamp_quality": "EXACT",
            }
        )
    return rows


def _human_fact(metric_name: str, note: str) -> dict[str, object]:
    return {
        "change_direction": "UNKNOWN",
        "metric": "OTHER",
        "metric_name": metric_name,
        "notes": note,
        "period_text": "by end of 2026",
        "role": "FORECAST",
        "unit": "PERCENT",
        "value": "75",
    }


def _frozen(tmp_path: Path):
    canonical = canonicalize_human_review(_human_file(tmp_path))
    old = tmp_path / "old-gold.jsonl"
    old.write_text("frozen\n", encoding="utf-8")
    old_hash = hashlib.sha256(old.read_bytes()).hexdigest()
    return freeze_canonical_dataset(
        canonical,
        output_directory=tmp_path / "gold",
        old_batch_001_path=old,
        git_sha="test-sha",
        expected_old_batch_001_sha256=old_hash,
    )


def _prediction(news_id: UUID, primary: EventType) -> BenchmarkPrediction:
    analysis = NewsEventAnalysis.create(
        news_id=news_id,
        status=EventAnalysisStatus.COMPLETE,
        primary_event_type=primary,
        events=[],
        financial_facts=[],
    )
    return BenchmarkPrediction(
        record_id=str(news_id),
        news_id=news_id,
        analysis=analysis,
        runtime={"latency_ms": 1},
    )


def _dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    items = cast("dict[object, object]", value)
    return {str(key): item for key, item in items.items()}
