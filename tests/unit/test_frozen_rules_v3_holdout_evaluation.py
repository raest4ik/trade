from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.events.domain.entities import DetectedEvent, NewsEventAnalysis
from src.events.domain.enums import EventAnalysisStatus, EventType
from src.events.domain.v3 import EVENT_ANALYSIS_V3_VERSION, rules_v3_fingerprint
from src.holdout_evaluation.application import (
    claim_single_run,
    run_single_holdout_evaluation,
    write_holdout_artifacts,
)
from src.holdout_evaluation.domain import (
    EXPECTED_CANDIDATE_NAME,
    EXPECTED_RULES_FINGERPRINT,
    EXPECTED_SPLIT_SHA256,
    HOLDOUT_DATASET_NAME,
    freeze_holdout_gold,
    verify_frozen_candidate,
)


def test_holdout_gold_contains_exactly_four_records(tmp_path: Path) -> None:
    assert len(_freeze(tmp_path).records) == 4


def test_holdout_gold_distribution_is_other_only(tmp_path: Path) -> None:
    assert _freeze(tmp_path).event_distribution == {"OTHER": 4}


def test_holdout_gold_contains_zero_financial_facts(tmp_path: Path) -> None:
    dataset = _freeze(tmp_path)
    assert all(not item.source_payload["human_financial_facts"] for item in dataset.records)


def test_holdout_gold_sha_is_reproducible(tmp_path: Path) -> None:
    first = _freeze(tmp_path / "first")
    second = _freeze(tmp_path / "second")
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.source_review_sha256 == second.source_review_sha256


def test_holdout_gold_preserves_frozen_split_sha(tmp_path: Path) -> None:
    assert _freeze(tmp_path).split_sha256 == EXPECTED_SPLIT_SHA256


def test_expected_candidate_fingerprint_is_current() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT


def test_frozen_candidate_is_verified_before_evaluation(tmp_path: Path) -> None:
    candidate = verify_frozen_candidate(_write_candidate(tmp_path))
    assert candidate.name == EXPECTED_CANDIDATE_NAME
    assert candidate.rules_fingerprint == EXPECTED_RULES_FINGERPRINT


def test_candidate_fingerprint_mismatch_stops(tmp_path: Path) -> None:
    path = _write_candidate(tmp_path, fingerprint="0" * 64)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_frozen_candidate(path)


def test_single_holdout_run_analyzes_each_record_once(tmp_path: Path) -> None:
    analyzer = CountingOtherAnalyzer()
    result = run_single_holdout_evaluation(_freeze(tmp_path), analyzer)
    assert analyzer.calls == 4
    assert result.metrics["primary_accuracy"] == 1.0


def test_single_run_marker_blocks_a_second_run(tmp_path: Path) -> None:
    claim_single_run(tmp_path)
    with pytest.raises(ValueError, match="already been claimed"):
        claim_single_run(tmp_path)


def test_holdout_metrics_include_other_and_confusion(tmp_path: Path) -> None:
    result = run_single_holdout_evaluation(_freeze(tmp_path), CountingOtherAnalyzer())
    assert result.metrics["micro"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result.metrics["macro_f1"] == 1.0
    assert result.metrics["per_class"]["OTHER"]["f1"] == 1.0
    assert result.metrics["confusion_matrix"] == []


def test_record_results_are_complete_and_correct(tmp_path: Path) -> None:
    result = run_single_holdout_evaluation(_freeze(tmp_path), CountingOtherAnalyzer())
    assert len(result.record_results) == 4
    assert all(item["gold"] == "OTHER" for item in result.record_results)
    assert all(item["prediction"] == "OTHER" for item in result.record_results)
    assert all(item["correct"] is True for item in result.record_results)


def test_artifacts_mark_holdout_observed_and_nlp_cycle_closed(tmp_path: Path) -> None:
    output = tmp_path / "output"
    dataset = freeze_holdout_gold(
        source_review_path=_write_review(tmp_path),
        split_manifest_path=_write_split(tmp_path),
        output_directory=output / "holdout-gold",
    )
    candidate = verify_frozen_candidate(_write_candidate(tmp_path))
    result = run_single_holdout_evaluation(dataset, CountingOtherAnalyzer())
    write_holdout_artifacts(
        output_directory=output,
        dataset=dataset,
        candidate=candidate,
        result=result,
    )
    manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["holdout_status"] == "OBSERVED_HOLDOUT"
    assert manifest["nlp_development_cycle"] == "CLOSED"
    assert manifest["post_holdout_tuning_allowed"] is False
    assert manifest["rules_changed_after_freeze"] is False


def test_holdout_workflow_has_no_qwen_hybrid_or_predictive_ml() -> None:
    source = "".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "src/holdout_evaluation/domain.py",
            "src/holdout_evaluation/application.py",
            "apps/cli/evaluate_frozen_rules_v3_holdout.py",
        )
    ).lower()
    assert "qwen" not in source
    assert '"hybrid": false' in source
    for model in ("xgboost", "catboost", "lightgbm", "logisticregression"):
        assert model not in source


def test_holdout_unit_tests_have_no_live_http() -> None:
    source = Path("src/holdout_evaluation/application.py").read_text(encoding="utf-8")
    assert "httpx" not in source
    assert "requests.get(" not in source


def test_holdout_dataset_name_is_versioned() -> None:
    assert HOLDOUT_DATASET_NAME == "ru-corporate-events-real-batch-004-holdout-gold-v1"


class CountingOtherAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, *, news_id: UUID, raw_content: str) -> NewsEventAnalysis:
        self.calls += 1
        event = DetectedEvent(
            id=uuid4(),
            analysis_id=UUID(int=0),
            event_type=EventType.OTHER,
            confidence=Decimal("0.9"),
            rule_id="synthetic.other",
            matched_rule="synthetic.other",
            evidence_text=raw_content,
            start_position=0,
            end_position=len(raw_content),
        )
        return NewsEventAnalysis.create(
            news_id=news_id,
            status=EventAnalysisStatus.COMPLETE,
            primary_event_type=EventType.OTHER,
            events=[event],
            financial_facts=[],
            analysis_version=EVENT_ANALYSIS_V3_VERSION,
        )


def _freeze(tmp_path: Path):
    return freeze_holdout_gold(
        source_review_path=_write_review(tmp_path),
        split_manifest_path=_write_split(tmp_path),
        output_directory=tmp_path / "gold",
    )


def _write_review(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "batch-004-holdout-human-review-v1.jsonl"
    payloads: list[dict[str, object]] = []
    for index in range(1, 5):
        text = f"Issuer-owned HOLDOUT excerpt {index}."
        payloads.append(
            {
                "annotation_text": text,
                "human_events": ["OTHER"],
                "human_financial_facts": [],
                "human_primary_event": "OTHER",
                "human_review_basis": "annotation_text_excerpt_only",
                "human_review_notes": "No supported specific event in the excerpt.",
                "human_review_status": "REVIEWED",
                "is_gold": False,
                "news_id": str(UUID(int=index)),
                "provenance": "REAL",
                "published_at": f"2026-01-{index:02d}T00:00:00Z",
                "raw_content_hash": hashlib.sha256(text.encode()).hexdigest(),
                "source": "ISSUER_OWNED_RSS",
                "source_item_id": f"holdout-{index}",
                "ticker": "TEST",
            }
        )
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in payloads),
        encoding="utf-8",
    )
    return path


def _write_split(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "split-manifest.json"
    path.write_text(
        json.dumps(
            {
                "assignments": [
                    {"news_id": str(UUID(int=index)), "split": "FRESH_HOLDOUT"}
                    for index in range(1, 5)
                ],
                "split_sha256": EXPECTED_SPLIT_SHA256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _write_candidate(tmp_path: Path, *, fingerprint: str = EXPECTED_RULES_FINGERPRINT) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "candidate_name": EXPECTED_CANDIDATE_NAME,
                "development_gold_sha256": "d" * 64,
                "frozen": True,
                "git_sha": "c" * 40,
                "rules_fingerprint_sha256": fingerprint,
                "split_sha256": EXPECTED_SPLIT_SHA256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path
