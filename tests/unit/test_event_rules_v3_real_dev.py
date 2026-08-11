from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import UUID

from src.events.domain.analyzer import EventAnalyzer
from src.events.domain.enums import EventType, FinancialMetric
from src.events.domain.rules import EVENT_RULES
from src.events.domain.v3 import (
    EVENT_ANALYSIS_V3_VERSION,
    V3_EVENT_RULES,
    EventAnalyzerV3,
    rules_v3_fingerprint,
)
from src.real_dev_rules.application import FROZEN_QWEN_CONFIG, frozen_qwen_manifest
from src.real_dev_rules.domain import (
    DEVELOPMENT_DATASET_NAME,
    EXPECTED_EVENT_DISTRIBUTION,
    EXPECTED_SPLIT_SHA256,
    freeze_development_gold,
    load_split_metadata,
)


def test_development_gold_contains_exactly_ten_records(tmp_path: Path) -> None:
    assert len(_freeze(tmp_path).records) == 10


def test_development_gold_has_expected_event_distribution(tmp_path: Path) -> None:
    assert _freeze(tmp_path).event_distribution == EXPECTED_EVENT_DISTRIBUTION


def test_development_gold_preserves_frozen_split_sha(tmp_path: Path) -> None:
    assert _freeze(tmp_path).split_sha256 == EXPECTED_SPLIT_SHA256


def test_holdout_is_exposed_as_count_metadata_only(tmp_path: Path) -> None:
    split_path = _write_split(tmp_path)
    split_sha, development_ids, holdout_count = load_split_metadata(split_path)
    assert split_sha == EXPECTED_SPLIT_SHA256
    assert len(development_ids) == 10
    assert holdout_count == 4
    assert "annotation_text" not in split_path.read_text(encoding="utf-8")


def test_holdout_text_is_not_an_evaluator_input() -> None:
    import inspect

    from src.real_dev_rules.application import evaluate_deterministic

    assert "holdout" not in inspect.signature(evaluate_deterministic).parameters


def test_candidate_policy_creates_no_holdout_predictions() -> None:
    source = Path("src/real_dev_rules/application.py").read_text(encoding="utf-8")
    assert '"holdout_predictions_created": False' in source


def test_event_rules_v2_remains_unchanged() -> None:
    assert all(not rule.rule_id.startswith("event.v3.") for rule in EVENT_RULES)
    result = EventAnalyzer().analyze(news_id=UUID(int=1), raw_content=_texts()[6])
    assert result.analysis_version == "event-rules-v2"
    assert result.primary_event_type == EventType.UNKNOWN


def test_event_rules_v3_is_deterministic() -> None:
    analyzer = EventAnalyzerV3()
    first = analyzer.analyze(news_id=UUID(int=1), raw_content=_texts()[6])
    second = analyzer.analyze(news_id=UUID(int=1), raw_content=_texts()[6])
    assert first.primary_event_type == second.primary_event_type
    assert [item.rule_id for item in first.events] == [item.rule_id for item in second.events]
    assert first.analysis_version == EVENT_ANALYSIS_V3_VERSION


def test_v3_rules_have_no_ticker_or_issuer_specific_patterns() -> None:
    material = " ".join(rule.pattern.pattern.lower() for rule in V3_EVENT_RULES)
    assert "rosn" not in material
    assert "rosneft" not in material
    assert "ticker" not in material


def test_v3_rules_have_no_record_identity_hardcoding() -> None:
    material = " ".join(rule.pattern.pattern.lower() for rule in V3_EVENT_RULES)
    assert "news_id" not in material
    assert "source_item" not in material
    assert "6e7ebcd5" not in material


def test_v3_maps_cooperation_agreements_to_other() -> None:
    analyzer = EventAnalyzerV3()
    for text in _texts()[:3]:
        assert analyzer.analyze(news_id=UUID(int=1), raw_content=text).primary_event_type == (
            EventType.OTHER
        )


def test_v3_does_not_map_arbitrary_unknown_text_to_other() -> None:
    result = EventAnalyzerV3().analyze(
        news_id=UUID(int=1), raw_content="A short issuer update without an identifiable event."
    )
    assert result.primary_event_type == EventType.UNKNOWN


def test_v3_preserves_dividend_event_rule() -> None:
    result = EventAnalyzerV3().analyze(news_id=UUID(int=1), raw_content=_texts()[3])
    assert result.primary_event_type == EventType.DIVIDEND
    assert result.events[0].rule_id == "event.dividend"


def test_v3_detects_restrictive_measures_as_sanctions() -> None:
    result = EventAnalyzerV3().analyze(news_id=UUID(int=1), raw_content=_texts()[4])
    assert result.primary_event_type == EventType.SANCTIONS


def test_v3_detects_elected_board_chair_as_management_change() -> None:
    result = EventAnalyzerV3().analyze(news_id=UUID(int=1), raw_content=_texts()[5])
    assert result.primary_event_type == EventType.MANAGEMENT_CHANGE


def test_v3_detects_ifrs_results_announcements() -> None:
    analyzer = EventAnalyzerV3()
    for text in _texts()[6:]:
        assert analyzer.analyze(news_id=UUID(int=1), raw_content=text).primary_event_type == (
            EventType.FINANCIAL_RESULTS
        )


def test_v3_extracts_reviewed_dividend_fact() -> None:
    result = EventAnalyzerV3().analyze(news_id=UUID(int=1), raw_content=_texts()[3])
    assert len(result.financial_facts) == 1
    fact = result.financial_facts[0]
    assert fact.metric == FinancialMetric.DIVIDEND_PER_SHARE
    assert str(fact.normalized_value) == "14.68"
    assert fact.currency.value == "RUB"
    assert fact.unit.value == "MONEY"
    assert fact.fact_role.value == "ACTUAL"
    assert fact.year == 2024


def test_canonical_development_sha_is_reproducible(tmp_path: Path) -> None:
    first = _freeze(tmp_path / "first")
    second = _freeze(tmp_path / "second")
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.source_review_sha256 == second.source_review_sha256


def test_rules_candidate_fingerprint_is_reproducible() -> None:
    first = rules_v3_fingerprint()
    assert first == rules_v3_fingerprint()
    assert len(first) == 64


def test_qwen_config_is_frozen() -> None:
    assert FROZEN_QWEN_CONFIG == {
        "context_length": 4096,
        "model": "qwen3.5:9b",
        "provider": "ollama",
        "random_seed": 0,
        "think": False,
    }
    manifest = frozen_qwen_manifest()
    assert manifest["prompt_sha256"]
    assert manifest["schema_sha256"]


def test_candidate_has_no_hybrid() -> None:
    source = Path("src/real_dev_rules/application.py").read_text(encoding="utf-8")
    assert '"hybrid": False' in source
    assert '"qwen_used_for_rule_design": False' in source


def test_development_gold_has_no_market_future_fields(tmp_path: Path) -> None:
    _freeze(tmp_path)
    text = (tmp_path / "gold" / "dataset.jsonl").read_text(encoding="utf-8")
    for field in ("abnormal_return", "future_price", "future_volume", "market_reaction"):
        assert field not in text


def test_development_dataset_name_is_versioned() -> None:
    assert DEVELOPMENT_DATASET_NAME == "ru-corporate-events-real-batch-004-development-gold-v1"


def test_unit_workflow_has_no_live_http_calls() -> None:
    source = "".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "src/events/domain/v3.py",
            "src/real_dev_rules/domain.py",
        )
    )
    assert "import httpx" not in source
    assert "requests.get(" not in source


def _freeze(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    review_path = tmp_path / "batch-004-development-human-review-v1.jsonl"
    review_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in _review_payloads()),
        encoding="utf-8",
    )
    return freeze_development_gold(
        source_review_path=review_path,
        split_manifest_path=_write_split(tmp_path),
        output_directory=tmp_path / "gold",
    )


def _write_split(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "split-manifest.json"
    assignments = [
        {"news_id": str(UUID(int=index)), "split": "DEVELOPMENT"} for index in range(1, 11)
    ] + [{"news_id": str(UUID(int=index)), "split": "FRESH_HOLDOUT"} for index in range(11, 15)]
    path.write_text(
        json.dumps(
            {"assignments": assignments, "split_sha256": EXPECTED_SPLIT_SHA256},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _review_payloads() -> list[dict[str, object]]:
    labels = [
        "OTHER",
        "OTHER",
        "OTHER",
        "DIVIDEND",
        "SANCTIONS",
        "MANAGEMENT_CHANGE",
        "FINANCIAL_RESULTS",
        "FINANCIAL_RESULTS",
        "FINANCIAL_RESULTS",
        "FINANCIAL_RESULTS",
    ]
    payloads: list[dict[str, object]] = []
    for index, (text, label) in enumerate(zip(_texts(), labels, strict=True), start=1):
        facts: list[dict[str, object]] = []
        if label == "DIVIDEND":
            facts.append(
                {
                    "change_direction": "UNKNOWN",
                    "comparison_type": "NONE",
                    "currency": "RUB",
                    "metric": "DIVIDEND_PER_SHARE",
                    "period_text": "2024",
                    "role": "ACTUAL",
                    "scale": "ONE",
                    "unit": "MONEY",
                    "value": "14.68",
                }
            )
        payloads.append(
            {
                "annotation_text": text,
                "human_events": [label],
                "human_financial_facts": facts,
                "human_primary_event": label,
                "human_review_basis": "annotation_text_excerpt_only",
                "human_review_notes": "Reviewed from permitted excerpt.",
                "human_review_status": "REVIEWED",
                "is_gold": False,
                "news_id": str(UUID(int=index)),
                "published_at": f"2025-01-{index:02d}T00:00:00Z",
                "raw_content_hash": hashlib.sha256(text.encode()).hexdigest(),
                "source": "ISSUER_OWNED_RSS",
                "source_item_id": f"item-{index}",
                "ticker": "TEST",
            }
        )
    return payloads


def _texts() -> list[str]:
    return [
        "The company signed a cooperation agreement with an environmental agency.",
        "The company concluded a trilateral agreement of cooperation in HR training.",
        "The company signed a Cooperation Agreement with the ministry.",
        "The shareholders approved dividends for 2024 in the amount of 14.68 roubles per share.",
        "The authority decided to impose restrictive measures on the refinery.",
        "A director has been elected Chairman of the Board of Directors.",
        "The company publishes its results for first half 2025 under IFRS.",
        "The company announces its results for 9M 2025 under IFRS.",
        "The company announces its results for 12M 2025 under IFRS.",
        "The company announces its results for Q1 2026 under IFRS.",
    ]
