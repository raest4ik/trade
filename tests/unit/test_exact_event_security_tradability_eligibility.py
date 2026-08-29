from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from apps.cli.gate_exact_event_security_tradability import build_parser
from src.exact_event_security_tradability_eligibility.application import (
    run_security_tradability_eligibility,
)
from src.exact_event_security_tradability_eligibility.domain import (
    ARTIFACT_VERSION,
    EventValidity,
    InstrumentIdentityStatus,
    MarketReactionEligibility,
    TradingEvidence,
    evaluate_event_eligibility,
    sha256_payload,
    should_attempt_market_maturation,
)


def test_cli_defaults_to_tradability_eligibility_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "a" * 40])
    assert args.diagnostic_dir == "artifacts/chep-security-history-diagnostics-v1"
    assert args.maturation_dir == "artifacts/chep-historical-exact-maturation-v1"
    assert args.output_dir == "artifacts/exact-event-security-tradability-eligibility-v1"


def test_valid_exact_event_remains_valid_when_market_ineligible() -> None:
    result = evaluate_event_eligibility(
        event_id="chep-one",
        ticker="CHEP",
        published_at_utc=datetime(2026, 7, 6, 9, 4, tzinfo=UTC),
        identity_status=InstrumentIdentityStatus.RESOLVED,
        evidence=_positive_non_trading_evidence(),
    )
    assert result.event_validity == EventValidity.VALID_EXACT_EVENT
    assert (
        result.market_reaction_eligibility
        == MarketReactionEligibility.SECURITY_NOT_TRADING_AT_EVENT_TIME
    )
    assert should_attempt_market_maturation(result) is False


def test_security_not_trading_requires_positive_evidence() -> None:
    empty_provider_only = TradingEvidence(
        ticker="CHEP",
        instrument_uid="uid",
        figi="figi",
        class_code="TQBR",
        source="TINVEST_EMPTY_CANDLES_ONLY",
        security_history_confirmed=False,
        event_date_trading_confirmed=None,
        last_confirmed_trading_date=None,
        current_trading_status=None,
        api_trade_available=None,
        buy_available=None,
        sell_available=None,
        evidence_detail="empty minute candles only",
    )
    result = evaluate_event_eligibility(
        event_id="empty-only",
        ticker="CHEP",
        published_at_utc=datetime(2026, 7, 6, 9, 4, tzinfo=UTC),
        identity_status=InstrumentIdentityStatus.RESOLVED,
        evidence=empty_provider_only,
    )
    assert (
        result.market_reaction_eligibility == MarketReactionEligibility.SECURITY_HISTORY_UNAVAILABLE
    )


def test_unresolved_and_ambiguous_identities_fail_closed() -> None:
    unresolved = evaluate_event_eligibility(
        event_id="unresolved",
        ticker="MISS",
        published_at_utc=datetime(2026, 7, 6, 9, 4, tzinfo=UTC),
        identity_status=InstrumentIdentityStatus.UNRESOLVED,
        evidence=None,
    )
    ambiguous = evaluate_event_eligibility(
        event_id="ambiguous",
        ticker="MISS",
        published_at_utc=datetime(2026, 7, 6, 9, 4, tzinfo=UTC),
        identity_status=InstrumentIdentityStatus.AMBIGUOUS,
        evidence=None,
    )
    assert (
        unresolved.market_reaction_eligibility
        == MarketReactionEligibility.INSTRUMENT_IDENTITY_UNRESOLVED
    )
    assert (
        ambiguous.market_reaction_eligibility
        == MarketReactionEligibility.INSTRUMENT_IDENTITY_AMBIGUOUS
    )
    assert should_attempt_market_maturation(unresolved) is False
    assert should_attempt_market_maturation(ambiguous) is False


def test_eligible_security_proceeds_to_maturation() -> None:
    result = evaluate_event_eligibility(
        event_id="eligible",
        ticker="SBER",
        published_at_utc=datetime(2026, 7, 6, 9, 4, tzinfo=UTC),
        identity_status=InstrumentIdentityStatus.RESOLVED,
        evidence=TradingEvidence(
            ticker="SBER",
            instrument_uid="uid",
            figi="figi",
            class_code="TQBR",
            source="OFFICIAL_EXCHANGE_HISTORY",
            security_history_confirmed=True,
            event_date_trading_confirmed=True,
            last_confirmed_trading_date=date(2026, 7, 6),
            current_trading_status="SECURITY_TRADING_STATUS_NORMAL_TRADING",
            api_trade_available=True,
            buy_available=True,
            sell_available=True,
            evidence_detail="official event-date trade exists",
        ),
    )
    assert result.market_reaction_eligibility == MarketReactionEligibility.ELIGIBLE
    assert should_attempt_market_maturation(result) is True
    assert result.market_history_request_avoided is False


def test_future_event_never_triggers_market_lookup() -> None:
    result = evaluate_event_eligibility(
        event_id="future",
        ticker="CHEP",
        published_at_utc=datetime(2026, 8, 11, 0, 0, tzinfo=UTC),
        identity_status=InstrumentIdentityStatus.RESOLVED,
        evidence=_positive_non_trading_evidence(),
    )
    assert (
        result.instrument_identity_status == InstrumentIdentityStatus.NOT_EVALUATED_FUTURE_HOLDOUT
    )
    assert result.market_reaction_eligibility == MarketReactionEligibility.FUTURE_METADATA_ONLY
    assert result.reaction_attempt_skipped is False
    assert result.market_history_request_avoided is False


def test_artifact_applies_chep_guard_and_preserves_events(tmp_path: Path) -> None:
    diagnostic_root, maturation_root = _write_artifacts(tmp_path)
    manifest = run_security_tradability_eligibility(
        diagnostic_root=diagnostic_root,
        maturation_root=maturation_root,
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="a" * 40,
        git_sha="b" * 40,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
    )
    assert manifest["CHEP_HISTORICAL_EXACT_EVENTS"] == 44
    assert manifest["CHEP_EVENT_VALID"] == 44
    assert manifest["CHEP_CANONICAL_EVENTS_PRESERVED"] == 44
    assert manifest["CHEP_MARKET_REACTION_ELIGIBLE"] == 0
    assert manifest["CHEP_SECURITY_NOT_TRADING"] == 44
    assert manifest["CHEP_REACTION_ATTEMPTS_SKIPPED"] == 44
    assert manifest["CHEP_FEATURE_ATTEMPTS_SKIPPED"] == 44
    assert manifest["FUTURE_CHEP_EVENTS"] == 6
    assert manifest["FUTURE_CHEP_PRICE_LOOKUPS"] == 0
    assert manifest["FUTURE_CHEP_REACTION_ATTEMPTS"] == 0
    assert manifest["FUTURE_CHEP_TARGET_ATTEMPTS"] == 0
    assert manifest["CHEP_COLLECTION_DECISION"] == "KEEP_METADATA_ONLY"
    rows = _read_jsonl(tmp_path / ARTIFACT_VERSION / "event-eligibility.jsonl")
    historical = [
        row for row in rows if row["market_reaction_eligibility"] != "FUTURE_METADATA_ONLY"
    ]
    assert len(historical) == 44
    assert {row["event_validity"] for row in historical} == {"VALID_EXACT_EVENT"}


def test_deterministic_eligibility_sha_and_replay(tmp_path: Path) -> None:
    diagnostic_root, maturation_root = _write_artifacts(tmp_path)
    first = run_security_tradability_eligibility(
        diagnostic_root=diagnostic_root,
        maturation_root=maturation_root,
        output_root=tmp_path / "first",
        base_main_sha="a" * 40,
        git_sha="b" * 40,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
    )
    second = run_security_tradability_eligibility(
        diagnostic_root=diagnostic_root,
        maturation_root=maturation_root,
        output_root=tmp_path / "second",
        base_main_sha="a" * 40,
        git_sha="b" * 40,
        created_at=datetime(2026, 8, 28, tzinfo=UTC),
    )
    assert first["ELIGIBILITY_RESULT_SHA"] == second["ELIGIBILITY_RESULT_SHA"]
    assert first["ARTIFACT_SHA"] == second["ARTIFACT_SHA"]
    assert first["ARTIFACT_SHA"] == sha256_payload({**first, "ARTIFACT_SHA": None})


def test_documentation_states_empty_candles_rule() -> None:
    text = (
        Path(__file__).parents[2] / "docs" / "exact-event-security-tradability-eligibility-v1.md"
    ).read_text(encoding="utf-8")
    assert "MODEL_TRAINING_PERFORMED=false" in text
    assert "FUTURE_EVENT_HOLDOUT_OBSERVED=false" in text
    assert "REAL_ORDER_SUBMISSION_ALLOWED=false" in text
    assert "Empty candle responses alone never prove non-trading" in text


def _positive_non_trading_evidence() -> TradingEvidence:
    return TradingEvidence(
        ticker="CHEP",
        instrument_uid="b1f4f4fc-dac5-4e29-ae56-95fe441416ee",
        figi="BBG000Q49F45",
        class_code="TQBR",
        source="CHEP_SECURITY_HISTORY_DIAGNOSTICS_V1",
        security_history_confirmed=True,
        event_date_trading_confirmed=False,
        last_confirmed_trading_date=date(2021, 9, 21),
        current_trading_status="SECURITY_TRADING_STATUS_NOT_AVAILABLE_FOR_TRADING",
        api_trade_available=False,
        buy_available=False,
        sell_available=False,
        evidence_detail="official MOEX no event-date rows and historical 2021 rows",
    )


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    diagnostic_root = tmp_path / "diagnostic"
    maturation_root = tmp_path / "maturation"
    diagnostic_root.mkdir()
    maturation_root.mkdir()
    _write_json(
        diagnostic_root / "manifest.json",
        {
            "ARTIFACT_SHA": "b31fca68eccde1aa009f0a992130f8afb4ce8281cf7db0a7808eaebc81740497",
            "INPUT_MATURATION_ARTIFACT_SHA": (
                "236ab1579cafda265eceeefc148b359d3ab2e5c54538d1d434bc789fc5775305"
            ),
            "PRIMARY_ROOT_CAUSE": "HISTORICAL_SECURITY_NOT_SUPPORTED",
            "RECOVERY_FEASIBILITY": "NOT_RECOVERABLE_WITH_ZERO_COST_SOURCES",
            "FUTURE_EVENT_HOLDOUT_USED": False,
            "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
            "MODEL_TRAINING_PERFORMED": False,
            "TEST_OUTCOME_USED": False,
            "TEST_EVALUATION_PERFORMED": False,
        },
    )
    _write_json(diagnostic_root / "diagnostic-report.json", _diagnostic_report())
    historical_start = datetime(2026, 6, 28, 9, 0, tzinfo=UTC)
    historical = [
        _cohort_row(f"chep-{index:02d}", historical_start + timedelta(days=index))
        for index in range(44)
    ]
    future = [
        _cohort_row(f"future-{index}", datetime(2026, 8, 11 + index, 9, 0, tzinfo=UTC))
        for index in range(6)
    ]
    events = [_event(row["event_id"], row["publication_timestamp_utc"]) for row in historical]
    events.extend(_event(row["event_id"], row["publication_timestamp_utc"]) for row in future)
    _write_json(
        maturation_root / "manifest.json",
        {"ARTIFACT_SHA": "236ab1579cafda265eceeefc148b359d3ab2e5c54538d1d434bc789fc5775305"},
    )
    _write_jsonl(maturation_root / "historical-cohort.jsonl", historical)
    _write_jsonl(maturation_root / "future-metadata-cohort.jsonl", future)
    _write_jsonl(maturation_root / "events.jsonl", events)
    _write_jsonl(maturation_root / "features.jsonl", [])
    return diagnostic_root, maturation_root


def _diagnostic_report() -> dict[str, object]:
    return {
        "metrics": {
            "PRIMARY_ROOT_CAUSE": "HISTORICAL_SECURITY_NOT_SUPPORTED",
            "RECOVERY_FEASIBILITY": "NOT_RECOVERABLE_WITH_ZERO_COST_SOURCES",
            "MOEX_SECURITY_HISTORY_CONFIRMED": True,
            "MOEX_EVENT_DATE_TRADING_CONFIRMED": False,
        },
        "instrument_candidates": [
            {
                "classification": "CURRENT_CONFIRMED",
                "ticker": "CHEP",
                "figi": "BBG000Q49F45",
                "instrument_uid": "b1f4f4fc-dac5-4e29-ae56-95fe441416ee",
                "class_code": "TQBR",
                "trading_status": "SECURITY_TRADING_STATUS_NOT_AVAILABLE_FOR_TRADING",
                "api_trade_available_flag": False,
                "buy_available_flag": False,
                "sell_available_flag": False,
            }
        ],
        "candle_probes": [
            {
                "label": "known_last_daily_window_current_identity",
                "interval": "1d",
                "returned_candle_count": 2,
                "last_returned_timestamp": "2021-09-21",
            }
        ],
        "moex_cross_check": {
            "MOEX_REQUESTS": [
                {
                    "label": "last_known_tinvest_daily_history",
                    "returned_row_count": 2,
                    "last_returned_timestamp": "2021-09-21",
                }
            ]
        },
    }


def _cohort_row(event_id: object, published_at: datetime | str) -> dict[str, object]:
    published = published_at if isinstance(published_at, str) else published_at.isoformat()
    return {
        "event_id": str(event_id),
        "ticker": "CHEP",
        "publication_timestamp_utc": published,
        "source_item_id": f"https://example.test/{event_id}",
    }


def _event(event_id_value: object, published_at: object) -> dict[str, object]:
    return {
        "metadata": {
            "event_id": str(event_id_value),
            "ticker": "CHEP",
            "publication_timestamp_utc": str(published_at),
        },
        "target_availability": {
            "reaction_ready": False,
            "feature_ready": False,
            "research_outcomes_visible": False,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
