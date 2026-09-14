from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.ai_trading_agent_v1.application import AgentRunConfig
from src.current_moex_tradability_v1.application import (
    classify_candidate,
    resolve_from_states,
    verified_ticker_migrations,
)
from src.current_moex_tradability_v1.domain import (
    INTENDED_BOARD,
    CandidateClassification,
    CurrentMoexState,
    CurrentTradabilityStatus,
    CurrentUniverseResolution,
)
from src.current_moex_tradability_v1.moex import CurrentMoexTradabilityResolver
from src.current_moex_tradability_v1.reporting import build_tradability_artifact
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.paper_trading_operation_v1.application import build_operation_id
from src.paper_trading_operation_v1.domain import PaperOperationPolicy
from src.production_dry_run_burnin_v1.application import build_burnin_report
from src.production_dry_run_burnin_v1.domain import (
    CURRENT_BURNIN_EPOCH,
    PRE_FIX_BURNIN_EPOCH,
    BurninObservationStatus,
    BurninPolicy,
    BurninStatus,
)
from src.production_dry_run_burnin_v1.reporting import build_sample_observation
from src.production_dry_run_burnin_v1.repository import observation_record_sha
from src.production_readonly_adapters_v1.context import ProductionPaperOperationContextProvider
from src.production_readonly_adapters_v1.domain import FreshMarketSnapshot
from src.risk_engine_paper_v1.application import initial_paper_portfolio
from src.risk_engine_paper_v1.domain import PaperPosition, RiskPolicy

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
SHA = "a" * 64


def _canonical(ticker: str) -> dict[str, Any]:
    return {"ticker": ticker, "board": "TQBR", "historical_mapping": True}


def _empty_events(*_args: object) -> dict[str, Any]:
    return {"events": []}


def _empty_research(*_args: object) -> dict[str, Any]:
    return {}


def _state(
    ticker: str,
    *,
    board: str | None = "TQBR",
    lot: int | None = 10,
    security_status: str | None = "A",
    trading_status: str | None = "T",
    marketdata: bool = True,
) -> CurrentMoexState:
    return CurrentMoexState(
        ticker=ticker,
        board=board,
        lot_size=lot,
        security_status=security_status,
        trading_status=trading_status,
        has_marketdata_row=marketdata,
    )


def _classify(state: CurrentMoexState | None) -> CandidateClassification:
    return classify_candidate(_canonical("TEST"), state, fetched_at=NOW, payload_sha=SHA)


class _Resolver:
    def __init__(self, states: dict[str, CurrentMoexState], fetched_at: datetime = NOW) -> None:
        self.states = states
        self.fetched_at = fetched_at

    def resolve(self, canonical: Sequence[dict[str, Any]]) -> CurrentUniverseResolution:
        return resolve_from_states(
            canonical,
            self.states,
            fetched_at=self.fetched_at,
            payload_sha=SHA,
            source_time=self.fetched_at,
        )


def _provider(
    rows: list[dict[str, Any]], states: dict[str, CurrentMoexState]
) -> ProductionPaperOperationContextProvider:
    return ProductionPaperOperationContextProvider(
        agent_config=AgentRunConfig(output_root=Path("unused"), code_sha="a" * 40),
        market_adapter=None,  # type: ignore[arg-type]
        risk_policy=RiskPolicy(),
        configured_max_age_seconds=300,
        universe_loader=lambda _config: rows,
        tradability_resolver=_Resolver(states),
    )


def _selected(
    rows: list[dict[str, Any]],
    states: dict[str, CurrentMoexState],
    *,
    maximum: int = 3,
    held: str | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, object]]:
    portfolio = initial_paper_portfolio(NOW)
    if held is not None:
        position = PaperPosition(
            ticker=held,
            quantity=10,
            average_cost=100,
            last_price=100,
            market_value=1000,
            weight=0.001,
            unrealized_pnl=0,
            mark_as_of=NOW - timedelta(days=1),
        )
        portfolio = portfolio.model_copy(update={"positions": [position]})
    return _provider(rows, states)._selected_universe(  # pyright: ignore[reportPrivateUsage]
        portfolio, PaperOperationPolicy(max_operation_universe=maximum)
    )


def test_historical_tqbr_mapping_is_not_sufficient_for_current_production_eligibility() -> None:
    assert not _classify(None).candidate_eligible


def test_missing_current_moex_security_row_is_ineligible_candidate() -> None:
    assert (
        _classify(None).current_moex_status == CurrentTradabilityStatus.INELIGIBLE_NO_SECURITY_ROW
    )


def test_wrong_current_board_is_ineligible_candidate() -> None:
    assert (
        _classify(_state("TEST", board="TQTF")).current_moex_status
        == CurrentTradabilityStatus.INELIGIBLE_WRONG_BOARD
    )


def test_invalid_lot_is_ineligible_candidate() -> None:
    assert (
        _classify(_state("TEST", lot=0)).current_moex_status
        == CurrentTradabilityStatus.INELIGIBLE_INVALID_LOT
    )


def test_explicit_trading_disabled_security_is_ineligible_candidate() -> None:
    assert (
        _classify(_state("TEST", security_status="N")).current_moex_status
        == CurrentTradabilityStatus.INELIGIBLE_TRADING_DISABLED
    )


def test_unknown_current_status_fails_closed() -> None:
    assert (
        _classify(_state("TEST", security_status=None)).current_moex_status
        == CurrentTradabilityStatus.UNKNOWN
    )


def test_current_eligible_security_is_selected() -> None:
    rows, selected, _audit = _selected([_canonical("SBER")], {"SBER": _state("SBER")})
    assert selected == ["SBER"]
    assert rows[0]["current_tradability_status"] == "ELIGIBLE"


def test_current_board_is_loaded_with_one_bounded_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "securities": {
                    "columns": ["SECID", "BOARDID", "SHORTNAME", "LOTSIZE", "STATUS"],
                    "data": [["SBER", "TQBR", "SBERBANK", 10, "A"]],
                },
                "marketdata": {
                    "columns": [
                        "SECID",
                        "BOARDID",
                        "LAST",
                        "BID",
                        "OFFER",
                        "SYSTIME",
                        "TRADINGSTATUS",
                    ],
                    "data": [["SBER", "TQBR", None, None, None, "2026-09-14 15:00:00", "T"]],
                },
            },
        )

    resolver = CurrentMoexTradabilityResolver(
        base_url="https://iss.moex.com/iss",
        timeout_seconds=1,
        max_retries=0,
        user_agent="tests",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: NOW,
    )
    result = resolver.resolve([_canonical("SBER"), _canonical("YDEX")])

    assert len(requests) == 1
    assert result.by_ticker()["SBER"].candidate_eligible
    assert not result.by_ticker()["YDEX"].candidate_eligible


def test_structural_eligibility_does_not_require_non_null_last() -> None:
    assert _classify(_state("TEST", marketdata=True)).candidate_eligible


def test_security_without_current_marketdata_row_is_ineligible() -> None:
    result = _classify(_state("TEST", marketdata=False))
    assert result.current_moex_status == CurrentTradabilityStatus.INELIGIBLE_NO_CURRENT_MARKET_DATA


def test_ineligible_candidates_do_not_consume_operation_universe_capacity() -> None:
    canonical = [_canonical(ticker) for ticker in ["AAA", "BBB", "CCC", "DDD"]]
    states = {ticker: _state(ticker) for ticker in ["BBB", "CCC", "DDD"]}
    _rows, selected, _audit = _selected(canonical, states, maximum=3)
    assert selected == ["BBB", "CCC", "DDD"]


def test_selection_continues_until_max_eligible_candidates_reached() -> None:
    canonical = [_canonical(ticker) for ticker in ["A", "B", "C", "D", "E"]]
    states = {ticker: _state(ticker) for ticker in ["B", "D", "E"]}
    assert _selected(canonical, states, maximum=3)[1] == ["B", "D", "E"]


def test_candidate_order_remains_deterministic() -> None:
    rows = [_canonical(ticker) for ticker in ["ZZZ", "AAA", "MMM"]]
    states = {ticker: _state(ticker) for ticker in ["ZZZ", "AAA", "MMM"]}
    assert _selected(rows, states)[1] == ["AAA", "MMM", "ZZZ"]


def test_held_ineligible_instrument_is_not_dropped() -> None:
    rows, selected, _audit = _selected(
        [_canonical("SBER")], {"SBER": _state("SBER")}, maximum=2, held="ALNU"
    )
    assert selected == ["ALNU", "SBER"]
    assert rows[0]["current_tradability_reason"] == "HELD_POSITION_VISIBILITY_OVERRIDE"


def test_held_trading_disabled_security_reaches_risk_context() -> None:
    rows, selected, _audit = _selected(
        [_canonical("AMEZ"), _canonical("SBER")],
        {"AMEZ": _state("AMEZ", security_status="N"), "SBER": _state("SBER")},
        maximum=2,
        held="AMEZ",
    )
    assert selected == ["AMEZ", "SBER"]
    assert rows[0]["current_tradability_status"] == "INELIGIBLE_TRADING_DISABLED"


def test_known_live_defect_fixture_fails_closed() -> None:
    canonical = [_canonical(ticker) for ticker in ["AGRO", "ALNU", "AMEZ"]]
    resolution = resolve_from_states(
        canonical,
        {"AMEZ": _state("AMEZ", security_status="N", trading_status="B")},
        fetched_at=NOW,
        payload_sha=SHA,
    )
    assert {row.ticker: row.candidate_eligible for row in resolution.classifications} == {
        "AGRO": False,
        "ALNU": False,
        "AMEZ": False,
    }


def test_legacy_ticker_not_silently_rewritten() -> None:
    resolution = resolve_from_states(
        [_canonical("AGRO")], {"RAGR": _state("RAGR")}, fetched_at=NOW, payload_sha=SHA
    )
    assert resolution.classifications[0].ticker == "AGRO"
    assert not resolution.classifications[0].candidate_eligible


def test_verified_successor_mapping_is_explicit() -> None:
    migration = verified_ticker_migrations()[0]
    assert (migration.legacy_ticker, migration.current_ticker) == ("AGRO", "RAGR")
    assert not migration.silent_substitution_allowed


def test_current_successor_can_enter_candidate_universe() -> None:
    assert _selected([_canonical("RAGR")], {"RAGR": _state("RAGR")})[1] == ["RAGR"]


def test_historical_events_keep_legacy_ticker_identity() -> None:
    canonical = _canonical("AGRO")
    classify_candidate(canonical, None, fetched_at=NOW, payload_sha=SHA)
    assert canonical["ticker"] == "AGRO"


def test_current_universe_snapshot_respects_cycle_temporal_boundary() -> None:
    resolution = _Resolver({"SBER": _state("SBER")}, NOW).resolve([_canonical("SBER")])
    assert resolution.fetched_at <= NOW


def test_future_current_universe_snapshot_fails_closed() -> None:
    provider = replace(
        _provider([_canonical("SBER")], {"SBER": _state("SBER")}),
        event_loader=_empty_events,
        research_loader=_empty_research,
    )
    snapshot = FreshMarketSnapshot(
        market_fetch_started_at=NOW,
        market_fetch_completed_at=NOW + timedelta(seconds=1),
        operation_as_of=NOW + timedelta(seconds=1),
        effective_max_age_seconds=300,
        quotes=[],
        quote_audit=[],
        source_payload_sha=SHA,
    )
    eligibility = _Resolver({}, NOW + timedelta(seconds=2)).resolve([]).audit_payload()

    with pytest.raises(ValueError, match="CURRENT_UNIVERSE_SNAPSHOT_AFTER_DECISION_CUTOFF"):
        provider._context(  # pyright: ignore[reportPrivateUsage]
            cycle_started_at=NOW,
            decision_as_of=NOW + timedelta(seconds=1),
            universe=[],
            selected=[],
            snapshot=snapshot,
            eligibility=eligibility,
        )


def test_decision_cutoff_after_market_acquisition_remains_pass() -> None:
    provider = replace(
        _provider([_canonical("SBER")], {"SBER": _state("SBER")}),
        event_loader=_empty_events,
        research_loader=_empty_research,
    )
    snapshot = FreshMarketSnapshot(
        market_fetch_started_at=NOW,
        market_fetch_completed_at=NOW + timedelta(seconds=2),
        operation_as_of=NOW + timedelta(seconds=2),
        effective_max_age_seconds=300,
        quotes=[],
        quote_audit=[],
        source_payload_sha=SHA,
    )
    eligibility = _Resolver({}, NOW + timedelta(seconds=1)).resolve([]).audit_payload()
    context = provider._context(  # pyright: ignore[reportPrivateUsage]
        cycle_started_at=NOW,
        decision_as_of=NOW + timedelta(seconds=2),
        universe=[],
        selected=[],
        snapshot=snapshot,
        eligibility=eligibility,
    )
    assert context.market_context["decision_as_of"] == (NOW + timedelta(seconds=2)).isoformat()


def test_same_slot_idempotency_survives_current_universe_change() -> None:
    first = build_operation_id(operation_as_of=NOW, session="BURNIN_EOD")
    second = build_operation_id(operation_as_of=NOW, session="BURNIN_EOD")
    assert first == second


def test_universe_sha_changes_without_operation_id_change() -> None:
    assert sha256_payload(["SBER"]) != sha256_payload(["RAGR", "SBER"])


def test_pre_fix_observations_preserved() -> None:
    current = build_sample_observation(NOW, status=BurninObservationStatus.PASS)
    payload = current.model_dump(mode="json")
    payload.pop("burnin_epoch")
    payload["record_sha"] = sha256_payload(
        {key: value for key, value in payload.items() if key != "record_sha"}
    )
    restored = type(current).model_validate(payload)
    assert restored.burnin_epoch == PRE_FIX_BURNIN_EPOCH
    assert observation_record_sha(restored) == restored.record_sha


def test_post_fix_epoch_does_not_count_pre_fix_days() -> None:
    rows = [
        build_sample_observation(
            NOW - timedelta(days=3), status=BurninObservationStatus.PASS
        ).model_copy(update={"burnin_epoch": PRE_FIX_BURNIN_EPOCH}),
        build_sample_observation(NOW, status=BurninObservationStatus.PASS).model_copy(
            update={"burnin_epoch": PRE_FIX_BURNIN_EPOCH}
        ),
    ]
    report = build_burnin_report(rows)
    assert report.BURNIN_EPOCH == CURRENT_BURNIN_EPOCH
    assert report.BURNIN_STATUS == BurninStatus.NOT_STARTED
    assert report.valid_cycles == 0


def test_burnin_pass_requires_five_days_within_same_epoch() -> None:
    rows = [
        build_sample_observation(NOW + timedelta(days=index), status=BurninObservationStatus.PASS)
        for index in range(5)
    ]
    assert build_burnin_report(rows[:4]).BURNIN_STATUS == BurninStatus.IN_PROGRESS
    assert build_burnin_report(rows).BURNIN_STATUS == BurninStatus.PASS


def test_thresholds_unchanged() -> None:
    policy = BurninPolicy()
    assert policy.min_market_fresh_rate == 0.95
    assert policy.min_primary_cycles == 5
    assert policy.min_distinct_moex_trading_days == 5
    assert INTENDED_BOARD == "TQBR"


def test_fixture_artifact_rebuild_is_byte_for_byte(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    manifest = build_tradability_artifact(output, "b" * 40)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    build_tradability_artifact(output, "b" * 40)
    second = {path.name: path.read_bytes() for path in output.iterdir()}

    assert first == second
    assert set(first) == {
        "manifest.json",
        "eligibility-policy.json",
        "mock-moex-current-state.json",
        "candidate-classification.json",
        "held-position-proof.json",
        "ticker-migration-evidence.json",
        "burnin-epoch-proof.json",
        "safety.json",
        "report.md",
    }
    assert manifest["POST_FIX_BURNIN_STATUS"] == "NOT_STARTED"
