from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from apps.cli.paper_operation import _policy_from_env  # pyright: ignore[reportPrivateUsage]
from src.ai_trading_agent_v1.application import (
    AgentDataContext,
    AgentRunConfig,
    FakeAgentModel,
    UnconfiguredAgentModel,
    build_read_only_tool_registry,
)
from src.ai_trading_agent_v1.domain import AgentModelResponse
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.paper_trading_operation_v1.application import (
    PaperOperationContext,
    SimulatedCrashAfterPaperFillError,
    StaticPaperOperationContextProvider,
    build_operation_contract_sha,
    build_operation_id,
    run_paper_operation,
)
from src.paper_trading_operation_v1.domain import (
    PaperOperationMode,
    PaperOperationPolicy,
    PaperOperationRun,
    PaperOperationStatus,
)
from src.paper_trading_operation_v1.reporting import (
    build_sample_operations,
    write_operation_artifact,
)
from src.paper_trading_operation_v1.repository import (
    InMemoryOperationAuditRepository,
    JsonlOperationAuditRepository,
    OperationAlreadyRunningError,
    OperationAuditRepository,
    operation_lock,
)
from src.risk_engine_paper_v1.application import (
    evaluate_agent_run,
    execute_paper_plan,
    initial_paper_portfolio,
    portfolio_state_sha,
    replay_portfolio,
)
from src.risk_engine_paper_v1.domain import (
    MarketQuote,
    PaperExecutionStatus,
    RiskPolicy,
)
from src.risk_engine_paper_v1.repository import (
    InMemoryPaperLedgerRepository,
    JsonlPaperLedgerRepository,
)

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)


def _universe() -> list[dict[str, Any]]:
    return [
        {
            "ticker": ticker,
            "legal_issuer": issuer,
            "instrument_uid": f"UID-{ticker}",
            "figi": f"FIGI-{ticker}",
            "board": "TQBR",
            "instrument_type": "INSTRUMENT_TYPE_SHARE",
            "active": True,
            "supported": True,
            "market_data_compatible": True,
            "feature_compatible": True,
            "lot_size": 10 if ticker == "SBER" else 1,
        }
        for ticker, issuer in (("SBER", "Sberbank"), ("YDEX", "Yandex"))
    ]


def _quote(ticker: str, as_of: datetime, *, price: float | None = None) -> MarketQuote:
    value = (
        price
        if price is not None
        else {"GAZP": 160.0, "ROSN": 520.0, "SBER": 100.0, "YDEX": 110.0}[ticker]
    )
    return MarketQuote(
        ticker=ticker,
        as_of=as_of,
        last_price=value,
        bid=value - 0.2,
        ask=value + 0.2,
        lot_size=10 if ticker == "SBER" else 1,
    )


def _context(
    as_of: datetime,
    *tickers: str,
    research_ready: bool = True,
    event_published_at: datetime | None = None,
) -> PaperOperationContext:
    quotes = [_quote(ticker, as_of) for ticker in tickers]
    by_ticker = {
        quote.ticker: {
            "ticker": quote.ticker,
            "market_data_as_of": quote.as_of.isoformat(),
            "last_price": quote.last_price,
            "bid": quote.bid,
            "ask": quote.ask,
            "stale": False,
        }
        for quote in quotes
    }
    events: list[dict[str, Any]] = []
    if event_published_at is not None:
        events.append(
            {
                "event_id": "event-1",
                "ticker": "SBER",
                "published_at": event_published_at.isoformat(),
            }
        )
    return PaperOperationContext(
        universe=_universe(),
        market_quotes=quotes,
        market_context={"as_of": as_of.isoformat(), "by_ticker": by_ticker},
        event_context={"events_as_of": as_of.isoformat(), "events": events},
        research_status={
            "research_status_as_of": as_of.isoformat(),
            "LIVE_RESEARCH_OPERATION_STATUS": "READY" if research_ready else "DEGRADED",
            "OPERATIONAL_BURN_IN": "PASS",
            "SOURCE_FAILURE_ISOLATION": True,
            "SOURCE_FAILURE_ISOLATION_PROOF_LEVEL": "APPLICATION_PROOF",
            "seal": {"sealed_epoch_verified": True, "violations": 0},
            "ML_V2_DATASET_STATUS": "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY",
        },
    )


def _proposal(ticker: str, action: str, weight: float) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "action": action,
        "agent_confidence": 0.6,
        "target_weight": weight,
        "holding_horizon": "1-5d",
        "thesis": ["deterministic operation test"],
        "risks": ["paper only"],
        "evidence": [],
        "data_quality": "GOOD",
    }


def _model(as_of: datetime, *proposals: dict[str, Any]) -> FakeAgentModel:
    output = {
        "as_of": as_of.isoformat(),
        "input_snapshot_as_of": {
            "portfolio_as_of": as_of.isoformat(),
            "market_data_as_of": as_of.isoformat(),
            "events_as_of": as_of.isoformat(),
        },
        "proposals": list(proposals),
    }
    return FakeAgentModel(
        [AgentModelResponse(final_output=json.dumps(output))],
        model_id="fake-operation-agent-v1",
    )


class PortfolioPriorityContextProvider:
    def __init__(self) -> None:
        self.loaded_universes: list[list[str]] = []
        self.canonical = [
            {
                "ticker": ticker,
                "legal_issuer": issuer,
                "instrument_uid": f"UID-{ticker}",
                "figi": f"FIGI-{ticker}",
                "board": "TQBR",
                "instrument_type": "INSTRUMENT_TYPE_SHARE",
                "active": True,
                "supported": True,
                "market_data_compatible": True,
                "feature_compatible": True,
                "lot_size": 10 if ticker == "SBER" else 1,
            }
            for ticker, issuer in (
                ("GAZP", "Gazprom"),
                ("ROSN", "Rosneft"),
                ("SBER", "Sberbank"),
                ("YDEX", "Yandex"),
            )
        ]

    def load(
        self,
        *,
        operation_as_of: datetime,
        portfolio: Any,
        policy: PaperOperationPolicy,
    ) -> PaperOperationContext:
        by_ticker = {str(row["ticker"]): row for row in self.canonical}
        held = [position.ticker for position in portfolio.positions]
        selected = list(dict.fromkeys([*held, *by_ticker]))[: policy.max_operation_universe]
        self.loaded_universes.append(selected)
        quotes = [_quote(ticker, operation_as_of) for ticker in selected]
        context = _context(operation_as_of, *selected)
        return replace(
            context,
            universe=[by_ticker[ticker] for ticker in selected],
            market_quotes=quotes,
        )


def _run(
    tmp_path: Path,
    *,
    as_of: datetime,
    proposals: tuple[dict[str, Any], ...],
    context: PaperOperationContext,
    paper: InMemoryPaperLedgerRepository | JsonlPaperLedgerRepository,
    audit: OperationAuditRepository,
    mode: PaperOperationMode = PaperOperationMode.PAPER_EXECUTE,
    model: FakeAgentModel | None = None,
    policy: PaperOperationPolicy | None = None,
    operation_slot: str | None = None,
    simulate_crash_after_fill: bool = False,
) -> PaperOperationRun:
    return run_paper_operation(
        operation_as_of=as_of,
        mode=mode,
        model=model or _model(as_of, *proposals),
        context_provider=StaticPaperOperationContextProvider(context),
        paper_repository=paper,
        audit_repository=audit,
        state_root=tmp_path / "operation-state",
        code_sha="a" * 40,
        policy=policy or PaperOperationPolicy(paper_execution_enabled=True),
        operation_slot=operation_slot,
        simulate_crash_after_fill=simulate_crash_after_fill,
    )


def test_operation_dry_run_does_not_mutate_portfolio(tmp_path: Path) -> None:
    paper = InMemoryPaperLedgerRepository()
    run = _run(
        tmp_path,
        as_of=NOW,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(NOW, "SBER", "YDEX"),
        paper=paper,
        audit=InMemoryOperationAuditRepository(),
        mode=PaperOperationMode.DRY_RUN,
    )

    assert run.status == PaperOperationStatus.SUCCESS
    assert run.paper_execution_status == "SKIPPED_DRY_RUN"
    assert run.safety.PAPER_ORDERS_FILLED == 0
    assert paper.events() == []


def test_operation_execute_runs_agent_risk_paper(tmp_path: Path) -> None:
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    run = _run(
        tmp_path,
        as_of=NOW,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(NOW, "SBER", "YDEX"),
        paper=paper,
        audit=audit,
    )

    assert run.status == PaperOperationStatus.SUCCESS
    assert run.agent_proposals[0]["ticker"] == "SBER"
    assert run.risk_plan_id
    assert run.paper_trade_ids
    assert run.replay_verified is True
    assert len(audit.events()) == 2


def test_operation_uses_existing_portfolio_and_next_day_sees_positions(tmp_path: Path) -> None:
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    first = _run(
        tmp_path,
        as_of=NOW,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(NOW, "SBER", "YDEX"),
        paper=paper,
        audit=audit,
    )
    next_day = NOW + timedelta(days=1)
    second = _run(
        tmp_path,
        as_of=next_day,
        proposals=(_proposal("SBER", "HOLD", 0.10), _proposal("YDEX", "BUY", 0.20)),
        context=_context(next_day, "SBER", "YDEX"),
        paper=paper,
        audit=audit,
    )

    assert second.portfolio_before["cash"] == first.portfolio_after["cash"]
    assert {row["ticker"] for row in second.portfolio_before["positions"]} == {"SBER"}
    assert second.day_transition_applied is True
    assert {row["ticker"] for row in second.portfolio_after["positions"]} == {"SBER", "YDEX"}


def test_operation_audit_written_and_replay_verified(tmp_path: Path) -> None:
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    run = _run(
        tmp_path,
        as_of=NOW,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(NOW, "SBER", "YDEX"),
        paper=paper,
        audit=audit,
    )

    assert audit.completed_run(run.operation_id) == run
    assert run.operation_slot_id == "2026-09-09:EOD"
    assert run.universe_sha is not None
    assert run.universe_sha == sha256_payload(run.universe)
    assert run.operation_contract_sha == build_operation_contract_sha(
        model_id=run.agent_model_id,
        universe_sha=run.universe_sha,
        policy_version=run.policy_version,
        code_sha=run.code_sha,
    )
    assert portfolio_state_sha(replay_portfolio(paper.events())) == run.portfolio_after_sha


def test_research_not_ready_blocks_operation_before_agent(tmp_path: Path) -> None:
    model = _model(NOW, _proposal("SBER", "BUY", 0.10))
    run = _run(
        tmp_path,
        as_of=NOW,
        proposals=(),
        context=_context(NOW, "SBER", "YDEX", research_ready=False),
        paper=InMemoryPaperLedgerRepository(),
        audit=InMemoryOperationAuditRepository(),
        model=model,
    )

    assert run.status == PaperOperationStatus.BLOCKED
    assert run.status_code == "LIVE_RESEARCH_NOT_READY"
    assert model.requests == []


def test_bad_ledger_blocks_before_agent(tmp_path: Path) -> None:
    ledger = tmp_path / "bad-ledger.jsonl"
    ledger.write_text("not-json\n", encoding="utf-8")
    model = _model(NOW, _proposal("SBER", "BUY", 0.10))
    run = _run(
        tmp_path,
        as_of=NOW,
        proposals=(),
        context=_context(NOW, "SBER", "YDEX"),
        paper=JsonlPaperLedgerRepository(ledger),
        audit=InMemoryOperationAuditRepository(),
        model=model,
    )

    assert run.status == PaperOperationStatus.BLOCKED
    assert run.status_code == "PAPER_LEDGER_INTEGRITY_FAILED"
    assert model.requests == []


def test_missing_held_mark_blocks_buy(tmp_path: Path) -> None:
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    _run(
        tmp_path,
        as_of=NOW,
        proposals=(_proposal("YDEX", "BUY", 0.10),),
        context=_context(NOW, "SBER", "YDEX"),
        paper=paper,
        audit=audit,
        operation_slot="EOD_MARK_COMPLETENESS_PROOF",
    )
    later = NOW + timedelta(minutes=1)
    degraded = _run(
        tmp_path,
        as_of=later,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(later, "SBER"),
        paper=paper,
        audit=audit,
    )

    assert degraded.status == PaperOperationStatus.DEGRADED
    assert degraded.safety.PAPER_ORDERS_FILLED == 0
    assert degraded.risk_decisions[0]["reason_codes"] == ["PORTFOLIO_MARK_INCOMPLETE"]


def test_stale_market_blocks_exposure_increase(tmp_path: Path) -> None:
    context = replace(
        _context(NOW, "SBER", "YDEX"),
        market_quotes=[_quote("SBER", NOW - timedelta(hours=1)), _quote("YDEX", NOW)],
    )
    run = _run(
        tmp_path,
        as_of=NOW,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=context,
        paper=InMemoryPaperLedgerRepository(),
        audit=InMemoryOperationAuditRepository(),
    )

    assert run.status == PaperOperationStatus.DEGRADED
    assert run.safety.PAPER_ORDERS_FILLED == 0
    assert run.risk_decisions[0]["reason_codes"] == ["STALE_DATA"]


@pytest.mark.parametrize(
    ("context", "reason"),
    [
        (_context(NOW, "SBER", event_published_at=NOW + timedelta(seconds=1)), "FUTURE_EVENT"),
        (
            replace(
                _context(NOW, "SBER"),
                market_quotes=[_quote("SBER", NOW + timedelta(seconds=1))],
            ),
            "FUTURE_MARKET_QUOTE",
        ),
    ],
)
def test_future_pit_input_rejected(
    tmp_path: Path, context: PaperOperationContext, reason: str
) -> None:
    run = _run(
        tmp_path,
        as_of=NOW,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=context,
        paper=InMemoryPaperLedgerRepository(),
        audit=InMemoryOperationAuditRepository(),
    )

    assert run.status == PaperOperationStatus.BLOCKED
    assert reason in run.reasons


def test_duplicate_operation_no_new_fill(tmp_path: Path) -> None:
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    model = _model(NOW, _proposal("SBER", "BUY", 0.10))
    first = _run(
        tmp_path,
        as_of=NOW,
        proposals=(),
        context=_context(NOW, "SBER", "YDEX"),
        paper=paper,
        audit=audit,
        model=model,
    )
    count = len(paper.events())
    duplicate = _run(
        tmp_path,
        as_of=NOW,
        proposals=(),
        context=_context(NOW, "SBER", "YDEX"),
        paper=paper,
        audit=audit,
        model=model,
    )

    assert first.status == PaperOperationStatus.SUCCESS
    assert duplicate.status == PaperOperationStatus.ALREADY_PROCESSED
    assert duplicate.safety.PAPER_ORDERS_FILLED == 0
    assert len(paper.events()) == count


def test_same_session_different_wall_clock_is_already_processed(tmp_path: Path) -> None:
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    first_at = datetime(2026, 9, 9, 15, 0, 1, tzinfo=UTC)
    second_at = datetime(2026, 9, 9, 15, 3, 44, tzinfo=UTC)
    first = _run(
        tmp_path,
        as_of=first_at,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(first_at, "SBER", "YDEX"),
        paper=paper,
        audit=audit,
    )
    event_count = len(paper.events())
    duplicate = _run(
        tmp_path,
        as_of=second_at,
        proposals=(_proposal("SBER", "BUY", 0.20),),
        context=_context(second_at, "SBER", "YDEX"),
        paper=paper,
        audit=audit,
    )

    assert first.status == PaperOperationStatus.SUCCESS
    assert first.operation_slot_id == "2026-09-09:EOD"
    assert duplicate.operation_id == first.operation_id
    assert duplicate.status == PaperOperationStatus.ALREADY_PROCESSED
    assert duplicate.safety.PAPER_ORDERS_FILLED == 0
    assert duplicate.safety.PAPER_PORTFOLIO_MUTATIONS == 0
    assert len(paper.events()) == event_count


def test_same_date_different_operation_slot_is_distinct() -> None:
    assert build_operation_id(
        operation_as_of=NOW,
        session="EOD",
    ) != build_operation_id(
        operation_as_of=NOW,
        session="EOD_RETRY_1",
    )


def test_next_moex_date_is_distinct_operation() -> None:
    assert build_operation_id(
        operation_as_of=NOW,
        session="EOD",
    ) != build_operation_id(
        operation_as_of=NOW + timedelta(days=1),
        session="EOD",
    )


def test_production_provider_universe_reorder_cannot_bypass_slot_idempotency(
    tmp_path: Path,
) -> None:
    provider = PortfolioPriorityContextProvider()
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    first_at = NOW
    second_at = NOW + timedelta(minutes=3)
    first_model = _model(first_at, _proposal("SBER", "BUY", 0.10))
    first = run_paper_operation(
        operation_as_of=first_at,
        mode=PaperOperationMode.PAPER_EXECUTE,
        model=first_model,
        context_provider=provider,
        paper_repository=paper,
        audit_repository=audit,
        state_root=tmp_path,
        code_sha="a" * 40,
        policy=PaperOperationPolicy(paper_execution_enabled=True),
    )
    event_count = len(paper.events())
    second_model = _model(second_at, _proposal("SBER", "BUY", 0.20))
    duplicate = run_paper_operation(
        operation_as_of=second_at,
        mode=PaperOperationMode.PAPER_EXECUTE,
        model=second_model,
        context_provider=provider,
        paper_repository=paper,
        audit_repository=audit,
        state_root=tmp_path,
        code_sha="a" * 40,
        policy=PaperOperationPolicy(paper_execution_enabled=True),
    )

    assert first.status == PaperOperationStatus.SUCCESS
    assert provider.loaded_universes == [
        ["GAZP", "ROSN", "SBER", "YDEX"],
        ["SBER", "GAZP", "ROSN", "YDEX"],
    ]
    assert first.universe_sha == sha256_payload(
        [provider.canonical[index] for index in (0, 1, 2, 3)]
    )
    assert first.universe_sha != sha256_payload(
        [provider.canonical[index] for index in (2, 0, 1, 3)]
    )
    assert duplicate.status == PaperOperationStatus.ALREADY_PROCESSED
    assert second_model.requests == []
    assert duplicate.safety.PAPER_RISK_PLANS == 0
    assert duplicate.safety.PAPER_ORDERS_FILLED == 0
    assert len(paper.events()) == event_count


def test_truncated_universe_membership_change_cannot_bypass_slot_idempotency(
    tmp_path: Path,
) -> None:
    provider = PortfolioPriorityContextProvider()
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    first = run_paper_operation(
        operation_as_of=NOW,
        mode=PaperOperationMode.PAPER_EXECUTE,
        model=_model(NOW, _proposal("YDEX", "BUY", 0.10)),
        context_provider=provider,
        paper_repository=paper,
        audit_repository=audit,
        state_root=tmp_path,
        code_sha="a" * 40,
        policy=PaperOperationPolicy(max_operation_universe=4, paper_execution_enabled=True),
    )
    event_count = len(paper.events())
    retry_model = _model(NOW + timedelta(minutes=2), _proposal("GAZP", "BUY", 0.10))
    duplicate = run_paper_operation(
        operation_as_of=NOW + timedelta(minutes=2),
        mode=PaperOperationMode.PAPER_EXECUTE,
        model=retry_model,
        context_provider=provider,
        paper_repository=paper,
        audit_repository=audit,
        state_root=tmp_path,
        code_sha="b" * 40,
        policy=PaperOperationPolicy(max_operation_universe=3, paper_execution_enabled=True),
    )

    assert first.status == PaperOperationStatus.SUCCESS
    assert provider.loaded_universes == [
        ["GAZP", "ROSN", "SBER", "YDEX"],
        ["YDEX", "GAZP", "ROSN"],
    ]
    assert duplicate.status == PaperOperationStatus.ALREADY_PROCESSED
    assert retry_model.requests == []
    assert duplicate.safety.PAPER_ORDERS_FILLED == 0
    assert len(paper.events()) == event_count


def test_contract_change_same_slot_is_already_processed(tmp_path: Path) -> None:
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    first = _run(
        tmp_path,
        as_of=NOW,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(NOW, "SBER", "YDEX"),
        paper=paper,
        audit=audit,
    )
    changed_universe = list(reversed(_universe()))
    changed_context = replace(_context(NOW, "SBER", "YDEX"), universe=changed_universe)
    changed_model = FakeAgentModel(
        _model(NOW + timedelta(minutes=1), _proposal("SBER", "BUY", 0.20)).responses,
        model_id="changed-operation-agent-v2",
    )
    changed_contract_sha = build_operation_contract_sha(
        model_id=changed_model.model_id,
        universe_sha=sha256_payload(changed_universe),
        policy_version=PaperOperationPolicy().policy_version,
        code_sha="b" * 40,
    )
    event_count = len(paper.events())
    duplicate = _run(
        tmp_path,
        as_of=NOW + timedelta(minutes=1),
        proposals=(),
        context=changed_context,
        paper=paper,
        audit=audit,
        model=changed_model,
    )

    assert changed_contract_sha != first.operation_contract_sha
    assert duplicate.operation_id == first.operation_id
    assert duplicate.status == PaperOperationStatus.ALREADY_PROCESSED
    assert changed_model.requests == []
    assert duplicate.safety.PAPER_ORDERS_FILLED == 0
    assert len(paper.events()) == event_count


def test_paper_execution_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAPER_EXECUTION_ENABLED", raising=False)

    assert PaperOperationPolicy().paper_execution_enabled is False
    assert _policy_from_env().paper_execution_enabled is False


def test_execute_flag_without_env_gate_is_blocked(tmp_path: Path) -> None:
    paper = InMemoryPaperLedgerRepository()
    run = _run(
        tmp_path,
        as_of=NOW,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(NOW, "SBER", "YDEX"),
        paper=paper,
        audit=InMemoryOperationAuditRepository(),
        policy=PaperOperationPolicy(),
    )

    assert run.status == PaperOperationStatus.BLOCKED
    assert run.status_code == "PAPER_EXECUTION_DISABLED"
    assert run.safety.PAPER_ORDERS_FILLED == 0
    assert paper.events() == []


def test_execute_requires_both_explicit_gates(tmp_path: Path) -> None:
    dry_paper = InMemoryPaperLedgerRepository()
    enabled = PaperOperationPolicy(paper_execution_enabled=True)
    dry = _run(
        tmp_path / "dry",
        as_of=NOW,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(NOW, "SBER", "YDEX"),
        paper=dry_paper,
        audit=InMemoryOperationAuditRepository(),
        mode=PaperOperationMode.DRY_RUN,
        policy=enabled,
    )
    execute_paper = InMemoryPaperLedgerRepository()
    executed = _run(
        tmp_path / "execute",
        as_of=NOW,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(NOW, "SBER", "YDEX"),
        paper=execute_paper,
        audit=InMemoryOperationAuditRepository(),
        policy=enabled,
    )

    assert dry.paper_execution_status == "SKIPPED_DRY_RUN"
    assert dry_paper.events() == []
    assert executed.status == PaperOperationStatus.SUCCESS
    assert executed.safety.PAPER_ORDERS_FILLED == 1


def test_agent_model_unavailable_is_blocked_without_traceback(tmp_path: Path) -> None:
    run = run_paper_operation(
        operation_as_of=NOW,
        mode=PaperOperationMode.DRY_RUN,
        model=UnconfiguredAgentModel(),
        context_provider=StaticPaperOperationContextProvider(_context(NOW, "SBER", "YDEX")),
        paper_repository=InMemoryPaperLedgerRepository(),
        audit_repository=InMemoryOperationAuditRepository(),
        state_root=tmp_path,
        code_sha="a" * 40,
    )

    assert run.status == PaperOperationStatus.BLOCKED
    assert run.status_code == "AGENT_MODEL_UNAVAILABLE"
    assert run.safety.PAPER_ORDERS_FILLED == 0


def test_concurrent_operation_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "operation.lock"
    with operation_lock(
        lock_path,
        operation_id="first",
        stale_after=timedelta(hours=1),
        now=NOW,
    ):
        with pytest.raises(OperationAlreadyRunningError, match="OPERATION_ALREADY_RUNNING"):
            with operation_lock(
                lock_path,
                operation_id="second",
                stale_after=timedelta(hours=1),
                now=NOW,
            ):
                pass


def test_restart_preserves_operation_state(tmp_path: Path) -> None:
    ledger_path = tmp_path / "portfolio-ledger.jsonl"
    first_repository = JsonlPaperLedgerRepository(ledger_path)
    audit = InMemoryOperationAuditRepository()
    first = _run(
        tmp_path,
        as_of=NOW,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(NOW, "SBER", "YDEX"),
        paper=first_repository,
        audit=audit,
    )

    restarted = JsonlPaperLedgerRepository(ledger_path)

    assert portfolio_state_sha(replay_portfolio(restarted.events())) == first.portfolio_after_sha


def test_restart_preserves_operation_audit_and_idempotency(tmp_path: Path) -> None:
    paper = InMemoryPaperLedgerRepository()
    audit_path = tmp_path / "operation-ledger.jsonl"
    first_audit = JsonlOperationAuditRepository(audit_path)
    model = _model(NOW, _proposal("SBER", "BUY", 0.10))
    first = _run(
        tmp_path,
        as_of=NOW,
        proposals=(),
        context=_context(NOW, "SBER", "YDEX"),
        paper=paper,
        audit=first_audit,
        model=model,
    )
    restarted_audit = JsonlOperationAuditRepository(audit_path)
    duplicate = _run(
        tmp_path,
        as_of=NOW,
        proposals=(),
        context=_context(NOW, "SBER", "YDEX"),
        paper=paper,
        audit=restarted_audit,
        model=model,
    )

    assert restarted_audit.completed_run(first.operation_id) == first
    assert duplicate.status == PaperOperationStatus.ALREADY_PROCESSED
    assert duplicate.safety.PAPER_ORDERS_FILLED == 0


def test_future_portfolio_mark_rejected(tmp_path: Path) -> None:
    paper = InMemoryPaperLedgerRepository()
    _run(
        tmp_path / "future-seed",
        as_of=NOW + timedelta(days=1),
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(NOW + timedelta(days=1), "SBER", "YDEX"),
        paper=paper,
        audit=InMemoryOperationAuditRepository(),
    )
    model = _model(NOW, _proposal("SBER", "HOLD", 0.10))
    run = _run(
        tmp_path / "past-operation",
        as_of=NOW,
        proposals=(),
        context=_context(NOW, "SBER", "YDEX"),
        paper=paper,
        audit=InMemoryOperationAuditRepository(),
        model=model,
    )

    assert run.status == PaperOperationStatus.BLOCKED
    assert "FUTURE_PORTFOLIO_MARK" in run.reasons
    assert model.requests == []


def test_fill_committed_before_audit_recovers_without_duplicate(tmp_path: Path) -> None:
    paper = InMemoryPaperLedgerRepository()
    audit = InMemoryOperationAuditRepository()
    model = _model(NOW, _proposal("SBER", "BUY", 0.10))
    with pytest.raises(SimulatedCrashAfterPaperFillError):
        _run(
            tmp_path,
            as_of=NOW,
            proposals=(),
            context=_context(NOW, "SBER", "YDEX"),
            paper=paper,
            audit=audit,
            model=model,
            simulate_crash_after_fill=True,
        )
    event_count = len(paper.events())

    recovered = _run(
        tmp_path,
        as_of=NOW + timedelta(minutes=3),
        proposals=(),
        context=_context(NOW + timedelta(minutes=3), "SBER", "YDEX"),
        paper=paper,
        audit=audit,
        model=model,
    )

    assert recovered.status == PaperOperationStatus.SUCCESS
    assert recovered.status_code == "RECOVERED_AFTER_COMMITTED_FILL"
    assert recovered.safety.PAPER_ORDERS_FILLED == 0
    assert len(recovered.paper_trade_ids) == 1
    assert len(paper.events()) == event_count


def test_real_execution_disabled_and_no_broker_capability_registered(tmp_path: Path) -> None:
    policy = PaperOperationPolicy()
    context = _context(NOW, "SBER", "YDEX")
    tools = build_read_only_tool_registry(
        AgentDataContext(
            as_of=NOW,
            allowed_universe=context.universe,
            portfolio_snapshot={},
            market_context_snapshot=context.market_context,
            event_context_snapshot=context.event_context,
            research_status_snapshot=context.research_status,
        ),
        AgentRunConfig(output_root=tmp_path / "agent", code_sha="a" * 40),
    )

    assert policy.real_execution_enabled is False
    assert not {"buy", "sell", "place_order", "cancel_order"} & set(tools)
    source = Path("src/paper_trading_operation_v1/application.py").read_text(encoding="utf-8")
    assert "tinvest_market.client" not in source
    assert "place_order" not in source


def test_stale_risk_plan_no_fill_and_expired_plan_no_fill() -> None:
    quote = _quote("SBER", NOW)
    portfolio = initial_paper_portfolio(NOW)
    plan = evaluate_agent_run(
        agent_run={
            "run_id": "operation-risk-proof",
            "AGENT_RESEARCH_CAPABILITY_READY": True,
            "AGENT_DECISION_STATUS": "VALID",
            "validation": {"status": "VALID"},
            "universe": _universe(),
            "research_status_snapshot": {
                "LIVE_RESEARCH_OPERATION_STATUS": "READY",
                "OPERATIONAL_BURN_IN": "PASS",
            },
            "final_proposals": [_proposal("SBER", "BUY", 0.10)],
        },
        portfolio=portfolio,
        market_snapshot=[quote],
        policy=RiskPolicy(),
        decision_as_of=NOW,
    )
    expired = execute_paper_plan(
        plan,
        InMemoryPaperLedgerRepository(),
        execution_as_of=NOW + timedelta(minutes=6),
    )
    changed_repository = InMemoryPaperLedgerRepository()
    execute_paper_plan(plan, changed_repository, execution_as_of=NOW)
    stale_plan = plan.model_copy(
        update={
            "agent_run_id": "other-run",
            "paper_orders": [
                order.model_copy(
                    update={
                        "run_id": "other-run",
                        "idempotency_key": f"other:{order.proposal_id}",
                    }
                )
                for order in plan.paper_orders
            ],
        }
    )
    stale = execute_paper_plan(stale_plan, changed_repository, execution_as_of=NOW)

    assert expired.execution_status == PaperExecutionStatus.PLAN_EXPIRED
    assert expired.safety.PAPER_ORDERS_FILLED == 0
    assert stale.execution_status == PaperExecutionStatus.STALE_RISK_PLAN
    assert stale.safety.PAPER_ORDERS_FILLED == 0


def test_operation_safety_counters_prove_no_outcome_or_real_reads(tmp_path: Path) -> None:
    run = _run(
        tmp_path,
        as_of=NOW,
        proposals=(_proposal("SBER", "BUY", 0.10),),
        context=_context(NOW, "SBER", "YDEX"),
        paper=InMemoryPaperLedgerRepository(),
        audit=InMemoryOperationAuditRepository(),
    )

    assert run.safety.REAL_BROKER_MUTATIONS == 0
    assert run.safety.REAL_ORDERS_SENT == 0
    assert run.safety.REAL_POSITIONS_CHANGED == 0
    assert run.safety.LIVE_OUTCOMES_READ == 0
    assert run.safety.LIVE_TARGETS_COMPUTED == 0
    assert run.safety.LIVE_POST_EVENT_PRICE_READS == 0
    assert run.safety.OLD_FUTURE_HOLDOUT_OPENED is False


def test_deterministic_operation_artifact_covers_acceptance(tmp_path: Path) -> None:
    first_sample = build_sample_operations(
        work_root=tmp_path / "first-work",
        code_sha="b" * 40,
    )
    second_sample = build_sample_operations(
        work_root=tmp_path / "second-work",
        code_sha="b" * 40,
    )
    first_root = tmp_path / "first-artifact"
    second_root = tmp_path / "second-artifact"
    first = write_operation_artifact(
        output_root=first_root,
        base_main_sha="a" * 40,
        head_sha="b" * 40,
        sample=first_sample,
    )
    second = write_operation_artifact(
        output_root=second_root,
        base_main_sha="a" * 40,
        head_sha="b" * 40,
        sample=second_sample,
    )

    assert first["PAPER_TRADING_OPERATION_READY"] == "YES"
    assert first["PAPER_OPERATION_RUNS"] == 4
    assert first["PAPER_ORDERS_FILLED"] == 3
    assert first["paper_ledger_event_count"] == 6
    assert first["DEGRADED_OPERATION_STATUS"] == "DEGRADED"
    assert first["STALE_PLAN_RESULT"] == "STALE_RISK_PLAN"
    assert first["DUPLICATE_OPERATION_RESULT"] == "ALREADY_PROCESSED"
    assert first["CRASH_RECOVERY_RESULT"] == "RECOVERED_AFTER_COMMITTED_FILL"
    assert first["SESSION_IDEMPOTENCY"] == "PASS"
    assert first["PRODUCTION_PROVIDER_SESSION_IDEMPOTENCY"] == "PASS"
    assert first["HELD_POSITION_UNIVERSE_REORDER_RETRY"] == "ALREADY_PROCESSED"
    assert first["TRUNCATED_UNIVERSE_CHANGE_RETRY"] == "ALREADY_PROCESSED"
    assert first["CONTRACT_CHANGE_SAME_SLOT"] == "ALREADY_PROCESSED"
    assert first["SAME_SESSION_DIFFERENT_TIMESTAMP"] == "ALREADY_PROCESSED"
    assert first["DIFFERENT_SESSION_DISTINCT"] == "YES"
    assert first["NEXT_TRADING_DAY_DISTINCT"] == "YES"
    assert first["PAPER_EXECUTION_DEFAULT"] is False
    assert first["PAPER_EXECUTION_DOUBLE_OPT_IN"] == "PASS"
    assert first["REAL_BROKER_MUTATIONS"] == 0
    assert first["REAL_ORDERS_SENT"] == 0
    assert first["ARTIFACT_SHA"] == second["ARTIFACT_SHA"]
    assert {
        "manifest.json",
        "operation-policy.json",
        "day-1-operation.json",
        "day-2-operation.json",
        "day-3-operation.json",
        "degraded-operation.json",
        "operation-history.json",
        "operation-ledger.jsonl",
        "paper-ledger.jsonl",
        "final-portfolio.json",
        "idempotency-verification.json",
        "stale-plan-verification.json",
        "crash-recovery-verification.json",
        "replay-verification.json",
        "safety.json",
        "report.md",
    } == {path.name for path in first_root.iterdir()}
    assert all(
        path.read_bytes() == (second_root / path.name).read_bytes() for path in first_root.iterdir()
    )
