from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.risk_engine_paper_v1.application import (
    close_paper_day,
    evaluate_agent_run,
    execute_paper_plan,
    initial_paper_portfolio,
    portfolio_state_sha,
    replay_portfolio,
    restore_operational_portfolio,
)
from src.risk_engine_paper_v1.domain import (
    LedgerEvent,
    LedgerEventType,
    MarketQuote,
    PaperExecutionResult,
    PaperExecutionStatus,
    PaperSide,
    RiskDecisionType,
    RiskPlan,
    RiskPolicy,
    RiskReasonCode,
)
from src.risk_engine_paper_v1.repository import (
    InMemoryPaperLedgerRepository,
    JsonlPaperLedgerRepository,
    LedgerIntegrityError,
)

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def _proposal(ticker: str, action: str, weight: float) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "action": action,
        "agent_confidence": 0.5,
        "target_weight": weight,
        "holding_horizon": "1-5d",
        "thesis": ["multi-run test"],
        "risks": ["paper only"],
        "evidence": [],
        "data_quality": "GOOD",
    }


def _agent(run_id: str, *proposals: dict[str, Any]) -> dict[str, Any]:
    tickers = sorted({str(row["ticker"]) for row in proposals})
    return {
        "run_id": run_id,
        "AGENT_RESEARCH_CAPABILITY_READY": True,
        "AGENT_DECISION_STATUS": "VALID",
        "validation": {"status": "VALID"},
        "ML_V2_DATASET_STATUS": "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY",
        "universe": [
            {"ticker": ticker, "supported": True, "market_data_compatible": True}
            for ticker in tickers
        ],
        "research_status_snapshot": {
            "LIVE_RESEARCH_OPERATION_STATUS": "READY",
            "OPERATIONAL_BURN_IN": "PASS",
        },
        "final_proposals": list(proposals),
    }


def _quotes(as_of: datetime, *tickers: str) -> list[MarketQuote]:
    prices = {"SBER": 100.0, "YDEX": 110.0}
    return [
        MarketQuote(
            ticker=ticker,
            as_of=as_of,
            last_price=prices[ticker],
            bid=prices[ticker] - 0.2,
            ask=prices[ticker] + 0.2,
            lot_size=10 if ticker == "SBER" else 1,
        )
        for ticker in tickers
    ]


def _plan(
    repository: InMemoryPaperLedgerRepository | JsonlPaperLedgerRepository,
    run_id: str,
    *proposals: dict[str, Any],
    as_of: datetime = NOW,
    policy: RiskPolicy | None = None,
) -> RiskPlan:
    quotes = _quotes(as_of, *(str(row["ticker"]) for row in proposals))
    portfolio = restore_operational_portfolio(
        repository,
        as_of=as_of,
        market_snapshot=quotes,
    )
    return evaluate_agent_run(
        agent_run=_agent(run_id, *proposals),
        portfolio=portfolio,
        market_snapshot=quotes,
        policy=policy or RiskPolicy(),
        decision_as_of=as_of,
        ledger_event_count=repository.last_sequence(),
    )


def _buy_once(
    repository: InMemoryPaperLedgerRepository | JsonlPaperLedgerRepository,
) -> tuple[RiskPlan, PaperExecutionResult]:
    plan = _plan(repository, "run-1", _proposal("SBER", "BUY", 0.10))
    return plan, execute_paper_plan(plan, repository)


def test_second_run_uses_persisted_portfolio() -> None:
    repository = InMemoryPaperLedgerRepository()
    _, first = _buy_once(repository)
    second = _plan(repository, "run-2", _proposal("SBER", "BUY", 0.15))

    assert second.initial_portfolio.cash == first.final_portfolio.cash
    assert second.initial_portfolio.positions[0].quantity > 0
    assert second.initial_portfolio.turnover_today == first.final_portfolio.turnover_today
    assert second.ledger_event_count == repository.last_sequence()


def test_second_buy_uses_delta_not_empty_portfolio() -> None:
    repository = InMemoryPaperLedgerRepository()
    _, first = _buy_once(repository)
    second = _plan(repository, "run-2", _proposal("SBER", "BUY", 0.15))
    empty = _plan(InMemoryPaperLedgerRepository(), "empty", _proposal("SBER", "BUY", 0.15))

    assert second.paper_orders[0].quantity < empty.paper_orders[0].quantity
    result = execute_paper_plan(second, repository)
    assert (
        result.final_portfolio.positions[0].quantity > first.final_portfolio.positions[0].quantity
    )


def test_sell_sees_position_from_previous_run() -> None:
    repository = InMemoryPaperLedgerRepository()
    _, first = _buy_once(repository)
    sell = _plan(repository, "run-2", _proposal("SBER", "SELL", 0.05))
    result = execute_paper_plan(sell, repository)

    assert sell.decisions[0].reason_codes != [RiskReasonCode.NO_EXISTING_POSITION]
    assert result.filled_orders[0].side == PaperSide.SELL
    assert (
        result.final_portfolio.positions[0].quantity < first.final_portfolio.positions[0].quantity
    )
    assert result.final_portfolio.cash > first.final_portfolio.cash
    assert result.final_portfolio.realized_pnl != 0


def test_full_exit_across_runs() -> None:
    repository = InMemoryPaperLedgerRepository()
    _buy_once(repository)
    exit_plan = _plan(repository, "run-2", _proposal("SBER", "SELL", 0.0))
    result = execute_paper_plan(exit_plan, repository)

    assert result.final_portfolio.positions == []
    assert result.final_portfolio.realized_pnl != 0
    assert replay_portfolio(repository.events()) == result.final_portfolio


def test_turnover_accumulates_same_day() -> None:
    repository = InMemoryPaperLedgerRepository()
    _, first = _buy_once(repository)
    second = _plan(repository, "run-2", _proposal("SBER", "BUY", 0.15))
    result = execute_paper_plan(second, repository)

    assert result.final_portfolio.turnover_today > first.final_portfolio.turnover_today


def test_new_day_turnover_reset_is_explicit() -> None:
    repository = InMemoryPaperLedgerRepository()
    _buy_once(repository)
    next_day = NOW + timedelta(days=1)
    transitioned = close_paper_day(repository, next_day_as_of=next_day)

    assert transitioned.turnover_today == 0
    assert transitioned.daily_pnl == 0
    assert transitioned.as_of == next_day
    assert repository.events()[-1].event_type == LedgerEventType.DAY_CLOSED


def test_replay_historical_fills_without_current_quote_dependency() -> None:
    repository = InMemoryPaperLedgerRepository()
    _, executed = _buy_once(repository)

    replayed = replay_portfolio(repository.events())

    assert replayed == executed.final_portfolio


def test_stale_risk_plan_rejected() -> None:
    repository = InMemoryPaperLedgerRepository()
    stale = _plan(repository, "run-a", _proposal("SBER", "BUY", 0.05))
    intervening = _plan(repository, "run-b", _proposal("SBER", "BUY", 0.10))
    execute_paper_plan(intervening, repository)
    before = replay_portfolio(repository.events())

    result = execute_paper_plan(stale, repository)

    assert result.execution_status == PaperExecutionStatus.STALE_RISK_PLAN
    assert result.status_code == "PORTFOLIO_STATE_MISMATCH"
    assert result.safety.PAPER_ORDERS_FILLED == 0
    assert result.safety.PAPER_PORTFOLIO_MUTATIONS == 0
    assert replay_portfolio(repository.events()) == before


def test_portfolio_sha_matches_at_execution() -> None:
    repository = InMemoryPaperLedgerRepository()
    plan = _plan(repository, "run-1", _proposal("SBER", "BUY", 0.10))

    assert plan.portfolio_state_sha == portfolio_state_sha(plan.initial_portfolio)
    assert execute_paper_plan(plan, repository).execution_status == PaperExecutionStatus.SUCCESS


def test_tampered_market_snapshot_sha_rejects_plan() -> None:
    repository = InMemoryPaperLedgerRepository()
    plan = _plan(repository, "run-1", _proposal("SBER", "BUY", 0.10))
    tampered = plan.model_copy(update={"market_snapshot_sha": "0" * 64})

    result = execute_paper_plan(tampered, repository)

    assert result.execution_status == PaperExecutionStatus.STALE_RISK_PLAN
    assert result.status_code == "STALE_RISK_PLAN"
    assert result.safety.PAPER_ORDERS_FILLED == 0


def test_expired_plan_fails_closed() -> None:
    repository = InMemoryPaperLedgerRepository()
    plan = _plan(repository, "run-1", _proposal("SBER", "BUY", 0.10))
    result = execute_paper_plan(plan, repository, execution_as_of=NOW + timedelta(minutes=6))

    assert result.execution_status == PaperExecutionStatus.PLAN_EXPIRED
    assert result.safety.PAPER_ORDERS_FILLED == 0
    assert repository.events() == []


def test_idempotency_survives_repository_restart(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    first_repository = JsonlPaperLedgerRepository(ledger_path)
    plan, first = _buy_once(first_repository)
    restarted = JsonlPaperLedgerRepository(ledger_path)

    duplicate = execute_paper_plan(plan, restarted)

    assert duplicate.duplicate_executions_skipped == len(plan.paper_orders)
    assert duplicate.filled_orders == []
    assert duplicate.final_portfolio == first.final_portfolio


def test_duplicate_portfolio_created_rejected() -> None:
    repository = InMemoryPaperLedgerRepository()
    _buy_once(repository)
    portfolio = replay_portfolio(repository.events())
    duplicate = LedgerEvent(
        sequence=repository.last_sequence() + 1,
        event_id="duplicate-create",
        event_type=LedgerEventType.PORTFOLIO_CREATED,
        portfolio_id=portfolio.portfolio_id,
        occurred_at=NOW,
        payload={"portfolio": portfolio.model_dump(mode="json")},
    )

    with pytest.raises(LedgerIntegrityError, match="DUPLICATE_PORTFOLIO_CREATED"):
        repository.append(duplicate)


def test_corrupted_ledger_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"sequence": 1', encoding="utf-8")

    with pytest.raises(LedgerIntegrityError, match="LEDGER_INTEGRITY_FAILURE"):
        JsonlPaperLedgerRepository(path).events()


def test_ledger_rejects_out_of_order_and_duplicate_event_id() -> None:
    portfolio = initial_paper_portfolio(NOW)
    created = LedgerEvent(
        sequence=1,
        event_id="created",
        event_type=LedgerEventType.PORTFOLIO_CREATED,
        portfolio_id=portfolio.portfolio_id,
        occurred_at=NOW,
        payload={"portfolio": portfolio.model_dump(mode="json")},
    )
    repository = InMemoryPaperLedgerRepository([created])
    invalid = created.model_copy(update={"sequence": 3})
    with pytest.raises(LedgerIntegrityError, match="LEDGER_SEQUENCE_VIOLATION"):
        repository.append(invalid)

    duplicate_id = LedgerEvent(
        sequence=2,
        event_id="created",
        event_type=LedgerEventType.DAY_CLOSED,
        portfolio_id=portfolio.portfolio_id,
        occurred_at=NOW + timedelta(days=1),
        payload={"start_of_day_equity": portfolio.equity},
    )
    with pytest.raises(LedgerIntegrityError, match="DUPLICATE_LEDGER_EVENT_ID"):
        repository.append(duplicate_id)


def test_risk_reducing_sell_exemption() -> None:
    repository = InMemoryPaperLedgerRepository()
    _buy_once(repository)
    policy = RiskPolicy(max_daily_turnover=0.05, max_single_order_notional_pct=0.01)
    plan = _plan(
        repository,
        "run-2",
        _proposal("SBER", "SELL", 0.0),
        policy=policy,
    )

    assert plan.decisions[0].risk_decision == RiskDecisionType.APPROVE
    assert plan.paper_orders[0].quantity <= plan.initial_portfolio.positions[0].quantity


def test_sell_policy_without_exemption() -> None:
    repository = InMemoryPaperLedgerRepository()
    _buy_once(repository)
    policy = RiskPolicy(
        max_daily_turnover=0.05,
        max_single_order_notional_pct=0.01,
        risk_reducing_sell_turnover_exempt=False,
        risk_reducing_sell_order_cap_exempt=False,
    )
    plan = _plan(
        repository,
        "run-2",
        _proposal("SBER", "SELL", 0.0),
        policy=policy,
    )

    assert plan.decisions[0].risk_decision == RiskDecisionType.REJECT
    assert RiskReasonCode.TURNOVER_LIMIT in plan.decisions[0].reason_codes
    assert RiskReasonCode.SINGLE_ORDER_LIMIT in plan.decisions[0].reason_codes
    assert plan.paper_orders == []
