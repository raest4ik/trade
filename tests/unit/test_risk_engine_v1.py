from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from src.ai_trading_agent_v1.application import (
    AgentRunConfig,
    build_read_only_tool_registry,
    sample_agent_context,
)
from src.risk_engine_paper_v1.application import evaluate_agent_run, initial_paper_portfolio
from src.risk_engine_paper_v1.domain import (
    MarketQuote,
    PaperPortfolio,
    PaperPosition,
    RiskDecisionType,
    RiskPlan,
    RiskPolicy,
    RiskReasonCode,
)

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def _proposal(ticker: str = "SBER", action: str = "BUY", weight: float = 0.10) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "action": action,
        "agent_confidence": 0.99,
        "target_weight": weight,
        "holding_horizon": "1-5d",
        "thesis": ["test"],
        "risks": ["test"],
        "evidence": [],
        "data_quality": "GOOD",
    }


def _agent(*proposals: dict[str, Any], ready: bool = True) -> dict[str, Any]:
    tickers = {str(row["ticker"]) for row in proposals} | {"SBER"}
    return {
        "run_id": "agent-run-1",
        "AGENT_RESEARCH_CAPABILITY_READY": ready,
        "AGENT_DECISION_STATUS": "VALID" if ready else "DEGRADED_STALE_DATA",
        "validation": {"status": "VALID" if ready else "DEGRADED"},
        "ML_V2_DATASET_STATUS": "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY",
        "universe": [
            {"ticker": ticker, "supported": True, "market_data_compatible": True}
            for ticker in sorted(tickers)
        ],
        "research_status_snapshot": {
            "LIVE_RESEARCH_OPERATION_STATUS": "READY",
            "OPERATIONAL_BURN_IN": "PASS",
        },
        "final_proposals": list(proposals),
    }


def _quote(
    ticker: str = "SBER",
    *,
    as_of: datetime = NOW,
    last: float | None = 100.0,
    lot: int | None = 10,
) -> MarketQuote:
    return MarketQuote(
        ticker=ticker,
        as_of=as_of,
        last_price=last,
        bid=None if last is None else last - 0.2,
        ask=None if last is None else last + 0.2,
        lot_size=lot,
    )


def _portfolio(
    *,
    cash: float = 1_000_000.0,
    quantity: int = 0,
    daily_pnl: float = 0.0,
    drawdown: float = 0.0,
    turnover: float = 0.0,
) -> PaperPortfolio:
    position = []
    market_value = quantity * 100.0
    equity = cash + market_value
    if quantity:
        position = [
            PaperPosition(
                ticker="SBER",
                quantity=quantity,
                average_cost=90.0,
                last_price=100.0,
                market_value=market_value,
                weight=market_value / equity,
                unrealized_pnl=quantity * 10.0,
                mark_as_of=NOW,
            )
        ]
    return PaperPortfolio(
        portfolio_id="paper-test",
        cash=cash,
        equity=equity,
        positions=position,
        daily_pnl=daily_pnl,
        peak_equity=max(1_000_000.0, equity),
        drawdown=drawdown,
        turnover_today=turnover,
        start_of_day_equity=1_000_000.0,
        as_of=NOW,
    )


def _evaluate(
    proposal: dict[str, Any],
    *,
    portfolio: PaperPortfolio | None = None,
    quote: MarketQuote | None = None,
    market_snapshot: list[MarketQuote] | None = None,
    policy: RiskPolicy | None = None,
    agent: dict[str, Any] | None = None,
) -> RiskPlan:
    source = agent or _agent(proposal)
    return evaluate_agent_run(
        agent_run=source,
        portfolio=portfolio or initial_paper_portfolio(NOW),
        market_snapshot=market_snapshot or [quote or _quote(str(proposal["ticker"]))],
        policy=policy or RiskPolicy(),
        decision_as_of=NOW,
    )


def test_valid_proposal_is_approved_and_confidence_does_not_size() -> None:
    low = _proposal(weight=0.10)
    low["agent_confidence"] = 0.01
    high = _proposal(weight=0.10)
    high["agent_confidence"] = 0.99

    low_plan = _evaluate(low)
    high_plan = _evaluate(high)

    assert low_plan.decisions[0].risk_decision == RiskDecisionType.APPROVE
    assert low_plan.paper_orders[0].quantity == high_plan.paper_orders[0].quantity
    assert low_plan.portfolio_id == low_plan.initial_portfolio.portfolio_id
    assert low_plan.portfolio_as_of == low_plan.initial_portfolio.as_of
    assert low_plan.ledger_event_count == 0
    assert low_plan.portfolio_state_sha
    assert low_plan.market_snapshot_sha


def test_position_above_policy_limit_is_reduced() -> None:
    plan = _evaluate(
        _proposal(weight=0.20),
        policy=RiskPolicy(max_single_order_notional_pct=0.20),
    )

    assert plan.decisions[0].risk_decision == RiskDecisionType.REDUCE
    assert RiskReasonCode.POSITION_LIMIT in plan.decisions[0].reason_codes
    assert plan.decisions[0].approved_target_weight <= 0.15


@pytest.mark.parametrize(
    ("policy", "reason"),
    [
        (RiskPolicy(max_gross_exposure=0.05), RiskReasonCode.PORTFOLIO_GROSS_EXPOSURE_LIMIT),
        (RiskPolicy(max_net_exposure=0.05), RiskReasonCode.PORTFOLIO_NET_EXPOSURE_LIMIT),
        (RiskPolicy(min_cash_buffer_pct=0.95), RiskReasonCode.CASH_LIMIT),
        (RiskPolicy(max_daily_turnover=0.05), RiskReasonCode.TURNOVER_LIMIT),
    ],
)
def test_portfolio_limits_reduce_buy(policy: RiskPolicy, reason: RiskReasonCode) -> None:
    plan = _evaluate(_proposal(weight=0.10), policy=policy)

    assert plan.decisions[0].risk_decision == RiskDecisionType.REDUCE
    assert reason in plan.decisions[0].reason_codes


def test_insufficient_cash_below_minimum_becomes_no_action() -> None:
    plan = _evaluate(
        _proposal(weight=0.10),
        portfolio=_portfolio(cash=500.0),
    )

    assert plan.decisions[0].risk_decision == RiskDecisionType.NO_ACTION
    assert plan.decisions[0].reason_codes == [RiskReasonCode.MIN_TRADE_NOTIONAL]


def test_cash_policy_can_reject_instead_of_reduce() -> None:
    plan = _evaluate(
        _proposal(ticker="GAZP", weight=0.10),
        portfolio=_portfolio(cash=50_000.0, quantity=9_500),
        market_snapshot=[_quote("GAZP"), _quote("SBER")],
        policy=RiskPolicy(reduce_to_available_cash=False),
    )

    assert plan.decisions[0].risk_decision == RiskDecisionType.REJECT
    assert plan.decisions[0].reason_codes == [RiskReasonCode.CASH_LIMIT]


@pytest.mark.parametrize(
    ("portfolio", "reason"),
    [
        (_portfolio(daily_pnl=-20_000.0), RiskReasonCode.DAILY_LOSS_LIMIT),
        (_portfolio(drawdown=0.10), RiskReasonCode.DRAWDOWN_LIMIT),
    ],
)
def test_defensive_gates_block_buy(portfolio: PaperPortfolio, reason: RiskReasonCode) -> None:
    plan = _evaluate(_proposal(), portfolio=portfolio)

    assert plan.decisions[0].risk_decision == RiskDecisionType.REJECT
    assert plan.decisions[0].reason_codes == [reason]


def test_sell_is_allowed_in_drawdown_and_does_not_liquidate_implicitly() -> None:
    plan = _evaluate(
        _proposal(action="SELL", weight=0.05),
        portfolio=_portfolio(cash=800_000.0, quantity=2_000, drawdown=0.20),
    )

    assert plan.decisions[0].risk_decision == RiskDecisionType.APPROVE
    assert plan.paper_orders[0].quantity <= 2_000


def test_kill_switch_blocks_buy_and_sell_but_hold_is_no_action() -> None:
    policy = RiskPolicy(kill_switch_enabled=True)
    buy = _evaluate(_proposal(), policy=policy)
    sell = _evaluate(
        _proposal(action="SELL", weight=0.0),
        policy=policy,
        portfolio=_portfolio(cash=900_000.0, quantity=1_000),
    )
    hold = _evaluate(_proposal(action="HOLD"), policy=policy)

    assert buy.decisions[0].reason_codes == [RiskReasonCode.KILL_SWITCH]
    assert sell.decisions[0].reason_codes == [RiskReasonCode.KILL_SWITCH]
    assert hold.decisions[0].risk_decision == RiskDecisionType.NO_ACTION


@pytest.mark.parametrize(
    ("quote", "reason"),
    [
        (_quote(as_of=NOW - timedelta(hours=1)), RiskReasonCode.STALE_DATA),
        (_quote(as_of=NOW + timedelta(seconds=1)), RiskReasonCode.FUTURE_MARKET_SNAPSHOT),
        (_quote(last=None), RiskReasonCode.PRICE_UNAVAILABLE),
        (_quote(last=0.0), RiskReasonCode.INVALID_PRICE),
        (_quote(lot=None), RiskReasonCode.LOT_SIZE_UNAVAILABLE),
    ],
)
def test_market_snapshot_fail_closed(quote: MarketQuote, reason: RiskReasonCode) -> None:
    plan = _evaluate(_proposal(), quote=quote)

    assert plan.decisions[0].risk_decision == RiskDecisionType.REJECT
    assert plan.decisions[0].reason_codes == [reason]


def test_stale_agent_and_degraded_research_fail_closed() -> None:
    proposal = _proposal()
    stale_agent = _evaluate(proposal, agent=_agent(proposal, ready=False))
    degraded = _agent(proposal)
    degraded["research_status_snapshot"]["LIVE_RESEARCH_OPERATION_STATUS"] = "DEGRADED"
    degraded_plan = _evaluate(proposal, agent=degraded)

    assert stale_agent.decisions[0].reason_codes == [RiskReasonCode.INVALID_PROPOSAL]
    assert degraded_plan.decisions[0].reason_codes == [RiskReasonCode.RESEARCH_HEALTH_DEGRADED]


def test_unsupported_and_conflicting_proposals_are_rejected() -> None:
    unsupported = _agent(_proposal("SBER"))
    unsupported["universe"] = []
    unsupported_plan = _evaluate(_proposal("SBER"), agent=unsupported)
    conflict_agent = _agent(_proposal(), _proposal(action="SELL", weight=0.0))
    conflict_plan = _evaluate(_proposal(), agent=conflict_agent)

    assert unsupported_plan.decisions[0].reason_codes == [RiskReasonCode.UNSUPPORTED_INSTRUMENT]
    assert not conflict_plan.paper_orders
    assert all(
        row.reason_codes == [RiskReasonCode.DUPLICATE_OR_CONFLICTING_PROPOSAL]
        for row in conflict_plan.decisions
    )


def test_malformed_agent_run_cannot_execute() -> None:
    malformed = _agent(_proposal())
    malformed["final_proposals"] = [{"ticker": "SBER", "action": "BUY"}]
    plan = evaluate_agent_run(
        agent_run=malformed,
        portfolio=initial_paper_portfolio(NOW),
        market_snapshot=[_quote()],
        policy=RiskPolicy(),
        decision_as_of=NOW,
    )

    assert plan.decisions == []
    assert plan.paper_orders == []


def test_short_margin_and_leverage_are_disabled_and_agent_tools_unchanged(
    tmp_path: Path,
) -> None:
    policy = RiskPolicy()
    tools = build_read_only_tool_registry(
        sample_agent_context(NOW),
        AgentRunConfig(output_root=tmp_path / "agent", code_sha="test"),
    )

    assert not policy.short_selling_enabled
    assert not policy.margin_enabled
    assert not policy.leverage_enabled
    assert policy.risk_reducing_sell_turnover_exempt
    assert policy.risk_reducing_sell_order_cap_exempt
    assert policy.trading_day_timezone == "Europe/Moscow"
    assert "paper_buy" not in tools
    assert "paper_sell" not in tools
    assert "execute_paper_plan" not in tools
