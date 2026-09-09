from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from apps.cli.paper import build_parser as build_paper_parser
from src.risk_engine_paper_v1.application import (
    apply_paper_order,
    evaluate_agent_run,
    execute_paper_plan,
    initial_paper_portfolio,
    mark_to_market,
    market_snapshot_sha,
    replay_portfolio,
)
from src.risk_engine_paper_v1.domain import (
    ExecutionPriceSource,
    MarketQuote,
    PaperOrder,
    PaperOrderStatus,
    PaperPortfolio,
    PaperPosition,
    PaperSide,
    RiskPlan,
    RiskPolicy,
)
from src.risk_engine_paper_v1.reporting import build_sample_execution, write_audit_artifact
from src.risk_engine_paper_v1.repository import (
    InMemoryPaperLedgerRepository,
    JsonlPaperLedgerRepository,
)

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def test_paper_cli_exposes_two_phase_evaluate_and_execute_commands() -> None:
    parser = build_paper_parser()

    evaluate = parser.parse_args(["evaluate-agent-run", "agent-run-1"])
    execute = parser.parse_args(["execute-agent-run", "agent-run-1"])
    portfolio = parser.parse_args(["portfolio"])
    replay = parser.parse_args(["replay"])

    assert evaluate.command == "evaluate-agent-run"
    assert execute.command == "execute-agent-run"
    assert portfolio.command == "portfolio"
    assert replay.command == "replay"
    assert not hasattr(portfolio, "run_id")
    assert not hasattr(replay, "run_id")


def _agent(action: str = "BUY", weight: float = 0.10) -> dict[str, Any]:
    return {
        "run_id": "paper-agent-run",
        "AGENT_RESEARCH_CAPABILITY_READY": True,
        "AGENT_DECISION_STATUS": "VALID",
        "validation": {"status": "VALID"},
        "universe": [{"ticker": "SBER", "supported": True, "market_data_compatible": True}],
        "research_status_snapshot": {
            "LIVE_RESEARCH_OPERATION_STATUS": "READY",
            "OPERATIONAL_BURN_IN": "PASS",
        },
        "final_proposals": [
            {
                "ticker": "SBER",
                "action": action,
                "agent_confidence": 0.5,
                "target_weight": weight,
                "holding_horizon": "1-5d",
                "thesis": ["test"],
                "risks": ["test"],
                "evidence": [],
                "data_quality": "GOOD",
            }
        ],
    }


def _quote(last: float = 100.0) -> MarketQuote:
    return MarketQuote(
        ticker="SBER",
        as_of=NOW,
        last_price=last,
        bid=last - 0.2,
        ask=last + 0.2,
        lot_size=10,
    )


def _positioned_portfolio() -> PaperPortfolio:
    return PaperPortfolio(
        portfolio_id="paper-portfolio-v1",
        cash=900_000.0,
        equity=1_000_000.0,
        positions=[
            PaperPosition(
                ticker="SBER",
                quantity=1_000,
                average_cost=90.0,
                last_price=100.0,
                market_value=100_000.0,
                weight=0.10,
                unrealized_pnl=10_000.0,
                mark_as_of=NOW,
            )
        ],
        peak_equity=1_000_000.0,
        drawdown=0.0,
        turnover_today=0.0,
        start_of_day_equity=1_000_000.0,
        as_of=NOW,
    )


def _plan(
    action: str = "BUY",
    weight: float = 0.10,
    portfolio: PaperPortfolio | None = None,
) -> RiskPlan:
    return evaluate_agent_run(
        agent_run=_agent(action, weight),
        portfolio=portfolio or initial_paper_portfolio(NOW),
        market_snapshot=[_quote()],
        policy=RiskPolicy(),
        decision_as_of=NOW,
    )


def test_buy_updates_cash_position_commission_slippage_and_lots() -> None:
    plan = _plan()
    result = execute_paper_plan(plan, InMemoryPaperLedgerRepository())
    order = result.filled_orders[0]
    position = result.final_portfolio.positions[0]

    assert order.status == PaperOrderStatus.FILLED
    assert order.price_source == ExecutionPriceSource.ASK_PLUS_SLIPPAGE
    assert order.execution_price > order.planned_price
    assert order.quantity % order.lot_size == 0
    assert order.commission > 0
    assert math.isclose(
        result.final_portfolio.cash,
        1_000_000.0 - order.gross_notional - order.commission,
        abs_tol=0.0001,
    )
    assert position.quantity == order.quantity
    assert math.isclose(
        position.average_cost,
        (order.gross_notional + order.commission) / order.quantity,
        abs_tol=0.0001,
    )
    assert order.gross_notional + order.commission <= 100_000.0
    assert result.final_portfolio.cash >= 900_000.0


def test_partial_sell_updates_cash_and_realized_pnl_without_negative_position() -> None:
    plan = _plan("SELL", 0.05, _positioned_portfolio())
    result = execute_paper_plan(plan, InMemoryPaperLedgerRepository())
    order = result.filled_orders[0]
    position = result.final_portfolio.positions[0]

    assert order.side == PaperSide.SELL
    assert order.price_source == ExecutionPriceSource.BID_MINUS_SLIPPAGE
    assert order.execution_price < order.planned_price
    assert position.quantity == 1_000 - order.quantity
    assert position.average_cost == 90.0
    assert math.isclose(
        result.final_portfolio.realized_pnl,
        (order.execution_price - 90.0) * order.quantity - order.commission,
        abs_tol=0.0001,
    )
    assert result.final_portfolio.cash >= 0


def test_sell_cannot_exceed_position() -> None:
    order = PaperOrder(
        paper_order_id="order",
        idempotency_key="run:proposal",
        run_id="run",
        proposal_id="proposal",
        ticker="SBER",
        side=PaperSide.SELL,
        quantity=2_000,
        lot_size=10,
        planned_price=99.8,
        execution_price=99.7002,
        price_source=ExecutionPriceSource.BID_MINUS_SLIPPAGE,
        slippage_bps=10,
        commission_bps=5,
        gross_notional=199_400.4,
        commission=99.7002,
        net_cash_effect=199_300.6998,
        created_at=NOW,
    )

    with pytest.raises(ValueError, match="INSUFFICIENT_PAPER_POSITION"):
        apply_paper_order(_positioned_portfolio(), order, [_quote()])


def test_replay_reproduces_portfolio_and_duplicate_execution_is_idempotent() -> None:
    repository = InMemoryPaperLedgerRepository()
    plan = _plan()
    first = execute_paper_plan(plan, repository)
    second = execute_paper_plan(plan, repository)
    replayed = replay_portfolio(repository.events())

    assert first.replay_verification.replay_matches
    assert second.duplicate_executions_skipped == len(plan.paper_orders)
    assert second.paper_trades == []
    assert second.final_portfolio == first.final_portfolio
    assert replayed == first.final_portfolio


def test_jsonl_repository_persists_append_only_ledger(tmp_path: Path) -> None:
    path = tmp_path / "state" / "ledger.jsonl"
    first_repo = JsonlPaperLedgerRepository(path)
    plan = _plan()
    result = execute_paper_plan(plan, first_repo)
    second_repo = JsonlPaperLedgerRepository(path)

    assert len(second_repo.events()) == result.replay_verification.event_count
    assert second_repo.contains_idempotency_key(plan.paper_orders[0].idempotency_key)


def test_mark_to_market_is_not_a_trade_and_rejects_future_snapshot() -> None:
    portfolio = _positioned_portfolio()
    marked = mark_to_market(portfolio, [_quote(110.0)], NOW)

    assert marked.positions[0].last_price == 110.0
    assert marked.turnover_today == portfolio.turnover_today
    with pytest.raises(ValueError, match="FUTURE_MARKET_SNAPSHOT"):
        mark_to_market(
            portfolio,
            [MarketQuote(ticker="SBER", as_of=NOW.replace(hour=13), last_price=110.0)],
            NOW,
        )


def test_execution_rejects_snapshot_after_decision() -> None:
    plan = _plan()
    future_quote = plan.market_snapshot[0].model_copy(update={"as_of": NOW.replace(hour=13)})
    future_plan = plan.model_copy(
        update={
            "market_snapshot": [future_quote],
            "market_snapshot_sha": market_snapshot_sha([future_quote]),
        }
    )

    with pytest.raises(ValueError, match="FUTURE_MARKET_SNAPSHOT"):
        execute_paper_plan(future_plan, InMemoryPaperLedgerRepository())


def test_no_target_outcome_holdout_or_real_broker_fields_exist() -> None:
    result = execute_paper_plan(_plan(), InMemoryPaperLedgerRepository())
    input_keys = set(RiskPlan.model_fields)

    assert "targets" not in input_keys
    assert "outcomes" not in input_keys
    assert result.safety.LIVE_OUTCOMES_READ == 0
    assert result.safety.LIVE_TARGETS_COMPUTED == 0
    assert result.safety.LIVE_POST_EVENT_PRICE_READS == 0
    assert result.safety.OLD_FUTURE_HOLDOUT_OPENED is False
    assert result.safety.REAL_BROKER_MUTATIONS == 0
    assert result.safety.REAL_ORDERS_SENT == 0


def test_deterministic_sample_covers_all_decisions_and_artifact(tmp_path: Path) -> None:
    agent_path = Path("artifacts/ai-trading-agent-v1/run.json")
    sample = build_sample_execution(agent_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = write_audit_artifact(
        output_root=first,
        code_sha="a" * 40,
        sample=sample,
    )
    second_manifest = write_audit_artifact(
        output_root=second,
        code_sha="a" * 40,
        sample=sample,
    )

    assert first_manifest["APPROVE_COUNT"] == 3
    assert first_manifest["REDUCE_COUNT"] == 1
    assert first_manifest["REJECT_COUNT"] == 1
    assert first_manifest["NO_ACTION_COUNT"] == 1
    assert first_manifest["PAPER_ORDERS_FILLED"] == 4
    assert first_manifest["REAL_EXECUTION_READY"] == "NO"
    assert first_manifest["MULTI_RUN_PAPER_READY"] == "YES"
    assert first_manifest["STALE_PLAN_PROTECTION"] == "YES"
    assert first_manifest["REPLAY_VERIFIED"] == "YES"
    assert first_manifest["IDEMPOTENCY_VERIFIED"] == "YES"
    assert first_manifest["PORTFOLIO_MARK_COMPLETENESS"] == "PASS"
    assert first_manifest["BUY_WITH_MISSING_HELD_MARK"] == "REJECT"
    assert first_manifest["RISK_REDUCING_SELL_WITH_OTHER_STALE_MARK"] == "ALLOWED"
    assert sample.mark_completeness_verification["PAPER_ORDERS_PLANNED"] == 0
    assert first_manifest["ARTIFACT_SHA"] == second_manifest["ARTIFACT_SHA"]
    required = {
        "manifest.json",
        "risk-policy.json",
        "run-1-agent.json",
        "run-1-risk-plan.json",
        "run-1-orders.json",
        "run-2-agent.json",
        "run-2-risk-plan.json",
        "run-2-orders.json",
        "run-3-agent.json",
        "run-3-risk-plan.json",
        "run-3-orders.json",
        "final-portfolio.json",
        "replay-verification.json",
        "multi-run-verification.json",
        "idempotency-verification.json",
        "stale-plan-verification.json",
        "mark-completeness-verification.json",
        "safety.json",
        "report.md",
    }
    assert required.issubset(path.name for path in first.iterdir())
    assert all(left.read_bytes() == (second / left.name).read_bytes() for left in first.iterdir())
    with pytest.raises(FileExistsError, match="immutable risk/paper output"):
        write_audit_artifact(
            output_root=first,
            code_sha="a" * 40,
            sample=sample,
        )
