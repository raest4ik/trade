from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from src.ai_trading_agent_v1.domain import TradeAction, TradeProposal
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.risk_engine_paper_v1.domain import (
    ExecutionPriceSource,
    LedgerEvent,
    LedgerEventType,
    MarketQuote,
    PaperExecutionResult,
    PaperExecutionStatus,
    PaperOrder,
    PaperOrderStatus,
    PaperPortfolio,
    PaperPosition,
    PaperSide,
    PaperTrade,
    PipelineSafety,
    PortfolioMarkStatus,
    PositionMarkStatus,
    ReplayVerification,
    RiskDecision,
    RiskDecisionType,
    RiskPlan,
    RiskPolicy,
    RiskReasonCode,
)
from src.risk_engine_paper_v1.repository import PaperLedgerRepository, validate_ledger_events

ARTIFACT_VERSION = "risk-engine-paper-portfolio-v1"
DEFAULT_ARTIFACT_ROOT = Path(f"artifacts/{ARTIFACT_VERSION}")
DEFAULT_LEDGER_PATH = Path("state/paper-portfolio-v1/ledger.jsonl")
MOEX_TIMEZONE = ZoneInfo("Europe/Moscow")


def evaluate_agent_run(
    *,
    agent_run: dict[str, Any],
    portfolio: PaperPortfolio,
    market_snapshot: Sequence[MarketQuote],
    policy: RiskPolicy,
    decision_as_of: datetime,
    ledger_event_count: int = 0,
) -> RiskPlan:
    if decision_as_of.tzinfo is None or decision_as_of.utcoffset() is None:
        raise ValueError("DECISION_TIME_MUST_BE_TIMEZONE_AWARE")
    agent_run_id = str(agent_run.get("run_id", ""))
    proposals = _proposals(agent_run)
    agent_run_sha = sha256_payload(agent_run)
    input_portfolio_sha = portfolio_state_sha(portfolio)
    snapshot_sha = market_snapshot_sha(market_snapshot)
    plan_id = _id(
        "risk-plan",
        agent_run_id,
        agent_run_sha,
        policy.policy_version,
        input_portfolio_sha,
        snapshot_sha,
    )
    quotes = {quote.ticker.upper(): quote for quote in market_snapshot}
    universe = {
        str(row.get("ticker", "")).upper()
        for row in cast("list[dict[str, Any]]", agent_run.get("universe", []))
        if row.get("supported") is True and row.get("market_data_compatible") is True
    }
    agent_valid = (
        agent_run.get("AGENT_RESEARCH_CAPABILITY_READY") is True
        and agent_run.get("AGENT_DECISION_STATUS") == "VALID"
        and cast("dict[str, Any]", agent_run.get("validation", {})).get("status") == "VALID"
    )
    research = cast("dict[str, Any]", agent_run.get("research_status_snapshot", {}))
    research_ready = (
        research.get("LIVE_RESEARCH_OPERATION_STATUS") == "READY"
        and research.get("OPERATIONAL_BURN_IN") == "PASS"
    )
    ticker_counts = Counter(proposal.ticker.upper() for proposal in proposals)
    has_conflict = any(count > 1 for count in ticker_counts.values())
    portfolio_mark_status, position_mark_statuses = portfolio_mark_health(
        portfolio,
        market_snapshot,
        decision_as_of=decision_as_of,
        max_stale_market_age=policy.max_stale_market_age,
    )

    projected_values = {
        position.ticker.upper(): position.quantity * position.last_price
        for position in portfolio.positions
    }
    projected_cash = portfolio.cash
    projected_turnover = portfolio.turnover_today * portfolio.start_of_day_equity
    decisions: list[RiskDecision] = []
    orders: list[PaperOrder] = []

    for index, proposal in enumerate(proposals):
        proposal_id = _id("proposal", agent_run_id, str(index), proposal.model_dump_json())
        ticker = proposal.ticker.upper()
        current_value = projected_values.get(ticker, 0.0)
        current_weight = current_value / portfolio.equity
        quote = quotes.get(ticker)
        rejection = _pre_trade_rejection(
            proposal=proposal,
            quote=quote,
            policy=policy,
            decision_as_of=decision_as_of,
            universe=universe,
            agent_valid=agent_valid,
            research_ready=research_ready,
            has_conflict=has_conflict,
            portfolio=portfolio,
            portfolio_marks_complete=portfolio_mark_status == PortfolioMarkStatus.COMPLETE,
        )
        if rejection is not None:
            decisions.append(
                _decision(
                    proposal_id,
                    agent_run_id,
                    proposal,
                    RiskDecisionType.REJECT,
                    current_weight,
                    [rejection],
                    decision_as_of,
                )
            )
            continue
        if proposal.action in {TradeAction.HOLD, TradeAction.AVOID}:
            decisions.append(
                _decision(
                    proposal_id,
                    agent_run_id,
                    proposal,
                    RiskDecisionType.NO_ACTION,
                    current_weight,
                    [RiskReasonCode.OK],
                    decision_as_of,
                )
            )
            continue
        assert quote is not None and quote.last_price is not None and quote.lot_size is not None
        if proposal.action == TradeAction.SELL and current_value <= 0:
            decisions.append(
                _decision(
                    proposal_id,
                    agent_run_id,
                    proposal,
                    RiskDecisionType.REJECT,
                    0.0,
                    [RiskReasonCode.NO_EXISTING_POSITION],
                    decision_as_of,
                )
            )
            continue

        desired_value = proposal.target_weight * portfolio.equity
        delta = desired_value - current_value
        if (
            proposal.action == TradeAction.BUY
            and delta <= policy.target_weight_tolerance * portfolio.equity
        ):
            decisions.append(
                _decision(
                    proposal_id,
                    agent_run_id,
                    proposal,
                    RiskDecisionType.NO_ACTION,
                    current_weight,
                    [RiskReasonCode.TARGET_WEIGHT_ALREADY_SATISFIED],
                    decision_as_of,
                )
            )
            continue
        if (
            proposal.action == TradeAction.SELL
            and delta >= -policy.target_weight_tolerance * portfolio.equity
        ):
            decisions.append(
                _decision(
                    proposal_id,
                    agent_run_id,
                    proposal,
                    RiskDecisionType.NO_ACTION,
                    current_weight,
                    [RiskReasonCode.TARGET_WEIGHT_ALREADY_SATISFIED],
                    decision_as_of,
                )
            )
            continue

        side = PaperSide.BUY if proposal.action == TradeAction.BUY else PaperSide.SELL
        reasons: list[RiskReasonCode] = []
        if side == PaperSide.BUY:
            desired_value, cap_reasons = _cap_buy_value(
                desired_value=desired_value,
                current_value=current_value,
                projected_values=projected_values,
                projected_cash=projected_cash,
                projected_turnover=projected_turnover,
                portfolio=portfolio,
                policy=policy,
            )
            reasons.extend(cap_reasons)
            delta = desired_value - current_value
            if RiskReasonCode.CASH_LIMIT in reasons and not policy.reduce_to_available_cash:
                decisions.append(
                    _decision(
                        proposal_id,
                        agent_run_id,
                        proposal,
                        RiskDecisionType.REJECT,
                        current_weight,
                        [RiskReasonCode.CASH_LIMIT],
                        decision_as_of,
                    )
                )
                continue
        else:
            desired_value = max(0.0, desired_value)
            desired_value, sell_reasons = _cap_sell_value(
                desired_value=desired_value,
                current_value=current_value,
                projected_turnover=projected_turnover,
                portfolio=portfolio,
                policy=policy,
            )
            reasons.extend(sell_reasons)
            delta = desired_value - current_value
            if delta >= 0:
                decisions.append(
                    _decision(
                        proposal_id,
                        agent_run_id,
                        proposal,
                        RiskDecisionType.REJECT,
                        current_weight,
                        reasons or [RiskReasonCode.INVALID_PROPOSAL],
                        decision_as_of,
                    )
                )
                continue

        order = _plan_order(
            agent_run_id=agent_run_id,
            proposal_id=proposal_id,
            ticker=ticker,
            side=side,
            delta_value=delta,
            quote=quote,
            policy=policy,
            decision_as_of=decision_as_of,
        )
        if order is None:
            reason = (
                RiskReasonCode.MIN_TRADE_NOTIONAL
                if abs(delta) > 0
                else RiskReasonCode.TARGET_WEIGHT_ALREADY_SATISFIED
            )
            decisions.append(
                _decision(
                    proposal_id,
                    agent_run_id,
                    proposal,
                    RiskDecisionType.NO_ACTION,
                    current_weight,
                    [reason],
                    decision_as_of,
                )
            )
            continue
        if side == PaperSide.SELL:
            current_quantity = _position_quantity(portfolio, ticker)
            if order.quantity > current_quantity:
                order = _replace_order_quantity(order, current_quantity, policy)
            if order.quantity <= 0:
                decisions.append(
                    _decision(
                        proposal_id,
                        agent_run_id,
                        proposal,
                        RiskDecisionType.REJECT,
                        current_weight,
                        [RiskReasonCode.NO_EXISTING_POSITION],
                        decision_as_of,
                    )
                )
                continue

        approved_value = (
            current_value + order.quantity * quote.last_price
            if side == PaperSide.BUY
            else current_value - order.quantity * quote.last_price
        )
        approved_weight = max(0.0, approved_value / portfolio.equity)
        reduced = (
            approved_weight + policy.target_weight_tolerance < proposal.target_weight
            if side == PaperSide.BUY
            else approved_weight - policy.target_weight_tolerance > proposal.target_weight
        )
        decision_type = RiskDecisionType.REDUCE if reduced else RiskDecisionType.APPROVE
        if not reasons:
            reasons.append(RiskReasonCode.OK)
        decisions.append(
            _decision(
                proposal_id,
                agent_run_id,
                proposal,
                decision_type,
                approved_weight,
                reasons,
                decision_as_of,
            )
        )
        orders.append(order)
        projected_values[ticker] = approved_value
        projected_cash += order.net_cash_effect
        projected_turnover += order.gross_notional

    return RiskPlan(
        plan_id=plan_id,
        agent_run_id=agent_run_id,
        agent_run_sha=agent_run_sha,
        policy_version=policy.policy_version,
        portfolio_id=portfolio.portfolio_id,
        portfolio_state_sha=input_portfolio_sha,
        portfolio_as_of=portfolio.as_of,
        ledger_event_count=ledger_event_count,
        market_snapshot_sha=snapshot_sha,
        portfolio_mark_status=portfolio_mark_status,
        position_mark_statuses=position_mark_statuses,
        max_stale_market_age=policy.max_stale_market_age,
        max_risk_plan_age=policy.max_risk_plan_age,
        decision_as_of=decision_as_of,
        initial_portfolio=portfolio,
        market_snapshot=list(market_snapshot),
        decisions=decisions,
        paper_orders=orders,
    )


def execute_paper_plan(
    plan: RiskPlan,
    repository: PaperLedgerRepository,
    *,
    execution_as_of: datetime | None = None,
) -> PaperExecutionResult:
    execution_time = execution_as_of or plan.decision_as_of
    if execution_time.tzinfo is None or execution_time.utcoffset() is None:
        raise ValueError("EXECUTION_TIME_MUST_BE_TIMEZONE_AWARE")
    events = repository.events()
    plan_contract_invalid = (
        plan.portfolio_id != plan.initial_portfolio.portfolio_id
        or plan.portfolio_as_of != plan.initial_portfolio.as_of
        or plan.portfolio_state_sha != portfolio_state_sha(plan.initial_portfolio)
        or plan.market_snapshot_sha != market_snapshot_sha(plan.market_snapshot)
    )
    if plan_contract_invalid:
        current = replay_portfolio(events) if events else plan.initial_portfolio
        return _execution_result(
            plan=plan,
            portfolio=current,
            repository=repository,
            execution_status=PaperExecutionStatus.STALE_RISK_PLAN,
            status_code="STALE_RISK_PLAN",
        )
    if (
        events
        and plan.paper_orders
        and all(
            repository.contains_idempotency_key(order.idempotency_key)
            for order in plan.paper_orders
        )
    ):
        current = replay_portfolio(events)
        return _execution_result(
            plan=plan,
            portfolio=current,
            repository=repository,
            duplicate_skips=len(plan.paper_orders),
        )

    if len(events) != plan.ledger_event_count:
        current = replay_portfolio(events) if events else plan.initial_portfolio
        return _execution_result(
            plan=plan,
            portfolio=current,
            repository=repository,
            execution_status=PaperExecutionStatus.STALE_RISK_PLAN,
            status_code="PORTFOLIO_STATE_MISMATCH",
        )

    current = (
        mark_to_market(
            replay_portfolio(events),
            plan.market_snapshot,
            plan.decision_as_of,
            max_stale_market_age=plan.max_stale_market_age,
            allow_incomplete=True,
        )
        if events
        else plan.initial_portfolio
    )
    if execution_time - plan.decision_as_of > plan.max_risk_plan_age:
        return _execution_result(
            plan=plan,
            portfolio=current,
            repository=repository,
            execution_status=PaperExecutionStatus.PLAN_EXPIRED,
            status_code="PLAN_EXPIRED",
        )
    state_mismatch = (
        current.portfolio_id != plan.portfolio_id
        or portfolio_state_sha(current) != plan.portfolio_state_sha
    )
    if state_mismatch:
        return _execution_result(
            plan=plan,
            portfolio=current,
            repository=repository,
            execution_status=PaperExecutionStatus.STALE_RISK_PLAN,
            status_code="PORTFOLIO_STATE_MISMATCH",
        )

    if not events:
        repository.append(
            LedgerEvent(
                sequence=1,
                event_id=_id("ledger", plan.plan_id, "portfolio-created"),
                event_type=LedgerEventType.PORTFOLIO_CREATED,
                portfolio_id=plan.initial_portfolio.portfolio_id,
                occurred_at=plan.decision_as_of,
                payload={"portfolio": current.model_dump(mode="json")},
            )
        )
    portfolio = replay_portfolio(repository.events())
    filled_orders: list[PaperOrder] = []
    trades: list[PaperTrade] = []
    duplicate_skips = 0
    for order in plan.paper_orders:
        if repository.contains_idempotency_key(order.idempotency_key):
            duplicate_skips += 1
            continue
        portfolio, filled, trade = apply_paper_order(portfolio, order, plan.market_snapshot)
        repository.append(
            LedgerEvent(
                sequence=len(repository.events()) + 1,
                event_id=_id("ledger", filled.paper_order_id, "filled"),
                event_type=LedgerEventType.PAPER_ORDER_FILLED,
                portfolio_id=portfolio.portfolio_id,
                idempotency_key=filled.idempotency_key,
                occurred_at=trade.executed_at,
                payload={
                    "order": filled.model_dump(mode="json"),
                    "trade": trade.model_dump(mode="json"),
                    "position_mark_price": _quote(plan.market_snapshot, filled.ticker).last_price,
                    "position_mark_as_of": _quote(
                        plan.market_snapshot, filled.ticker
                    ).as_of.isoformat(),
                },
            )
        )
        filled_orders.append(filled)
        trades.append(trade)

    replayed = mark_to_market(
        replay_portfolio(repository.events()),
        plan.market_snapshot,
        plan.decision_as_of,
        max_stale_market_age=plan.max_stale_market_age,
        allow_incomplete=True,
    )
    expected = mark_to_market(
        portfolio,
        plan.market_snapshot,
        plan.decision_as_of,
        max_stale_market_age=plan.max_stale_market_age,
        allow_incomplete=True,
    )
    expected_sha = portfolio_state_sha(expected)
    replayed_sha = portfolio_state_sha(replayed)
    verification = ReplayVerification(
        replay_matches=expected_sha == replayed_sha,
        expected_sha=expected_sha,
        replayed_sha=replayed_sha,
        event_count=len(repository.events()),
    )
    safety = PipelineSafety(
        PAPER_ORDERS_PLANNED=len(plan.paper_orders),
        PAPER_ORDERS_FILLED=len(filled_orders),
        PAPER_PORTFOLIO_MUTATIONS=len(filled_orders),
    )
    replay_ok = verification.replay_matches
    return PaperExecutionResult(
        plan=plan,
        filled_orders=filled_orders,
        paper_trades=trades,
        final_portfolio=replayed,
        replay_verification=verification,
        duplicate_executions_skipped=duplicate_skips,
        safety=safety,
        execution_status=(
            PaperExecutionStatus.SUCCESS
            if replay_ok
            else PaperExecutionStatus.LEDGER_INTEGRITY_FAILURE
        ),
        status_code="OK" if replay_ok else "REPLAY_VERIFICATION_FAILED",
    )


def _execution_result(
    *,
    plan: RiskPlan,
    portfolio: PaperPortfolio,
    repository: PaperLedgerRepository,
    duplicate_skips: int = 0,
    execution_status: PaperExecutionStatus = PaperExecutionStatus.SUCCESS,
    status_code: str = "OK",
) -> PaperExecutionResult:
    state_sha = portfolio_state_sha(portfolio)
    return PaperExecutionResult(
        plan=plan,
        filled_orders=[],
        paper_trades=[],
        final_portfolio=portfolio,
        replay_verification=ReplayVerification(
            replay_matches=True,
            expected_sha=state_sha,
            replayed_sha=state_sha,
            event_count=repository.last_sequence(),
        ),
        duplicate_executions_skipped=duplicate_skips,
        safety=PipelineSafety(
            PAPER_ORDERS_PLANNED=len(plan.paper_orders),
            PAPER_ORDERS_FILLED=0,
            PAPER_PORTFOLIO_MUTATIONS=0,
        ),
        execution_status=execution_status,
        status_code=status_code,
    )


def apply_paper_order(
    portfolio: PaperPortfolio,
    order: PaperOrder,
    market_snapshot: Sequence[MarketQuote],
) -> tuple[PaperPortfolio, PaperOrder, PaperTrade]:
    if order.status != PaperOrderStatus.PLANNED:
        raise ValueError("ORDER_NOT_PLANNED")
    quote = _quote(market_snapshot, order.ticker)
    if quote.as_of > order.created_at:
        raise ValueError("FUTURE_MARKET_SNAPSHOT")
    positions = {position.ticker: position for position in portfolio.positions}
    existing = positions.get(order.ticker)
    current_quantity = 0 if existing is None else existing.quantity
    if order.side == PaperSide.SELL and order.quantity > current_quantity:
        raise ValueError("INSUFFICIENT_PAPER_POSITION")
    if order.side == PaperSide.BUY and portfolio.cash + order.net_cash_effect < -0.0001:
        raise ValueError("NEGATIVE_PAPER_CASH")

    executed_at = order.created_at
    trade = PaperTrade(
        paper_trade_id=_id("paper-trade", order.paper_order_id),
        paper_order_id=order.paper_order_id,
        ticker=order.ticker,
        side=order.side,
        quantity=order.quantity,
        fill_price=order.execution_price,
        gross_value=order.gross_notional,
        commission=order.commission,
        cash_delta=order.net_cash_effect,
        executed_at=executed_at,
    )
    updated = _apply_order_economics(
        portfolio,
        order,
        mark_price=quote.last_price or order.execution_price,
        mark_as_of=quote.as_of,
        as_of=executed_at,
    )
    filled = order.model_copy(
        update={"status": PaperOrderStatus.FILLED, "executed_at": executed_at}
    )
    return updated, filled, trade


def _apply_order_economics(
    portfolio: PaperPortfolio,
    order: PaperOrder,
    *,
    mark_price: float,
    mark_as_of: datetime,
    as_of: datetime,
) -> PaperPortfolio:
    positions = {position.ticker: position for position in portfolio.positions}
    existing = positions.get(order.ticker)
    current_quantity = 0 if existing is None else existing.quantity
    if order.side == PaperSide.SELL and order.quantity > current_quantity:
        raise ValueError("INSUFFICIENT_PAPER_POSITION")
    if order.side == PaperSide.BUY and portfolio.cash + order.net_cash_effect < -0.0001:
        raise ValueError("NEGATIVE_PAPER_CASH")
    new_cash = _money(portfolio.cash + order.net_cash_effect)
    new_realized = portfolio.realized_pnl
    if order.side == PaperSide.BUY:
        old_quantity = current_quantity
        old_cost = 0.0 if existing is None else existing.average_cost * old_quantity
        new_quantity = old_quantity + order.quantity
        average_cost = (old_cost + order.gross_notional + order.commission) / new_quantity
    else:
        assert existing is not None
        new_quantity = current_quantity - order.quantity
        average_cost = existing.average_cost if new_quantity else 0.0
        new_realized += (
            order.execution_price - existing.average_cost
        ) * order.quantity - order.commission
    if new_quantity:
        positions[order.ticker] = _position(
            order.ticker,
            new_quantity,
            average_cost,
            mark_price,
            0.0,
            mark_as_of,
        )
    else:
        positions.pop(order.ticker, None)
    updated = _portfolio_from_positions(
        portfolio=portfolio,
        cash=new_cash,
        positions=list(positions.values()),
        realized_pnl=new_realized,
        turnover_today=portfolio.turnover_today
        + order.gross_notional / portfolio.start_of_day_equity,
        as_of=as_of,
    )
    return updated


def replay_portfolio(events: Sequence[LedgerEvent]) -> PaperPortfolio:
    validate_ledger_events(list(events))
    if not events:
        raise ValueError("EMPTY_PAPER_LEDGER")
    created = events[0]
    if created.event_type != LedgerEventType.PORTFOLIO_CREATED:
        raise ValueError("MISSING_PORTFOLIO_CREATED")
    portfolio = PaperPortfolio.model_validate(created.payload["portfolio"])
    for event in events:
        if event.portfolio_id != portfolio.portfolio_id:
            raise ValueError("LEDGER_PORTFOLIO_ID_MISMATCH")
        if event.event_type == LedgerEventType.PORTFOLIO_CREATED:
            continue
        if event.event_type == LedgerEventType.PAPER_ORDER_FILLED:
            order = PaperOrder.model_validate(event.payload["order"])
            mark_price = float(
                cast("int | float", event.payload.get("position_mark_price", order.execution_price))
            )
            mark_as_of_raw = event.payload.get("position_mark_as_of")
            mark_as_of = (
                datetime.fromisoformat(str(mark_as_of_raw).replace("Z", "+00:00"))
                if mark_as_of_raw is not None
                else event.occurred_at
            )
            portfolio = _apply_order_economics(
                portfolio,
                order,
                mark_price=mark_price,
                mark_as_of=mark_as_of,
                as_of=event.occurred_at,
            )
        elif event.event_type == LedgerEventType.DAY_CLOSED:
            portfolio = _apply_day_closed(portfolio, event)
        else:
            raise ValueError("UNSUPPORTED_LEDGER_EVENT_TYPE")
    return portfolio


def mark_to_market(
    portfolio: PaperPortfolio,
    market_snapshot: Sequence[MarketQuote],
    as_of: datetime,
    *,
    max_stale_market_age: timedelta | None = timedelta(minutes=30),
    allow_incomplete: bool = False,
) -> PaperPortfolio:
    if as_of < portfolio.as_of:
        raise ValueError("MARK_TIME_PRECEDES_PORTFOLIO_STATE")
    quotes = {quote.ticker.upper(): quote for quote in market_snapshot}
    positions: list[PaperPosition] = []
    for position in portfolio.positions:
        quote = quotes.get(position.ticker.upper())
        status = _position_mark_status(
            position,
            quote,
            decision_as_of=as_of,
            max_stale_market_age=max_stale_market_age,
        )
        if status != PositionMarkStatus.FRESH:
            if allow_incomplete:
                positions.append(position)
                continue
            if status == PositionMarkStatus.FUTURE:
                raise ValueError("FUTURE_MARKET_SNAPSHOT")
            if status == PositionMarkStatus.STALE:
                raise ValueError("STALE_MARKET_SNAPSHOT")
            raise ValueError(f"MARK_PRICE_UNAVAILABLE:{position.ticker}")
        assert quote is not None and quote.last_price is not None
        positions.append(
            _position(
                position.ticker,
                position.quantity,
                position.average_cost,
                quote.last_price,
                0.0,
                quote.as_of,
            )
        )
    return _portfolio_from_positions(
        portfolio=portfolio,
        cash=portfolio.cash,
        positions=positions,
        realized_pnl=portfolio.realized_pnl,
        turnover_today=portfolio.turnover_today,
        as_of=as_of,
    )


def portfolio_mark_health(
    portfolio: PaperPortfolio,
    market_snapshot: Sequence[MarketQuote],
    *,
    decision_as_of: datetime,
    max_stale_market_age: timedelta,
) -> tuple[PortfolioMarkStatus, dict[str, PositionMarkStatus]]:
    quotes = {quote.ticker.upper(): quote for quote in market_snapshot}
    statuses = {
        position.ticker: _position_mark_status(
            position,
            quotes.get(position.ticker.upper()),
            decision_as_of=decision_as_of,
            max_stale_market_age=max_stale_market_age,
        )
        for position in portfolio.positions
    }
    overall = (
        PortfolioMarkStatus.COMPLETE
        if all(status == PositionMarkStatus.FRESH for status in statuses.values())
        else PortfolioMarkStatus.DEGRADED
    )
    return overall, dict(sorted(statuses.items()))


def _position_mark_status(
    position: PaperPosition,
    quote: MarketQuote | None,
    *,
    decision_as_of: datetime,
    max_stale_market_age: timedelta | None,
) -> PositionMarkStatus:
    if quote is None:
        return PositionMarkStatus.MISSING
    if quote.as_of > decision_as_of:
        return PositionMarkStatus.FUTURE
    if quote.last_price is None or quote.last_price <= 0:
        return PositionMarkStatus.INVALID
    if max_stale_market_age is not None and decision_as_of - quote.as_of > max_stale_market_age:
        return PositionMarkStatus.STALE
    if quote.as_of < position.mark_as_of:
        return PositionMarkStatus.STALE
    return PositionMarkStatus.FRESH


def restore_operational_portfolio(
    repository: PaperLedgerRepository,
    *,
    as_of: datetime,
    market_snapshot: Sequence[MarketQuote] = (),
    max_stale_market_age: timedelta = timedelta(minutes=30),
) -> PaperPortfolio:
    events = repository.events()
    portfolio = replay_portfolio(events) if events else initial_paper_portfolio(as_of)
    if market_snapshot:
        portfolio = mark_to_market(
            portfolio,
            market_snapshot,
            as_of,
            max_stale_market_age=max_stale_market_age,
            allow_incomplete=True,
        )
    return portfolio


def close_paper_day(
    repository: PaperLedgerRepository,
    *,
    next_day_as_of: datetime,
) -> PaperPortfolio:
    if next_day_as_of.tzinfo is None or next_day_as_of.utcoffset() is None:
        raise ValueError("DAY_TRANSITION_TIME_MUST_BE_TIMEZONE_AWARE")
    events = repository.events()
    if not events:
        raise ValueError("EMPTY_PAPER_LEDGER")
    portfolio = replay_portfolio(events)
    if _moex_day(next_day_as_of) <= _moex_day(portfolio.as_of):
        raise ValueError("DAY_TRANSITION_MUST_ADVANCE_MOEX_DATE")
    repository.append(
        LedgerEvent(
            sequence=repository.last_sequence() + 1,
            event_id=_id(
                "ledger", portfolio.portfolio_id, "day-closed", next_day_as_of.isoformat()
            ),
            event_type=LedgerEventType.DAY_CLOSED,
            portfolio_id=portfolio.portfolio_id,
            occurred_at=next_day_as_of,
            payload={"start_of_day_equity": portfolio.equity},
        )
    )
    return replay_portfolio(repository.events())


def _apply_day_closed(portfolio: PaperPortfolio, event: LedgerEvent) -> PaperPortfolio:
    start_equity = float(cast("int | float", event.payload.get("start_of_day_equity", 0.0)))
    if start_equity <= 0 or _moex_day(event.occurred_at) <= _moex_day(portfolio.as_of):
        raise ValueError("INVALID_DAY_CLOSED_EVENT")
    return portfolio.model_copy(
        update={
            "start_of_day_equity": _money(start_equity),
            "turnover_today": 0.0,
            "daily_pnl": 0.0,
            "as_of": event.occurred_at,
        }
    )


def _moex_day(value: datetime) -> date:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("TIME_MUST_BE_TIMEZONE_AWARE")
    return value.astimezone(MOEX_TIMEZONE).date()


def initial_paper_portfolio(as_of: datetime, cash: float = 1_000_000.0) -> PaperPortfolio:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("PORTFOLIO_TIME_MUST_BE_TIMEZONE_AWARE")
    return PaperPortfolio(
        portfolio_id="paper-portfolio-v1",
        cash=cash,
        equity=cash,
        positions=[],
        peak_equity=cash,
        drawdown=0.0,
        turnover_today=0.0,
        start_of_day_equity=cash,
        as_of=as_of,
    )


def _pre_trade_rejection(
    *,
    proposal: TradeProposal,
    quote: MarketQuote | None,
    policy: RiskPolicy,
    decision_as_of: datetime,
    universe: set[str],
    agent_valid: bool,
    research_ready: bool,
    has_conflict: bool,
    portfolio: PaperPortfolio,
    portfolio_marks_complete: bool,
) -> RiskReasonCode | None:
    ticker = proposal.ticker.upper()
    if not agent_valid:
        return RiskReasonCode.INVALID_PROPOSAL
    if has_conflict:
        return RiskReasonCode.DUPLICATE_OR_CONFLICTING_PROPOSAL
    if ticker not in universe or quote is None or not quote.supported:
        return RiskReasonCode.UNSUPPORTED_INSTRUMENT
    if proposal.action == TradeAction.BUY and not portfolio_marks_complete:
        return RiskReasonCode.PORTFOLIO_MARK_INCOMPLETE
    if quote.as_of > decision_as_of:
        return RiskReasonCode.FUTURE_MARKET_SNAPSHOT
    if decision_as_of - quote.as_of > policy.max_stale_market_age:
        return RiskReasonCode.STALE_DATA
    if not research_ready and proposal.action in {TradeAction.BUY, TradeAction.SELL}:
        return RiskReasonCode.RESEARCH_HEALTH_DEGRADED
    if policy.kill_switch_enabled and proposal.action in {TradeAction.BUY, TradeAction.SELL}:
        return RiskReasonCode.KILL_SWITCH
    if quote.last_price is None:
        return RiskReasonCode.PRICE_UNAVAILABLE
    if (
        quote.last_price <= 0
        or (quote.ask is not None and quote.ask <= 0)
        or (quote.bid is not None and quote.bid <= 0)
    ):
        return RiskReasonCode.INVALID_PRICE
    if quote.lot_size is None:
        return RiskReasonCode.LOT_SIZE_UNAVAILABLE
    if proposal.action == TradeAction.BUY:
        if portfolio.daily_pnl / portfolio.start_of_day_equity <= -policy.max_daily_loss_pct:
            return RiskReasonCode.DAILY_LOSS_LIMIT
        if portfolio.drawdown >= policy.max_portfolio_drawdown_pct:
            return RiskReasonCode.DRAWDOWN_LIMIT
    return None


def _cap_buy_value(
    *,
    desired_value: float,
    current_value: float,
    projected_values: dict[str, float],
    projected_cash: float,
    projected_turnover: float,
    portfolio: PaperPortfolio,
    policy: RiskPolicy,
) -> tuple[float, list[RiskReasonCode]]:
    equity = portfolio.equity
    caps: list[tuple[float, RiskReasonCode]] = [
        (policy.max_position_weight * equity, RiskReasonCode.POSITION_LIMIT),
        (
            current_value
            + max(0.0, policy.max_gross_exposure * equity - sum(projected_values.values())),
            RiskReasonCode.PORTFOLIO_GROSS_EXPOSURE_LIMIT,
        ),
        (
            current_value
            + max(0.0, policy.max_net_exposure * equity - sum(projected_values.values())),
            RiskReasonCode.PORTFOLIO_NET_EXPOSURE_LIMIT,
        ),
        (
            current_value + max(0.0, projected_cash - policy.min_cash_buffer_pct * equity),
            RiskReasonCode.CASH_LIMIT,
        ),
        (
            current_value
            + max(
                0.0,
                policy.max_daily_turnover * portfolio.start_of_day_equity - projected_turnover,
            ),
            RiskReasonCode.TURNOVER_LIMIT,
        ),
        (
            current_value + policy.max_single_order_notional_pct * equity,
            RiskReasonCode.SINGLE_ORDER_LIMIT,
        ),
    ]
    approved = desired_value
    reasons: list[RiskReasonCode] = []
    for cap, reason in caps:
        if cap < desired_value:
            reasons.append(reason)
        approved = min(approved, cap)
    return max(current_value, approved), reasons


def _cap_sell_value(
    *,
    desired_value: float,
    current_value: float,
    projected_turnover: float,
    portfolio: PaperPortfolio,
    policy: RiskPolicy,
) -> tuple[float, list[RiskReasonCode]]:
    requested_reduction = max(0.0, current_value - desired_value)
    caps: list[tuple[float, RiskReasonCode]] = []
    if not policy.risk_reducing_sell_turnover_exempt:
        caps.append(
            (
                max(
                    0.0,
                    policy.max_daily_turnover * portfolio.start_of_day_equity - projected_turnover,
                ),
                RiskReasonCode.TURNOVER_LIMIT,
            )
        )
    if not policy.risk_reducing_sell_order_cap_exempt:
        caps.append(
            (
                policy.max_single_order_notional_pct * portfolio.equity,
                RiskReasonCode.SINGLE_ORDER_LIMIT,
            )
        )
    allowed_reduction = requested_reduction
    reasons: list[RiskReasonCode] = []
    for cap, reason in caps:
        if cap < requested_reduction:
            reasons.append(reason)
        allowed_reduction = min(allowed_reduction, cap)
    return current_value - max(0.0, allowed_reduction), reasons


def _plan_order(
    *,
    agent_run_id: str,
    proposal_id: str,
    ticker: str,
    side: PaperSide,
    delta_value: float,
    quote: MarketQuote,
    policy: RiskPolicy,
    decision_as_of: datetime,
) -> PaperOrder | None:
    base_price, source = _execution_base_price(quote, side)
    direction = 1.0 if side == PaperSide.BUY else -1.0
    execution_price = base_price * (1.0 + direction * policy.slippage_bps / 10_000)
    unit_budget_cost = execution_price
    if side == PaperSide.BUY:
        unit_budget_cost *= 1.0 + policy.commission_bps / 10_000
    raw_quantity = abs(delta_value) / unit_budget_cost
    assert quote.lot_size is not None
    quantity = math.floor(raw_quantity / quote.lot_size) * quote.lot_size
    gross = _money(quantity * execution_price)
    if quantity <= 0 or gross < policy.min_trade_notional:
        return None
    commission = _money(gross * policy.commission_bps / 10_000)
    cash_effect = -(gross + commission) if side == PaperSide.BUY else gross - commission
    order_id = _id("paper-order", agent_run_id, proposal_id)
    return PaperOrder(
        paper_order_id=order_id,
        idempotency_key=f"{agent_run_id}:{proposal_id}",
        run_id=agent_run_id,
        proposal_id=proposal_id,
        ticker=ticker,
        side=side,
        quantity=quantity,
        lot_size=quote.lot_size,
        planned_price=base_price,
        execution_price=_money(execution_price),
        price_source=source,
        slippage_bps=policy.slippage_bps,
        commission_bps=policy.commission_bps,
        gross_notional=gross,
        commission=commission,
        net_cash_effect=_money(cash_effect),
        created_at=decision_as_of,
    )


def _execution_base_price(
    quote: MarketQuote, side: PaperSide
) -> tuple[float, ExecutionPriceSource]:
    assert quote.last_price is not None
    if side == PaperSide.BUY and quote.ask is not None:
        return quote.ask, ExecutionPriceSource.ASK_PLUS_SLIPPAGE
    if side == PaperSide.SELL and quote.bid is not None:
        return quote.bid, ExecutionPriceSource.BID_MINUS_SLIPPAGE
    return quote.last_price, ExecutionPriceSource.LAST_PLUS_SLIPPAGE


def _replace_order_quantity(order: PaperOrder, quantity: int, policy: RiskPolicy) -> PaperOrder:
    quantity = quantity - quantity % order.lot_size
    gross = _money(quantity * order.execution_price)
    commission = _money(gross * policy.commission_bps / 10_000)
    cash_effect = gross - commission
    return order.model_copy(
        update={
            "quantity": quantity,
            "gross_notional": gross,
            "commission": commission,
            "net_cash_effect": _money(cash_effect),
        }
    )


def _decision(
    proposal_id: str,
    agent_run_id: str,
    proposal: TradeProposal,
    decision: RiskDecisionType,
    approved_weight: float,
    reasons: list[RiskReasonCode],
    as_of: datetime,
) -> RiskDecision:
    return RiskDecision(
        decision_id=_id("risk-decision", agent_run_id, proposal_id),
        proposal_id=proposal_id,
        agent_run_id=agent_run_id,
        ticker=proposal.ticker.upper(),
        proposal_action=proposal.action.value,
        agent_target_weight=proposal.target_weight,
        risk_decision=decision,
        approved_target_weight=round(approved_weight, 8),
        reason_codes=list(dict.fromkeys(reasons)),
        decision_as_of=as_of,
    )


def _proposals(agent_run: dict[str, Any]) -> list[TradeProposal]:
    raw: object = agent_run.get("final_proposals", [])
    if not isinstance(raw, list):
        return []
    try:
        return [TradeProposal.model_validate(row) for row in cast("list[object]", raw)]
    except Exception:
        return []


def _portfolio_from_positions(
    *,
    portfolio: PaperPortfolio,
    cash: float,
    positions: list[PaperPosition],
    realized_pnl: float,
    turnover_today: float,
    as_of: datetime,
) -> PaperPortfolio:
    market_values = sum(position.quantity * position.last_price for position in positions)
    equity = _money(cash + market_values)
    weighted = [
        _position(
            position.ticker,
            position.quantity,
            position.average_cost,
            position.last_price,
            (position.quantity * position.last_price / equity) if equity else 0.0,
            position.mark_as_of,
        )
        for position in positions
    ]
    unrealized = sum(position.unrealized_pnl for position in weighted)
    peak = max(portfolio.peak_equity, equity)
    drawdown = 0.0 if peak <= 0 else max(0.0, (peak - equity) / peak)
    return PaperPortfolio(
        portfolio_id=portfolio.portfolio_id,
        cash=_money(cash),
        equity=equity,
        positions=sorted(weighted, key=lambda row: row.ticker),
        realized_pnl=_money(realized_pnl),
        unrealized_pnl=_money(unrealized),
        daily_pnl=_money(equity - portfolio.start_of_day_equity),
        peak_equity=_money(peak),
        drawdown=round(drawdown, 8),
        turnover_today=round(turnover_today, 8),
        start_of_day_equity=portfolio.start_of_day_equity,
        as_of=as_of,
    )


def _position(
    ticker: str,
    quantity: int,
    average_cost: float,
    last_price: float,
    weight: float,
    mark_as_of: datetime,
) -> PaperPosition:
    market_value = quantity * last_price
    return PaperPosition(
        ticker=ticker,
        quantity=quantity,
        average_cost=_money(average_cost),
        last_price=_money(last_price),
        market_value=_money(market_value),
        weight=round(weight, 8),
        unrealized_pnl=_money((last_price - average_cost) * quantity),
        mark_as_of=mark_as_of,
    )


def _position_quantity(portfolio: PaperPortfolio, ticker: str) -> int:
    return next(
        (position.quantity for position in portfolio.positions if position.ticker == ticker), 0
    )


def _quote(snapshot: Sequence[MarketQuote], ticker: str) -> MarketQuote:
    for quote in snapshot:
        if quote.ticker.upper() == ticker.upper():
            return quote
    raise ValueError(f"QUOTE_NOT_FOUND:{ticker}")


def portfolio_state_sha(portfolio: PaperPortfolio) -> str:
    return sha256_payload(portfolio.model_dump(mode="json"))


def market_snapshot_sha(snapshot: Sequence[MarketQuote]) -> str:
    return sha256_payload(
        [
            row.model_dump(mode="json")
            for row in sorted(snapshot, key=lambda value: value.ticker.upper())
        ]
    )


def _id(prefix: str, *parts: str) -> str:
    return f"{prefix}-{sha256_payload(list(parts))[:20]}"


def _money(value: float) -> float:
    return round(value, 4)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return cast("dict[str, Any]", value)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


def total_commission(trades: Iterable[PaperTrade]) -> float:
    return _money(sum(trade.commission for trade in trades))
