from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from src.ai_trading_agent_v1.domain import TradeAction, TradeProposal
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.risk_engine_paper_v1.domain import (
    ExecutionPriceSource,
    LedgerEvent,
    LedgerEventType,
    MarketQuote,
    PaperExecutionResult,
    PaperOrder,
    PaperOrderStatus,
    PaperPortfolio,
    PaperPosition,
    PaperSide,
    PaperTrade,
    PipelineSafety,
    ReplayVerification,
    RiskDecision,
    RiskDecisionType,
    RiskPlan,
    RiskPolicy,
    RiskReasonCode,
)
from src.risk_engine_paper_v1.repository import PaperLedgerRepository

ARTIFACT_VERSION = "risk-engine-paper-portfolio-v1"
DEFAULT_ARTIFACT_ROOT = Path(f"artifacts/{ARTIFACT_VERSION}")
DEFAULT_LEDGER_PATH = Path("state/paper-portfolio-v1/ledger.jsonl")


def evaluate_agent_run(
    *,
    agent_run: dict[str, Any],
    portfolio: PaperPortfolio,
    market_snapshot: Sequence[MarketQuote],
    policy: RiskPolicy,
    decision_as_of: datetime,
) -> RiskPlan:
    agent_run_id = str(agent_run.get("run_id", ""))
    proposals = _proposals(agent_run)
    agent_run_sha = sha256_payload(agent_run)
    plan_id = _id("risk-plan", agent_run_id, agent_run_sha, policy.policy_version)
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
            delta = desired_value - current_value

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
        reduced = approved_weight + policy.target_weight_tolerance < proposal.target_weight
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
        decision_as_of=decision_as_of,
        initial_portfolio=portfolio,
        market_snapshot=list(market_snapshot),
        decisions=decisions,
        paper_orders=orders,
    )


def execute_paper_plan(
    plan: RiskPlan,
    repository: PaperLedgerRepository,
) -> PaperExecutionResult:
    if not repository.events():
        repository.append(
            LedgerEvent(
                sequence=1,
                event_id=_id("ledger", plan.plan_id, "portfolio-created"),
                event_type=LedgerEventType.PORTFOLIO_CREATED,
                portfolio_id=plan.initial_portfolio.portfolio_id,
                occurred_at=plan.decision_as_of,
                payload={"portfolio": plan.initial_portfolio.model_dump(mode="json")},
            )
        )
    portfolio = replay_portfolio(plan.initial_portfolio, repository.events(), plan.market_snapshot)
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
                },
            )
        )
        filled_orders.append(filled)
        trades.append(trade)

    replayed = replay_portfolio(plan.initial_portfolio, repository.events(), plan.market_snapshot)
    expected_sha = portfolio_state_sha(portfolio)
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
    return PaperExecutionResult(
        plan=plan,
        filled_orders=filled_orders,
        paper_trades=trades,
        final_portfolio=replayed,
        replay_verification=verification,
        duplicate_executions_skipped=duplicate_skips,
        safety=safety,
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
    new_cash = _money(portfolio.cash + trade.cash_delta)
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
            order.ticker, new_quantity, average_cost, quote.last_price or order.execution_price, 0.0
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
        as_of=executed_at,
    )
    filled = order.model_copy(
        update={"status": PaperOrderStatus.FILLED, "executed_at": executed_at}
    )
    return updated, filled, trade


def replay_portfolio(
    initial: PaperPortfolio,
    events: Sequence[LedgerEvent],
    market_snapshot: Sequence[MarketQuote],
) -> PaperPortfolio:
    portfolio = initial
    for event in events:
        if event.portfolio_id != initial.portfolio_id:
            continue
        if event.event_type != LedgerEventType.PAPER_ORDER_FILLED:
            continue
        order = PaperOrder.model_validate(event.payload["order"])
        planned = order.model_copy(update={"status": PaperOrderStatus.PLANNED, "executed_at": None})
        portfolio, _, _ = apply_paper_order(portfolio, planned, market_snapshot)
    return portfolio


def mark_to_market(
    portfolio: PaperPortfolio,
    market_snapshot: Sequence[MarketQuote],
    as_of: datetime,
) -> PaperPortfolio:
    quotes = {quote.ticker.upper(): quote for quote in market_snapshot}
    positions: list[PaperPosition] = []
    for position in portfolio.positions:
        quote = quotes.get(position.ticker.upper())
        if quote is None or quote.last_price is None or quote.last_price <= 0:
            raise ValueError(f"MARK_PRICE_UNAVAILABLE:{position.ticker}")
        if quote.as_of > as_of:
            raise ValueError("FUTURE_MARKET_SNAPSHOT")
        positions.append(
            _position(
                position.ticker,
                position.quantity,
                position.average_cost,
                quote.last_price,
                0.0,
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


def initial_paper_portfolio(as_of: datetime, cash: float = 1_000_000.0) -> PaperPortfolio:
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
) -> RiskReasonCode | None:
    ticker = proposal.ticker.upper()
    if not agent_valid:
        return RiskReasonCode.INVALID_PROPOSAL
    if has_conflict:
        return RiskReasonCode.DUPLICATE_OR_CONFLICTING_PROPOSAL
    if ticker not in universe or quote is None or not quote.supported:
        return RiskReasonCode.UNSUPPORTED_INSTRUMENT
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
            RiskReasonCode.LIQUIDITY_LIMIT,
        ),
    ]
    approved = desired_value
    reasons: list[RiskReasonCode] = []
    for cap, reason in caps:
        if cap < desired_value:
            reasons.append(reason)
        approved = min(approved, cap)
    return max(current_value, approved), reasons


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
