from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from src.risk_engine_paper_v1.application import (
    DEFAULT_LEDGER_PATH,
    evaluate_agent_run,
    load_json,
    restore_operational_portfolio,
    write_json,
)
from src.risk_engine_paper_v1.domain import MarketQuote, RiskPlan, RiskPolicy
from src.risk_engine_paper_v1.repository import JsonlPaperLedgerRepository


def run(args: argparse.Namespace) -> int:
    if args.command == "policy":
        print(RiskPolicy().model_dump_json(indent=2))
        return 0
    state_root = Path(args.state_root)
    if args.command == "inspect":
        for path in sorted((state_root / "plans").glob("*.json")):
            plan = RiskPlan.model_validate_json(path.read_text(encoding="utf-8"))
            for decision in plan.decisions:
                if decision.decision_id == args.decision_id:
                    print(decision.model_dump_json(indent=2))
                    return 0
        raise SystemExit(f"risk decision not found: {args.decision_id}")

    agent_path = Path(args.agent_root) / "run.json"
    agent_run = load_json(agent_path)
    if agent_run.get("run_id") != args.run_id:
        raise SystemExit(f"agent run not found: {args.run_id}")
    decision_as_of = _datetime(args.as_of) if args.as_of else datetime.now(UTC)
    quotes = _quotes(agent_run)
    ledger = JsonlPaperLedgerRepository(state_root / "ledger.jsonl")
    policy = RiskPolicy()
    portfolio = restore_operational_portfolio(
        ledger,
        as_of=decision_as_of,
        market_snapshot=quotes,
        max_stale_market_age=policy.max_stale_market_age,
    )
    plan = evaluate_agent_run(
        agent_run=agent_run,
        portfolio=portfolio,
        market_snapshot=quotes,
        policy=policy,
        decision_as_of=decision_as_of,
        ledger_event_count=ledger.last_sequence(),
    )
    output = state_root / "plans" / f"{args.run_id}.json"
    write_json(output, plan.model_dump(mode="json"))
    print(
        json.dumps(
            {
                "plan_id": plan.plan_id,
                "agent_run_id": plan.agent_run_id,
                "decisions": len(plan.decisions),
                "paper_orders_planned": len(plan.paper_orders),
                "portfolio_mutated": False,
                "portfolio_state_sha": plan.portfolio_state_sha,
                "ledger_event_count": plan.ledger_event_count,
                "portfolio_mark_status": plan.portfolio_mark_status,
                "position_mark_statuses": plan.position_mark_statuses,
                "real_execution_ready": False,
                "ledger_path": str(state_root / DEFAULT_LEDGER_PATH.name),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="paper-risk-policy-v1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("policy")
    evaluate = subparsers.add_parser("evaluate-agent-run")
    evaluate.add_argument("run_id")
    evaluate.add_argument("--agent-root", default="artifacts/ai-trading-agent-v1")
    evaluate.add_argument("--state-root", default="state/paper-portfolio-v1")
    evaluate.add_argument("--as-of")
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("decision_id")
    inspect.add_argument("--state-root", default="state/paper-portfolio-v1")
    return parser


def _quotes(agent_run: dict[str, Any]) -> list[MarketQuote]:
    market = cast("dict[str, Any]", agent_run.get("market_context_snapshot", {}))
    by_ticker = cast("dict[str, dict[str, Any]]", market.get("by_ticker", {}))
    universe = {
        str(row.get("ticker", "")).upper(): row
        for row in cast("list[dict[str, Any]]", agent_run.get("universe", []))
    }
    rows: list[MarketQuote] = []
    for ticker, value in sorted(by_ticker.items()):
        timestamp = value.get("market_data_as_of")
        if not isinstance(timestamp, str):
            continue
        instrument = universe.get(ticker.upper(), {})
        rows.append(
            MarketQuote(
                ticker=ticker.upper(),
                as_of=_datetime(timestamp),
                last_price=_number(value.get("last_price")),
                bid=_number(value.get("bid")),
                ask=_number(value.get("ask")),
                lot_size=cast("int | None", instrument.get("lot_size")),
                supported=instrument.get("supported") is True,
            )
        )
    return rows


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include timezone")
    return parsed.astimezone(UTC)


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
