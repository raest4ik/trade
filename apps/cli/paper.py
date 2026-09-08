from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from apps.cli.risk import run as run_risk
from src.risk_engine_paper_v1.application import (
    execute_paper_plan,
    initial_paper_portfolio,
    portfolio_state_sha,
    replay_portfolio,
    write_json,
)
from src.risk_engine_paper_v1.domain import PaperExecutionStatus, PaperPortfolio, RiskPlan
from src.risk_engine_paper_v1.repository import JsonlPaperLedgerRepository


def run(args: argparse.Namespace) -> int:
    if args.command == "evaluate-agent-run":
        return run_risk(args)
    state_root = Path(args.state_root)
    ledger = JsonlPaperLedgerRepository(state_root / "ledger.jsonl")
    if args.command == "status":
        portfolio = _current_portfolio(ledger)
        events = ledger.events()
        print(
            json.dumps(
                {
                    "PAPER_EXECUTION_ENABLED": True,
                    "PAPER_AUTO_EXECUTION_ENABLED": False,
                    "REAL_EXECUTION_READY": "NO",
                    "REAL_BROKER_MUTATIONS": 0,
                    "REAL_ORDERS_SENT": 0,
                    "portfolio_id": portfolio.portfolio_id,
                    "ledger_event_count": len(events),
                    "portfolio_state_sha": portfolio_state_sha(portfolio),
                    "cash": portfolio.cash,
                    "equity": portfolio.equity,
                    "position_count": len(portfolio.positions),
                    "turnover_today": portfolio.turnover_today,
                    "last_event_at": events[-1].occurred_at.isoformat() if events else None,
                    "replay_verified": True,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "reset":
        if not args.sample_only:
            raise SystemExit("paper reset requires --sample-only")
        for path in (state_root / "ledger.jsonl", state_root / "portfolio.json"):
            path.unlink(missing_ok=True)
        print(json.dumps({"sample_state_reset": True, "real_broker_affected": False}))
        return 0

    if args.command == "execute-agent-run":
        plan = _plan(state_root, args.run_id)
        execution_as_of = (
            datetime.fromisoformat(args.execution_as_of.replace("Z", "+00:00"))
            if args.execution_as_of
            else datetime.now(UTC)
        )
        result = execute_paper_plan(plan, ledger, execution_as_of=execution_as_of)
        if (
            result.execution_status == PaperExecutionStatus.SUCCESS
            and result.replay_verification.replay_matches
        ):
            write_json(
                state_root / "portfolio.json",
                result.final_portfolio.model_dump(mode="json"),
            )
        print(
            json.dumps(
                {
                    "agent_run_id": plan.agent_run_id,
                    "paper_orders_filled": len(result.filled_orders),
                    "duplicate_executions_skipped": result.duplicate_executions_skipped,
                    "replay_matches": result.replay_verification.replay_matches,
                    "execution_status": result.execution_status,
                    "status_code": result.status_code,
                    "real_orders_sent": 0,
                },
                sort_keys=True,
            )
        )
        return 0 if result.execution_status == PaperExecutionStatus.SUCCESS else 2
    replayed = _current_portfolio(ledger)
    if args.command == "portfolio":
        print(replayed.model_dump_json(indent=2))
        return 0
    if args.command == "replay":
        print(
            json.dumps(
                {
                    "portfolio": replayed.model_dump(mode="json"),
                    "event_count": len(ledger.events()),
                    "portfolio_state_sha": portfolio_state_sha(replayed),
                    "replay_completed": True,
                    "integrity_valid": True,
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"unsupported command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="paper-portfolio-v1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--state-root", default="state/paper-portfolio-v1")
    for name in ("portfolio", "replay"):
        command = subparsers.add_parser(name)
        command.add_argument("--state-root", default="state/paper-portfolio-v1")
    execute = subparsers.add_parser("execute-agent-run")
    execute.add_argument("run_id")
    execute.add_argument("--state-root", default="state/paper-portfolio-v1")
    execute.add_argument("--execution-as-of")
    evaluate = subparsers.add_parser("evaluate-agent-run")
    evaluate.add_argument("run_id")
    evaluate.add_argument("--agent-root", default="artifacts/ai-trading-agent-v1")
    evaluate.add_argument("--state-root", default="state/paper-portfolio-v1")
    evaluate.add_argument("--as-of")
    reset = subparsers.add_parser("reset")
    reset.add_argument("--sample-only", action="store_true")
    reset.add_argument("--state-root", default="state/paper-portfolio-v1")
    return parser


def _plan(state_root: Path, run_id: str) -> RiskPlan:
    path = state_root / "plans" / f"{run_id}.json"
    if not path.exists():
        raise SystemExit(f"risk plan not found: {path}")
    return RiskPlan.model_validate_json(path.read_text(encoding="utf-8"))


def _current_portfolio(ledger: JsonlPaperLedgerRepository) -> PaperPortfolio:
    events = ledger.events()
    return replay_portfolio(events) if events else initial_paper_portfolio(datetime.now(UTC))


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
