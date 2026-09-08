from __future__ import annotations

import argparse
import json
from pathlib import Path

from apps.cli.risk import run as run_risk
from src.risk_engine_paper_v1.application import execute_paper_plan, replay_portfolio, write_json
from src.risk_engine_paper_v1.domain import RiskPlan
from src.risk_engine_paper_v1.repository import JsonlPaperLedgerRepository


def run(args: argparse.Namespace) -> int:
    if args.command == "evaluate-agent-run":
        return run_risk(args)
    state_root = Path(args.state_root)
    ledger = JsonlPaperLedgerRepository(state_root / "ledger.jsonl")
    if args.command == "status":
        print(
            json.dumps(
                {
                    "PAPER_EXECUTION_ENABLED": True,
                    "PAPER_AUTO_EXECUTION_ENABLED": False,
                    "REAL_EXECUTION_READY": "NO",
                    "REAL_BROKER_MUTATIONS": 0,
                    "REAL_ORDERS_SENT": 0,
                    "ledger_events": len(ledger.events()),
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

    plan = _plan(state_root, args.run_id)
    if args.command == "execute-agent-run":
        result = execute_paper_plan(plan, ledger)
        write_json(state_root / "portfolio.json", result.final_portfolio.model_dump(mode="json"))
        print(
            json.dumps(
                {
                    "agent_run_id": plan.agent_run_id,
                    "paper_orders_filled": len(result.filled_orders),
                    "duplicate_executions_skipped": result.duplicate_executions_skipped,
                    "replay_matches": result.replay_verification.replay_matches,
                    "real_orders_sent": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    replayed = replay_portfolio(plan.initial_portfolio, ledger.events(), plan.market_snapshot)
    if args.command == "portfolio":
        print(replayed.model_dump_json(indent=2))
        return 0
    if args.command == "replay":
        print(
            json.dumps(
                {
                    "portfolio": replayed.model_dump(mode="json"),
                    "event_count": len(ledger.events()),
                    "replay_completed": True,
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
        command.add_argument("run_id", nargs="?")
        command.add_argument("--state-root", default="state/paper-portfolio-v1")
    execute = subparsers.add_parser("execute-agent-run")
    execute.add_argument("run_id")
    execute.add_argument("--state-root", default="state/paper-portfolio-v1")
    evaluate = subparsers.add_parser("evaluate-agent-run")
    evaluate.add_argument("run_id")
    evaluate.add_argument("--agent-root", default="artifacts/ai-trading-agent-v1")
    evaluate.add_argument("--state-root", default="state/paper-portfolio-v1")
    evaluate.add_argument("--as-of")
    reset = subparsers.add_parser("reset")
    reset.add_argument("--sample-only", action="store_true")
    reset.add_argument("--state-root", default="state/paper-portfolio-v1")
    return parser


def _plan(state_root: Path, run_id: str | None) -> RiskPlan:
    if run_id is None:
        plans = sorted((state_root / "plans").glob("*.json"))
        if not plans:
            raise SystemExit("no risk plans found")
        path = plans[-1]
    else:
        path = state_root / "plans" / f"{run_id}.json"
    if not path.exists():
        raise SystemExit(f"risk plan not found: {path}")
    return RiskPlan.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
