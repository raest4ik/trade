from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.ai_trading_agent_v1.application import (
    ARTIFACT_VERSION,
    DEFAULT_OUTPUT_ROOT,
    AgentRunConfig,
    UnconfiguredAgentModel,
    git_sha,
    run_read_only_research_agent_v1,
    sample_agent_context,
    sample_fake_agent_model,
)


def run(args: argparse.Namespace) -> int:
    created_at = _created_at(args.created_at) if args.created_at else datetime.now(UTC)
    if args.command == "status":
        print(
            json.dumps(
                {
                    "ARTIFACT_VERSION": ARTIFACT_VERSION,
                    "AGENT_MODE": "READ_ONLY_RESEARCH",
                    "EXTERNAL_WEB_SEARCH_ENABLED": False,
                    "BROKER_MUTATIONS": 0,
                    "REAL_ORDERS_SENT": 0,
                    "PAPER_ORDERS_SENT": 0,
                    "PORTFOLIO_MUTATIONS": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "inspect-run":
        manifest_path = Path(args.output_root) / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("run_id") != args.run_id:
            raise SystemExit(f"run_id not found in {manifest_path}")
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    output_root = Path(args.output_root)
    config = AgentRunConfig(
        output_root=output_root,
        code_sha=git_sha(),
        run_id="sample-agent-run-v1" if args.command == "dry-run" else None,
        created_at=created_at,
    )
    if args.command == "dry-run":
        context = sample_agent_context(created_at)
        model = sample_fake_agent_model(created_at)
    else:
        context = None
        model = UnconfiguredAgentModel()
    result = run_read_only_research_agent_v1(
        config=config,
        model=model,
        deterministic_context=context,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "AGENT_RESEARCH_CAPABILITY_READY": result.AGENT_RESEARCH_CAPABILITY_READY,
                "AGENT_DECISION_STATUS": result.AGENT_DECISION_STATUS.value,
                "BROKER_MUTATIONS": result.safety.BROKER_MUTATIONS,
                "REAL_ORDERS_SENT": result.safety.REAL_ORDERS_SENT,
                "PAPER_ORDERS_SENT": result.safety.PAPER_ORDERS_SENT,
                "PORTFOLIO_MUTATIONS": result.safety.PORTFOLIO_MUTATIONS,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=ARTIFACT_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "dry-run"):
        child = subparsers.add_parser(name)
        child.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
        child.add_argument("--created-at", default=None, help=argparse.SUPPRESS)
    inspect = subparsers.add_parser("inspect-run")
    inspect.add_argument("run_id")
    inspect.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    subparsers.add_parser("status")
    return parser


def _created_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("created-at must include timezone")
    return parsed.astimezone(UTC)


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
