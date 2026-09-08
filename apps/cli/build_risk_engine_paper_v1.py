from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ai_trading_agent_v1.application import git_sha
from src.risk_engine_paper_v1.application import DEFAULT_ARTIFACT_ROOT
from src.risk_engine_paper_v1.reporting import build_sample_execution, write_audit_artifact


def run(args: argparse.Namespace) -> int:
    sample = build_sample_execution(Path(args.agent_run))
    manifest = write_audit_artifact(
        output_root=Path(args.output_root),
        code_sha=git_sha(),
        sample=sample,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic Risk/Paper V1 artifact")
    parser.add_argument("--agent-run", default="artifacts/ai-trading-agent-v1/run.json")
    parser.add_argument("--output-root", default=str(DEFAULT_ARTIFACT_ROOT))
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
