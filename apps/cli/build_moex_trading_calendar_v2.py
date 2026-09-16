from __future__ import annotations

import argparse
from pathlib import Path

from src.ai_trading_agent_v1.application import git_sha
from src.moex_trading_calendar_v2.reporting import (
    DEFAULT_OUTPUT_ROOT,
    build_calendar_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic MOEX calendar V2 audit")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--code-sha", default=None)
    parser.add_argument("--ledger-path", type=Path, default=None)
    args = parser.parse_args()
    manifest = build_calendar_artifact(
        Path(args.output_root),
        base_main_sha=args.base_main_sha,
        code_sha=args.code_sha or git_sha(),
        **({"ledger_path": args.ledger_path} if args.ledger_path is not None else {}),
    )
    print(manifest)


if __name__ == "__main__":
    main()
