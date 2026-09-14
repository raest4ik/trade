from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ai_trading_agent_v1.application import git_sha
from src.current_moex_tradability_v1.reporting import (
    DEFAULT_OUTPUT_ROOT,
    build_tradability_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="build current MOEX tradability V1 artifact")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--code-sha", default=None)
    args = parser.parse_args()
    manifest = build_tradability_artifact(args.output_root, args.code_sha or git_sha())
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
