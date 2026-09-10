from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ai_trading_agent_v1.application import git_sha
from src.production_dry_run_burnin_v1.domain import BURNIN_ARTIFACT_VERSION
from src.production_dry_run_burnin_v1.reporting import build_burnin_artifact


def run(args: argparse.Namespace) -> int:
    manifest = build_burnin_artifact(
        output_root=Path(args.output_root),
        work_root=Path(args.work_root),
        base_main_sha=args.base_main_sha,
        head_sha=args.head_sha or git_sha(),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=BURNIN_ARTIFACT_VERSION)
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--head-sha", default=None)
    parser.add_argument("--output-root", default=f"artifacts/{BURNIN_ARTIFACT_VERSION}")
    parser.add_argument("--work-root", default=f".tmp/{BURNIN_ARTIFACT_VERSION}-work")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
