from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ai_trading_agent_v1.application import git_sha
from src.paper_trading_operation_v1.reporting import (
    ARTIFACT_VERSION,
    build_sample_operations,
    write_operation_artifact,
)


def run(args: argparse.Namespace) -> int:
    head_sha = git_sha()
    work_root = Path(args.work_root)
    sample = build_sample_operations(work_root=work_root, code_sha=head_sha)
    manifest = write_operation_artifact(
        output_root=Path(args.output_root),
        base_main_sha=args.base_main_sha,
        head_sha=head_sha,
        sample=sample,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=ARTIFACT_VERSION)
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--output-root", default=f"artifacts/{ARTIFACT_VERSION}")
    parser.add_argument("--work-root", default=f".tmp/{ARTIFACT_VERSION}-work")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
