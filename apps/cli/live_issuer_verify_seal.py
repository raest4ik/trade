from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.free_live_issuer_accumulation.operation import (
    DEFAULT_OPERATION_ARTIFACT_ROOT,
    verify_operation_seal,
)


def run(args: argparse.Namespace) -> int:
    payload = verify_operation_seal(Path(args.artifact_root))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["sealed_epoch_verified"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify free live issuer sealed epoch guards.")
    parser.add_argument("--artifact-root", default=str(DEFAULT_OPERATION_ARTIFACT_ROOT))
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
