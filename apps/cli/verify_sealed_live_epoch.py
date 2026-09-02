from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.free_live_issuer_accumulation.application import verify_sealed_live_epoch


def run(args: argparse.Namespace) -> int:
    result = verify_sealed_live_epoch(Path(args.artifact_dir))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["sealed_epoch_verified"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify sealed live epoch counters and shadow rows."
    )
    parser.add_argument("--artifact-dir", default="artifacts/free-live-issuer-accumulation-v1")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
