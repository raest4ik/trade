from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.free_live_issuer_accumulation.application import load_shadow_corpus_stats


def run(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            load_shadow_corpus_stats(Path(args.artifact_dir)), ensure_ascii=False, sort_keys=True
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build no-outcome stats for live shadow corpus.")
    parser.add_argument("--artifact-dir", default="artifacts/free-live-issuer-accumulation-v1")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
