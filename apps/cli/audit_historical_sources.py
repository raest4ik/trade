from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.reaction_ready_corpus.source_audit import write_source_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the verified source audit for the reaction-ready corpus universe."
    )
    parser.add_argument(
        "--output",
        default="artifacts/historical-news-v1/source-audit.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = write_source_audit(Path(args.output))
    print(
        json.dumps(
            {
                "output": args.output,
                "audited_sources": len(payload["sources"]),
                "approved_real_source_codes": payload["approved_real_source_codes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
