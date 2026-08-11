from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, cast

from src.daily_corpus.application import build_daily_corpus
from src.daily_corpus.reporting import write_daily_corpus_reports
from src.daily_corpus.source_registry import daily_source_verifications
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    engine = create_engine(get_settings().database_url)
    try:
        async with create_session_factory(engine)() as session:
            result = await build_daily_corpus(session)
    finally:
        await engine.dispose()
    paths = write_daily_corpus_reports(
        Path(args.output_dir),
        result=result,
        verifications=daily_source_verifications(),
        intraday=_load_intraday(Path(args.reaction_manifest)),
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe daily reaction and feature corpus reports."
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/free-daily-historical-v1",
    )
    parser.add_argument(
        "--reaction-manifest",
        default="artifacts/reaction-ready-corpus-v3/manifest.json",
    )
    return parser


def _load_intraday(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"real_exact": 0, "reaction_ready": 0, "feature_ready": 0}
    payload = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    return {
        "real_exact": int(payload.get("real_exact", payload.get("total_real", 40))),
        "reaction_ready": int(payload.get("reaction_ready", 0)),
        "feature_ready": int(payload.get("feature_ready", 0)),
    }


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
