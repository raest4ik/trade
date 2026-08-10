from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from apps.cli.historical_news_common import add_ingestion_arguments, ingest_from_client
from src.historical_news.domain.enums import HistoricalNewsSourceKind
from src.historical_news.infrastructure.disclosure_archive import (
    InterfaxDisclosureArchiveAdapter,
)
from src.historical_news.infrastructure.local_archive import LocalArchiveNewsSource


async def run(args: argparse.Namespace) -> int:
    source_kind = HistoricalNewsSourceKind(args.source_kind)
    adapter = (
        InterfaxDisclosureArchiveAdapter(Path(args.input), max_items=100_000)
        if source_kind
        in {
            HistoricalNewsSourceKind.DISCLOSURE_ARCHIVE,
            HistoricalNewsSourceKind.INTERFAX_DISCLOSURE,
        }
        else LocalArchiveNewsSource(Path(args.input), max_items=100_000)
    )
    return await ingest_from_client(args, client=adapter, source_kind=source_kind)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a bounded, authorized historical-news JSONL archive."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--source-kind",
        choices=[
            HistoricalNewsSourceKind.LOCAL_ARCHIVE.value,
            HistoricalNewsSourceKind.MANUAL_RESEARCH.value,
            HistoricalNewsSourceKind.DISCLOSURE_ARCHIVE.value,
            HistoricalNewsSourceKind.INTERFAX_DISCLOSURE.value,
            HistoricalNewsSourceKind.ISSUER_JSON.value,
        ],
        default=HistoricalNewsSourceKind.LOCAL_ARCHIVE.value,
    )
    add_ingestion_arguments(parser)
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
