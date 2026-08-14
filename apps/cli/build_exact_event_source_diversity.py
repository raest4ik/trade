from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from collections.abc import Awaitable
from datetime import date
from pathlib import Path
from typing import Any, cast

from src.exact_event_corpus.domain import ExactEvent
from src.exact_event_diversity.application import (
    build_diversity_dataset,
    build_diversity_source_registry,
)
from src.exact_event_diversity.sources import (
    OfficialSourceProfile,
    acquire_embedded_app_state,
    acquire_moex_rss,
    acquire_tbank_public_news,
    acquire_vk_next_state,
    acquire_x5_wordpress,
)
from src.tinvest_market.client import TInvestContour, TInvestReadOnlyClient

DEFAULT_OUTPUT = Path("artifacts/exact-event-market-dataset-v2")


async def run(args: argparse.Namespace) -> int:
    token = os.environ.get("TINVEST_READONLY_TOKEN", "")
    if not token:
        raise ValueError("TINVEST_READONLY_TOKEN_REQUIRED")
    mapping_path = Path(args.instrument_mapping)
    registry = build_diversity_source_registry(mapping_path, Path(args.v1_source_registry))
    by_ticker = {item.ticker: item for item in registry}
    source_cache = Path(args.output) / "raw-source-cache"
    date_from = date.fromisoformat(args.date_from)
    date_to = date.fromisoformat(args.date_to)
    events: list[ExactEvent] = []
    source_diagnostics: list[dict[str, object]] = []

    async def collect(source_family: str, operation: Awaitable[list[ExactEvent]]) -> None:
        try:
            rows = await operation
        except (RuntimeError, ValueError) as exc:
            source_diagnostics.append(
                {
                    "source_family": source_family,
                    "status": "FAILED_CLOSED",
                    "sanitized_error": str(exc).split(":", maxsplit=1)[0],
                }
            )
            return
        events.extend(rows)
        source_diagnostics.append(
            {"source_family": source_family, "status": "SUCCESS", "exact_events": len(rows)}
        )

    await collect(
        "X5_OFFICIAL_WORDPRESS_REST",
        acquire_x5_wordpress(
            _profile(
                by_ticker,
                "X5",
                "X5_OFFICIAL_WORDPRESS_EXACT",
                "https://www.x5.ru/wp-json/wp/v2/news",
                "www.x5.ru",
                "WordPress REST news.date_gmt (UTC)",
            ),
            date_from=date_from,
            date_to=date_to,
            item_limit=args.per_source_limit,
            cache_dir=source_cache / "X5_OFFICIAL_WORDPRESS_REST",
        ),
    )
    await collect(
        "VK_OFFICIAL_NEXT_PUBLIC_STATE",
        acquire_vk_next_state(
            _profile(
                by_ticker,
                "VKCO",
                "VK_OFFICIAL_NEXT_STATE_EXACT",
                "https://vk.company/ru/press/releases/",
                "vk.company",
                "public __NEXT_DATA__.pageProps.publications.pub_date",
            ),
            date_from=date_from,
            date_to=date_to,
            item_limit=min(30, args.per_source_limit),
            cache_dir=source_cache / "VK_OFFICIAL_NEXT_PUBLIC_STATE",
        ),
    )
    await collect(
        "TBANK_OFFICIAL_PUBLIC_NEWS_API",
        acquire_tbank_public_news(
            _profile(
                by_ticker,
                "T",
                "TBANK_OFFICIAL_PUBLIC_NEWS_EXACT",
                "https://cfg.tbank.ru/about/public/api/news/platform/v1/getArticles",
                "cfg.tbank.ru",
                "public getArticles response.items.publishedAt",
            ),
            date_from=date_from,
            date_to=date_to,
            item_limit=args.per_source_limit,
            cache_dir=source_cache / "TBANK_OFFICIAL_PUBLIC_NEWS_API",
        ),
    )
    await collect(
        "NOVABEV_OFFICIAL_APP_STATE",
        acquire_embedded_app_state(
            _profile(
                by_ticker,
                "BELU",
                "NOVABEV_OFFICIAL_APP_STATE_EXACT",
                "https://novabev.com/en/investors/news/",
                "novabev.com",
                "embedded App.news.items.activeFrom Unix epoch seconds",
            ),
            date_from=date_from,
            date_to=date_to,
            item_limit=min(100, args.per_source_limit),
            cache_dir=source_cache / "NOVABEV_OFFICIAL_APP_STATE",
        ),
    )
    await collect(
        "MOEX_OFFICIAL_SHAREHOLDER_RSS",
        acquire_moex_rss(
            _profile(
                by_ticker,
                "MOEX",
                "MOEX_OFFICIAL_SHAREHOLDER_RSS_EXACT",
                "https://www.moex.com/export/news.aspx?cat=120",
                "www.moex.com",
                "RSS item pubDate with explicit +0300 offset",
            ),
            date_from=date_from,
            date_to=date_to,
            item_limit=min(100, args.per_source_limit),
            cache_dir=source_cache / "MOEX_OFFICIAL_SHAREHOLDER_RSS",
        ),
    )
    await collect(
        "MOEX_OFFICIAL_ISSUER_NOTICE_RSS:SMLT",
        acquire_moex_rss(
            _profile(
                by_ticker,
                "SMLT",
                "MOEX_OFFICIAL_SMLT_NOTICE_RSS_EXACT",
                "https://www.moex.com/export/news.aspx?cat=100",
                "www.moex.com",
                "RSS item pubDate with explicit +0300 offset",
            ),
            date_from=date_from,
            date_to=date_to,
            item_limit=min(20, args.per_source_limit),
            cache_dir=source_cache / "MOEX_OFFICIAL_ISSUER_NOTICE_RSS",
            required_phrases=('Группа компаний "Самолет', 'ГК "Самолет', "SMLT"),
            rejected_phrases=("ценового коридора", "дискретный аукцион"),
        ),
    )
    await collect(
        "MOEX_OFFICIAL_ISSUER_NOTICE_RSS:VTBR",
        acquire_moex_rss(
            _profile(
                by_ticker,
                "VTBR",
                "MOEX_OFFICIAL_VTBR_NOTICE_RSS_EXACT",
                "https://www.moex.com/export/news.aspx?cat=100",
                "www.moex.com",
                "RSS item pubDate with explicit +0300 offset",
            ),
            date_from=date_from,
            date_to=date_to,
            item_limit=min(20, args.per_source_limit),
            cache_dir=source_cache / "MOEX_OFFICIAL_ISSUER_NOTICE_RSS_VTBR",
            required_phrases=("Банк ВТБ", "VTBR"),
            rejected_phrases=("ценового коридора", "дискретный аукцион"),
        ),
    )
    async with TInvestReadOnlyClient(
        token=token,
        contour=TInvestContour.READONLY_PRODUCTION,
    ) as client:
        report = await build_diversity_dataset(
            new_events=events,
            registry=registry,
            client=client,
            v1_dir=Path(args.v1_dataset),
            output_dir=Path(args.output),
            benchmark_instrument_uid=_instrument_uid(mapping_path, "IMOEX"),
            git_sha=_git_sha(),
            source_acquisition_diagnostics=source_diagnostics,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand exact issuer diversity without training or evaluating a model."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--date-from", default="2025-01-01")
    parser.add_argument("--date-to", default=date.today().isoformat())
    parser.add_argument("--per-source-limit", type=int, default=100)
    parser.add_argument(
        "--instrument-mapping",
        default="artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json",
    )
    parser.add_argument("--v1-dataset", default="artifacts/exact-event-market-dataset-v1")
    parser.add_argument(
        "--v1-source-registry",
        default="artifacts/exact-event-market-dataset-v1/source-registry.jsonl",
    )
    return parser


def _profile(
    by_ticker: dict[str, Any],
    ticker: str,
    source_code: str,
    source_url: str,
    allowed_host: str,
    timestamp_field: str,
) -> OfficialSourceProfile:
    identity = by_ticker[ticker]
    return OfficialSourceProfile(
        source_code=source_code,
        ticker=ticker,
        issuer=identity.issuer,
        instrument_uid=identity.instrument_uid,
        source_url=source_url,
        allowed_host=allowed_host,
        timestamp_field=timestamp_field,
    )


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _instrument_uid(mapping_path: Path, ticker: str) -> str:
    payload = cast("dict[str, Any]", json.loads(mapping_path.read_text(encoding="utf-8")))
    instruments = cast("list[dict[str, Any]]", payload["instruments"])
    matches = [str(item["instrument_uid"]) for item in instruments if item["ticker"] == ticker]
    if len(matches) != 1:
        raise ValueError(f"INSTRUMENT_IDENTITY_MISSING_OR_AMBIGUOUS:{ticker}")
    return matches[0]


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
