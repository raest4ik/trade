from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.exact_event_live_official_collection.http_client import FetchResult
from src.free_live_issuer_accumulation.application import (
    DEFAULT_HISTORICAL_TICKER_SUMMARY_PATH,
    collect_live_issuer_news,
    read_registry,
)
from src.free_live_issuer_accumulation.domain import DEFAULT_SOURCE_REGISTRY_PATH


def run(args: argparse.Namespace) -> int:
    client = _FixtureClient(Path(args.source_registry)) if args.fixture_smoke else None
    manifest = collect_live_issuer_news(
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        registry_path=Path(args.source_registry),
        historical_ticker_summary_path=Path(args.historical_ticker_summary),
        state_path=Path(args.state_file) if args.state_file else None,
        client=client,
        created_at=datetime.fromisoformat(args.created_at) if args.created_at else None,
        max_sources=args.max_sources,
        source_id=args.source,
    )
    print(json.dumps(_summary(manifest), ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect bounded free live issuer shadow corpus.")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--source-registry", default=DEFAULT_SOURCE_REGISTRY_PATH)
    parser.add_argument(
        "--historical-ticker-summary", default=DEFAULT_HISTORICAL_TICKER_SUMMARY_PATH
    )
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--output-dir", default="artifacts/free-live-issuer-accumulation-v1")
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--max-sources", type=int, default=5)
    parser.add_argument("--source", default=None)
    parser.add_argument("--once", action="store_true", help="Run one bounded polling pass.")
    parser.add_argument(
        "--fixture-smoke",
        action="store_true",
        help="Use deterministic fixture RSS instead of network access.",
    )
    return parser


def _summary(manifest: dict[str, object]) -> dict[str, object]:
    keys = (
        "ARTIFACT_SHA",
        "STRICT_ANSWER",
        "EVENTS_COLLECTED",
        "RAW_SNAPSHOTS_FROZEN",
        "DUPLICATES_ENCOUNTERED",
        "SEMANTIC_READY_EVENTS",
        "UNKNOWN_EVENTS",
        "UNKNOWN_RATE",
        "PRE_EVENT_FEATURE_READY_EVENTS",
        "LIVE_POST_EVENT_PRICE_READS",
        "LIVE_TARGETS_COMPUTED",
        "LIVE_OUTCOMES_READ",
        "LIVE_DIVERSITY_STATUS",
        "LIVE_DIVERSITY_ACCUMULATION_STATUS",
        "SOURCE_READY",
        "NEW_ITEM_OBSERVED",
        "READY_ISSUER_TICKERS",
        "NEW_TICKERS_RELATIVE_TO_HISTORICAL_7",
        "FREE_BLOCKER",
    )
    return {key: manifest[key] for key in keys}


class _FixtureClient:
    def __init__(self, registry_path: Path) -> None:
        self._by_url = {
            source.discovery_url: _rss_fixture(source.ticker, index)
            for index, source in enumerate(read_registry(registry_path).sources, start=1)
            if source.enabled
        }

    def get(self, url: str) -> FetchResult:
        body = self._by_url[url]
        return FetchResult(
            request_url=url,
            final_url=url,
            status=200,
            content_type="application/rss+xml",
            body=body,
            redirects=0,
            redirect_chain=(),
            blocker=None,
        )


def _rss_fixture(ticker: str, index: int) -> bytes:
    published = datetime(2026, 9, 2, 9 + index, 15, tzinfo=UTC)
    local = published.strftime("%a, %d %b %Y %H:%M:%S +0000")
    title = f"{ticker} announces results under IFRS"
    content = f"{ticker} announces its results under IFRS and confirms guidance."
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>{ticker} official feed</title>
    <item>
      <title>{title}</title>
      <description>{content}</description>
      <content:encoded><![CDATA[{content}]]></content:encoded>
      <link>https://example.invalid/{ticker.lower()}/2026-09-02-{index}</link>
      <guid>{ticker}-fixture-{index}</guid>
      <pubDate>{local}</pubDate>
    </item>
  </channel>
</rss>
""".encode()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
