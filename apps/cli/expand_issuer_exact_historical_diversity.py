from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from src.issuer_exact_historical_diversity_expansion.application import (
    ActiveMarketClient,
    run_issuer_exact_historical_diversity_expansion,
)
from src.issuer_exact_historical_diversity_expansion.domain import (
    ARTIFACT_VERSION,
    DEFAULT_READINESS_AUDIT_ROOT,
)
from src.tinvest_market.client import TInvestContour, TInvestReadOnlyClient
from src.tinvest_market.config import load_readonly_token


def run(args: argparse.Namespace) -> int:
    market_client_factory: Callable[[], ActiveMarketClient] | None = (
        _readonly_market_client_factory if args.live_readonly else None
    )
    manifest = run_issuer_exact_historical_diversity_expansion(
        readiness_root=Path(args.readiness_dir),
        output_root=Path(args.output_dir),
        base_main_sha=args.base_main_sha,
        git_sha=_git_sha(),
        created_at=(
            datetime.fromisoformat(args.created_at) if args.created_at is not None else None
        ),
        market_client_factory=market_client_factory,
        extra_cache_roots=tuple(Path(path) for path in args.extra_cache_dir),
        universe_root=Path(args.universe_dir),
    )
    print(
        json.dumps(
            {
                "ARTIFACT_SHA": manifest["ARTIFACT_SHA"],
                "SELECTED_SOURCES": manifest["SELECTED_SOURCES"],
                "NEW_EXACT_EVENTS_COLLECTED": manifest["NEW_EXACT_EVENTS_COLLECTED"],
                "NEW_HISTORICAL_EVENTS_COLLECTED": manifest["NEW_HISTORICAL_EVENTS_COLLECTED"],
                "NEW_FEATURE_READY_EVENTS": manifest["NEW_FEATURE_READY_EVENTS"],
                "DIVERSITY_DECISION": manifest["DIVERSITY_DECISION"],
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand issuer-originated strict-EXACT historical source diversity."
    )
    parser.add_argument("--readiness-dir", default=DEFAULT_READINESS_AUDIT_ROOT)
    parser.add_argument("--output-dir", default=f"artifacts/{ARTIFACT_VERSION}")
    parser.add_argument("--universe-dir", default="artifacts/tinvest-market-universe-raw-v1")
    parser.add_argument("--base-main-sha", required=True)
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--extra-cache-dir", action="append", default=[])
    parser.add_argument(
        "--live-readonly",
        action="store_true",
        help="Use existing TINVEST_READONLY_TOKEN for bounded production read-only candles.",
    )
    return parser


def _readonly_market_client_factory() -> TInvestReadOnlyClient:
    credentials: dict[str, Any] = {
        "token": load_readonly_token(),
        "contour": TInvestContour.READONLY_PRODUCTION,
        "max_retries": 1,
    }
    return TInvestReadOnlyClient(**credentials)


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
