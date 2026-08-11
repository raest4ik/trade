from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import UUID

from apps.cli.historical_news_common import parse_range_datetime
from src.corpus_quality.application import (
    load_publication_time_records,
    source_acceptance_evidence,
)
from src.corpus_quality.domain import ShadowPrediction
from src.corpus_quality.reporting import write_quality_artifacts
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def run(args: argparse.Namespace) -> int:
    date_from = parse_range_datetime(args.date_from, end_of_day=False)
    date_to = parse_range_datetime(args.date_to, end_of_day=True)
    feature_ids = _feature_news_ids(Path(args.feature_corpus))
    engine = create_engine(get_settings().database_url)
    try:
        async with create_session_factory(engine)() as session:
            records = await load_publication_time_records(
                session,
                date_from=date_from,
                date_to=date_to,
                feature_news_ids=feature_ids,
                limit=args.limit,
            )
    finally:
        await engine.dispose()
    baseline_path = Path(args.output) / "rosn-baseline.json"
    baseline_records = _baseline_records(records, baseline_path, args.baseline_count)
    shadow = _shadow_predictions(Path(args.shadow_predictions))
    rss_audit = {
        "source_url": "https://www.rosneft.com/press/releases/rss/",
        "payload_type": "application/xml",
        "payload_shape": "RSS item title + HTML description excerpt + issuer-owned link",
        "full_text_in_feed": False,
        "content_assessment": "CONTENT_TOO_THIN_FOR_EVENT_EXTRACTION",
        "issuer_owned_release_link": True,
        "sample_link": "https://www.rosneft.com/press/releases/item/224187/",
        "controlled_observation": {
            "rss_status": args.rss_status,
            "rss_bytes": args.rss_bytes,
            "release_page_status": args.release_page_status,
            "release_page_bytes": args.release_page_bytes,
            "feed_excerpt_present_on_release_page": args.feed_excerpt_present,
        },
        "article_enrichment_implemented": False,
        "article_enrichment_blocker": (
            "Current EXCERPT_ALLOWED policy does not establish permission to persist issuer full "
            "text; no usage-policy assumption was made."
        ),
    }
    paths = write_quality_artifacts(
        Path(args.output),
        Path(args.v2_output),
        baseline_records=baseline_records,
        cumulative_records=records,
        source_evidence=source_acceptance_evidence(),
        batch_001_gold_path=Path(args.batch_001_gold),
        shadow_predictions=shadow,
        rss_audit=rss_audit,
    )
    print(
        json.dumps(
            {
                "baseline_records": len(baseline_records),
                "cumulative_records": len(records),
                "shadow_predictions": len(shadow),
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build real-corpus UNKNOWN diagnostics and v2 readiness artifacts."
    )
    parser.add_argument("--from", dest="date_from", required=True)
    parser.add_argument("--to", dest="date_to", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--baseline-count", type=int, default=10)
    parser.add_argument("--rss-status", type=int)
    parser.add_argument("--rss-bytes", type=int)
    parser.add_argument("--release-page-status", type=int)
    parser.add_argument("--release-page-bytes", type=int)
    parser.add_argument("--feed-excerpt-present", action="store_true")
    parser.add_argument("--output", default="artifacts/corpus-quality-v1")
    parser.add_argument("--v2-output", default="artifacts/reaction-ready-corpus-v2")
    parser.add_argument(
        "--feature-corpus",
        default="artifacts/reaction-ready-corpus-v1/corpus.jsonl",
    )
    parser.add_argument(
        "--batch-001-gold",
        default="artifacts/seed/batch-001-gold-v1-reviewed-only.jsonl",
    )
    parser.add_argument(
        "--shadow-predictions",
        default="artifacts/corpus-quality-v1/qwen-shadow/predictions.jsonl",
    )
    return parser


def _feature_news_ids(path: Path) -> set[UUID]:
    if not path.exists():
        return set()
    identifiers: set[UUID] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        identifiers.add(UUID(str(payload["metadata"]["news_id"])))
    return identifiers


def _baseline_records(records: list[Any], path: Path, baseline_count: int) -> list[Any]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        baseline_ids = {UUID(value) for value in payload["news_ids"]}
        selected = [item for item in records if item.news_id in baseline_ids]
    else:
        selected = [item for item in records if item.ticker == "ROSN"][:baseline_count]
    if len(selected) != baseline_count:
        raise ValueError("frozen ROSN baseline cannot be reconstructed from the database")
    return selected


def _shadow_predictions(path: Path) -> list[ShadowPrediction]:
    if not path.exists():
        return []
    predictions: list[ShadowPrediction] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        successful = payload.get("status") != "FAILED"
        predictions.append(
            ShadowPrediction(
                news_id=UUID(str(payload["news_id"])),
                primary_event=str(payload.get("primary_event_type", "UNKNOWN")),
                event_count=len(payload.get("events", [])),
                fact_count=len(payload.get("financial_facts", [])),
                successful=successful,
            )
        )
    return predictions


def main() -> None:
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
