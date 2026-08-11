from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from src.free_historical_data.registry import free_source_audits, source_volumes
from src.free_historical_data.reporting import (
    CumulativeCorpusState,
    PilotState,
    write_zero_cost_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the zero-cost historical source audit and readiness reports."
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/free-historical-data-v1",
    )
    parser.add_argument(
        "--preflight-corpus",
        default="artifacts/fresh-real-corpus-v1/preflight-corpus.jsonl",
    )
    parser.add_argument(
        "--reaction-manifest",
        default="artifacts/reaction-ready-corpus-v3/manifest.json",
    )
    parser.add_argument(
        "--historical-quality-report",
        default="artifacts/historical-news-v1/data-quality-report.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cumulative = _load_cumulative_state(
        Path(args.preflight_corpus),
        Path(args.reaction_manifest),
        Path(args.historical_quality_report),
    )
    paths = write_zero_cost_reports(
        Path(args.output_dir),
        audits=free_source_audits(),
        volumes=source_volumes(),
        cumulative=cumulative,
        pilot=PilotState(),
        discovered_items=[],
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, sort_keys=True))


def _load_cumulative_state(
    preflight_path: Path,
    reaction_manifest_path: Path,
    quality_report_path: Path,
) -> CumulativeCorpusState:
    rows = [
        cast("dict[str, Any]", json.loads(line))
        for line in preflight_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    real = [row for row in rows if row.get("source_code") != "ML_FEATURE_SYNTHETIC_SMOKE"]
    reaction = cast(
        "dict[str, Any]", json.loads(reaction_manifest_path.read_text(encoding="utf-8"))
    )
    quality = cast("dict[str, Any]", json.loads(quality_report_path.read_text(encoding="utf-8")))
    timestamps = sorted(str(row["published_at"]) for row in real)
    synthetic_count = int(
        cast("dict[str, int]", quality["by_source"]).get("ML_FEATURE_SYNTHETIC_SMOKE", 0)
    )
    source_distribution = Counter(
        {
            source: count
            for source, count in cast("dict[str, int]", quality["by_source"]).items()
            if source != "ML_FEATURE_SYNTHETIC_SMOKE"
        }
    )
    ticker_distribution = Counter(
        {
            ticker: count
            for ticker, count in cast("dict[str, int]", quality["by_ticker"]).items()
            if ticker != "SBER"
        }
    )
    total_real = int(quality["total_candidates"]) - synthetic_count
    real_exact = (
        int(cast("dict[str, int]", quality["by_timestamp_quality"])["EXACT"]) - synthetic_count
    )
    matched = int(quality["matched_count"]) - synthetic_count
    ambiguous = int(quality["ambiguous_count"])
    months = {str(row["published_at"])[:7] for row in real}
    return CumulativeCorpusState(
        real=total_real,
        real_exact=real_exact,
        matched=matched - ambiguous,
        ambiguous=ambiguous,
        unmatched=total_real - matched,
        reaction_ready=int(reaction["reaction_ready"]),
        feature_ready=int(reaction["feature_ready"]),
        ticker_distribution=dict(sorted(ticker_distribution.items())),
        source_distribution=dict(sorted(source_distribution.items())),
        date_from=timestamps[0] if timestamps else None,
        date_to=timestamps[-1] if timestamps else None,
        month_count=len(months),
    )


if __name__ == "__main__":
    main()
