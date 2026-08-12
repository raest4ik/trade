from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from src.live_corpus_operations.domain import TELEGRAM_API_POLICY
from src.live_corpus_operations.state import OperationsStateStore


def run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    root = (repo_root / args.artifact_root).resolve()
    health = OperationsStateStore(root).health()
    readiness = _read_optional(repo_root / args.readiness)
    payload = {
        "collector_status": health.get("status", "NEVER_RUN"),
        "last_successful_run": health.get("last_success_at"),
        "source_health": health.get("sources", {}),
        "database_status": health.get("database_status", "UNKNOWN"),
        "moex_status": health.get("moex_status", "UNKNOWN"),
        "daily_feature_ready": int(readiness.get("daily_feature_ready", 0)),
        "intraday_feature_ready": int(readiness.get("intraday_feature_ready", 0)),
        "ticker_count": int(readiness.get("ticker_count", 0)),
        "source_count": int(readiness.get("source_count", 0)),
        "rows_to_100": int(readiness.get("rows_to_100", 100)),
        "rows_to_500": int(readiness.get("rows_to_500", 500)),
        "rows_to_1000": int(readiness.get("rows_to_1000", 1000)),
        "training_gate": readiness.get("training_gate", "TRAINING_BLOCKED"),
        "automatic_training": False,
        "zero_cost": True,
        "telegram_api": TELEGRAM_API_POLICY,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show local live corpus operations health.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--artifact-root",
        default="artifacts/live-corpus-operations-v1",
    )
    parser.add_argument(
        "--readiness",
        default="artifacts/predictive-baseline-v1/readiness.json",
    )
    return parser


def _read_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return {str(key): item for key, item in cast("dict[object, Any]", value).items()}


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
