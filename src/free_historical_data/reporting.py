from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.free_historical_data.domain import (
    DATA_BUDGET,
    FreeSourceAudit,
    FreeSourceStatus,
    SourceVolume,
    readiness_for_feature_rows,
    source_volume_summary,
)

SCHEMA_VERSION = "free-historical-data-v1"


@dataclass(frozen=True, slots=True)
class CumulativeCorpusState:
    real: int
    real_exact: int
    matched: int
    ambiguous: int
    unmatched: int
    reaction_ready: int
    feature_ready: int
    ticker_distribution: dict[str, int]
    source_distribution: dict[str, int]
    date_from: str | None
    date_to: str | None
    month_count: int


@dataclass(frozen=True, slots=True)
class PilotState:
    discovered: int = 0
    new_real_imported: int = 0
    new_exact: int = 0
    matched: int = 0
    ambiguous: int = 0
    unmatched: int = 0
    reaction_ready: int = 0
    feature_ready: int = 0


def write_zero_cost_reports(
    output_dir: Path,
    *,
    audits: tuple[FreeSourceAudit, ...],
    volumes: tuple[SourceVolume, ...],
    cumulative: CumulativeCorpusState,
    pilot: PilotState,
    discovered_items: list[dict[str, Any]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for audit in audits:
        audit.validate()
    observed_statuses = Counter(audit.status.value for audit in audits)
    statuses = {status.value: observed_statuses[status.value] for status in FreeSourceStatus}
    exact_sources = [
        audit.source_code for audit in audits if audit.status == FreeSourceStatus.COMPLIANT_EXACT
    ]
    audit_payload = {
        "schema_version": SCHEMA_VERSION,
        "data_budget": DATA_BUDGET,
        "audited_at": datetime.now(UTC).isoformat(),
        "total_sources": len(audits),
        "status_counts": dict(sorted(statuses.items())),
        "compliant_exact_source_codes": exact_sources,
        "sources": [audit.payload() for audit in audits],
        "paid_services_used": False,
        "purchases_required": False,
        "access_restrictions_bypassed": False,
        "prohibited_scraping_used": False,
    }
    volume_payload = {"schema_version": SCHEMA_VERSION, **source_volume_summary(volumes)}
    pilot_payload = {
        "schema_version": SCHEMA_VERSION,
        **asdict(pilot),
        "limit": 200,
        "selection_order": "source, published_at, source_item_id",
        "uses_rules_predictions": False,
        "uses_qwen_predictions": False,
        "uses_event_class": False,
        "uses_future_returns": False,
        "uses_price_or_volume_movement": False,
        "outcome": (
            "NO_NEW_COMPLIANT_SOURCE_ITEMS"
            if pilot.new_real_imported == 0
            else "BOUNDED_PILOT_COMPLETED"
        ),
    }
    readiness = readiness_for_feature_rows(
        cumulative.feature_ready,
        ticker_count=len(cumulative.ticker_distribution),
        source_count=len(cumulative.source_distribution),
        month_count=cumulative.month_count,
    )
    readiness_payload = {
        "schema_version": SCHEMA_VERSION,
        **readiness,
        "cumulative": asdict(cumulative),
        "reaction_data_readiness": (
            "NOT_READY" if cumulative.reaction_ready < 100 else "PILOT_ONLY"
        ),
        "event_feature_quality": "FROZEN_NOT_EVALUATED_IN_THIS_PR",
    }
    growth_payload = {
        "schema_version": SCHEMA_VERSION,
        "strategy": "free history first, then bounded issuer-owned live collection",
        "current_real_exact": cumulative.real_exact,
        "current_reaction_ready": cumulative.reaction_ready,
        "current_feature_ready": cumulative.feature_ready,
        "new_records_last_run": pilot.new_real_imported,
        "estimated_monthly_growth": "UNKNOWN",
        "estimated_time_to_100": "UNKNOWN",
        "estimated_time_to_500": "UNKNOWN",
        "estimated_time_to_1000": "UNKNOWN",
        "reason": "insufficient repeated live collection observations",
        "free_path_to_100": "poll accepted issuer RSS and re-audit official archives",
        "free_path_to_500": "expand compliant issuers plus accumulate own live archive",
        "free_path_to_1000": "long-term own archive; no paid fallback",
    }
    paths = {
        "source_audits": output_dir / "source-audits.json",
        "source_volume": output_dir / "source-volume.json",
        "discovered_items": output_dir / "discovered-items.jsonl",
        "pilot_manifest": output_dir / "pilot-manifest.json",
        "readiness": output_dir / "readiness.json",
        "growth_plan": output_dir / "growth-plan.json",
    }
    _write_json(paths["source_audits"], audit_payload)
    _write_json(paths["source_volume"], volume_payload)
    _write_jsonl(paths["discovered_items"], discovered_items)
    _write_json(paths["pilot_manifest"], pilot_payload)
    _write_json(paths["readiness"], readiness_payload)
    _write_json(paths["growth_plan"], growth_payload)
    return paths


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
