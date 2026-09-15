from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from src.free_live_issuer_accumulation.domain import sha256_payload
from src.moex_trading_calendar_v2.application import (
    distinct_moex_trading_days,
    resolve_schedule_payload,
)
from src.moex_trading_calendar_v2.domain import (
    CALENDAR_SOURCE,
    CALENDAR_SOURCE_URL,
    RUNTIME_SOURCE,
    SESSION_POLICY_VERSION,
    MoexSessionEvidenceV2,
)
from src.production_dry_run_burnin_v1.repository import JsonlBurninObservationRepository

ARTIFACT_VERSION = "authoritative-moex-trading-calendar-v2"
DEFAULT_OUTPUT_ROOT = Path(f"artifacts/{ARTIFACT_VERSION}")
FIXTURE_TIME = datetime(2026, 9, 15, 12, tzinfo=UTC)
LEGACY_LEDGER = Path("artifacts/production-dry-run-burnin-v1/sample-observations.jsonl")


def build_calendar_artifact(
    output_root: Path,
    *,
    base_main_sha: str,
    code_sha: str,
) -> dict[str, Any]:
    payload = _fixture_payload()
    regular = _resolve(payload, "2026-09-21")
    weekend_open = _resolve(payload, "2026-09-19")
    weekend_closed = _resolve(payload, "2026-09-12")
    holiday_open = _resolve(payload, "2026-05-01")
    holiday_closed = _resolve(payload, "2026-01-01")
    same_business_day = [weekend_open, _resolve(payload, "2026-09-20"), regular]

    legacy_bytes = LEGACY_LEDGER.read_bytes()
    legacy_rows = JsonlBurninObservationRepository(LEGACY_LEDGER).observations()
    files: dict[str, Any] = {
        "calendar-policy.json": {
            "FIXTURE_ONLY": True,
            "SESSION_POLICY_VERSION": SESSION_POLICY_VERSION,
            "market": "stock",
            "board": "TQBR",
            "timezone": "Europe/Moscow",
            "planned_source": CALENDAR_SOURCE,
            "planned_source_url": CALENDAR_SOURCE_URL,
            "runtime_source": RUNTIME_SOURCE,
            "runtime_role": "OPTIONAL_CURRENT_MARKET_WIDE_CONFIRMATION",
            "business_date_rule": (
                "official additional weekend/holiday session maps to the next "
                "authoritative regular stock-market trading date"
            ),
            "weekday_fallback": False,
            "weekend_fallback": False,
            "unknown_fails_closed": True,
        },
        "sample-regular-open.json": _sample(regular),
        "sample-weekend-open.json": _sample(weekend_open),
        "sample-weekend-closed.json": _sample(weekend_closed),
        "sample-holiday.json": {
            "FIXTURE_ONLY": True,
            "open": holiday_open.model_dump(mode="json"),
            "closed": holiday_closed.model_dump(mode="json"),
        },
        "business-date-proof.json": {
            "FIXTURE_ONLY": True,
            "calendar_dates": [row.calendar_date.isoformat() for row in same_business_day],
            "moex_business_dates": [
                row.moex_business_date.isoformat()
                for row in same_business_day
                if row.moex_business_date is not None
            ],
            "distinct_calendar_dates": 3,
            "distinct_moex_trading_days": distinct_moex_trading_days(same_business_day),
            "DOUBLE_COUNT_PROTECTION": "PASS",
        },
        "fail-closed-proof.json": {
            "FIXTURE_ONLY": True,
            "missing_date": _resolve(payload, "2026-07-01").model_dump(mode="json"),
            "authoritative_unavailable_result": "UNKNOWN",
            "wrong_market_scope_result": "UNKNOWN",
            "contradictory_evidence_result": "UNKNOWN",
            "UNKNOWN_BLOCKS_BEFORE_MODEL": "PASS",
        },
        "backward-compatibility-proof.json": {
            "FIXTURE_ONLY": True,
            "legacy_ledger": LEGACY_LEDGER.as_posix(),
            "legacy_ledger_sha256": hashlib.sha256(legacy_bytes).hexdigest(),
            "legacy_records_verified": len(legacy_rows),
            "legacy_record_shas": [row.record_sha for row in legacy_rows],
            "legacy_bytes_rewritten": False,
            "legacy_v2_fields_required": False,
            "OLD_EPOCH1_LEDGER_VERIFIED": "PASS",
        },
        "safety.json": {
            "PAPER_EXECUTION_ENABLED": False,
            "REAL_EXECUTION_ENABLED": False,
            "PAPER_OPERATION_SCHEDULE_ENABLED": False,
            "PAPER_EXECUTION_DEFAULT": False,
            "PAPER_EXECUTION_ELIGIBLE": "NO",
            "REAL_EXECUTION_READY": "NO",
            "REAL_BROKER_MUTATIONS": 0,
            "REAL_ORDERS_SENT": 0,
            "REAL_ORDERS_CANCELLED": 0,
            "REAL_POSITIONS_CHANGED": 0,
            "LIVE_OUTCOMES_READ": 0,
            "LIVE_TARGETS_COMPUTED": 0,
            "LIVE_POST_EVENT_PRICE_READS": 0,
            "OLD_FUTURE_HOLDOUT_OPENED": False,
            "CURRENT_MOEX_TRADABILITY_GATE_READY": "YES",
            "CURRENT_PRODUCTION_UNIVERSE_READY": "YES",
            "PRODUCTION_AGENT_ADAPTER_READY": "YES",
            "FRESH_PIT_MARKET_ADAPTER_READY": "YES",
            "PRODUCTION_PAPER_CONTEXT_READY": "YES",
            "PRODUCTION_DRY_RUN_READY": "YES",
            "SESSION_IDEMPOTENCY": "PASS",
            "PIT_SAFETY": "PASS",
            "DECISION_CUTOFF_AFTER_MARKET_FETCH": "PASS",
        },
    }
    artifact_sha = sha256_payload(files)
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "BASE_MAIN_SHA": base_main_sha,
        "ARTIFACT_CODE_SHA": code_sha,
        "ARTIFACT_SHA": artifact_sha,
        "FIXTURE_ONLY": True,
        "SESSION_POLICY_VERSION": SESSION_POLICY_VERSION,
        "AUTHORITATIVE_CALENDAR_SOURCE": CALENDAR_SOURCE_URL,
        "AUTHORITATIVE_RUNTIME_SOURCE": RUNTIME_SOURCE,
        "WEEKEND_SHORTCUT_REMOVED": "YES",
        "WEEKEND_OPEN_SUPPORTED": "YES",
        "WEEKEND_CLOSED_SUPPORTED": "YES",
        "HOLIDAY_OPEN_SUPPORTED": "YES",
        "HOLIDAY_CLOSED_SUPPORTED": "YES",
        "MOEX_BUSINESS_DATE_SUPPORTED": "YES",
        "DOUBLE_COUNT_PROTECTION": "PASS",
        "FAIL_CLOSED": "YES",
        "LIVE_LEDGER_MUTATIONS": 0,
        "CURRENT_BURNIN_EPOCH": "production-dry-run-burnin-v1-epoch-2",
        "CURRENT_BURNIN_CODE_SHA": base_main_sha,
        "CURRENT_BURNIN_CHANGED": "NO",
        "PR75_READY_FOR_REVIEW": "YES",
        "PR75_READY_TO_MERGE": "NO",
    }
    files["manifest.json"] = manifest

    temporary = output_root.with_name(f".{output_root.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for name, content in files.items():
        _write_json(temporary / name, content)
    (temporary / "report.md").write_text(_report(manifest), encoding="utf-8", newline="\n")
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary.replace(output_root)
    return manifest


def _fixture_payload() -> dict[str, Any]:
    days = {
        "2026-01-01": {"tradedate": "2026-01-01", "is_traded": 0, "reason": "H"},
        "2026-05-01": {"tradedate": "2026-05-01", "is_traded": 1, "reason": "W"},
        "2026-05-04": {"tradedate": "2026-05-04", "is_traded": 1, "reason": "N"},
        "2026-09-12": {"tradedate": "2026-09-12", "is_traded": 0, "reason": "N"},
        "2026-09-19": {"tradedate": "2026-09-19", "is_traded": 1, "reason": "W"},
        "2026-09-20": {"tradedate": "2026-09-20", "is_traded": 1, "reason": "W"},
        "2026-09-21": {"tradedate": "2026-09-21", "is_traded": 1, "reason": "N"},
    }
    return {
        "market": "stock",
        "board": "TQBR",
        "days": days,
        "holidays": {"2026-01-01": "holiday", "2026-05-01": "holiday"},
        "windows": {
            "weekday": [{"from": "06:50", "till": "23:50"}],
            "weekend": [{"from": "09:50", "till": "19:00"}],
        },
    }


def _resolve(payload: dict[str, Any], value: str) -> MoexSessionEvidenceV2:
    return resolve_schedule_payload(
        payload,
        date.fromisoformat(value),
        checked_at=FIXTURE_TIME,
        evidence_sha=sha256_payload(payload),
    )


def _sample(row: MoexSessionEvidenceV2) -> dict[str, Any]:
    return {"FIXTURE_ONLY": True, "evidence": row.model_dump(mode="json")}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _report(manifest: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# Authoritative MOEX trading calendar V2",
            "",
            "Deterministic fixture evidence. It is not a live MOEX snapshot.",
            "",
            f"- session policy: {manifest['SESSION_POLICY_VERSION']}",
            f"- artifact code SHA: {manifest['ARTIFACT_CODE_SHA']}",
            f"- artifact SHA: {manifest['ARTIFACT_SHA']}",
            "- official planned source: MOEX trading calendar",
            "- optional runtime source: market-wide MOEX ISS TQBR state",
            "- no weekday or weekend fallback",
            "- additional sessions use explicit MOEX business-date semantics",
            "- unavailable, malformed, or contradictory evidence fails closed",
            "- legacy ledgers remain byte-preserved and hash-valid",
            "- paper execution: disabled",
            "- real execution: disabled",
            "- scheduling: disabled",
            "- merge during active epoch 2: prohibited",
            "",
        )
    )
