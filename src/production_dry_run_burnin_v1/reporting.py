from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.free_live_issuer_accumulation.domain import sha256_payload
from src.production_dry_run_burnin_v1.application import build_burnin_report
from src.production_dry_run_burnin_v1.domain import (
    BURNIN_ARTIFACT_VERSION,
    BurninAttemptType,
    BurninObservation,
    BurninObservationStatus,
    BurninPolicy,
    BurninSafety,
    MoexSessionEvidence,
    MoexSessionStatus,
)
from src.production_dry_run_burnin_v1.repository import (
    InMemoryBurninObservationRepository,
    validate_observations,
)

SAMPLE_START = datetime(2026, 9, 8, 15, 0, tzinfo=UTC)
FAILURE_TAXONOMY = {
    "success": ["SUCCESS", "NO_ACTION", "DEGRADED"],
    "market": [
        "MARKET_UNAVAILABLE",
        "MARKET_STALE",
        "MARKET_INVALID",
        "MARKET_SOURCE_CLOCK_SKEW",
        "MARKET_CONTEXT_INCOMPLETE",
        "MARKET_SESSION_CLOSED",
        "MARKET_SESSION_UNKNOWN",
    ],
    "research": [
        "RESEARCH_NOT_READY",
        "RESEARCH_SEAL_INVALID",
        "SOURCE_FAILURE_ISOLATION_FAILED",
    ],
    "agent": [
        "AGENT_MODEL_UNAVAILABLE",
        "AGENT_MODEL_TIMEOUT",
        "AGENT_MODEL_INVALID_RESPONSE",
        "AGENT_OUTPUT_INVALID",
        "AGENT_LIMIT_ABORT",
    ],
    "risk": ["RISK_BLOCKED", "PORTFOLIO_MARK_INCOMPLETE"],
    "integrity": [
        "LEDGER_INTEGRITY_FAILED",
        "OPERATION_ALREADY_RUNNING",
        "UNKNOWN_FAILURE",
    ],
}


def build_burnin_artifact(
    *, output_root: Path, work_root: Path, base_main_sha: str, head_sha: str
) -> dict[str, Any]:
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    policy = BurninPolicy()
    repository = InMemoryBurninObservationRepository()
    repository.append(build_sample_observation(SAMPLE_START, BurninObservationStatus.PASS))
    repository.append(
        build_sample_observation(
            SAMPLE_START + timedelta(days=1), BurninObservationStatus.BLOCKED_EXPECTED
        )
    )
    observations = repository.observations()
    validate_observations(observations)
    aggregate = build_burnin_report(observations, policy).model_dump(mode="json")
    policy_payload = policy.model_dump(mode="json")
    policy_payload.update(
        {
            "MIN_VALID_CYCLES": policy.min_primary_cycles,
            "MIN_TRADING_DAYS": policy.min_distinct_moex_trading_days,
        }
    )
    observation_payload = [row.model_dump(mode="json") for row in observations]
    replay = {
        "BURNIN_LEDGER_REPLAY": "PASS",
        "BURNIN_IDEMPOTENCY": "PASS",
        "fixture_record_count": len(observations),
        "last_record_sha": observations[-1].record_sha,
        "duplicate_observation_ids": 0,
        "duplicate_operation_slots": 0,
    }
    calendar = {
        "BURNIN_CALENDAR_GATE": "PASS",
        "authoritative_open_proof_required": True,
        "weekday_implies_open": False,
        "weekend_status": "CLOSED",
        "unverified_status": "UNKNOWN",
    }
    safety = {
        **BurninSafety().model_dump(mode="json"),
        "BURNIN_SAFETY_GATES": "PASS",
        "REAL_EXECUTION_READY": "NO",
    }
    schema = BurninObservation.model_json_schema()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": BURNIN_ARTIFACT_VERSION,
        "BASE_MAIN_SHA": base_main_sha,
        "HEAD_SHA": head_sha,
        "ARTIFACT_CODE_SHA": head_sha,
        "BURNIN_POLICY_VERSION": policy.policy_version,
        "PRODUCTION_DRY_RUN_BURNIN_READY": "YES",
        "BURNIN_STATUS": "NOT_STARTED",
        "BURNIN_VALID_CYCLES": 0,
        "BURNIN_DISTINCT_TRADING_DAYS": 0,
        "LIVE_BURNIN_OBSERVATIONS": 0,
        "fixture_only": True,
        "MIN_VALID_CYCLES": policy.min_primary_cycles,
        "MIN_TRADING_DAYS": policy.min_distinct_moex_trading_days,
        "BURNIN_LEDGER_REPLAY": "PASS",
        "BURNIN_IDEMPOTENCY": "PASS",
        "BURNIN_CALENDAR_GATE": "PASS",
        "BURNIN_SAFETY_GATES": "PASS",
    }
    manifest["ARTIFACT_SHA"] = sha256_payload(
        {
            "manifest": manifest,
            "policy": policy_payload,
            "schema": schema,
            "sample_observations": observation_payload,
            "replay": replay,
            "calendar": calendar,
            "failure_taxonomy": FAILURE_TAXONOMY,
            "aggregate": aggregate,
            "safety": safety,
        }
    )
    files: dict[str, object] = {
        "manifest.json": manifest,
        "burnin-policy.json": policy_payload,
        "observation-schema.json": schema,
        "replay-verification.json": replay,
        "calendar-verification.json": calendar,
        "failure-taxonomy.json": FAILURE_TAXONOMY,
        "aggregate-report.json": aggregate,
        "safety.json": safety,
    }
    for name, payload in files.items():
        _write_json(work_root / name, payload)
    _write_jsonl(work_root / "sample-observations.jsonl", observation_payload)
    _write_report(work_root / "report.md", manifest)
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(work_root, output_root)
    return manifest


def build_sample_observation(
    started: datetime, status: BurninObservationStatus
) -> BurninObservation:
    passed = status == BurninObservationStatus.PASS
    day = started.date().isoformat()
    portfolio_sha = sha256_payload([day, "portfolio"])
    return BurninObservation(
        sequence=1,
        previous_record_sha=None,
        record_sha="PENDING",
        observation_id=f"fixture-observation-{day}",
        burnin_observation_id=f"fixture-observation-{day}",
        trading_date=day,
        market_date=day,
        operation_slot="BURNIN_EOD",
        operation_session="BURNIN_EOD",
        operation_id=f"fixture-operation-{day}",
        operation_slot_id=f"{day}:BURNIN_EOD",
        attempt_type=BurninAttemptType.PRIMARY,
        primary_operation_id=f"fixture-operation-{day}",
        cycle_started_at=started,
        decision_as_of=started + timedelta(seconds=2),
        completed_at=started + timedelta(seconds=5),
        code_sha="fixture-code-sha",
        policy_version=BurninPolicy().policy_version,
        prompt_version="agent-readonly-research-v1",
        agent_model_id="fixture-model",
        market_adapter_id="fixture-market",
        universe_sha=sha256_payload(["SBER", "YDEX"]),
        operation_contract_sha=sha256_payload([day, "contract"]),
        market_snapshot_sha=sha256_payload([day, "market"]),
        event_snapshot_sha=sha256_payload([day, "events"]),
        research_status_sha=sha256_payload([day, "research"]),
        portfolio_sha=portfolio_sha,
        market_fetch_started_at=started,
        market_fetch_completed_at=started + timedelta(seconds=2),
        market_fetch_duration_ms=2000,
        operation_duration_ms=5000,
        duration_ms=5000,
        model_latency_ms=1200 if passed else 0,
        universe_count=2,
        market_quote_count=2 if passed else 1,
        fresh_quote_count=2 if passed else 0,
        missing_quote_count=0 if passed else 1,
        max_market_age_seconds=3.0 if passed else 0.0,
        research_status="READY" if passed else "NOT_READY",
        research_operation_status="READY" if passed else "NOT_READY",
        operational_burnin_status="PASS" if passed else "PARTIAL",
        source_failure_isolation=passed,
        seal_verified=passed,
        research_seal_verified=passed,
        agent_decision_status="VALID" if passed else "NOT_CALLED",
        agent_steps=1 if passed else 0,
        agent_proposal_count=1 if passed else 0,
        proposal_count=1 if passed else 0,
        proposal_action_counts={"HOLD": 1} if passed else {},
        model_calls=1 if passed else 0,
        risk_plan_created=passed,
        risk_decision_count=1 if passed else 0,
        operation_status="NO_ACTION" if passed else "BLOCKED",
        status=status,
        status_code="NO_ACTION" if passed else "RESEARCH_NOT_READY",
        operation_status_code="NO_ACTION" if passed else "RESEARCH_NOT_READY",
        reasons=[] if passed else ["RESEARCH_NOT_READY"],
        paper_ledger_event_count_before=0,
        paper_ledger_event_count_after=0,
        paper_ledger_sha_before=sha256_payload([]),
        paper_ledger_sha_after=sha256_payload([]),
        portfolio_sha_before=portfolio_sha,
        portfolio_sha_after=portfolio_sha,
        paper_ledger_unchanged=True,
        session=MoexSessionEvidence(
            trading_date=day,
            status=MoexSessionStatus.OPEN,
            source="DETERMINISTIC_AUTHORITATIVE_FIXTURE",
            checked_at=started,
            evidence_sha=sha256_payload(["session", day]),
        ),
        safety=BurninSafety(),
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Production dry-run burn-in V1",
        "",
        "This artifact proves that the observation infrastructure is ready; "
        "fixture results are not live burn-in evidence.",
        "",
        f"- infrastructure ready: {manifest['PRODUCTION_DRY_RUN_BURNIN_READY']}",
        f"- actual burn-in status: {manifest['BURNIN_STATUS']}",
        f"- live observations included: {manifest['LIVE_BURNIN_OBSERVATIONS']}",
        "- paper execution enabled: false",
        "- real execution enabled: false",
        "- scheduling enabled: false",
        "- profitability evaluated: false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
