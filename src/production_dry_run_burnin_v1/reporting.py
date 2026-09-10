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


def build_burnin_artifact(
    *,
    output_root: Path,
    work_root: Path,
    base_main_sha: str,
    head_sha: str,
) -> dict[str, Any]:
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    policy = BurninPolicy()
    repository = InMemoryBurninObservationRepository()
    repository.append(build_sample_observation(SAMPLE_START, BurninObservationStatus.PASS))
    repository.append(
        build_sample_observation(
            SAMPLE_START + timedelta(days=1),
            BurninObservationStatus.BLOCKED_EXPECTED,
        )
    )
    observations = repository.observations()
    validate_observations(observations)
    report = build_burnin_report(observations, policy)
    policy_payload = policy.model_dump(mode="json")
    observation_payload = [row.model_dump(mode="json") for row in observations]
    integrity = {
        "BURNIN_LEDGER_INTEGRITY": "PASS",
        "record_count": len(observations),
        "last_record_sha": observations[-1].record_sha,
        "duplicate_observation_ids": 0,
        "duplicate_primary_operation_ids": 0,
    }
    failures = {
        "BLOCKED_EXPECTED": [
            "MARKET_CONTEXT_UNAVAILABLE",
            "STALE_MARKET",
            "BID_ABOVE_ASK",
            "SOURCE_CLOCK_SKEW",
            "LIVE_RESEARCH_NOT_READY",
        ],
        "FAIL": ["BURNIN_INTERNAL_FAILURE"],
        "SAFETY_VIOLATION": [
            "BURNIN_PAPER_LEDGER_MUTATED",
            "FUTURE_MARKET_QUOTE",
            "HOLDOUT_OPENED",
        ],
    }
    safety = BurninSafety().model_dump(mode="json")
    report_payload = report.model_dump(mode="json")
    manifest = {
        "ARTIFACT_VERSION": BURNIN_ARTIFACT_VERSION,
        "BASE_MAIN_SHA": base_main_sha,
        "HEAD_SHA": head_sha,
        "ARTIFACT_CODE_SHA": head_sha,
        "BURNIN_POLICY_VERSION": policy.policy_version,
        "BURNIN_FRAMEWORK_READY": "YES",
        "BURNIN_LEDGER_INTEGRITY": "PASS",
        "BURNIN_IDEMPOTENCY": "PASS",
        "BURNIN_SAFETY_GATES": "PASS",
        "BURNIN_COLLECTION_STATUS": "IN_PROGRESS",
        "PRODUCTION_DRY_RUN_BURNIN_READY": "NO",
        "MIN_DISTINCT_MOEX_TRADING_DAYS": policy.min_distinct_moex_trading_days,
        "fixture_distinct_trading_days": report.distinct_trading_days,
        "fixture_only": True,
        "safety": safety,
    }
    manifest["ARTIFACT_SHA"] = sha256_payload(
        {
            "manifest": manifest,
            "policy": policy_payload,
            "observations": observation_payload,
            "integrity": integrity,
            "report": report_payload,
            "failure_cases": failures,
            "safety": safety,
        }
    )
    _write_json(work_root / "manifest.json", manifest)
    _write_json(work_root / "burnin-policy.json", policy_payload)
    _write_jsonl(work_root / "sample-observations.jsonl", observation_payload)
    _write_json(work_root / "integrity-verification.json", integrity)
    _write_json(work_root / "sample-report.json", report_payload)
    _write_json(work_root / "failure-cases.json", failures)
    _write_json(work_root / "safety.json", safety)
    _write_report(work_root / "report.md", manifest, report_payload)
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(work_root, output_root)
    return manifest


def build_sample_observation(
    started: datetime,
    status: BurninObservationStatus,
) -> BurninObservation:
    passed = status == BurninObservationStatus.PASS
    suffix = started.date().isoformat()
    session = MoexSessionEvidence(
        trading_date=suffix,
        status=MoexSessionStatus.TRADING_DAY,
        source="DETERMINISTIC_FIXTURE",
        checked_at=started,
        evidence_sha=sha256_payload(["session", suffix]),
    )
    return BurninObservation(
        sequence=1,
        previous_record_sha=None,
        record_sha="PENDING",
        observation_id=f"fixture-observation-{suffix}",
        trading_date=suffix,
        operation_slot="BURNIN_EOD",
        attempt_type=BurninAttemptType.PRIMARY,
        primary_operation_id=f"fixture-operation-{suffix}",
        cycle_started_at=started,
        decision_as_of=started + timedelta(seconds=2),
        completed_at=started + timedelta(seconds=5),
        code_sha="fixture-code-sha",
        policy_version=BurninPolicy().policy_version,
        prompt_version="agent-readonly-research-v1",
        agent_model_id="fixture-model",
        market_adapter_id="fixture-market",
        universe_sha=sha256_payload(["SBER", "YDEX"]),
        operation_contract_sha=sha256_payload([suffix, "contract"]),
        market_snapshot_sha=sha256_payload([suffix, "market"]),
        event_snapshot_sha=sha256_payload([suffix, "events"]),
        research_status_sha=sha256_payload([suffix, "research"]),
        market_fetch_duration_ms=2000,
        operation_duration_ms=5000,
        model_latency_ms=1200 if passed else 0,
        universe_count=2,
        market_quote_count=2 if passed else 1,
        fresh_quote_count=2 if passed else 0,
        missing_quote_count=0 if passed else 1,
        max_market_age_seconds=3.0 if passed else 0.0,
        max_source_clock_delta_seconds=-1.0 if passed else 0.0,
        research_status="READY" if passed else "NOT_READY",
        source_failure_isolation=passed,
        seal_verified=passed,
        agent_decision_status="VALID" if passed else "NOT_CALLED",
        agent_proposal_count=1 if passed else 0,
        proposal_action_counts={"HOLD": 1} if passed else {},
        model_calls=1 if passed else 0,
        risk_plan_created=passed,
        risk_decision_count=1 if passed else 0,
        operation_status="NO_ACTION" if passed else "BLOCKED",
        status=status,
        status_code="NO_ACTION" if passed else "LIVE_RESEARCH_NOT_READY",
        reasons=[] if passed else ["LIVE_RESEARCH_NOT_READY"],
        paper_ledger_event_count_before=0,
        paper_ledger_event_count_after=0,
        paper_ledger_sha_before=sha256_payload([]),
        paper_ledger_sha_after=sha256_payload([]),
        portfolio_sha_before=sha256_payload([suffix, "portfolio"]),
        portfolio_sha_after=sha256_payload([suffix, "portfolio"]),
        paper_ledger_unchanged=True,
        session=session,
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


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> None:
    lines = [
        "# Production dry-run burn-in V1",
        "",
        "Burn-in proves operational reliability only. It does not prove profitability or alpha.",
        "",
        f"- framework ready: {manifest['BURNIN_FRAMEWORK_READY']}",
        f"- ledger integrity: {manifest['BURNIN_LEDGER_INTEGRITY']}",
        f"- collection status: {manifest['BURNIN_COLLECTION_STATUS']}",
        f"- production burn-in ready: {manifest['PRODUCTION_DRY_RUN_BURNIN_READY']}",
        f"- fixture primary cycles: {report['primary_cycles']}",
        "- paper execution enabled: false",
        "- real execution enabled: false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
