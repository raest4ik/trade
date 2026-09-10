from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from apps.cli.production_dry_run_burnin import build_parser
from src.ai_trading_agent_v1.application import AgentModelRequest
from src.ai_trading_agent_v1.domain import AgentModelResponse
from src.free_live_issuer_accumulation.domain import sha256_payload
from src.paper_trading_operation_v1.application import (
    PaperOperationContext,
    StaticPaperOperationContextProvider,
    build_operation_id,
)
from src.paper_trading_operation_v1.domain import (
    OperationAuditEvent,
    OperationAuditRecordType,
    PaperOperationMode,
    PaperOperationPolicy,
    PaperOperationRun,
    PaperOperationSafety,
    PaperOperationStatus,
)
from src.paper_trading_operation_v1.repository import InMemoryOperationAuditRepository
from src.production_dry_run_burnin_v1.application import (
    build_burnin_report,
    run_burnin_once,
)
from src.production_dry_run_burnin_v1.domain import (
    BurninAttemptType,
    BurninObservationStatus,
    BurninSafety,
    MoexSessionEvidence,
    MoexSessionStatus,
)
from src.production_dry_run_burnin_v1.policy import MoexIssSessionVerifier
from src.production_dry_run_burnin_v1.reporting import (
    build_burnin_artifact,
    build_sample_observation,
)
from src.production_dry_run_burnin_v1.repository import (
    BurninLedgerIntegrityError,
    InMemoryBurninObservationRepository,
    JsonlBurninObservationRepository,
    validate_observations,
)
from src.risk_engine_paper_v1.domain import MarketQuote
from src.risk_engine_paper_v1.repository import InMemoryPaperLedgerRepository

NOW = datetime(2026, 9, 10, 15, 0, tzinfo=UTC)


class StaticSessionVerifier:
    def __init__(self, status: MoexSessionStatus = MoexSessionStatus.TRADING_DAY) -> None:
        self.status = status
        self.calls = 0

    def verify(self, trading_date: Any) -> MoexSessionEvidence:
        self.calls += 1
        return MoexSessionEvidence(
            trading_date=trading_date.isoformat(),
            status=self.status,
            source="TEST",
            checked_at=NOW,
            reason=(
                None
                if self.status == MoexSessionStatus.TRADING_DAY
                else "MOEX_SESSION_STATUS_UNKNOWN"
            ),
        )


class CountingModel:
    model_id = "counting-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: AgentModelRequest) -> AgentModelResponse:
        self.calls += 1
        return AgentModelResponse(final_output=json.dumps({"unused": True}))


def _operation_run(
    *,
    status: PaperOperationStatus = PaperOperationStatus.NO_ACTION,
    status_code: str = "NO_ACTION",
    future_quotes: int = 0,
    safety: PaperOperationSafety | None = None,
) -> PaperOperationRun:
    operation_id = build_operation_id(operation_as_of=NOW, session="BURNIN_EOD")
    return PaperOperationRun(
        operation_id=operation_id,
        operation_slot_id="2026-09-10:BURNIN_EOD",
        operation_contract_sha=sha256_payload(["contract"]),
        universe_sha=sha256_payload(["SBER"]),
        research_status_sha=sha256_payload(["research"]),
        market_adapter_id="test-market",
        market_source="TEST",
        market_audit={
            "market_fetch_started_at": NOW.isoformat(),
            "market_fetch_completed_at": (NOW + timedelta(seconds=2)).isoformat(),
            "quote_count": 1,
            "fresh_quote_count": 1 if future_quotes == 0 else 0,
            "stale_quote_count": 0,
            "future_quote_count": future_quotes,
            "missing_quote_count": 0,
            "invalid_quote_count": 0,
            "market_source_clock_delta_seconds": -1.0 if future_quotes == 0 else 1.0,
            "quotes": [{"age_seconds": 1.0}],
        },
        operation_as_of=NOW + timedelta(seconds=2),
        cycle_started_at=NOW,
        decision_as_of=NOW + timedelta(seconds=2),
        mode=PaperOperationMode.DRY_RUN,
        status=status,
        status_code=status_code,
        code_sha="test-code",
        policy_version="paper-trading-operation-policy-v1",
        prompt_version="agent-readonly-research-v1",
        agent_model_id="counting-model",
        research_status={
            "LIVE_RESEARCH_OPERATION_STATUS": "READY",
            "SOURCE_FAILURE_ISOLATION": True,
            "seal": {"sealed_epoch_verified": True, "violations": 0},
        },
        universe=[{"ticker": "SBER"}],
        portfolio_before_sha=sha256_payload(["portfolio"]),
        portfolio_before={},
        market_snapshot_sha=sha256_payload(["market"]),
        event_snapshot_sha=sha256_payload(["events"]),
        agent_run_id="agent-run" if status != PaperOperationStatus.BLOCKED else None,
        agent_proposals=(
            [{"ticker": "SBER", "action": "HOLD", "thesis": ["test"]}]
            if status != PaperOperationStatus.BLOCKED
            else []
        ),
        risk_plan_id="risk-plan" if status != PaperOperationStatus.BLOCKED else None,
        risk_decisions=(
            [{"risk_decision": "NO_ACTION"}] if status != PaperOperationStatus.BLOCKED else []
        ),
        paper_execution_status="SKIPPED_DRY_RUN",
        portfolio_after_sha=sha256_payload(["portfolio"]),
        portfolio_after={},
        replay_verified=True,
        steps=[],
        reasons=[] if status != PaperOperationStatus.BLOCKED else [status_code],
        safety=safety or PaperOperationSafety(PAPER_RISK_PLANS=1),
    )


def _runner(
    tmp_path: Path,
    *,
    run: PaperOperationRun | None = None,
    observations: InMemoryBurninObservationRepository | None = None,
    audit: InMemoryOperationAuditRepository | None = None,
    model: CountingModel | None = None,
    session: StaticSessionVerifier | None = None,
    retry_index: int = 0,
    retry_reason: str | None = None,
    operation_runner: Any = None,
) -> Any:
    selected_run = run or _operation_run()
    selected_model = model or CountingModel()

    def fake_operation_runner(**kwargs: Any) -> PaperOperationRun:
        kwargs["model"].complete(
            AgentModelRequest(
                prompt_version="test",
                system_prompt="test",
                allowed_universe=[],
                transcript=[],
                safety_policy={},
            )
        )
        return selected_run

    return run_burnin_once(
        cycle_started_at=NOW,
        model=selected_model,
        context_provider=object(),
        paper_repository=InMemoryPaperLedgerRepository(),
        audit_repository=audit or InMemoryOperationAuditRepository(),
        observation_repository=observations or InMemoryBurninObservationRepository(),
        session_verifier=session or StaticSessionVerifier(),
        operation_state_root=tmp_path / "operation",
        code_sha="test-code",
        retry_index=retry_index,
        retry_reason=retry_reason,
        clock=lambda: NOW + timedelta(seconds=5),
        operation_runner=operation_runner or fake_operation_runner,
    )


def test_burnin_observation_append_only() -> None:
    repository = InMemoryBurninObservationRepository()
    first = repository.append(build_sample_observation(NOW, BurninObservationStatus.PASS))
    second = repository.append(
        build_sample_observation(NOW + timedelta(days=1), BurninObservationStatus.PASS)
    )
    assert second.sequence == 2
    assert second.previous_record_sha == first.record_sha


def test_burnin_hash_chain_valid() -> None:
    repository = InMemoryBurninObservationRepository()
    repository.append(build_sample_observation(NOW, BurninObservationStatus.PASS))
    validate_observations(repository.observations())


def test_burnin_corruption_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    repository = JsonlBurninObservationRepository(path)
    repository.append(build_sample_observation(NOW, BurninObservationStatus.PASS))
    path.write_text(path.read_text(encoding="utf-8").replace("PASS", "FAIL", 1), encoding="utf-8")
    with pytest.raises(BurninLedgerIntegrityError):
        repository.observations()


def test_duplicate_primary_observation_not_appended(tmp_path: Path) -> None:
    repository = InMemoryBurninObservationRepository()
    first = _runner(tmp_path, observations=repository)
    second = _runner(tmp_path, observations=repository)
    assert first.observation is not None
    assert second.status == "ALREADY_PROCESSED"
    assert len(repository.observations()) == 1


def test_retry_does_not_increment_distinct_trading_days() -> None:
    primary = build_sample_observation(NOW, BurninObservationStatus.BLOCKED_EXPECTED)
    retry = build_sample_observation(NOW, BurninObservationStatus.PASS).model_copy(
        update={
            "observation_id": "retry",
            "operation_slot": "BURNIN_EOD_RETRY_1",
            "attempt_type": BurninAttemptType.RETRY,
            "retry_index": 1,
            "retry_reason": "TRANSIENT",
        }
    )
    report = build_burnin_report([primary, retry])
    assert report.distinct_trading_days == 1
    assert report.primary_cycles == 1


def test_recovery_from_operation_audit_does_not_recall_model(tmp_path: Path) -> None:
    run = _operation_run()
    audit = InMemoryOperationAuditRepository(
        [
            OperationAuditEvent(
                sequence=1,
                record_id="completed",
                operation_id=run.operation_id,
                record_type=OperationAuditRecordType.COMPLETED,
                occurred_at=run.operation_as_of,
                payload=run.model_dump(mode="json"),
            )
        ]
    )
    model = CountingModel()
    session = StaticSessionVerifier()
    result = _runner(tmp_path, audit=audit, model=model, session=session)
    assert result.observation is not None
    assert result.observation.status_code == "RECOVERED_FROM_OPERATION_AUDIT"
    assert result.observation.model_calls == 0
    assert model.calls == 0
    assert session.calls == 0


def test_burnin_runner_cannot_enable_paper_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BURNIN_EXECUTION_CAPABILITY_FORBIDDEN"):
        run_burnin_once(
            cycle_started_at=NOW,
            model=CountingModel(),
            context_provider=object(),
            paper_repository=InMemoryPaperLedgerRepository(),
            audit_repository=InMemoryOperationAuditRepository(),
            observation_repository=InMemoryBurninObservationRepository(),
            session_verifier=StaticSessionVerifier(),
            operation_state_root=tmp_path,
            operation_policy=PaperOperationPolicy(paper_execution_enabled=True),
        )


def test_burnin_runner_never_accepts_execute_paper_flag() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-once", "--execute-paper"])


def test_paper_ledger_unchanged_after_successful_burnin(tmp_path: Path) -> None:
    result = _runner(tmp_path, run=_operation_run())
    assert result.observation is not None
    assert result.observation.paper_ledger_unchanged is True
    assert result.observation.paper_ledger_sha_before == result.observation.paper_ledger_sha_after


def test_paper_ledger_unchanged_after_blocked_burnin(tmp_path: Path) -> None:
    blocked = _operation_run(
        status=PaperOperationStatus.BLOCKED,
        status_code="MARKET_CONTEXT_UNAVAILABLE",
    )
    result = _runner(tmp_path, run=blocked)
    assert result.observation is not None
    assert result.observation.paper_ledger_unchanged is True
    assert result.observation.paper_ledger_sha_before == result.observation.paper_ledger_sha_after


def test_real_execution_remains_disabled(tmp_path: Path) -> None:
    result = _runner(tmp_path)
    assert result.observation is not None
    assert result.observation.safety.REAL_EXECUTION_ENABLED is False
    assert result.observation.safety.REAL_BROKER_MUTATIONS == 0


def _preflight_failure(
    tmp_path: Path,
    model: CountingModel,
    *,
    future: bool,
    research_ready: bool,
) -> Any:
    quote_as_of = NOW + timedelta(seconds=10) if future else NOW
    context = PaperOperationContext(
        universe=[{"ticker": "SBER", "supported": True, "market_data_compatible": True}],
        market_quotes=[MarketQuote(ticker="SBER", as_of=quote_as_of, last_price=300, lot_size=10)],
        market_context={
            "market_adapter_id": "test",
            "market_source": "TEST",
            "future_quote_count": 1 if future else 0,
            "fresh_quote_count": 0 if future else 1,
            "quote_count": 1,
        },
        event_context={"events": []},
        research_status={
            "research_status_as_of": NOW.isoformat(),
            "LIVE_RESEARCH_OPERATION_STATUS": "READY" if research_ready else "BLOCKED",
            "OPERATIONAL_BURN_IN": "PASS",
            "SOURCE_FAILURE_ISOLATION": True,
            "SOURCE_FAILURE_ISOLATION_PROOF_LEVEL": "APPLICATION_PROOF",
            "seal": {"sealed_epoch_verified": True, "violations": 0},
        },
    )
    return run_burnin_once(
        cycle_started_at=NOW,
        model=model,
        context_provider=StaticPaperOperationContextProvider(context),
        paper_repository=InMemoryPaperLedgerRepository(),
        audit_repository=InMemoryOperationAuditRepository(),
        observation_repository=InMemoryBurninObservationRepository(),
        session_verifier=StaticSessionVerifier(),
        operation_state_root=tmp_path / "operation",
        code_sha="test-code",
        clock=lambda: NOW + timedelta(seconds=20),
    )


def test_future_market_data_blocks_before_model(tmp_path: Path) -> None:
    model = CountingModel()
    result = _preflight_failure(tmp_path, model, future=True, research_ready=True)
    assert result.observation is not None
    assert result.observation.model_calls == 0
    assert model.calls == 0


def test_research_failure_blocks_before_model(tmp_path: Path) -> None:
    model = CountingModel()
    result = _preflight_failure(tmp_path, model, future=False, research_ready=False)
    assert result.observation is not None
    assert result.observation.model_calls == 0
    assert model.calls == 0


def test_holdout_paths_never_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if "holdout" in str(path).lower():
            raise AssertionError("holdout path read")
        return cast("Any", original(path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", guarded_open)
    result = _runner(tmp_path)
    assert result.observation is not None
    assert result.observation.safety.OLD_FUTURE_HOLDOUT_OPENED is False


def test_primary_success_rate_excludes_retries() -> None:
    primary = build_sample_observation(NOW, BurninObservationStatus.BLOCKED_EXPECTED)
    retry = build_sample_observation(NOW, BurninObservationStatus.PASS).model_copy(
        update={
            "attempt_type": BurninAttemptType.RETRY,
            "retry_index": 1,
            "retry_reason": "TRANSIENT",
        }
    )
    report = build_burnin_report([primary, retry])
    assert report.primary_pass_rate == 0.0


def test_distinct_days_deduplicated() -> None:
    first = build_sample_observation(NOW, BurninObservationStatus.PASS)
    second = first.model_copy(
        update={"observation_id": "second", "primary_operation_id": "second-operation"}
    )
    assert build_burnin_report([first, second]).distinct_trading_days == 1


def test_blocked_expected_not_counted_as_pass() -> None:
    row = build_sample_observation(NOW, BurninObservationStatus.BLOCKED_EXPECTED)
    report = build_burnin_report([row])
    assert report.primary_pass_cycles == 0
    assert report.primary_blocked_cycles == 1


def test_safety_violation_forces_readiness_no() -> None:
    rows = [
        build_sample_observation(NOW + timedelta(days=index), BurninObservationStatus.PASS)
        for index in range(5)
    ]
    rows[-1] = rows[-1].model_copy(
        update={
            "status": BurninObservationStatus.SAFETY_VIOLATION,
            "safety": BurninSafety(PAPER_PORTFOLIO_MUTATIONS=1),
        }
    )
    assert build_burnin_report(rows).PRODUCTION_DRY_RUN_BURNIN_READY == "NO"


def test_minimum_days_required_before_readiness() -> None:
    row = build_sample_observation(NOW, BurninObservationStatus.PASS)
    report = build_burnin_report([row])
    assert report.BURNIN_COLLECTION_STATUS.value == "IN_PROGRESS"
    assert report.PRODUCTION_DRY_RUN_BURNIN_READY == "NO"


def test_readiness_yes_only_after_all_fixed_thresholds() -> None:
    rows = [
        build_sample_observation(NOW + timedelta(days=index), BurninObservationStatus.PASS)
        for index in range(5)
    ]
    report = build_burnin_report(rows)
    assert report.BURNIN_COLLECTION_STATUS.value == "COMPLETE"
    assert report.PRODUCTION_DRY_RUN_BURNIN_READY == "YES"


def test_burnin_uses_final_decision_as_of(tmp_path: Path) -> None:
    result = _runner(tmp_path)
    assert result.observation is not None
    assert result.observation.decision_as_of == NOW + timedelta(seconds=2)
    assert result.observation.cycle_started_at == NOW


def test_all_observed_input_timestamps_lte_decision_as_of(tmp_path: Path) -> None:
    result = _runner(tmp_path)
    assert result.observation is not None
    assert result.observation.future_quote_count == 0
    assert result.observation.max_market_age_seconds >= 0


def test_no_post_decision_market_data_read(tmp_path: Path) -> None:
    result = _runner(tmp_path)
    assert result.observation is not None
    assert result.observation.decision_as_of < result.observation.completed_at
    assert result.observation.safety.LIVE_POST_EVENT_PRICE_READS == 0


def test_source_clock_skew_preserved(tmp_path: Path) -> None:
    result = _runner(tmp_path)
    assert result.observation is not None
    assert result.observation.max_source_clock_delta_seconds == -1.0


def test_unknown_trading_day_blocks_without_model(tmp_path: Path) -> None:
    model = CountingModel()
    result = _runner(
        tmp_path,
        model=model,
        session=StaticSessionVerifier(MoexSessionStatus.UNKNOWN),
    )
    assert result.observation is not None
    assert result.observation.status == BurninObservationStatus.BLOCKED_EXPECTED
    assert result.observation.status_code == "MOEX_SESSION_STATUS_UNKNOWN"
    assert model.calls == 0


def test_unknown_session_does_not_increment_distinct_trading_days() -> None:
    row = build_sample_observation(NOW, BurninObservationStatus.BLOCKED_EXPECTED)
    row = row.model_copy(
        update={"session": row.session.model_copy(update={"status": MoexSessionStatus.UNKNOWN})}
    )
    assert build_burnin_report([row]).distinct_trading_days == 0


def test_moex_session_verifier_uses_official_daily_candle() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "iss.moex.com"
        assert request.url.params["from"] == "2026-09-10"
        return httpx.Response(
            200,
            json={
                "candles": {
                    "columns": ["open", "close", "begin", "end"],
                    "data": [[300.0, 301.0, "2026-09-10 10:00:00", "2026-09-10 23:49:59"]],
                }
            },
        )

    verifier = MoexIssSessionVerifier(
        base_url="https://iss.moex.com/iss",
        timeout_seconds=1,
        user_agent="test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: NOW,
    )
    evidence = verifier.verify(date(2026, 9, 10))
    assert evidence.status == MoexSessionStatus.TRADING_DAY
    assert evidence.evidence_sha is not None


def test_artifact_rebuilds_byte_for_byte(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    manifest = build_burnin_artifact(
        output_root=first,
        work_root=tmp_path / "work-first",
        base_main_sha="base",
        head_sha="head",
    )
    build_burnin_artifact(
        output_root=second,
        work_root=tmp_path / "work-second",
        base_main_sha="base",
        head_sha="head",
    )
    assert manifest["BURNIN_FRAMEWORK_READY"] == "YES"
    assert manifest["PRODUCTION_DRY_RUN_BURNIN_READY"] == "NO"
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
