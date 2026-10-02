from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

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
    CURRENT_BURNIN_EPOCH,
    MIXED_CODE_BURNIN_EPOCH,
    PRE_FIX_BURNIN_EPOCH,
    SAFETY_FAILED_BURNIN_EPOCH,
    BurninAttemptType,
    BurninObservation,
    BurninObservationStatus,
    BurninPolicy,
    BurninSafety,
    MoexSessionEvidence,
    MoexSessionStatus,
)
from src.production_dry_run_burnin_v1.reporting import (
    build_burnin_artifact,
    build_sample_observation,
)
from src.production_dry_run_burnin_v1.repository import (
    BurninAlreadyRunningError,
    BurninCodeShaHomogeneityError,
    BurninLedgerIntegrityError,
    BurninSingleFlightLock,
    InMemoryBurninObservationRepository,
    JsonlBurninObservationRepository,
    chain_observation,
    observation_record_sha,
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
    code_sha: str = "test-code",
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
        code_sha=code_sha,
        retry_index=retry_index,
        retry_reason=retry_reason,
        clock=lambda: NOW + timedelta(seconds=5),
        operation_runner=operation_runner or fake_operation_runner,
    )


def test_burnin_observation_append_and_replay() -> None:
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


def test_nine_historical_records_remain_readable_without_byte_rewrite(tmp_path: Path) -> None:
    specifications = (
        (NOW - timedelta(days=8), PRE_FIX_BURNIN_EPOCH, "legacy-code"),
        (NOW - timedelta(days=7), PRE_FIX_BURNIN_EPOCH, "legacy-code"),
        (NOW - timedelta(days=6), MIXED_CODE_BURNIN_EPOCH, "code-a"),
        (NOW - timedelta(days=5), MIXED_CODE_BURNIN_EPOCH, "code-b"),
        (NOW - timedelta(days=4), SAFETY_FAILED_BURNIN_EPOCH, "epoch-3-code"),
        (NOW - timedelta(days=3), SAFETY_FAILED_BURNIN_EPOCH, "epoch-3-code"),
        (NOW - timedelta(days=2), SAFETY_FAILED_BURNIN_EPOCH, "epoch-3-code"),
        (NOW - timedelta(days=1), SAFETY_FAILED_BURNIN_EPOCH, "epoch-3-code"),
        (NOW, SAFETY_FAILED_BURNIN_EPOCH, "epoch-3-code"),
    )
    rows: list[BurninObservation] = []
    for started, epoch, code_sha in specifications:
        candidate = build_sample_observation(started, BurninObservationStatus.PASS).model_copy(
            update={"burnin_epoch": epoch, "code_sha": code_sha}
        )
        rows.append(chain_observation(candidate, rows))
    path = tmp_path / "observations.jsonl"
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    before = path.read_bytes()
    restored = JsonlBurninObservationRepository(path).observations()
    assert len(restored) == 9
    assert path.read_bytes() == before
    validate_observations(restored)
    assert observation_record_sha(restored[-1]) == restored[-1].record_sha


def test_repository_rejects_mixed_code_primary_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    repository = JsonlBurninObservationRepository(path)
    repository.append(build_sample_observation(NOW, BurninObservationStatus.PASS))
    before = path.read_bytes()
    mismatched = build_sample_observation(
        NOW + timedelta(days=1), BurninObservationStatus.PASS
    ).model_copy(update={"code_sha": "different-code"})
    with pytest.raises(BurninCodeShaHomogeneityError, match="PRIMARY_CODE_SHA_MISMATCH"):
        repository.append(mismatched)
    assert path.read_bytes() == before


def test_primary_code_sha_guard_blocks_before_session_model_and_operation(tmp_path: Path) -> None:
    observations = InMemoryBurninObservationRepository()
    observations.append(
        build_sample_observation(NOW - timedelta(days=1), BurninObservationStatus.PASS)
    )
    model = CountingModel()
    session = StaticSessionVerifier()
    operation_called = False

    def operation_runner(**_kwargs: Any) -> PaperOperationRun:
        nonlocal operation_called
        operation_called = True
        return _operation_run()

    with pytest.raises(BurninCodeShaHomogeneityError, match="PRIMARY_CODE_SHA_MISMATCH"):
        _runner(
            tmp_path,
            observations=observations,
            model=model,
            session=session,
            operation_runner=operation_runner,
            code_sha="different-code",
        )
    assert session.calls == 0
    assert model.calls == 0
    assert operation_called is False
    assert len(observations.observations()) == 1


def test_old_epoch1_observations_still_verify() -> None:
    repository = JsonlBurninObservationRepository(
        Path("artifacts/production-dry-run-burnin-v1/sample-observations.jsonl")
    )
    rows = repository.observations()
    assert len(rows) == 2
    assert all(row.session.session_policy_version is None for row in rows)


def test_existing_epoch2_observation_still_verifies() -> None:
    repository = InMemoryBurninObservationRepository()
    saved = repository.append(build_sample_observation(NOW, BurninObservationStatus.PASS))
    assert saved.record_sha == observation_record_sha(saved)
    validate_observations(repository.observations())


def test_old_record_hashes_are_not_rewritten() -> None:
    path = Path("artifacts/production-dry-run-burnin-v1/sample-observations.jsonl")
    before = path.read_bytes()
    JsonlBurninObservationRepository(path).observations()
    assert path.read_bytes() == before


def test_old_records_without_v2_calendar_fields_are_backward_compatible() -> None:
    raw = json.loads(
        Path("artifacts/production-dry-run-burnin-v1/sample-observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert "session_policy_version" not in raw["session"]
    repository = JsonlBurninObservationRepository(
        Path("artifacts/production-dry-run-burnin-v1/sample-observations.jsonl")
    )
    row = repository.observations()[0]
    assert row.session.calendar_date is None
    assert row.record_sha == observation_record_sha(row)


def test_burnin_hash_chain_detects_corruption(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    repository = JsonlBurninObservationRepository(path)
    repository.append(build_sample_observation(NOW, BurninObservationStatus.PASS))
    path.write_text(path.read_text(encoding="utf-8").replace("PASS", "FAIL", 1), encoding="utf-8")
    with pytest.raises(BurninLedgerIntegrityError):
        repository.observations()


def test_duplicate_operation_slot_is_not_recorded_twice(tmp_path: Path) -> None:
    repository = InMemoryBurninObservationRepository()
    first = _runner(tmp_path, observations=repository)
    second = _runner(tmp_path, observations=repository)
    assert first.observation is not None
    assert second.status == "ALREADY_OBSERVED"
    assert len(repository.observations()) == 1


def test_duplicate_logical_slot_fails_ledger_validation() -> None:
    first = build_sample_observation(NOW, BurninObservationStatus.PASS)
    second = first.model_copy(
        update={
            "observation_id": "different-observation",
            "burnin_observation_id": "different-observation",
            "primary_operation_id": "different-operation",
        }
    )
    repository = InMemoryBurninObservationRepository()
    repository.append(first)
    with pytest.raises(BurninLedgerIntegrityError, match="DUPLICATE_BURNIN_OPERATION_SLOT"):
        repository.append(second)


def test_jsonl_append_writes_compact_snapshot(tmp_path: Path) -> None:
    repository = JsonlBurninObservationRepository(tmp_path / "observations.jsonl")
    saved = repository.append(build_sample_observation(NOW, BurninObservationStatus.PASS))
    snapshot = tmp_path / "snapshots" / f"{saved.burnin_observation_id}.json"
    assert json.loads(snapshot.read_text(encoding="utf-8"))["record_sha"] == saved.record_sha


def test_burnin_single_flight_rejects_second_runner(tmp_path: Path) -> None:
    lock_path = tmp_path / "burnin.lock"
    with BurninSingleFlightLock(lock_path):
        with pytest.raises(BurninAlreadyRunningError, match="BURNIN_ALREADY_RUNNING"):
            with BurninSingleFlightLock(lock_path):
                pass
    assert not lock_path.exists()


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


def test_production_burnin_cli_exposes_required_commands() -> None:
    parser = build_parser()
    for command in ("run", "status", "history", "report", "calendar-status"):
        args = parser.parse_args([command])
        assert args.command == command
    range_args = parser.parse_args(["calendar-range", "--from", "2026-09-19", "--to", "2026-09-21"])
    assert range_args.command == "calendar-range"


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


def test_distinct_days_use_v2_business_date_without_double_counting() -> None:
    saturday = build_sample_observation(NOW, BurninObservationStatus.PASS)
    monday = build_sample_observation(NOW + timedelta(days=2), BurninObservationStatus.PASS)
    saturday = saturday.model_copy(
        update={
            "session": saturday.session.model_copy(
                update={
                    "session_policy_version": "authoritative-moex-trading-calendar-v2",
                    "calendar_date": saturday.market_date,
                    "moex_business_date": monday.market_date,
                    "session_kind": "WEEKEND_ADDITIONAL",
                }
            )
        }
    )
    monday = monday.model_copy(
        update={
            "session": monday.session.model_copy(
                update={
                    "session_policy_version": "authoritative-moex-trading-calendar-v2",
                    "calendar_date": monday.market_date,
                    "moex_business_date": monday.market_date,
                    "session_kind": "REGULAR",
                }
            )
        }
    )
    assert build_burnin_report([saturday, monday]).distinct_trading_days == 1


def test_epoch2_mixed_code_is_explicitly_non_qualifying() -> None:
    first = build_sample_observation(NOW, BurninObservationStatus.PASS).model_copy(
        update={"burnin_epoch": MIXED_CODE_BURNIN_EPOCH, "code_sha": "code-a"}
    )
    second = build_sample_observation(
        NOW + timedelta(days=1), BurninObservationStatus.PASS
    ).model_copy(update={"burnin_epoch": MIXED_CODE_BURNIN_EPOCH, "code_sha": "code-b"})
    report = build_burnin_report(
        [first, second],
        BurninPolicy(burnin_epoch=MIXED_CODE_BURNIN_EPOCH),
    )
    assert report.CODE_SHA_HOMOGENEITY == "FAIL"
    assert report.QUALIFICATION_STATUS == "NON_QUALIFYING_MIXED_CODE"
    assert report.observed_code_shas == ["code-a", "code-b"]
    assert report.valid_cycles == 0
    assert report.distinct_trading_days == 0
    assert report.BURNIN_STATUS.value == "FAIL"


def test_epoch3_future_quote_safety_violation_is_preserved_and_non_qualifying() -> None:
    rows = [
        build_sample_observation(
            NOW + timedelta(days=index), BurninObservationStatus.PASS
        ).model_copy(
            update={
                "burnin_epoch": SAFETY_FAILED_BURNIN_EPOCH,
                "code_sha": "epoch-3-code",
            }
        )
        for index in range(5)
    ]
    rows[-1] = rows[-1].model_copy(
        update={
            "status": BurninObservationStatus.SAFETY_VIOLATION,
            "status_code": "FUTURE_MARKET_QUOTE",
            "future_quote_count": 1,
            "max_source_clock_delta_seconds": 0.400851,
            "reasons": ["FUTURE_MARKET_QUOTE"],
        }
    )
    report = build_burnin_report(
        rows,
        BurninPolicy(burnin_epoch=SAFETY_FAILED_BURNIN_EPOCH),
    )
    assert rows[-1].status == BurninObservationStatus.SAFETY_VIOLATION
    assert rows[-1].status_code == "FUTURE_MARKET_QUOTE"
    assert rows[-1].max_source_clock_delta_seconds == 0.400851
    assert report.CODE_SHA_HOMOGENEITY == "PASS"
    assert report.QUALIFICATION_STATUS == "NON_QUALIFYING_SAFETY_VIOLATION"
    assert report.future_data_violation_count == 1
    assert report.BURNIN_STATUS.value == "FAIL"


def test_epoch4_starts_zero_zero_with_all_historical_epochs() -> None:
    rows = [
        build_sample_observation(NOW, BurninObservationStatus.PASS).model_copy(
            update={"burnin_epoch": PRE_FIX_BURNIN_EPOCH, "code_sha": "legacy-code"}
        ),
        build_sample_observation(NOW + timedelta(days=1), BurninObservationStatus.PASS).model_copy(
            update={"burnin_epoch": MIXED_CODE_BURNIN_EPOCH, "code_sha": "code-a"}
        ),
        build_sample_observation(
            NOW + timedelta(days=2), BurninObservationStatus.SAFETY_VIOLATION
        ).model_copy(
            update={
                "burnin_epoch": SAFETY_FAILED_BURNIN_EPOCH,
                "code_sha": "epoch-3-code",
                "status_code": "FUTURE_MARKET_QUOTE",
                "future_quote_count": 1,
            }
        ),
    ]
    report = build_burnin_report(rows)
    assert report.BURNIN_EPOCH == CURRENT_BURNIN_EPOCH
    assert report.CODE_SHA_HOMOGENEITY == "NOT_STARTED"
    assert report.QUALIFICATION_STATUS == "NOT_STARTED"
    assert report.valid_cycles == 0
    assert report.distinct_trading_days == 0
    assert report.BURNIN_STATUS.value == "NOT_STARTED"


def test_epoch4_preserves_fixed_thresholds_and_disabled_execution() -> None:
    policy = BurninPolicy()
    fixed_thresholds = {
        "min_distinct_moex_trading_days": policy.min_distinct_moex_trading_days,
        "min_primary_cycles": policy.min_primary_cycles,
        "max_blocked_primary_cycle_rate": policy.max_blocked_primary_cycle_rate,
        "min_market_fresh_rate": policy.min_market_fresh_rate,
        "min_research_ready_rate": policy.min_research_ready_rate,
        "min_agent_valid_rate": policy.min_agent_valid_rate,
        "min_risk_completion_rate": policy.min_risk_completion_rate,
        "max_source_clock_skew_seconds": policy.max_source_clock_skew_seconds,
    }
    assert json.dumps(fixed_thresholds, sort_keys=True, separators=(",", ":")) == (
        '{"max_blocked_primary_cycle_rate":0.2,"max_source_clock_skew_seconds":5.0,'
        '"min_agent_valid_rate":0.9,"min_distinct_moex_trading_days":5,'
        '"min_market_fresh_rate":0.95,"min_primary_cycles":5,'
        '"min_research_ready_rate":0.9,"min_risk_completion_rate":0.9}'
    )
    assert policy.burnin_epoch == CURRENT_BURNIN_EPOCH
    assert policy.paper_execution_enabled is False
    assert policy.real_execution_enabled is False
    assert policy.paper_operation_schedule_enabled is False
    assert BurninSafety().violation_count() == 0


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
    report = build_burnin_report(rows)
    assert report.PRODUCTION_DRY_RUN_BURNIN_READY == "YES"
    assert report.BURNIN_STATUS.value == "FAIL"


def test_minimum_days_required_before_readiness() -> None:
    row = build_sample_observation(NOW, BurninObservationStatus.PASS)
    report = build_burnin_report([row])
    assert report.BURNIN_COLLECTION_STATUS.value == "IN_PROGRESS"
    assert report.PRODUCTION_DRY_RUN_BURNIN_READY == "YES"
    assert report.BURNIN_STATUS.value == "IN_PROGRESS"


def test_readiness_yes_only_after_all_fixed_thresholds() -> None:
    rows = [
        build_sample_observation(NOW + timedelta(days=index), BurninObservationStatus.PASS)
        for index in range(5)
    ]
    report = build_burnin_report(rows)
    assert report.BURNIN_COLLECTION_STATUS.value == "COMPLETE"
    assert report.PRODUCTION_DRY_RUN_BURNIN_READY == "YES"
    assert report.BURNIN_STATUS.value == "PASS"


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
    assert result.observation.status_code == "MARKET_SESSION_UNKNOWN"
    assert model.calls == 0


def test_closed_session_blocks_before_model(tmp_path: Path) -> None:
    model = CountingModel()
    result = _runner(
        tmp_path,
        model=model,
        session=StaticSessionVerifier(MoexSessionStatus.CLOSED),
    )
    assert result.observation is not None
    assert result.observation.status == BurninObservationStatus.BLOCKED_EXPECTED
    assert result.observation.status_code == "MARKET_SESSION_CLOSED"
    assert model.calls == 0


def test_open_session_allows_existing_dry_run_path(tmp_path: Path) -> None:
    model = CountingModel()
    result = _runner(tmp_path, model=model, session=StaticSessionVerifier())
    assert result.observation is not None
    assert result.observation.status == BurninObservationStatus.PASS
    assert model.calls == 1


def test_unknown_session_does_not_increment_distinct_trading_days() -> None:
    row = build_sample_observation(NOW, BurninObservationStatus.BLOCKED_EXPECTED)
    row = row.model_copy(
        update={"session": row.session.model_copy(update={"status": MoexSessionStatus.UNKNOWN})}
    )
    assert build_burnin_report([row]).distinct_trading_days == 0


def test_burnin_status_not_started() -> None:
    report = build_burnin_report([])
    assert report.PRODUCTION_DRY_RUN_BURNIN_READY == "YES"
    assert report.BURNIN_STATUS.value == "NOT_STARTED"


def test_mocked_observations_do_not_create_profitability_metrics() -> None:
    keys = {key.lower() for key in build_burnin_report([]).model_dump()}
    assert not keys & {"pnl", "returns", "sharpe", "sortino", "alpha", "win_rate"}


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
    assert manifest["PRODUCTION_DRY_RUN_BURNIN_READY"] == "YES"
    assert manifest["BURNIN_STATUS"] == "NOT_STARTED"
    assert manifest["LIVE_BURNIN_OBSERVATIONS"] == 0
    assert {path.name for path in first.iterdir()} == {
        "manifest.json",
        "burnin-policy.json",
        "observation-schema.json",
        "sample-observations.jsonl",
        "replay-verification.json",
        "calendar-verification.json",
        "failure-taxonomy.json",
        "aggregate-report.json",
        "safety.json",
        "report.md",
    }
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
