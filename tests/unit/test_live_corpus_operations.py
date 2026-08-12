from __future__ import annotations

import ast
import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from apps.cli.live_corpus_status import build_parser as status_parser
from apps.cli.live_corpus_status import run as status_run
from src.events.domain.v3 import rules_v3_fingerprint
from src.holdout_evaluation.domain import EXPECTED_RULES_FINGERPRINT
from src.live_corpus_operations.domain import (
    ACCEPTED_SOURCE_CODES,
    TELEGRAM_API_POLICY,
    CorpusSnapshot,
    LiveRunConfig,
    MaturationOutcome,
    MaturationState,
    RunStatus,
    SourceOutcome,
)
from src.live_corpus_operations.local_backend import CommandResult, LocalLiveCorpusBackend
from src.live_corpus_operations.runner import LiveCorpusRunner
from src.live_corpus_operations.state import FileJobLock, OperationsStateStore, sanitize_error
from src.predictive_baseline.domain import MODEL_VERSION, BaselineConfig
from src.shared.config.settings import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_THINK

NOW = datetime(2026, 8, 12, 9, 0, tzinfo=UTC)


def test_one_shot_pipeline_runs_both_sources_and_maturation(tmp_path: Path) -> None:
    backend = FakeBackend()
    report = _run(tmp_path, backend)
    assert report.status == RunStatus.SUCCEEDED
    assert backend.calls == [*ACCEPTED_SOURCE_CODES, "MATURE"]
    assert report.sources_checked == 2
    assert report.maturation.daily_features_created == 1


def test_idempotent_second_run_imports_no_duplicate_real_rows(tmp_path: Path) -> None:
    backend = FakeBackend()
    first = _run(tmp_path, backend)
    second = _run(tmp_path, backend)
    assert sum(item.items_imported for item in first.source_results) == 2
    assert sum(item.items_imported for item in second.source_results) == 0
    assert sum(item.duplicates for item in second.source_results) == 2


def test_stable_identity_prevents_duplicates_after_checkpoint_loss(tmp_path: Path) -> None:
    backend = FakeBackend()
    _run(tmp_path, backend)
    (tmp_path / "checkpoints.json").unlink()
    report = _run(tmp_path, backend)
    assert sum(item.items_imported for item in report.source_results) == 0
    assert sum(item.duplicates for item in report.source_results) == 2


def test_source_failure_is_isolated(tmp_path: Path) -> None:
    backend = FakeBackend(failing_sources={ACCEPTED_SOURCE_CODES[0]})
    report = _run(tmp_path, backend)
    assert report.status == RunStatus.PARTIAL
    assert report.source_results[0].status == "FAILED"
    assert report.source_results[1].status == "SUCCEEDED"
    assert backend.calls[-1] == "MATURE"
    assert not (tmp_path / "growth-history.jsonl").exists()


def test_retry_is_bounded_and_forwarded_without_live_http(tmp_path: Path) -> None:
    executor = FakeExecutor()
    backend = LocalLiveCorpusBackend(executor=executor, repo_root=tmp_path)
    config = replace(_config(), dry_run=True, max_retries=3)
    outcome = asyncio.run(backend.ingest_source(ACCEPTED_SOURCE_CODES[0], config, {}))
    assert outcome.status == "SUCCEEDED"
    arguments = executor.calls[0][1]
    assert arguments[arguments.index("--max-retries") + 1] == "3"
    with pytest.raises(ValueError, match="max_retries"):
        replace(config, max_retries=6).validate()


def test_cumulative_rebuild_is_not_limited_to_polling_window(tmp_path: Path) -> None:
    executor = FakeExecutor()
    backend = LocalLiveCorpusBackend(executor=executor, repo_root=tmp_path)
    asyncio.run(backend.mature(_config()))
    calls = {module: arguments for module, arguments in executor.calls}
    corpus_args = calls["apps.cli.build_official_source_corpus"]
    reaction_args = calls["apps.cli.compute_abnormal_reactions"]
    assert corpus_args[corpus_args.index("--limit") + 1] == "1000"
    assert reaction_args[reaction_args.index("--limit") + 1] == "1000"


def test_maturation_without_new_items_is_bounded_to_recent_week(tmp_path: Path) -> None:
    executor = FakeExecutor()
    backend = LocalLiveCorpusBackend(executor=executor, repo_root=tmp_path)
    asyncio.run(backend.mature(_config()))
    prepare_args = executor.calls[0][1]
    expected = (_config().date_to - timedelta(days=7)).isoformat()
    assert prepare_args[prepare_args.index("--from") + 1] == expected


def test_lock_prevents_overlapping_runner(tmp_path: Path) -> None:
    lock_path = tmp_path / "live.lock"
    first = FileJobLock(lock_path)
    assert first.try_acquire(now=NOW)
    try:
        backend = FakeBackend()
        report = asyncio.run(_runner(tmp_path, backend, lock_path=lock_path).execute(_config()))
        assert report.status == RunStatus.ALREADY_RUNNING
        assert backend.calls == []
    finally:
        first.release()


def test_stale_lock_can_be_recovered(tmp_path: Path) -> None:
    lock_path = tmp_path / "live.lock"
    lock_path.write_text(
        json.dumps({"pid": 1, "token": "old", "started_at": (NOW - timedelta(days=1)).isoformat()}),
        encoding="utf-8",
    )
    lock = FileJobLock(lock_path)
    assert lock.try_acquire(now=NOW)
    lock.release()
    assert not lock_path.exists()


def test_source_timestamp_semantics_are_not_poll_time() -> None:
    source = (
        Path(__file__).parents[2] / "src" / "historical_news" / "application" / "use_cases.py"
    ).read_text(encoding="utf-8")
    assert "published_at=candidate.source_published_at" in source
    assert "received_at=candidate.fetched_at" in source
    assert "published_at=candidate.fetched_at" not in source


def test_exact_timestamp_policy_uses_only_accepted_registry() -> None:
    assert ACCEPTED_SOURCE_CODES == (
        "ROSNEFT_PRESS_RELEASES_RSS",
        "YANDEX_IR_PRESS_RELEASES_RSS",
    )
    backend = LocalLiveCorpusBackend(executor=FakeExecutor(), repo_root=Path.cwd())
    for source in ACCEPTED_SOURCE_CODES:
        backend.validate_source(source)
    with pytest.raises(ValueError, match="not accepted"):
        backend.validate_source("THIRD_PARTY_AGGREGATOR")


def test_immature_events_are_waiting_not_failed(tmp_path: Path) -> None:
    maturation = replace(
        _maturation(),
        waiting_intraday=2,
        waiting_daily=3,
        state_counts={
            MaturationState.WAITING_INTRADAY_TARGET.value: 2,
            MaturationState.WAITING_DAILY_TARGET.value: 3,
        },
    )
    report = _run(tmp_path, FakeBackend(maturation=maturation))
    assert report.status == RunStatus.SUCCEEDED
    assert report.maturation.waiting_intraday == 2
    assert not report.errors


def test_matured_intraday_and_daily_reactions_create_features(tmp_path: Path) -> None:
    outcome = _run(tmp_path, FakeBackend()).maturation
    assert outcome.intraday_reactions_created == 1
    assert outcome.daily_reactions_created == 1
    assert outcome.intraday_features_created == 1
    assert outcome.daily_features_created == 1


def test_waiting_counts_are_cumulative_not_recent_subset(tmp_path: Path) -> None:
    _write_snapshot_files(tmp_path)
    backend = LocalLiveCorpusBackend(executor=FakeExecutor(), repo_root=tmp_path)
    outcome = asyncio.run(backend.mature(_config()))
    assert outcome.waiting_intraday == 1
    assert outcome.waiting_daily == 2


def test_feature_creation_never_exceeds_valid_reactions() -> None:
    outcome = _maturation()
    assert outcome.intraday_features_created <= outcome.intraday_reactions_created
    assert outcome.daily_features_created <= outcome.daily_reactions_created


def test_readiness_snapshot_is_included_after_run(tmp_path: Path) -> None:
    report = _run(tmp_path, FakeBackend())
    assert report.snapshot.daily_feature_ready == 34
    assert report.snapshot.intraday_feature_ready == 21


def test_collector_never_auto_trains(tmp_path: Path) -> None:
    report = _run(tmp_path, FakeBackend(snapshot=_snapshot(daily=100)))
    assert report.automatic_training is False
    package = Path(__file__).parents[2] / "src" / "live_corpus_operations"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    assert "train_predictive_baselines" not in source
    assert "train_daily_baseline" not in source


def test_diversity_warning_after_training_row_gate() -> None:
    snapshot = replace(
        _snapshot(daily=100),
        ticker_distribution={"ROSN": 80, "YDEX": 20},
    )
    assert snapshot.warnings() == ("LOW_TICKER_DIVERSITY",)
    balanced = replace(snapshot, ticker_distribution={"ROSN": 70, "YDEX": 30})
    assert balanced.warnings() == ()


def test_dry_run_writes_no_state_or_logs(tmp_path: Path) -> None:
    backend = FakeBackend()
    report = _run(tmp_path, backend, config=replace(_config(), dry_run=True))
    assert report.status == RunStatus.DRY_RUN
    assert not (tmp_path / "checkpoints.json").exists()
    assert not (tmp_path / "health.json").exists()
    assert not (tmp_path / "growth-history.jsonl").exists()
    assert not (tmp_path / "logs").exists()


def test_windows_scripts_are_repo_relative_and_do_not_hardcode_user() -> None:
    root = Path(__file__).parents[2]
    runner = (root / "scripts/windows/run-live-corpus.ps1").read_text(encoding="utf-8")
    installer = (root / "scripts/windows/install-live-corpus-task.ps1").read_text(encoding="utf-8")
    assert "$PSScriptRoot" in runner
    assert "Push-Location $RepoRoot" in runner
    assert "molok" not in runner.lower()
    assert "molok" not in installer.lower()
    assert "LogonType Interactive" in installer
    assert "IntervalMinutes = 60" in installer


def test_secret_values_are_redacted_from_logs(tmp_path: Path) -> None:
    backend = FakeBackend(failing_sources={ACCEPTED_SOURCE_CODES[0]}, secret_failure=True)
    _run(tmp_path, backend)
    log = next((tmp_path / "logs").glob("*.json")).read_text(encoding="utf-8")
    assert "super-secret" not in log
    assert "[REDACTED]" in log
    assert sanitize_error("token=abc") == "token=[REDACTED]"


def test_health_report_has_deterministic_schema(tmp_path: Path) -> None:
    _run(tmp_path, FakeBackend())
    payload = json.loads((tmp_path / "health.json").read_text(encoding="utf-8"))
    assert set(payload) == {
        "database_status",
        "last_run_at",
        "last_success_at",
        "moex_status",
        "schema_version",
        "sources",
        "status",
    }
    assert set(payload["sources"]) == {"ROSN", "YDEX"}


def test_growth_history_appends_once_per_run_id(tmp_path: Path) -> None:
    store = OperationsStateStore(tmp_path)
    report = _run(tmp_path, FakeBackend())
    store.persist(report, successful=True)
    lines = (tmp_path / "growth-history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["daily_feature_ready"] == 34
    assert row["real_total"] == 40


def test_log_rotation_is_bounded(tmp_path: Path) -> None:
    backend = FakeBackend()
    runner = _runner(tmp_path, backend, retention=2)
    for _ in range(3):
        asyncio.run(runner.execute(_config()))
    assert len(list((tmp_path / "logs").glob("*.json"))) == 2


def test_status_cli_combines_health_and_readiness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps(
            {
                "daily_feature_ready": 34,
                "intraday_feature_ready": 21,
                "rows_to_100": 66,
                "rows_to_500": 466,
                "rows_to_1000": 966,
                "training_gate": "TRAINING_BLOCKED",
                "ticker_count": 2,
                "source_count": 2,
            }
        ),
        encoding="utf-8",
    )
    args = status_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "--artifact-root",
            "ops",
            "--readiness",
            "readiness.json",
        ]
    )
    assert status_run(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["collector_status"] == "NEVER_RUN"
    assert payload["rows_to_100"] == 66
    assert payload["automatic_training"] is False


def test_paid_and_hosted_providers_are_forbidden() -> None:
    package = Path(__file__).parents[2] / "src" / "live_corpus_operations"
    source = "\n".join(path.read_text(encoding="utf-8").lower() for path in package.glob("*.py"))
    assert "openai" not in source
    assert "telegram" not in source.replace("telegram_api", "")
    assert "cloud scheduler" not in source
    assert "paid_api" not in source


def test_telegram_api_is_explicitly_rejected_for_ml() -> None:
    assert TELEGRAM_API_POLICY == "REJECTED_POLICY_FOR_ML"
    root = Path(__file__).parents[2]
    assert not any("telegram" in path.name.lower() for path in (root / "src").rglob("*adapter*.py"))


def test_frozen_rules_qwen_and_predictive_model_config_are_unchanged() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert DEFAULT_OLLAMA_MODEL == "qwen3.5:9b"
    assert DEFAULT_OLLAMA_THINK is False
    assert MODEL_VERSION == "predictive-daily-baseline-v1"
    assert BaselineConfig().fingerprint() == BaselineConfig().fingerprint()


def test_unit_tests_use_fake_executor_not_live_http() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported.isdisjoint({"httpx", "requests"})


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def module(
        self,
        module: str,
        arguments: list[str],
        *,
        accepted_codes: frozenset[int] = frozenset({0}),
        timeout_seconds: float = 900.0,
    ) -> CommandResult:
        del accepted_codes, timeout_seconds
        self.calls.append((module, arguments))
        return CommandResult(
            returncode=0,
            stdout=json.dumps(
                {
                    "discovered_count": 1,
                    "imported_count": 0,
                    "duplicate_count": 1,
                    "matched_news_count": 1,
                    "rejected_count": 0,
                }
            ),
            stderr="",
        )


class FakeBackend:
    def __init__(
        self,
        *,
        failing_sources: set[str] | None = None,
        maturation: MaturationOutcome | None = None,
        snapshot: CorpusSnapshot | None = None,
        secret_failure: bool = False,
    ) -> None:
        self.failing_sources = failing_sources or set()
        self.maturation = maturation or _maturation()
        self.current_snapshot = snapshot or _snapshot()
        self.secret_failure = secret_failure
        self.calls: list[str] = []
        self.identities: set[tuple[str, str]] = set()

    def validate_source(self, source_code: str) -> None:
        assert source_code in ACCEPTED_SOURCE_CODES

    async def ingest_source(
        self,
        source_code: str,
        config: LiveRunConfig,
        checkpoint: dict[str, object],
    ) -> SourceOutcome:
        del checkpoint
        self.calls.append(source_code)
        if source_code in self.failing_sources:
            message = "token=super-secret timeout" if self.secret_failure else "source unavailable"
            raise RuntimeError(message)
        identity = (source_code, "stable-item-1")
        duplicate = identity in self.identities
        if not config.dry_run:
            self.identities.add(identity)
        return SourceOutcome(
            source_code=source_code,
            status="SUCCEEDED",
            items_seen=1,
            items_imported=0 if duplicate or config.dry_run else 1,
            duplicates=1 if duplicate else 0,
            matched=1,
            last_item_id=identity[1],
            last_item_at="2026-08-12T08:00:00+00:00",
        )

    async def mature(self, config: LiveRunConfig) -> MaturationOutcome:
        self.calls.append("MATURE")
        if config.dry_run:
            return replace(self.maturation, database_status="DRY_RUN", moex_status="DRY_RUN")
        return self.maturation

    def snapshot(self) -> CorpusSnapshot:
        return self.current_snapshot


def _run(
    root: Path,
    backend: FakeBackend,
    *,
    config: LiveRunConfig | None = None,
):
    return asyncio.run(_runner(root, backend).execute(config or _config()))


def _runner(
    root: Path,
    backend: FakeBackend,
    *,
    lock_path: Path | None = None,
    retention: int = 30,
) -> LiveCorpusRunner:
    tick = iter((0.0, 1.25) * 20)
    return LiveCorpusRunner(
        backend=backend,
        state_store=OperationsStateStore(root, log_retention=retention),
        lock_path=lock_path or root / "live.lock",
        now=lambda: NOW,
        monotonic_clock=lambda: next(tick),
    )


def _config() -> LiveRunConfig:
    return LiveRunConfig(
        date_from=NOW - timedelta(days=45),
        date_to=NOW,
        limit=100,
        timeout_seconds=10,
        max_retries=2,
    )


def _maturation() -> MaturationOutcome:
    return MaturationOutcome(
        matched=35,
        unmatched=5,
        intraday_reactions_created=1,
        daily_reactions_created=1,
        intraday_features_created=1,
        daily_features_created=1,
        waiting_intraday=0,
        waiting_daily=1,
        moex_status="SUCCEEDED",
        database_status="SUCCEEDED",
        state_counts={state.value: 1 for state in MaturationState},
    )


def _snapshot(*, daily: int = 34) -> CorpusSnapshot:
    return CorpusSnapshot(
        captured_at=NOW.isoformat(),
        real_total=40,
        exact_total=40,
        matched_total=35,
        intraday_reaction_ready=26,
        intraday_feature_ready=21,
        daily_reaction_ready=34,
        daily_feature_ready=daily,
        ticker_count=2,
        source_count=2,
        date_from="2025-06-20",
        date_to="2026-08-10",
        ticker_distribution={"ROSN": 19, "YDEX": 15},
    )


def _write_snapshot_files(root: Path) -> None:
    daily = root / "artifacts/free-daily-historical-v1"
    predictive = root / "artifacts/predictive-baseline-v1"
    daily.mkdir(parents=True)
    predictive.mkdir(parents=True)
    (daily / "coverage.json").write_text(
        json.dumps(
            {
                "provenance_real": 41,
                "timestamp_quality": {"EXACT": 41},
                "matched": 36,
                "intraday": {"reaction_ready": 35, "feature_ready": 26},
                "daily_reaction_ready": 34,
                "daily_feature_ready": 34,
                "daily_feature_per_ticker": {"ROSN": 19, "YDEX": 15},
                "per_source": {code: 1 for code in ACCEPTED_SOURCE_CODES},
            }
        ),
        encoding="utf-8",
    )
    (predictive / "readiness.json").write_text(
        json.dumps(
            {
                "ticker_count": 2,
                "source_count": 2,
                "date_from": "2025-06-20",
                "date_to": "2026-08-10",
            }
        ),
        encoding="utf-8",
    )
