from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast
from uuid import uuid4

from src.live_corpus_operations.domain import (
    ACCEPTED_SOURCE_CODES,
    OPERATIONS_VERSION,
    CorpusSnapshot,
    LiveRunConfig,
    LiveRunReport,
    MaturationOutcome,
    RunStatus,
    SourceOutcome,
)
from src.live_corpus_operations.state import FileJobLock, OperationsStateStore, sanitize_error


class LiveCorpusBackend(Protocol):
    def validate_source(self, source_code: str) -> None: ...

    async def ingest_source(
        self,
        source_code: str,
        config: LiveRunConfig,
        checkpoint: dict[str, object],
    ) -> SourceOutcome: ...

    async def mature(self, config: LiveRunConfig) -> MaturationOutcome: ...

    def snapshot(self) -> CorpusSnapshot: ...


class LiveCorpusRunner:
    def __init__(
        self,
        *,
        backend: LiveCorpusBackend,
        state_store: OperationsStateStore,
        lock_path: Path,
        now: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._backend = backend
        self._state_store = state_store
        self._lock_path = lock_path
        self._now = now or (lambda: datetime.now(UTC))
        self._monotonic = monotonic_clock or monotonic

    async def execute(self, config: LiveRunConfig) -> LiveRunReport:
        config.validate()
        started = self._now()
        run_id = uuid4().hex
        lock = FileJobLock(self._lock_path)
        if not lock.try_acquire(now=started):
            return self._report(
                run_id=run_id,
                status=RunStatus.ALREADY_RUNNING,
                started=started,
                duration=0.0,
                config=config,
                sources=(),
                maturation=MaturationOutcome(),
                snapshot=self._backend.snapshot(),
                errors=(),
            )
        timer = self._monotonic()
        try:
            checkpoints = self._state_store.checkpoints()
            raw_source_checkpoints = checkpoints.get("sources", {})
            source_checkpoints = (
                cast("dict[str, Any]", raw_source_checkpoints)
                if isinstance(raw_source_checkpoints, dict)
                else {}
            )
            source_results: list[SourceOutcome] = []
            errors: list[str] = []
            for source_code in ACCEPTED_SOURCE_CODES:
                try:
                    self._backend.validate_source(source_code)
                    raw_checkpoint = source_checkpoints.get(source_code, {})
                    checkpoint = (
                        cast("dict[str, object]", raw_checkpoint)
                        if isinstance(raw_checkpoint, dict)
                        else {}
                    )
                    source_results.append(
                        await self._backend.ingest_source(
                            source_code,
                            config,
                            checkpoint,
                        )
                    )
                except Exception as exc:
                    error = sanitize_error(exc)
                    errors.append(f"{source_code}: {error}")
                    source_results.append(
                        SourceOutcome(source_code=source_code, status="FAILED", error=error)
                    )
            try:
                maturation = await self._backend.mature(config)
                errors.extend(maturation.errors)
                snapshot = self._backend.snapshot()
            except Exception as exc:
                error = sanitize_error(exc)
                errors.append(f"pipeline: {error}")
                maturation = MaturationOutcome(
                    database_status="FAILED",
                    moex_status="FAILED",
                    errors=(error,),
                )
                snapshot = self._backend.snapshot()
            successful_sources = sum(item.status == "SUCCEEDED" for item in source_results)
            if config.dry_run:
                status = RunStatus.DRY_RUN
            elif successful_sources == len(ACCEPTED_SOURCE_CODES) and not errors:
                status = RunStatus.SUCCEEDED
            elif successful_sources or maturation.database_status == "SUCCEEDED":
                status = RunStatus.PARTIAL
            else:
                status = RunStatus.FAILED
            report = self._report(
                run_id=run_id,
                status=status,
                started=started,
                duration=max(0.0, self._monotonic() - timer),
                config=config,
                sources=tuple(source_results),
                maturation=maturation,
                snapshot=snapshot,
                errors=tuple(errors),
            )
            if not config.dry_run:
                self._state_store.persist(
                    report,
                    successful=status == RunStatus.SUCCEEDED,
                )
            return report
        finally:
            lock.release()

    def _report(
        self,
        *,
        run_id: str,
        status: RunStatus,
        started: datetime,
        duration: float,
        config: LiveRunConfig,
        sources: tuple[SourceOutcome, ...],
        maturation: MaturationOutcome,
        snapshot: CorpusSnapshot,
        errors: tuple[str, ...],
    ) -> LiveRunReport:
        finished = self._now()
        return LiveRunReport(
            schema_version=OPERATIONS_VERSION,
            run_id=run_id,
            status=status,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            duration_seconds=round(duration, 6),
            dry_run=config.dry_run,
            sources_checked=len(sources),
            source_results=sources,
            maturation=maturation,
            snapshot=snapshot,
            errors=errors,
        )
