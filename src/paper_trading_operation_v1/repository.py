from __future__ import annotations

import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from src.paper_trading_operation_v1.domain import (
    OperationAuditEvent,
    OperationAuditRecordType,
    PaperOperationRun,
)


class OperationAuditIntegrityError(ValueError):
    pass


class OperationAlreadyRunningError(RuntimeError):
    pass


def validate_operation_events(events: list[OperationAuditEvent]) -> None:
    record_ids: set[str] = set()
    completed: set[str] = set()
    prepared: set[str] = set()
    for expected, event in enumerate(events, start=1):
        if event.sequence != expected:
            raise OperationAuditIntegrityError("OPERATION_AUDIT_SEQUENCE_VIOLATION")
        if event.record_id in record_ids:
            raise OperationAuditIntegrityError("DUPLICATE_OPERATION_AUDIT_RECORD_ID")
        record_ids.add(event.record_id)
        if event.record_type == OperationAuditRecordType.PREPARED:
            if event.operation_id in prepared or event.operation_id in completed:
                raise OperationAuditIntegrityError("DUPLICATE_OPERATION_PREPARED")
            prepared.add(event.operation_id)
        else:
            if event.operation_id in completed:
                raise OperationAuditIntegrityError("DUPLICATE_OPERATION_COMPLETION")
            PaperOperationRun.model_validate(event.payload)
            completed.add(event.operation_id)


class OperationAuditRepository(Protocol):
    def events(self) -> list[OperationAuditEvent]: ...

    def append(self, event: OperationAuditEvent) -> None: ...

    def latest(self, operation_id: str) -> OperationAuditEvent | None: ...

    def completed_run(self, operation_id: str) -> PaperOperationRun | None: ...


class InMemoryOperationAuditRepository:
    def __init__(self, events: list[OperationAuditEvent] | None = None) -> None:
        self._events = list(events or [])
        validate_operation_events(self._events)

    def events(self) -> list[OperationAuditEvent]:
        validate_operation_events(self._events)
        return list(self._events)

    def append(self, event: OperationAuditEvent) -> None:
        validate_operation_events([*self._events, event])
        self._events.append(event)

    def latest(self, operation_id: str) -> OperationAuditEvent | None:
        return next(
            (row for row in reversed(self.events()) if row.operation_id == operation_id), None
        )

    def completed_run(self, operation_id: str) -> PaperOperationRun | None:
        event = self.latest(operation_id)
        if event is None or event.record_type == OperationAuditRecordType.PREPARED:
            return None
        return PaperOperationRun.model_validate(event.payload)


class JsonlOperationAuditRepository(InMemoryOperationAuditRepository):
    def __init__(self, path: Path) -> None:
        self.path = path

    def events(self) -> list[OperationAuditEvent]:
        if not self.path.exists():
            return []
        rows: list[OperationAuditEvent] = []
        try:
            with self.path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        raise OperationAuditIntegrityError("EMPTY_OPERATION_AUDIT_LINE")
                    rows.append(OperationAuditEvent.model_validate_json(line))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise OperationAuditIntegrityError("OPERATION_AUDIT_INTEGRITY_FAILURE") from exc
        validate_operation_events(rows)
        return rows

    def append(self, event: OperationAuditEvent) -> None:
        validate_operation_events([*self.events(), event])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(event.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())


@contextmanager
def operation_lock(
    path: Path,
    *,
    operation_id: str,
    stale_after: timedelta,
    now: datetime | None = None,
) -> Generator[None]:
    acquired_at = now or datetime.now(UTC)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(str(payload["created_at"]).replace("Z", "+00:00"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise OperationAlreadyRunningError("OPERATION_ALREADY_RUNNING") from exc
        if acquired_at - created <= stale_after:
            raise OperationAlreadyRunningError("OPERATION_ALREADY_RUNNING")
        path.unlink()
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(
            descriptor,
            json.dumps(
                {"operation_id": operation_id, "created_at": acquired_at.isoformat()},
                sort_keys=True,
            ).encode("utf-8"),
        )
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        yield
    except FileExistsError as exc:
        raise OperationAlreadyRunningError("OPERATION_ALREADY_RUNNING") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            if path.exists():
                current = json.loads(path.read_text(encoding="utf-8"))
                if current.get("operation_id") == operation_id:
                    path.unlink()
        except (OSError, json.JSONDecodeError):
            pass
