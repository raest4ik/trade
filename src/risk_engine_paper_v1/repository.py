from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from src.risk_engine_paper_v1.domain import (
    LedgerEvent,
    LedgerEventType,
    PaperOrder,
    PaperOrderStatus,
    PaperPortfolio,
)


class LedgerIntegrityError(ValueError):
    pass


def validate_ledger_events(events: list[LedgerEvent]) -> None:
    if not events:
        return
    event_ids: set[str] = set()
    idempotency_keys: set[str] = set()
    portfolio_id = events[0].portfolio_id
    portfolio_created = 0
    for expected_sequence, event in enumerate(events, start=1):
        if event.sequence != expected_sequence:
            raise LedgerIntegrityError("LEDGER_SEQUENCE_VIOLATION")
        if event.event_id in event_ids:
            raise LedgerIntegrityError("DUPLICATE_LEDGER_EVENT_ID")
        event_ids.add(event.event_id)
        if event.portfolio_id != portfolio_id:
            raise LedgerIntegrityError("LEDGER_PORTFOLIO_ID_MISMATCH")
        if event.idempotency_key:
            if event.idempotency_key in idempotency_keys:
                raise LedgerIntegrityError("DUPLICATE_LEDGER_IDEMPOTENCY_KEY")
            idempotency_keys.add(event.idempotency_key)
        try:
            if event.event_type == LedgerEventType.PORTFOLIO_CREATED:
                portfolio_created += 1
                if expected_sequence != 1 or portfolio_created != 1:
                    raise LedgerIntegrityError("DUPLICATE_PORTFOLIO_CREATED")
                portfolio = PaperPortfolio.model_validate(event.payload.get("portfolio"))
                if portfolio.portfolio_id != event.portfolio_id:
                    raise LedgerIntegrityError("LEDGER_PORTFOLIO_ID_MISMATCH")
            elif event.event_type == LedgerEventType.PAPER_ORDER_FILLED:
                if portfolio_created != 1:
                    raise LedgerIntegrityError("MISSING_PORTFOLIO_CREATED")
                order = PaperOrder.model_validate(event.payload.get("order"))
                if order.idempotency_key != event.idempotency_key:
                    raise LedgerIntegrityError("LEDGER_IDEMPOTENCY_PAYLOAD_MISMATCH")
                if order.status != PaperOrderStatus.FILLED or order.executed_at is None:
                    raise LedgerIntegrityError("LEDGER_FILL_STATUS_INVALID")
            elif event.event_type == LedgerEventType.DAY_CLOSED:
                if portfolio_created != 1:
                    raise LedgerIntegrityError("MISSING_PORTFOLIO_CREATED")
            else:
                raise LedgerIntegrityError("UNSUPPORTED_LEDGER_EVENT_TYPE")
        except ValidationError as exc:
            raise LedgerIntegrityError("LEDGER_PAYLOAD_INVALID") from exc
    if portfolio_created != 1:
        raise LedgerIntegrityError("MISSING_PORTFOLIO_CREATED")


class PaperLedgerRepository(Protocol):
    def events(self) -> list[LedgerEvent]: ...

    def append(self, event: LedgerEvent) -> None: ...

    def contains_idempotency_key(self, key: str) -> bool: ...

    def last_sequence(self) -> int: ...


class InMemoryPaperLedgerRepository:
    def __init__(self, events: list[LedgerEvent] | None = None) -> None:
        self._events = list(events or [])
        validate_ledger_events(self._events)

    def events(self) -> list[LedgerEvent]:
        validate_ledger_events(self._events)
        return list(self._events)

    def append(self, event: LedgerEvent) -> None:
        validate_ledger_events([*self._events, event])
        self._events.append(event)

    def contains_idempotency_key(self, key: str) -> bool:
        return any(event.idempotency_key == key for event in self._events)

    def last_sequence(self) -> int:
        return len(self.events())


class JsonlPaperLedgerRepository:
    """Append-only operational paper state, separate from immutable run artifacts."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def events(self) -> list[LedgerEvent]:
        if not self.path.exists():
            return []
        rows: list[LedgerEvent] = []
        try:
            with self.path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        raise LedgerIntegrityError("EMPTY_LEDGER_LINE")
                    rows.append(LedgerEvent.model_validate_json(line))
        except (OSError, ValidationError) as exc:
            raise LedgerIntegrityError("LEDGER_INTEGRITY_FAILURE") from exc
        validate_ledger_events(rows)
        return rows

    def append(self, event: LedgerEvent) -> None:
        existing = self.events()
        validate_ledger_events([*existing, event])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(event.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def contains_idempotency_key(self, key: str) -> bool:
        return any(event.idempotency_key == key for event in self.events())

    def last_sequence(self) -> int:
        return len(self.events())
