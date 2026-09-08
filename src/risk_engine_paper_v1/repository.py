from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from src.risk_engine_paper_v1.domain import LedgerEvent


class PaperLedgerRepository(Protocol):
    def events(self) -> list[LedgerEvent]: ...

    def append(self, event: LedgerEvent) -> None: ...

    def contains_idempotency_key(self, key: str) -> bool: ...


class InMemoryPaperLedgerRepository:
    def __init__(self) -> None:
        self._events: list[LedgerEvent] = []

    def events(self) -> list[LedgerEvent]:
        return list(self._events)

    def append(self, event: LedgerEvent) -> None:
        if event.sequence != len(self._events) + 1:
            raise ValueError("LEDGER_SEQUENCE_VIOLATION")
        if event.idempotency_key and self.contains_idempotency_key(event.idempotency_key):
            raise ValueError("DUPLICATE_LEDGER_IDEMPOTENCY_KEY")
        self._events.append(event)

    def contains_idempotency_key(self, key: str) -> bool:
        return any(event.idempotency_key == key for event in self._events)


class JsonlPaperLedgerRepository:
    """Append-only operational paper state, separate from immutable run artifacts."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def events(self) -> list[LedgerEvent]:
        if not self.path.exists():
            return []
        rows: list[LedgerEvent] = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(LedgerEvent.model_validate_json(line))
        return rows

    def append(self, event: LedgerEvent) -> None:
        existing = self.events()
        if event.sequence != len(existing) + 1:
            raise ValueError("LEDGER_SEQUENCE_VIOLATION")
        if event.idempotency_key and any(
            row.idempotency_key == event.idempotency_key for row in existing
        ):
            raise ValueError("DUPLICATE_LEDGER_IDEMPOTENCY_KEY")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(event.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def contains_idempotency_key(self, key: str) -> bool:
        return any(event.idempotency_key == key for event in self.events())
