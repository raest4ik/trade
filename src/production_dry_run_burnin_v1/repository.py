from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from src.free_live_issuer_accumulation.domain import sha256_payload
from src.production_dry_run_burnin_v1.domain import (
    BurninAttemptType,
    BurninObservation,
)


class BurninLedgerIntegrityError(ValueError):
    pass


def observation_record_sha(observation: BurninObservation) -> str:
    return sha256_payload(observation.model_dump(mode="json", exclude={"record_sha"}))


def chain_observation(
    observation: BurninObservation,
    existing: list[BurninObservation],
) -> BurninObservation:
    previous_sha = existing[-1].record_sha if existing else None
    chained = observation.model_copy(
        update={
            "sequence": len(existing) + 1,
            "previous_record_sha": previous_sha,
            "record_sha": "PENDING",
        }
    )
    return chained.model_copy(update={"record_sha": observation_record_sha(chained)})


def validate_observations(observations: list[BurninObservation]) -> None:
    ids: set[str] = set()
    primary_ids: set[str] = set()
    previous_sha: str | None = None
    for expected, row in enumerate(observations, start=1):
        if row.sequence != expected:
            raise BurninLedgerIntegrityError("BURNIN_LEDGER_SEQUENCE_VIOLATION")
        if row.previous_record_sha != previous_sha:
            raise BurninLedgerIntegrityError("BURNIN_LEDGER_CHAIN_VIOLATION")
        if row.record_sha != observation_record_sha(row):
            raise BurninLedgerIntegrityError("BURNIN_LEDGER_RECORD_SHA_MISMATCH")
        if row.observation_id in ids:
            raise BurninLedgerIntegrityError("DUPLICATE_BURNIN_OBSERVATION_ID")
        ids.add(row.observation_id)
        if row.attempt_type == BurninAttemptType.PRIMARY:
            if row.primary_operation_id in primary_ids:
                raise BurninLedgerIntegrityError("DUPLICATE_PRIMARY_BURNIN_OBSERVATION")
            primary_ids.add(row.primary_operation_id)
        previous_sha = row.record_sha


class BurninObservationRepository(Protocol):
    def observations(self) -> list[BurninObservation]: ...

    def append(self, observation: BurninObservation) -> BurninObservation: ...

    def primary(self, primary_operation_id: str) -> BurninObservation | None: ...

    def get(self, observation_id: str) -> BurninObservation | None: ...


class InMemoryBurninObservationRepository:
    def __init__(self, observations: list[BurninObservation] | None = None) -> None:
        self._observations = list(observations or [])
        validate_observations(self._observations)

    def observations(self) -> list[BurninObservation]:
        validate_observations(self._observations)
        return list(self._observations)

    def append(self, observation: BurninObservation) -> BurninObservation:
        existing = self.observations()
        chained = chain_observation(observation, existing)
        validate_observations([*existing, chained])
        self._observations.append(chained)
        return chained

    def primary(self, primary_operation_id: str) -> BurninObservation | None:
        return next(
            (
                row
                for row in self.observations()
                if row.attempt_type == BurninAttemptType.PRIMARY
                and row.primary_operation_id == primary_operation_id
            ),
            None,
        )

    def get(self, observation_id: str) -> BurninObservation | None:
        return next(
            (row for row in self.observations() if row.observation_id == observation_id),
            None,
        )


class JsonlBurninObservationRepository(InMemoryBurninObservationRepository):
    def __init__(self, path: Path) -> None:
        self.path = path

    def observations(self) -> list[BurninObservation]:
        if not self.path.exists():
            return []
        rows: list[BurninObservation] = []
        try:
            with self.path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        raise BurninLedgerIntegrityError("EMPTY_BURNIN_LEDGER_LINE")
                    rows.append(BurninObservation.model_validate_json(line))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise BurninLedgerIntegrityError("BURNIN_LEDGER_INTEGRITY_FAILED") from exc
        validate_observations(rows)
        return rows

    def append(self, observation: BurninObservation) -> BurninObservation:
        existing = self.observations()
        chained = chain_observation(observation, existing)
        validate_observations([*existing, chained])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(chained.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return chained
