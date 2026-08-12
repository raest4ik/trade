from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

OPERATIONS_VERSION = "live-corpus-operations-v1"
ACCEPTED_SOURCE_CODES = (
    "ROSNEFT_PRESS_RELEASES_RSS",
    "YANDEX_IR_PRESS_RELEASES_RSS",
)
SOURCE_TICKERS = {
    "ROSNEFT_PRESS_RELEASES_RSS": "ROSN",
    "YANDEX_IR_PRESS_RELEASES_RSS": "YDEX",
}
TELEGRAM_API_POLICY = "REJECTED_POLICY_FOR_ML"
DEFAULT_CADENCE_MINUTES = 60
DEFAULT_LIMIT = 100
DEFAULT_LOOKBACK_DAYS = 45
DEFAULT_LOG_RETENTION = 30
LOW_TICKER_DIVERSITY_THRESHOLD = 0.70


def _empty_counts() -> dict[str, int]:
    return {}


class RunStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    ALREADY_RUNNING = "ALREADY_RUNNING"
    DRY_RUN = "DRY_RUN"


class MaturationState(StrEnum):
    INGESTED = "INGESTED"
    MATCHED = "MATCHED"
    WAITING_INTRADAY_TARGET = "WAITING_INTRADAY_TARGET"
    INTRADAY_READY = "INTRADAY_READY"
    WAITING_DAILY_TARGET = "WAITING_DAILY_TARGET"
    DAILY_READY = "DAILY_READY"
    FEATURE_READY = "FEATURE_READY"


@dataclass(frozen=True, slots=True)
class LiveRunConfig:
    date_from: datetime
    date_to: datetime
    limit: int = DEFAULT_LIMIT
    timeout_seconds: float = 10.0
    max_retries: int = 2
    dry_run: bool = False

    def validate(self) -> None:
        if self.date_to < self.date_from:
            raise ValueError("date_to must not be before date_from")
        if not 1 <= self.limit <= 100:
            raise ValueError("live run limit must be between 1 and 100")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")


@dataclass(frozen=True, slots=True)
class SourceOutcome:
    source_code: str
    status: str
    items_seen: int = 0
    items_imported: int = 0
    duplicates: int = 0
    matched: int = 0
    rejected: int = 0
    last_item_id: str | None = None
    last_item_at: str | None = None
    error: str | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MaturationOutcome:
    matched: int = 0
    unmatched: int = 0
    intraday_reactions_created: int = 0
    daily_reactions_created: int = 0
    intraday_features_created: int = 0
    daily_features_created: int = 0
    waiting_intraday: int = 0
    waiting_daily: int = 0
    moex_status: str = "NOT_RUN"
    database_status: str = "UNKNOWN"
    state_counts: dict[str, int] = field(default_factory=_empty_counts)
    errors: tuple[str, ...] = ()

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["errors"] = list(self.errors)
        return payload


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    captured_at: str
    real_total: int
    exact_total: int
    matched_total: int
    intraday_reaction_ready: int
    intraday_feature_ready: int
    daily_reaction_ready: int
    daily_feature_ready: int
    ticker_count: int
    source_count: int
    date_from: str | None
    date_to: str | None
    ticker_distribution: dict[str, int] = field(default_factory=_empty_counts)

    def warnings(self) -> tuple[str, ...]:
        if self.daily_feature_ready < 100 or not self.ticker_distribution:
            return ()
        concentration = max(self.ticker_distribution.values()) / max(
            sum(self.ticker_distribution.values()), 1
        )
        return ("LOW_TICKER_DIVERSITY",) if concentration > LOW_TICKER_DIVERSITY_THRESHOLD else ()

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings())
        return payload


@dataclass(frozen=True, slots=True)
class LiveRunReport:
    schema_version: str
    run_id: str
    status: RunStatus
    started_at: str
    finished_at: str
    duration_seconds: float
    dry_run: bool
    sources_checked: int
    source_results: tuple[SourceOutcome, ...]
    maturation: MaturationOutcome
    snapshot: CorpusSnapshot
    errors: tuple[str, ...]
    automatic_training: bool = False
    zero_cost: bool = True
    telegram_api: str = TELEGRAM_API_POLICY

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "dry_run": self.dry_run,
            "sources_checked": self.sources_checked,
            "source_results": [item.payload() for item in self.source_results],
            "items_seen": sum(item.items_seen for item in self.source_results),
            "new_real": sum(item.items_imported for item in self.source_results),
            "duplicates": sum(item.duplicates for item in self.source_results),
            "matched": self.maturation.matched,
            "unmatched": self.maturation.unmatched,
            "reactions_created": (
                self.maturation.intraday_reactions_created + self.maturation.daily_reactions_created
            ),
            "features_created": (
                self.maturation.intraday_features_created + self.maturation.daily_features_created
            ),
            "maturation": self.maturation.payload(),
            "snapshot": self.snapshot.payload(),
            "errors": list(self.errors),
            "automatic_training": self.automatic_training,
            "zero_cost": self.zero_cost,
            "telegram_api": self.telegram_api,
        }
