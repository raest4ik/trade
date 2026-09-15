from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

SESSION_POLICY_VERSION = "authoritative-moex-trading-calendar-v2"
CALENDAR_SOURCE = "MOEX_OFFICIAL_TRADING_CALENDAR"
CALENDAR_SOURCE_URL = "https://www.moex.com/ru/tradingcalendar"
RUNTIME_SOURCE = "MOEX_ISS_CURRENT_TQBR_BOARD"
MARKET_SCOPE = "stock"
BOARD_SCOPE = "TQBR"


class SessionStatus(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class SessionKind(StrEnum):
    REGULAR = "REGULAR"
    WEEKEND_ADDITIONAL = "WEEKEND_ADDITIONAL"
    HOLIDAY_ADDITIONAL = "HOLIDAY_ADDITIONAL"
    SPECIAL = "SPECIAL"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class RuntimeSessionStatus(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class MoexSessionEvidenceV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = SESSION_POLICY_VERSION
    calendar_date: date
    market: str = MARKET_SCOPE
    board: str = BOARD_SCOPE
    status: SessionStatus
    session_kind: SessionKind
    moex_business_date: date | None = None
    scheduled_open_at: datetime | None = None
    scheduled_close_at: datetime | None = None
    checked_at: datetime
    source: str = CALENDAR_SOURCE
    source_url: str = CALENDAR_SOURCE_URL
    source_id: str = "offDays.stock"
    source_published_at: datetime | None = None
    schedule_version: str | None = None
    effective_at: datetime | None = None
    evidence_sha: str | None = None
    reason: str
    runtime_status: RuntimeSessionStatus = RuntimeSessionStatus.NOT_APPLICABLE
    runtime_source: str | None = None
    runtime_evidence_sha: str | None = None

    @property
    def is_session_scheduled(self) -> bool:
        return self.status == SessionStatus.OPEN

    def is_within_session_window(self, value: datetime) -> bool:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("SESSION_CHECK_TIME_MUST_BE_TIMEZONE_AWARE")
        return bool(
            self.status == SessionStatus.OPEN
            and self.scheduled_open_at is not None
            and self.scheduled_close_at is not None
            and self.scheduled_open_at <= value <= self.scheduled_close_at
        )

    @model_validator(mode="after")
    def validate_semantics(self) -> MoexSessionEvidenceV2:
        for value in (
            self.checked_at,
            self.scheduled_open_at,
            self.scheduled_close_at,
            self.source_published_at,
            self.effective_at,
        ):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("SESSION_TIMESTAMPS_MUST_BE_TIMEZONE_AWARE")
        if self.status == SessionStatus.OPEN:
            if self.moex_business_date is None:
                raise ValueError("OPEN_SESSION_REQUIRES_BUSINESS_DATE")
            if self.scheduled_open_at is None or self.scheduled_close_at is None:
                raise ValueError("OPEN_SESSION_REQUIRES_WINDOW")
            if self.scheduled_close_at <= self.scheduled_open_at:
                raise ValueError("SESSION_WINDOW_INVALID")
        return self
