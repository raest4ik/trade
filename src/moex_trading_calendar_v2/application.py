from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

from src.moex_trading_calendar_v2.domain import (
    BOARD_SCOPE,
    MARKET_SCOPE,
    MoexSessionEvidenceV2,
    RuntimeSessionStatus,
    SessionKind,
    SessionStatus,
)

MOEX_TIMEZONE = ZoneInfo("Europe/Moscow")


def resolve_schedule_payload(
    payload: dict[str, Any],
    calendar_date: date,
    *,
    checked_at: datetime,
    evidence_sha: str,
    runtime_status: RuntimeSessionStatus = RuntimeSessionStatus.NOT_APPLICABLE,
    runtime_evidence_sha: str | None = None,
) -> MoexSessionEvidenceV2:
    try:
        market = payload["market"]
        board = payload["board"]
        raw_days = payload["days"]
        raw_holidays = payload["holidays"]
        windows = payload["windows"]
        if market != MARKET_SCOPE or board != BOARD_SCOPE:
            return unknown_evidence(
                calendar_date,
                checked_at=checked_at,
                evidence_sha=evidence_sha,
                reason="WRONG_MARKET_SCOPE",
            )
        if not isinstance(raw_days, dict) or not isinstance(raw_holidays, dict):
            raise TypeError("calendar collections must be objects")
        days = cast("dict[str, Any]", raw_days)
        holidays = cast("dict[str, Any]", raw_holidays)
        raw_row = days.get(calendar_date.isoformat())
        if not isinstance(raw_row, dict):
            raise KeyError("calendar date missing")
        row = cast("dict[str, Any]", raw_row)
        if row.get("tradedate") != calendar_date.isoformat():
            raise ValueError("calendar date mismatch")
        is_traded = row.get("is_traded")
        reason_code = row.get("reason")
        if is_traded not in (0, 1) or not isinstance(reason_code, str):
            raise ValueError("calendar row malformed")
        if is_traded == 0:
            if runtime_status == RuntimeSessionStatus.OPEN:
                return unknown_evidence(
                    calendar_date,
                    checked_at=checked_at,
                    evidence_sha=evidence_sha,
                    reason="PLANNED_RUNTIME_CONTRADICTION",
                    runtime_status=runtime_status,
                    runtime_evidence_sha=runtime_evidence_sha,
                )
            return MoexSessionEvidenceV2(
                calendar_date=calendar_date,
                status=SessionStatus.CLOSED,
                session_kind=SessionKind.CLOSED,
                checked_at=checked_at,
                evidence_sha=evidence_sha,
                reason="AUTHORITATIVE_SCHEDULE_CLOSED",
                runtime_status=runtime_status,
                runtime_source=_runtime_source(runtime_status),
                runtime_evidence_sha=runtime_evidence_sha,
            )
        kind = _session_kind(calendar_date, reason_code, holidays)
        business_date = (
            calendar_date
            if reason_code == "N"
            else _next_regular_business_date(calendar_date, days)
        )
        if business_date is None:
            raise ValueError("business date unresolved")
        window_key = "weekday" if kind == SessionKind.REGULAR else "weekend"
        scheduled_open, scheduled_close = _session_window(calendar_date, windows, window_key)
        return MoexSessionEvidenceV2(
            calendar_date=calendar_date,
            status=SessionStatus.OPEN,
            session_kind=kind,
            moex_business_date=business_date,
            scheduled_open_at=scheduled_open,
            scheduled_close_at=scheduled_close,
            checked_at=checked_at,
            evidence_sha=evidence_sha,
            reason="AUTHORITATIVE_SCHEDULE_OPEN",
            runtime_status=runtime_status,
            runtime_source=_runtime_source(runtime_status),
            runtime_evidence_sha=runtime_evidence_sha,
        )
    except (KeyError, TypeError, ValueError):
        return unknown_evidence(
            calendar_date,
            checked_at=checked_at,
            evidence_sha=evidence_sha,
            reason="AUTHORITATIVE_CALENDAR_INVALID",
            runtime_status=runtime_status,
            runtime_evidence_sha=runtime_evidence_sha,
        )


def unknown_evidence(
    calendar_date: date,
    *,
    checked_at: datetime,
    evidence_sha: str | None,
    reason: str,
    runtime_status: RuntimeSessionStatus = RuntimeSessionStatus.NOT_APPLICABLE,
    runtime_evidence_sha: str | None = None,
) -> MoexSessionEvidenceV2:
    return MoexSessionEvidenceV2(
        calendar_date=calendar_date,
        status=SessionStatus.UNKNOWN,
        session_kind=SessionKind.UNKNOWN,
        checked_at=checked_at,
        evidence_sha=evidence_sha,
        reason=reason,
        runtime_status=runtime_status,
        runtime_source=_runtime_source(runtime_status),
        runtime_evidence_sha=runtime_evidence_sha,
    )


def distinct_moex_trading_days(evidence: Sequence[MoexSessionEvidenceV2]) -> int:
    return len(
        {
            row.moex_business_date
            for row in evidence
            if row.status == SessionStatus.OPEN and row.moex_business_date is not None
        }
    )


def _session_kind(
    calendar_date: date,
    reason_code: str,
    holidays: dict[str, Any],
) -> SessionKind:
    if reason_code == "N":
        return SessionKind.REGULAR
    if holidays.get(calendar_date.isoformat()) == "holiday":
        return SessionKind.HOLIDAY_ADDITIONAL
    if reason_code == "W" and calendar_date.weekday() >= 5:
        return SessionKind.WEEKEND_ADDITIONAL
    if reason_code in {"W", "H"}:
        return SessionKind.SPECIAL
    raise ValueError("unknown session reason")


def _next_regular_business_date(calendar_date: date, days: dict[str, Any]) -> date | None:
    later = sorted(day for day in days if day > calendar_date.isoformat())
    for value in later:
        raw_row = days[value]
        if not isinstance(raw_row, dict):
            continue
        row = cast("dict[str, Any]", raw_row)
        if row.get("is_traded") == 1 and row.get("reason") == "N":
            return date.fromisoformat(value)
    return None


def _session_window(
    calendar_date: date,
    windows: object,
    window_key: str,
) -> tuple[datetime, datetime]:
    if not isinstance(windows, dict):
        raise TypeError("windows must be an object")
    typed_windows = cast("dict[str, Any]", windows)
    raw_intervals = typed_windows.get(window_key)
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise ValueError("session window missing")
    intervals = cast("list[Any]", raw_intervals)
    first = intervals[0]
    last = intervals[-1]
    if not isinstance(first, dict) or not isinstance(last, dict):
        raise TypeError("session interval malformed")
    first = cast("dict[str, Any]", first)
    last = cast("dict[str, Any]", last)
    opened = datetime.combine(
        calendar_date,
        time.fromisoformat(str(first["from"])),
        tzinfo=MOEX_TIMEZONE,
    )
    closed = datetime.combine(
        calendar_date,
        time.fromisoformat(str(last["till"])),
        tzinfo=MOEX_TIMEZONE,
    )
    if closed <= opened:
        closed += timedelta(days=1)
    return opened, closed


def _runtime_source(status: RuntimeSessionStatus) -> str | None:
    if status == RuntimeSessionStatus.NOT_APPLICABLE:
        return None
    return "MOEX_ISS_CURRENT_TQBR_BOARD"
