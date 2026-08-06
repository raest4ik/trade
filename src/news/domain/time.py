from __future__ import annotations

from datetime import UTC, datetime

from src.news.domain.exceptions import DomainError


def ensure_aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def utc_now() -> datetime:
    return datetime.now(UTC)
