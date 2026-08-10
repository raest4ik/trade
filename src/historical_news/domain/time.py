from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.news.domain.enums import PublicationTimestampQuality

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class ParsedPublicationTimestamp:
    published_at: datetime | None
    quality: PublicationTimestampQuality
    source_timezone: str | None
    error: str | None = None


def parse_publication_timestamp(
    value: str,
    *,
    source_timezone: str | None,
) -> ParsedPublicationTimestamp:
    normalized = value.strip()
    if not normalized:
        return ParsedPublicationTimestamp(
            published_at=None,
            quality=PublicationTimestampQuality.UNKNOWN,
            source_timezone=source_timezone,
            error="timestamp_empty",
        )
    if _DATE_ONLY.fullmatch(normalized):
        try:
            parsed_date = date.fromisoformat(normalized)
        except ValueError:
            return _unknown(source_timezone, "timestamp_invalid")
        return ParsedPublicationTimestamp(
            published_at=datetime.combine(parsed_date, time.min, tzinfo=UTC),
            quality=PublicationTimestampQuality.DATE_ONLY,
            source_timezone=source_timezone,
        )
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return _unknown(source_timezone, "timestamp_invalid")
    if parsed.tzinfo is not None:
        return ParsedPublicationTimestamp(
            published_at=parsed.astimezone(UTC),
            quality=PublicationTimestampQuality.EXACT,
            source_timezone=source_timezone or str(parsed.tzinfo),
        )
    if source_timezone is None or not source_timezone.strip():
        return _unknown(None, "timezone_unknown")
    try:
        timezone = ZoneInfo(source_timezone)
    except ZoneInfoNotFoundError:
        return _unknown(source_timezone, "timezone_invalid")
    return ParsedPublicationTimestamp(
        published_at=parsed.replace(tzinfo=timezone).astimezone(UTC),
        quality=PublicationTimestampQuality.EXACT,
        source_timezone=source_timezone,
    )


def _unknown(source_timezone: str | None, error: str) -> ParsedPublicationTimestamp:
    return ParsedPublicationTimestamp(
        published_at=None,
        quality=PublicationTimestampQuality.UNKNOWN,
        source_timezone=source_timezone,
        error=error,
    )
