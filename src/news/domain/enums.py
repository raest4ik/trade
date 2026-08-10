from __future__ import annotations

from enum import StrEnum


class PublicationTimestampQuality(StrEnum):
    EXACT = "EXACT"
    DATE_ONLY = "DATE_ONLY"
    UNKNOWN = "UNKNOWN"
