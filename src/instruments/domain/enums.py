from __future__ import annotations

from enum import StrEnum


class InstrumentType(StrEnum):
    COMMON_STOCK = "COMMON_STOCK"
    PREFERRED_STOCK = "PREFERRED_STOCK"


class AliasType(StrEnum):
    OFFICIAL_NAME = "OFFICIAL_NAME"
    SHORT_NAME = "SHORT_NAME"
    BRAND = "BRAND"
    TICKER = "TICKER"
    LEGAL_NAME = "LEGAL_NAME"
    MANUAL = "MANUAL"


class MatchType(StrEnum):
    EXACT_TICKER = "EXACT_TICKER"
    EXACT_ALIAS = "EXACT_ALIAS"
