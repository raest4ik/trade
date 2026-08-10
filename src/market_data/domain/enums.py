from __future__ import annotations

from enum import StrEnum


class MarketDataProvider(StrEnum):
    MOEX_ISS = "MOEX_ISS"


class MarketDataImportStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class CandleInterval(StrEnum):
    MINUTE = "MINUTE"


class MarketDataSetType(StrEnum):
    SECURITY = "SECURITY"
    BENCHMARK = "BENCHMARK"
