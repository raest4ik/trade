from __future__ import annotations


class HistoricalNewsApplicationError(Exception):
    """Base historical news application error."""


class HistoricalNewsStorageError(HistoricalNewsApplicationError):
    """Raised when historical news persistence fails."""


class HistoricalNewsIngestionError(HistoricalNewsApplicationError):
    """Raised when historical source ingestion cannot complete."""
