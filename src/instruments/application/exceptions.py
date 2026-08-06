from __future__ import annotations


class InstrumentApplicationError(RuntimeError):
    """Base class for instrument application failures."""


class InstrumentStorageError(InstrumentApplicationError):
    """Raised when instrument persistence fails."""


class InstrumentNotFoundError(InstrumentApplicationError):
    """Raised when a referenced instrument does not exist."""


class NewsForMatchingNotFoundError(InstrumentApplicationError):
    """Raised when a news item cannot be found for matching."""
