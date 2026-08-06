from __future__ import annotations


class EventAnalysisApplicationError(Exception):
    """Base event analysis application error."""


class EventAnalysisNewsNotFoundError(EventAnalysisApplicationError):
    """Raised when the news item does not exist."""


class EventAnalysisStorageError(EventAnalysisApplicationError):
    """Raised when event analysis storage fails."""
