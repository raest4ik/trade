from __future__ import annotations


class ReactionApplicationError(Exception):
    """Base reaction application error."""


class ReactionStorageError(ReactionApplicationError):
    """Raised when reaction storage fails."""


class ReactionNewsNotFoundError(ReactionApplicationError):
    """Raised when the news item does not exist."""


class ReactionMissingInstrumentMatchesError(ReactionApplicationError):
    """Raised when matching was not run before reaction calculation."""


class ReactionTimestampIneligibleError(ReactionApplicationError):
    """Raised when publication time is not trusted enough for reaction labels."""
