from __future__ import annotations


class ApplicationError(RuntimeError):
    """Base class for application-layer failures."""


class NewsStorageError(ApplicationError):
    """Raised when news persistence fails."""
