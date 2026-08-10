from __future__ import annotations


class AIEventError(Exception):
    """Base error for AI event extraction."""


class AIOutputValidationError(AIEventError):
    """The model output violates a domain invariant."""


class AIModelError(AIEventError):
    """A sanitized non-retryable model failure."""


class AIModelTransientError(AIModelError):
    """A sanitized model failure that may be retried."""


class AIConfigurationError(AIEventError):
    """The runtime configuration cannot execute an AI request."""
