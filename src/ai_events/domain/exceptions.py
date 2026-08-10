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


class OllamaUnavailableError(AIModelTransientError):
    error_code = "OLLAMA_UNAVAILABLE"

    def __init__(self, base_url: str) -> None:
        self.public_message = f"Ollama is unavailable at {base_url}"
        super().__init__(self.public_message)


class OllamaModelNotFoundError(AIModelError):
    error_code = "OLLAMA_MODEL_NOT_FOUND"

    def __init__(self, model: str) -> None:
        self.public_message = f"Ollama model {model} is not installed; run: ollama pull {model}"
        super().__init__(self.public_message)


class OllamaInvalidStructuredOutputError(AIModelTransientError):
    error_code = "OLLAMA_INVALID_STRUCTURED_OUTPUT"
    public_message = "Ollama returned invalid structured output"

    def __init__(self) -> None:
        super().__init__(self.public_message)


class OllamaTimeoutError(AIModelTransientError):
    error_code = "OLLAMA_TIMEOUT"
    public_message = "Ollama request timed out"

    def __init__(self) -> None:
        super().__init__(self.public_message)
