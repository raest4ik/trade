from __future__ import annotations

from enum import StrEnum


class AIProvider(StrEnum):
    OLLAMA = "ollama"
    OPENAI = "openai"
