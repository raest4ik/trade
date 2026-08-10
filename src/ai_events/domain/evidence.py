from __future__ import annotations

from dataclasses import dataclass

from src.ai_events.domain.exceptions import AIOutputValidationError


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    start: int
    end: int
    warning: str | None = None


def resolve_exact_evidence(raw_content: str, evidence_text: str) -> EvidenceSpan:
    if not evidence_text:
        raise AIOutputValidationError("evidence_text must not be empty")
    positions: list[int] = []
    start = 0
    while True:
        position = raw_content.find(evidence_text, start)
        if position < 0:
            break
        positions.append(position)
        start = position + 1
    if not positions:
        raise AIOutputValidationError("evidence_text is not an exact substring of input")
    warning = None
    if len(positions) > 1:
        warning = f"duplicate evidence occurrence; selected first of {len(positions)}"
    selected = positions[0]
    return EvidenceSpan(
        start=selected,
        end=selected + len(evidence_text),
        warning=warning,
    )
