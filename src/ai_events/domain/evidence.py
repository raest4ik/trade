from __future__ import annotations

from dataclasses import dataclass

from src.ai_events.domain.exceptions import AIOutputValidationError


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    start: int | None
    end: int | None
    valid: bool
    warning: str | None = None


UNRESOLVED_EVIDENCE_OFFSET = -1

_QUOTE_EQUIVALENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
}


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
        valid=True,
        warning=warning,
    )


def resolve_evidence(raw_content: str, evidence_text: str) -> EvidenceSpan:
    try:
        return resolve_exact_evidence(raw_content, evidence_text)
    except AIOutputValidationError:
        pass
    normalized_raw, offsets = _normalize_with_offsets(raw_content)
    normalized_evidence, _ = _normalize_with_offsets(evidence_text)
    positions = _positions(normalized_raw, normalized_evidence)
    if positions and normalized_evidence:
        selected = positions[0]
        end_index = selected + len(normalized_evidence) - 1
        warning = "evidence aligned after safe whitespace/unicode normalization"
        if len(positions) > 1:
            warning += f"; selected first of {len(positions)}"
        return EvidenceSpan(
            start=offsets[selected][0],
            end=offsets[end_index][1],
            valid=True,
            warning=warning,
        )
    return EvidenceSpan(
        start=None,
        end=None,
        valid=False,
        warning="evidence is not an exact or safely normalized substring; offsets unavailable",
    )


def _positions(value: str, substring: str) -> list[int]:
    if not substring:
        return []
    result: list[int] = []
    start = 0
    while True:
        position = value.find(substring, start)
        if position < 0:
            return result
        result.append(position)
        start = position + 1


def _normalize_with_offsets(value: str) -> tuple[str, list[tuple[int, int]]]:
    normalized: list[str] = []
    offsets: list[tuple[int, int]] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character.isspace():
            start = index
            index += 1
            while index < len(value) and value[index].isspace():
                index += 1
            normalized.append(" ")
            offsets.append((start, index))
            continue
        normalized.append(_QUOTE_EQUIVALENTS.get(character, character))
        offsets.append((index, index + 1))
        index += 1
    return "".join(normalized), offsets
