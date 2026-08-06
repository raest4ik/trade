from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

WORD_CHARS_PATTERN = r"0-9a-zа-я_"
TOKEN_LEFT_BOUNDARY = rf"(?<![{WORD_CHARS_PATTERN}])"
TOKEN_RIGHT_BOUNDARY = rf"(?![{WORD_CHARS_PATTERN}])"
QUOTE_TRANSLATION = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
    }
)


@dataclass(frozen=True, slots=True)
class NormalizedText:
    value: str
    original_positions: tuple[int, ...]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        if start < 0 or end <= start or end > len(self.original_positions):
            raise ValueError("invalid normalized span")
        return self.original_positions[start], self.original_positions[end - 1] + 1


def normalize_text(value: str) -> str:
    return normalize_text_with_mapping(value).value


def normalize_text_with_mapping(value: str) -> NormalizedText:
    chars: list[str] = []
    positions: list[int] = []

    for original_index, raw_char in enumerate(value):
        normalized_chunk = unicodedata.normalize("NFKC", raw_char).translate(QUOTE_TRANSLATION)
        for normalized_char in normalized_chunk:
            char = normalized_char.lower().replace("ё", "е")
            if char.isspace():
                chars.append(" ")
                positions.append(original_index)
            elif _is_punctuation(char):
                chars.extend((" ", char, " "))
                positions.extend((original_index, original_index, original_index))
            else:
                chars.append(char)
                positions.append(original_index)

    collapsed_chars: list[str] = []
    collapsed_positions: list[int] = []
    previous_was_space = True
    for char, position in zip(chars, positions, strict=True):
        if char.isspace():
            if not previous_was_space:
                collapsed_chars.append(" ")
                collapsed_positions.append(position)
            previous_was_space = True
            continue
        collapsed_chars.append(char)
        collapsed_positions.append(position)
        previous_was_space = False

    if collapsed_chars and collapsed_chars[-1] == " ":
        collapsed_chars.pop()
        collapsed_positions.pop()

    return NormalizedText(
        value="".join(collapsed_chars), original_positions=tuple(collapsed_positions)
    )


def compile_token_pattern(normalized_needle: str) -> re.Pattern[str]:
    return re.compile(
        rf"{TOKEN_LEFT_BOUNDARY}{re.escape(normalized_needle)}{TOKEN_RIGHT_BOUNDARY}",
        flags=re.IGNORECASE,
    )


def _is_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P")
