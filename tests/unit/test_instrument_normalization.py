from __future__ import annotations

from src.instruments.domain.normalization import normalize_text, normalize_text_with_mapping


def test_yo_and_ie_are_normalized_equally() -> None:
    assert normalize_text("эмитёр") == normalize_text("эмитер")


def test_spaces_and_newlines_are_collapsed() -> None:
    assert normalize_text("ПАО\n\n  Газпром") == "пао газпром"


def test_quotes_are_normalized_and_punctuation_is_separated() -> None:
    assert normalize_text("ПАО «Газпром»") == 'пао " газпром "'


def test_original_positions_are_preserved() -> None:
    raw_text = 'ПАО "Газпром" сообщило'
    normalized = normalize_text_with_mapping(raw_text)
    start = normalized.value.index("газпром")
    end = start + len("газпром")

    assert raw_text[slice(*normalized.original_span(start, end))] == "Газпром"
