from __future__ import annotations

from uuid import UUID, uuid4

from src.instruments.domain.entities import MatchCandidate
from src.instruments.domain.enums import AliasType, MatchType
from src.instruments.domain.matcher import InstrumentMatcher


def candidate(
    *,
    instrument_id: UUID,
    ticker: str,
    alias: str,
    normalized_alias: str,
    alias_type: AliasType,
    issuer_name: str = "Issuer",
) -> MatchCandidate:
    return MatchCandidate(
        instrument_id=instrument_id,
        ticker=ticker,
        issuer_name=issuer_name,
        matched_alias=alias,
        normalized_alias=normalized_alias,
        alias_type=alias_type,
        priority=100,
    )


def test_sber_is_matched_as_ticker() -> None:
    instrument_id = uuid4()
    matches = InstrumentMatcher().match(
        "SBER вырос после отчета",
        [
            candidate(
                instrument_id=instrument_id,
                ticker="SBER",
                alias="SBER",
                normalized_alias="sber",
                alias_type=AliasType.TICKER,
            )
        ],
    )

    assert len(matches) == 1
    assert matches[0].ticker == "SBER"
    assert matches[0].match_type == MatchType.EXACT_TICKER
    assert matches[0].confidence == 1.0


def test_sberp_does_not_match_sber() -> None:
    sber_id = uuid4()
    sberp_id = uuid4()
    matches = InstrumentMatcher().match(
        "SBERP выросла",
        [
            candidate(
                instrument_id=sber_id,
                ticker="SBER",
                alias="SBER",
                normalized_alias="sber",
                alias_type=AliasType.TICKER,
            ),
            candidate(
                instrument_id=sberp_id,
                ticker="SBERP",
                alias="SBERP",
                normalized_alias="sberp",
                alias_type=AliasType.TICKER,
            ),
        ],
    )

    assert [match.ticker for match in matches] == ["SBERP"]


def test_ticker_inside_another_word_does_not_match() -> None:
    instrument_id = uuid4()
    matches = InstrumentMatcher().match(
        "NEWSBER не является тикером",
        [
            candidate(
                instrument_id=instrument_id,
                ticker="SBER",
                alias="SBER",
                normalized_alias="sber",
                alias_type=AliasType.TICKER,
            )
        ],
    )

    assert matches == []


def test_shared_alias_is_returned_as_ambiguous_for_both_instruments() -> None:
    sber_id = uuid4()
    sberp_id = uuid4()
    matches = InstrumentMatcher().match(
        "Сбербанк сообщил результаты",
        [
            candidate(
                instrument_id=sber_id,
                ticker="SBER",
                alias="Сбербанк",
                normalized_alias="сбербанк",
                alias_type=AliasType.OFFICIAL_NAME,
            ),
            candidate(
                instrument_id=sberp_id,
                ticker="SBERP",
                alias="Сбербанк",
                normalized_alias="сбербанк",
                alias_type=AliasType.OFFICIAL_NAME,
            ),
        ],
    )

    assert {match.ticker for match in matches} == {"SBER", "SBERP"}
    assert all(match.is_ambiguous for match in matches)


def test_repeated_mentions_are_merged_per_instrument() -> None:
    instrument_id = uuid4()
    matches = InstrumentMatcher().match(
        "Газпром сообщил, Газпром подтвердил",
        [
            candidate(
                instrument_id=instrument_id,
                ticker="GAZP",
                alias="Газпром",
                normalized_alias="газпром",
                alias_type=AliasType.OFFICIAL_NAME,
            )
        ],
    )

    assert len(matches) == 1
    assert matches[0].start_position == 0
    assert matches[0].end_position == len("Газпром")


def test_lukoil_variants_match_same_instrument() -> None:
    instrument_id = uuid4()
    first = InstrumentMatcher().match(
        "ЛУКОЙЛ опубликовал отчет",
        [
            candidate(
                instrument_id=instrument_id,
                ticker="LKOH",
                alias="ЛУКОЙЛ",
                normalized_alias="лукойл",
                alias_type=AliasType.OFFICIAL_NAME,
            )
        ],
    )
    second = InstrumentMatcher().match(
        "Лукойл опубликовал отчет",
        [
            candidate(
                instrument_id=instrument_id,
                ticker="LKOH",
                alias="Лукойл",
                normalized_alias="лукойл",
                alias_type=AliasType.BRAND,
            )
        ],
    )

    assert first[0].ticker == second[0].ticker == "LKOH"
