from __future__ import annotations

import time
from uuid import uuid4

from src.instruments.domain.entities import MatchCandidate
from src.instruments.domain.enums import AliasType
from src.instruments.domain.matcher import InstrumentMatcher


def test_matcher_handles_1000_aliases_and_10000_characters_without_large_regression() -> None:
    target_id = uuid4()
    candidates = [
        MatchCandidate(
            instrument_id=uuid4(),
            ticker=f"TST{index}",
            issuer_name=f"Issuer {index}",
            matched_alias=f"Компания {index}",
            normalized_alias=f"компания {index}",
            alias_type=AliasType.OFFICIAL_NAME,
            priority=100,
        )
        for index in range(999)
    ]
    candidates.append(
        MatchCandidate(
            instrument_id=target_id,
            ticker="GAZP",
            issuer_name="ПАО Газпром",
            matched_alias="Газпром",
            normalized_alias="газпром",
            alias_type=AliasType.OFFICIAL_NAME,
            priority=100,
        )
    )
    text = ("нейтральный текст " * 600) + " Газпром"

    started_at = time.perf_counter()
    matches = InstrumentMatcher().match(text, candidates)
    duration = time.perf_counter() - started_at

    assert [match.ticker for match in matches] == ["GAZP"]
    assert duration < 2.0
