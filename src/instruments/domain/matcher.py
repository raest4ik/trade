from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from src.instruments.domain.entities import InstrumentMatch, MatchCandidate
from src.instruments.domain.enums import AliasType, MatchType
from src.instruments.domain.normalization import compile_token_pattern, normalize_text_with_mapping

MATCH_CONFIDENCE: dict[AliasType, float] = {
    AliasType.OFFICIAL_NAME: 0.98,
    AliasType.LEGAL_NAME: 0.97,
    AliasType.SHORT_NAME: 0.95,
    AliasType.BRAND: 0.92,
    AliasType.MANUAL: 0.90,
    AliasType.TICKER: 1.00,
}


@dataclass(frozen=True, slots=True)
class _FoundMatch:
    match: InstrumentMatch
    normalized_alias: str
    priority: int


class InstrumentMatcher:
    def match(self, raw_text: str, candidates: list[MatchCandidate]) -> list[InstrumentMatch]:
        normalized_text = normalize_text_with_mapping(raw_text)
        if not normalized_text.value or not candidates:
            return []

        found: list[_FoundMatch] = []
        alias_group_sizes = self._alias_group_sizes(candidates)
        for candidate in candidates:
            pattern = compile_token_pattern(candidate.normalized_alias)
            for regex_match in pattern.finditer(normalized_text.value):
                start, end = normalized_text.original_span(
                    regex_match.start(),
                    regex_match.end(),
                )
                match_type = (
                    MatchType.EXACT_TICKER
                    if candidate.alias_type == AliasType.TICKER
                    else MatchType.EXACT_ALIAS
                )
                found.append(
                    _FoundMatch(
                        match=InstrumentMatch(
                            instrument_id=candidate.instrument_id,
                            ticker=candidate.ticker,
                            issuer_name=candidate.issuer_name,
                            matched_alias=raw_text[start:end],
                            alias_type=candidate.alias_type,
                            match_type=match_type,
                            confidence=MATCH_CONFIDENCE[candidate.alias_type],
                            start_position=start,
                            end_position=end,
                            is_ambiguous=alias_group_sizes[candidate.normalized_alias] > 1,
                        ),
                        normalized_alias=candidate.normalized_alias,
                        priority=candidate.priority,
                    )
                )

        return self._choose_best_per_instrument(found)

    @staticmethod
    def _alias_group_sizes(candidates: list[MatchCandidate]) -> dict[str, int]:
        instrument_ids_by_alias: dict[str, set[UUID]] = {}
        for candidate in candidates:
            instrument_ids_by_alias.setdefault(candidate.normalized_alias, set()).add(
                candidate.instrument_id
            )
        return {
            normalized_alias: len(instrument_ids)
            for normalized_alias, instrument_ids in instrument_ids_by_alias.items()
        }

    @staticmethod
    def _choose_best_per_instrument(found: list[_FoundMatch]) -> list[InstrumentMatch]:
        best_by_instrument: dict[UUID, _FoundMatch] = {}
        for item in found:
            current = best_by_instrument.get(item.match.instrument_id)
            if current is None or _sort_key(item) < _sort_key(current):
                best_by_instrument[item.match.instrument_id] = item

        return [
            item.match
            for item in sorted(
                best_by_instrument.values(),
                key=lambda value: (value.match.start_position, value.match.ticker),
            )
        ]


def _sort_key(item: _FoundMatch) -> tuple[float, int, int, str]:
    return (-item.match.confidence, item.priority, item.match.start_position, item.normalized_alias)
