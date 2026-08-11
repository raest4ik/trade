from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from uuid import UUID, uuid4

from src.events.domain.analyzer import EventAnalyzer
from src.events.domain.entities import DetectedEvent, ExtractedFinancialFact, NewsEventAnalysis
from src.events.domain.enums import (
    ChangeDirection,
    ComparisonType,
    Currency,
    EventAnalysisStatus,
    EventType,
    FactRole,
    FactUnit,
    FinancialMetric,
    PeriodType,
    ValueScale,
)
from src.events.domain.rules import EventRule, compile_rule

EVENT_ANALYSIS_V3_VERSION = "event-rules-v3"
FINANCIAL_FACTS_V3_VERSION = "financial-facts-v3"

V3_EVENT_RULES: tuple[EventRule, ...] = (
    EventRule(
        "event.v3.financial_results.ifrs_announcement.en",
        EventType.FINANCIAL_RESULTS,
        compile_rule(
            r"\b(?:publishes|announces)\s+(?:its\s+)?results\b.{0,180}"
            r"\b(?:ifrs|international\s+financial\s+reporting\s+standards)\b"
        ),
        10,
        "0.96",
    ),
    EventRule(
        "event.v3.sanctions.restrictive_measures.en",
        EventType.SANCTIONS,
        compile_rule(
            r"\b(?:impos(?:e|es|ed|ing)\s+restrictive\s+measures|"
            r"restrictive\s+measures\s+(?:on|against))\b"
        ),
        20,
        "0.95",
    ),
    EventRule(
        "event.v3.management.elected_leadership.en",
        EventType.MANAGEMENT_CHANGE,
        compile_rule(
            r"\b(?:has\s+been\s+)?elected\s+(?:as\s+)?"
            r"(?:chair(?:man|woman|person)|chief\s+executive|ceo|president)\b"
        ),
        30,
        "0.94",
    ),
    EventRule(
        "event.v3.other.cooperation_agreement.en",
        EventType.OTHER,
        compile_rule(
            r"\b(?:signed|concluded)\s+(?:a\s+)?(?:trilateral\s+)?"
            r"(?:cooperation\s+agreement|agreement\s+of\s+cooperation)\b"
        ),
        40,
        "0.90",
    ),
)

_DIVIDEND_PER_SHARE_PATTERN = compile_rule(
    r"\bdividends?\b.{0,100}?"
    r"(?P<amount>\d+(?:[.,]\d+)?)\s+(?:roubles?|rub)\s+per\s+share\b"
)
_YEAR_PATTERN = re.compile(r"\b(?:for\s+)?(?P<year>20\d{2})\b", re.IGNORECASE)


class EventAnalyzerV3:
    """A frozen v2 analysis plus narrowly scoped, general DEVELOPMENT-derived rules."""

    def __init__(self) -> None:
        self._v2 = EventAnalyzer()

    def analyze(self, *, news_id: UUID, raw_content: str) -> NewsEventAnalysis:
        baseline = self._v2.analyze(news_id=news_id, raw_content=raw_content)
        events = list(baseline.events)
        detected_types = {event.event_type for event in events}
        additions: list[tuple[int, DetectedEvent]] = []
        for rule in V3_EVENT_RULES:
            if rule.event_type in detected_types:
                continue
            match = rule.pattern.search(raw_content)
            if match is None:
                continue
            additions.append(
                (
                    rule.priority,
                    DetectedEvent(
                        id=uuid4(),
                        analysis_id=UUID(int=0),
                        event_type=rule.event_type,
                        confidence=Decimal(rule.confidence),
                        rule_id=rule.rule_id,
                        matched_rule=rule.rule_id,
                        evidence_text=raw_content[match.start() : match.end()],
                        start_position=match.start(),
                        end_position=match.end(),
                    ),
                )
            )
            detected_types.add(rule.event_type)
        events.extend(event for _, event in additions)

        facts = list(baseline.financial_facts)
        if not any(fact.metric == FinancialMetric.DIVIDEND_PER_SHARE for fact in facts):
            dividend_fact = _extract_dividend_per_share(raw_content)
            if dividend_fact is not None:
                facts.append(dividend_fact)

        primary = baseline.primary_event_type
        if primary == EventType.UNKNOWN and additions:
            primary = min(additions, key=lambda item: item[0])[1].event_type
        return NewsEventAnalysis.create(
            news_id=news_id,
            status=_status(events, facts),
            primary_event_type=primary,
            events=events,
            financial_facts=facts,
            analysis_version=EVENT_ANALYSIS_V3_VERSION,
        )


def rules_v3_fingerprint() -> str:
    material = {
        "analysis_version": EVENT_ANALYSIS_V3_VERSION,
        "fact_version": FINANCIAL_FACTS_V3_VERSION,
        "event_rules": [
            {
                "confidence": rule.confidence,
                "event_type": rule.event_type.value,
                "pattern": rule.pattern.pattern,
                "priority": rule.priority,
                "rule_id": rule.rule_id,
            }
            for rule in V3_EVENT_RULES
        ],
        "fact_rules": [
            {
                "pattern": _DIVIDEND_PER_SHARE_PATTERN.pattern,
                "rule_id": "fact.v3.dividend_per_share.en",
            }
        ],
    }
    canonical = json.dumps(material, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _extract_dividend_per_share(raw_content: str) -> ExtractedFinancialFact | None:
    match = _DIVIDEND_PER_SHARE_PATTERN.search(raw_content)
    if match is None:
        return None
    amount_text = match.group("amount")
    amount_start = match.start("amount")
    evidence_end_match = re.search(
        r"\d+(?:[.,]\d+)?\s+(?:roubles?|rub)\s+per\s+share\b",
        raw_content[amount_start:],
        re.IGNORECASE,
    )
    if evidence_end_match is None:
        return None
    evidence_end = amount_start + evidence_end_match.end()
    year_match = _YEAR_PATTERN.search(raw_content)
    year = None if year_match is None else int(year_match.group("year"))
    value = Decimal(amount_text.replace(",", "."))
    return ExtractedFinancialFact(
        id=uuid4(),
        analysis_id=UUID(int=0),
        metric=FinancialMetric.DIVIDEND_PER_SHARE,
        raw_value=value,
        normalized_value=value,
        unit=FactUnit.MONEY,
        currency=Currency.RUB,
        scale=ValueScale.ONE,
        period_type=PeriodType.YEAR if year is not None else PeriodType.UNKNOWN,
        year=year,
        quarter=None,
        month=None,
        date_from=None,
        date_to=None,
        raw_period=None if year is None else str(year),
        comparison_type=ComparisonType.NONE,
        fact_role=FactRole.ACTUAL,
        change_direction=ChangeDirection.UNKNOWN,
        change_value=None,
        change_unit=None,
        confidence=Decimal("0.97"),
        rule_id="fact.v3.dividend_per_share.en",
        evidence_text=raw_content[amount_start:evidence_end],
        start_position=amount_start,
        end_position=evidence_end,
        extractor_version=FINANCIAL_FACTS_V3_VERSION,
        matched_rule="fact.v3.dividend_per_share.en",
    )


def _status(
    events: list[DetectedEvent], facts: list[ExtractedFinancialFact]
) -> EventAnalysisStatus:
    if not events and not facts:
        return EventAnalysisStatus.NO_EVENT_FOUND
    if len({event.event_type for event in events}) > 1:
        return EventAnalysisStatus.AMBIGUOUS
    if any(fact.metric == FinancialMetric.OTHER for fact in facts):
        return EventAnalysisStatus.PARTIAL
    return EventAnalysisStatus.COMPLETE
