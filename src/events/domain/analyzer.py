from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, InvalidOperation
from uuid import UUID, uuid4

from src.events.domain.entities import (
    EVENT_ANALYSIS_VERSION,
    FINANCIAL_FACTS_VERSION,
    DetectedEvent,
    ExtractedFinancialFact,
    NewsEventAnalysis,
)
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
from src.events.domain.rules import EVENT_RULES, METRIC_RULES, MetricRule

_NUMBER_PATTERN = re.compile(
    r"(?P<prefix>[$€¥₽])?\s*(?P<number>-?\d[\d\s]*(?:[,.]\d+)?)\s*"
    r"(?P<suffix>%|процентн(?:ых|ого|ые)?\s+пункт\w*|п\.п\.|pp|млн|млрд|трлн|тыс\.?|thousand|million|billion|trillion)?\s*"
    r"(?P<currency>руб\.?|рублей|rub|₽|доллар\w*|долл\.?|usd|\$|eur|евро|€|юан\w*|cny|тонн\w*|tons?|tonnes?|баррел\w*|barrels?|акци\w*|shares?|куб\.?\s*м(?:етр\w*)?)?",
    flags=re.IGNORECASE | re.UNICODE,
)

_SENTENCE_PATTERN = re.compile(r".+?(?:[.!?](?=\s|$)|\n|$)", flags=re.UNICODE)
_INLINE_ABBREVIATION_PATTERN = re.compile(
    r"(?i)\b(?:руб|долл|тыс|куб|кв)\.(?=\s*[,а-яё])|\bп\.п\.(?=\s*[,а-яё])"
)
_YEAR_ONLY_PATTERN = re.compile(r"^\d{4}$")


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Period:
    period_type: PeriodType
    year: int | None
    quarter: int | None
    month: int | None
    date_from: date | None
    date_to: date | None
    raw_period: str | None


class EventAnalyzer:
    def __init__(self, analysis_version: str = EVENT_ANALYSIS_VERSION) -> None:
        self._analysis_version = analysis_version

    def analyze(self, *, news_id: UUID, raw_content: str) -> NewsEventAnalysis:
        events = self._detect_events(raw_content)
        facts = self._extract_facts(raw_content)
        status = _analysis_status(events, facts)
        primary = _primary_event_type(events)
        return NewsEventAnalysis.create(
            news_id=news_id,
            status=status,
            primary_event_type=primary,
            events=events,
            financial_facts=facts,
            analysis_version=self._analysis_version,
        )

    def _detect_events(self, raw_content: str) -> list[DetectedEvent]:
        detected: dict[EventType, DetectedEvent] = {}
        for rule in EVENT_RULES:
            match = rule.pattern.search(raw_content)
            if match is None:
                continue
            evidence = raw_content[match.start() : match.end()]
            current = detected.get(rule.event_type)
            candidate = DetectedEvent(
                id=uuid4(),
                analysis_id=UUID(int=0),
                event_type=rule.event_type,
                confidence=Decimal(rule.confidence),
                rule_id=rule.rule_id,
                matched_rule=rule.rule_id,
                evidence_text=evidence,
                start_position=match.start(),
                end_position=match.end(),
            )
            if current is None or (rule.priority, match.start()) < (
                _rule_priority(current.matched_rule),
                current.start_position,
            ):
                detected[rule.event_type] = candidate
        return sorted(detected.values(), key=lambda item: (-item.confidence, item.start_position))

    def _extract_facts(self, raw_content: str) -> list[ExtractedFinancialFact]:
        facts: list[ExtractedFinancialFact] = []
        active_period: Period | None = None
        for sentence in _sentences(raw_content):
            period = _extract_period(sentence.text)
            if period.period_type == PeriodType.UNKNOWN and active_period is not None:
                period = active_period
            elif period.period_type != PeriodType.UNKNOWN:
                active_period = period
            metric_matches = _metric_matches(sentence.text)
            sentence_facts: list[ExtractedFinancialFact] = []
            for match in _NUMBER_PATTERN.finditer(sentence.text):
                raw_number = match.group("number")
                if _looks_like_date_or_year(sentence.text, match.start(), match.end(), raw_number):
                    continue
                parsed = _parse_number(raw_number)
                if parsed is None:
                    continue
                value_start = sentence.start + match.start()
                value_end = sentence.start + match.end()
                metric_rule = _nearest_metric(metric_matches, match.start())
                unit = _unit(match)
                currency = _currency(match)
                scale = _scale(match)
                metric, rule_id = _metric_for_value(
                    sentence.text,
                    match.start(),
                    match.end(),
                    metric_rule,
                    unit,
                )
                if _should_ignore_untyped_number(
                    sentence.text,
                    match.start(),
                    match.end(),
                    metric,
                    unit,
                    scale,
                ):
                    continue
                normalized = _normalize_value(parsed, scale)
                comparison = _comparison_type(sentence.text)
                direction = _change_direction(sentence.text, match.start())
                role = _fact_role(sentence.text, match.start(), metric, direction)
                target_index = _attached_change_target(
                    sentence_facts,
                    metric=metric,
                    role=role,
                    direction=direction,
                    unit=unit,
                )
                if target_index is not None:
                    sentence_facts[target_index] = replace(
                        sentence_facts[target_index],
                        comparison_type=comparison,
                        change_direction=direction,
                        change_value=parsed,
                        change_unit=unit,
                    )
                    continue
                confidence = _fact_confidence(metric != FinancialMetric.OTHER, period, role)
                sentence_facts.append(
                    ExtractedFinancialFact(
                        id=uuid4(),
                        analysis_id=UUID(int=0),
                        metric=metric,
                        raw_value=parsed,
                        normalized_value=normalized,
                        unit=unit,
                        currency=currency,
                        scale=scale,
                        period_type=period.period_type,
                        year=period.year,
                        quarter=period.quarter,
                        month=period.month,
                        date_from=period.date_from,
                        date_to=period.date_to,
                        raw_period=period.raw_period,
                        comparison_type=comparison,
                        fact_role=role,
                        change_direction=direction,
                        change_value=parsed if role == FactRole.CHANGE else None,
                        change_unit=unit if role == FactRole.CHANGE else None,
                        confidence=confidence,
                        rule_id=rule_id,
                        evidence_text=raw_content[value_start:value_end],
                        start_position=value_start,
                        end_position=value_end,
                        extractor_version=FINANCIAL_FACTS_VERSION,
                        matched_rule=rule_id,
                    )
                )
            facts.extend(sentence_facts)
        return facts


def _sentences(raw_content: str) -> list[Sentence]:
    shadow = _INLINE_ABBREVIATION_PATTERN.sub(
        lambda match: match.group(0).replace(".", "\0"), raw_content
    )
    return [
        Sentence(
            text=raw_content[match.start() : match.end()],
            start=match.start(),
            end=match.end(),
        )
        for match in _SENTENCE_PATTERN.finditer(shadow)
        if match.group(0).strip()
    ]


def _metric_matches(sentence: str) -> list[tuple[MetricRule, int, int]]:
    matches: list[tuple[MetricRule, int, int]] = []
    for rule in METRIC_RULES:
        for match in rule.pattern.finditer(sentence):
            matches.append((rule, match.start(), match.end()))
    return sorted(matches, key=lambda item: (item[1], item[0].priority))


def _nearest_metric(
    metric_matches: list[tuple[MetricRule, int, int]],
    value_start: int,
) -> MetricRule | None:
    nearby = [
        (abs(value_start - end), rule)
        for rule, start, end in metric_matches
        if abs(value_start - start) <= 80 or abs(value_start - end) <= 80
    ]
    if not nearby:
        return None
    return sorted(nearby, key=lambda item: (item[0], item[1].priority))[0][1]


def _metric_for_value(
    sentence: str,
    value_start: int,
    value_end: int,
    metric_rule: MetricRule | None,
    unit: FactUnit,
) -> tuple[FinancialMetric, str]:
    text = sentence.lower()
    after = text[value_end : min(len(text), value_end + 36)]
    around = text[max(0, value_start - 80) : min(len(text), value_end + 36)]
    if "дивиденд" in text and re.search(r"(?:на\s+акци\w*|per\s+share)", after):
        return FinancialMetric.DIVIDEND_PER_SHARE, "metric.dividend_per_share.context"
    if "дивиденд" in text and re.search(r"(?:общ\w+\s+сумм\w+|всего\s+выплат)", text):
        return FinancialMetric.DIVIDEND_TOTAL, "metric.dividend_total.context"
    if unit in {FactUnit.TONNES, FactUnit.BARRELS, FactUnit.CUBIC_METERS} and re.search(
        r"(?:производств\w*|добыч\w*|добы[лт]\w*|production|output)", text
    ):
        return FinancialMetric.PRODUCTION_VOLUME, "metric.production.unit_context"
    if unit == FactUnit.PERCENT and re.search(r"(?:акци\w*|дол[яи]|ownership|stake)", around):
        return FinancialMetric.OWNERSHIP_PERCENT, "metric.ownership.share_context"
    if metric_rule is not None:
        return metric_rule.metric, f"{metric_rule.rule_id}.value_proximity"
    return FinancialMetric.OTHER, "metric.unknown.value"


def _attached_change_target(
    facts: list[ExtractedFinancialFact],
    *,
    metric: FinancialMetric,
    role: FactRole,
    direction: ChangeDirection,
    unit: FactUnit,
) -> int | None:
    if (
        role != FactRole.CHANGE
        or direction not in {ChangeDirection.UP, ChangeDirection.DOWN}
        or unit not in {FactUnit.PERCENT, FactUnit.PERCENTAGE_POINTS}
    ):
        return None
    candidates = [
        index
        for index, fact in enumerate(facts)
        if fact.fact_role != FactRole.CHANGE
        and fact.change_value is None
        and (metric == FinancialMetric.OTHER or fact.metric == metric)
    ]
    return candidates[-1] if candidates else None


def _parse_number(raw: str) -> Decimal | None:
    normalized = raw.replace(" ", "").replace(",", ".")
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def _looks_like_date_or_year(sentence: str, start: int, end: int, raw_number: str) -> bool:
    stripped = raw_number.replace(" ", "")
    before = sentence[max(0, start - 18) : start]
    after = sentence[end : min(len(sentence), end + 24)]
    around = before + stripped + after
    if _YEAR_ONLY_PATTERN.fullmatch(stripped) and 1900 <= int(stripped) <= 2099:
        return True
    if re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}", around, re.I):
        return True
    if stripped in {"1", "2", "3", "4"} and (
        re.search(r"(?:квартал|quarter)", after, re.I)
        or re.search(r"(?:q|кв\.?)\s*$", before, re.I)
    ):
        return True
    if re.search(
        r"^(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)",
        after.strip(),
        re.I,
    ):
        return True
    return bool(re.search(r"\d[./-]$", before) or re.search(r"^[./-]\d", after))


def _should_ignore_untyped_number(
    sentence: str,
    start: int,
    end: int,
    metric: FinancialMetric,
    unit: FactUnit,
    scale: ValueScale,
) -> bool:
    if metric != FinancialMetric.OTHER:
        return False
    if unit in {FactUnit.TONNES, FactUnit.BARRELS, FactUnit.CUBIC_METERS, FactUnit.SHARES}:
        return True
    if unit != FactUnit.UNSPECIFIED or scale != ValueScale.ONE:
        return False
    stripped_digits = re.sub(r"\D", "", sentence[start:end])
    if len(stripped_digits) >= 8:
        return True
    return True


def _unit(match: re.Match[str]) -> FactUnit:
    suffix = (match.group("suffix") or "").lower()
    tail = (match.group("currency") or "").lower()
    if "%" in suffix:
        return FactUnit.PERCENT
    if "п.п" in suffix or "пункт" in suffix or suffix == "pp":
        return FactUnit.PERCENTAGE_POINTS
    if "тонн" in tail or tail in {"ton", "tons", "tonne", "tonnes"}:
        return FactUnit.TONNES
    if "баррел" in tail or tail in {"barrel", "barrels"}:
        return FactUnit.BARRELS
    if "куб" in tail:
        return FactUnit.CUBIC_METERS
    if "акци" in tail or tail in {"share", "shares"}:
        return FactUnit.SHARES
    currency = match.group("prefix") or tail
    if _is_currency_text(currency):
        return FactUnit.MONEY
    return FactUnit.UNSPECIFIED


def _currency(match: re.Match[str]) -> Currency:
    text = f"{match.group('prefix') or ''} {match.group('currency') or ''}".lower()
    if "₽" in text or "руб" in text or "rub" in text:
        return Currency.RUB
    if "$" in text or "usd" in text or "дол" in text:
        return Currency.USD
    if "€" in text or "eur" in text or "евро" in text:
        return Currency.EUR
    if "cny" in text or "юан" in text:
        return Currency.CNY
    return Currency.UNSPECIFIED


def _is_currency_text(text: str) -> bool:
    return bool(re.search(r"(₽|руб|rub|\$|usd|дол|€|eur|евро|юан|cny)", text, re.I))


def _scale(match: re.Match[str]) -> ValueScale:
    suffix = (match.group("suffix") or "").lower()
    if "трлн" in suffix or "trillion" in suffix:
        return ValueScale.TRILLION
    if "млрд" in suffix or "billion" in suffix:
        return ValueScale.BILLION
    if "млн" in suffix or "million" in suffix:
        return ValueScale.MILLION
    if "тыс" in suffix or "thousand" in suffix:
        return ValueScale.THOUSAND
    return ValueScale.ONE


def _normalize_value(value: Decimal, scale: ValueScale) -> Decimal:
    multipliers = {
        ValueScale.ONE: Decimal("1"),
        ValueScale.THOUSAND: Decimal("1000"),
        ValueScale.MILLION: Decimal("1000000"),
        ValueScale.BILLION: Decimal("1000000000"),
        ValueScale.TRILLION: Decimal("1000000000000"),
    }
    return value * multipliers[scale]


def _fact_role(
    sentence: str,
    value_start: int,
    metric: FinancialMetric,
    direction: ChangeDirection,
) -> FactRole:
    prefix = sentence[max(0, value_start - 140) : value_start].lower()
    if metric == FinancialMetric.OWNERSHIP_PERCENT and re.search(
        r"(?:план\w*|намер\w*|цел\w*|консолидир\w*).*(?:до|не\s+менее)\s*$", prefix
    ):
        return FactRole.TARGET
    if re.search(
        r"(прогноз|forecast|guidance|ожидает|outlook|ориентир|цел\w*|планиру\w*|намерен\w*|рекомендовал\w*)",
        prefix,
    ):
        return FactRole.FORECAST
    if re.search(r"(консенсус|consensus)", prefix):
        return FactRole.CONSENSUS
    if re.search(r"(годом ранее|ранее|previous)", prefix):
        return FactRole.PREVIOUS
    if direction in {ChangeDirection.UP, ChangeDirection.DOWN} and re.search(
        r"(?:на|by)\s*$", prefix
    ):
        return FactRole.CHANGE
    if re.search(r"(вырос\w*|увеличил\w*|снизил\w*|сократил\w*)\s+до", prefix):
        return FactRole.ACTUAL
    return FactRole.ACTUAL


def _comparison_type(sentence: str) -> ComparisonType:
    text = sentence.lower()
    if re.search(
        r"(г/г|год\s+к\s+году|относительно\s+20\d{2}\s+год\w*|year[-\s]?over[-\s]?year)",
        text,
    ):
        return ComparisonType.YEAR_OVER_YEAR
    if re.search(r"(кв/кв|квартал\s+к\s+кварталу|quarter[-\s]?over[-\s]?quarter)", text):
        return ComparisonType.QUARTER_OVER_QUARTER
    if re.search(r"(м/м|месяц\s+к\s+месяцу|month[-\s]?over[-\s]?month)", text):
        return ComparisonType.MONTH_OVER_MONTH
    if re.search(r"(выше\s+прогноз|ниже\s+прогноз|versus\s+forecast)", text):
        return ComparisonType.VERSUS_FORECAST
    if re.search(r"(выше\s+консенсус|ниже\s+консенсус|versus\s+consensus)", text):
        return ComparisonType.VERSUS_FORECAST
    if re.search(r"(годом ранее|previous)", text):
        return ComparisonType.VERSUS_PREVIOUS
    return ComparisonType.NONE


def _change_direction(sentence: str, value_start: int) -> ChangeDirection:
    prefix = sentence[max(0, value_start - 80) : value_start].lower()
    if re.search(r"(вырос\w*|увелич\w*|повысил\w*|поднял\w*|grew|increased|rose)", prefix):
        return ChangeDirection.UP
    if re.search(r"(сниз\w*|сократил\w*|понизил\w*|упал\w*|decreased|fell|declined)", prefix):
        return ChangeDirection.DOWN
    if re.search(
        r"(без\s+изменени\w*|не\s+изменил\w*|остал\w*\s+на\s+уровне|сохранил\w*\s+на\s+уровне|unchanged|flat)",
        prefix,
    ):
        return ChangeDirection.UNCHANGED
    return ChangeDirection.UNKNOWN


def _extract_period(sentence: str) -> Period:
    patterns: tuple[tuple[PeriodType, re.Pattern[str]], ...] = (
        (
            PeriodType.QUARTER,
            re.compile(
                r"(?P<raw>(?:за\s+)?(?P<q>[1-4]|I{1,3}|IV)\s+квартал\w*\s+(?P<year>20\d{2}))", re.I
            ),
        ),
        (PeriodType.QUARTER, re.compile(r"(?P<raw>Q(?P<q>[1-4])\s*(?P<year>20\d{2}))", re.I)),
        (
            PeriodType.HALF_YEAR,
            re.compile(
                r"(?P<raw>(?:(?:за\s+)?перв\w+\s+полугоди\w+|1H)"
                r"\s*(?P<year>20\d{2}))",
                re.I,
            ),
        ),
        (
            PeriodType.NINE_MONTHS,
            re.compile(r"(?P<raw>(?:9|девят\w+)\s+месяц\w+\s+(?P<year>20\d{2}))", re.I),
        ),
        (
            PeriodType.YEAR,
            re.compile(
                r"(?P<raw>(?:(?:за|в|на|FY)\s*|по\s+итогам\s+)(?P<year>20\d{2})\s*(?:год\w*)?)",
                re.I,
            ),
        ),
        (
            PeriodType.MONTH,
            re.compile(
                r"(?P<raw>по\s+итогам\s+(?P<month>июля|января|февраля|марта|апреля|мая|июня|августа|сентября|октября|ноября|декабря)\s+(?P<year>20\d{2}))",
                re.I,
            ),
        ),
    )
    for period_type, pattern in patterns:
        match = pattern.search(sentence)
        if match is None:
            continue
        year = int(match.group("year"))
        quarter = _quarter(match.groupdict().get("q"))
        month = _month(match.groupdict().get("month"))
        return Period(
            period_type=period_type,
            year=year,
            quarter=quarter,
            month=month,
            date_from=None,
            date_to=None,
            raw_period=match.group("raw"),
        )
    return Period(
        period_type=PeriodType.UNKNOWN,
        year=None,
        quarter=None,
        month=None,
        date_from=None,
        date_to=None,
        raw_period=None,
    )


def _quarter(raw: str | None) -> int | None:
    if raw is None:
        return None
    mapping = {"I": 1, "II": 2, "III": 3, "IV": 4}
    return mapping.get(raw.upper(), int(raw) if raw.isdigit() else None)


def _month(raw: str | None) -> int | None:
    months = {
        "января": 1,
        "февраля": 2,
        "марта": 3,
        "апреля": 4,
        "мая": 5,
        "июня": 6,
        "июля": 7,
        "августа": 8,
        "сентября": 9,
        "октября": 10,
        "ноября": 11,
        "декабря": 12,
    }
    return None if raw is None else months.get(raw.lower())


def _fact_confidence(
    metric_known: bool,
    period: Period,
    role: FactRole,
) -> Decimal:
    confidence = Decimal("0.91") if metric_known else Decimal("0.72")
    if period.period_type == PeriodType.UNKNOWN:
        confidence -= Decimal("0.08")
    if role in {FactRole.FORECAST, FactRole.CHANGE, FactRole.TARGET}:
        confidence += Decimal("0.02")
    return confidence


def _analysis_status(
    events: list[DetectedEvent],
    facts: list[ExtractedFinancialFact],
) -> EventAnalysisStatus:
    if not events and not facts:
        return EventAnalysisStatus.NO_EVENT_FOUND
    if len({event.event_type for event in events}) > 1:
        return EventAnalysisStatus.AMBIGUOUS
    if any(fact.metric == FinancialMetric.OTHER for fact in facts):
        return EventAnalysisStatus.PARTIAL
    return EventAnalysisStatus.COMPLETE


def _primary_event_type(events: list[DetectedEvent]) -> EventType:
    if not events:
        return EventType.UNKNOWN
    return sorted(
        events,
        key=lambda item: (
            -item.confidence,
            _rule_priority(item.rule_id),
            -(item.end_position - item.start_position),
            item.start_position,
        ),
    )[0].event_type


def _rule_priority(rule_id: str) -> int:
    for rule in EVENT_RULES:
        if rule.rule_id == rule_id:
            return rule.priority
    return 9999
