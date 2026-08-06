from __future__ import annotations

from decimal import Decimal
from time import perf_counter
from uuid import uuid4

import pytest

from src.events.domain.analyzer import EventAnalyzer
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Компания раскрыла финансовые результаты по МСФО.", EventType.FINANCIAL_RESULTS),
        ("Совет директоров рекомендовал дивиденды за 2025 год.", EventType.DIVIDEND),
        ("Менеджмент повысил прогноз EBITDA на 2026 год.", EventType.GUIDANCE),
        ("Эмитент объявил о приобретении доли в активе.", EventType.MERGER_ACQUISITION),
        ("Компания попала под санкции OFAC.", EventType.SANCTIONS),
        ("Совет утвердил обратный выкуп акций.", EventType.BUYBACK),
        ("Назначил нового генерального директора.", EventType.MANAGEMENT_CHANGE),
        ("Подписан крупный контракт на поставку.", EventType.MAJOR_CONTRACT),
        ("Добыча нефти выросла за квартал.", EventType.PRODUCTION_UPDATE),
        ("Агентство подтвердило кредитный рейтинг.", EventType.CREDIT_RATING),
        ("Компания разместит облигации.", EventType.DEBT_FINANCING),
        ("The company published financial results.", EventType.FINANCIAL_RESULTS),
        ("Board approved a dividend per share.", EventType.DIVIDEND),
        ("Management updated guidance for the year.", EventType.GUIDANCE),
        ("The issuer announced an acquisition.", EventType.MERGER_ACQUISITION),
        ("The company signed a major contract.", EventType.MAJOR_CONTRACT),
        ("Компания объявила допэмиссию акций.", EventType.SHARE_ISSUANCE),
        ("Банк России выдал предписание эмитенту.", EventType.REGULATORY_ACTION),
        ("Компания раскрыла судебный спор.", EventType.LITIGATION),
        ("Совет одобрил сплит акций.", EventType.CORPORATE_ACTION),
    ],
)
def test_event_detection_scenarios(text: str, expected: EventType) -> None:
    analysis = EventAnalyzer().analyze(news_id=uuid4(), raw_content=text)

    assert expected in {event.event_type for event in analysis.events}


@pytest.mark.parametrize(
    ("text", "metric", "raw_value", "normalized", "unit", "currency", "scale"),
    [
        (
            "Выручка за 2025 год составила 1,2 млрд рублей.",
            FinancialMetric.REVENUE,
            Decimal("1.2"),
            Decimal("1200000000.0"),
            FactUnit.MONEY,
            Currency.RUB,
            ValueScale.BILLION,
        ),
        (
            "Чистая прибыль за 2025 год составила 55 млн руб.",
            FinancialMetric.NET_PROFIT,
            Decimal("55"),
            Decimal("55000000"),
            FactUnit.MONEY,
            Currency.RUB,
            ValueScale.MILLION,
        ),
        (
            "EBITDA for FY2025 reached $2.5 billion.",
            FinancialMetric.EBITDA,
            Decimal("2.5"),
            Decimal("2500000000.0"),
            FactUnit.MONEY,
            Currency.USD,
            ValueScale.BILLION,
        ),
        (
            "Operating profit for 2025 was 700 million eur.",
            FinancialMetric.OPERATING_PROFIT,
            Decimal("700"),
            Decimal("700000000"),
            FactUnit.MONEY,
            Currency.EUR,
            ValueScale.MILLION,
        ),
        (
            "Свободный денежный поток за 2025 год достиг 10 млрд руб.",
            FinancialMetric.FREE_CASH_FLOW,
            Decimal("10"),
            Decimal("10000000000"),
            FactUnit.MONEY,
            Currency.RUB,
            ValueScale.BILLION,
        ),
        (
            "Капзатраты за 2025 год составили 300 млн руб.",
            FinancialMetric.CAPEX,
            Decimal("300"),
            Decimal("300000000"),
            FactUnit.MONEY,
            Currency.RUB,
            ValueScale.MILLION,
        ),
        (
            "Чистый долг за 2025 год снизился до 4 млрд руб.",
            FinancialMetric.NET_DEBT,
            Decimal("4"),
            Decimal("4000000000"),
            FactUnit.MONEY,
            Currency.RUB,
            ValueScale.BILLION,
        ),
        (
            "Дивиденд на акцию за 2025 год составит 12,5 руб.",
            FinancialMetric.DIVIDEND_PER_SHARE,
            Decimal("12.5"),
            Decimal("12.5"),
            FactUnit.MONEY,
            Currency.RUB,
            ValueScale.ONE,
        ),
        (
            "Добыча за 2025 год составила 8 млн тонн.",
            FinancialMetric.PRODUCTION_VOLUME,
            Decimal("8"),
            Decimal("8000000"),
            FactUnit.TONNES,
            Currency.UNSPECIFIED,
            ValueScale.MILLION,
        ),
        (
            "Contract value for 2025 reached 3 billion usd.",
            FinancialMetric.CONTRACT_VALUE,
            Decimal("3"),
            Decimal("3000000000"),
            FactUnit.MONEY,
            Currency.USD,
            ValueScale.BILLION,
        ),
        (
            "Доля в компании достигла 25%.",
            FinancialMetric.OWNERSHIP_PERCENT,
            Decimal("25"),
            Decimal("25"),
            FactUnit.PERCENT,
            Currency.UNSPECIFIED,
            ValueScale.ONE,
        ),
        (
            "Маржа EBITDA составила 14 п.п.",
            FinancialMetric.EBITDA,
            Decimal("14"),
            Decimal("14"),
            FactUnit.PERCENTAGE_POINTS,
            Currency.UNSPECIFIED,
            ValueScale.ONE,
        ),
    ],
)
def test_financial_fact_value_scenarios(
    text: str,
    metric: FinancialMetric,
    raw_value: Decimal,
    normalized: Decimal,
    unit: FactUnit,
    currency: Currency,
    scale: ValueScale,
) -> None:
    analysis = EventAnalyzer().analyze(news_id=uuid4(), raw_content=text)

    fact = analysis.financial_facts[0]
    assert fact.metric == metric
    assert fact.raw_value == raw_value
    assert fact.normalized_value == normalized
    assert fact.unit == unit
    assert fact.currency == currency
    assert fact.scale == scale


@pytest.mark.parametrize(
    ("text", "period_type", "year", "quarter", "month"),
    [
        ("Выручка за 2025 год составила 1 млрд руб.", PeriodType.YEAR, 2025, None, None),
        ("Выручка за 1 квартал 2025 составила 1 млрд руб.", PeriodType.QUARTER, 2025, 1, None),
        ("Выручка за IV квартал 2025 составила 1 млрд руб.", PeriodType.QUARTER, 2025, 4, None),
        ("Revenue Q2 2025 reached $1 billion.", PeriodType.QUARTER, 2025, 2, None),
        (
            "EBITDA за первое полугодие 2025 составила 10 млрд руб.",
            PeriodType.HALF_YEAR,
            2025,
            None,
            None,
        ),
        ("Чистая прибыль 1H 2025 достигла 5 млрд руб.", PeriodType.HALF_YEAR, 2025, None, None),
        (
            "Выручка за 9 месяцев 2025 составила 9 млрд руб.",
            PeriodType.NINE_MONTHS,
            2025,
            None,
            None,
        ),
        ("Выручка по итогам июля 2025 составила 1 млрд руб.", PeriodType.MONTH, 2025, None, 7),
    ],
)
def test_period_detection_scenarios(
    text: str,
    period_type: PeriodType,
    year: int | None,
    quarter: int | None,
    month: int | None,
) -> None:
    analysis = EventAnalyzer().analyze(news_id=uuid4(), raw_content=text)

    fact = analysis.financial_facts[0]
    assert fact.period_type == period_type
    assert fact.year == year
    assert fact.quarter == quarter
    assert fact.month == month


@pytest.mark.parametrize(
    ("text", "role", "comparison", "direction"),
    [
        (
            "Выручка за 2025 год выросла на 15% г/г.",
            FactRole.CHANGE,
            ComparisonType.YEAR_OVER_YEAR,
            ChangeDirection.UP,
        ),
        (
            "Чистая прибыль за 2025 год снизилась на 8% год к году.",
            FactRole.CHANGE,
            ComparisonType.YEAR_OVER_YEAR,
            ChangeDirection.DOWN,
        ),
        (
            "Прогноз EBITDA на 2026 год составляет 100 млрд руб.",
            FactRole.FORECAST,
            ComparisonType.NONE,
            ChangeDirection.UNCHANGED,
        ),
        (
            "Консенсус по выручке на 2026 год составляет 1 трлн руб.",
            FactRole.CONSENSUS,
            ComparisonType.NONE,
            ChangeDirection.UNCHANGED,
        ),
        (
            "Выручка годом ранее составляла 900 млрд руб.",
            FactRole.PREVIOUS,
            ComparisonType.VERSUS_PREVIOUS,
            ChangeDirection.UNCHANGED,
        ),
    ],
)
def test_role_comparison_and_direction_scenarios(
    text: str,
    role: FactRole,
    comparison: ComparisonType,
    direction: ChangeDirection,
) -> None:
    analysis = EventAnalyzer().analyze(news_id=uuid4(), raw_content=text)

    fact = analysis.financial_facts[0]
    assert fact.fact_role == role
    assert fact.comparison_type == comparison
    assert fact.change_direction == direction


def test_no_event_found_status_for_text_without_numbers_or_rules() -> None:
    analysis = EventAnalyzer().analyze(
        news_id=uuid4(), raw_content="Обычное корпоративное сообщение."
    )

    assert analysis.status == EventAnalysisStatus.NO_EVENT_FOUND
    assert analysis.primary_event_type == EventType.UNKNOWN


def test_analysis_links_child_records_to_analysis_id() -> None:
    analysis = EventAnalyzer().analyze(
        news_id=uuid4(),
        raw_content="Выручка за 2025 год составила 1 млрд руб.",
    )

    assert all(event.analysis_id == analysis.id for event in analysis.events)
    assert all(fact.analysis_id == analysis.id for fact in analysis.financial_facts)


@pytest.mark.parametrize(
    ("text", "raw_value", "normalized", "currency", "scale"),
    [
        (
            "Выручка составила 12.5 млрд руб.",
            Decimal("12.5"),
            Decimal("12500000000.0"),
            Currency.RUB,
            ValueScale.BILLION,
        ),
        (
            "Выручка составила ₽12,5 млрд.",
            Decimal("12.5"),
            Decimal("12500000000.0"),
            Currency.RUB,
            ValueScale.BILLION,
        ),
        (
            "Выручка составила 1 250 млн рублей.",
            Decimal("1250"),
            Decimal("1250000000"),
            Currency.RUB,
            ValueScale.MILLION,
        ),
        (
            "Выручка составила 2,4 млрд долларов.",
            Decimal("2.4"),
            Decimal("2400000000.0"),
            Currency.USD,
            ValueScale.BILLION,
        ),
        (
            "Выручка составила -3,5 млрд рублей.",
            Decimal("-3.5"),
            Decimal("-3500000000.0"),
            Currency.RUB,
            ValueScale.BILLION,
        ),
        (
            "Revenue amounted to 7 thousand RUB.",
            Decimal("7"),
            Decimal("7000"),
            Currency.RUB,
            ValueScale.THOUSAND,
        ),
        (
            "Revenue amounted to 4 million CNY.",
            Decimal("4"),
            Decimal("4000000"),
            Currency.CNY,
            ValueScale.MILLION,
        ),
        (
            "Revenue amounted to 5 million евро.",
            Decimal("5"),
            Decimal("5000000"),
            Currency.EUR,
            ValueScale.MILLION,
        ),
    ],
)
def test_required_number_formats(
    text: str,
    raw_value: Decimal,
    normalized: Decimal,
    currency: Currency,
    scale: ValueScale,
) -> None:
    fact = EventAnalyzer().analyze(news_id=uuid4(), raw_content=text).financial_facts[0]

    assert fact.raw_value == raw_value
    assert fact.normalized_value == normalized
    assert fact.currency == currency
    assert fact.scale == scale
    assert fact.extractor_version == "financial-facts-v1"
    assert fact.rule_id


def test_dates_years_uuid_http_status_and_ticker_are_not_financial_values() -> None:
    text = (
        "Документ № 15 от 01.02.2026. UUID 123e4567-e89b-12d3-a456-426614174000. "
        "HTTP status 404. Ticker SBER."
    )

    analysis = EventAnalyzer().analyze(news_id=uuid4(), raw_content=text)

    assert analysis.financial_facts == []


def test_two_metrics_and_two_values_in_one_sentence_are_linked_locally() -> None:
    analysis = EventAnalyzer().analyze(
        news_id=uuid4(),
        raw_content="Выручка составила 120 млрд рублей, чистая прибыль — 15 млрд рублей.",
    )

    values_by_metric = {fact.metric: fact.raw_value for fact in analysis.financial_facts}
    assert values_by_metric[FinancialMetric.REVENUE] == Decimal("120")
    assert values_by_metric[FinancialMetric.NET_PROFIT] == Decimal("15")


def test_change_on_vs_target_to_are_distinguished() -> None:
    analysis = EventAnalyzer().analyze(
        news_id=uuid4(),
        raw_content="Рентабельность выросла на 3 п.п. до 15%.",
    )

    change, target = analysis.financial_facts
    assert change.fact_role == FactRole.CHANGE
    assert change.change_direction == ChangeDirection.UP
    assert change.change_value == Decimal("3")
    assert target.fact_role == FactRole.ACTUAL
    assert target.change_value is None


def test_multiple_events_are_preserved_and_mark_analysis_ambiguous() -> None:
    analysis = EventAnalyzer().analyze(
        news_id=uuid4(),
        raw_content="Компания раскрыла финансовые результаты и рекомендовала дивиденды.",
    )

    assert {event.event_type for event in analysis.events} == {
        EventType.FINANCIAL_RESULTS,
        EventType.DIVIDEND,
    }
    assert analysis.status == EventAnalysisStatus.AMBIGUOUS


def test_evidence_spans_point_to_original_raw_content() -> None:
    raw_content = (
        "Сбербанк сообщил, что чистая прибыль по МСФО за 2025 год "
        "выросла на 18% до 850 млрд рублей."
    )

    analysis = EventAnalyzer().analyze(news_id=uuid4(), raw_content=raw_content)

    for event in analysis.events:
        assert raw_content[event.start_position : event.end_position] == event.evidence_text
        assert event.rule_id == event.matched_rule
    for fact in analysis.financial_facts:
        assert raw_content[fact.start_position : fact.end_position] == fact.evidence_text
        assert fact.rule_id == fact.matched_rule


def test_long_text_regression_stays_linear_enough() -> None:
    sentence = "Выручка составила 1 млрд рублей, EBITDA составила 2 млрд рублей. "
    raw_content = sentence * 320

    started = perf_counter()
    analysis = EventAnalyzer().analyze(news_id=uuid4(), raw_content=raw_content)
    elapsed = perf_counter() - started

    assert len(raw_content) > 10_000
    assert len(analysis.financial_facts) >= 100
    assert elapsed < 5
