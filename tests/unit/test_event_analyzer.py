from __future__ import annotations

from decimal import Decimal
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
            ChangeDirection.INCREASE,
        ),
        (
            "Чистая прибыль за 2025 год снизилась на 8% год к году.",
            FactRole.CHANGE,
            ComparisonType.YEAR_OVER_YEAR,
            ChangeDirection.DECREASE,
        ),
        (
            "Прогноз EBITDA на 2026 год составляет 100 млрд руб.",
            FactRole.FORECAST,
            ComparisonType.NONE,
            ChangeDirection.NONE,
        ),
        (
            "Консенсус по выручке на 2026 год составляет 1 трлн руб.",
            FactRole.CONSENSUS,
            ComparisonType.NONE,
            ChangeDirection.NONE,
        ),
        (
            "Выручка годом ранее составляла 900 млрд руб.",
            FactRole.PREVIOUS,
            ComparisonType.VERSUS_PREVIOUS,
            ChangeDirection.NONE,
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
