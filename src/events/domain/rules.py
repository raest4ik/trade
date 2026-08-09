from __future__ import annotations

import re
from dataclasses import dataclass

from src.events.domain.enums import EventType, FinancialMetric


@dataclass(frozen=True, slots=True)
class EventRule:
    rule_id: str
    event_type: EventType
    pattern: re.Pattern[str]
    priority: int
    confidence: str
    negative_pattern: re.Pattern[str] | None = None


@dataclass(frozen=True, slots=True)
class MetricRule:
    rule_id: str
    metric: FinancialMetric
    pattern: re.Pattern[str]
    priority: int


def compile_rule(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)


EVENT_RULES: tuple[EventRule, ...] = (
    EventRule(
        "event.financial_results.ru",
        EventType.FINANCIAL_RESULTS,
        compile_rule(
            r"(?<![\w])(?:финансов\w*\s+результат\w*|мсфо|рсбу|выручк\w*|чист\w+\s+прибыл\w*|ebitda|операционн\w+\s+прибыл\w*)(?![\w])"
        ),
        10,
        "0.96",
    ),
    EventRule(
        "event.financial_results.en",
        EventType.FINANCIAL_RESULTS,
        compile_rule(
            r"(?<![\w])(?:financial\s+results|revenue|net\s+profit|ebitda|operating\s+profit)(?![\w])"
        ),
        20,
        "0.94",
    ),
    EventRule(
        "event.dividend",
        EventType.DIVIDEND,
        compile_rule(
            r"(?<![\w])(?:дивиденд\w*|совет\s+директоров\s+рекомендовал|дата\s+закрытия\s+реестра|дивидендн\w+\s+доходност\w*|dividend\w*)(?![\w])"
        ),
        30,
        "0.97",
    ),
    EventRule(
        "event.guidance",
        EventType.GUIDANCE,
        compile_rule(
            r"(?<![\w])(?:прогноз\w*|ориентир\w*|цел\w*|ожидает\w*|планирует\w*|подтвердил\w*\s+прогноз|пересмотрел\w*\s+прогноз|повысил\w*\s+прогноз|понизил\w*\s+прогноз|улучшил\w*\s+прогноз|guidance|forecast|outlook)(?![\w])"
        ),
        5,
        "0.98",
    ),
    EventRule(
        "event.ma",
        EventType.MERGER_ACQUISITION,
        compile_rule(
            r"(?<![\w])(?:приобретени\w*|покупк\w+\s+дол[ии]|продаж\w+\s+актив\w*|слияни\w*|поглощени\w*|консолидир\w+\s+(?:до\s+)?\d+(?:[,.]\d+)?%?|acquisition|merger)(?![\w])"
        ),
        50,
        "0.93",
    ),
    EventRule(
        "event.sanctions",
        EventType.SANCTIONS,
        compile_rule(r"(?<![\w])(?:санкци\w*|санкционн\w+|ofac|sdn|sanction\w*)(?![\w])"),
        60,
        "0.95",
    ),
    EventRule(
        "event.buyback",
        EventType.BUYBACK,
        compile_rule(r"(?<![\w])(?:buyback|обратн\w+\s+выкуп\w*|выкуп\w+\s+акци\w*)(?![\w])"),
        70,
        "0.91",
    ),
    EventRule(
        "event.management",
        EventType.MANAGEMENT_CHANGE,
        compile_rule(
            r"(?<![\w])(?:назначил\w*|покинул\w+\s+пост|генеральн\w+\s+директор|management\s+change|ceo)(?![\w])"
        ),
        80,
        "0.89",
    ),
    EventRule(
        "event.contract",
        EventType.MAJOR_CONTRACT,
        compile_rule(r"(?<![\w])(?:контракт\w*|договор\w*|major\s+contract)(?![\w])"),
        90,
        "0.88",
    ),
    EventRule(
        "event.production",
        EventType.PRODUCTION_UPDATE,
        compile_rule(r"(?<![\w])(?:производств\w*|добыч\w*|добы[лт]\w*|production|output)(?![\w])"),
        100,
        "0.88",
    ),
    EventRule(
        "event.rating",
        EventType.CREDIT_RATING,
        compile_rule(
            r"(?<![\w])(?:кредитн\w+\s+рейтинг|credit\s+rating|рейтинг\w+\s+агентств\w*)(?![\w])"
        ),
        110,
        "0.9",
    ),
    EventRule(
        "event.debt",
        EventType.DEBT_FINANCING,
        compile_rule(r"(?<![\w])(?:облигаци\w*|кредит\w*|заем\w*|debt|bond\w*)(?![\w])"),
        120,
        "0.87",
    ),
    EventRule(
        "event.share_issuance",
        EventType.SHARE_ISSUANCE,
        compile_rule(
            r"(?<![\w])(?:допэмисси\w*|размещени\w+\s+акци\w*|share\s+issuance|secondary\s+offering)(?![\w])"
        ),
        130,
        "0.86",
    ),
    EventRule(
        "event.regulatory",
        EventType.REGULATORY_ACTION,
        compile_rule(
            r"(?<![\w])(?:цб\s+рф|банк\s+россии|регулятор\w*|лицензи\w*|regulatory\s+action)(?![\w])"
        ),
        140,
        "0.85",
    ),
    EventRule(
        "event.litigation",
        EventType.LITIGATION,
        compile_rule(
            r"(?<![\w])(?:иск\w*|судебн\w+\s+спор\w*|арбитраж\w*|litigation|lawsuit)(?![\w])"
        ),
        150,
        "0.85",
    ),
    EventRule(
        "event.corporate_action",
        EventType.CORPORATE_ACTION,
        compile_rule(
            r"(?<![\w])(?:сплит\w*|консолидаци\w+\s+акци\w*|реорганизаци\w*|corporate\s+action|stock\s+split)(?![\w])"
        ),
        160,
        "0.84",
    ),
)

METRIC_RULES: tuple[MetricRule, ...] = (
    MetricRule(
        "metric.revenue",
        FinancialMetric.REVENUE,
        compile_rule(r"(?<![\w])(?:выручк\w*|revenue)(?![\w])"),
        10,
    ),
    MetricRule(
        "metric.net_profit",
        FinancialMetric.NET_PROFIT,
        compile_rule(
            r"(?<![\w])(?:чист\w+\s+прибыл\w*|(?:нормализованн\w+\s+)?прибыл\w+\s+акционер\w*|net\s+profit)(?![\w])"
        ),
        20,
    ),
    MetricRule(
        "metric.ebitda", FinancialMetric.EBITDA, compile_rule(r"(?<![\w])ebitda(?![\w])"), 30
    ),
    MetricRule(
        "metric.operating_profit",
        FinancialMetric.OPERATING_PROFIT,
        compile_rule(r"(?<![\w])(?:операционн\w+\s+прибыл\w*|operating\s+profit)(?![\w])"),
        40,
    ),
    MetricRule(
        "metric.fcf",
        FinancialMetric.FREE_CASH_FLOW,
        compile_rule(r"(?<![\w])(?:свободн\w+\s+денежн\w+\s+поток|free\s+cash\s+flow|fcf)(?![\w])"),
        50,
    ),
    MetricRule(
        "metric.capex",
        FinancialMetric.CAPEX,
        compile_rule(r"(?<![\w])(?:капзатрат\w*|capex)(?![\w])"),
        60,
    ),
    MetricRule(
        "metric.net_debt",
        FinancialMetric.NET_DEBT,
        compile_rule(r"(?<![\w])(?:чист\w+\s+долг|net\s+debt)(?![\w])"),
        70,
    ),
    MetricRule(
        "metric.dividend_per_share",
        FinancialMetric.DIVIDEND_PER_SHARE,
        compile_rule(r"(?<![\w])(?:дивиденд\w*\s+на\s+акци\w*|dividend\s+per\s+share)(?![\w])"),
        80,
    ),
    MetricRule(
        "metric.dividend_total",
        FinancialMetric.DIVIDEND_TOTAL,
        compile_rule(r"(?<![\w])(?:общ\w+\s+сумм\w+\s+дивиденд\w*|dividend\s+total)(?![\w])"),
        90,
    ),
    MetricRule(
        "metric.production",
        FinancialMetric.PRODUCTION_VOLUME,
        compile_rule(r"(?<![\w])(?:производств\w*|добыч\w*|добы[лт]\w*|production|output)(?![\w])"),
        100,
    ),
    MetricRule(
        "metric.contract_value",
        FinancialMetric.CONTRACT_VALUE,
        compile_rule(r"(?<![\w])(?:стоимост\w+\s+контракт\w*|contract\s+value)(?![\w])"),
        110,
    ),
    MetricRule(
        "metric.ownership",
        FinancialMetric.OWNERSHIP_PERCENT,
        compile_rule(r"(?<![\w])(?:дол[яи]|ownership|stake)(?![\w])"),
        120,
    ),
)
