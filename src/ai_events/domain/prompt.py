from __future__ import annotations

import hashlib
import json

from src.ai_events.domain.schema import AIEventOutput
from src.events.domain.enums import (
    ChangeDirection,
    ComparisonType,
    Currency,
    EventType,
    FactRole,
    FactUnit,
    FinancialMetric,
    PeriodType,
    ValueScale,
)

PROMPT_VERSION = "ai-event-prompt-v0"
SCHEMA_VERSION = "ai-event-schema-v0"
ANALYSIS_VERSION = "ai-event-v0"
FACT_EXTRACTOR_VERSION = "ai-financial-facts-v0"


def _values(enum_type: type[object]) -> str:
    return ", ".join(str(item.value) for item in enum_type)  # type: ignore[attr-defined]


SYSTEM_PROMPT = f"""You extract corporate events and financial facts from one news text.
Return only data matching the provided structured-output schema. This is zero-shot extraction.

Event ontology (use only these values): {_values(EventType)}.
Financial metrics (use only these values): {_values(FinancialMetric)}.
Units: {_values(FactUnit)}. Currencies: {_values(Currency)}. Scales: {_values(ValueScale)}.
Fact roles: {_values(FactRole)}. Period types: {_values(PeriodType)}.
Comparison types: {_values(ComparisonType)}. Change directions: {_values(ChangeDirection)}.

Rules:
- Extract only facts explicitly supported by the text. It is valid to return no events and no facts.
- Mark at most one event as primary. Use UNKNOWN rather than inventing a type.
- evidence_text must be one exact, contiguous, non-empty substring copied from the input.
- Use decimal strings with a dot, no thousands separators, and no exponent.
- normalized_value is the numeric value after applying scale semantics; keep unit, currency,
  and scale.
- Use metric OTHER with a concise metric_name for unsupported named KPIs such as ROE or NIM.
- A cooperation or partnership without a disclosed material commercial contract is OTHER,
  not MAJOR_CONTRACT.
- Use UNKNOWN for missing or uncertain fact_role, period_type, comparison_type, or change_direction.
- Use UNCHANGED only when the evidence explicitly says the value did not change.
- YEAR means a full year, HALF_YEAR six months, QUARTER one quarter, NINE_MONTHS nine months,
  MONTH one month, DATE_RANGE an explicit range. Do not infer a period from publication date.
- Roman I/II/III/IV quarter wording maps to QUARTER. Apply a shared reporting period to all
  listed metrics only when the grammar explicitly makes it common to them.
- For dividends, a board recommendation is FORECAST; shareholder approval or a paid/declared
  historical amount is ACTUAL. Distinguish DIVIDEND_PER_SHARE from DIVIDEND_TOTAL.
- Guidance, forecasts, targets, and outlook statements use event GUIDANCE and fact role FORECAST
  or TARGET as the wording supports. Historical results mentioned as context remain ACTUAL.
- Attach an explicit growth or decline amount to its main fact through change_direction,
  change_value, and change_unit; do not create a separate numeric fact for the change.
- Put non-fatal ambiguity notes in warnings. Never include hidden reasoning or chain-of-thought.
"""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_hash() -> str:
    return sha256_text(SYSTEM_PROMPT)


def output_schema() -> dict[str, object]:
    return AIEventOutput.model_json_schema(mode="serialization")


def schema_hash() -> str:
    return sha256_text(canonical_json(output_schema()))
