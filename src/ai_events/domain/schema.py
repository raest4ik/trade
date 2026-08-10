from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

DECIMAL_PATTERN = r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?$"
_EXPLICIT_UNCHANGED = re.compile(
    r"(?:без\s+изменен|не\s+измен|остал(?:ся|ась|ось|ись)\s+на\s+уровне|unchanged|flat)",
    re.IGNORECASE,
)


class StrictOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AIEventPrediction(StrictOutputModel):
    event_type: EventType
    is_primary: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str = Field(min_length=1)


class AIFinancialFactPrediction(StrictOutputModel):
    metric: FinancialMetric
    metric_name: str | None
    normalized_value: str = Field(pattern=DECIMAL_PATTERN)
    unit: FactUnit
    currency: Currency
    scale: ValueScale
    fact_role: FactRole
    period_type: PeriodType
    period_year: int | None
    period_quarter: int | None = Field(ge=1, le=4)
    comparison_type: ComparisonType
    change_direction: ChangeDirection
    change_value: str | None = Field(pattern=DECIMAL_PATTERN)
    change_unit: FactUnit | None
    evidence_text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if self.metric == FinancialMetric.OTHER and not (self.metric_name or "").strip():
            raise ValueError("metric_name is required when metric is OTHER")
        if self.period_type == PeriodType.QUARTER and self.period_quarter is None:
            raise ValueError("period_quarter is required for QUARTER period")
        if self.period_type != PeriodType.QUARTER and self.period_quarter is not None:
            raise ValueError("period_quarter is only valid for QUARTER period")
        if self.change_direction == ChangeDirection.UNCHANGED and not _EXPLICIT_UNCHANGED.search(
            self.evidence_text
        ):
            raise ValueError("UNCHANGED requires explicit unchanged evidence")
        if self.change_value is None and self.change_unit is not None:
            raise ValueError("change_unit requires change_value")
        return self

    def decimal_value(self) -> Decimal:
        return parse_decimal_string(self.normalized_value)

    def decimal_change_value(self) -> Decimal | None:
        if self.change_value is None:
            return None
        return parse_decimal_string(self.change_value)


class AIEventOutput(StrictOutputModel):
    events: list[AIEventPrediction]
    financial_facts: list[AIFinancialFactPrediction]
    warnings: list[str]

    @model_validator(mode="after")
    def validate_primary_event(self) -> Self:
        if sum(event.is_primary for event in self.events) > 1:
            raise ValueError("at most one event may be primary")
        return self


def parse_decimal_string(value: str) -> Decimal:
    if not re.fullmatch(DECIMAL_PATTERN, value):
        raise ValueError("decimal value must use a dot and no exponent")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal value") from exc
