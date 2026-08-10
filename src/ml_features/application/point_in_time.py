from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
from itertools import pairwise
from typing import Protocol

from src.ml_features.domain.entities import FEATURE_HORIZONS_MINUTES


class CompletedCandle(Protocol):
    @property
    def end_at(self) -> datetime: ...

    @property
    def close(self) -> Decimal: ...

    @property
    def volume(self) -> Decimal: ...


class PointInTimeViolationError(ValueError):
    """Raised when a future observation reaches feature computation."""


@dataclass(frozen=True, slots=True)
class PointInTimeMarketFeatures:
    returns: dict[int, Decimal | None]
    log_returns: dict[int, Decimal | None]
    realized_volatility: dict[int, Decimal | None]
    volume_last_1m: Decimal | None
    volume_sums: Mapping[int, Decimal | None]
    volume_ratio_5m_vs_60m: Decimal | None
    last_observation_end_at: datetime | None
    missing: tuple[str, ...]


class PointInTimeFeatureBuilder:
    def build(
        self,
        *,
        candles: Sequence[CompletedCandle],
        as_of: datetime,
        prefix: str,
    ) -> PointInTimeMarketFeatures:
        ordered = sorted(candles, key=lambda item: item.end_at)
        future = [item for item in ordered if item.end_at > as_of]
        if future:
            raise PointInTimeViolationError(
                f"{prefix} candle ends after point-in-time cutoff: {future[0].end_at.isoformat()}"
            )
        if not ordered:
            return PointInTimeMarketFeatures(
                returns={horizon: None for horizon in FEATURE_HORIZONS_MINUTES},
                log_returns={horizon: None for horizon in FEATURE_HORIZONS_MINUTES},
                realized_volatility={horizon: None for horizon in (15, 30, 60)},
                volume_last_1m=None,
                volume_sums={horizon: None for horizon in (5, 15, 60)},
                volume_ratio_5m_vs_60m=None,
                last_observation_end_at=None,
                missing=(f"{prefix}_candles",),
            )
        current = ordered[-1]
        returns: dict[int, Decimal | None] = {}
        log_returns: dict[int, Decimal | None] = {}
        missing: list[str] = []
        for horizon in FEATURE_HORIZONS_MINUTES:
            baseline = _last_at_or_before(ordered, as_of - timedelta(minutes=horizon))
            if baseline is None or baseline.close <= 0 or current.close <= 0:
                returns[horizon] = None
                log_returns[horizon] = None
                missing.append(f"{prefix}_pre_return_{horizon}m")
                continue
            ratio = current.close / baseline.close
            returns[horizon] = ratio - Decimal("1")
            with localcontext() as context:
                context.prec = 28
                log_returns[horizon] = ratio.ln()
        volatility = {
            horizon: _realized_volatility(
                [
                    candle.close
                    for candle in ordered
                    if candle.end_at > as_of - timedelta(minutes=horizon)
                ]
            )
            for horizon in (15, 30, 60)
        }
        for horizon, value in volatility.items():
            if value is None:
                missing.append(f"{prefix}_realized_volatility_{horizon}m")
        volume_sums: dict[int, Decimal | None] = {
            horizon: sum(
                (
                    candle.volume
                    for candle in ordered
                    if candle.end_at > as_of - timedelta(minutes=horizon)
                ),
                Decimal("0"),
            )
            for horizon in (5, 15, 60)
        }
        denominator = volume_sums[60]
        numerator = volume_sums[5]
        ratio = (
            None
            if denominator is None or numerator is None or denominator <= 0
            else numerator / denominator
        )
        return PointInTimeMarketFeatures(
            returns=returns,
            log_returns=log_returns,
            realized_volatility=volatility,
            volume_last_1m=current.volume,
            volume_sums=volume_sums,
            volume_ratio_5m_vs_60m=ratio,
            last_observation_end_at=current.end_at,
            missing=tuple(sorted(missing)),
        )


def _last_at_or_before(
    candles: Sequence[CompletedCandle], cutoff: datetime
) -> CompletedCandle | None:
    return next((item for item in reversed(candles) if item.end_at <= cutoff), None)


def _realized_volatility(closes: list[Decimal]) -> Decimal | None:
    if len(closes) < 3 or any(value <= 0 for value in closes):
        return None
    with localcontext() as context:
        context.prec = 28
        log_returns = [(current / previous).ln() for previous, current in pairwise(closes)]
        mean = sum(log_returns, Decimal("0")) / Decimal(len(log_returns))
        variance = sum(((value - mean) ** 2 for value in log_returns), Decimal("0")) / Decimal(
            len(log_returns)
        )
        return variance.sqrt()
