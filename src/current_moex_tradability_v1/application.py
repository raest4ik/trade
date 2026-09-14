from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from src.current_moex_tradability_v1.domain import (
    INTENDED_BOARD,
    TRADABILITY_SOURCE,
    CandidateClassification,
    CurrentMoexState,
    CurrentTradabilityStatus,
    CurrentUniverseResolution,
    TickerMigrationEvidence,
)

ACTIVE_SECURITY_STATUS = "A"
EXPLICIT_DISABLED_TRADING_STATUSES = {"D"}
RUSAGRO_MIGRATION_SOURCES = (
    "https://www.rusagrogroup.ru/investors/shares/",
    "https://www.rusagrogroup.ru/investors/news-events/press-releases/single-view/article/1384/",
    "https://www.rusagrogroup.ru/investors/news-events/press-releases/single-view/article/1396/",
)


def verified_ticker_migrations() -> tuple[TickerMigrationEvidence, ...]:
    return (
        TickerMigrationEvidence(
            legacy_ticker="AGRO",
            current_ticker="RAGR",
            relation="ISSUER_SUCCESSION_WITH_DISTINCT_INSTRUMENT_IDENTITIES",
            legacy_instrument_identity="US7496552057:ROS_AGRO_PLC_GDR",
            current_instrument_identity="RU000A0JQUZ6:PJSC_RUSAGRO_GROUP_COMMON_SHARE",
            sources=RUSAGRO_MIGRATION_SOURCES,
        ),
    )


def classify_candidate(
    canonical: dict[str, Any],
    current: CurrentMoexState | None,
    *,
    fetched_at: datetime,
    payload_sha: str,
) -> CandidateClassification:
    ticker = str(canonical["ticker"]).strip().upper()
    if current is None:
        return _classification(
            ticker,
            CurrentTradabilityStatus.INELIGIBLE_NO_SECURITY_ROW,
            "NO_CURRENT_MOEX_SECURITY_ROW",
            fetched_at,
            payload_sha,
        )
    if current.board != INTENDED_BOARD:
        return _classification(
            ticker,
            CurrentTradabilityStatus.INELIGIBLE_WRONG_BOARD,
            "CURRENT_BOARD_NOT_TQBR",
            fetched_at,
            payload_sha,
            current=current,
        )
    if current.lot_size is None or current.lot_size <= 0:
        return _classification(
            ticker,
            CurrentTradabilityStatus.INELIGIBLE_INVALID_LOT,
            "CURRENT_LOT_NOT_POSITIVE",
            fetched_at,
            payload_sha,
            current=current,
        )
    if current.security_status is None:
        return _classification(
            ticker,
            CurrentTradabilityStatus.UNKNOWN,
            "CURRENT_SECURITY_STATUS_UNKNOWN",
            fetched_at,
            payload_sha,
            current=current,
        )
    if (
        current.security_status != ACTIVE_SECURITY_STATUS
        or current.trading_status in EXPLICIT_DISABLED_TRADING_STATUSES
    ):
        return _classification(
            ticker,
            CurrentTradabilityStatus.INELIGIBLE_TRADING_DISABLED,
            "CURRENT_TRADING_DISABLED",
            fetched_at,
            payload_sha,
            current=current,
        )
    if not current.has_marketdata_row:
        return _classification(
            ticker,
            CurrentTradabilityStatus.INELIGIBLE_NO_CURRENT_MARKET_DATA,
            "NO_CURRENT_MARKETDATA_ROW",
            fetched_at,
            payload_sha,
            current=current,
        )
    return _classification(
        ticker,
        CurrentTradabilityStatus.ELIGIBLE,
        "CURRENT_MOEX_TRADABLE",
        fetched_at,
        payload_sha,
        current=current,
    )


def resolve_from_states(
    canonical: Sequence[dict[str, Any]],
    states: dict[str, CurrentMoexState],
    *,
    fetched_at: datetime,
    payload_sha: str,
    source_time: datetime | None = None,
) -> CurrentUniverseResolution:
    return CurrentUniverseResolution(
        fetched_at=_utc(fetched_at),
        source_time=None if source_time is None else _utc(source_time),
        payload_sha=payload_sha,
        classifications=tuple(
            classify_candidate(
                row,
                states.get(str(row["ticker"]).strip().upper()),
                fetched_at=fetched_at,
                payload_sha=payload_sha,
            )
            for row in canonical
        ),
    )


def _classification(
    ticker: str,
    status: CurrentTradabilityStatus,
    reason: str,
    fetched_at: datetime,
    payload_sha: str,
    current: CurrentMoexState | None = None,
) -> CandidateClassification:
    return CandidateClassification(
        ticker=ticker,
        current_moex_status=status,
        candidate_eligible=status == CurrentTradabilityStatus.ELIGIBLE,
        reason=reason,
        source=TRADABILITY_SOURCE,
        fetched_at=_utc(fetched_at),
        payload_sha=payload_sha,
        board=None if current is None else current.board,
        lot_size=None if current is None else current.lot_size,
        security_status=None if current is None else current.security_status,
        trading_status=None if current is None else current.trading_status,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("CURRENT_TRADABILITY_TIMESTAMP_NAIVE")
    return value.astimezone(UTC)
