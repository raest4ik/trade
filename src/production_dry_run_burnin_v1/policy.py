from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime
from typing import Protocol

import httpx

from src.moex_trading_calendar_v2.domain import CALENDAR_SOURCE_URL, MoexSessionEvidenceV2
from src.moex_trading_calendar_v2.moex import MoexTradingCalendarResolver
from src.production_dry_run_burnin_v1.domain import MoexSessionEvidence, MoexSessionStatus


class MoexSessionVerifier(Protocol):
    def verify(self, trading_date: date) -> MoexSessionEvidence: ...


class MoexIssSessionVerifier:
    """Backward-compatible burn-in adapter for the authoritative V2 calendar."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        user_agent: str,
        reference_ticker: str = "SBER",
        max_retries: int = 0,
        calendar_url: str = CALENDAR_SOURCE_URL,
        http_client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        del reference_ticker
        self._resolver = MoexTradingCalendarResolver(
            calendar_url=calendar_url,
            runtime_base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            user_agent=user_agent,
            http_client=http_client,
            clock=clock,
        )

    def verify(self, trading_date: date) -> MoexSessionEvidence:
        return _adapt(self._resolver.resolve(trading_date))

    def verify_many(self, trading_dates: Sequence[date]) -> list[MoexSessionEvidence]:
        return [_adapt(evidence) for evidence in self._resolver.resolve_many(trading_dates)]


def _adapt(evidence: MoexSessionEvidenceV2) -> MoexSessionEvidence:
    return MoexSessionEvidence(
        trading_date=evidence.calendar_date.isoformat(),
        status=MoexSessionStatus(evidence.status.value),
        source=evidence.source,
        checked_at=evidence.checked_at,
        evidence_sha=evidence.evidence_sha,
        reason=evidence.reason,
        session_policy_version=evidence.policy_version,
        calendar_date=evidence.calendar_date.isoformat(),
        market=evidence.market,
        board=evidence.board,
        session_kind=evidence.session_kind.value,
        moex_business_date=(
            evidence.moex_business_date.isoformat()
            if evidence.moex_business_date is not None
            else None
        ),
        scheduled_open_at=evidence.scheduled_open_at,
        scheduled_close_at=evidence.scheduled_close_at,
        source_url=evidence.source_url,
        source_id=evidence.source_id,
        source_published_at=evidence.source_published_at,
        schedule_version=evidence.schedule_version,
        effective_at=evidence.effective_at,
        runtime_status=evidence.runtime_status.value,
        runtime_source=evidence.runtime_source,
        runtime_evidence_sha=evidence.runtime_evidence_sha,
    )
