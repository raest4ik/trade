from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, Protocol, cast

import httpx

from src.free_live_issuer_accumulation.domain import sha256_payload
from src.production_dry_run_burnin_v1.domain import MoexSessionEvidence, MoexSessionStatus

MOEX_SESSION_SOURCE = "MOEX_ISS_TQBR_DAILY_CANDLE"


class MoexSessionVerifier(Protocol):
    def verify(self, trading_date: date) -> MoexSessionEvidence: ...


class MoexIssSessionVerifier:
    """Confirms a TQBR trading day from an official daily candle, fail closed otherwise."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        user_agent: str,
        reference_ticker: str = "SBER",
        http_client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        parsed = httpx.URL(base_url.rstrip("/"))
        if parsed.scheme != "https" or parsed.host != "iss.moex.com":
            raise ValueError("MOEX ISS base URL is not allowed")
        self._base_url = str(parsed).rstrip("/")
        self._timeout = timeout_seconds
        self._user_agent = user_agent
        self._ticker = reference_ticker.strip().upper()
        self._client = http_client
        self._clock = clock or (lambda: datetime.now(UTC))

    def verify(self, trading_date: date) -> MoexSessionEvidence:
        checked_at = self._clock().astimezone(UTC)
        if trading_date.weekday() >= 5:
            return MoexSessionEvidence(
                trading_date=trading_date.isoformat(),
                status=MoexSessionStatus.CLOSED,
                source="MOEX_WEEKEND_RULE",
                checked_at=checked_at,
                reason="MARKET_SESSION_CLOSED",
            )
        try:
            payload = self._request(trading_date)
            table = cast("dict[str, Any]", payload["candles"])
            columns = cast("list[str]", table["columns"])
            rows = cast("list[list[Any]]", table["data"])
            begin_index = columns.index("begin")
            matched = any(
                str(row[begin_index]).startswith(trading_date.isoformat()) for row in rows
            )
            if matched:
                return MoexSessionEvidence(
                    trading_date=trading_date.isoformat(),
                    status=MoexSessionStatus.OPEN,
                    source=MOEX_SESSION_SOURCE,
                    checked_at=checked_at,
                    evidence_sha=sha256_payload(payload),
                )
            return MoexSessionEvidence(
                trading_date=trading_date.isoformat(),
                status=MoexSessionStatus.UNKNOWN,
                source=MOEX_SESSION_SOURCE,
                checked_at=checked_at,
                evidence_sha=sha256_payload(payload),
                reason="MOEX_SESSION_STATUS_UNKNOWN",
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return MoexSessionEvidence(
                trading_date=trading_date.isoformat(),
                status=MoexSessionStatus.UNKNOWN,
                source=MOEX_SESSION_SOURCE,
                checked_at=checked_at,
                reason="MOEX_SESSION_STATUS_UNKNOWN",
            )

    def _request(self, trading_date: date) -> dict[str, Any]:
        endpoint = (
            f"{self._base_url}/engines/stock/markets/shares/boards/TQBR/"
            f"securities/{self._ticker}/candles.json"
        )
        params = {
            "from": trading_date.isoformat(),
            "till": trading_date.isoformat(),
            "interval": "24",
            "iss.meta": "off",
            "iss.only": "candles",
        }
        if self._client is not None:
            response = self._client.get(
                endpoint,
                params=params,
                headers={"User-Agent": self._user_agent},
            )
        else:
            response = httpx.get(
                endpoint,
                params=params,
                headers={"User-Agent": self._user_agent},
                timeout=self._timeout,
                follow_redirects=False,
            )
        response.raise_for_status()
        if len(response.content) > 1_000_000:
            raise ValueError("MOEX_SESSION_RESPONSE_TOO_LARGE")
        return cast("dict[str, Any]", response.json())
