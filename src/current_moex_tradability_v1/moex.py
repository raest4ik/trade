from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

import httpx

from src.current_moex_tradability_v1.application import resolve_from_states
from src.current_moex_tradability_v1.domain import (
    CurrentMoexState,
    CurrentUniverseResolution,
)
from src.free_live_issuer_accumulation.domain import sha256_payload

MAX_RESPONSE_BYTES = 2_000_000
MOEX_TIMEZONE = ZoneInfo("Europe/Moscow")


class CurrentMoexTradabilityError(RuntimeError):
    pass


class CurrentMoexTradabilityResolver:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        user_agent: str,
        http_client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        parsed = httpx.URL(base_url.rstrip("/"))
        if parsed.scheme != "https" or parsed.host != "iss.moex.com":
            raise ValueError("MOEX ISS base URL is not allowed")
        self._base_url = str(parsed).rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._user_agent = user_agent
        self._client = http_client
        self._clock = clock or (lambda: datetime.now(UTC))

    def resolve(self, canonical: Sequence[dict[str, Any]]) -> CurrentUniverseResolution:
        payload = self._request()
        fetched_at = self._clock().astimezone(UTC)
        payload_sha = sha256_payload(payload)
        securities = _table(payload, "securities", "SECID")
        marketdata = _table(payload, "marketdata", "SECID")
        states: dict[str, CurrentMoexState] = {}
        source_times: list[datetime] = []
        for ticker, security in securities.items():
            market = marketdata.get(ticker)
            states[ticker] = CurrentMoexState(
                ticker=ticker,
                board=_text(security.get("BOARDID")),
                short_name=_text(security.get("SHORTNAME")),
                lot_size=_integer(security.get("LOTSIZE")),
                security_status=_text(security.get("STATUS")),
                trading_status=None if market is None else _text(market.get("TRADINGSTATUS")),
                has_marketdata_row=market is not None,
            )
            if market is not None and (parsed := _moex_time(market.get("SYSTIME"))) is not None:
                source_times.append(parsed)
        return resolve_from_states(
            canonical,
            states,
            fetched_at=fetched_at,
            payload_sha=payload_sha,
            source_time=max(source_times, default=None),
        )

    def _request(self) -> dict[str, Any]:
        endpoint = f"{self._base_url}/engines/stock/markets/shares/boards/TQBR/securities.json"
        params = {
            "iss.meta": "off",
            "iss.only": "securities,marketdata",
            "securities.columns": "SECID,BOARDID,SHORTNAME,LOTSIZE,STATUS",
            "marketdata.columns": "SECID,BOARDID,LAST,BID,OFFER,SYSTIME,TRADINGSTATUS",
        }
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = (
                    self._client.get(
                        endpoint, params=params, headers={"User-Agent": self._user_agent}
                    )
                    if self._client is not None
                    else httpx.get(
                        endpoint,
                        params=params,
                        headers={"User-Agent": self._user_agent},
                        timeout=self._timeout,
                        follow_redirects=False,
                    )
                )
                response.raise_for_status()
                if len(response.content) > MAX_RESPONSE_BYTES:
                    raise CurrentMoexTradabilityError("MOEX_CURRENT_STATE_RESPONSE_TOO_LARGE")
                return cast("dict[str, Any]", response.json())
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    time.sleep(min(0.25 * (attempt + 1), 1.0))
        raise CurrentMoexTradabilityError("MOEX_CURRENT_STATE_UNAVAILABLE") from last_error


def _table(payload: dict[str, Any], name: str, key: str) -> dict[str, dict[str, Any]]:
    try:
        table = cast("dict[str, Any]", payload[name])
        columns = cast("list[str]", table["columns"])
        rows = cast("list[list[Any]]", table["data"])
        key_index = columns.index(key)
        return {str(row[key_index]).upper(): dict(zip(columns, row, strict=True)) for row in rows}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise CurrentMoexTradabilityError("MOEX_CURRENT_STATE_MALFORMED") from exc


def _text(value: object) -> str | None:
    return None if value is None or not str(value).strip() else str(value).strip().upper()


def _integer(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed


def _moex_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=MOEX_TIMEZONE).astimezone(UTC)
    except ValueError:
        return None
