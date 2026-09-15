from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

import httpx

from src.free_live_issuer_accumulation.domain import sha256_payload
from src.moex_trading_calendar_v2.application import (
    resolve_schedule_payload,
    unknown_evidence,
)
from src.moex_trading_calendar_v2.domain import (
    BOARD_SCOPE,
    CALENDAR_SOURCE_URL,
    MARKET_SCOPE,
    MoexSessionEvidenceV2,
    RuntimeSessionStatus,
)

MAX_CALENDAR_RESPONSE_BYTES = 1_000_000
MAX_RUNTIME_RESPONSE_BYTES = 2_000_000
MOEX_TIMEZONE = ZoneInfo("Europe/Moscow")
NEXT_DATA_PATTERN = re.compile(
    r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
    re.DOTALL,
)


class MoexTradingCalendarResolver:
    def __init__(
        self,
        *,
        calendar_url: str = CALENDAR_SOURCE_URL,
        runtime_base_url: str,
        timeout_seconds: float,
        max_retries: int,
        user_agent: str,
        http_client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        calendar = httpx.URL(calendar_url)
        runtime = httpx.URL(runtime_base_url.rstrip("/"))
        if (
            calendar.scheme != "https"
            or calendar.host != "www.moex.com"
            or calendar.path.rstrip("/") != "/ru/tradingcalendar"
        ):
            raise ValueError("MOEX calendar URL is not allowed")
        if runtime.scheme != "https" or runtime.host != "iss.moex.com":
            raise ValueError("MOEX ISS base URL is not allowed")
        self._calendar_url = str(calendar)
        self._runtime_base_url = str(runtime).rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._user_agent = user_agent
        self._client = http_client
        self._clock = clock or (lambda: datetime.now(UTC))

    def resolve(self, calendar_date: date) -> MoexSessionEvidenceV2:
        return self.resolve_many([calendar_date])[0]

    def resolve_many(self, calendar_dates: Sequence[date]) -> list[MoexSessionEvidenceV2]:
        if not calendar_dates:
            return []
        checked_at = self._clock().astimezone(UTC)
        runtime_status = RuntimeSessionStatus.NOT_APPLICABLE
        runtime_sha: str | None = None
        try:
            calendar_content = self._get_text(
                self._calendar_url,
                max_bytes=MAX_CALENDAR_RESPONSE_BYTES,
            )
            page_sha = sha256_payload(calendar_content)
            payload = extract_stock_schedule_payload(calendar_content)
            current_date = checked_at.astimezone(MOEX_TIMEZONE).date()
            if current_date in calendar_dates:
                runtime_status, runtime_sha = self._current_runtime_status()
            return [
                resolve_schedule_payload(
                    payload,
                    calendar_date,
                    checked_at=checked_at,
                    evidence_sha=page_sha,
                    runtime_status=(
                        runtime_status
                        if calendar_date == current_date
                        else RuntimeSessionStatus.NOT_APPLICABLE
                    ),
                    runtime_evidence_sha=(runtime_sha if calendar_date == current_date else None),
                )
                for calendar_date in calendar_dates
            ]
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return [
                unknown_evidence(
                    calendar_date,
                    checked_at=checked_at,
                    evidence_sha=None,
                    reason="AUTHORITATIVE_CALENDAR_UNAVAILABLE",
                )
                for calendar_date in calendar_dates
            ]

    def _current_runtime_status(self) -> tuple[RuntimeSessionStatus, str | None]:
        endpoint = (
            f"{self._runtime_base_url}/engines/stock/markets/shares/boards/TQBR/securities.json"
        )
        try:
            response = self._get_json(
                endpoint,
                params={
                    "iss.meta": "off",
                    "iss.only": "marketdata",
                    "marketdata.columns": "SECID,BOARDID,SYSTIME,TRADINGSTATUS",
                },
                max_bytes=MAX_RUNTIME_RESPONSE_BYTES,
            )
            table = cast("dict[str, Any]", response["marketdata"])
            columns = cast("list[str]", table["columns"])
            rows = cast("list[list[Any]]", table["data"])
            board_index = columns.index("BOARDID")
            status_index = columns.index("TRADINGSTATUS")
            scoped = [row for row in rows if str(row[board_index]) == BOARD_SCOPE]
            if not scoped:
                raise ValueError("runtime marketdata missing")
            status = (
                RuntimeSessionStatus.OPEN
                if any(str(row[status_index]).upper() == "T" for row in scoped)
                else RuntimeSessionStatus.CLOSED
            )
            return status, sha256_payload(response)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, IndexError):
            return RuntimeSessionStatus.UNKNOWN, None

    def _get_text(self, url: str, *, max_bytes: int) -> str:
        response = self._request(url)
        if len(response.content) > max_bytes:
            raise ValueError("MOEX_CALENDAR_RESPONSE_TOO_LARGE")
        return response.text

    def _get_json(
        self,
        url: str,
        *,
        params: dict[str, str],
        max_bytes: int,
    ) -> dict[str, Any]:
        response = self._request(url, params=params)
        if len(response.content) > max_bytes:
            raise ValueError("MOEX_RUNTIME_RESPONSE_TOO_LARGE")
        return cast("dict[str, Any]", response.json())

    def _request(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = (
                    self._client.get(
                        url,
                        params=params,
                        headers={"User-Agent": self._user_agent},
                        timeout=self._timeout,
                        follow_redirects=False,
                    )
                    if self._client is not None
                    else httpx.get(
                        url,
                        params=params,
                        headers={"User-Agent": self._user_agent},
                        timeout=self._timeout,
                        follow_redirects=False,
                    )
                )
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    time.sleep(min(0.25 * (attempt + 1), 1.0))
        assert last_error is not None
        raise last_error


def extract_stock_schedule_payload(content: str) -> dict[str, Any]:
    match = NEXT_DATA_PATTERN.search(content)
    if match is None:
        raise ValueError("MOEX_CALENDAR_NEXT_DATA_MISSING")
    page = cast("dict[str, Any]", json.loads(match.group(1)))
    page_props = cast("dict[str, Any]", page["props"])["pageProps"]
    init_data = _find_dict(
        page_props,
        lambda row: isinstance(row.get("offDays"), dict) and isinstance(row.get("holidays"), dict),
    )
    schedule_container = _find_dict(
        page_props,
        lambda row: isinstance(row.get("marketsSchedules"), list),
    )
    markets = cast("list[dict[str, Any]]", schedule_container["marketsSchedules"])
    stock = next(
        row
        for row in markets
        if row.get("market") == MARKET_SCOPE and row.get("id") == "stocks-market"
    )
    day_schedules = cast("list[dict[str, Any]]", stock["daysSchedules"])
    windows = {
        str(row["type"]): cast("list[dict[str, str]]", row["intervals"]) for row in day_schedules
    }
    off_days = cast("dict[str, Any]", init_data["offDays"])
    stock_years = cast("dict[str, dict[str, Any]]", off_days[MARKET_SCOPE])
    days = {day: row for year in stock_years.values() for day, row in year.items()}
    holiday_years = cast("dict[str, list[dict[str, str]]]", init_data["holidays"])
    holidays = {row["date"]: row["type"] for year in holiday_years.values() for row in year}
    return {
        "market": MARKET_SCOPE,
        "board": BOARD_SCOPE,
        "days": days,
        "holidays": holidays,
        "windows": windows,
    }


def _find_dict(
    value: object,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    queue = [value]
    while queue:
        item = queue.pop()
        if isinstance(item, dict):
            typed = cast("dict[str, Any]", item)
            if predicate(typed):
                return typed
            queue.extend(typed.values())
        elif isinstance(item, list):
            queue.extend(cast("list[Any]", item))
    raise KeyError("MOEX_CALENDAR_COMPONENT_MISSING")
