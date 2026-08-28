from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, cast

import httpx

from src.chep_security_history_diagnostics.domain import ProbeWindow

MOEX_BASE_URL = "https://iss.moex.com/iss"


class MoexIssClient:
    """Zero-cost official MOEX ISS diagnostics client."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))

    async def __aenter__(self) -> MoexIssClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def security_card(self, secid: str) -> dict[str, object]:
        url = f"{MOEX_BASE_URL}/securities/{secid}.json"
        return await self._get(url, {"iss.meta": "off", "iss.only": "description,boards"})

    async def history(self, secid: str, *, date_from: date, date_to: date) -> dict[str, object]:
        url = f"{MOEX_BASE_URL}/history/engines/stock/markets/shares/securities/{secid}.json"
        return await self._get(
            url,
            {
                "iss.meta": "off",
                "from": date_from.isoformat(),
                "till": date_to.isoformat(),
            },
        )

    async def candles(
        self,
        secid: str,
        *,
        board: str,
        date_from: datetime,
        date_to: datetime,
        interval: int,
    ) -> dict[str, object]:
        url = (
            f"{MOEX_BASE_URL}/engines/stock/markets/shares/boards/"
            f"{board}/securities/{secid}/candles.json"
        )
        return await self._get(
            url,
            {
                "iss.meta": "off",
                "from": _moex_time(date_from),
                "till": _moex_time(date_to),
                "interval": str(interval),
            },
        )

    async def _get(self, url: str, params: dict[str, str]) -> dict[str, object]:
        response = await self._client.get(url, params=params)
        if response.status_code >= 400:
            return {
                "url": str(response.url),
                "status": f"HTTP_{response.status_code}",
                "rows": [],
                "error": "MOEX_REQUEST_FAILED",
            }
        raw: object = response.json()
        if not isinstance(raw, dict):
            return {
                "url": str(response.url),
                "status": "INVALID_JSON",
                "rows": [],
                "error": "MOEX_RESPONSE_INVALID",
            }
        return {"url": str(response.url), "status": "PASS", "payload": raw}


async def run_moex_cross_check(
    *,
    client: MoexIssClient,
    secid: str,
    board: str,
    windows: tuple[ProbeWindow, ...],
    last_known_tinvest_daily: date | None,
) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    card = await client.security_card(secid)
    card_rows = _table_rows(card, "description") + _table_rows(card, "boards")
    requests.append(
        {
            "source": "MOEX_ISS_OFFICIAL",
            "label": "security_card",
            "url": card.get("url"),
            "api_status": card.get("status"),
            "returned_row_count": len(card_rows),
            "first_returned_timestamp": None,
            "last_returned_timestamp": None,
            "api_error": card.get("error"),
        }
    )
    event_history_rows = 0
    event_minute_rows = 0
    for window in windows:
        history = await client.history(
            secid, date_from=window.publication_timestamp_utc.date(), date_to=window.daily_to
        )
        history_rows = _table_rows(history, "history")
        event_history_rows += len(history_rows)
        requests.append(
            {
                "source": "MOEX_ISS_OFFICIAL",
                "label": f"{window.label}_history",
                "url": history.get("url"),
                "api_status": history.get("status"),
                "interval": "1d",
                "from": window.publication_timestamp_utc.date().isoformat(),
                "to": window.daily_to.isoformat(),
                "returned_row_count": len(history_rows),
                "first_returned_timestamp": _min_value(history_rows, "TRADEDATE"),
                "last_returned_timestamp": _max_value(history_rows, "TRADEDATE"),
                "api_error": history.get("error"),
            }
        )
        candles = await client.candles(
            secid,
            board=board,
            date_from=window.minute_from,
            date_to=window.minute_to,
            interval=1,
        )
        candle_rows = _table_rows(candles, "candles")
        event_minute_rows += len(candle_rows)
        requests.append(
            {
                "source": "MOEX_ISS_OFFICIAL",
                "label": f"{window.label}_minute_candles",
                "url": candles.get("url"),
                "api_status": candles.get("status"),
                "interval": "1m",
                "from": window.minute_from.isoformat(),
                "to": window.minute_to.isoformat(),
                "returned_row_count": len(candle_rows),
                "first_returned_timestamp": _min_value(candle_rows, "begin"),
                "last_returned_timestamp": _max_value(candle_rows, "end"),
                "api_error": candles.get("error"),
            }
        )
    known_history_rows = 0
    if last_known_tinvest_daily is not None:
        history = await client.history(
            secid, date_from=last_known_tinvest_daily, date_to=last_known_tinvest_daily
        )
        known_rows = _table_rows(history, "history")
        known_history_rows = len(known_rows)
        requests.append(
            {
                "source": "MOEX_ISS_OFFICIAL",
                "label": "last_known_tinvest_daily_history",
                "url": history.get("url"),
                "api_status": history.get("status"),
                "interval": "1d",
                "from": last_known_tinvest_daily.isoformat(),
                "to": last_known_tinvest_daily.isoformat(),
                "returned_row_count": len(known_rows),
                "first_returned_timestamp": _min_value(known_rows, "TRADEDATE"),
                "last_returned_timestamp": _max_value(known_rows, "TRADEDATE"),
                "api_error": history.get("error"),
            }
        )
    return {
        "MOEX_SECURITY_HISTORY_CONFIRMED": bool(card_rows or known_history_rows),
        "MOEX_EVENT_DATE_TRADING_CONFIRMED": bool(event_history_rows),
        "MOEX_MINUTE_HISTORY_EVIDENCE": bool(event_minute_rows),
        "MOEX_REQUESTS": requests,
        "MOEX_EVENT_HISTORY_ROWS": event_history_rows,
        "MOEX_EVENT_MINUTE_ROWS": event_minute_rows,
        "MOEX_LAST_KNOWN_HISTORY_ROWS": known_history_rows,
        "MOEX_PROVENANCE": "MOEX_ISS_PUBLIC_ZERO_COST_DIAGNOSTIC_ONLY",
    }


def _table_rows(response: dict[str, object], table: str) -> list[dict[str, object]]:
    payload = response.get("payload")
    if not isinstance(payload, dict):
        return []
    typed_payload = cast("dict[str, object]", payload)
    value = typed_payload.get(table)
    if not isinstance(value, dict):
        return []
    typed_table = cast("dict[str, object]", value)
    columns = typed_table.get("columns")
    data = typed_table.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        return []
    result: list[dict[str, object]] = []
    typed_columns = cast("list[object]", columns)
    typed_data = cast("list[object]", data)
    for raw_row in typed_data:
        if not isinstance(raw_row, list):
            continue
        typed_raw_row = cast("list[object]", raw_row)
        row: dict[str, object] = {}
        for index, column in enumerate(typed_columns):
            row[str(column)] = typed_raw_row[index] if index < len(typed_raw_row) else None
        result.append(row)
    return result


def _min_value(rows: list[dict[str, object]], key: str) -> str | None:
    return min((str(row[key]) for row in rows if row.get(key) is not None), default=None)


def _max_value(rows: list[dict[str, object]], key: str) -> str | None:
    return max((str(row[key]) for row in rows if row.get(key) is not None), default=None)


def _moex_time(value: datetime) -> str:
    return value.astimezone(UTC).replace(tzinfo=None).isoformat(timespec="seconds")
