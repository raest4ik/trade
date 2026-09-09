from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, cast
from zoneinfo import ZoneInfo

import httpx

from src.free_live_issuer_accumulation.domain import sha256_payload
from src.production_readonly_adapters_v1.domain import (
    MARKET_ADAPTER_ID,
    MARKET_SOURCE,
    FreshMarketSnapshot,
    MarketQuoteAudit,
    MarketQuoteStatus,
)

MAX_RESPONSE_BYTES = 1_000_000
MOEX_TIMEZONE = ZoneInfo("Europe/Moscow")


class FreshMarketAdapterError(RuntimeError):
    pass


class MarketTimestampUnavailableError(FreshMarketAdapterError):
    pass


class MarketResponseInvalidError(FreshMarketAdapterError):
    pass


class MarketQuoteMissingError(FreshMarketAdapterError):
    pass


class MoexIssFreshMarketAdapter:
    adapter_id = MARKET_ADAPTER_ID
    source = MARKET_SOURCE

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        user_agent: str,
        max_age_seconds: float,
        http_client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        parsed = httpx.URL(base_url.rstrip("/"))
        if parsed.scheme != "https" or parsed.host != "iss.moex.com":
            raise ValueError("MOEX ISS base URL is not allowed")
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        self._base_url = str(parsed).rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._user_agent = user_agent
        self._max_age_seconds = max_age_seconds
        self._client = http_client
        self._clock = clock or (lambda: datetime.now(UTC))

    def fetch(
        self,
        *,
        universe: Sequence[dict[str, Any]],
        operation_as_of: datetime,
    ) -> FreshMarketSnapshot:
        if operation_as_of.tzinfo is None or operation_as_of.utcoffset() is None:
            raise ValueError("operation_as_of must be timezone-aware")
        tickers = [str(row.get("ticker", "")).strip().upper() for row in universe]
        if not all(tickers) or len(set(tickers)) != len(tickers):
            raise MarketResponseInvalidError("MISSING_OR_DUPLICATE_TICKER")
        started = self._clock()
        quote_rows: list[dict[str, Any]] = []
        audits: list[MarketQuoteAudit] = []
        payloads: list[dict[str, Any]] = []
        for instrument in universe:
            ticker = str(instrument["ticker"]).upper()
            board = str(instrument.get("board") or "").upper()
            if instrument.get("supported") is not True:
                audits.append(_invalid(ticker, board, "UNSUPPORTED_INSTRUMENT"))
                continue
            if board != "TQBR":
                audits.append(_invalid(ticker, board, "WRONG_BOARD"))
                continue
            try:
                payload = self._request(ticker, board)
                fetched_at = self._clock()
                payloads.append({"ticker": ticker, "payload": payload})
                quote, audit = _parse_quote(
                    payload,
                    expected_ticker=ticker,
                    expected_board=board,
                    operation_as_of=operation_as_of.astimezone(UTC),
                    max_age_seconds=self._max_age_seconds,
                    fetched_at=fetched_at,
                )
                quote_rows.append(quote)
                audits.append(audit)
            except MarketTimestampUnavailableError as exc:
                audits.append(_invalid(ticker, board, str(exc), status=MarketQuoteStatus.MISSING))
            except MarketQuoteMissingError as exc:
                audits.append(_invalid(ticker, board, str(exc), status=MarketQuoteStatus.MISSING))
            except (
                KeyError,
                TypeError,
                ValueError,
                InvalidOperation,
                MarketResponseInvalidError,
            ) as exc:
                audits.append(_invalid(ticker, board, str(exc)))
        completed = self._clock()
        return FreshMarketSnapshot(
            market_fetch_started_at=started,
            market_fetch_completed_at=completed,
            operation_as_of=operation_as_of,
            effective_max_age_seconds=self._max_age_seconds,
            quotes=quote_rows,
            quote_audit=audits,
            source_payload_sha=sha256_payload(payloads),
        )

    def _request(self, ticker: str, board: str) -> dict[str, Any]:
        url = (
            f"{self._base_url}/engines/stock/markets/shares/boards/{board}/securities/{ticker}.json"
        )
        params = {
            "iss.meta": "off",
            "iss.only": "securities,marketdata",
            "securities.columns": "SECID,BOARDID,LOTSIZE",
            "marketdata.columns": "SECID,BOARDID,LAST,BID,OFFER,SYSTIME",
        }
        for attempt in range(self._max_retries + 1):
            try:
                response = self._get(url, params)
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                if attempt >= self._max_retries:
                    raise FreshMarketAdapterError("MOEX_REQUEST_FAILED") from exc
                continue
            if response.status_code in {429} or response.status_code >= 500:
                if attempt < self._max_retries:
                    continue
                raise FreshMarketAdapterError("MOEX_REQUEST_FAILED")
            if response.status_code >= 400:
                raise FreshMarketAdapterError(f"MOEX_HTTP_{response.status_code}")
            if len(response.content) > MAX_RESPONSE_BYTES:
                raise MarketResponseInvalidError("MOEX_RESPONSE_TOO_LARGE")
            try:
                payload: object = response.json()
            except ValueError as exc:
                raise MarketResponseInvalidError("MOEX_INVALID_JSON") from exc
            if not isinstance(payload, dict):
                raise MarketResponseInvalidError("MOEX_INVALID_ROOT")
            return cast("dict[str, Any]", payload)
        raise FreshMarketAdapterError("MOEX_REQUEST_FAILED")

    def _get(self, url: str, params: dict[str, str]) -> httpx.Response:
        if self._client is not None:
            return self._client.get(
                url,
                params=params,
                headers={"User-Agent": self._user_agent},
                timeout=self._timeout,
            )
        with httpx.Client(
            timeout=self._timeout,
            headers={"User-Agent": self._user_agent},
        ) as client:
            return client.get(url, params=params)


def _parse_quote(
    payload: dict[str, Any],
    *,
    expected_ticker: str,
    expected_board: str,
    operation_as_of: datetime,
    max_age_seconds: float,
    fetched_at: datetime,
) -> tuple[dict[str, Any], MarketQuoteAudit]:
    security = _one_row(payload, "securities")
    market = _one_row(payload, "marketdata")
    ticker = _required_string(market.get("SECID"))
    board = _required_string(market.get("BOARDID"))
    if ticker != expected_ticker or _required_string(security.get("SECID")) != expected_ticker:
        raise MarketResponseInvalidError("TICKER_MAPPING_MISMATCH")
    if board != expected_board or _required_string(security.get("BOARDID")) != expected_board:
        raise MarketResponseInvalidError("WRONG_BOARD")
    lot_size = _positive_int(security.get("LOTSIZE"), "MISSING_OR_INVALID_LOT_SIZE")
    last = _positive_decimal(market.get("LAST"), "INVALID_LAST_PRICE")
    bid = _optional_positive_decimal(market.get("BID"), "INVALID_BID")
    ask = _optional_positive_decimal(market.get("OFFER"), "INVALID_ASK")
    if bid is not None and ask is not None and bid > ask:
        raise MarketResponseInvalidError("BID_ABOVE_ASK")
    source_time = market.get("SYSTIME")
    if not isinstance(source_time, str) or not source_time.strip():
        raise MarketTimestampUnavailableError("MARKET_TIMESTAMP_UNAVAILABLE")
    market_as_of = _source_timestamp(source_time)
    age = (operation_as_of - market_as_of).total_seconds()
    status = (
        MarketQuoteStatus.FUTURE
        if age < 0
        else MarketQuoteStatus.STALE
        if age > max_age_seconds
        else MarketQuoteStatus.FRESH
    )
    payload_sha = sha256_payload({"securities": security, "marketdata": market})
    quote = {
        "ticker": ticker,
        "board": board,
        "last_price": float(last),
        "bid": None if bid is None else float(bid),
        "ask": None if ask is None else float(ask),
        "lot_size": lot_size,
        "market_data_as_of": market_as_of.isoformat(),
        "source": MARKET_SOURCE,
        "fetched_at": fetched_at.isoformat(),
        "source_payload_sha": payload_sha,
        "status": status.value,
        "age_seconds": age,
    }
    return quote, MarketQuoteAudit(
        ticker=ticker,
        board=board,
        market_data_as_of=market_as_of,
        age_seconds=age,
        status=status,
        payload_sha=payload_sha,
    )


def _one_row(payload: dict[str, Any], key: str) -> dict[str, Any]:
    raw_table = payload.get(key)
    if not isinstance(raw_table, dict):
        raise MarketResponseInvalidError(f"MISSING_{key.upper()}_TABLE")
    table = cast("dict[str, Any]", raw_table)
    columns_obj: object = table.get("columns")
    rows_obj: object = table.get("data")
    if not isinstance(columns_obj, list):
        raise MarketResponseInvalidError(f"INVALID_{key.upper()}_COLUMNS")
    raw_columns = cast("list[object]", columns_obj)
    if not all(isinstance(item, str) for item in raw_columns):
        raise MarketResponseInvalidError(f"INVALID_{key.upper()}_COLUMNS")
    if not isinstance(rows_obj, list):
        raise MarketResponseInvalidError(f"MISSING_OR_DUPLICATE_{key.upper()}_ROW")
    raw_rows = cast("list[object]", rows_obj)
    if not raw_rows:
        raise MarketQuoteMissingError(f"MISSING_{key.upper()}_ROW")
    if len(raw_rows) != 1 or not isinstance(raw_rows[0], list):
        raise MarketResponseInvalidError(f"MISSING_OR_DUPLICATE_{key.upper()}_ROW")
    columns = cast("list[str]", raw_columns)
    values = cast("list[Any]", raw_rows[0])
    if len(values) != len(columns):
        raise MarketResponseInvalidError(f"INVALID_{key.upper()}_ROW")
    return dict(zip(columns, values, strict=True))


def _source_timestamp(value: str) -> datetime:
    normalized = value.strip().replace(" ", "T")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=MOEX_TIMEZONE)
    return parsed.astimezone(UTC)


def _required_string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarketResponseInvalidError("MISSING_STRING_FIELD")
    return value.strip().upper()


def _positive_decimal(value: object, reason: str) -> Decimal:
    if value is None or isinstance(value, bool):
        raise MarketResponseInvalidError(reason)
    result = Decimal(str(value))
    if not result.is_finite() or result <= 0:
        raise MarketResponseInvalidError(reason)
    return result


def _optional_positive_decimal(value: object, reason: str) -> Decimal | None:
    return None if value is None else _positive_decimal(value, reason)


def _positive_int(value: object, reason: str) -> int:
    if value is None or isinstance(value, bool):
        raise MarketResponseInvalidError(reason)
    try:
        result = int(cast("int | str", value))
    except (TypeError, ValueError) as exc:
        raise MarketResponseInvalidError(reason) from exc
    if result <= 0:
        raise MarketResponseInvalidError(reason)
    return result


def _invalid(
    ticker: str,
    board: str,
    reason: str,
    *,
    status: MarketQuoteStatus = MarketQuoteStatus.INVALID,
) -> MarketQuoteAudit:
    return MarketQuoteAudit(ticker=ticker, board=board, status=status, reason=reason)
