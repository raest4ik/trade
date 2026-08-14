from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import cast

import httpx

from src.tinvest_market.config import tbank_tls_context

_PRODUCTION_BASE_URL = "https://invest-public-api.tbank.ru/rest"
_SANDBOX_BASE_URL = "https://sandbox-invest-public-api.tbank.ru/rest"
_SERVICE_PREFIX = "/tinkoff.public.invest.api.contract.v1."
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class TInvestContour(StrEnum):
    READONLY_PRODUCTION = "READONLY_PRODUCTION"
    SANDBOX_READONLY_CONNECTIVITY = "SANDBOX_READONLY_CONNECTIVITY"


class TInvestClientError(RuntimeError):
    """Sanitized T-Invest client failure."""


class TInvestAuthError(TInvestClientError):
    pass


class TInvestRateLimitError(TInvestClientError):
    pass


class TInvestContractError(TInvestClientError):
    pass


@dataclass(frozen=True, slots=True)
class TInvestInstrument:
    ticker: str
    class_code: str
    instrument_uid: str
    figi: str | None
    instrument_type: str
    first_1day_candle_date: date | None
    name: str
    exchange: str | None = None
    currency: str | None = None
    real_exchange: str | None = None
    trading_status: str | None = None
    api_trade_available: bool | None = None
    buy_available: bool | None = None
    sell_available: bool | None = None
    last_1day_candle_date: date | None = None

    def payload(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "class_code": self.class_code,
            "instrument_uid": self.instrument_uid,
            "figi": self.figi,
            "instrument_type": self.instrument_type,
            "first_1day_candle_date": (
                self.first_1day_candle_date.isoformat()
                if self.first_1day_candle_date is not None
                else None
            ),
            "name": self.name,
            "exchange": self.exchange,
            "currency": self.currency,
            "real_exchange": self.real_exchange,
            "trading_status": self.trading_status,
            "api_trade_available_flag": self.api_trade_available,
            "buy_available_flag": self.buy_available,
            "sell_available_flag": self.sell_available,
            "last_1day_candle_date": (
                self.last_1day_candle_date.isoformat()
                if self.last_1day_candle_date is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class TInvestDailyCandle:
    instrument_uid: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    is_complete: bool


@dataclass(frozen=True, slots=True)
class TInvestCandleBatch:
    candles: tuple[TInvestDailyCandle, ...]
    rejected_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TInvestMinuteCandle:
    instrument_uid: str
    begin_at: datetime
    end_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    is_complete: bool


@dataclass(frozen=True, slots=True)
class TInvestMinuteCandleBatch:
    candles: tuple[TInvestMinuteCandle, ...]
    rejected_reasons: tuple[str, ...]


class TInvestReadOnlyClient:
    """Explicitly allowlisted read client with no account or order service surface."""

    def __init__(
        self,
        *,
        token: str,
        contour: TInvestContour,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
        sleep: bool = True,
    ) -> None:
        if not token.strip():
            raise ValueError("token must not be blank")
        self._token = token
        self._contour = contour
        self._base_url = (
            _PRODUCTION_BASE_URL
            if contour == TInvestContour.READONLY_PRODUCTION
            else _SANDBOX_BASE_URL
        )
        self._max_retries = max(0, max_retries)
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            verify=tbank_tls_context(),
        )

    @property
    def contour(self) -> TInvestContour:
        return self._contour

    async def __aenter__(self) -> TInvestReadOnlyClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def find_instruments(
        self, query: str, *, instrument_kind: str
    ) -> tuple[TInvestInstrument, ...]:
        normalized = query.strip()
        if not normalized or len(normalized) > 128:
            raise ValueError("instrument query is invalid")
        payload = await self._post_read(
            "InstrumentsService/FindInstrument",
            {"query": normalized, "instrumentKind": instrument_kind},
        )
        rows = _object_list(payload, "instruments")
        return tuple(_parse_instrument(item) for item in rows)

    async def list_indicatives(self) -> tuple[TInvestInstrument, ...]:
        payload = await self._post_read("InstrumentsService/Indicatives", {})
        rows = _object_list(payload, "instruments")
        return tuple(_parse_instrument(item) for item in rows)

    async def list_shares(self) -> tuple[TInvestInstrument, ...]:
        payload = await self._post_read(
            "InstrumentsService/Shares",
            {"instrumentStatus": "INSTRUMENT_STATUS_ALL"},
        )
        rows = _object_list(payload, "instruments")
        return tuple(_parse_instrument(item, default_kind="INSTRUMENT_TYPE_SHARE") for item in rows)

    async def get_instrument_by_uid(self, instrument_uid: str) -> TInvestInstrument:
        uid = _safe_identifier(instrument_uid)
        payload = await self._post_read(
            "InstrumentsService/GetInstrumentBy",
            {"idType": "INSTRUMENT_ID_TYPE_UID", "classCode": "", "id": uid},
        )
        instrument = payload.get("instrument")
        if not isinstance(instrument, dict):
            raise TInvestContractError("TINVEST_RESPONSE_INVALID")
        return _parse_instrument(cast("dict[str, object]", instrument))

    async def fetch_daily_candles(
        self,
        *,
        instrument_uid: str,
        date_from: date,
        date_to: date,
    ) -> tuple[TInvestDailyCandle, ...]:
        batch = await self.fetch_daily_candles_audited(
            instrument_uid=instrument_uid, date_from=date_from, date_to=date_to
        )
        if batch.rejected_reasons:
            raise TInvestContractError(batch.rejected_reasons[0])
        return batch.candles

    async def fetch_daily_candles_audited(
        self,
        *,
        instrument_uid: str,
        date_from: date,
        date_to: date,
    ) -> TInvestCandleBatch:
        uid = _safe_identifier(instrument_uid)
        if date_to < date_from:
            raise ValueError("date_to must not be before date_from")
        if (date_to - date_from).days > 2192:
            raise ValueError("daily candle request must not exceed six years")
        payload = await self._post_read(
            "MarketDataService/GetCandles",
            {
                "instrumentId": uid,
                "from": datetime.combine(date_from, datetime.min.time(), UTC).isoformat(),
                "to": datetime.combine(date_to, datetime.max.time(), UTC).isoformat(),
                "interval": "CANDLE_INTERVAL_DAY",
                "candleSourceType": "CANDLE_SOURCE_EXCHANGE",
            },
        )
        rows = _object_list(payload, "candles")
        candles: list[TInvestDailyCandle] = []
        rejected: list[str] = []
        for item in rows:
            try:
                candles.append(_parse_candle(item, instrument_uid=uid))
            except TInvestContractError as exc:
                rejected.append(str(exc))
        return TInvestCandleBatch(tuple(candles), tuple(rejected))

    async def fetch_minute_candles_audited(
        self,
        *,
        instrument_uid: str,
        date_from: datetime,
        date_to: datetime,
    ) -> TInvestMinuteCandleBatch:
        uid = _safe_identifier(instrument_uid)
        if date_from.tzinfo is None or date_to.tzinfo is None:
            raise ValueError("minute candle bounds must be timezone-aware")
        begin = date_from.astimezone(UTC)
        end = date_to.astimezone(UTC)
        if end <= begin:
            raise ValueError("date_to must be after date_from")
        if end - begin > timedelta(days=1):
            raise ValueError("minute candle request must not exceed one day")
        payload = await self._post_read(
            "MarketDataService/GetCandles",
            {
                "instrumentId": uid,
                "from": begin.isoformat(),
                "to": end.isoformat(),
                "interval": "CANDLE_INTERVAL_1_MIN",
                "candleSourceType": "CANDLE_SOURCE_EXCHANGE",
            },
        )
        rows = _object_list(payload, "candles")
        candles: list[TInvestMinuteCandle] = []
        rejected: list[str] = []
        for item in rows:
            try:
                candles.append(_parse_minute_candle(item, instrument_uid=uid))
            except TInvestContractError as exc:
                rejected.append(str(exc))
        return TInvestMinuteCandleBatch(tuple(candles), tuple(rejected))

    async def fetch_schedules(
        self, *, date_from: date, date_to: date, exchange: str = ""
    ) -> dict[str, object]:
        if date_to < date_from or (date_to - date_from).days > 7:
            raise ValueError("schedule request must cover at most seven days")
        return await self._post_read(
            "InstrumentsService/TradingSchedules",
            {
                "exchange": exchange,
                "from": datetime.combine(date_from, datetime.min.time(), UTC).isoformat(),
                "to": datetime.combine(date_to, datetime.max.time(), UTC).isoformat(),
            },
        )

    async def _post_read(self, method: str, body: dict[str, object]) -> dict[str, object]:
        allowed = {
            "InstrumentsService/FindInstrument",
            "InstrumentsService/GetInstrumentBy",
            "InstrumentsService/Indicatives",
            "InstrumentsService/Shares",
            "MarketDataService/GetCandles",
            "InstrumentsService/TradingSchedules",
        }
        if method not in allowed:
            raise TInvestClientError("read method is not allowlisted")
        url = f"{self._base_url}{_SERVICE_PREFIX}{method}"
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    url,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type": "application/json",
                        "User-Agent": "trade-ai-news-mvp/0.1",
                    },
                )
            except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError) as exc:
                if attempt >= self._max_retries:
                    raise TInvestClientError("TINVEST_API_UNAVAILABLE") from exc
                await self._backoff(attempt)
                continue
            if response.status_code in {401, 403}:
                raise TInvestAuthError("TINVEST_AUTH_FAILED")
            if response.status_code == 429:
                if attempt >= self._max_retries:
                    raise TInvestRateLimitError("TINVEST_RATE_LIMITED")
                await self._backoff(attempt, response.headers.get("Retry-After"))
                continue
            if response.status_code >= 500:
                if attempt >= self._max_retries:
                    raise TInvestClientError("TINVEST_API_UNAVAILABLE")
                await self._backoff(attempt)
                continue
            if response.status_code >= 400:
                raise TInvestClientError(f"TINVEST_REQUEST_REJECTED_{response.status_code}")
            try:
                raw: object = response.json()
            except ValueError as exc:
                raise TInvestContractError("TINVEST_RESPONSE_INVALID") from exc
            if not isinstance(raw, dict):
                raise TInvestContractError("TINVEST_RESPONSE_INVALID")
            return cast("dict[str, object]", raw)
        raise TInvestClientError("TINVEST_API_UNAVAILABLE")

    async def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        delay = min(0.5 * (2**attempt), 8.0)
        if retry_after is not None:
            try:
                delay = min(max(float(retry_after), 0.0), 30.0)
            except ValueError:
                pass
        if self._sleep:
            await asyncio.sleep(delay)


def _parse_instrument(
    payload: dict[str, object], *, default_kind: str = "UNKNOWN"
) -> TInvestInstrument:
    ticker = _required_string(payload, "ticker").upper()
    uid = _required_string(payload, "uid")
    return TInvestInstrument(
        ticker=ticker,
        class_code=_optional_string(payload.get("classCode")) or "",
        instrument_uid=_safe_identifier(uid),
        figi=_optional_string(payload.get("figi")),
        instrument_type=(
            _optional_string(payload.get("instrumentType"))
            or _optional_string(payload.get("instrumentKind"))
            or default_kind
        ),
        first_1day_candle_date=_optional_datetime_date(payload.get("first1dayCandleDate")),
        name=_optional_string(payload.get("name")) or ticker,
        exchange=_optional_string(payload.get("exchange")),
        currency=_optional_string(payload.get("currency")),
        real_exchange=_optional_scalar(payload.get("realExchange")),
        trading_status=_optional_scalar(payload.get("tradingStatus")),
        api_trade_available=_optional_bool(payload.get("apiTradeAvailableFlag")),
        buy_available=_optional_bool(payload.get("buyAvailableFlag")),
        sell_available=_optional_bool(payload.get("sellAvailableFlag")),
        last_1day_candle_date=_optional_datetime_date(payload.get("last1dayCandleDate")),
    )


def _parse_candle(payload: dict[str, object], *, instrument_uid: str) -> TInvestDailyCandle:
    open_price = _quotation(payload.get("open"))
    high = _quotation(payload.get("high"))
    low = _quotation(payload.get("low"))
    close = _quotation(payload.get("close"))
    volume = payload.get("volume")
    timestamp = payload.get("time")
    if min(open_price, high, low, close) <= 0:
        raise TInvestContractError("TINVEST_CANDLE_INVALID_PRICE")
    if low > high or not low <= open_price <= high or not low <= close <= high:
        raise TInvestContractError("TINVEST_CANDLE_INVALID_OHLC")
    if isinstance(volume, bool) or not isinstance(volume, (int, str)):
        raise TInvestContractError("TINVEST_CANDLE_INVALID_VOLUME")
    try:
        parsed_volume = int(volume)
    except ValueError as exc:
        raise TInvestContractError("TINVEST_CANDLE_INVALID_VOLUME") from exc
    if parsed_volume < 0 or not isinstance(timestamp, str):
        raise TInvestContractError("TINVEST_CANDLE_INVALID")
    return TInvestDailyCandle(
        instrument_uid=instrument_uid,
        trade_date=datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date(),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=parsed_volume,
        is_complete=bool(payload.get("isComplete", False)),
    )


def _parse_minute_candle(payload: dict[str, object], *, instrument_uid: str) -> TInvestMinuteCandle:
    daily = _parse_candle(payload, instrument_uid=instrument_uid)
    timestamp = payload.get("time")
    if not isinstance(timestamp, str):
        raise TInvestContractError("TINVEST_CANDLE_INVALID")
    try:
        begin_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as exc:
        raise TInvestContractError("TINVEST_CANDLE_INVALID") from exc
    if begin_at.second or begin_at.microsecond:
        raise TInvestContractError("TINVEST_MINUTE_CANDLE_NOT_ALIGNED")
    return TInvestMinuteCandle(
        instrument_uid=instrument_uid,
        begin_at=begin_at,
        end_at=begin_at + timedelta(minutes=1),
        open=daily.open,
        high=daily.high,
        low=daily.low,
        close=daily.close,
        volume=daily.volume,
        is_complete=daily.is_complete,
    )


def _quotation(value: object) -> Decimal:
    if not isinstance(value, dict):
        raise TInvestContractError("TINVEST_QUOTATION_INVALID")
    typed = cast("dict[str, object]", value)
    units = typed.get("units", 0)
    nano = typed.get("nano", 0)
    try:
        return Decimal(str(units)) + Decimal(str(nano)) / Decimal("1000000000")
    except (InvalidOperation, TypeError) as exc:
        raise TInvestContractError("TINVEST_QUOTATION_INVALID") from exc


def _object_list(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise TInvestContractError("TINVEST_RESPONSE_INVALID")
    result: list[dict[str, object]] = []
    for item in cast("list[object]", value):
        if not isinstance(item, dict):
            raise TInvestContractError("TINVEST_RESPONSE_INVALID")
        result.append(cast("dict[str, object]", item))
    return result


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TInvestContractError("TINVEST_RESPONSE_INVALID")
    return value.strip()


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_scalar(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (str, int)):
        normalized = str(value).strip()
        return normalized or None
    return None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_datetime_date(value: object) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _safe_identifier(value: str) -> str:
    normalized = value.strip()
    if not _SAFE_IDENTIFIER.fullmatch(normalized):
        raise ValueError("instrument identifier is invalid")
    return normalized
