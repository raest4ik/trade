from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from time import monotonic
from typing import Any, cast
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import httpx

from src.market_data.application.exceptions import (
    BenchmarkDataPartialProviderError,
    MarketDataPartialProviderError,
    MarketDataProviderContractError,
    MarketDataProviderUnavailableError,
    MarketDataValidationError,
)
from src.market_data.domain.entities import (
    MOEX_ADAPTER_VERSION,
    MOEX_ENGINE_STOCK,
    MOEX_MARKET_SHARES,
    MOEX_SOURCE_TIMEZONE,
    SUPPORTED_INTERVAL_MINUTES,
    BenchmarkCandle,
    MarketBenchmark,
    MarketCandle,
    RejectedCandleRow,
)

logger = logging.getLogger(__name__)

_MARKET_CODE_PATTERN = re.compile(r"^[A-Z0-9_-]{1,32}$")
_REQUIRED_COLUMNS = ("open", "close", "high", "low", "value", "volume", "begin", "end")
_MAX_REQUEST_DAYS = 31
_PAGE_SIZE_FALLBACK = 500
_DAILY_INTERVAL_MINUTES = 24


@dataclass(frozen=True, slots=True)
class MoexFetchResult:
    candles: list[MarketCandle]
    pages_received: int
    rows_received: int
    rows_valid: int
    rows_rejected: int
    rejected_rows: list[RejectedCandleRow]


@dataclass(frozen=True, slots=True)
class MoexBenchmarkFetchResult:
    candles: list[BenchmarkCandle]
    pages_received: int
    rows_received: int
    rows_valid: int
    rows_rejected: int
    rejected_rows: list[RejectedCandleRow]


@dataclass(frozen=True, slots=True)
class MoexDailyCandle:
    security_code: str
    board: str
    trade_date: date
    open: Decimal
    close: Decimal
    high: Decimal
    low: Decimal
    value: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class MoexDailyFetchResult:
    candles: list[MoexDailyCandle]
    pages_received: int
    rows_received: int
    rows_valid: int
    rows_rejected: int
    rejected_rows: list[RejectedCandleRow]


class MoexIssClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        max_pages: int,
        user_agent: str,
        client: httpx.AsyncClient | None = None,
        sleep: bool = True,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        self._timeout_seconds = timeout_seconds
        self._max_retries = max(0, max_retries)
        self._max_pages = max_pages
        self._user_agent = user_agent
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": user_agent},
        )
        self._sleep = sleep

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> MoexIssClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    async def fetch_daily_candles(
        self,
        *,
        security_code: str,
        engine: str,
        market: str,
        board: str,
        date_from: date,
        date_till: date,
    ) -> MoexDailyFetchResult:
        """Fetch source daily candles without weakening the minute-candle contract."""
        code = validate_market_path_value(security_code, "security_code")
        board = validate_market_path_value(board, "board")
        engine = validate_market_path_value(engine, "engine").lower()
        market = validate_market_path_value(market, "market").lower()
        if date_till < date_from:
            raise MarketDataValidationError("date_till must not be before date_from")

        candles: list[MoexDailyCandle] = []
        rejected_rows: list[RejectedCandleRow] = []
        rows_received = 0
        start = 0
        pages_received = 0
        for page in range(self._max_pages):
            payload = await self._request_page(
                security_code=code,
                engine=engine,
                market=market,
                board=board,
                date_from=date_from,
                date_till=date_till,
                interval_minutes=_DAILY_INTERVAL_MINUTES,
                start=start,
                page=page,
            )
            columns, rows = _candle_columns_and_rows(payload)
            column_index = {column: index for index, column in enumerate(columns)}
            pages_received += 1
            rows_received += len(rows)
            for row_index, row in enumerate(rows):
                if not isinstance(row, list):
                    rejected_rows.append(
                        RejectedCandleRow(page=page, row_index=row_index, reason="row_not_array")
                    )
                    continue
                try:
                    candles.append(
                        _row_to_daily_candle(
                            cast("list[object]", row),
                            column_index,
                            security_code=code,
                            board=board,
                        )
                    )
                except (InvalidOperation, TypeError, ValueError) as exc:
                    rejected_rows.append(
                        RejectedCandleRow(page=page, row_index=row_index, reason=str(exc))
                    )
            if not rows or len(rows) < _PAGE_SIZE_FALLBACK:
                break
            start += len(rows)
        else:
            raise MarketDataProviderContractError(
                "MOEX daily pagination exceeded configured max pages"
            )
        return MoexDailyFetchResult(
            candles=candles,
            pages_received=pages_received,
            rows_received=rows_received,
            rows_valid=len(candles),
            rows_rejected=len(rejected_rows),
            rejected_rows=rejected_rows,
        )

    async def fetch_candles(
        self,
        *,
        instrument_id: UUID,
        ticker: str,
        board: str,
        date_from: date,
        date_till: date,
        interval_minutes: int,
    ) -> tuple[list[MarketCandle], int, int, int, int]:
        result = await self.fetch_candles_with_rejections(
            instrument_id=instrument_id,
            ticker=ticker,
            board=board,
            date_from=date_from,
            date_till=date_till,
            interval_minutes=interval_minutes,
        )
        return (
            result.candles,
            result.pages_received,
            result.rows_received,
            result.rows_valid,
            result.rows_rejected,
        )

    async def fetch_candles_with_rejections(
        self,
        *,
        instrument_id: UUID,
        ticker: str,
        board: str,
        date_from: date,
        date_till: date,
        interval_minutes: int,
    ) -> MoexFetchResult:
        ticker = validate_market_path_value(ticker, "ticker")
        board = validate_market_path_value(board, "board")
        validate_date_range(date_from, date_till)
        if interval_minutes != SUPPORTED_INTERVAL_MINUTES:
            raise MarketDataValidationError("only interval_minutes=1 is supported")

        candles: list[MarketCandle] = []
        rejected_rows: list[RejectedCandleRow] = []
        rows_received = 0
        rows_valid = 0
        rows_rejected = 0
        start = 0
        pages_received = 0

        for page in range(self._max_pages):
            try:
                payload = await self._request_page(
                    security_code=ticker,
                    engine=MOEX_ENGINE_STOCK,
                    market=MOEX_MARKET_SHARES,
                    board=board,
                    date_from=date_from,
                    date_till=date_till,
                    interval_minutes=interval_minutes,
                    start=start,
                    page=page,
                )
            except MarketDataProviderUnavailableError as exc:
                if pages_received == 0:
                    raise
                raise MarketDataPartialProviderError(
                    "MOEX failed after returning partial candle data",
                    candles=candles,
                    pages_received=pages_received,
                    rows_received=rows_received,
                    rows_valid=rows_valid,
                    rows_rejected=rows_rejected,
                ) from exc
            parsed = _parse_page(
                payload,
                instrument_id=instrument_id,
                ticker=ticker,
                board=board,
                interval_minutes=interval_minutes,
                page=page,
            )
            pages_received += 1
            rows_received += parsed.rows_received
            rows_valid += len(parsed.candles)
            rows_rejected += len(parsed.rejected_rows)
            candles.extend(parsed.candles)
            rejected_rows.extend(parsed.rejected_rows)
            if parsed.rows_received == 0:
                break
            start += parsed.rows_received
            if parsed.rows_received < _PAGE_SIZE_FALLBACK:
                break
        else:
            raise MarketDataProviderContractError("MOEX pagination exceeded configured max pages")

        return MoexFetchResult(
            candles=candles,
            pages_received=pages_received,
            rows_received=rows_received,
            rows_valid=rows_valid,
            rows_rejected=rows_rejected,
            rejected_rows=rejected_rows,
        )

    async def fetch_benchmark_candles(
        self,
        *,
        benchmark: MarketBenchmark,
        date_from: date,
        date_till: date,
        interval_minutes: int,
    ) -> tuple[list[BenchmarkCandle], int, int, int, int]:
        result = await self.fetch_benchmark_candles_with_rejections(
            benchmark=benchmark,
            date_from=date_from,
            date_till=date_till,
            interval_minutes=interval_minutes,
        )
        return (
            result.candles,
            result.pages_received,
            result.rows_received,
            result.rows_valid,
            result.rows_rejected,
        )

    async def fetch_benchmark_candles_with_rejections(
        self,
        *,
        benchmark: MarketBenchmark,
        date_from: date,
        date_till: date,
        interval_minutes: int,
    ) -> MoexBenchmarkFetchResult:
        code = validate_market_path_value(benchmark.code, "benchmark_code")
        board = validate_market_path_value(benchmark.board, "board")
        engine = validate_market_path_value(benchmark.engine, "engine").lower()
        market = validate_market_path_value(benchmark.market, "market").lower()
        validate_date_range(date_from, date_till)
        if interval_minutes != SUPPORTED_INTERVAL_MINUTES:
            raise MarketDataValidationError("only interval_minutes=1 is supported")

        candles: list[BenchmarkCandle] = []
        rejected_rows: list[RejectedCandleRow] = []
        rows_received = 0
        rows_valid = 0
        rows_rejected = 0
        start = 0
        pages_received = 0
        for page in range(self._max_pages):
            try:
                payload = await self._request_page(
                    security_code=code,
                    engine=engine,
                    market=market,
                    board=board,
                    date_from=date_from,
                    date_till=date_till,
                    interval_minutes=interval_minutes,
                    start=start,
                    page=page,
                )
            except MarketDataProviderUnavailableError as exc:
                if pages_received == 0:
                    raise
                raise BenchmarkDataPartialProviderError(
                    "MOEX failed after returning partial benchmark candle data",
                    candles=candles,
                    pages_received=pages_received,
                    rows_received=rows_received,
                    rows_valid=rows_valid,
                    rows_rejected=rows_rejected,
                ) from exc
            parsed = _parse_benchmark_page(
                payload,
                benchmark_id=benchmark.id,
                interval_minutes=interval_minutes,
                page=page,
            )
            pages_received += 1
            rows_received += parsed.rows_received
            rows_valid += len(parsed.candles)
            rows_rejected += len(parsed.rejected_rows)
            candles.extend(parsed.candles)
            rejected_rows.extend(parsed.rejected_rows)
            if parsed.rows_received == 0:
                break
            start += parsed.rows_received
            if parsed.rows_received < _PAGE_SIZE_FALLBACK:
                break
        else:
            raise MarketDataProviderContractError("MOEX pagination exceeded configured max pages")
        return MoexBenchmarkFetchResult(
            candles=candles,
            pages_received=pages_received,
            rows_received=rows_received,
            rows_valid=rows_valid,
            rows_rejected=rows_rejected,
            rejected_rows=rejected_rows,
        )

    async def _request_page(
        self,
        *,
        security_code: str,
        engine: str,
        market: str,
        board: str,
        date_from: date,
        date_till: date,
        interval_minutes: int,
        start: int,
        page: int,
    ) -> dict[str, object]:
        request_id = str(uuid4())
        url = (
            f"{self._base_url}/engines/{engine}/markets/{market}/boards/{board}"
            f"/securities/{security_code}/candles.json"
        )
        params = {
            "interval": str(interval_minutes),
            "from": date_from.isoformat(),
            "till": date_till.isoformat(),
            "start": str(start),
            "iss.meta": "off",
            "iss.only": "candles",
        }
        for attempt in range(self._max_retries + 1):
            started = monotonic()
            try:
                response = await self._client.get(
                    url,
                    params=params,
                    headers={"User-Agent": self._user_agent},
                )
            except httpx.TimeoutException as exc:
                await self._backoff_or_raise(exc, attempt, None)
                continue
            except httpx.HTTPError as exc:
                raise MarketDataProviderUnavailableError("MOEX request failed") from exc
            duration_ms = int((monotonic() - started) * 1000)
            logger.info(
                "moex_iss_request",
                extra={
                    "request_id": request_id,
                    "security_code": security_code,
                    "board": board,
                    "date_from": date_from.isoformat(),
                    "date_till": date_till.isoformat(),
                    "page": page,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
            if response.status_code == 429 or 500 <= response.status_code < 600:
                await self._backoff_or_raise(
                    MarketDataProviderUnavailableError("MOEX temporary failure"),
                    attempt,
                    response,
                )
                continue
            if 400 <= response.status_code < 500:
                raise MarketDataValidationError("MOEX rejected request parameters")
            try:
                raw_payload: object = response.json()
            except ValueError as exc:
                raise MarketDataProviderContractError("MOEX returned malformed JSON") from exc
            if not isinstance(raw_payload, dict):
                raise MarketDataProviderContractError("MOEX JSON root must be an object")
            return cast("dict[str, Any]", raw_payload)
        raise MarketDataProviderUnavailableError("MOEX request retries exhausted")

    async def _backoff_or_raise(
        self,
        exc: Exception,
        attempt: int,
        response: httpx.Response | None,
    ) -> None:
        if attempt >= self._max_retries:
            if isinstance(exc, MarketDataProviderUnavailableError):
                raise exc
            raise MarketDataProviderUnavailableError("MOEX request retries exhausted") from exc
        retry_after = None if response is None else response.headers.get("Retry-After")
        delay = retry_delay_seconds(attempt, retry_after)
        if self._sleep:
            await asyncio.sleep(delay)


@dataclass(frozen=True, slots=True)
class _ParsedPage:
    candles: list[MarketCandle]
    rows_received: int
    rejected_rows: list[RejectedCandleRow]


@dataclass(frozen=True, slots=True)
class _ParsedBenchmarkPage:
    candles: list[BenchmarkCandle]
    rows_received: int
    rejected_rows: list[RejectedCandleRow]


def validate_market_path_value(value: str, field_name: str) -> str:
    normalized = value.strip().upper()
    if not _MARKET_CODE_PATTERN.fullmatch(normalized):
        raise MarketDataValidationError(f"{field_name} contains unsupported characters")
    return normalized


def validate_date_range(date_from: date, date_till: date) -> None:
    if date_till < date_from:
        raise MarketDataValidationError("date_till must not be before date_from")
    if (date_till - date_from).days > _MAX_REQUEST_DAYS:
        raise MarketDataValidationError("date range must not exceed 31 calendar days")


def _validate_base_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    parsed = httpx.URL(stripped)
    if parsed.scheme != "https" or parsed.host != "iss.moex.com":
        raise MarketDataValidationError("MOEX ISS base URL is not allowed")
    return stripped


def _parse_page(
    payload: dict[str, Any],
    *,
    instrument_id: UUID,
    ticker: str,
    board: str,
    interval_minutes: int,
    page: int,
) -> _ParsedPage:
    candles_obj = payload.get("candles")
    if not isinstance(candles_obj, dict):
        raise MarketDataProviderContractError("MOEX response misses candles object")
    typed_candles = cast("dict[str, Any]", candles_obj)
    columns: object = typed_candles.get("columns")
    data: object = typed_candles.get("data")
    if not isinstance(columns, list):
        raise MarketDataProviderContractError("MOEX candles.columns must be a string array")
    raw_columns = cast("list[object]", columns)
    if not all(isinstance(item, str) for item in raw_columns):
        raise MarketDataProviderContractError("MOEX candles.columns must be a string array")
    if not isinstance(data, list):
        raise MarketDataProviderContractError("MOEX candles.data must be an array")
    typed_columns = cast("list[str]", raw_columns)
    typed_rows = cast("list[Any]", data)
    column_index = {column: index for index, column in enumerate(typed_columns)}
    missing = [column for column in _REQUIRED_COLUMNS if column not in column_index]
    if missing:
        raise MarketDataProviderContractError(
            f"MOEX candles.columns misses required fields: {', '.join(missing)}"
        )
    parsed: list[MarketCandle] = []
    rejected: list[RejectedCandleRow] = []
    for row_index, row in enumerate(typed_rows):
        if not isinstance(row, list):
            rejected.append(
                RejectedCandleRow(page=page, row_index=row_index, reason="row_not_array")
            )
            continue
        try:
            parsed.append(
                _row_to_candle(
                    cast("list[object]", row),
                    column_index,
                    instrument_id=instrument_id,
                    ticker=ticker,
                    board=board,
                    interval_minutes=interval_minutes,
                )
            )
        except (InvalidOperation, TypeError, ValueError) as exc:
            rejected.append(RejectedCandleRow(page=page, row_index=row_index, reason=str(exc)))
    return _ParsedPage(candles=parsed, rows_received=len(typed_rows), rejected_rows=rejected)


def _parse_benchmark_page(
    payload: dict[str, Any],
    *,
    benchmark_id: UUID,
    interval_minutes: int,
    page: int,
) -> _ParsedBenchmarkPage:
    columns, rows = _candle_columns_and_rows(payload)
    column_index = {column: index for index, column in enumerate(columns)}
    parsed: list[BenchmarkCandle] = []
    rejected: list[RejectedCandleRow] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            rejected.append(
                RejectedCandleRow(page=page, row_index=row_index, reason="row_not_array")
            )
            continue
        try:
            parsed.append(
                _row_to_benchmark_candle(
                    cast("list[object]", row),
                    column_index,
                    benchmark_id=benchmark_id,
                    interval_minutes=interval_minutes,
                )
            )
        except (InvalidOperation, TypeError, ValueError) as exc:
            rejected.append(RejectedCandleRow(page=page, row_index=row_index, reason=str(exc)))
    return _ParsedBenchmarkPage(
        candles=parsed,
        rows_received=len(rows),
        rejected_rows=rejected,
    )


def _candle_columns_and_rows(payload: dict[str, Any]) -> tuple[list[str], list[Any]]:
    candles_obj = payload.get("candles")
    if not isinstance(candles_obj, dict):
        raise MarketDataProviderContractError("MOEX response misses candles object")
    typed_candles = cast("dict[str, Any]", candles_obj)
    columns: object = typed_candles.get("columns")
    data: object = typed_candles.get("data")
    if not isinstance(columns, list):
        raise MarketDataProviderContractError("MOEX candles.columns must be a string array")
    raw_columns = cast("list[object]", columns)
    if not all(isinstance(item, str) for item in raw_columns):
        raise MarketDataProviderContractError("MOEX candles.columns must be a string array")
    if not isinstance(data, list):
        raise MarketDataProviderContractError("MOEX candles.data must be an array")
    typed_columns = cast("list[str]", raw_columns)
    typed_rows = cast("list[Any]", data)
    missing = [column for column in _REQUIRED_COLUMNS if column not in typed_columns]
    if missing:
        raise MarketDataProviderContractError(
            f"MOEX candles.columns misses required fields: {', '.join(missing)}"
        )
    return typed_columns, typed_rows


def _row_to_candle(
    row: list[object],
    column_index: dict[str, int],
    *,
    instrument_id: UUID,
    ticker: str,
    board: str,
    interval_minutes: int,
) -> MarketCandle:
    open_price = _decimal(row, column_index, "open")
    close_price = _decimal(row, column_index, "close")
    high_price = _decimal(row, column_index, "high")
    low_price = _decimal(row, column_index, "low")
    value = _decimal(row, column_index, "value")
    volume = _decimal(row, column_index, "volume")
    begin_at = _moex_datetime(row, column_index, "begin")
    end_at = _moex_datetime(row, column_index, "end")
    return MarketCandle.create(
        instrument_id=instrument_id,
        board=board,
        ticker_snapshot=ticker,
        interval_minutes=interval_minutes,
        begin_at=begin_at,
        end_at=end_at,
        open_price=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
        value=value,
        adapter_version=MOEX_ADAPTER_VERSION,
    )


def _row_to_benchmark_candle(
    row: list[object],
    column_index: dict[str, int],
    *,
    benchmark_id: UUID,
    interval_minutes: int,
) -> BenchmarkCandle:
    return BenchmarkCandle.create(
        benchmark_id=benchmark_id,
        interval_minutes=interval_minutes,
        begin_at=_moex_datetime(row, column_index, "begin"),
        end_at=_moex_datetime(row, column_index, "end"),
        open_price=_decimal(row, column_index, "open"),
        high=_decimal(row, column_index, "high"),
        low=_decimal(row, column_index, "low"),
        close=_decimal(row, column_index, "close"),
        volume=_decimal(row, column_index, "volume"),
        value=_decimal(row, column_index, "value"),
        adapter_version=MOEX_ADAPTER_VERSION,
    )


def _row_to_daily_candle(
    row: list[object],
    column_index: dict[str, int],
    *,
    security_code: str,
    board: str,
) -> MoexDailyCandle:
    open_price = _decimal(row, column_index, "open")
    close_price = _decimal(row, column_index, "close")
    high_price = _decimal(row, column_index, "high")
    low_price = _decimal(row, column_index, "low")
    value = _decimal(row, column_index, "value")
    volume = _decimal(row, column_index, "volume")
    trade_date = _moex_trade_date(row, column_index, "begin")
    if min(open_price, close_price, high_price, low_price) <= 0:
        raise ValueError("daily OHLC prices must be positive")
    if low_price > high_price or not low_price <= open_price <= high_price:
        raise ValueError("daily open must be inside low-high range")
    if not low_price <= close_price <= high_price:
        raise ValueError("daily close must be inside low-high range")
    if volume < 0 or value < 0:
        raise ValueError("daily value and volume must be non-negative")
    return MoexDailyCandle(
        security_code=security_code,
        board=board,
        trade_date=trade_date,
        open=open_price,
        close=close_price,
        high=high_price,
        low=low_price,
        value=value,
        volume=volume,
    )


def _decimal(row: list[object], column_index: dict[str, int], column: str) -> Decimal:
    value = _field(row, column_index, column)
    if value is None or value == "":
        raise ValueError(f"{column}_missing")
    return Decimal(str(value))


def _moex_datetime(row: list[object], column_index: dict[str, int], column: str) -> datetime:
    value = _field(row, column_index, column)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{column}_missing")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed.replace(tzinfo=ZoneInfo(MOEX_SOURCE_TIMEZONE)).astimezone(UTC)


def _moex_trade_date(row: list[object], column_index: dict[str, int], column: str) -> date:
    value = _field(row, column_index, column)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{column}_missing")
    return datetime.fromisoformat(value).date()


def _field(row: list[object], column_index: dict[str, int], column: str) -> object:
    index = column_index[column]
    if index >= len(row):
        raise ValueError(f"{column}_missing")
    return row[index]


def retry_delay_seconds(attempt: int, retry_after: str | None) -> float:
    if retry_after is not None:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            return 1.0
    return min(0.25 * (2**attempt), 5.0)
