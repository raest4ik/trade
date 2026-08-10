from __future__ import annotations

from src.market_data.domain.entities import BenchmarkCandle, MarketCandle


class MarketDataApplicationError(Exception):
    """Base market data application error."""


class MarketDataStorageError(MarketDataApplicationError):
    """Raised when market data storage is unavailable."""


class MarketDataProviderError(MarketDataApplicationError):
    """Raised when an external market data provider fails."""


class MarketDataProviderContractError(MarketDataProviderError):
    """Raised when the provider response shape is invalid."""


class MarketDataProviderUnavailableError(MarketDataProviderError):
    """Raised when the provider is temporarily unavailable."""


class MarketDataPartialProviderError(MarketDataProviderError):
    """Raised when the provider fails after returning usable rows."""

    def __init__(
        self,
        message: str,
        *,
        candles: list[MarketCandle],
        pages_received: int,
        rows_received: int,
        rows_valid: int,
        rows_rejected: int,
    ) -> None:
        super().__init__(message)
        self.candles = candles
        self.pages_received = pages_received
        self.rows_received = rows_received
        self.rows_valid = rows_valid
        self.rows_rejected = rows_rejected


class BenchmarkDataPartialProviderError(MarketDataProviderError):
    """Raised when benchmark fetching fails after returning usable rows."""

    def __init__(
        self,
        message: str,
        *,
        candles: list[BenchmarkCandle],
        pages_received: int,
        rows_received: int,
        rows_valid: int,
        rows_rejected: int,
    ) -> None:
        super().__init__(message)
        self.candles = candles
        self.pages_received = pages_received
        self.rows_received = rows_received
        self.rows_valid = rows_valid
        self.rows_rejected = rows_rejected


class MarketDataValidationError(MarketDataApplicationError):
    """Raised when request parameters are invalid."""


class InstrumentMarketDataNotFoundError(MarketDataApplicationError):
    """Raised when an instrument does not exist."""


class InstrumentMarketDataConflictError(MarketDataApplicationError):
    """Raised when an instrument lacks required market data configuration."""
