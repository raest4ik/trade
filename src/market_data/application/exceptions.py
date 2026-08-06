from __future__ import annotations


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


class MarketDataValidationError(MarketDataApplicationError):
    """Raised when request parameters are invalid."""


class InstrumentMarketDataNotFoundError(MarketDataApplicationError):
    """Raised when an instrument does not exist."""


class InstrumentMarketDataConflictError(MarketDataApplicationError):
    """Raised when an instrument lacks required market data configuration."""
