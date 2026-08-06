from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from apps.api.dependencies import (
    get_instrument_repository,
    get_market_data_repository,
    get_moex_client,
)
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.market_data.application.exceptions import (
    InstrumentMarketDataConflictError,
    InstrumentMarketDataNotFoundError,
    MarketDataProviderContractError,
    MarketDataProviderUnavailableError,
    MarketDataStorageError,
    MarketDataValidationError,
)
from src.market_data.application.use_cases import (
    BackfillInstrumentCandles,
    BackfillInstrumentCandlesCommand,
    GetMarketDataImport,
    ListInstrumentCandles,
)
from src.market_data.infrastructure.moex_client import MoexIssClient
from src.market_data.infrastructure.repositories import SqlAlchemyMarketDataRepository
from src.market_data.presentation.schemas import (
    BackfillCandlesRequest,
    BackfillCandlesResponse,
    MarketCandleResponse,
    MarketDataImportResponse,
)

router = APIRouter(tags=["market-data"])


@router.post(
    "/instruments/{instrument_id}/candles/backfill",
    response_model=BackfillCandlesResponse,
)
async def backfill_instrument_candles(
    instrument_id: UUID,
    payload: BackfillCandlesRequest,
    instrument_repository: SqlAlchemyInstrumentRepository = Depends(get_instrument_repository),
    market_data_repository: SqlAlchemyMarketDataRepository = Depends(get_market_data_repository),
    moex_client: MoexIssClient = Depends(get_moex_client),
) -> BackfillCandlesResponse:
    use_case = BackfillInstrumentCandles(
        instrument_repository=instrument_repository,
        market_data_repository=market_data_repository,
        provider=moex_client,
    )
    try:
        result = await use_case.execute(
            BackfillInstrumentCandlesCommand(
                instrument_id=instrument_id,
                date_from=payload.date_from,
                date_till=payload.date_till,
                interval_minutes=payload.interval_minutes,
            )
        )
    except InstrumentMarketDataNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="instrument not found"
        ) from exc
    except InstrumentMarketDataConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="instrument lacks ticker or primary_board",
        ) from exc
    except MarketDataValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except MarketDataProviderContractError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except MarketDataProviderUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MOEX ISS is unavailable",
        ) from exc
    except MarketDataStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="market data storage is unavailable",
        ) from exc
    return BackfillCandlesResponse.from_import(result.import_record)


@router.get(
    "/instruments/{instrument_id}/candles",
    response_model=list[MarketCandleResponse],
)
async def list_instrument_candles(
    instrument_id: UUID,
    from_at: datetime = Query(alias="from"),
    till_at: datetime = Query(alias="till"),
    interval_minutes: int = Query(default=1, ge=1, le=60),
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    market_data_repository: SqlAlchemyMarketDataRepository = Depends(get_market_data_repository),
) -> list[MarketCandleResponse]:
    try:
        candles = await ListInstrumentCandles(market_data_repository).execute(
            instrument_id=instrument_id,
            from_at=from_at,
            till_at=till_at,
            interval_minutes=interval_minutes,
            limit=limit,
            offset=offset,
        )
    except MarketDataValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except MarketDataStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="market data storage is unavailable",
        ) from exc
    return [MarketCandleResponse.from_entity(candle) for candle in candles]


@router.get(
    "/market-data/imports/{import_id}",
    response_model=MarketDataImportResponse,
)
async def get_market_data_import(
    import_id: UUID,
    market_data_repository: SqlAlchemyMarketDataRepository = Depends(get_market_data_repository),
) -> MarketDataImportResponse:
    try:
        item = await GetMarketDataImport(market_data_repository).execute(import_id)
    except MarketDataStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="market data storage is unavailable",
        ) from exc
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="import not found")
    return MarketDataImportResponse.from_entity(item)
