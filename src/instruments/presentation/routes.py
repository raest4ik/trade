from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from apps.api.dependencies import get_instrument_repository, get_news_repository
from src.instruments.application.exceptions import (
    InstrumentNotFoundError,
    InstrumentStorageError,
    NewsForMatchingNotFoundError,
)
from src.instruments.application.use_cases import (
    CreateInstrument,
    CreateIssuerAlias,
    GetNewsInstrumentMatches,
    ListInstruments,
    MatchNewsInstruments,
)
from src.instruments.domain.exceptions import InstrumentDomainError
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.instruments.presentation.schemas import (
    InstrumentCreateRequest,
    InstrumentResponse,
    IssuerAliasCreateRequest,
    IssuerAliasResponse,
    MatchNewsInstrumentsResponse,
    NewsInstrumentMatchResponse,
)
from src.news.infrastructure.repositories import SqlAlchemyNewsRepository

router = APIRouter(tags=["instruments"])


@router.post(
    "/instruments",
    response_model=InstrumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_instrument(
    payload: InstrumentCreateRequest,
    response: Response,
    repository: SqlAlchemyInstrumentRepository = Depends(get_instrument_repository),
) -> InstrumentResponse:
    try:
        result = await CreateInstrument(repository).execute(payload.to_command())
    except InstrumentDomainError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InstrumentStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="instrument storage is unavailable",
        ) from exc
    response.status_code = status.HTTP_201_CREATED if result.created else status.HTTP_200_OK
    return InstrumentResponse.from_entity(result.instrument)


@router.get("/instruments", response_model=list[InstrumentResponse])
async def list_instruments(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: SqlAlchemyInstrumentRepository = Depends(get_instrument_repository),
) -> list[InstrumentResponse]:
    try:
        instruments = await ListInstruments(repository).execute(limit=limit, offset=offset)
    except InstrumentStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="instrument storage is unavailable",
        ) from exc
    return [InstrumentResponse.from_entity(instrument) for instrument in instruments]


@router.post(
    "/instruments/{instrument_id}/aliases",
    response_model=IssuerAliasResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_issuer_alias(
    instrument_id: UUID,
    payload: IssuerAliasCreateRequest,
    response: Response,
    repository: SqlAlchemyInstrumentRepository = Depends(get_instrument_repository),
) -> IssuerAliasResponse:
    try:
        result = await CreateIssuerAlias(repository).execute(payload.to_command(instrument_id))
    except InstrumentDomainError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InstrumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="instrument not found"
        ) from exc
    except InstrumentStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="instrument storage is unavailable",
        ) from exc
    response.status_code = status.HTTP_201_CREATED if result.created else status.HTTP_200_OK
    return IssuerAliasResponse.from_entity(result.alias)


@router.post(
    "/news/{news_id}/match-instruments",
    response_model=MatchNewsInstrumentsResponse,
)
async def match_news_instruments(
    news_id: UUID,
    news_repository: SqlAlchemyNewsRepository = Depends(get_news_repository),
    instrument_repository: SqlAlchemyInstrumentRepository = Depends(get_instrument_repository),
) -> MatchNewsInstrumentsResponse:
    use_case = MatchNewsInstruments(news_repository, instrument_repository)
    try:
        result = await use_case.execute(news_id)
    except NewsForMatchingNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="news item not found"
        ) from exc
    except InstrumentStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="instrument storage is unavailable",
        ) from exc
    return MatchNewsInstrumentsResponse(
        news_id=result.news_id,
        matcher_version=result.matcher_version,
        matches=[NewsInstrumentMatchResponse.from_entity(match) for match in result.matches],
    )


@router.get(
    "/news/{news_id}/instruments",
    response_model=MatchNewsInstrumentsResponse,
)
async def get_news_instruments(
    news_id: UUID,
    repository: SqlAlchemyInstrumentRepository = Depends(get_instrument_repository),
) -> MatchNewsInstrumentsResponse:
    try:
        result = await GetNewsInstrumentMatches(repository).execute(news_id)
    except InstrumentStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="instrument storage is unavailable",
        ) from exc
    return MatchNewsInstrumentsResponse(
        news_id=result.news_id,
        matcher_version=result.matcher_version,
        matches=[NewsInstrumentMatchResponse.from_entity(match) for match in result.matches],
    )
