from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.dependencies import get_evaluation_repository
from src.evaluation.application.exceptions import (
    EvaluationDatasetNotFoundError,
    EvaluationThresholdError,
)
from src.evaluation.application.use_cases import run_evaluation
from src.evaluation.infrastructure.repositories import SqlAlchemyEvaluationRepository
from src.evaluation.presentation.schemas import (
    CreateEvaluationRunRequest,
    EvaluationDatasetDetailResponse,
    EvaluationDatasetSummaryResponse,
    EvaluationRunResponse,
)

router = APIRouter(tags=["evaluation"])


@router.get(
    "/evaluation/datasets",
    response_model=list[EvaluationDatasetSummaryResponse],
)
async def list_evaluation_datasets(
    repository: SqlAlchemyEvaluationRepository = Depends(get_evaluation_repository),
) -> list[EvaluationDatasetSummaryResponse]:
    datasets = await repository.list_datasets()
    return [EvaluationDatasetSummaryResponse.from_entity(dataset) for dataset in datasets]


@router.get(
    "/evaluation/datasets/{dataset_id}",
    response_model=EvaluationDatasetDetailResponse,
)
async def get_evaluation_dataset(
    dataset_id: UUID,
    repository: SqlAlchemyEvaluationRepository = Depends(get_evaluation_repository),
) -> EvaluationDatasetDetailResponse:
    dataset = await repository.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="dataset not found")
    return EvaluationDatasetDetailResponse.from_entity(dataset)


@router.post(
    "/evaluation/datasets/{dataset_id}/runs",
    response_model=EvaluationRunResponse,
)
async def create_evaluation_run(
    dataset_id: UUID,
    request: CreateEvaluationRunRequest,
    repository: SqlAlchemyEvaluationRepository = Depends(get_evaluation_repository),
) -> EvaluationRunResponse:
    try:
        result = await run_evaluation(
            repository=repository,
            dataset_id=dataset_id,
            split=request.split,
            thresholds_path=Path(request.thresholds_path),
            output_dir=Path(request.output_dir),
            fail_below_thresholds=False,
        )
    except EvaluationDatasetNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="dataset not found"
        ) from exc
    except EvaluationThresholdError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return EvaluationRunResponse.from_entity(result.run)


@router.get(
    "/evaluation/runs/{run_id}",
    response_model=EvaluationRunResponse,
)
async def get_evaluation_run(
    run_id: UUID,
    repository: SqlAlchemyEvaluationRepository = Depends(get_evaluation_repository),
) -> EvaluationRunResponse:
    run = await repository.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return EvaluationRunResponse.from_entity(run)
