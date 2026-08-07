from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel

from src.evaluation.domain.entities import EvaluationDataset, EvaluationRun
from src.evaluation.domain.enums import DatasetSplit, EvaluationRunStatus


class EvaluationDatasetSummaryResponse(BaseModel):
    id: UUID
    name: str
    schema_version: str
    description: str | None
    imported_at: datetime
    example_count: int
    reviewed_count: int
    train_count: int
    validation_count: int
    test_count: int

    @classmethod
    def from_entity(cls, dataset: EvaluationDataset) -> EvaluationDatasetSummaryResponse:
        return cls(
            id=dataset.id,
            name=dataset.name,
            schema_version=dataset.schema_version,
            description=dataset.description,
            imported_at=dataset.imported_at,
            example_count=dataset.example_count,
            reviewed_count=dataset.reviewed_count,
            train_count=dataset.train_count,
            validation_count=dataset.validation_count,
            test_count=dataset.test_count,
        )


class EvaluationDatasetDetailResponse(EvaluationDatasetSummaryResponse):
    source_file_hash: str
    created_at: datetime
    split_strategy: str | None
    train_until: date | None
    validation_until: date | None

    @classmethod
    def from_entity(cls, dataset: EvaluationDataset) -> EvaluationDatasetDetailResponse:
        return cls(
            id=dataset.id,
            name=dataset.name,
            schema_version=dataset.schema_version,
            description=dataset.description,
            source_file_hash=dataset.source_file_hash,
            created_at=dataset.created_at,
            imported_at=dataset.imported_at,
            example_count=dataset.example_count,
            reviewed_count=dataset.reviewed_count,
            train_count=dataset.train_count,
            validation_count=dataset.validation_count,
            test_count=dataset.test_count,
            split_strategy=dataset.split_strategy,
            train_until=dataset.train_until,
            validation_until=dataset.validation_until,
        )


class CreateEvaluationRunRequest(BaseModel):
    split: DatasetSplit = DatasetSplit.TEST
    thresholds_path: str = "config/evaluation_thresholds.toml"
    output_dir: str = "artifacts/evaluation"


class EvaluationRunResponse(BaseModel):
    id: UUID
    dataset_id: UUID
    split: DatasetSplit
    analysis_version: str
    extractor_version: str
    started_at: datetime
    finished_at: datetime | None
    status: EvaluationRunStatus
    example_count: int
    metrics_json: dict[str, object]
    error_count: int
    git_commit_sha: str
    config_json: dict[str, object]

    @classmethod
    def from_entity(cls, run: EvaluationRun) -> EvaluationRunResponse:
        return cls(
            id=run.id,
            dataset_id=run.dataset_id,
            split=run.split,
            analysis_version=run.analysis_version,
            extractor_version=run.extractor_version,
            started_at=run.started_at,
            finished_at=run.finished_at,
            status=run.status,
            example_count=run.example_count,
            metrics_json=run.metrics_json,
            error_count=run.error_count,
            git_commit_sha=run.git_commit_sha,
            config_json=run.config_json,
        )
