from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.evaluation.domain.entities import (
    AnnotationExample,
    EvaluationDataset,
    EvaluationRun,
)
from src.evaluation.domain.enums import DatasetSplit, EvaluationRunStatus, ReviewStatus
from src.evaluation.infrastructure.models import (
    EvaluationDatasetRecord,
    EvaluationExampleRecord,
    EvaluationRunRecord,
    GoldEventRecord,
    GoldFinancialFactRecord,
)
from src.news.domain.time import utc_now
from src.news.infrastructure.models import NewsItemRecord


@dataclass(frozen=True, slots=True)
class ImportDatasetResult:
    dataset: EvaluationDataset
    created: bool


@dataclass(frozen=True, slots=True)
class EvaluationExampleWithNews:
    example: EvaluationExampleRecord
    news: NewsItemRecord


class EvaluationStorageError(Exception):
    pass


class SqlAlchemyEvaluationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_datasets(self) -> list[EvaluationDataset]:
        result = await self._session.execute(
            select(EvaluationDatasetRecord).order_by(EvaluationDatasetRecord.imported_at.desc())
        )
        return [record.to_entity() for record in result.scalars()]

    async def get_dataset(self, dataset_id: UUID) -> EvaluationDataset | None:
        record = await self._get_dataset_record(dataset_id)
        return None if record is None else record.to_entity()

    async def get_run(self, run_id: UUID) -> EvaluationRun | None:
        result = await self._session.execute(
            select(EvaluationRunRecord).where(EvaluationRunRecord.id == run_id)
        )
        record = result.scalar_one_or_none()
        return None if record is None else record.to_entity()

    async def import_dataset(
        self,
        *,
        dataset: EvaluationDataset,
        examples: Sequence[AnnotationExample],
    ) -> ImportDatasetResult:
        existing = await self._get_dataset_by_hash(dataset.source_file_hash)
        if existing is not None:
            return ImportDatasetResult(dataset=existing.to_entity(), created=False)
        record = EvaluationDatasetRecord.from_entity(dataset)
        now = utc_now()
        record.examples = []
        for example in examples:
            example_id = uuid4()
            example_record = EvaluationExampleRecord(
                id=example_id,
                dataset_id=dataset.id,
                news_id=example.news_id,
                published_at=example.published_at,
                raw_content_hash=example.raw_content_hash,
                split=example.split.value,
                review_status=example.review_status.value,
                annotator=example.annotator,
                notes=example.notes,
                predicted_events=example.predicted_events,
                predicted_financial_facts=example.predicted_financial_facts,
                created_at=now,
                updated_at=now,
            )
            example_record.gold_events = [
                GoldEventRecord.from_entity(example_id, event) for event in example.gold_events
            ]
            example_record.gold_financial_facts = [
                GoldFinancialFactRecord.from_entity(example_id, fact)
                for fact in example.gold_financial_facts
            ]
            record.examples.append(example_record)
        self._session.add(record)
        try:
            await self._session.commit()
        except IntegrityError:
            await self._session.rollback()
            existing = await self._get_dataset_by_hash(dataset.source_file_hash)
            if existing is not None:
                return ImportDatasetResult(dataset=existing.to_entity(), created=False)
            raise
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise EvaluationStorageError("could not import evaluation dataset") from exc
        saved = await self._get_dataset_record(dataset.id)
        if saved is None:
            raise EvaluationStorageError("evaluation dataset save could not be resolved")
        return ImportDatasetResult(dataset=saved.to_entity(), created=True)

    async def list_examples_with_news(
        self,
        *,
        dataset_id: UUID,
        split: DatasetSplit | None = None,
    ) -> list[EvaluationExampleWithNews]:
        query = (
            select(EvaluationExampleRecord, NewsItemRecord)
            .join(NewsItemRecord, EvaluationExampleRecord.news_id == NewsItemRecord.id)
            .where(EvaluationExampleRecord.dataset_id == dataset_id)
            .options(
                selectinload(EvaluationExampleRecord.gold_events),
                selectinload(EvaluationExampleRecord.gold_financial_facts),
            )
            .order_by(EvaluationExampleRecord.published_at, EvaluationExampleRecord.news_id)
        )
        if split is not None:
            query = query.where(EvaluationExampleRecord.split == split.value)
        result = await self._session.execute(query)
        return [
            EvaluationExampleWithNews(example=example, news=news) for example, news in result.all()
        ]

    async def assign_temporal_split(
        self,
        *,
        dataset_id: UUID,
        train_until: date,
        validation_until: date,
    ) -> dict[DatasetSplit, int]:
        rows = await self.list_examples_with_news(dataset_id=dataset_id)
        counts = {split: 0 for split in DatasetSplit}
        for row in rows:
            published_date = row.news.published_at.date()
            if published_date <= train_until:
                split = DatasetSplit.TRAIN
            elif published_date <= validation_until:
                split = DatasetSplit.VALIDATION
            else:
                split = DatasetSplit.TEST
            row.example.split = split.value
            row.example.updated_at = utc_now()
            counts[split] += 1
        dataset = await self._get_dataset_record(dataset_id)
        if dataset is None:
            raise EvaluationStorageError("evaluation dataset not found")
        dataset.train_count = counts[DatasetSplit.TRAIN]
        dataset.validation_count = counts[DatasetSplit.VALIDATION]
        dataset.test_count = counts[DatasetSplit.TEST]
        dataset.split_strategy = "temporal"
        dataset.train_until = train_until
        dataset.validation_until = validation_until
        await self._session.commit()
        return counts

    async def save_run(self, run: EvaluationRun) -> EvaluationRun:
        self._session.add(EvaluationRunRecord.from_entity(run))
        await self._session.commit()
        saved = await self.get_run(run.id)
        if saved is None:
            raise EvaluationStorageError("evaluation run save could not be resolved")
        return saved

    async def finish_run(
        self,
        *,
        run_id: UUID,
        status: EvaluationRunStatus,
        example_count: int,
        metrics_json: dict[str, object],
        error_count: int,
    ) -> EvaluationRun:
        result = await self._session.execute(
            select(EvaluationRunRecord).where(EvaluationRunRecord.id == run_id)
        )
        record = result.scalar_one()
        record.status = status.value
        record.finished_at = utc_now()
        record.example_count = example_count
        record.metrics_json = metrics_json
        record.error_count = error_count
        await self._session.commit()
        saved = await self.get_run(run_id)
        if saved is None:
            raise EvaluationStorageError("evaluation run update could not be resolved")
        return saved

    async def raw_news_maps(
        self,
        news_ids: Sequence[UUID],
    ) -> tuple[dict[UUID, str], dict[UUID, str]]:
        if not news_ids:
            return {}, {}
        result = await self._session.execute(
            select(
                NewsItemRecord.id,
                NewsItemRecord.raw_content,
                NewsItemRecord.raw_content_hash,
            ).where(NewsItemRecord.id.in_(news_ids))
        )
        content_by_id: dict[UUID, str] = {}
        hash_by_id: dict[UUID, str] = {}
        for news_id, raw_content, raw_content_hash in result.all():
            content_by_id[news_id] = raw_content
            hash_by_id[news_id] = raw_content_hash
        return content_by_id, hash_by_id

    async def count_news(self) -> int:
        result = await self._session.execute(select(func.count()).select_from(NewsItemRecord))
        return int(result.scalar_one())

    async def _get_dataset_record(self, dataset_id: UUID) -> EvaluationDatasetRecord | None:
        result = await self._session.execute(
            select(EvaluationDatasetRecord).where(EvaluationDatasetRecord.id == dataset_id)
        )
        return result.scalar_one_or_none()

    async def _get_dataset_by_hash(self, source_file_hash: str) -> EvaluationDatasetRecord | None:
        result = await self._session.execute(
            select(EvaluationDatasetRecord).where(
                EvaluationDatasetRecord.source_file_hash == source_file_hash
            )
        )
        return result.scalar_one_or_none()


def dataset_from_examples(
    *,
    name: str,
    source_file_hash: str,
    examples: Sequence[AnnotationExample],
    description: str | None,
) -> EvaluationDataset:
    return EvaluationDataset.create(
        name=name,
        source_file_hash=source_file_hash,
        description=description,
        example_count=len(examples),
        reviewed_count=sum(
            1 for example in examples if example.review_status == ReviewStatus.REVIEWED
        ),
        train_count=sum(1 for example in examples if example.split == DatasetSplit.TRAIN),
        validation_count=sum(1 for example in examples if example.split == DatasetSplit.VALIDATION),
        test_count=sum(1 for example in examples if example.split == DatasetSplit.TEST),
    )
