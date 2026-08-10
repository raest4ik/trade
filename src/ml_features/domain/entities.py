from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from src.events.domain.entities import EVENT_ANALYSIS_VERSION, FINANCIAL_FACTS_VERSION
from src.ml_features.domain.enums import FeatureDatasetRunStatus, FeatureExclusionReason
from src.news.domain.time import ensure_aware_utc, utc_now
from src.reactions.domain.entities import REACTION_VERSION

DATASET_VERSION = "ml-feature-dataset-v1"
FEATURE_VERSION = "ml-features-v1"
MARKET_CONTEXT_VERSION = "pre-event-market-v1"
BENCHMARK_CODE = "IMOEX"
FEATURE_HORIZONS_MINUTES = (5, 15, 30, 60)
LABEL_HORIZONS_MINUTES = (1, 5, 15, 30, 60)


@dataclass(frozen=True, slots=True)
class FeatureDatasetConfig:
    date_from: datetime
    date_to: datetime
    tickers: tuple[str, ...] = ()
    limit: int = 10_000
    require_label_horizon: int | None = None
    classification_threshold: Decimal | None = None
    dataset_version: str = DATASET_VERSION
    feature_version: str = FEATURE_VERSION
    event_analysis_version: str = EVENT_ANALYSIS_VERSION
    fact_extractor_version: str = FINANCIAL_FACTS_VERSION
    reaction_version: str = REACTION_VERSION
    market_context_version: str = MARKET_CONTEXT_VERSION
    benchmark_code: str = BENCHMARK_CODE

    def normalized(self) -> FeatureDatasetConfig:
        date_from = ensure_aware_utc(self.date_from, "date_from")
        date_to = ensure_aware_utc(self.date_to, "date_to")
        if date_to < date_from:
            raise ValueError("date_to must not be before date_from")
        if not 1 <= self.limit <= 100_000:
            raise ValueError("limit must be between 1 and 100000")
        if (
            self.require_label_horizon is not None
            and self.require_label_horizon not in LABEL_HORIZONS_MINUTES
        ):
            raise ValueError("require_label_horizon is unsupported")
        if self.classification_threshold is not None and self.classification_threshold < 0:
            raise ValueError("classification_threshold must not be negative")
        return replace(
            self,
            date_from=date_from,
            date_to=date_to,
            tickers=tuple(
                sorted({ticker.strip().upper() for ticker in self.tickers if ticker.strip()})
            ),
            benchmark_code=self.benchmark_code.strip().upper(),
        )

    def hash(self) -> str:
        normalized = self.normalized()
        payload = {
            "benchmark_code": normalized.benchmark_code,
            "classification_threshold": _decimal_text(normalized.classification_threshold),
            "dataset_version": normalized.dataset_version,
            "date_from": normalized.date_from.isoformat(),
            "date_to": normalized.date_to.isoformat(),
            "event_analysis_version": normalized.event_analysis_version,
            "fact_extractor_version": normalized.fact_extractor_version,
            "feature_version": normalized.feature_version,
            "limit": normalized.limit,
            "market_context_version": normalized.market_context_version,
            "reaction_version": normalized.reaction_version,
            "require_label_horizon": normalized.require_label_horizon,
            "tickers": list(normalized.tickers),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class MlFeatureDatasetRun:
    id: UUID
    dataset_version: str
    feature_version: str
    date_from: datetime
    date_to: datetime
    started_at: datetime
    finished_at: datetime | None
    candidate_count: int
    eligible_count: int
    built_count: int
    excluded_count: int
    failed_count: int
    config_hash: str
    git_sha: str
    status: FeatureDatasetRunStatus
    error: str | None

    @classmethod
    def start(cls, config: FeatureDatasetConfig, *, git_sha: str) -> MlFeatureDatasetRun:
        normalized = config.normalized()
        return cls(
            id=uuid4(),
            dataset_version=normalized.dataset_version,
            feature_version=normalized.feature_version,
            date_from=normalized.date_from,
            date_to=normalized.date_to,
            started_at=utc_now(),
            finished_at=None,
            candidate_count=0,
            eligible_count=0,
            built_count=0,
            excluded_count=0,
            failed_count=0,
            config_hash=normalized.hash(),
            git_sha=git_sha.strip() or "UNKNOWN",
            status=FeatureDatasetRunStatus.RUNNING,
            error=None,
        )

    def finish(
        self,
        *,
        status: FeatureDatasetRunStatus,
        candidate_count: int,
        eligible_count: int,
        built_count: int,
        excluded_count: int,
        failed_count: int,
        error: str | None = None,
    ) -> MlFeatureDatasetRun:
        return replace(
            self,
            finished_at=utc_now(),
            candidate_count=candidate_count,
            eligible_count=eligible_count,
            built_count=built_count,
            excluded_count=excluded_count,
            failed_count=failed_count,
            status=status,
            error=error,
        )


@dataclass(frozen=True, slots=True)
class FeatureDatasetRow:
    metadata: dict[str, Any]
    features: dict[str, Any]
    labels: dict[str, Any]
    quality: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "features": self.features,
            "labels": self.labels,
            "quality": self.quality,
        }


@dataclass(frozen=True, slots=True)
class FeatureExclusion:
    news_id: UUID
    instrument_id: UUID | None
    reason: FeatureExclusionReason
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class FeatureDatasetBuildResult:
    rows: list[FeatureDatasetRow]
    exclusions: list[FeatureExclusion]
    run: MlFeatureDatasetRun


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)
