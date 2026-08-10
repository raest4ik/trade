from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from src.events.application.ports import EventAnalysisRepository
from src.events.domain.entities import ExtractedFinancialFact, NewsEventAnalysis
from src.events.domain.enums import (
    ChangeDirection,
    EventType,
    FactRole,
    FactUnit,
    FinancialMetric,
)
from src.ml_features.application.point_in_time import (
    PointInTimeFeatureBuilder,
    PointInTimeMarketFeatures,
    PointInTimeViolationError,
)
from src.ml_features.application.ports import FeatureCandidate, MlFeatureRepository
from src.ml_features.domain.entities import (
    FEATURE_HORIZONS_MINUTES,
    LABEL_HORIZONS_MINUTES,
    FeatureDatasetBuildResult,
    FeatureDatasetConfig,
    FeatureDatasetRow,
    FeatureExclusion,
    MlFeatureDatasetRun,
)
from src.ml_features.domain.enums import (
    ClassificationLabel,
    FeatureDatasetRunStatus,
    FeatureExclusionReason,
)
from src.news.domain.enums import PublicationTimestampQuality
from src.news.domain.time import utc_now
from src.reactions.application.ports import ReactionRepository
from src.reactions.domain.entities import NewsMarketReaction
from src.reactions.domain.enums import BenchmarkAdjustmentStatus, ReactionPointStatus

EXCHANGE_TIMEZONE = ZoneInfo("Europe/Moscow")
LOOKBACK_MINUTES = 120

_FACT_NAMES: dict[FinancialMetric, str] = {
    FinancialMetric.REVENUE: "revenue",
    FinancialMetric.NET_PROFIT: "net_profit",
    FinancialMetric.EBITDA: "ebitda",
    FinancialMetric.OPERATING_PROFIT: "operating_profit",
    FinancialMetric.FREE_CASH_FLOW: "fcf",
    FinancialMetric.CAPEX: "capex",
    FinancialMetric.NET_DEBT: "net_debt",
    FinancialMetric.DIVIDEND_PER_SHARE: "dividend_per_share",
    FinancialMetric.DIVIDEND_TOTAL: "total_dividend",
    FinancialMetric.PRODUCTION_VOLUME: "production",
    FinancialMetric.CONTRACT_VALUE: "contract_value",
    FinancialMetric.OWNERSHIP_PERCENT: "ownership_pct",
}
_CHANGE_NAMES: dict[FinancialMetric, str] = {
    FinancialMetric.NET_PROFIT: "net_profit_change_pct",
    FinancialMetric.REVENUE: "revenue_change_pct",
    FinancialMetric.EBITDA: "ebitda_change_pct",
    FinancialMetric.DIVIDEND_PER_SHARE: "dividend_change_pct",
    FinancialMetric.DIVIDEND_TOTAL: "dividend_change_pct",
    FinancialMetric.PRODUCTION_VOLUME: "production_change_pct",
}
_ROLE_PRIORITY = {
    FactRole.ACTUAL: 0,
    FactRole.FORECAST: 1,
    FactRole.TARGET: 2,
    FactRole.PREVIOUS: 3,
    FactRole.CONSENSUS: 4,
    FactRole.CHANGE: 5,
    FactRole.UNKNOWN: 6,
}


class BuildMlFeatureDataset:
    def __init__(
        self,
        *,
        repository: MlFeatureRepository,
        event_repository: EventAnalysisRepository,
        reaction_repository: ReactionRepository,
        point_in_time_builder: PointInTimeFeatureBuilder | None = None,
    ) -> None:
        self._repository = repository
        self._event_repository = event_repository
        self._reaction_repository = reaction_repository
        self._point_in_time = point_in_time_builder or PointInTimeFeatureBuilder()

    async def execute(
        self,
        *,
        config: FeatureDatasetConfig,
        git_sha: str,
        dry_run: bool,
        generated_at: datetime | None = None,
    ) -> FeatureDatasetBuildResult:
        normalized = config.normalized()
        run = MlFeatureDatasetRun.start(normalized, git_sha=git_sha)
        if not dry_run:
            run = await self._repository.create_run(run)
        rows: list[FeatureDatasetRow] = []
        exclusions: list[FeatureExclusion] = []
        candidates: list[FeatureCandidate] = []
        eligible_count = 0
        failed_count = 0
        generated = generated_at or utc_now()
        try:
            candidates = await self._repository.list_candidates(normalized)
            for candidate in candidates:
                static_reason = await self._static_exclusion(candidate, normalized)
                if isinstance(static_reason, FeatureExclusion):
                    exclusions.append(static_reason)
                    continue
                analysis, reaction = static_reason
                eligible_count += 1
                instrument_match = candidate.matches[0]
                try:
                    security_candles = await self._repository.list_security_candles_as_of(
                        instrument_id=instrument_match.instrument.id,
                        as_of=candidate.news.published_at,
                        lookback_minutes=LOOKBACK_MINUTES,
                    )
                    benchmark_candles = await self._repository.list_benchmark_candles_as_of(
                        benchmark_code=normalized.benchmark_code,
                        as_of=candidate.news.published_at,
                        lookback_minutes=LOOKBACK_MINUTES,
                    )
                    if not security_candles:
                        exclusions.append(
                            _exclude(candidate, FeatureExclusionReason.MARKET_DATA_MISSING)
                        )
                        continue
                    if not benchmark_candles:
                        exclusions.append(
                            _exclude(candidate, FeatureExclusionReason.BENCHMARK_DATA_MISSING)
                        )
                        continue
                    security = self._point_in_time.build(
                        candles=security_candles,
                        as_of=candidate.news.published_at,
                        prefix="security",
                    )
                    benchmark = self._point_in_time.build(
                        candles=benchmark_candles,
                        as_of=candidate.news.published_at,
                        prefix="imoex",
                    )
                except PointInTimeViolationError as exc:
                    exclusions.append(
                        _exclude(
                            candidate,
                            FeatureExclusionReason.POINT_IN_TIME_VIOLATION,
                            detail=str(exc),
                        )
                    )
                    continue
                labels = build_labels(reaction, normalized)
                row = build_row(
                    candidate=candidate,
                    analysis=analysis,
                    security=security,
                    benchmark=benchmark,
                    labels=labels,
                    config=normalized,
                    generated_at=generated,
                )
                assert_label_separation(row)
                rows.append(row)
        except Exception as exc:
            failed_count += 1
            failed_run = run.finish(
                status=FeatureDatasetRunStatus.FAILED,
                candidate_count=len(candidates),
                eligible_count=eligible_count,
                built_count=len(rows),
                excluded_count=len(exclusions),
                failed_count=failed_count,
                error=str(exc)[:2000],
            )
            if not dry_run:
                await self._repository.finish_run(failed_run)
            raise
        status = (
            FeatureDatasetRunStatus.PARTIAL
            if exclusions or failed_count
            else FeatureDatasetRunStatus.SUCCEEDED
        )
        finished = run.finish(
            status=status,
            candidate_count=len(candidates),
            eligible_count=eligible_count,
            built_count=len(rows),
            excluded_count=len(exclusions),
            failed_count=failed_count,
        )
        if not dry_run:
            finished = await self._repository.finish_run(finished)
        return FeatureDatasetBuildResult(rows=rows, exclusions=exclusions, run=finished)

    async def _static_exclusion(
        self,
        candidate: FeatureCandidate,
        config: FeatureDatasetConfig,
    ) -> FeatureExclusion | tuple[NewsEventAnalysis, NewsMarketReaction]:
        if candidate.news.publication_timestamp_quality != PublicationTimestampQuality.EXACT:
            return _exclude(candidate, FeatureExclusionReason.TIMESTAMP_NOT_EXACT)
        if not candidate.matches:
            return _exclude(candidate, FeatureExclusionReason.NO_INSTRUMENT_MATCH)
        if len(candidate.matches) != 1 or candidate.matches[0].match.is_ambiguous:
            return _exclude(candidate, FeatureExclusionReason.AMBIGUOUS_INSTRUMENT)
        analysis = await self._event_repository.get_by_news_id(
            news_id=candidate.news.id,
            analysis_version=config.event_analysis_version,
        )
        if analysis is None:
            return _exclude(candidate, FeatureExclusionReason.NO_EVENT_ANALYSIS)
        reactions = await self._reaction_repository.get_news_reactions(
            news_id=candidate.news.id,
            reaction_version=config.reaction_version,
        )
        reaction = next(
            (
                item
                for item in reactions
                if item.instrument_id == candidate.matches[0].instrument.id
            ),
            None,
        )
        if reaction is None:
            return _exclude(candidate, FeatureExclusionReason.NO_REACTION_LABEL)
        valid_horizons = _valid_label_horizons(reaction)
        if config.require_label_horizon is not None:
            if config.require_label_horizon not in valid_horizons:
                reason = (
                    FeatureExclusionReason.BENCHMARK_DATA_MISSING
                    if _security_label_exists(reaction, config.require_label_horizon)
                    else FeatureExclusionReason.NO_REACTION_LABEL
                )
                return _exclude(candidate, reason)
        elif not valid_horizons:
            reason = (
                FeatureExclusionReason.BENCHMARK_DATA_MISSING
                if any(point.status == ReactionPointStatus.AVAILABLE for point in reaction.points)
                else FeatureExclusionReason.NO_REACTION_LABEL
            )
            return _exclude(candidate, reason)
        return analysis, reaction


def build_row(
    *,
    candidate: FeatureCandidate,
    analysis: NewsEventAnalysis,
    security: PointInTimeMarketFeatures,
    benchmark: PointInTimeMarketFeatures,
    labels: dict[str, Any],
    config: FeatureDatasetConfig,
    generated_at: datetime,
) -> FeatureDatasetRow:
    instrument = candidate.matches[0].instrument
    features = build_event_fact_features(analysis, config.fact_extractor_version)
    features.update(_text_features(candidate.news.title, candidate.news.raw_content))
    features.update(_time_features(candidate.news.published_at))
    features.update(_market_features(security, benchmark))
    missing = sorted(set(security.missing + benchmark.missing))
    metadata: dict[str, Any] = {
        "news_id": candidate.news.id,
        "instrument_id": instrument.id,
        "ticker": instrument.ticker,
        "published_at": candidate.news.published_at,
        "timestamp_quality": candidate.news.publication_timestamp_quality.value,
        "source": candidate.source_code,
        "source_item_id": candidate.source_item_id,
        "dataset_version": config.dataset_version,
        "feature_version": config.feature_version,
        "event_analysis_version": config.event_analysis_version,
        "fact_extractor_version": config.fact_extractor_version,
        "reaction_version": config.reaction_version,
        "market_context_version": config.market_context_version,
        "benchmark_code": config.benchmark_code,
        "generated_at": generated_at,
        "ai_analysis_available": False,
        "ai_analyzer_version": None,
        "ai_model": None,
    }
    quality: dict[str, Any] = {
        "missing_features": missing,
        "market_data_complete": not security.missing,
        "benchmark_data_complete": not benchmark.missing,
        "security_observation_end_at": security.last_observation_end_at,
        "benchmark_observation_end_at": benchmark.last_observation_end_at,
        "point_in_time_cutoff": candidate.news.published_at,
        "classification_policy": (
            None if config.classification_threshold is None else "RESEARCH_DEFAULT_NOT_CALIBRATED"
        ),
        "classification_threshold": config.classification_threshold,
    }
    return FeatureDatasetRow(
        metadata=metadata,
        features=features,
        labels=labels,
        quality=quality,
    )


def build_event_fact_features(
    analysis: NewsEventAnalysis,
    fact_extractor_version: str,
) -> dict[str, Any]:
    event_counts = Counter(event.event_type for event in analysis.events)
    features: dict[str, Any] = {
        "primary_event_type": analysis.primary_event_type.value,
        "event_count": len(analysis.events),
        "has_financial_results": bool(event_counts[EventType.FINANCIAL_RESULTS]),
        "has_dividend": bool(event_counts[EventType.DIVIDEND]),
        "has_guidance": bool(event_counts[EventType.GUIDANCE]),
        "has_ma": bool(event_counts[EventType.MERGER_ACQUISITION]),
        "has_production_update": bool(event_counts[EventType.PRODUCTION_UPDATE]),
        "has_sanctions": bool(event_counts[EventType.SANCTIONS]),
        "has_regulatory_action": bool(event_counts[EventType.REGULATORY_ACTION]),
        "has_other": bool(event_counts[EventType.OTHER] or event_counts[EventType.UNKNOWN]),
    }
    for event_type in EventType:
        features[f"event_type_{event_type.value.lower()}"] = bool(event_counts[event_type])
    facts = [
        fact
        for fact in analysis.financial_facts
        if fact.extractor_version == fact_extractor_version
    ]
    features["fact_count"] = len(facts)
    for metric, name in _FACT_NAMES.items():
        metric_facts = [fact for fact in facts if fact.metric == metric]
        selected = select_metric_fact(metric_facts)
        features[f"has_{name}"] = selected is not None
        features[f"{name}_value"] = None if selected is None else selected.normalized_value
        features[f"{name}_unit"] = None if selected is None else selected.unit.value
        features[f"{name}_currency"] = None if selected is None else selected.currency.value
        features[f"{name}_scale"] = None if selected is None else selected.scale.value
        features[f"{name}_role"] = None if selected is None else selected.fact_role.value
    for feature_name in sorted(set(_CHANGE_NAMES.values())):
        change_name = feature_name.removesuffix("_pct")
        features[feature_name] = None
        features[f"{change_name}_direction"] = None
        features[f"{change_name}_unit"] = None
        features[f"{change_name}_comparison_type"] = None
    for feature_name in sorted(set(_CHANGE_NAMES.values())):
        candidates = [
            fact
            for fact in facts
            if _CHANGE_NAMES.get(fact.metric) == feature_name
            and fact.change_unit == FactUnit.PERCENT
            and fact.change_value is not None
            and fact.change_direction != ChangeDirection.UNKNOWN
        ]
        selected_change = select_metric_fact(candidates)
        if selected_change is not None:
            assert selected_change.change_unit is not None
            change_name = feature_name.removesuffix("_pct")
            features[feature_name] = _signed_change(selected_change)
            features[f"{change_name}_direction"] = selected_change.change_direction.value
            features[f"{change_name}_unit"] = selected_change.change_unit.value
            features[f"{change_name}_comparison_type"] = selected_change.comparison_type.value
    guidance_facts = [
        fact for fact in facts if fact.fact_role in {FactRole.FORECAST, FactRole.TARGET}
    ]
    features["guidance_fact_count"] = len(guidance_facts)
    features["guidance_up_count"] = sum(
        fact.change_direction == ChangeDirection.UP for fact in guidance_facts
    )
    features["guidance_down_count"] = sum(
        fact.change_direction == ChangeDirection.DOWN for fact in guidance_facts
    )
    features["guidance_unchanged_count"] = sum(
        fact.change_direction == ChangeDirection.UNCHANGED for fact in guidance_facts
    )
    features["dividend_per_share"] = features["dividend_per_share_value"]
    features["dividend_role"] = features["dividend_per_share_role"]
    return features


def select_metric_fact(facts: list[ExtractedFinancialFact]) -> ExtractedFinancialFact | None:
    if not facts:
        return None
    return sorted(
        facts,
        key=lambda fact: (
            _ROLE_PRIORITY[fact.fact_role],
            -(fact.date_to.toordinal() if fact.date_to else 0),
            -(fact.year or 0),
            -(fact.quarter or 0),
            -(fact.month or 0),
            -fact.confidence,
            fact.start_position,
            str(fact.id),
        ),
    )[0]


def build_labels(
    reaction: NewsMarketReaction,
    config: FeatureDatasetConfig,
) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    points = {point.horizon_minutes: point for point in reaction.points}
    for horizon in LABEL_HORIZONS_MINUTES:
        point = points.get(horizon)
        adjustment = None if point is None else point.benchmark_adjustment
        if (
            point is not None
            and point.status == ReactionPointStatus.AVAILABLE
            and adjustment is not None
            and adjustment.status == BenchmarkAdjustmentStatus.AVAILABLE
        ):
            available = True
            security_simple = point.simple_return
            benchmark_simple = adjustment.simple_return
            abnormal_simple = adjustment.abnormal_simple_return
            security_log = point.log_return
            benchmark_log = adjustment.log_return
            abnormal_log = adjustment.abnormal_log_return
        else:
            available = False
            security_simple = None
            benchmark_simple = None
            abnormal_simple = None
            security_log = None
            benchmark_log = None
            abnormal_log = None
        labels[f"{horizon}m"] = {
            "available": available,
            "security_simple_return": security_simple,
            "benchmark_simple_return": benchmark_simple,
            "abnormal_simple_return": abnormal_simple,
            "security_log_return": security_log,
            "benchmark_log_return": benchmark_log,
            "abnormal_log_return": abnormal_log,
            "classification": classify_abnormal_return(
                abnormal_simple,
                threshold=config.classification_threshold,
            ),
        }
    return labels


def classify_abnormal_return(
    value: Decimal | None,
    *,
    threshold: Decimal | None,
) -> str | None:
    if value is None or threshold is None:
        return None
    if value > threshold:
        return ClassificationLabel.UP.value
    if value < -threshold:
        return ClassificationLabel.DOWN.value
    return ClassificationLabel.FLAT.value


def _market_features(
    security: PointInTimeMarketFeatures,
    benchmark: PointInTimeMarketFeatures,
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for horizon in FEATURE_HORIZONS_MINUTES:
        security_return = security.returns[horizon]
        benchmark_return = benchmark.returns[horizon]
        features[f"pre_return_{horizon}m"] = security_return
        features[f"pre_log_return_{horizon}m"] = security.log_returns[horizon]
        features[f"imoex_pre_return_{horizon}m"] = benchmark_return
        features[f"imoex_pre_log_return_{horizon}m"] = benchmark.log_returns[horizon]
        features[f"pre_abnormal_return_{horizon}m"] = (
            None
            if security_return is None or benchmark_return is None
            else security_return - benchmark_return
        )
    for horizon in (15, 30, 60):
        features[f"realized_volatility_{horizon}m"] = security.realized_volatility[horizon]
    features["volume_last_1m"] = security.volume_last_1m
    for horizon in (5, 15, 60):
        features[f"volume_sum_{horizon}m"] = security.volume_sums[horizon]
    features["volume_ratio_5m_vs_60m"] = security.volume_ratio_5m_vs_60m
    return features


def _text_features(title: str, content: str) -> dict[str, int]:
    combined = f"{title}\n{content}"
    return {
        "title_length": len(title),
        "content_length": len(content),
        "word_count": len(re.findall(r"\b\w+\b", combined, flags=re.UNICODE)),
        "number_count": len(re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?", combined)),
        "percentage_count": len(re.findall(r"\d+(?:[.,]\d+)?\s*%", combined)),
        "currency_mention_count": len(
            re.findall(
                "\\b(?:RUB|USD|EUR|CNY|\\u0440\\u0443\\u0431\\w*|"
                "\\u0434\\u043e\\u043b\\u043b\\u0430\\u0440\\w*|"
                "\\u0435\\u0432\\u0440\\u043e|\\u044e\\u0430\\u043d\\w*)\\b",
                combined,
                flags=re.IGNORECASE,
            )
        ),
    }


def _time_features(published_at: datetime) -> dict[str, int | bool]:
    local = published_at.astimezone(EXCHANGE_TIMEZONE)
    return {
        "publication_hour_local": local.hour,
        "publication_minute_local": local.minute,
        "day_of_week": local.weekday(),
        "is_weekend": local.weekday() >= 5,
    }


def _signed_change(fact: ExtractedFinancialFact) -> Decimal | None:
    if fact.change_value is None:
        return None
    magnitude = abs(fact.change_value)
    if fact.change_direction == ChangeDirection.UP:
        return magnitude
    if fact.change_direction == ChangeDirection.DOWN:
        return -magnitude
    if fact.change_direction == ChangeDirection.UNCHANGED:
        return Decimal("0")
    return None


def _valid_label_horizons(reaction: NewsMarketReaction) -> set[int]:
    return {
        point.horizon_minutes
        for point in reaction.points
        if point.status == ReactionPointStatus.AVAILABLE
        and point.benchmark_adjustment is not None
        and point.benchmark_adjustment.status == BenchmarkAdjustmentStatus.AVAILABLE
        and point.benchmark_adjustment.abnormal_simple_return is not None
    }


def _security_label_exists(reaction: NewsMarketReaction, horizon: int) -> bool:
    return any(
        point.horizon_minutes == horizon and point.status == ReactionPointStatus.AVAILABLE
        for point in reaction.points
    )


def _exclude(
    candidate: FeatureCandidate,
    reason: FeatureExclusionReason,
    *,
    detail: str | None = None,
) -> FeatureExclusion:
    instrument_id = candidate.matches[0].instrument.id if len(candidate.matches) == 1 else None
    return FeatureExclusion(
        news_id=candidate.news.id,
        instrument_id=instrument_id,
        reason=reason,
        detail=detail,
    )


def assert_label_separation(row: FeatureDatasetRow) -> None:
    forbidden = {
        "security_simple_return",
        "benchmark_simple_return",
        "abnormal_simple_return",
        "security_log_return",
        "benchmark_log_return",
        "abnormal_log_return",
    }
    if forbidden.intersection(row.features):
        raise AssertionError("post-event reaction label leaked into features")
    serialized_features = set(row.features)
    if any(key.startswith("label_") for key in serialized_features):
        raise AssertionError("label-prefixed key leaked into features")
