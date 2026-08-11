from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID
from zoneinfo import ZoneInfo

from src.news.domain.enums import PublicationTimestampQuality

DATASET_VERSION = "free-daily-historical-v1"
LABEL_FAMILY = "DATE_SAFE_DAILY"
REACTION_VERSION = "date-safe-daily-reaction-v1"
FEATURE_VERSION = "ml-daily-features-v1"
INTRADAY_LABEL_FAMILY = "EXACT_INTRADAY"
INTRADAY_REACTION_VERSION = "reaction-v2-benchmark-adjusted"
MAX_HISTORICAL_IMPORT = 500
MAX_SOURCE_SAMPLE = 20
MOEX_TIMEZONE = ZoneInfo("Europe/Moscow")
MINIMUM_COMPLETE_SESSION_CLOSE = time(18, 39)


class SourceAcceptanceStatus(StrEnum):
    COMPLIANT_DATE_SAFE_DAILY = "COMPLIANT_DATE_SAFE_DAILY"
    COMPLIANT_EXACT = "COMPLIANT_EXACT"
    BLOCKED = "BLOCKED"
    REJECTED_PAID = "REJECTED_PAID"


class SourceBlocker(StrEnum):
    ACCESS_POLICY = "ACCESS_POLICY"
    ROBOTS = "ROBOTS"
    UNSTABLE_PAGINATION = "UNSTABLE_PAGINATION"
    NO_PUBLICATION_DATE = "NO_PUBLICATION_DATE"
    NO_TIMEZONE = "NO_TIMEZONE"
    NO_STABLE_ID = "NO_STABLE_ID"
    STORAGE_POLICY = "STORAGE_POLICY"
    TECHNICAL_PARSE = "TECHNICAL_PARSE"
    OTHER = "OTHER"


class DailyExclusionReason(StrEnum):
    NOT_REAL = "NOT_REAL"
    SOURCE_NOT_COMPLIANT = "SOURCE_NOT_COMPLIANT"
    NO_SOURCE_PUBLICATION_DATE = "NO_SOURCE_PUBLICATION_DATE"
    TIMESTAMP_UNKNOWN = "TIMESTAMP_UNKNOWN"
    CRAWL_DATE_SUBSTITUTION = "CRAWL_DATE_SUBSTITUTION"
    DUPLICATE = "DUPLICATE"
    NO_INSTRUMENT_MATCH = "NO_INSTRUMENT_MATCH"
    AMBIGUOUS_INSTRUMENT = "AMBIGUOUS_INSTRUMENT"
    SECURITY_MARKET_DATA_MISSING = "SECURITY_MARKET_DATA_MISSING"
    BENCHMARK_MARKET_DATA_MISSING = "BENCHMARK_MARKET_DATA_MISSING"
    COMMON_SESSION_WINDOW_MISSING = "COMMON_SESSION_WINDOW_MISSING"


class DailyReadiness(StrEnum):
    NOT_READY = "NOT_READY"
    DAILY_PILOT_READY = "DAILY_PILOT_READY"
    DAILY_BASELINE_EXPERIMENT_READY = "DAILY_BASELINE_EXPERIMENT_READY"
    DAILY_BASELINE_TRAINING_READY = "DAILY_BASELINE_TRAINING_READY"


class TemporalSplit(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


@dataclass(frozen=True, slots=True)
class SourceVerification:
    source_code: str
    tickers: tuple[str, ...]
    source_url: str
    status: SourceAcceptanceStatus
    blockers: tuple[SourceBlocker, ...]
    official_or_provenance_verified: bool
    free: bool
    automation_allowed: bool
    stable_identity_verified: bool
    storage_policy_verified: bool
    estimated_items: int
    sample_limit: int
    sampled_items: int
    verified_accessible_items: int
    verified_date_items: int
    verified_exact_items: int
    verified_daily_eligible_items: int
    sampling_order: str
    evidence_note: str

    def validate(self) -> None:
        if (
            not self.source_code.strip()
            or not self.tickers
            or not self.source_url.startswith("https://")
        ):
            raise ValueError("source verification identity is incomplete")
        counts = (
            self.estimated_items,
            self.sample_limit,
            self.sampled_items,
            self.verified_accessible_items,
            self.verified_date_items,
            self.verified_exact_items,
            self.verified_daily_eligible_items,
        )
        if min(counts) < 0:
            raise ValueError("source verification counts cannot be negative")
        if self.sample_limit > MAX_SOURCE_SAMPLE or self.sampled_items > self.sample_limit:
            raise ValueError("source verification sample exceeds the bounded limit")
        if not (
            self.verified_daily_eligible_items
            <= self.verified_date_items
            <= self.verified_accessible_items
            <= self.sampled_items
        ):
            raise ValueError("source verification funnel is inconsistent")
        if self.verified_exact_items > self.verified_date_items:
            raise ValueError("exact items cannot exceed date-verified items")
        if self.status in {
            SourceAcceptanceStatus.COMPLIANT_DATE_SAFE_DAILY,
            SourceAcceptanceStatus.COMPLIANT_EXACT,
        }:
            if self.blockers:
                raise ValueError("compliant source cannot have blockers")
            if not all(
                (
                    self.official_or_provenance_verified,
                    self.free,
                    self.automation_allowed,
                    self.stable_identity_verified,
                    self.storage_policy_verified,
                    self.verified_daily_eligible_items > 0,
                )
            ):
                raise ValueError("compliant source lacks required evidence")
        elif not self.blockers:
            raise ValueError("blocked source requires a specific blocker")
        if self.status == SourceAcceptanceStatus.REJECTED_PAID and self.free:
            raise ValueError("paid source cannot be marked free")
        if not self.sampling_order.strip() or not self.evidence_note.strip():
            raise ValueError("source verification evidence is required")

    def payload(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["status"] = self.status.value
        result["blockers"] = [item.value for item in self.blockers]
        result["tickers"] = list(self.tickers)
        return result


@dataclass(frozen=True, slots=True)
class DailyCandidate:
    news_id: UUID
    source_code: str
    source_item_id: str
    source_url: str
    ticker: str | None
    instrument_id: UUID | None
    publication_date: date | None
    timestamp_quality: PublicationTimestampQuality
    publication_date_from_source: bool
    provenance: str
    source_compliant: bool
    duplicate: bool
    match_count: int
    ambiguous_match: bool
    text_length: int

    def selection_key(self) -> tuple[str, str, date, str, str]:
        return (
            self.ticker or "~",
            self.source_code,
            self.publication_date or date.max,
            self.source_item_id,
            str(self.news_id),
        )


def daily_eligibility(candidate: DailyCandidate) -> DailyExclusionReason | None:
    if candidate.provenance != "REAL":
        return DailyExclusionReason.NOT_REAL
    if not candidate.source_compliant:
        return DailyExclusionReason.SOURCE_NOT_COMPLIANT
    if candidate.publication_date is None:
        return DailyExclusionReason.NO_SOURCE_PUBLICATION_DATE
    if candidate.timestamp_quality == PublicationTimestampQuality.UNKNOWN:
        return DailyExclusionReason.TIMESTAMP_UNKNOWN
    if not candidate.publication_date_from_source:
        return DailyExclusionReason.CRAWL_DATE_SUBSTITUTION
    if candidate.duplicate:
        return DailyExclusionReason.DUPLICATE
    if candidate.match_count == 0 or candidate.instrument_id is None or candidate.ticker is None:
        return DailyExclusionReason.NO_INSTRUMENT_MATCH
    if candidate.match_count != 1 or candidate.ambiguous_match:
        return DailyExclusionReason.AMBIGUOUS_INSTRUMENT
    return None


class CandleLike(Protocol):
    @property
    def end_at(self) -> datetime: ...

    @property
    def close(self) -> Decimal: ...

    @property
    def volume(self) -> Decimal: ...


@dataclass(frozen=True, slots=True)
class SessionClose:
    session_date: date
    observed_at: datetime
    close: Decimal
    volume: Decimal

    def validate(self) -> None:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("session close timestamp must be timezone-aware")
        if self.close <= 0:
            raise ValueError("session close price must be positive")


def collapse_complete_session_closes(candles: Sequence[CandleLike]) -> list[SessionClose]:
    latest: dict[date, CandleLike] = {}
    for candle in candles:
        if candle.end_at.tzinfo is None or candle.end_at.utcoffset() is None:
            raise ValueError("market candle timestamps must be timezone-aware")
        local_end = candle.end_at.astimezone(MOEX_TIMEZONE)
        session_date = local_end.date()
        current = latest.get(session_date)
        if current is None or candle.end_at > current.end_at:
            latest[session_date] = candle
    result: list[SessionClose] = []
    for session_date, candle in sorted(latest.items()):
        local_end = candle.end_at.astimezone(MOEX_TIMEZONE)
        if local_end.time().replace(tzinfo=None) < MINIMUM_COMPLETE_SESSION_CLOSE:
            continue
        close = SessionClose(
            session_date=session_date,
            observed_at=candle.end_at,
            close=candle.close,
            volume=candle.volume,
        )
        close.validate()
        result.append(close)
    return result


@dataclass(frozen=True, slots=True)
class DailyReaction:
    news_id: UUID
    source: str
    ticker: str
    publication_date: date
    timestamp_quality: PublicationTimestampQuality
    baseline_session_date: date
    target_session_date: date
    baseline_security_close: Decimal
    target_security_close: Decimal
    baseline_imoex_close: Decimal
    target_imoex_close: Decimal
    security_return: Decimal
    benchmark_return: Decimal
    abnormal_return: Decimal
    label_family: str = LABEL_FAMILY
    reaction_version: str = REACTION_VERSION

    def validate(self) -> None:
        if not self.baseline_session_date < self.publication_date < self.target_session_date:
            raise ValueError("daily session dates must be strictly outside publication date")
        if self.label_family != LABEL_FAMILY or self.reaction_version != REACTION_VERSION:
            raise ValueError("daily reaction identity is invalid")
        expected_security = self.target_security_close / self.baseline_security_close - Decimal(1)
        expected_benchmark = self.target_imoex_close / self.baseline_imoex_close - Decimal(1)
        if self.security_return != expected_security:
            raise ValueError("security return is inconsistent")
        if self.benchmark_return != expected_benchmark:
            raise ValueError("benchmark return is inconsistent")
        if self.abnormal_return != self.security_return - self.benchmark_return:
            raise ValueError("abnormal return is inconsistent")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "news_id": str(self.news_id),
            "source": self.source,
            "ticker": self.ticker,
            "publication_date": self.publication_date.isoformat(),
            "timestamp_quality": self.timestamp_quality.value,
            "baseline_session_date": self.baseline_session_date.isoformat(),
            "target_session_date": self.target_session_date.isoformat(),
            "baseline_security_close": str(self.baseline_security_close),
            "target_security_close": str(self.target_security_close),
            "baseline_imoex_close": str(self.baseline_imoex_close),
            "target_imoex_close": str(self.target_imoex_close),
            "security_return": str(self.security_return),
            "benchmark_return": str(self.benchmark_return),
            "abnormal_return": str(self.abnormal_return),
            "label_family": self.label_family,
            "reaction_version": self.reaction_version,
        }


def build_daily_reaction(
    candidate: DailyCandidate,
    *,
    security_closes: list[SessionClose],
    benchmark_closes: list[SessionClose],
) -> tuple[DailyReaction | None, DailyExclusionReason | None]:
    eligibility = daily_eligibility(candidate)
    if eligibility is not None:
        return None, eligibility
    assert candidate.publication_date is not None
    assert candidate.ticker is not None
    if not security_closes:
        return None, DailyExclusionReason.SECURITY_MARKET_DATA_MISSING
    if not benchmark_closes:
        return None, DailyExclusionReason.BENCHMARK_MARKET_DATA_MISSING
    security = {item.session_date: item for item in security_closes}
    benchmark = {item.session_date: item for item in benchmark_closes}
    common_dates = sorted(security.keys() & benchmark.keys())
    baseline_dates = [item for item in common_dates if item < candidate.publication_date]
    target_dates = [item for item in common_dates if item > candidate.publication_date]
    if not baseline_dates or not target_dates:
        return None, DailyExclusionReason.COMMON_SESSION_WINDOW_MISSING
    baseline_date = baseline_dates[-1]
    target_date = target_dates[0]
    baseline_security = security[baseline_date]
    target_security = security[target_date]
    baseline_benchmark = benchmark[baseline_date]
    target_benchmark = benchmark[target_date]
    security_return = target_security.close / baseline_security.close - Decimal(1)
    benchmark_return = target_benchmark.close / baseline_benchmark.close - Decimal(1)
    reaction = DailyReaction(
        news_id=candidate.news_id,
        source=candidate.source_code,
        ticker=candidate.ticker,
        publication_date=candidate.publication_date,
        timestamp_quality=candidate.timestamp_quality,
        baseline_session_date=baseline_date,
        target_session_date=target_date,
        baseline_security_close=baseline_security.close,
        target_security_close=target_security.close,
        baseline_imoex_close=baseline_benchmark.close,
        target_imoex_close=target_benchmark.close,
        security_return=security_return,
        benchmark_return=benchmark_return,
        abnormal_return=security_return - benchmark_return,
    )
    reaction.validate()
    return reaction, None


@dataclass(frozen=True, slots=True)
class DailyFeatureRow:
    news_id: UUID
    source: str
    ticker: str
    publication_date: date
    timestamp_quality: PublicationTimestampQuality
    feature_available_at: datetime
    baseline_session_date: date
    features: dict[str, Decimal | str]
    labels: dict[str, Decimal | str]
    feature_version: str = FEATURE_VERSION
    label_family: str = LABEL_FAMILY
    reaction_version: str = REACTION_VERSION

    def validate(self) -> None:
        if self.feature_available_at.astimezone(MOEX_TIMEZONE).date() >= self.publication_date:
            raise ValueError("daily features must be available strictly before publication date")
        forbidden = {"security_return", "benchmark_return", "abnormal_return", "target_close"}
        if forbidden & self.features.keys():
            raise ValueError("reaction labels must not appear in daily features")
        if self.feature_version != FEATURE_VERSION:
            raise ValueError("daily feature version is invalid")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "metadata": {
                "news_id": str(self.news_id),
                "source": self.source,
                "ticker": self.ticker,
                "publication_date": self.publication_date.isoformat(),
                "timestamp_quality": self.timestamp_quality.value,
                "baseline_session_date": self.baseline_session_date.isoformat(),
                "feature_available_at": self.feature_available_at.isoformat(),
                "feature_version": self.feature_version,
                "label_family": self.label_family,
                "reaction_version": self.reaction_version,
            },
            "features": {key: str(value) for key, value in self.features.items()},
            "labels": {key: str(value) for key, value in self.labels.items()},
        }


def build_daily_feature_row(
    reaction: DailyReaction,
    *,
    baseline_security: SessionClose,
    baseline_benchmark: SessionClose,
) -> DailyFeatureRow:
    if baseline_security.session_date != reaction.baseline_session_date:
        raise ValueError("security feature session does not match reaction baseline")
    if baseline_benchmark.session_date != reaction.baseline_session_date:
        raise ValueError("benchmark feature session does not match reaction baseline")
    row = DailyFeatureRow(
        news_id=reaction.news_id,
        source=reaction.source,
        ticker=reaction.ticker,
        publication_date=reaction.publication_date,
        timestamp_quality=reaction.timestamp_quality,
        feature_available_at=max(baseline_security.observed_at, baseline_benchmark.observed_at),
        baseline_session_date=reaction.baseline_session_date,
        features={
            "baseline_security_close": baseline_security.close,
            "baseline_security_volume": baseline_security.volume,
            "baseline_imoex_close": baseline_benchmark.close,
        },
        labels={
            "security_return": reaction.security_return,
            "benchmark_return": reaction.benchmark_return,
            "abnormal_return": reaction.abnormal_return,
        },
    )
    row.validate()
    return row


def select_historical_import(
    candidates: list[DailyCandidate], *, limit: int = MAX_HISTORICAL_IMPORT
) -> list[DailyCandidate]:
    if not 1 <= limit <= MAX_HISTORICAL_IMPORT:
        raise ValueError(f"historical import limit must be between 1 and {MAX_HISTORICAL_IMPORT}")
    selected: dict[tuple[str, str], DailyCandidate] = {}
    for candidate in sorted(candidates, key=DailyCandidate.selection_key):
        if daily_eligibility(candidate) is None:
            selected.setdefault((candidate.source_code, candidate.source_item_id), candidate)
    return list(selected.values())[:limit]


def deterministic_temporal_split(
    rows: list[DailyFeatureRow],
) -> dict[UUID, TemporalSplit]:
    ordered = sorted(rows, key=lambda item: (item.publication_date, str(item.news_id)))
    count = len(ordered)
    if count == 0:
        return {}
    if count < 3:
        return {
            row.news_id: (TemporalSplit.TRAIN if index == 0 else TemporalSplit.TEST)
            for index, row in enumerate(ordered)
        }
    train_end = max(1, int(count * 0.70))
    validation_end = max(train_end + 1, int(count * 0.85))
    validation_end = min(validation_end, count - 1)
    assignments: dict[UUID, TemporalSplit] = {}
    for index, row in enumerate(ordered):
        if index < train_end:
            split = TemporalSplit.TRAIN
        elif index < validation_end:
            split = TemporalSplit.VALIDATION
        else:
            split = TemporalSplit.TEST
        assignments[row.news_id] = split
    return assignments


def daily_readiness(
    feature_ready: int, *, ticker_count: int, source_count: int, month_count: int
) -> dict[str, Any]:
    if feature_ready < 100:
        status = DailyReadiness.NOT_READY
    elif feature_ready < 500:
        status = DailyReadiness.DAILY_PILOT_READY
    elif feature_ready < 1000:
        status = DailyReadiness.DAILY_BASELINE_EXPERIMENT_READY
    else:
        status = DailyReadiness.DAILY_BASELINE_TRAINING_READY
    blockers: list[str] = []
    if ticker_count < 3:
        blockers.append("INSUFFICIENT_TICKER_DIVERSITY")
    if source_count < 2:
        blockers.append("INSUFFICIENT_SOURCE_DIVERSITY")
    if month_count < 6:
        blockers.append("INSUFFICIENT_TIME_DIVERSITY")
    if blockers and status not in {DailyReadiness.NOT_READY, DailyReadiness.DAILY_PILOT_READY}:
        status = DailyReadiness.DAILY_PILOT_READY
    return {
        "status": status.value,
        "feature_ready": feature_ready,
        "diversity_blockers": blockers,
        "predictive_ml_trained": False,
    }


def exclusion_counts(reasons: list[DailyExclusionReason]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for reason in reasons:
        counts[reason.value] += 1
    return dict(sorted(counts.items()))
