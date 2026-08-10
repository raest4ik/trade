from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from src.events.domain.entities import EVENT_ANALYSIS_VERSION, FINANCIAL_FACTS_VERSION
from src.historical_news.domain.enums import ContentStoragePolicy
from src.ml_features.domain.entities import FEATURE_VERSION, LABEL_HORIZONS_MINUTES
from src.news.domain.enums import PublicationTimestampQuality
from src.news.domain.time import ensure_aware_utc
from src.reactions.domain.entities import REACTION_VERSION

CORPUS_VERSION = "reaction-ready-corpus-v1"
UNIVERSE = ("SBER", "SBERP", "GAZP", "LKOH", "ROSN", "NVTK", "YDEX", "T", "VTBR", "GMKN")
REAL_SOURCE_CODES = frozenset({"ROSNEFT_PRESS_RELEASES_RSS"})


class CorpusProvenance(StrEnum):
    REAL = "REAL"
    SYNTHETIC = "SYNTHETIC"
    SEED = "SEED"
    OTHER = "OTHER"


class MatchStatus(StrEnum):
    MATCHED = "MATCHED"
    AMBIGUOUS = "AMBIGUOUS"
    UNMATCHED = "UNMATCHED"


class MarketDataStatus(StrEnum):
    MARKET_DATA_READY = "MARKET_DATA_READY"
    MARKET_DATA_MISSING_SECURITY = "MARKET_DATA_MISSING_SECURITY"
    MARKET_DATA_MISSING_BENCHMARK = "MARKET_DATA_MISSING_BENCHMARK"
    NON_TRADING_EVENT = "NON_TRADING_EVENT"
    OTHER = "OTHER"


class ExclusionReason(StrEnum):
    DATE_ONLY = "DATE_ONLY"
    UNKNOWN_TIMESTAMP = "UNKNOWN_TIMESTAMP"
    UNMATCHED = "UNMATCHED"
    AMBIGUOUS = "AMBIGUOUS"
    NO_EVENT_ANALYSIS = "NO_EVENT_ANALYSIS"
    SECURITY_MARKET_DATA_MISSING = "SECURITY_MARKET_DATA_MISSING"
    IMOEX_DATA_MISSING = "IMOEX_DATA_MISSING"
    NO_VALID_REACTION = "NO_VALID_REACTION"
    STORAGE_POLICY = "STORAGE_POLICY"
    SOURCE_ERROR = "SOURCE_ERROR"
    NON_REAL_PROVENANCE = "NON_REAL_PROVENANCE"


class SourceAuditStatus(StrEnum):
    COMPLIANT = "COMPLIANT"
    NO_COMPLIANT_LIVE_SOURCE_AVAILABLE = "NO_COMPLIANT_LIVE_SOURCE_AVAILABLE"
    TIMESTAMP_DATE_ONLY = "TIMESTAMP_DATE_ONLY"
    SOURCE_ERROR = "SOURCE_ERROR"
    LICENSED_SOURCE_REQUIRED = "LICENSED_SOURCE_REQUIRED"


class ReadinessStatus(StrEnum):
    NOT_READY = "NOT_READY"
    PILOT_ONLY = "PILOT_ONLY"
    BASELINE_EXPERIMENT_READY = "BASELINE_EXPERIMENT_READY"
    BASELINE_TRAINING_READY = "BASELINE_TRAINING_READY"


@dataclass(frozen=True, slots=True)
class SourceAuditEntry:
    tickers: tuple[str, ...]
    issuer: str
    source_url: str
    source_owner: str
    source_kind: str
    https: bool
    historical_depth_observed: str
    timestamp_precision: str
    timezone_semantics: str
    full_text_availability: str
    storage_policy: ContentStoragePolicy
    pagination_archive_capability: str
    robots_access_restrictions: str
    status: SourceAuditStatus
    blocker: str | None = None
    source_code: str | None = None

    def validate(self) -> None:
        if not self.tickers or any(ticker not in UNIVERSE for ticker in self.tickers):
            raise ValueError("source audit tickers must belong to the configured universe")
        parsed = urlparse(self.source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source audit URL must be absolute HTTP(S)")
        if self.https != (parsed.scheme == "https"):
            raise ValueError("source audit HTTPS flag does not match source URL")
        required = (
            self.issuer,
            self.source_owner,
            self.source_kind,
            self.historical_depth_observed,
            self.timestamp_precision,
            self.timezone_semantics,
            self.full_text_availability,
            self.pagination_archive_capability,
            self.robots_access_restrictions,
        )
        if any(not value.strip() for value in required):
            raise ValueError("source audit fields must not be empty")
        if self.status == SourceAuditStatus.COMPLIANT:
            if self.source_code not in REAL_SOURCE_CODES:
                raise ValueError("compliant source must have an approved REAL source code")
            if not self.https:
                raise ValueError("compliant source must use HTTPS")
        elif not self.blocker:
            raise ValueError("non-compliant source audit entry must document a blocker")

    def payload(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["storage_policy"] = self.storage_policy.value
        payload["status"] = self.status.value
        payload["tickers"] = list(self.tickers)
        return payload


@dataclass(frozen=True, slots=True)
class MarketBackfillWindow:
    ticker: str
    date_from: datetime
    date_to: datetime
    interval_minutes: int = 1
    benchmark_code: str = "IMOEX"

    def __post_init__(self) -> None:
        start = ensure_aware_utc(self.date_from, "date_from")
        end = ensure_aware_utc(self.date_to, "date_to")
        if self.ticker not in UNIVERSE:
            raise ValueError("market window ticker is outside the configured universe")
        if end < start:
            raise ValueError("market window date_to must not be before date_from")
        if end - start > timedelta(days=14):
            raise ValueError("market window must remain bounded to 14 days")
        if self.interval_minutes != 1:
            raise ValueError("reaction-ready corpus requires one-minute candles")

    def payload(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "benchmark_code": self.benchmark_code,
            "interval_minutes": self.interval_minutes,
            "date_from": self.date_from.isoformat(),
            "date_to": self.date_to.isoformat(),
        }


def classify_provenance(source_code: str, source_name: str | None = None) -> CorpusProvenance:
    code = source_code.strip().upper()
    name = "" if source_name is None else source_name.strip().lower()
    if code in REAL_SOURCE_CODES:
        return CorpusProvenance.REAL
    if code.startswith(("SEED", "BATCH_001")) or name == "seed-dataset":
        return CorpusProvenance.SEED
    if code.startswith(("TEST_", "SYNTHETIC_")) or name.startswith("synthetic"):
        return CorpusProvenance.SYNTHETIC
    return CorpusProvenance.OTHER


def match_status(match_count: int, has_ambiguous_match: bool) -> MatchStatus:
    if match_count == 0:
        return MatchStatus.UNMATCHED
    if match_count != 1 or has_ambiguous_match:
        return MatchStatus.AMBIGUOUS
    return MatchStatus.MATCHED


def timestamp_exclusion(quality: PublicationTimestampQuality) -> ExclusionReason | None:
    if quality == PublicationTimestampQuality.DATE_ONLY:
        return ExclusionReason.DATE_ONLY
    if quality == PublicationTimestampQuality.UNKNOWN:
        return ExclusionReason.UNKNOWN_TIMESTAMP
    return None


def plan_market_windows(
    publications: list[tuple[str, datetime]],
    *,
    pre_safety_days: int = 3,
    post_safety_days: int = 7,
) -> list[MarketBackfillWindow]:
    if not 1 <= pre_safety_days <= 7 or not 1 <= post_safety_days <= 7:
        raise ValueError("market window safety margins must be between one and seven days")
    grouped: dict[str, list[datetime]] = {}
    for ticker, published_at in publications:
        normalized_ticker = ticker.strip().upper()
        if normalized_ticker not in UNIVERSE:
            continue
        grouped.setdefault(normalized_ticker, []).append(
            ensure_aware_utc(published_at, "published_at")
        )
    windows: list[MarketBackfillWindow] = []
    for ticker, timestamps in sorted(grouped.items()):
        earliest = min(timestamps)
        latest = max(timestamps)
        start = datetime.combine(
            (earliest - timedelta(days=pre_safety_days)).date(),
            datetime.min.time(),
            tzinfo=UTC,
        )
        end = datetime.combine(
            (latest + timedelta(days=post_safety_days)).date(),
            datetime.max.time(),
            tzinfo=UTC,
        )
        windows.append(MarketBackfillWindow(ticker=ticker, date_from=start, date_to=end))
    return windows


def readiness_status(real_feature_rows: int) -> ReadinessStatus:
    if real_feature_rows < 100:
        return ReadinessStatus.NOT_READY
    if real_feature_rows < 500:
        return ReadinessStatus.PILOT_ONLY
    if real_feature_rows < 1000:
        return ReadinessStatus.BASELINE_EXPERIMENT_READY
    return ReadinessStatus.BASELINE_TRAINING_READY


def source_audit_payload(entries: tuple[SourceAuditEntry, ...]) -> dict[str, Any]:
    audited = {ticker for entry in entries for ticker in entry.tickers}
    if audited != set(UNIVERSE):
        missing = sorted(set(UNIVERSE) - audited)
        extra = sorted(audited - set(UNIVERSE))
        raise ValueError(f"source audit universe mismatch: missing={missing}, extra={extra}")
    return {
        "schema_version": "historical-news-source-audit-v1",
        "audited_at": datetime.now(UTC).isoformat(),
        "universe": list(UNIVERSE),
        "approved_real_source_codes": sorted(REAL_SOURCE_CODES),
        "sources": [entry.payload() for entry in entries],
    }


VERSION_SUMMARY = {
    "event_analysis_version": EVENT_ANALYSIS_VERSION,
    "fact_extractor_version": FINANCIAL_FACTS_VERSION,
    "reaction_version": REACTION_VERSION,
    "feature_version": FEATURE_VERSION,
    "label_horizons_minutes": list(LABEL_HORIZONS_MINUTES),
}
