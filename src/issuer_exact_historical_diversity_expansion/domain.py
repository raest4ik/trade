from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any, cast
from zoneinfo import ZoneInfo

ARTIFACT_VERSION = "issuer-exact-historical-diversity-expansion-v1"
DEFAULT_READINESS_AUDIT_ROOT = "artifacts/exact-dataset-readiness-audit-v1"
EXPECTED_READINESS_AUDIT_SHA = "fa96c9e71534e8d4cc27201e7f17c8cf4c40a5af90075d2031044e86d80758f2"
EXPECTED_RULES_V3_FINGERPRINT = "3510511d1f7b3ce02a4efa245816b9422e6014088f1595b0339dcfd5be9e7f06"
FUTURE_EVENT_HOLDOUT_START = date(2026, 8, 11)
HORIZONS = ("1m", "5m", "15m", "30m", "60m")
MOSCOW_TZ = ZoneInfo("Europe/Moscow")
MVID_INSTRUMENT_UID = "cf1c6158-a303-43ac-89eb-9b1db8f96043"
IMOEX_INSTRUMENT_UID = "f7c8fca9-efe4-4d1b-97e9-a2b48babbcc1"


class SourceMechanism(StrEnum):
    RSS = "RSS"
    ATOM = "ATOM"
    PUBLIC_JSON = "PUBLIC_JSON"
    PUBLIC_HTML_ARCHIVE = "PUBLIC_HTML_ARCHIVE"
    PUBLIC_IR_NEWS_ARCHIVE = "PUBLIC_IR_NEWS_ARCHIVE"
    OTHER_OFFICIAL_PUBLIC = "OTHER_OFFICIAL_PUBLIC"


class CandidateStatus(StrEnum):
    NEW_EXACT_HISTORICAL_CAPABLE = "NEW_EXACT_HISTORICAL_CAPABLE"
    DATE_ONLY = "DATE_ONLY"
    TECHNICAL_BLOCKER = "TECHNICAL_BLOCKER"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    ALREADY_COVERED = "ALREADY_COVERED"
    NOT_ISSUER_ORIGINATED = "NOT_ISSUER_ORIGINATED"
    ORIGIN_AMBIGUOUS = "ORIGIN_AMBIGUOUS"


class FinalDecision(StrEnum):
    ISSUER_DIVERSITY_GAIN_STRONG = "ISSUER_DIVERSITY_GAIN_STRONG"
    ISSUER_DIVERSITY_GAIN_MODEST = "ISSUER_DIVERSITY_GAIN_MODEST"
    STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING = "STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING"
    MARKET_HISTORY_UNAVAILABLE_AFTER_READONLY_ACQUISITION = (
        "MARKET_HISTORY_UNAVAILABLE_AFTER_READONLY_ACQUISITION"
    )
    SESSION_ALIGNMENT_BLOCKERS_DOMINATE = "SESSION_ALIGNMENT_BLOCKERS_DOMINATE"
    EVENT_COUNT_GAIN_WITHOUT_DIVERSITY = "EVENT_COUNT_GAIN_WITHOUT_DIVERSITY"
    HISTORICAL_ISSUER_SOURCE_YIELD_TOO_LOW = "HISTORICAL_ISSUER_SOURCE_YIELD_TOO_LOW"
    EXACT_HISTORICAL_SOURCE_AVAILABILITY_EXHAUSTED = (
        "EXACT_HISTORICAL_SOURCE_AVAILABILITY_EXHAUSTED"
    )
    DATA_QUALITY_REVIEW_REQUIRED = "DATA_QUALITY_REVIEW_REQUIRED"


FORBIDDEN_SELECTION_KEYS = {
    "abnormal_return",
    "accuracy",
    "auc",
    "backtest",
    "f1",
    "future_return",
    "label",
    "market_reaction",
    "model_performance",
    "outcome",
    "precision",
    "profit",
    "reaction",
    "recall",
    "rmse",
    "sharpe",
    "target",
    "target_return",
}
_MVIDEO_TIMESTAMP = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})\b")


@dataclass(frozen=True, slots=True)
class CandidateSource:
    ticker: str
    issuer: str
    official_domain: str
    source_url: str
    source_family: str
    source_id: str
    mechanism: SourceMechanism
    status: CandidateStatus
    event_origin: str
    exact_timestamp_supported: bool
    publication_material_available: bool
    historical_depth_estimate: str
    historical_depth_score: int
    parser_profile: str
    current_feature_ready_count: int = 0
    current_feature_ready_share: str = "0.000000"
    already_in_corpus: bool = False
    ticker_attribution_quality: str = "UNVERIFIED"
    zero_cost_public: bool = True
    source_selection_notes: str = ""
    instrument_uid: str | None = None
    figi: str | None = None
    evidence_urls: tuple[str, ...] = field(default_factory=tuple)

    def payload(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "issuer": self.issuer,
            "official_domain": self.official_domain,
            "source_url": self.source_url,
            "source_family": self.source_family,
            "source_id": self.source_id,
            "mechanism": self.mechanism.value,
            "status": self.status.value,
            "event_origin": self.event_origin,
            "exact_timestamp_supported": self.exact_timestamp_supported,
            "publication_material_available": self.publication_material_available,
            "historical_depth_estimate": self.historical_depth_estimate,
            "historical_depth_score": self.historical_depth_score,
            "parser_profile": self.parser_profile,
            "current_feature_ready_count": self.current_feature_ready_count,
            "current_feature_ready_share": self.current_feature_ready_share,
            "already_in_corpus": self.already_in_corpus,
            "ticker_attribution_quality": self.ticker_attribution_quality,
            "zero_cost_public": self.zero_cost_public,
            "source_selection_notes": self.source_selection_notes,
            "instrument_uid": self.instrument_uid,
            "figi": self.figi,
            "evidence_urls": list(self.evidence_urls),
        }


def safety_flags() -> dict[str, bool | int | str]:
    return {
        "RESEARCH_ONLY": True,
        "DATA_COST_RUB": 0,
        "MODEL_TRAINING_PERFORMED": False,
        "TEST_OUTCOME_USED": False,
        "TEST_EVALUATION_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
        "CONFIRMED_SIGNAL": False,
        "RULES_V3_CHANGED": False,
        "QWEN_CHANGED": False,
        "NLP_TUNING_PERFORMED": False,
        "FEATURE_DEFINITION_CHANGED": False,
        "REACTION_METHODOLOGY_CHANGED": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "FUTURE_EVENT_HOLDOUT_USED": False,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "FUTURE_PRICE_LOOKUPS": 0,
        "FUTURE_REACTIONS_COMPUTED": 0,
        "FUTURE_TARGETS_COMPUTED": 0,
        "FUTURE_OUTCOMES_READ": 0,
        "FUTURE_TARGETS_READ": 0,
        "SOURCE_SELECTION_USED_MARKET_OUTCOMES": False,
        "SOURCE_SELECTION_USED_MODEL_PERFORMANCE": False,
        "MOEX_RISK_PARAMETERS_SELECTED": False,
        "EXCHANGE_ORIGINATED_EVENTS_SELECTED": False,
        "ORIGIN_AMBIGUOUS_EVENTS_SELECTED": False,
        "FORWARD_FILL_USED": False,
        "MOEX_SUBSTITUTION_USED": False,
        "REAL_TRADING_ALLOWED": False,
        "REAL_ORDER_SUBMISSION_ALLOWED": False,
        "REAL_STOP_ORDER_ALLOWED": False,
        "REAL_MONEY_MOVEMENT_ALLOWED": False,
        "BROKER_ACCOUNT_MUTATION_ALLOWED": False,
        "MARGIN_TRADING_ALLOWED": False,
        "LIVE_EXECUTION_ALLOWED": False,
        "PAPER_TRADING_ALLOWED": False,
        "SANDBOX_ORDER_SUBMISSION_ALLOWED": False,
    }


def parse_local_timestamp(value: str) -> datetime | None:
    match = _MVIDEO_TIMESTAMP.search(value)
    if match is None:
        return None
    day, month, year, hour, minute = (int(part) for part in match.groups())
    return datetime(year, month, day, hour, minute)


def parse_verified_exact_timestamp(value: str, timezone_value: str | None) -> datetime | None:
    local = parse_local_timestamp(value)
    if local is None or timezone_value is None:
        return None
    normalized = timezone_value.strip().upper().replace(" ", "")
    if normalized in {"MSK", "EUROPE/MOSCOW", "MOSCOWTIME"}:
        return local.replace(tzinfo=MOSCOW_TZ).astimezone(UTC)
    if normalized in {"UTC", "Z", "GMT"}:
        return local.replace(tzinfo=UTC).astimezone(UTC)
    offset = re.fullmatch(r"(?:UTC|GMT)?([+-])(\d{2}):?(\d{2})", normalized)
    if offset is None:
        return None
    sign = 1 if offset.group(1) == "+" else -1
    hours = int(offset.group(2))
    minutes = int(offset.group(3))
    tz = timezone(sign * timedelta(hours=hours, minutes=minutes))
    return local.replace(tzinfo=tz).astimezone(UTC)


def publication_material(snapshot: dict[str, Any]) -> str:
    pieces = [
        str(snapshot.get("title") or ""),
        str(snapshot.get("description") or ""),
        str(snapshot.get("content") or ""),
    ]
    return "\n\n".join(piece.strip() for piece in pieces if piece.strip())


def publication_material_sha(snapshot: dict[str, Any]) -> str:
    return sha256_payload(publication_material(snapshot))


def validate_selection_payload(payload: Any) -> None:
    if isinstance(payload, Mapping):
        typed = cast("Mapping[Any, Any]", payload)
        for key, value in typed.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_SELECTION_KEYS:
                raise ValueError(f"SOURCE_SELECTION_FORBIDDEN_FIELD:{key}")
            validate_selection_payload(value)
    elif isinstance(payload, Sequence) and not isinstance(payload, str | bytes | bytearray):
        sequence = cast("Sequence[Any]", payload)
        for item in sequence:
            validate_selection_payload(item)


def share(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000000"
    return fmt_decimal(Decimal(numerator) / Decimal(denominator))


def hhi(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    if not total:
        return "0.000000"
    value = sum(
        ((Decimal(count) / Decimal(total)) ** 2 for count in counts.values()),
        Decimal("0"),
    )
    return fmt_decimal(value)


def effective_count(counts: dict[str, int]) -> str:
    value = Decimal(hhi(counts))
    if value == 0:
        return "0.000000"
    return fmt_decimal(Decimal("1") / value)


def top_share(counts: dict[str, int], top_n: int) -> str:
    total = sum(counts.values())
    if total == 0:
        return "0.000000"
    return share(sum(sorted(counts.values(), reverse=True)[:top_n]), total)


def fmt_decimal(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.000001'))}"


def sha256_payload(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def artifact_sha(manifest: dict[str, Any]) -> str:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"ARTIFACT_SHA", "created_at", "git_sha"}
    }
    return sha256_payload(core)
