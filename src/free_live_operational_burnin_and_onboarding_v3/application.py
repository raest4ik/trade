from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from src.exact_event_live_official_collection.http_client import BoundedHttpClient, HttpClient
from src.free_live_issuer_accumulation.application import DEFAULT_HISTORICAL_TICKER_SUMMARY_PATH
from src.free_live_issuer_accumulation.domain import (
    DEFAULT_SOURCE_REGISTRY_PATH,
    LiveIssuerSource,
    SourceQualificationStatus,
    live_accumulation_safety_flags,
    sha256_payload,
)
from src.free_live_issuer_accumulation.operation import (
    DEFAULT_OPERATION_ARTIFACT_ROOT,
    build_operation_status,
    load_collector_state,
    verify_operation_seal,
)
from src.free_live_issuer_expansion_v2.application import (
    HISTORICAL_ISSUER_TICKERS,
    SourceProbeConfig,
    SourceProbeResult,
    SourceStatus,
    TimestampLevel,
    default_candidate_sources,
    probe_candidate_source,
)
from src.instruments.infrastructure.seed import SEED_INSTRUMENTS, SeedInstrument

ARTIFACT_VERSION = "free-live-operational-burnin-and-onboarding-v3"
BASELINE_LIVE_TICKERS = ("ROSN", "YDEX")
DEFAULT_INSTRUMENT_MAPPING_PATH = Path(
    "artifacts/tinvest-market-universe-raw-v1/instrument-mapping.json"
)
ALLOWED_TARGET_EXCHANGES = ("moex_mrng_evng_e_wknd_dlr",)
APPROVED_BENCHMARK_TICKER = "IMOEX"


class DiversityStatus(StrEnum):
    NO_NEW_TARGET_ELIGIBLE_ISSUERS = "NO_NEW_TARGET_ELIGIBLE_ISSUERS"
    ONE_NEW_TARGET_ELIGIBLE_ISSUER = "ONE_NEW_TARGET_ELIGIBLE_ISSUER"
    TWO_NEW_TARGET_ELIGIBLE_ISSUERS = "TWO_NEW_TARGET_ELIGIBLE_ISSUERS"
    THREE_PLUS_NEW_TARGET_ELIGIBLE_ISSUERS = "THREE_PLUS_NEW_TARGET_ELIGIBLE_ISSUERS"
    FREE_SOURCE_UNIVERSE_EXHAUSTED_FOR_NOW = "FREE_SOURCE_UNIVERSE_EXHAUSTED_FOR_NOW"


@dataclass(frozen=True, slots=True)
class CanonicalTargetInstrument:
    ticker: str
    legal_issuer: str
    primary_board: str
    instrument_type: str

    @classmethod
    def from_seed(cls, instrument: SeedInstrument) -> CanonicalTargetInstrument:
        return cls(
            ticker=instrument.ticker.upper(),
            legal_issuer=instrument.issuer_name,
            primary_board=instrument.primary_board,
            instrument_type=instrument.instrument_type.value,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "legal_issuer": self.legal_issuer,
            "primary_board": self.primary_board,
            "instrument_type": self.instrument_type,
        }


@dataclass(frozen=True, slots=True)
class TargetEligibilityResult:
    source_id: str
    source_ticker: str
    source_legal_issuer: str
    free_official_source_ready: bool
    canonical_instrument: CanonicalTargetInstrument | None
    canonical_mapping_ready: bool
    target_instrument_eligible: bool
    feature_pipeline_compatible: bool
    counts_toward_ml_diversity: bool
    legal_issuer_key_value: str | None
    blocker: str | None
    mapping_confidence: str
    mapping_row: dict[str, Any] | None
    benchmark_path_ready: bool

    def payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_ticker": self.source_ticker,
            "source_legal_issuer": self.source_legal_issuer,
            "FREE_OFFICIAL_SOURCE_READY": self.free_official_source_ready,
            "SOURCE_READY": self.free_official_source_ready,
            "TARGET_INSTRUMENT_ELIGIBLE": self.target_instrument_eligible,
            "CANONICAL_INSTRUMENT_MAPPING_READY": self.canonical_mapping_ready,
            "FEATURE_PIPELINE_COMPATIBLE": self.feature_pipeline_compatible,
            "COUNTS_TOWARD_ML_DIVERSITY": self.counts_toward_ml_diversity,
            "canonical_instrument": None
            if self.canonical_instrument is None
            else self.canonical_instrument.payload(),
            "source_to_canonical_mapping": {
                "source_ticker": self.source_ticker,
                "canonical_ticker": None
                if self.canonical_instrument is None
                else self.canonical_instrument.ticker,
                "mapping_confidence": self.mapping_confidence,
                "exact_ticker_match_required": True,
                "alias_or_suffix_inferred": False,
            },
            "market_data_mapping": self.mapping_row,
            "benchmark_path_ready": self.benchmark_path_ready,
            "legal_issuer_key": self.legal_issuer_key_value,
            "blocker": self.blocker,
        }


@dataclass(frozen=True, slots=True)
class CandidateBacklogEntry:
    issuer: str
    tickers: tuple[str, ...]
    official_url: str
    last_checked: date
    previous_blocker: str
    new_hypothesis: str
    current_status: str
    timestamp_evidence: str | None
    identity_evidence: str | None
    next_possible_free_mechanism: str
    slow_recheck_after_days: int = 7

    def validate(self) -> None:
        if not self.issuer.strip():
            raise ValueError("CANDIDATE_ISSUER_REQUIRED")
        if not self.tickers:
            raise ValueError("CANDIDATE_TICKER_REQUIRED")
        if self.previous_blocker and not self.new_hypothesis.strip():
            raise ValueError("RECHECK_REQUIRES_NEW_HYPOTHESIS")
        if self.slow_recheck_after_days < 1:
            raise ValueError("SLOW_CADENCE_MUST_BE_POSITIVE")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "issuer": self.issuer,
            "tickers": list(self.tickers),
            "official_url": self.official_url,
            "last_checked": self.last_checked.isoformat(),
            "previous_blocker": self.previous_blocker,
            "new_hypothesis": self.new_hypothesis,
            "current_status": self.current_status,
            "timestamp_evidence": self.timestamp_evidence,
            "identity_evidence": self.identity_evidence,
            "next_possible_free_mechanism": self.next_possible_free_mechanism,
            "slow_recheck_after_days": self.slow_recheck_after_days,
        }


def assert_recheck_has_new_hypothesis(
    *,
    previous_url: str,
    previous_mechanism: str,
    candidate_url: str,
    candidate_mechanism: str,
    new_hypothesis: str,
) -> None:
    if not new_hypothesis.strip():
        raise ValueError("RECHECK_REQUIRES_NEW_HYPOTHESIS")
    repeated_probe = previous_url == candidate_url and previous_mechanism == candidate_mechanism
    hypothesis_tokens = set(_normalize_text(new_hypothesis).split())
    if repeated_probe and len(hypothesis_tokens) < 3:
        raise ValueError("RECHECK_REPEATS_URL_AND_METHOD_WITHOUT_MATERIAL_HYPOTHESIS")


def legal_issuer_key(*, ticker: str, legal_issuer: str) -> str:
    ticker_key = ticker.strip().upper()
    share_class_groups = {
        "SBER": "sberbank",
        "SBERP": "sberbank",
        "TATN": "tatneft",
        "TATNP": "tatneft",
    }
    if ticker_key in share_class_groups:
        return share_class_groups[ticker_key]
    normalized = _normalize_text(legal_issuer)
    normalized = re.sub(r"\b(pao|pjsc|ao|jsc|public joint stock company|bank)\b", "", normalized)
    return " ".join(normalized.split())


def canonical_target_registry_from_seed(
    seed_instruments: Sequence[SeedInstrument] = SEED_INSTRUMENTS,
) -> dict[str, CanonicalTargetInstrument]:
    return {
        instrument.ticker.upper(): CanonicalTargetInstrument.from_seed(instrument)
        for instrument in seed_instruments
    }


def load_instrument_mapping_rows(
    path: Path = DEFAULT_INSTRUMENT_MAPPING_PATH,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = _read_json(path)
    rows_object: object = payload.get("instruments", [])
    if not isinstance(rows_object, list):
        return []
    rows = cast("list[object]", rows_object)
    typed_rows: list[dict[str, Any]] = []
    for row_object in rows:
        if isinstance(row_object, dict):
            typed_rows.append(cast("dict[str, Any]", row_object))
    return typed_rows


def evaluate_target_eligibility(
    accepted_sources: Sequence[dict[str, Any]],
    *,
    canonical_registry: dict[str, CanonicalTargetInstrument] | None = None,
    instrument_mapping_rows: Sequence[dict[str, Any]] = (),
    frozen_tickers: Sequence[str] = HISTORICAL_ISSUER_TICKERS,
    allowed_exchanges: Sequence[str] = ALLOWED_TARGET_EXCHANGES,
    benchmark_ticker: str = APPROVED_BENCHMARK_TICKER,
) -> list[TargetEligibilityResult]:
    registry = (
        canonical_target_registry_from_seed() if canonical_registry is None else canonical_registry
    )
    mapping_by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in instrument_mapping_rows:
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        mapping_by_ticker.setdefault(ticker, []).append(dict(row))

    benchmark_rows = mapping_by_ticker.get(benchmark_ticker.upper(), [])
    benchmark_path_ready = any(
        str(row.get("instrument_uid", "")).strip()
        and str(row.get("figi", "")).strip()
        and str(row.get("first_1day_candle_date", "")).strip()
        for row in benchmark_rows
    )
    frozen = {ticker.upper() for ticker in frozen_tickers}
    allowed_exchange_set = {exchange.lower() for exchange in allowed_exchanges}
    results: list[TargetEligibilityResult] = []
    for source in accepted_sources:
        source_ticker = str(source["ticker"]).strip().upper()
        source_issuer = str(source["legal_issuer"])
        canonical = registry.get(source_ticker)
        rows = mapping_by_ticker.get(source_ticker, [])
        row = rows[0] if len(rows) == 1 else None
        canonical_mapping_ready = canonical is not None and row is not None
        target_instrument_eligible = False
        feature_pipeline_compatible = False
        counts_toward_ml_diversity = False
        mapping_confidence = "NONE"
        blocker: str | None = None
        legal_key: str | None = None

        if canonical is None:
            blocker = "CANONICAL_INSTRUMENT_NOT_IN_PROJECT_TARGET_REGISTRY"
        elif len(rows) == 0:
            blocker = "CANONICAL_INSTRUMENT_MAPPING_MISSING"
        elif len(rows) > 1:
            blocker = "AMBIGUOUS_CANONICAL_INSTRUMENT_MAPPING"
        else:
            assert row is not None
            mapping_confidence = "EXACT_CANONICAL_TICKER"
            exchange = str(row.get("exchange", "")).strip().lower()
            class_code = str(row.get("class_code", "")).strip().upper()
            instrument_uid = str(row.get("instrument_uid", "")).strip()
            figi = str(row.get("figi", "")).strip()
            candle_start = str(row.get("first_1day_candle_date", "")).strip()
            if canonical.primary_board.upper() != class_code:
                blocker = "CANONICAL_BOARD_MISMATCH"
            elif exchange not in allowed_exchange_set:
                blocker = "UNSUPPORTED_EXCHANGE_OR_TRADING_VENUE"
            elif not instrument_uid or not figi:
                blocker = "CANONICAL_INSTRUMENT_MAPPING_INCOMPLETE"
            elif not candle_start:
                blocker = "INACTIVE_OR_UNUSABLE_INSTRUMENT"
            elif source_ticker in frozen:
                blocker = "ALREADY_IN_FROZEN_HISTORICAL_TARGET_UNIVERSE"
            elif not benchmark_path_ready:
                blocker = "APPROVED_BENCHMARK_PATH_MISSING"
            else:
                target_instrument_eligible = True
                feature_pipeline_compatible = True
                counts_toward_ml_diversity = True
                legal_key = legal_issuer_key(
                    ticker=canonical.ticker, legal_issuer=canonical.legal_issuer
                )

        if canonical is not None and legal_key is None:
            legal_key = legal_issuer_key(
                ticker=canonical.ticker, legal_issuer=canonical.legal_issuer
            )
        results.append(
            TargetEligibilityResult(
                source_id=str(source.get("source_id", f"{source_ticker}_SOURCE")),
                source_ticker=source_ticker,
                source_legal_issuer=source_issuer,
                free_official_source_ready=True,
                canonical_instrument=canonical,
                canonical_mapping_ready=canonical_mapping_ready,
                target_instrument_eligible=target_instrument_eligible,
                feature_pipeline_compatible=feature_pipeline_compatible,
                counts_toward_ml_diversity=counts_toward_ml_diversity,
                legal_issuer_key_value=legal_key,
                blocker=blocker,
                mapping_confidence=mapping_confidence,
                mapping_row=row,
                benchmark_path_ready=benchmark_path_ready,
            )
        )
    return results


def distinct_new_target_eligible_legal_issuers(
    eligibility_results: Sequence[TargetEligibilityResult],
    *,
    frozen_tickers: Sequence[str] = HISTORICAL_ISSUER_TICKERS,
) -> list[dict[str, str]]:
    frozen = {ticker.upper() for ticker in frozen_tickers}
    unique: dict[str, dict[str, str]] = {}
    for result in eligibility_results:
        if not result.counts_toward_ml_diversity or result.canonical_instrument is None:
            continue
        ticker = result.canonical_instrument.ticker
        if ticker in frozen:
            continue
        key = result.legal_issuer_key_value or legal_issuer_key(
            ticker=ticker, legal_issuer=result.canonical_instrument.legal_issuer
        )
        unique.setdefault(
            key,
            {
                "legal_issuer_key": key,
                "legal_issuer": result.canonical_instrument.legal_issuer,
                "ticker": ticker,
            },
        )
    return sorted(unique.values(), key=lambda row: row["legal_issuer_key"])


def distinct_new_legal_issuers(
    accepted_sources: Sequence[dict[str, Any]],
    *,
    frozen_tickers: Sequence[str] = HISTORICAL_ISSUER_TICKERS,
) -> list[dict[str, str]]:
    eligibility = evaluate_target_eligibility(
        accepted_sources,
        instrument_mapping_rows=load_instrument_mapping_rows(),
        frozen_tickers=frozen_tickers,
    )
    return distinct_new_target_eligible_legal_issuers(eligibility)


def diversity_status(new_legal_issuer_count: int, *, universe_exhausted: bool = False) -> str:
    if universe_exhausted and new_legal_issuer_count < 3:
        return DiversityStatus.FREE_SOURCE_UNIVERSE_EXHAUSTED_FOR_NOW.value
    if new_legal_issuer_count <= 0:
        return DiversityStatus.NO_NEW_TARGET_ELIGIBLE_ISSUERS.value
    if new_legal_issuer_count == 1:
        return DiversityStatus.ONE_NEW_TARGET_ELIGIBLE_ISSUER.value
    if new_legal_issuer_count == 2:
        return DiversityStatus.TWO_NEW_TARGET_ELIGIBLE_ISSUERS.value
    return DiversityStatus.THREE_PLUS_NEW_TARGET_ELIGIBLE_ISSUERS.value


def diversity_eligibility_payload(
    eligibility_results: Sequence[TargetEligibilityResult],
    new_issuers: Sequence[dict[str, str]],
) -> dict[str, Any]:
    source_ready = [result for result in eligibility_results if result.free_official_source_ready]
    target_eligible = [
        result for result in eligibility_results if result.target_instrument_eligible
    ]
    feature_compatible = [
        result for result in eligibility_results if result.feature_pipeline_compatible
    ]
    diversity_eligible = [
        result for result in eligibility_results if result.counts_toward_ml_diversity
    ]
    return {
        "FREE_OFFICIAL_SOURCE_READY": bool(source_ready),
        "SOURCE_READY_DOES_NOT_IMPLY_ML_DIVERSITY_ELIGIBLE": True,
        "NEW_FREE_OFFICIAL_SOURCE_COUNT": len(source_ready),
        "NEW_TARGET_ELIGIBLE_SOURCE_COUNT": len(target_eligible),
        "NEW_TARGET_ELIGIBLE_TICKER_COUNT": len(
            {result.source_ticker for result in target_eligible}
        ),
        "NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUER_COUNT": len(new_issuers),
        "FEATURE_PIPELINE_COMPATIBLE_SOURCE_COUNT": len(feature_compatible),
        "COUNTS_TOWARD_ML_DIVERSITY_SOURCE_COUNT": len(diversity_eligible),
        "TARGET_ELIGIBLE_DIVERSITY": "YES" if len(new_issuers) >= 3 else "NO",
        "DIVERSITY": "YES" if len(new_issuers) >= 3 else "NO",
        "FINAL_DIVERSITY_STATUS": diversity_status(len(new_issuers)),
        "target_eligible_sources": [result.payload() for result in target_eligible],
        "target_ineligible_sources": [
            result.payload()
            for result in eligibility_results
            if not result.target_instrument_eligible
        ],
        "diversity_eligible_sources": [result.payload() for result in diversity_eligible],
        "rejected_from_diversity": [
            result.payload()
            for result in eligibility_results
            if not result.counts_toward_ml_diversity
        ],
        "new_target_eligible_distinct_legal_issuers": list(new_issuers),
    }


def registry_delta_from_accepted_sources(
    accepted_sources: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in accepted_sources:
        row = {
            "canonical_domain": source["domain"],
            "official_domain": source["domain"],
            "content_path": ["generic.title", "generic.content"],
            "discovery_type": source["mechanism"],
            "discovery_url": source["discovery_url"],
            "enabled": False,
            "expected_publication_frequency": "unknown; discovered by bounded live probe",
            "identity_path": source["identity_mechanism"],
            "issuer": source["legal_issuer"],
            "parser": source["parser_version"],
            "parser_type": source["parser_version"],
            "polling_policy": {"interval_minutes": 180, "max_items_per_poll": 5},
            "source_id": source["source_id"],
            "source_origin": "ISSUER_ORIGINATED",
            "source_status": SourceQualificationStatus.LIVE_STRICT_EXACT_READY.value,
            "source_version": 1,
            "stable_identity": source["identity_mechanism"],
            "ticker": source["ticker"],
            "ticker_binding": {
                "binding": "single_issuer_source",
                "publication_date_validity": "strict exact timestamp accepted by v3 probe",
            },
            "timestamp_contract": {
                "evidence_type": f"TIMESTAMP_EVIDENCE_TYPE={source['timezone_evidence']}",
                "evidence_value": source["timestamp_field"],
                "policy": (
                    "accept only item-level publication timestamp with explicit offset "
                    "or documented timezone"
                ),
            },
            "timestamp_path": source["timestamp_field"],
        }
        LiveIssuerSource.from_payload(row)
        rows.append(row)
    return rows


def ready_sources_visible_to_worker(
    registry_rows: Sequence[dict[str, Any]],
) -> bool:
    return all(
        LiveIssuerSource.from_payload(dict(row)).source_status
        == SourceQualificationStatus.LIVE_STRICT_EXACT_READY
        for row in registry_rows
    )


def default_v3_candidate_sources() -> tuple[SourceProbeConfig, ...]:
    return (
        *default_candidate_sources(),
        SourceProbeConfig(
            source_id="AGRO_RUSAGRO_OFFICIAL_RSS_EXACT_LIVE_V3",
            ticker="AGRO",
            legal_issuer="Ros Agro PLC",
            official_domain="www.rusagrogroup.ru",
            url="https://www.rusagrogroup.ru/en/?type=100&tx_ttnews%5Bcat%5D=9",
            mechanism="official_issuer_rss",
            parser="rss-item-pubdate-explicit-offset-v1",
            timestamp_field="rss.channel.item.pubDate",
            identity_field="rss.channel.item.guid || rss.channel.item.link",
            content_fields=("rss.channel.item.title", "rss.channel.item.description"),
            new_hypothesis=(
                "Use first-party TYPO3 RSS alternate discovered on Rusagro investor page."
            ),
        ),
        SourceProbeConfig(
            source_id="MSFT_MICROSOFT_NEWS_OFFICIAL_RSS_EXACT_LIVE_V3",
            ticker="MSFT",
            legal_issuer="Microsoft Corporation",
            official_domain="news.microsoft.com",
            url="https://news.microsoft.com/feed/",
            mechanism="official_issuer_rss",
            parser="rss-item-pubdate-explicit-offset-v1",
            timestamp_field="rss.channel.item.pubDate",
            identity_field="rss.channel.item.guid || rss.channel.item.link",
            content_fields=("rss.channel.item.title", "rss.channel.item.description"),
            new_hypothesis=(
                "Use first-party Microsoft News RSS item pubDate as explicit UTC offset "
                "publication timestamp."
            ),
        ),
        SourceProbeConfig(
            source_id="NVDA_NVIDIA_NEWSROOM_RELEASES_RSS_EXACT_LIVE_V3",
            ticker="NVDA",
            legal_issuer="NVIDIA Corporation",
            official_domain="nvidianews.nvidia.com",
            url="https://nvidianews.nvidia.com/releases.xml",
            mechanism="official_issuer_rss",
            parser="rss-item-pubdate-explicit-offset-v1",
            timestamp_field="rss.channel.item.pubDate",
            identity_field="rss.channel.item.guid || rss.channel.item.link",
            content_fields=("rss.channel.item.title", "rss.channel.item.description"),
            new_hypothesis=(
                "Use first-party NVIDIA Newsroom releases RSS item pubDate with GMT "
                "publication timestamp."
            ),
        ),
        SourceProbeConfig(
            source_id="DIS_WALT_DISNEY_COMPANY_OFFICIAL_RSS_EXACT_LIVE_V3",
            ticker="DIS",
            legal_issuer="The Walt Disney Company",
            official_domain="thewaltdisneycompany.com",
            url="https://thewaltdisneycompany.com/feed/",
            mechanism="official_issuer_rss",
            parser="rss-item-pubdate-explicit-offset-v1",
            timestamp_field="rss.channel.item.pubDate",
            identity_field="rss.channel.item.guid || rss.channel.item.link",
            content_fields=("rss.channel.item.title", "rss.channel.item.description"),
            new_hypothesis=(
                "Use first-party Walt Disney Company RSS feed with item pubDate +0000 "
                "and canonical https news URLs."
            ),
        ),
        SourceProbeConfig(
            source_id="DSV_DSV_INVESTOR_NEWS_RSS_EXACT_LIVE_V3",
            ticker="DSV",
            legal_issuer="DSV A/S",
            official_domain="investor.dsv.com",
            url="https://investor.dsv.com/rss/news-releases.xml?category=company%20announcements",
            mechanism="official_issuer_rss",
            parser="rss-item-pubdate-explicit-offset-v1",
            timestamp_field="rss.channel.item.pubDate",
            identity_field="rss.channel.item.guid || rss.channel.item.link",
            content_fields=("rss.channel.item.title", "rss.channel.item.description"),
            new_hypothesis=(
                "Use DSV documented first-party investor RSS company announcements feed "
                "with item pubDate explicit numeric offset."
            ),
        ),
        SourceProbeConfig(
            source_id="DENTSU_GROUP_OFFICIAL_RELEASES_RSS_EXACT_LIVE_V3",
            ticker="4324",
            legal_issuer="Dentsu Group Inc.",
            official_domain="www.group.dentsu.com",
            url="https://www.group.dentsu.com/en/news/release/index.xml",
            mechanism="official_issuer_rss",
            parser="rss-item-pubdate-explicit-offset-v1",
            timestamp_field="rss.channel.item.pubDate",
            identity_field="rss.channel.item.guid || rss.channel.item.link",
            content_fields=("rss.channel.item.title", "rss.channel.item.description"),
            new_hypothesis=(
                "Use Dentsu Group first-party releases RSS with item pubDate +0900 "
                "and canonical https release links."
            ),
        ),
    )


def build_burnin_summary(
    poll_cycles: Sequence[dict[str, Any]],
    *,
    status: dict[str, Any],
    collector_state: dict[str, Any],
    seal: dict[str, Any],
    source_isolation_proof: str = "REAL_BURN_IN_PROOF",
) -> dict[str, Any]:
    source_results: list[dict[str, Any]] = []
    for cycle in poll_cycles:
        raw_rows_object: object = cycle.get("source_results", ())
        if not isinstance(raw_rows_object, list):
            continue
        raw_rows = cast("list[object]", raw_rows_object)
        source_results.extend(
            cast("dict[str, Any]", row_object)
            for row_object in raw_rows
            if isinstance(row_object, dict)
        )
    statuses = {str(row.get("status")) for row in source_results}
    state_sources_object: object = collector_state.get("sources")
    state_sources = (
        cast("dict[str, Any]", state_sources_object)
        if isinstance(state_sources_object, dict)
        else None
    )
    state_persistence = (
        state_sources is not None and len(state_sources) > 0 and len(poll_cycles) >= 2
    )
    repeat_no_duplicate = (
        any(
            str(row.get("status")) == "NO_NEW_ITEMS" and int(row.get("accepted", 0)) == 0
            for row in source_results
        )
        or int(status.get("duplicates", 0)) > 0
    )
    isolated_failure = any(
        cycle.get("LIVE_RESEARCH_OPERATION_STATUS") == "DEGRADED"
        and any(
            str(row.get("status")) == "SOURCE_FAILURE" for row in cycle.get("source_results", ())
        )
        and any(
            str(row.get("status")) in {"SUCCESS", "NO_NEW_ITEMS"}
            for row in cycle.get("source_results", ())
        )
        for cycle in poll_cycles
    )
    health_real = bool(status.get("source_health")) and bool(status.get("enabled_sources"))
    safety_counters_zero = (
        status.get("outcome_counters", {}).get("LIVE_OUTCOMES_READ", 0) == 0
        and status.get("outcome_counters", {}).get("LIVE_TARGETS_COMPUTED", 0) == 0
        and status.get("outcome_counters", {}).get("LIVE_POST_EVENT_PRICE_READS", 0) == 0
        and status.get("outcome_counters", {}).get("BROKER_MUTATIONS", 0) == 0
    )
    core_ready_without_isolation = (
        state_persistence
        and repeat_no_duplicate
        and health_real
        and bool(seal.get("sealed_epoch_verified"))
        and int(status.get("sealed_violations", 0)) == 0
        and int(status.get("timestamp_rejections", 0)) == 0
        and safety_counters_zero
    )
    proof_level = source_isolation_proof if isolated_failure else "NOT_PROVEN"
    if not core_ready_without_isolation:
        operational_burn_in = "FAIL"
    elif isolated_failure:
        operational_burn_in = "PASS"
    else:
        operational_burn_in = "PARTIAL"
    return {
        "cycles_observed": len(poll_cycles),
        "statuses_observed": sorted(statuses),
        "state_persistence_proven": state_persistence,
        "repeat_poll_no_duplicate": repeat_no_duplicate,
        "one_source_failure_isolated": isolated_failure,
        "source_failure_isolation_proof_level": proof_level,
        "health_status_real": health_real,
        "raw_snapshots_immutable": bool(seal.get("sealed_epoch_verified")),
        "timestamp_violations": int(status.get("timestamp_rejections", 0)),
        "pre_event_features_bounded": status.get("feature_status", {}).get("upper_bound_policy")
        == "end_at <= published_at",
        "seal_pass": bool(seal.get("sealed_epoch_verified")),
        "safety_counters_zero": safety_counters_zero,
        "OPERATIONAL_BURN_IN": operational_burn_in,
        "OPERATION": "YES" if operational_burn_in == "PASS" else "NO",
    }


def run_free_live_operational_burnin_and_onboarding_v3(
    *,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    operation_root: Path = DEFAULT_OPERATION_ARTIFACT_ROOT,
    source_registry_path: Path = Path(DEFAULT_SOURCE_REGISTRY_PATH),
    historical_ticker_summary_path: Path = DEFAULT_HISTORICAL_TICKER_SUMMARY_PATH,
    instrument_mapping_path: Path = DEFAULT_INSTRUMENT_MAPPING_PATH,
    candidate_configs: Sequence[SourceProbeConfig] | None = None,
    client: HttpClient | None = None,
    network_check: bool = True,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable v3 output already exists")
    now = created_at or datetime.now(UTC)
    status = build_operation_status(
        operation_root,
        registry_path=source_registry_path,
        historical_ticker_summary_path=historical_ticker_summary_path,
    )
    seal = verify_operation_seal(operation_root)
    collector_state = load_collector_state(operation_root)
    poll_cycles = _poll_cycles(operation_root)
    candidates = tuple(candidate_configs or default_v3_candidate_sources())
    http = client or BoundedHttpClient(timeout_seconds=12.0, redirect_limit=3)
    probe_results = (
        [probe_candidate_source(config, client=http, fetched_at=now) for config in candidates]
        if network_check
        else []
    )
    accepted_sources = [
        _accepted_source_payload(result) for result in probe_results if _ready(result)
    ]
    rejected_sources = [
        _rejected_source_payload(result) for result in probe_results if not _ready(result)
    ]
    instrument_mapping_rows = load_instrument_mapping_rows(instrument_mapping_path)
    target_eligibility = evaluate_target_eligibility(
        accepted_sources,
        instrument_mapping_rows=instrument_mapping_rows,
    )
    new_issuers = distinct_new_target_eligible_legal_issuers(target_eligibility)
    diversity_eligibility = diversity_eligibility_payload(target_eligibility, new_issuers)
    diversity = diversity_status(len(new_issuers), universe_exhausted=False)
    registry_delta = registry_delta_from_accepted_sources(accepted_sources)
    burnin = build_burnin_summary(
        poll_cycles,
        status=status,
        collector_state=collector_state,
        seal=seal,
    )
    shadow = _shadow_stats(operation_root, accepted_sources)
    feature_status = _feature_status(status, accepted_sources)
    safety = _safety_payload(status)
    source_health = {
        "operation": status.get("LIVE_RESEARCH_OPERATION_STATUS"),
        "enabled_sources": status.get("enabled_sources", []),
        "healthy_sources": status.get("healthy_sources", []),
        "degraded_sources": status.get("degraded_sources", []),
        "source_health": status.get("source_health", []),
        "new_ready_registry_delta_visible_to_worker": ready_sources_visible_to_worker(
            registry_delta
        )
        if registry_delta
        else True,
    }
    backlog = _candidate_backlog_payload(candidates, probe_results, now.date())
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": now.isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "HEAD_SHA": git_sha,
        "FREE_SOURCES_ONLY": True,
        "CANDIDATES_AUDITED": len(candidates),
        "NEW_HYPOTHESES_TESTED": sum(bool(item.new_hypothesis.strip()) for item in candidates),
        "NEW_READY_SOURCES": len(accepted_sources),
        "NEW_FREE_OFFICIAL_SOURCE_COUNT": len(accepted_sources),
        "NEW_ITEM_OBSERVED": any(source["real_item_observed"] for source in accepted_sources),
        "SOURCE_READY": bool(accepted_sources),
        "FREE_OFFICIAL_SOURCE_READY": bool(accepted_sources),
        "SOURCE_READY_DOES_NOT_IMPLY_ML_DIVERSITY_ELIGIBLE": True,
        "NEW_TARGET_ELIGIBLE_SOURCE_COUNT": diversity_eligibility[
            "NEW_TARGET_ELIGIBLE_SOURCE_COUNT"
        ],
        "NEW_TARGET_ELIGIBLE_TICKER_COUNT": diversity_eligibility[
            "NEW_TARGET_ELIGIBLE_TICKER_COUNT"
        ],
        "NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUER_COUNT": diversity_eligibility[
            "NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUER_COUNT"
        ],
        "FEATURE_PIPELINE_COMPATIBLE_SOURCE_COUNT": diversity_eligibility[
            "FEATURE_PIPELINE_COMPATIBLE_SOURCE_COUNT"
        ],
        "COUNTS_TOWARD_ML_DIVERSITY_SOURCE_COUNT": diversity_eligibility[
            "COUNTS_TOWARD_ML_DIVERSITY_SOURCE_COUNT"
        ],
        "NEW_DISTINCT_LEGAL_ISSUER_COUNT": len(new_issuers),
        "NEW_DISTINCT_LEGAL_ISSUERS": new_issuers,
        "FINAL_DIVERSITY_STATUS": diversity,
        "DIVERSITY_COLLECTION_CAPABILITY_READY": diversity
        == DiversityStatus.THREE_PLUS_NEW_TARGET_ELIGIBLE_ISSUERS.value,
        "TARGET_ELIGIBLE_DIVERSITY": diversity_eligibility["TARGET_ELIGIBLE_DIVERSITY"],
        "ML_V2_DATASET_STATUS": "NOT_OPENED_BY_V3_ONBOARDING",
        "OPERATION": burnin["OPERATION"],
        "OPERATIONAL_BURN_IN": burnin["OPERATIONAL_BURN_IN"],
        "SOURCE_FAILURE_ISOLATION_PROOF_LEVEL": burnin["source_failure_isolation_proof_level"],
        "DIVERSITY": "YES" if len(new_issuers) >= 3 else "NO",
        "LIVE_RESEARCH_OPERATION_STATUS": status.get("LIVE_RESEARCH_OPERATION_STATUS"),
        "PAID_SOURCE_FALLBACK_CONSIDERED": False,
        "registry_delta_count": len(registry_delta),
        "safety": safety,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = sha256_payload(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at", "HEAD_SHA", "ARTIFACT_SHA"}
        }
    )
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "burnin-summary.json", burnin)
    _write_jsonl(output_root / "poll-cycles.jsonl", poll_cycles)
    _write_json(output_root / "source-health.json", source_health)
    _write_json(output_root / "candidate-backlog.json", {"candidates": backlog})
    _write_jsonl(
        output_root / "source-probes.jsonl",
        [_source_probe_payload(result, now) for result in probe_results],
    )
    _write_json(output_root / "accepted-sources.json", {"sources": accepted_sources})
    _write_jsonl(output_root / "rejected-sources.jsonl", rejected_sources)
    _write_json(output_root / "ready-registry-delta.json", {"sources": registry_delta})
    _write_jsonl(
        output_root / "instrument-eligibility.jsonl",
        [result.payload() for result in target_eligibility],
    )
    _write_jsonl(
        output_root / "target-mapping.jsonl",
        [result.payload()["source_to_canonical_mapping"] for result in target_eligibility],
    )
    _write_json(output_root / "diversity-eligibility.json", diversity_eligibility)
    _write_json(
        output_root / "burnin-gate.json",
        {
            "LIVE_RESEARCH_OPERATION_STATUS": status.get("LIVE_RESEARCH_OPERATION_STATUS"),
            "OPERATIONAL_BURN_IN": burnin["OPERATIONAL_BURN_IN"],
            "OPERATION": burnin["OPERATION"],
            "source_failure_isolation_proof_level": burnin["source_failure_isolation_proof_level"],
            "one_source_failure_isolated": burnin["one_source_failure_isolated"],
            "seal_pass": burnin["seal_pass"],
            "timestamp_violations": burnin["timestamp_violations"],
            "safety_counters_zero": burnin["safety_counters_zero"],
        },
    )
    _write_json(output_root / "shadow-stats.json", shadow)
    _write_json(output_root / "feature-status.json", feature_status)
    _write_json(output_root / "safety.json", safety)
    _write_report(
        output_root / "report.md",
        manifest,
        burnin,
        shadow,
        feature_status,
        diversity_eligibility,
    )
    return manifest


def _poll_cycles(operation_root: Path) -> list[dict[str, Any]]:
    run_root = operation_root / "runs"
    if not run_root.exists():
        return []
    cycles: dict[str, dict[str, Any]] = {}
    for source_dir in sorted(path for path in run_root.glob("*/*") if path.is_dir()):
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        run_id = str(source_dir.parent.name)
        cycle = cycles.setdefault(
            run_id,
            {
                "run_id": run_id,
                "source_results": [],
                "HTTP_REQUESTS": 0,
                "duplicates": 0,
                "new_publications": 0,
            },
        )
        cycle["HTTP_REQUESTS"] = int(cycle["HTTP_REQUESTS"]) + int(
            manifest.get("BOUNDED_HTTP_REQUESTS", 0)
        )
        cycle["duplicates"] = int(cycle["duplicates"]) + int(
            manifest.get("DUPLICATES_ENCOUNTERED", 0)
        )
        cycle["new_publications"] = int(cycle["new_publications"]) + int(
            manifest.get("EVENTS_COLLECTED", 0)
        )
        source_rows = _read_jsonl(source_dir / "source-polls.jsonl")
        shadow_rows = _read_jsonl(source_dir / "live-shadow-corpus.jsonl")
        source_id = str(manifest.get("source_id") or source_dir.name)
        cycle["source_results"].append(
            {
                "source_id": source_id,
                "status": _source_poll_status(source_rows, len(shadow_rows)),
                "accepted": len(shadow_rows),
                "duplicates": manifest.get("DUPLICATES_ENCOUNTERED", 0),
                "http_requests": manifest.get("BOUNDED_HTTP_REQUESTS", 0),
            }
        )
    for cycle in cycles.values():
        statuses = {row["status"] for row in cycle["source_results"]}
        cycle["LIVE_RESEARCH_OPERATION_STATUS"] = (
            "DEGRADED" if "SOURCE_FAILURE" in statuses else "READY"
        )
    return list(cycles.values())


def _source_poll_status(source_rows: list[dict[str, Any]], accepted: int) -> str:
    if any(str(row.get("status")) == "SOURCE_FAILURE" for row in source_rows):
        return "SOURCE_FAILURE"
    if accepted:
        return "SUCCESS"
    return "NO_NEW_ITEMS"


def _accepted_source_payload(result: SourceProbeResult) -> dict[str, Any]:
    first = result.items_observed[0]
    payload = {
        "source_id": result.config.source_id,
        "ticker": result.config.ticker,
        "legal_issuer": result.config.legal_issuer,
        "domain": result.config.official_domain,
        "discovery_url": result.config.url,
        "mechanism": result.config.mechanism,
        "parser_version": result.config.parser,
        "timestamp_field": result.config.timestamp_field,
        "timezone_evidence": result.timestamp_level.value,
        "identity_mechanism": result.config.identity_field,
        "real_item_observed": result.real_item_observed,
        "first_item": {
            "source_item_id": first.source_item_id,
            "canonical_url": first.canonical_url,
            "published_at": first.published_at.isoformat(),
            "published_raw": first.published_raw,
            "title": first.title,
        },
        "SOURCE_READY": True,
        "NEW_ITEM_OBSERVED": result.real_item_observed,
    }
    return payload | {"contract_sha": sha256_payload(payload)}


def _rejected_source_payload(result: SourceProbeResult) -> dict[str, Any]:
    blocker = result.blocker
    if result.status == SourceStatus.LIVE_STRICT_EXACT_READY and not _has_v3_stable_identity(
        result
    ):
        blocker = "STABLE_HTTPS_IDENTITY_REQUIRED"
    return {
        "source_id": result.config.source_id,
        "ticker": result.config.ticker,
        "legal_issuer": result.config.legal_issuer,
        "url": result.config.url,
        "status": result.status.value,
        "blocker": blocker,
        "timestamp_level": result.timestamp_level.value,
        "new_hypothesis": result.config.new_hypothesis,
        "paid_fallback_considered": False,
    }


def _source_probe_payload(result: SourceProbeResult, observed_at: datetime) -> dict[str, Any]:
    response = result.response
    return {
        "source_id": result.config.source_id,
        "ticker": result.config.ticker,
        "legal_issuer": result.config.legal_issuer,
        "official_url": result.config.url,
        "mechanism": result.config.mechanism,
        "parser": result.config.parser,
        "new_hypothesis": result.config.new_hypothesis,
        "current_status": result.status.value,
        "blocker": result.blocker,
        "timestamp_evidence": result.timestamp_level.value,
        "identity_evidence": result.config.identity_field
        if result.status != SourceStatus.LIVE_NO_STABLE_ID
        else None,
        "http_status": None if response is None else response.status,
        "final_url": None if response is None else response.final_url,
        "content_type": None if response is None else response.content_type,
        "items_observed": len(result.items_observed),
        "alternate_links": list(result.alternate_links),
        "observed_at": observed_at.isoformat(),
    }


def _candidate_backlog_payload(
    candidates: Sequence[SourceProbeConfig],
    results: Sequence[SourceProbeResult],
    checked_on: date,
) -> list[dict[str, Any]]:
    by_source_id = {result.config.source_id: result for result in results}
    rows: list[dict[str, Any]] = []
    for config in candidates:
        result = by_source_id.get(config.source_id)
        current_status = result.status.value if result else "NOT_PROBED_THIS_RUN"
        entry = CandidateBacklogEntry(
            issuer=config.legal_issuer,
            tickers=(config.ticker,),
            official_url=config.url,
            last_checked=checked_on,
            previous_blocker=config.prior_rejection_source_id or "",
            new_hypothesis=config.new_hypothesis,
            current_status=current_status,
            timestamp_evidence=None if result is None else result.timestamp_level.value,
            identity_evidence=config.identity_field,
            next_possible_free_mechanism=config.mechanism,
        )
        rows.append(entry.payload())
    return rows


def _shadow_stats(
    operation_root: Path, accepted_sources: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    rows = _read_jsonl(operation_root / "live-shadow-corpus.jsonl")
    ticker_counts = Counter(str(row.get("ticker", "UNKNOWN")) for row in rows)
    source_counts = Counter(str(row.get("source_id", "UNKNOWN")) for row in rows)
    semantic_ready = len(rows)
    unknown = sum(
        1
        for row in rows
        if cast("dict[str, Any]", row.get("semantic_output", {})).get("semantic_unknown") is True
    )
    top1 = max(ticker_counts.values(), default=0)
    total = sum(ticker_counts.values())
    top3 = sum(count for _, count in ticker_counts.most_common(3))
    return {
        "total_events": len(rows),
        "by_ticker": dict(sorted(ticker_counts.items())),
        "by_source": dict(sorted(source_counts.items())),
        "new_ready_sources_by_ticker": dict(
            sorted(Counter(str(source["ticker"]) for source in accepted_sources).items())
        ),
        "semantic_ready": semantic_ready,
        "UNKNOWN_count": unknown,
        "UNKNOWN_rate": _rate(unknown, semantic_ready),
        "feature_ready": sum(
            1
            for row in rows
            if cast("dict[str, Any]", row.get("pre_event_feature_availability", {})).get(
                "available"
            )
            is True
        ),
        "top1_live_ticker_share": _rate(top1, total),
        "top3_live_ticker_share": _rate(top3, total),
        "ticker_hhi": _hhi(ticker_counts),
        "target_metrics_included": False,
    }


def _feature_status(
    status: dict[str, Any], accepted_sources: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    feature = dict(cast("dict[str, Any]", status.get("feature_status", {})))
    feature["new_source_feature_status"] = [
        {
            "source_id": source["source_id"],
            "ticker": source["ticker"],
            "feature_ready": False,
            "blocker": "NEW_SOURCE_SMOKE_DID_NOT_READ_MARKET_DATA",
        }
        for source in accepted_sources
    ]
    feature["target_metrics_included"] = False
    return feature


def _safety_payload(status: dict[str, Any]) -> dict[str, Any]:
    counters = cast("dict[str, Any]", status.get("outcome_counters", {}))
    return {
        **live_accumulation_safety_flags(),
        "FREE_SOURCES_ONLY": True,
        "PAID_SOURCE_FALLBACK_CONSIDERED": False,
        "PAID_SOURCES_USED": 0,
        "MODEL_TRAINING_PERFORMED": False,
        "MODEL_PREDICTIONS_PERFORMED": False,
        "LIVE_OUTCOMES_READ": int(counters.get("LIVE_OUTCOMES_READ", 0)),
        "LIVE_TARGETS_COMPUTED": int(counters.get("LIVE_TARGETS_COMPUTED", 0)),
        "LIVE_POST_EVENT_PRICE_READS": int(counters.get("LIVE_POST_EVENT_PRICE_READS", 0)),
        "BROKER_MUTATIONS": int(counters.get("BROKER_MUTATIONS", 0)),
    }


def _ready(result: SourceProbeResult) -> bool:
    return (
        result.status == SourceStatus.LIVE_STRICT_EXACT_READY
        and result.timestamp_level in {TimestampLevel.LEVEL_A, TimestampLevel.LEVEL_B}
        and result.real_item_observed
        and _has_v3_stable_identity(result)
    )


def _has_v3_stable_identity(result: SourceProbeResult) -> bool:
    if not result.items_observed:
        return False
    for item in result.items_observed:
        identity = item.source_item_id.strip()
        canonical_url = item.canonical_url.strip()
        if not identity:
            return False
        if "://" in identity and not identity.startswith("https://"):
            return False
        if "://" in canonical_url and not canonical_url.startswith("https://"):
            return False
    return True


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\u0430-\u044f\u0451]+", " ", value.lower()).strip()


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 6)


def _hhi(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return round(sum((count / total) ** 2 for count in counter.values()), 6)


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(cast("dict[str, Any]", json.loads(line)))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    burnin: dict[str, Any],
    shadow: dict[str, Any],
    feature_status: dict[str, Any],
    diversity_eligibility: dict[str, Any],
) -> None:
    lines = [
        "# Free live operational burn-in and onboarding v3",
        "",
        f"- OPERATION: {manifest['OPERATION']}",
        f"- Operational burn-in: {manifest['OPERATIONAL_BURN_IN']}",
        f"- Source failure isolation proof: {manifest['SOURCE_FAILURE_ISOLATION_PROOF_LEVEL']}",
        f"- DIVERSITY: {manifest['DIVERSITY']}",
        f"- Final diversity status: {manifest['FINAL_DIVERSITY_STATUS']}",
        (
            "- Source-ready / target-eligible / diversity-eligible: "
            f"{manifest['NEW_FREE_OFFICIAL_SOURCE_COUNT']} / "
            f"{manifest['NEW_TARGET_ELIGIBLE_SOURCE_COUNT']} / "
            f"{manifest['COUNTS_TOWARD_ML_DIVERSITY_SOURCE_COUNT']}"
        ),
        (
            "- New target-eligible distinct legal issuer count: "
            f"{manifest['NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUER_COUNT']}"
        ),
        f"- New READY sources: {manifest['NEW_READY_SOURCES']}",
        (f"- Rejected from diversity: {len(diversity_eligibility['rejected_from_diversity'])}"),
        f"- Burn-in cycles observed: {burnin['cycles_observed']}",
        f"- State persistence proven: {burnin['state_persistence_proven']}",
        f"- Repeat poll no duplicate: {burnin['repeat_poll_no_duplicate']}",
        f"- Source failure isolated: {burnin['one_source_failure_isolated']}",
        f"- Health/status real: {burnin['health_status_real']}",
        f"- Seal pass: {burnin['seal_pass']}",
        f"- Timestamp violations: {burnin['timestamp_violations']}",
        f"- Total shadow events: {shadow['total_events']}",
        f"- Semantic ready: {shadow['semantic_ready']}",
        f"- UNKNOWN rate: {shadow['UNKNOWN_rate']}",
        f"- Feature ready: {feature_status.get('feature_ready', 0)}",
        f"- Paid sources used: {manifest['PAID_SOURCES_USED']}",
        f"- Live outcomes read: {manifest['LIVE_OUTCOMES_READ']}",
        f"- Live targets computed: {manifest['LIVE_TARGETS_COMPUTED']}",
        f"- Broker mutations: {manifest['BROKER_MUTATIONS']}",
        "",
        "ML dataset remains closed by this artifact.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
