from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from src.exact_event_live_official_collection.http_client import (
    BoundedHttpClient,
    FetchResult,
    HttpClient,
)
from src.free_live_issuer_accumulation.domain import (
    SOURCE_REGISTRY_VERSION,
    live_accumulation_safety_flags,
    sha256_payload,
)
from src.free_live_issuer_accumulation.operation import (
    FreeLiveResearchOperation,
    OperationConfig,
    SourcePollStatus,
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
    probe_candidate_source,
)
from src.free_live_operational_burnin_and_onboarding_v3.application import (
    DEFAULT_INSTRUMENT_MAPPING_PATH,
    TargetEligibilityResult,
    build_burnin_summary,
    distinct_new_target_eligible_legal_issuers,
    diversity_eligibility_payload,
    evaluate_target_eligibility,
    load_instrument_mapping_rows,
)
from src.instruments.infrastructure.seed import SEED_INSTRUMENTS

ARTIFACT_VERSION = "moex-target-source-expansion-v4"
DEFAULT_OUTPUT_ROOT = Path(f"artifacts/{ARTIFACT_VERSION}")
DEFAULT_OPERATION_ROOT = Path("artifacts/free-live-research-operation-v1")
MAX_TARGET_CANDIDATES = 20


@dataclass(frozen=True, slots=True)
class TargetCandidate:
    ticker: str
    legal_issuer: str
    legal_issuer_key: str
    rank: int
    acquisition_usefulness_score: int
    eligibility: TargetEligibilityResult

    def payload(self) -> dict[str, Any]:
        payload = self.eligibility.payload()
        return {
            "rank": self.rank,
            "ticker": self.ticker,
            "legal_issuer": self.legal_issuer,
            "legal_issuer_key": self.legal_issuer_key,
            "acquisition_usefulness_score": self.acquisition_usefulness_score,
            "ranking_inputs": {
                "canonical_eligibility": self.eligibility.target_instrument_eligible,
                "market_data_mapping": self.eligibility.canonical_mapping_ready,
                "feature_compatibility": self.eligibility.feature_pipeline_compatible,
                "liquidity_research_relevance_proxy": "seed_registry_order_and_mapping_history",
                "official_public_channel_hypothesis_available": bool(
                    default_v4_candidate_by_ticker().get(self.ticker)
                ),
                "returns_or_model_performance_used": False,
            },
            **payload,
        }


def build_target_candidate_universe(
    *,
    instrument_mapping_rows: Sequence[dict[str, Any]],
    frozen_tickers: Sequence[str] = HISTORICAL_ISSUER_TICKERS,
    max_candidates: int = MAX_TARGET_CANDIDATES,
) -> tuple[list[TargetCandidate], list[dict[str, Any]]]:
    pseudo_sources = [
        {
            "source_id": f"{instrument.ticker}_CANONICAL_TARGET_CHECK",
            "ticker": instrument.ticker,
            "legal_issuer": instrument.issuer_name,
        }
        for instrument in SEED_INSTRUMENTS
    ]
    eligibility = evaluate_target_eligibility(
        pseudo_sources,
        instrument_mapping_rows=instrument_mapping_rows,
        frozen_tickers=frozen_tickers,
    )
    candidates: list[TargetCandidate] = []
    skipped: list[dict[str, Any]] = []
    seen_issuers: set[str] = set()
    rank = 1
    for result in eligibility:
        canonical = result.canonical_instrument
        legal_key = result.legal_issuer_key_value
        if canonical is None or legal_key is None:
            skipped.append(_skip_payload(result, "TARGET_INSTRUMENT_INELIGIBLE"))
            continue
        if not result.target_instrument_eligible or not result.feature_pipeline_compatible:
            skipped.append(_skip_payload(result, result.blocker or "TARGET_INSTRUMENT_INELIGIBLE"))
            continue
        if legal_key in seen_issuers:
            skipped.append(_skip_payload(result, "DUPLICATE_LEGAL_ISSUER_SHARE_CLASS_COLLAPSE"))
            continue
        seen_issuers.add(legal_key)
        candidates.append(
            TargetCandidate(
                ticker=canonical.ticker,
                legal_issuer=canonical.legal_issuer,
                legal_issuer_key=legal_key,
                rank=rank,
                acquisition_usefulness_score=_usefulness_score(result, rank),
                eligibility=result,
            )
        )
        rank += 1
        if len(candidates) >= max_candidates:
            break
    return candidates, skipped


def default_v4_candidate_by_ticker() -> dict[str, SourceProbeConfig]:
    return {
        "GAZP": SourceProbeConfig(
            source_id="GAZP_GAZPROM_IR_RELEASES_ARCHIVE_JSONLD_V4",
            ticker="GAZP",
            legal_issuer='ПАО "Газпром"',
            official_domain="www.gazprom.com",
            url="https://www.gazprom.com/investors/disclosure/irreleases/",
            mechanism="official_investor_irrelease_archive_jsonld_probe",
            parser="html-alternate-jsonld-v1",
            timestamp_field="jsonld.datePublished || official IR release item timestamp",
            identity_field="jsonld.url || canonical url || official release id",
            content_fields=("headline", "description", "articleBody"),
            new_hypothesis=(
                "Probe first-party investor IR releases archive instead of generic press page."
            ),
            prior_rejection_source_id="GAZP_GAZPROM_PRESS_HTML_ALT_JSONLD_V2",
        ),
        "LKOH": SourceProbeConfig(
            source_id="LKOH_LUKOIL_INVESTOR_FILTERED_PRESS_JSONLD_V4",
            ticker="LKOH",
            legal_issuer="ПАО ЛУКОЙЛ",
            official_domain="www.lukoil.com",
            url="https://www.lukoil.com/PressCenter/Pressreleases?tags=38CXjmeic02Sgectxn85Pg%2C1%3B",
            mechanism="official_investor_filtered_press_jsonld_probe",
            parser="html-alternate-jsonld-v1",
            timestamp_field="jsonld.datePublished || filtered press release timestamp",
            identity_field="jsonld.url || canonical url || release id",
            content_fields=("headline", "description", "articleBody"),
            new_hypothesis=(
                "Probe first-party investor/shareholder filtered press page instead of broad "
                "press listing."
            ),
            prior_rejection_source_id="LKOH_LUKOIL_PRESS_HTML_ALT_JSONLD_V2",
        ),
        "NVTK": SourceProbeConfig(
            source_id="NVTK_NOVATEK_INDEX_PHP_RELEASES_JSONLD_V4",
            ticker="NVTK",
            legal_issuer="ПАО НОВАТЭК",
            official_domain="www.novatek.ru",
            url="https://www.novatek.ru/en/press/releases/index.php",
            mechanism="official_release_index_php_jsonld_probe",
            parser="html-alternate-jsonld-v1",
            timestamp_field="jsonld.datePublished || release index item timestamp",
            identity_field="jsonld.url || canonical url || release id",
            content_fields=("headline", "description", "articleBody"),
            new_hypothesis=(
                "Probe explicit index.php release archive variant for machine-readable metadata."
            ),
            prior_rejection_source_id="NVTK_NOVATEK_PRESS_HTML_ALT_JSONLD_V2",
        ),
        "SBER": SourceProbeConfig(
            source_id="SBER_SBERBANK_RU_PRESS_CENTER_JSONLD_V4",
            ticker="SBER",
            legal_issuer="ПАО Сбербанк",
            official_domain="www.sberbank.ru",
            url="https://www.sberbank.ru/ru/press_center/all",
            mechanism="official_ru_press_center_jsonld_probe",
            parser="html-alternate-jsonld-v1",
            timestamp_field="jsonld.datePublished || official ru press item timestamp",
            identity_field="jsonld.url || canonical url || release id",
            content_fields=("headline", "description", "articleBody"),
            new_hypothesis=(
                "Probe first-party Russian press center instead of sberbank.com English page."
            ),
            prior_rejection_source_id="SBER_SBERBANK_PRESS_HTML_ALT_JSONLD_V2",
        ),
        "VTBR": SourceProbeConfig(
            source_id="VTBR_VTB_RU_PRESS_CENTER_JSONLD_V4",
            ticker="VTBR",
            legal_issuer="Банк ВТБ",
            official_domain="www.vtb.ru",
            url="https://www.vtb.ru/about/press/",
            mechanism="official_ru_press_center_jsonld_probe",
            parser="html-alternate-jsonld-v1",
            timestamp_field="jsonld.datePublished || official ru press item timestamp",
            identity_field="jsonld.url || canonical url || release id",
            content_fields=("headline", "description", "articleBody"),
            new_hypothesis="Probe first-party Russian VTB press center instead of vtb.com path.",
            prior_rejection_source_id="VTBR_VTB_PRESS_HTML_ALT_JSONLD_V2",
        ),
    }


def build_v4_candidate_hypotheses(
    target_candidates: Sequence[TargetCandidate],
) -> tuple[list[SourceProbeConfig], list[dict[str, Any]]]:
    configured = default_v4_candidate_by_ticker()
    probes: list[SourceProbeConfig] = []
    rows: list[dict[str, Any]] = []
    for candidate in target_candidates:
        config = configured.get(candidate.ticker)
        if config is None:
            rows.append(
                {
                    "ticker": candidate.ticker,
                    "legal_issuer": candidate.legal_issuer,
                    "network_probe_allowed": False,
                    "skip_reason": "NO_NEW_FREE_OFFICIAL_SOURCE_HYPOTHESIS",
                }
            )
            continue
        if not config.new_hypothesis.strip():
            rows.append(
                {
                    "ticker": candidate.ticker,
                    "legal_issuer": candidate.legal_issuer,
                    "network_probe_allowed": False,
                    "skip_reason": "RECHECK_REQUIRES_NEW_HYPOTHESIS",
                }
            )
            continue
        probes.append(config)
        rows.append(
            {
                "source_id": config.source_id,
                "ticker": config.ticker,
                "legal_issuer": config.legal_issuer,
                "official_url": config.url,
                "official_domain": config.official_domain,
                "PREVIOUS_BLOCKER": config.prior_rejection_source_id,
                "PREVIOUS_MECHANISM": "see previous artifact source_id",
                "NEW_HYPOTHESIS": config.new_hypothesis,
                "NEW_MECHANISM": config.mechanism,
                "network_probe_allowed": True,
            }
        )
    return probes, rows


def run_moex_target_source_expansion_v4(
    *,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    operation_root: Path = DEFAULT_OPERATION_ROOT,
    instrument_mapping_path: Path = DEFAULT_INSTRUMENT_MAPPING_PATH,
    client: HttpClient | None = None,
    network_check: bool = True,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable v4 output already exists")
    output_root.mkdir(parents=True, exist_ok=False)
    now = created_at or datetime.now(UTC)
    mapping_rows = load_instrument_mapping_rows(instrument_mapping_path)
    target_candidates, skipped = build_target_candidate_universe(
        instrument_mapping_rows=mapping_rows
    )
    probe_configs, hypothesis_rows = build_v4_candidate_hypotheses(target_candidates)
    http = client or BoundedHttpClient(timeout_seconds=8.0, redirect_limit=3)
    probe_results = (
        [probe_candidate_source(config, client=http, fetched_at=now) for config in probe_configs]
        if network_check
        else []
    )
    accepted_sources = [
        _accepted_source_payload(result) for result in probe_results if _ready(result)
    ]
    rejected_sources = [
        _rejected_source_payload(result) for result in probe_results if not _ready(result)
    ]
    source_eligibility = evaluate_target_eligibility(
        accepted_sources,
        instrument_mapping_rows=mapping_rows,
    )
    new_issuers = distinct_new_target_eligible_legal_issuers(source_eligibility)
    diversity = diversity_eligibility_payload(source_eligibility, new_issuers)
    source_isolation = prove_source_isolation_application_path(
        output_root=output_root / "source-isolation-operation",
        base_main_sha=base_main_sha,
        git_sha=git_sha,
        created_at=now,
    )
    live_status = build_operation_status(operation_root)
    live_seal = verify_operation_seal(operation_root)
    safety = _safety_payload(live_status)
    operational_burnin = _operational_burnin_payload(source_isolation, live_status, live_seal)
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": now.isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "HEAD_SHA": git_sha,
        "SOURCE_REGISTRY_VERSION": SOURCE_REGISTRY_VERSION,
        "FREE_SOURCES_ONLY": True,
        "PAID_SOURCES_USED": False,
        "PAID_SOURCE_FALLBACK_CONSIDERED": False,
        "PAID_API_CALLS": 0,
        "TARGET_CANDIDATE_ISSUER_COUNT": len(target_candidates),
        "TARGET_CANDIDATE_TICKERS": [candidate.ticker for candidate in target_candidates],
        "CANDIDATES_SKIPPED_INELIGIBLE_BEFORE_NETWORK": len(skipped),
        "CANDIDATES_ACTUALLY_PROBED": len(probe_results),
        "NEW_HYPOTHESES_TESTED": sum(
            bool(config.new_hypothesis.strip()) for config in probe_configs
        ),
        "NEW_FREE_OFFICIAL_SOURCE_COUNT": len(accepted_sources),
        "OFFICIAL_FREE_SOURCES_READY": len(accepted_sources),
        "NEW_TARGET_ELIGIBLE_SOURCE_COUNT": diversity["NEW_TARGET_ELIGIBLE_SOURCE_COUNT"],
        "FEATURE_PIPELINE_COMPATIBLE_SOURCE_COUNT": diversity[
            "FEATURE_PIPELINE_COMPATIBLE_SOURCE_COUNT"
        ],
        "NEW_TARGET_ELIGIBLE_TICKER_COUNT": diversity["NEW_TARGET_ELIGIBLE_TICKER_COUNT"],
        "NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUER_COUNT": diversity[
            "NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUER_COUNT"
        ],
        "TARGET_ELIGIBLE_DIVERSITY": diversity["TARGET_ELIGIBLE_DIVERSITY"],
        "DIVERSITY": diversity["DIVERSITY"],
        "FINAL_DIVERSITY_STATUS": diversity["FINAL_DIVERSITY_STATUS"],
        "OPERATIONAL_ISOLATION": "YES"
        if source_isolation["SOURCE_ISOLATION_APPLICATION_PROOF"]
        else "NO",
        "SOURCE_ISOLATION_UNIT_PROOF": source_isolation["SOURCE_ISOLATION_UNIT_PROOF"],
        "SOURCE_ISOLATION_APPLICATION_PROOF": source_isolation[
            "SOURCE_ISOLATION_APPLICATION_PROOF"
        ],
        "SOURCE_ISOLATION_REAL_NETWORK_PROOF": source_isolation[
            "SOURCE_ISOLATION_REAL_NETWORK_PROOF"
        ],
        "OPERATIONAL_BURN_IN": operational_burnin["OPERATIONAL_BURN_IN"],
        "OPERATION": operational_burnin["OPERATION"],
        "LIVE_RESEARCH_OPERATION_STATUS": live_status["LIVE_RESEARCH_OPERATION_STATUS"],
        "TOTAL_LIVE_SHADOW_EVENTS": live_status["total_shadow_events"],
        "ML_V2_DATASET_STATUS": "NOT_OPENED_BY_V4_SOURCE_EXPANSION",
        "TARGET_METRICS_INCLUDED": False,
        "MODEL_TRAINING_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
        "OLD_FUTURE_HOLDOUT_OPENED": False,
        "REAL_TRADING_ALLOWED": False,
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
    _write_json(output_root / "manifest.json", manifest)
    _write_jsonl(
        output_root / "target-candidate-universe.jsonl",
        [candidate.payload() for candidate in target_candidates],
    )
    _write_jsonl(
        output_root / "instrument-eligibility.jsonl",
        [candidate.eligibility.payload() for candidate in target_candidates] + skipped,
    )
    _write_jsonl(output_root / "candidate-hypotheses.jsonl", hypothesis_rows)
    _write_jsonl(
        output_root / "source-probes.jsonl", [_source_probe_payload(r, now) for r in probe_results]
    )
    _write_json(output_root / "accepted-sources.json", {"sources": accepted_sources})
    _write_jsonl(output_root / "rejected-sources.jsonl", rejected_sources)
    _write_json(output_root / "source-isolation-proof.json", source_isolation)
    _write_json(output_root / "operational-burnin.json", operational_burnin)
    _write_jsonl(
        output_root / "feature-compatibility.jsonl",
        [result.payload() for result in source_eligibility],
    )
    _write_json(output_root / "diversity-status.json", diversity)
    _write_json(output_root / "safety.json", safety)
    _write_report(output_root / "report.md", manifest, target_candidates, skipped, rejected_sources)
    return manifest


def prove_source_isolation_application_path(
    *,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    created_at: datetime,
) -> dict[str, Any]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    registry_path = output_root / "controlled-registry.json"
    _write_json(registry_path, _controlled_registry_payload())
    client = _ScriptedIsolationClient(
        {
            "https://issuer-a.test/rss": [
                _failure("https://issuer-a.test/rss"),
                _rss_item("AAA", "recovered", domain="issuer-a.test"),
            ],
            "https://issuer-b.test/rss": [
                _rss_item("BBB", "healthy", domain="issuer-b.test"),
                _rss_item("BBB", "healthy", domain="issuer-b.test"),
            ],
        }
    )
    config = OperationConfig(
        artifact_root=output_root / "operation",
        registry_path=registry_path,
        historical_ticker_summary_path=output_root / "missing-tickers.jsonl",
        default_interval_minutes=10,
        max_retries=0,
        failure_threshold=3,
        cooldown_minutes=30,
        max_items_per_poll=5,
    )

    def client_factory(source: Any) -> _ScriptedIsolationClient:
        return client.for_source(str(source.discovery_url))

    first = FreeLiveResearchOperation(
        config,
        client_factory=client_factory,
        now=lambda: created_at,
    ).poll_once(base_main_sha=base_main_sha, git_sha=git_sha, force=True)
    second_time = created_at + timedelta(hours=2)
    second = FreeLiveResearchOperation(
        config,
        client_factory=client_factory,
        now=lambda: second_time,
    ).poll_once(base_main_sha=base_main_sha, git_sha=git_sha, force=True)
    status = build_operation_status(
        config.artifact_root,
        registry_path=registry_path,
        historical_ticker_summary_path=config.historical_ticker_summary_path,
    )
    seal = verify_operation_seal(config.artifact_root)
    state = load_collector_state(config.artifact_root)
    burnin = build_burnin_summary(
        [first, second],
        status=status,
        collector_state=state,
        seal=seal,
        source_isolation_proof="APPLICATION_PROOF",
    )
    by_first = {row["source_id"]: row for row in first["source_results"]}
    by_second = {row["source_id"]: row for row in second["source_results"]}
    application_proof = (
        first["LIVE_RESEARCH_OPERATION_STATUS"] == "DEGRADED"
        and by_first["AAA_SOURCE_ISOLATION_V4"]["status"] == SourcePollStatus.SOURCE_FAILURE.value
        and by_first["BBB_SOURCE_ISOLATION_V4"]["status"] == SourcePollStatus.SUCCESS.value
        and second["LIVE_RESEARCH_OPERATION_STATUS"] == "READY"
        and by_second["AAA_SOURCE_ISOLATION_V4"]["status"] == SourcePollStatus.SUCCESS.value
        and by_second["BBB_SOURCE_ISOLATION_V4"]["status"] == SourcePollStatus.NO_NEW_ITEMS.value
        and status["collector_process_alive"] is True
        and "BBB_SOURCE_ISOLATION_V4" in cast("dict[str, Any]", state["sources"])
        and seal["sealed_epoch_verified"] is True
        and burnin["OPERATIONAL_BURN_IN"] == "PASS"
    )
    return {
        "SOURCE_ISOLATION_UNIT_PROOF": True,
        "SOURCE_ISOLATION_APPLICATION_PROOF": application_proof,
        "SOURCE_ISOLATION_REAL_NETWORK_PROOF": False,
        "intentional_external_source_damage": False,
        "controlled_dependency_failure": True,
        "first_cycle_status": first["LIVE_RESEARCH_OPERATION_STATUS"],
        "second_cycle_status": second["LIVE_RESEARCH_OPERATION_STATUS"],
        "failed_source": by_first["AAA_SOURCE_ISOLATION_V4"],
        "healthy_source_continued": by_first["BBB_SOURCE_ISOLATION_V4"],
        "recovered_source": by_second["AAA_SOURCE_ISOLATION_V4"],
        "state_b_persisted": "BBB_SOURCE_ISOLATION_V4" in cast("dict[str, Any]", state["sources"]),
        "seal": seal,
        "burnin": burnin,
        "client_calls": client.calls,
    }


def _operational_burnin_payload(
    source_isolation: dict[str, Any],
    live_status: dict[str, Any],
    live_seal: dict[str, Any],
) -> dict[str, Any]:
    counters = cast("dict[str, Any]", live_status.get("outcome_counters", {}))
    safety_zero = (
        int(counters.get("LIVE_OUTCOMES_READ", 0)) == 0
        and int(counters.get("LIVE_TARGETS_COMPUTED", 0)) == 0
        and int(counters.get("LIVE_POST_EVENT_PRICE_READS", 0)) == 0
        and int(counters.get("BROKER_MUTATIONS", 0)) == 0
    )
    pass_gate = (
        source_isolation["SOURCE_ISOLATION_APPLICATION_PROOF"] is True
        and live_status["LIVE_RESEARCH_OPERATION_STATUS"] in {"READY", "DEGRADED"}
        and live_seal["sealed_epoch_verified"] is True
        and int(live_status.get("timestamp_rejections", 0)) == 0
        and int(live_status.get("sealed_violations", 0)) == 0
        and safety_zero
    )
    return {
        "OPERATIONAL_BURN_IN": "PASS" if pass_gate else "FAIL",
        "OPERATION": "YES" if pass_gate else "NO",
        "source_isolation_required": True,
        "source_isolation_application_proof": source_isolation[
            "SOURCE_ISOLATION_APPLICATION_PROOF"
        ],
        "live_research_operation_status": live_status["LIVE_RESEARCH_OPERATION_STATUS"],
        "live_collector_rosn_ydex_ready": live_status["LIVE_RESEARCH_OPERATION_STATUS"] == "READY",
        "live_seal": live_seal,
        "safety_counters_zero": safety_zero,
        "timestamp_violations": int(live_status.get("timestamp_rejections", 0)),
        "sealed_violations": int(live_status.get("sealed_violations", 0)),
    }


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
        "FREE_OFFICIAL_SOURCE_READY": True,
        "SOURCE_READY": True,
        "READY_DISABLED": True,
    }
    return payload | {"contract_sha": sha256_payload(payload)}


def _rejected_source_payload(result: SourceProbeResult) -> dict[str, Any]:
    return {
        "source_id": result.config.source_id,
        "ticker": result.config.ticker,
        "legal_issuer": result.config.legal_issuer,
        "url": result.config.url,
        "status": result.status.value,
        "blocker": result.blocker,
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


def _ready(result: SourceProbeResult) -> bool:
    return (
        result.status == SourceStatus.LIVE_STRICT_EXACT_READY
        and result.timestamp_level in {TimestampLevel.LEVEL_A, TimestampLevel.LEVEL_B}
        and result.real_item_observed
        and all(
            item.source_item_id.startswith("https://") or "://" not in item.source_item_id
            for item in result.items_observed
        )
        and all(
            item.canonical_url.startswith("https://") or "://" not in item.canonical_url
            for item in result.items_observed
        )
    )


def _skip_payload(result: TargetEligibilityResult, reason: str) -> dict[str, Any]:
    payload = result.payload()
    payload["skip_reason"] = reason
    payload["network_probe_allowed"] = False
    return payload


def _usefulness_score(result: TargetEligibilityResult, rank: int) -> int:
    return (
        100
        + (20 if result.canonical_mapping_ready else 0)
        + (20 if result.feature_pipeline_compatible else 0)
        + max(0, 20 - rank)
    )


def _safety_payload(status: dict[str, Any]) -> dict[str, Any]:
    counters = cast("dict[str, Any]", status.get("outcome_counters", {}))
    return {
        **live_accumulation_safety_flags(),
        "FREE_SOURCES_ONLY": True,
        "PAID_SOURCES_USED": False,
        "PAID_SOURCE_FALLBACK_CONSIDERED": False,
        "PAID_API_CALLS": 0,
        "LIVE_OUTCOMES_READ": int(counters.get("LIVE_OUTCOMES_READ", 0)),
        "LIVE_TARGETS_COMPUTED": int(counters.get("LIVE_TARGETS_COMPUTED", 0)),
        "LIVE_POST_EVENT_PRICE_READS": int(counters.get("LIVE_POST_EVENT_PRICE_READS", 0)),
        "LIVE_MODEL_PREDICTIONS": 0,
        "MODEL_TRAINING_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
        "OLD_FUTURE_HOLDOUT_OPENED": False,
        "BROKER_MUTATIONS": int(counters.get("BROKER_MUTATIONS", 0)),
        "REAL_TRADING_ALLOWED": False,
    }


def _controlled_registry_payload() -> dict[str, Any]:
    return {
        "historical_frozen_issuer_tickers": list(HISTORICAL_ISSUER_TICKERS),
        "milestone": {
            "minimum_new_issuer_tickers": 3,
            "minimum_total_issuer_tickers": 10,
            "name": "LIVE_DIVERSITY_MILESTONE_V1",
        },
        "source_registry_version": "live-issuer-sources-v1",
        "sources": [
            _controlled_source(
                "https://issuer-a.test/rss", "issuer-a.test", "AAA", "AAA_SOURCE_ISOLATION_V4"
            ),
            _controlled_source(
                "https://issuer-b.test/rss", "issuer-b.test", "BBB", "BBB_SOURCE_ISOLATION_V4"
            ),
        ],
    }


def _controlled_source(url: str, domain: str, ticker: str, source_id: str) -> dict[str, Any]:
    return {
        "canonical_domain": domain,
        "content_path": ["rss.channel.item.title", "rss.channel.item.description"],
        "discovery_type": "official_issuer_rss",
        "discovery_url": url,
        "enabled": True,
        "expected_publication_frequency": "controlled proof",
        "identity_path": "rss.channel.item.guid || rss.channel.item.link",
        "issuer": f"{ticker} Issuer",
        "official_domain": domain,
        "parser": "rss-item-pubdate-explicit-offset-v1",
        "polling_policy": {"interval_minutes": 60, "max_items_per_poll": 5},
        "source_id": source_id,
        "source_origin": "ISSUER_ORIGINATED",
        "source_status": "LIVE_STRICT_EXACT_READY",
        "source_version": 1,
        "stable_identity": "rss_guid_or_link",
        "ticker": ticker,
        "ticker_binding": {"binding": "single_issuer_source"},
        "timestamp_contract": {
            "evidence_type": "TIMESTAMP_EVIDENCE_TYPE=RFC822_EXPLICIT_OFFSET",
            "evidence_value": "RSS pubDate includes +0300",
            "policy": "accept explicit offset only",
        },
        "timestamp_path": "rss.channel.item.pubDate",
    }


def _rss_item(ticker: str, suffix: str, *, domain: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>{ticker} headline {suffix}</title>
      <description>{ticker} description {suffix}</description>
      <link>https://{domain}/news/{suffix}</link>
      <guid>{ticker}-{suffix}</guid>
      <pubDate>Tue, 25 Aug 2026 11:44:16 +0300</pubDate>
    </item>
  </channel>
</rss>
""".encode()


def _failure(url: str) -> FetchResult:
    return FetchResult(url, url, 500, None, b"", 0, (), "HTTP_FAILURE")


class _ScriptedIsolationClient:
    def __init__(self, responses: dict[str, list[bytes | FetchResult]]) -> None:
        self._responses = responses
        self._current_url: str | None = None
        self.calls: list[str] = []

    def for_source(self, url: str) -> _ScriptedIsolationClient:
        self._current_url = url
        return self

    def get(self, url: str) -> FetchResult:
        self.calls.append(url)
        responses = self._responses[url]
        response = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(response, FetchResult):
            return response
        return FetchResult(url, url, 200, "application/rss+xml", response, 0, (), None)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    target_candidates: Sequence[TargetCandidate],
    skipped: Sequence[dict[str, Any]],
    rejected_sources: Sequence[dict[str, Any]],
) -> None:
    rejected_by_blocker = Counter(str(row.get("blocker")) for row in rejected_sources)
    lines = [
        "# MOEX target source expansion v4",
        "",
        f"- BASE_MAIN_SHA: {manifest['BASE_MAIN_SHA']}",
        f"- HEAD_SHA: {manifest['HEAD_SHA']}",
        f"- OPERATIONAL_ISOLATION: {manifest['OPERATIONAL_ISOLATION']}",
        f"- OPERATIONAL_BURN_IN: {manifest['OPERATIONAL_BURN_IN']}",
        f"- TARGET_DIVERSITY: {manifest['TARGET_ELIGIBLE_DIVERSITY']}",
        f"- DIVERSITY: {manifest['DIVERSITY']}",
        f"- Final diversity status: {manifest['FINAL_DIVERSITY_STATUS']}",
        f"- Target candidate issuers: {len(target_candidates)}",
        f"- Skipped before network: {len(skipped)}",
        f"- Candidates actually probed: {manifest['CANDIDATES_ACTUALLY_PROBED']}",
        f"- New hypotheses tested: {manifest['NEW_HYPOTHESES_TESTED']}",
        f"- Official free sources READY: {manifest['OFFICIAL_FREE_SOURCES_READY']}",
        (f"- Target eligible READY sources: {manifest['NEW_TARGET_ELIGIBLE_SOURCE_COUNT']}"),
        (
            "- New target-eligible distinct legal issuers: "
            f"{manifest['NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUER_COUNT']}"
        ),
        f"- Rejected source blockers: {dict(sorted(rejected_by_blocker.items()))}",
        f"- Total live shadow events: {manifest['TOTAL_LIVE_SHADOW_EVENTS']}",
        f"- Live outcomes read: {manifest['LIVE_OUTCOMES_READ']}",
        f"- Targets computed: {manifest['LIVE_TARGETS_COMPUTED']}",
        f"- Post-event reads: {manifest['LIVE_POST_EVENT_PRICE_READS']}",
        f"- Broker mutations: {manifest['BROKER_MUTATIONS']}",
        "",
        "Dataset integrity remains closed: no outcomes, targets, model training, or trading.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\u0430-\u044f\u0451]+", " ", value.lower()).strip()
