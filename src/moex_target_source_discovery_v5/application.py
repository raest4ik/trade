from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from src.exact_event_live_official_collection.domain import SourceStatus as HttpSourceStatus
from src.exact_event_live_official_collection.http_client import (
    BoundedHttpClient,
    HttpClient,
)
from src.free_live_issuer_accumulation.domain import (
    live_accumulation_safety_flags,
    sha256_payload,
)
from src.free_live_issuer_accumulation.operation import (
    build_operation_status,
    verify_operation_seal,
)
from src.free_live_issuer_expansion_v2.application import (
    HISTORICAL_ISSUER_TICKERS,
    LIVE_READY_BASELINE_TICKERS,
    SourceProbeConfig,
    SourceProbeResult,
    SourceStatus,
    TimestampLevel,
    probe_candidate_source,
)
from src.free_live_operational_burnin_and_onboarding_v3.application import (
    APPROVED_BENCHMARK_TICKER,
    DEFAULT_INSTRUMENT_MAPPING_PATH,
    CanonicalTargetInstrument,
    TargetEligibilityResult,
    distinct_new_target_eligible_legal_issuers,
    diversity_eligibility_payload,
    evaluate_target_eligibility,
    load_instrument_mapping_rows,
)

ARTIFACT_VERSION = "moex-target-source-discovery-v5"
DEFAULT_OUTPUT_ROOT = Path(f"artifacts/{ARTIFACT_VERSION}")
DEFAULT_OPERATION_ROOT = Path("artifacts/free-live-research-operation-v1")
DEFAULT_PREVIOUS_V4_ROOT = Path("artifacts/moex-target-source-expansion-v4")
MAX_DISTINCT_ISSUER_CANDIDATES = 25
MIN_DISTINCT_DIVERSITY_ISSUERS = 3
V4_RECHECK_TICKERS = frozenset({"SBER", "GAZP", "LKOH", "NVTK", "VTBR"})


@dataclass(frozen=True, slots=True)
class Candidate:
    ticker: str
    legal_issuer: str
    legal_issuer_key: str
    rank: int
    score: int
    eligibility: TargetEligibilityResult
    mapping_row: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "ticker": self.ticker,
            "legal_issuer": self.legal_issuer,
            "legal_issuer_key": self.legal_issuer_key,
            "instrument_uid": self.mapping_row.get("instrument_uid"),
            "figi": self.mapping_row.get("figi"),
            "board": self.mapping_row.get("class_code"),
            "exchange": self.mapping_row.get("exchange"),
            "market_data_compatible": self.eligibility.canonical_mapping_ready,
            "feature_pipeline_compatible": self.eligibility.feature_pipeline_compatible,
            "score": self.score,
            "ranking_inputs": {
                "canonical_target_eligibility": self.eligibility.target_instrument_eligible,
                "deterministic_issuer_mapping": True,
                "supported_moex_instrument": True,
                "t_invest_market_data_mapping": self.eligibility.canonical_mapping_ready,
                "benchmark_compatibility": self.eligibility.benchmark_path_ready,
                "official_issuer_web_presence": self.ticker in default_v5_source_configs(),
                "machine_readable_publication_channel_probability": _channel_score(self.ticker),
                "publication_frequency_proxy": _frequency_score(self.ticker),
                "future_returns_or_ml_used": False,
            },
            "eligibility": self.eligibility.payload(),
        }


def run_moex_target_source_discovery_v5(
    *,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    operation_root: Path = DEFAULT_OPERATION_ROOT,
    instrument_mapping_path: Path = DEFAULT_INSTRUMENT_MAPPING_PATH,
    previous_v4_root: Path = DEFAULT_PREVIOUS_V4_ROOT,
    source_configs: Sequence[SourceProbeConfig] | None = None,
    client: HttpClient | None = None,
    network_check: bool = True,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable v5 output already exists")
    output_root.mkdir(parents=True, exist_ok=False)

    now = created_at or datetime.now(UTC)
    mapping_rows = load_instrument_mapping_rows(instrument_mapping_path)
    registry = canonical_registry_from_mapping(mapping_rows)
    previous_rejections = load_previous_v4_rejections(previous_v4_root)
    candidates, excluded = build_candidate_universe(
        mapping_rows=mapping_rows,
        canonical_registry=registry,
        previous_v4_rejections=previous_rejections,
    )
    if source_configs is None:
        configs_by_ticker = default_v5_source_configs()
    else:
        configs_by_ticker = {config.ticker.upper(): config for config in source_configs}
    probe_configs, hypothesis_rows, deferred = build_probe_plan(
        candidates,
        configs_by_ticker=configs_by_ticker,
        previous_v4_rejections=previous_rejections,
    )
    http = client or BoundedHttpClient(
        timeout_seconds=8.0,
        redirect_limit=3,
        max_response_bytes=512_000,
    )
    probe_results = (
        [probe_candidate_source(config, client=http, fetched_at=now) for config in probe_configs]
        if network_check
        else []
    )

    accepted_sources = [
        accepted_source_payload(result) for result in probe_results if source_ready(result)
    ]
    rejected_sources = [
        rejected_source_payload(result) for result in probe_results if not source_ready(result)
    ]
    source_eligibility = evaluate_target_eligibility(
        accepted_sources,
        canonical_registry=registry,
        instrument_mapping_rows=mapping_rows,
    )
    new_issuers = distinct_new_target_eligible_legal_issuers(source_eligibility)
    diversity = diversity_eligibility_payload(source_eligibility, new_issuers)
    live_status = build_operation_status(operation_root)
    live_seal = verify_operation_seal(operation_root)
    safety = safety_payload(live_status)
    burnin = operational_burnin_payload(live_status, live_seal, safety)
    blockers = Counter(
        row["current_status"]
        if row["current_status"] in {"TIMEOUT", "RECHECK_DEFERRED_NO_NEW_HYPOTHESIS"}
        else row.get("blocker") or row["current_status"]
        for row in [*rejected_sources, *deferred]
    )
    total_requests = sum(result.request_attempts for result in probe_results)
    previous_reprobed = [
        config.ticker for config in probe_configs if config.ticker.upper() in previous_rejections
    ]
    new_probed = [
        config.ticker
        for config in probe_configs
        if config.ticker.upper() not in previous_rejections
    ]

    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": now.isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "HEAD_SHA": git_sha,
        "FREE_SOURCES_ONLY": True,
        "PAID_SOURCES_USED": False,
        "PAID_API_CALLS": 0,
        "PAID_SOURCE_FALLBACK_CONSIDERED": False,
        "CANONICAL_TARGET_ISSUERS_CONSIDERED": len(candidates),
        "CANONICAL_TARGET_TICKERS_CONSIDERED": [candidate.ticker for candidate in candidates],
        "ISSUERS_EXCLUDED_BEFORE_NETWORK": len(excluded),
        "DISTINCT_ISSUERS_PROBED": len({config.ticker for config in probe_configs}),
        "TOTAL_NETWORK_REQUESTS": total_requests,
        "NEW_HYPOTHESES_TESTED": len(probe_configs),
        "PREVIOUS_CANDIDATES_REPROBED": len(previous_reprobed),
        "PREVIOUS_CANDIDATE_TICKERS_REPROBED": previous_reprobed,
        "NEW_CANDIDATES_PROBED": len(new_probed),
        "NEW_CANDIDATE_TICKERS_PROBED": new_probed,
        "FREE_OFFICIAL_SOURCES_READY": len(accepted_sources),
        "NEW_FREE_OFFICIAL_SOURCE_COUNT": len(accepted_sources),
        "TARGET_ELIGIBLE_SOURCES": diversity["NEW_TARGET_ELIGIBLE_SOURCE_COUNT"],
        "FEATURE_COMPATIBLE_SOURCES": diversity["FEATURE_PIPELINE_COMPATIBLE_SOURCE_COUNT"],
        "NEW_TARGET_ELIGIBLE_TICKERS": diversity["NEW_TARGET_ELIGIBLE_TICKER_COUNT"],
        "NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUERS": diversity[
            "NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUER_COUNT"
        ],
        "NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUER_COUNT": diversity[
            "NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUER_COUNT"
        ],
        "ACCEPTED_SOURCE_IDS": [source["source_id"] for source in accepted_sources],
        "BLOCKERS_BY_CATEGORY": dict(sorted(blockers.items())),
        "TARGET_DIVERSITY": diversity["TARGET_ELIGIBLE_DIVERSITY"],
        "TARGET_ELIGIBLE_DIVERSITY": diversity["TARGET_ELIGIBLE_DIVERSITY"],
        "DIVERSITY": diversity["DIVERSITY"],
        "FINAL_DIVERSITY_STATUS": diversity["FINAL_DIVERSITY_STATUS"],
        "LIVE_RESEARCH_OPERATION_STATUS": live_status["LIVE_RESEARCH_OPERATION_STATUS"],
        "OPERATIONAL_BURN_IN": burnin["OPERATIONAL_BURN_IN"],
        "OPERATION": burnin["OPERATION"],
        "SOURCE_READY_DOES_NOT_IMPLY_ML_DIVERSITY_ELIGIBLE": True,
        "ML_V2_DATASET_STATUS": "NOT_OPENED_BY_V5_SOURCE_DISCOVERY",
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
    _write_json(
        output_root / "target-universe.json",
        {
            "candidate_count": len(candidates),
            "candidates": [candidate.payload() for candidate in candidates],
            "excluded_before_network": excluded,
            "registry_source": str(instrument_mapping_path),
        },
    )
    _write_json(
        output_root / "candidate-ranking.json",
        {
            "ranking_basis": _ranking_basis(),
            "candidates": [candidate.payload() for candidate in candidates],
        },
    )
    _write_jsonl(
        output_root / "source-probes.jsonl",
        [source_probe_payload(result, now) for result in probe_results],
    )
    _write_json(output_root / "accepted-sources.json", {"sources": accepted_sources})
    _write_jsonl(output_root / "rejected-sources.jsonl", [*rejected_sources, *deferred])
    _write_jsonl(
        output_root / "target-mapping.jsonl",
        [mapping_payload(candidate) for candidate in candidates],
    )
    _write_jsonl(
        output_root / "instrument-eligibility.jsonl",
        [candidate.eligibility.payload() for candidate in candidates] + excluded,
    )
    _write_json(output_root / "diversity-status.json", diversity)
    _write_json(
        output_root / "candidate-backlog.json",
        {
            "next_recheck_date": (now.date() + timedelta(days=7)).isoformat(),
            "candidates": candidate_backlog(
                candidates,
                hypothesis_rows=hypothesis_rows,
                probe_results=probe_results,
                previous_v4_rejections=previous_rejections,
                checked_on=now.date(),
            ),
        },
    )
    _write_json(output_root / "safety.json", safety)
    _write_report(output_root / "report.md", manifest, candidates, excluded)
    return manifest


def canonical_registry_from_mapping(
    mapping_rows: Sequence[dict[str, Any]],
) -> dict[str, CanonicalTargetInstrument]:
    registry: dict[str, CanonicalTargetInstrument] = {}
    for row in mapping_rows:
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker or ticker == APPROVED_BENCHMARK_TICKER:
            continue
        if str(row.get("instrument_type")) != "INSTRUMENT_TYPE_SHARE":
            continue
        registry[ticker] = CanonicalTargetInstrument(
            ticker=ticker,
            legal_issuer=str(row.get("name") or ticker),
            primary_board=str(row.get("class_code") or "TQBR"),
            instrument_type=str(row.get("instrument_type") or "INSTRUMENT_TYPE_SHARE"),
        )
    return registry


def build_candidate_universe(
    *,
    mapping_rows: Sequence[dict[str, Any]],
    canonical_registry: dict[str, CanonicalTargetInstrument],
    previous_v4_rejections: dict[str, dict[str, Any]] | None = None,
    max_distinct_issuers: int = MAX_DISTINCT_ISSUER_CANDIDATES,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    previous = previous_v4_rejections or {}
    pseudo_sources = [
        {
            "source_id": f"{ticker}_V5_CANONICAL_TARGET_CHECK",
            "ticker": ticker,
            "legal_issuer": target.legal_issuer,
        }
        for ticker, target in sorted(canonical_registry.items())
    ]
    eligibility = evaluate_target_eligibility(
        pseudo_sources,
        canonical_registry=canonical_registry,
        instrument_mapping_rows=mapping_rows,
    )
    mapping_by_ticker = {
        str(row.get("ticker", "")).strip().upper(): dict(row) for row in mapping_rows
    }
    candidates: list[Candidate] = []
    excluded: list[dict[str, Any]] = []
    seen_issuer_keys: set[str] = set()
    for result in sorted(eligibility, key=lambda item: _candidate_sort_key(item, previous)):
        ticker = result.source_ticker
        legal_key = broad_legal_issuer_key(
            ticker=ticker,
            legal_issuer=result.source_legal_issuer,
        )
        if ticker in set(HISTORICAL_ISSUER_TICKERS):
            excluded.append(exclusion_payload(result, "FROZEN_HISTORICAL_ISSUER"))
            continue
        if ticker in set(LIVE_READY_BASELINE_TICKERS):
            excluded.append(exclusion_payload(result, "EXISTING_LIVE_READY_ISSUER"))
            continue
        if not result.target_instrument_eligible:
            excluded.append(exclusion_payload(result, result.blocker or "NOT_TARGET_ELIGIBLE"))
            continue
        if not result.feature_pipeline_compatible:
            excluded.append(exclusion_payload(result, "FEATURE_INCOMPATIBLE"))
            continue
        if legal_key in seen_issuer_keys:
            excluded.append(
                exclusion_payload(result, "DUPLICATE_LEGAL_ISSUER_SHARE_CLASS_COLLAPSE")
            )
            continue
        seen_issuer_keys.add(legal_key)
        if len(candidates) >= max_distinct_issuers:
            excluded.append(exclusion_payload(result, "CANDIDATE_BUDGET_LIMIT"))
            continue
        row = mapping_by_ticker[ticker]
        candidates.append(
            Candidate(
                ticker=ticker,
                legal_issuer=result.source_legal_issuer,
                legal_issuer_key=legal_key,
                rank=len(candidates) + 1,
                score=candidate_score(result, previous),
                eligibility=result,
                mapping_row=row,
            )
        )
    return candidates, excluded


def build_probe_plan(
    candidates: Sequence[Candidate],
    *,
    configs_by_ticker: dict[str, SourceProbeConfig],
    previous_v4_rejections: dict[str, dict[str, Any]],
) -> tuple[list[SourceProbeConfig], list[dict[str, Any]], list[dict[str, Any]]]:
    probe_configs: list[SourceProbeConfig] = []
    hypotheses: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for candidate in candidates:
        config = configs_by_ticker.get(candidate.ticker)
        previous = previous_v4_rejections.get(candidate.ticker)
        if config is None:
            hypotheses.append(
                hypothesis_payload(candidate, None, previous, "PENDING", network_probe=False)
            )
            continue
        if previous is not None and not recheck_has_new_hypothesis(config, previous):
            row = {
                **hypothesis_payload(
                    candidate,
                    config,
                    previous,
                    "RECHECK_DEFERRED_NO_NEW_HYPOTHESIS",
                    network_probe=False,
                ),
                "blocker": "RECHECK_DEFERRED_NO_NEW_HYPOTHESIS",
            }
            hypotheses.append(row)
            deferred.append(row)
            continue
        hypotheses.append(
            hypothesis_payload(candidate, config, previous, "PENDING", network_probe=True)
        )
        probe_configs.append(config)
    return probe_configs, hypotheses, deferred


def default_v5_source_configs() -> dict[str, SourceProbeConfig]:
    rows = [
        (
            "ALRS",
            "АЛРОСА",
            "www.alrosa.ru",
            "https://www.alrosa.ru/press-center/",
            "html-alternate-jsonld-v1",
            "first-party press-center JSON-LD/alternate feed probe",
        ),
        (
            "MTSS",
            "MTS",
            "ir.mts.ru",
            "https://ir.mts.ru/news_and_events/corporate_releases",
            "html-alternate-jsonld-v1",
            "first-party IR corporate releases page JSON-LD/alternate feed probe",
        ),
        (
            "PHOR",
            "ФосАгро",
            "www.phosagro.ru",
            "https://www.phosagro.ru/press/company/",
            "html-alternate-jsonld-v1",
            "first-party company news page JSON-LD/alternate feed probe",
        ),
        (
            "CHMF",
            "Северсталь",
            "severstal.com",
            "https://severstal.com/rus/media/news/",
            "html-alternate-jsonld-v1",
            "first-party media news page JSON-LD/alternate feed probe",
        ),
        (
            "MOEX",
            "Московская Биржа",
            "www.moex.com",
            "https://www.moex.com/export/news.aspx?cat=100",
            "rss-item-pubdate-explicit-offset-v1",
            "first-party MOEX official RSS with explicit pubDate probe",
        ),
        (
            "AFLT",
            "Аэрофлот",
            "www.aeroflot.ru",
            "https://www.aeroflot.ru/ru-ru/news",
            "html-alternate-jsonld-v1",
            "first-party news page JSON-LD/alternate feed probe",
        ),
        (
            "AFKS",
            "АФК Система",
            "sistema.com",
            "https://sistema.com/press/press-releases/",
            "html-alternate-jsonld-v1",
            "first-party press releases JSON-LD/alternate feed probe",
        ),
        (
            "AKRN",
            "Акрон",
            "www.acron.ru",
            "https://www.acron.ru/press-center/news/",
            "html-alternate-jsonld-v1",
            "first-party press center news JSON-LD/alternate feed probe",
        ),
        (
            "BSPB",
            "Банк Санкт-Петербург",
            "www.bspb.ru",
            "https://www.bspb.ru/news/",
            "html-alternate-jsonld-v1",
            "first-party bank news JSON-LD/alternate feed probe",
        ),
        (
            "CBOM",
            "МКБ",
            "mkb.ru",
            "https://mkb.ru/news",
            "html-alternate-jsonld-v1",
            "first-party bank news JSON-LD/alternate feed probe",
        ),
        (
            "FLOT",
            "Совкомфлот",
            "www.scf-group.ru",
            "https://www.scf-group.ru/press_office/press_releases/",
            "html-alternate-jsonld-v1",
            "first-party press releases JSON-LD/alternate feed probe",
        ),
        (
            "IRAO",
            "Inter RAO",
            "www.interrao.ru",
            "https://www.interrao.ru/press/news/",
            "html-alternate-jsonld-v1",
            "first-party press news JSON-LD/alternate feed probe",
        ),
        (
            "MAGN",
            "MMK",
            "mmk.ru",
            "https://mmk.ru/press-center/news/",
            "html-alternate-jsonld-v1",
            "first-party press center news JSON-LD/alternate feed probe",
        ),
        (
            "PIKK",
            "ПИК",
            "www.pik.ru",
            "https://www.pik.ru/news",
            "html-alternate-jsonld-v1",
            "first-party news JSON-LD/alternate feed probe",
        ),
        (
            "PLZL",
            "Полюс",
            "polyus.com",
            "https://polyus.com/en/media/press-releases/",
            "html-alternate-jsonld-v1",
            "first-party press releases JSON-LD/alternate feed probe",
        ),
        (
            "RUAL",
            "РУСАЛ",
            "rusal.ru",
            "https://rusal.ru/press-center/press-releases/",
            "html-alternate-jsonld-v1",
            "first-party press releases JSON-LD/alternate feed probe",
        ),
        (
            "SMLT",
            "Самолет",
            "samolet.ru",
            "https://samolet.ru/press-center/news/",
            "html-alternate-jsonld-v1",
            "first-party press-center JSON-LD/alternate feed probe",
        ),
        (
            "POSI",
            "Группа Позитив",
            "group-positive.com",
            "https://group-positive.com/press/news/",
            "html-alternate-jsonld-v1",
            "first-party press news JSON-LD/alternate feed probe",
        ),
        (
            "OZON",
            "Ozon",
            "corp.ozon.com",
            "https://corp.ozon.com/media/news/",
            "html-alternate-jsonld-v1",
            "first-party media news JSON-LD/alternate feed probe",
        ),
        (
            "HEAD",
            "HeadHunter",
            "ir.hh.ru",
            "https://ir.hh.ru/news-and-events/press-releases",
            "html-alternate-jsonld-v1",
            "first-party IR press releases JSON-LD/alternate feed probe",
        ),
        (
            "SBER",
            "ПАО Сбербанк",
            "www.sberbank.com",
            "https://www.sberbank.com/ru/investor-relations/news",
            "html-alternate-jsonld-v1",
            "alternate official investor-relations news page after V4 timeout",
        ),
        (
            "GAZP",
            "ПАО Газпром",
            "www.gazprom.ru",
            "https://www.gazprom.ru/press/news/",
            "html-alternate-jsonld-v1",
            "alternate official RU press news page after V4 timeout",
        ),
        (
            "LKOH",
            "ПАО ЛУКОЙЛ",
            "www.lukoil.ru",
            "https://www.lukoil.ru/PressCenter/Pressreleases",
            "html-alternate-jsonld-v1",
            "alternate official RU release archive after V4 timestamp blocker",
        ),
        (
            "NVTK",
            "ПАО НОВАТЭК",
            "www.novatek.ru",
            "https://www.novatek.ru/ru/press/releases/",
            "html-alternate-jsonld-v1",
            "alternate official RU release archive after V4 timestamp blocker",
        ),
        (
            "VTBR",
            "Банк ВТБ",
            "www.vtb.com",
            "https://www.vtb.com/o-banke/press-centr/novosti-i-press-relizy/",
            "html-alternate-jsonld-v1",
            "alternate official corporate press-center page after V4 technical blocker",
        ),
    ]
    return {
        ticker: SourceProbeConfig(
            source_id=f"{ticker}_{domain.upper().replace('.', '_')}_V5",
            ticker=ticker,
            legal_issuer=issuer,
            official_domain=domain,
            url=url,
            mechanism=mechanism,
            parser=parser,
            timestamp_field="rss.channel.item.pubDate"
            if parser.startswith("rss")
            else "jsonld.datePublished || discovered feed item timestamp",
            identity_field="rss.channel.item.guid || rss.channel.item.link"
            if parser.startswith("rss")
            else "jsonld.url || jsonld.@id || canonical first-party URL",
            content_fields=("rss.channel.item.title", "rss.channel.item.description")
            if parser.startswith("rss")
            else ("headline", "description", "articleBody"),
            new_hypothesis=mechanism,
            prior_rejection_source_id=f"{ticker}_V4_REJECTION"
            if ticker in V4_RECHECK_TICKERS
            else None,
        )
        for ticker, issuer, domain, url, parser, mechanism in rows
    }


def load_previous_v4_rejections(previous_v4_root: Path) -> dict[str, dict[str, Any]]:
    path = previous_v4_root / "rejected-sources.jsonl"
    if not path.exists():
        return {}
    rows = _read_jsonl(path)
    return {str(row.get("ticker", "")).upper(): row for row in rows}


def recheck_has_new_hypothesis(config: SourceProbeConfig, previous: dict[str, Any]) -> bool:
    if not config.new_hypothesis.strip():
        return False
    previous_url = str(previous.get("url") or "")
    previous_mechanism = str(previous.get("new_hypothesis") or previous.get("mechanism") or "")
    return config.url != previous_url or config.mechanism != previous_mechanism


def source_ready(result: SourceProbeResult) -> bool:
    return (
        result.status == SourceStatus.LIVE_STRICT_EXACT_READY
        and result.timestamp_level in {TimestampLevel.LEVEL_A, TimestampLevel.LEVEL_B}
        and result.real_item_observed
        and all(
            stable_identity(item.source_item_id, item.canonical_url)
            for item in result.items_observed
        )
    )


def stable_identity(source_item_id: str, canonical_url: str) -> bool:
    if not source_item_id.strip() or not canonical_url.strip():
        return False
    return all(
        "://" not in value or value.startswith("https://")
        for value in (source_item_id.strip(), canonical_url.strip())
    )


def accepted_source_payload(result: SourceProbeResult) -> dict[str, Any]:
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
        "FREE_OFFICIAL_SOURCE_READY": True,
        "SOURCE_READY": True,
        "NEW_ITEM_OBSERVED": result.real_item_observed,
        "first_item": {
            "source_item_id": first.source_item_id,
            "canonical_url": first.canonical_url,
            "published_at": first.published_at.isoformat(),
            "published_raw": first.published_raw,
            "title": first.title,
        },
    }
    return payload | {"contract_sha": sha256_payload(payload)}


def rejected_source_payload(result: SourceProbeResult) -> dict[str, Any]:
    current_status = candidate_status(result)
    first = result.items_observed[0] if result.items_observed else None
    return {
        "source_id": result.config.source_id,
        "ticker": result.config.ticker,
        "legal_issuer": result.config.legal_issuer,
        "url": result.config.url,
        "mechanism": result.config.mechanism,
        "status": result.status.value,
        "current_status": current_status,
        "blocker": result.blocker,
        "timestamp_level": result.timestamp_level.value,
        "published_raw": None if first is None else first.published_raw,
        "identity_evidence": None if first is None else first.source_item_id,
        "new_hypothesis": result.config.new_hypothesis,
        "paid_fallback_considered": False,
    }


def source_probe_payload(result: SourceProbeResult, observed_at: datetime) -> dict[str, Any]:
    response = result.response
    first = result.items_observed[0] if result.items_observed else None
    return {
        "source_id": result.config.source_id,
        "ticker": result.config.ticker,
        "legal_issuer": result.config.legal_issuer,
        "official_url": result.config.url,
        "official_domain": result.config.official_domain,
        "mechanism": result.config.mechanism,
        "parser": result.config.parser,
        "previous_rejection_source_id": result.config.prior_rejection_source_id,
        "new_hypothesis": result.config.new_hypothesis,
        "current_source_status": candidate_status(result),
        "raw_status": result.status.value,
        "blocker": result.blocker,
        "http_status": None if response is None else response.status,
        "final_url": None if response is None else response.final_url,
        "content_type": None if response is None else response.content_type,
        "request_attempts": result.request_attempts,
        "max_items": 5,
        "max_redirects": 3,
        "max_response_bytes": 512_000,
        "items_observed": len(result.items_observed),
        "observed_item": None
        if first is None
        else {
            "source_item_id": first.source_item_id,
            "canonical_url": first.canonical_url,
            "raw_timestamp": first.published_raw,
            "parsed_timestamp": first.published_at.isoformat(),
            "timezone_evidence": first.timestamp_level.value,
            "title": first.title,
        },
        "alternate_links": list(result.alternate_links),
        "observed_at": observed_at.isoformat(),
    }


def candidate_status(result: SourceProbeResult) -> str:
    if result.status == SourceStatus.LIVE_STRICT_EXACT_READY:
        return "STRICT_EXACT_READY" if source_ready(result) else "STABLE_IDENTITY_UNVERIFIED"
    if result.status in {
        SourceStatus.LIVE_TIMESTAMP_UNVERIFIED,
        SourceStatus.LIVE_DATE_ONLY,
        SourceStatus.LIVE_CLOCK_WITHOUT_TIMEZONE,
    }:
        return "TIMESTAMP_UNVERIFIED"
    if result.status == SourceStatus.LIVE_NO_STABLE_ID:
        return "STABLE_IDENTITY_UNVERIFIED"
    if result.status == SourceStatus.LIVE_TECHNICAL_BLOCKER:
        return (
            "TIMEOUT" if result.blocker == HttpSourceStatus.TIMEOUT.value else "TECHNICAL_BLOCKER"
        )
    return "NOT_MACHINE_READABLE"


def candidate_backlog(
    candidates: Sequence[Candidate],
    *,
    hypothesis_rows: Sequence[dict[str, Any]],
    probe_results: Sequence[SourceProbeResult],
    previous_v4_rejections: dict[str, dict[str, Any]],
    checked_on: date,
) -> list[dict[str, Any]]:
    hypothesis_by_ticker = {str(row["ticker"]): row for row in hypothesis_rows}
    result_by_ticker = {result.config.ticker: result for result in probe_results}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        hypothesis = hypothesis_by_ticker.get(candidate.ticker, {})
        result = result_by_ticker.get(candidate.ticker)
        previous = previous_v4_rejections.get(candidate.ticker, {})
        rows.append(
            {
                "legal_issuer": candidate.legal_issuer,
                "canonical_tickers": [candidate.ticker],
                "instrument_uid": candidate.mapping_row.get("instrument_uid"),
                "board": candidate.mapping_row.get("class_code"),
                "market_data_compatible": candidate.eligibility.canonical_mapping_ready,
                "official_domains": [hypothesis.get("official_domain")]
                if hypothesis.get("official_domain")
                else [],
                "mechanism_attempted": hypothesis.get("new_mechanism"),
                "previous_blocker": previous.get("blocker"),
                "new_hypothesis": hypothesis.get("new_hypothesis"),
                "timestamp_status": None if result is None else result.timestamp_level.value,
                "identity_status": "VERIFIED"
                if result is not None and source_ready(result)
                else "UNVERIFIED",
                "current_source_status": hypothesis.get("current_status")
                if result is None
                else candidate_status(result),
                "next_free_hypothesis": next_free_hypothesis(
                    candidate.ticker,
                    None if result is None else candidate_status(result),
                ),
                "next_recheck_date": (checked_on + timedelta(days=7)).isoformat(),
            }
        )
    return rows


def next_free_hypothesis(ticker: str, status: str | None) -> str:
    if status == "STRICT_EXACT_READY":
        return "Bounded smoke ingestion only after explicit onboarding change."
    if ticker in V4_RECHECK_TICKERS:
        return "Search alternate first-party RSS/Atom or sitemap endpoint not used by V4/V5."
    return (
        "Search first-party RSS, Atom, JSON, JSON-LD, AJAX, sitemap, or investor disclosure feed."
    )


def mapping_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "ticker": candidate.ticker,
        "legal_issuer": candidate.legal_issuer,
        "official_source_to_legal_issuer": True,
        "canonical_ticker": candidate.ticker,
        "instrument_uid": candidate.mapping_row.get("instrument_uid"),
        "figi": candidate.mapping_row.get("figi"),
        "board": candidate.mapping_row.get("class_code"),
        "market_data_mapping_ready": candidate.eligibility.canonical_mapping_ready,
        "feature_pipeline_compatible": candidate.eligibility.feature_pipeline_compatible,
        "benchmark_ticker": APPROVED_BENCHMARK_TICKER,
    }


def hypothesis_payload(
    candidate: Candidate,
    config: SourceProbeConfig | None,
    previous: dict[str, Any] | None,
    current_status: str,
    *,
    network_probe: bool,
) -> dict[str, Any]:
    return {
        "ticker": candidate.ticker,
        "legal_issuer": candidate.legal_issuer,
        "previous_blocker": None if previous is None else previous.get("blocker"),
        "previous_mechanism": None if previous is None else previous.get("new_hypothesis"),
        "previous_url": None if previous is None else previous.get("url"),
        "official_domain": None if config is None else config.official_domain,
        "official_url": None if config is None else config.url,
        "new_mechanism": None if config is None else config.mechanism,
        "new_hypothesis": None if config is None else config.new_hypothesis,
        "network_probe_allowed": network_probe,
        "current_status": current_status,
        "NEW_HYPOTHESIS": config is not None and bool(config.new_hypothesis.strip()),
    }


def exclusion_payload(result: TargetEligibilityResult, reason: str) -> dict[str, Any]:
    payload = result.payload()
    payload["current_status"] = (
        "FEATURE_INCOMPATIBLE" if reason == "FEATURE_INCOMPATIBLE" else "NOT_TARGET_ELIGIBLE"
    )
    payload["skip_reason"] = reason
    payload["network_probe_allowed"] = False
    return payload


def broad_legal_issuer_key(*, ticker: str, legal_issuer: str) -> str:
    ticker_key = ticker.strip().upper()
    if ticker_key.endswith("P") and len(ticker_key) > 2:
        ticker_key = ticker_key[:-1]
    normalized = _normalize_text(legal_issuer)
    normalized = re.sub(
        r"\b(акции|акция|привилегированные|обыкновенные|пao|pao|pjsc|ao|jsc|public|company|group)\b",
        " ",
        normalized,
    )
    return " ".join(normalized.split()) or ticker_key.lower()


def candidate_score(
    result: TargetEligibilityResult, previous_v4_rejections: dict[str, dict[str, Any]]
) -> int:
    ticker = result.source_ticker
    return (
        100
        + (25 if result.target_instrument_eligible else 0)
        + (20 if result.canonical_mapping_ready else 0)
        + (20 if result.feature_pipeline_compatible else 0)
        + (15 if ticker in default_v5_source_configs() else 0)
        + _channel_score(ticker)
        + _frequency_score(ticker)
        - (15 if ticker in previous_v4_rejections else 0)
    )


def _candidate_sort_key(
    result: TargetEligibilityResult, previous_v4_rejections: dict[str, dict[str, Any]]
) -> tuple[int, str]:
    return (-candidate_score(result, previous_v4_rejections), result.source_ticker)


def _channel_score(ticker: str) -> int:
    return 10 if ticker in default_v5_source_configs() else 0


def _frequency_score(ticker: str) -> int:
    high_frequency = {"ALRS", "MTSS", "PHOR", "CHMF", "MOEX", "AFLT", "IRAO", "MAGN"}
    return 10 if ticker in high_frequency else 5 if ticker in default_v5_source_configs() else 0


def _ranking_basis() -> list[str]:
    return [
        "canonical target eligibility",
        "deterministic issuer mapping",
        "liquid/supported MOEX instrument proxy from mapping availability",
        "existing T-Invest market-data mapping",
        "IMOEX benchmark compatibility",
        "official issuer web presence",
        "probability of machine-readable publication channel",
        "publication frequency proxy",
        "no future returns, targets, outcomes, or ML metrics",
    ]


def safety_payload(status: dict[str, Any]) -> dict[str, Any]:
    counters = cast("dict[str, Any]", status.get("outcome_counters", {}))
    return {
        **live_accumulation_safety_flags(),
        "FREE_SOURCES_ONLY": True,
        "PAID_SOURCES_USED": False,
        "PAID_API_CALLS": 0,
        "PAID_SOURCE_FALLBACK_CONSIDERED": False,
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


def operational_burnin_payload(
    live_status: dict[str, Any], live_seal: dict[str, Any], safety: dict[str, Any]
) -> dict[str, Any]:
    safety_zero = all(
        safety[key] == expected
        for key, expected in {
            "LIVE_OUTCOMES_READ": 0,
            "LIVE_TARGETS_COMPUTED": 0,
            "LIVE_POST_EVENT_PRICE_READS": 0,
            "LIVE_MODEL_PREDICTIONS": 0,
            "MODEL_TRAINING_PERFORMED": False,
            "BACKTEST_PERFORMED": False,
            "OLD_FUTURE_HOLDOUT_OPENED": False,
            "BROKER_MUTATIONS": 0,
        }.items()
    )
    pass_gate = (
        live_status["LIVE_RESEARCH_OPERATION_STATUS"] == "READY"
        and live_seal["sealed_epoch_verified"] is True
        and int(live_status.get("timestamp_rejections", 0)) == 0
        and int(live_status.get("sealed_violations", 0)) == 0
        and safety_zero
    )
    return {
        "OPERATIONAL_BURN_IN": "PASS" if pass_gate else "FAIL",
        "OPERATION": "YES" if pass_gate else "NO",
        "LIVE_RESEARCH_OPERATION_STATUS": live_status["LIVE_RESEARCH_OPERATION_STATUS"],
        "seal_pass": live_seal["sealed_epoch_verified"],
        "safety_counters_zero": safety_zero,
    }


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\u0430-\u044f\u0451]+", " ", value.lower()).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    candidates: Sequence[Candidate],
    excluded: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# MOEX target source discovery v5",
        "",
        f"- BASE_MAIN_SHA: {manifest['BASE_MAIN_SHA']}",
        f"- HEAD_SHA: {manifest['HEAD_SHA']}",
        f"- ARTIFACT_SHA: {manifest['ARTIFACT_SHA']}",
        f"- Canonical target issuers considered: {len(candidates)}",
        f"- Issuers excluded before network: {len(excluded)}",
        f"- Distinct issuers probed: {manifest['DISTINCT_ISSUERS_PROBED']}",
        f"- Total network requests: {manifest['TOTAL_NETWORK_REQUESTS']}",
        f"- New hypotheses tested: {manifest['NEW_HYPOTHESES_TESTED']}",
        f"- Previous candidates reprobed: {manifest['PREVIOUS_CANDIDATES_REPROBED']}",
        f"- New candidates probed: {manifest['NEW_CANDIDATES_PROBED']}",
        f"- Free official sources ready: {manifest['FREE_OFFICIAL_SOURCES_READY']}",
        f"- Target-eligible sources: {manifest['TARGET_ELIGIBLE_SOURCES']}",
        f"- Feature-compatible sources: {manifest['FEATURE_COMPATIBLE_SOURCES']}",
        f"- New target-eligible tickers: {manifest['NEW_TARGET_ELIGIBLE_TICKERS']}",
        (
            "- New target-eligible distinct legal issuers: "
            f"{manifest['NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUERS']}"
        ),
        f"- Accepted sources: {manifest['ACCEPTED_SOURCE_IDS']}",
        f"- Blockers by category: {manifest['BLOCKERS_BY_CATEGORY']}",
        f"- TARGET_DIVERSITY: {manifest['TARGET_DIVERSITY']}",
        f"- LIVE_RESEARCH_OPERATION_STATUS: {manifest['LIVE_RESEARCH_OPERATION_STATUS']}",
        f"- OPERATIONAL_BURN_IN: {manifest['OPERATIONAL_BURN_IN']}",
        "",
        (
            "ML remains closed: no outcomes, targets, future holdout, backtest, "
            "model, or broker mutation."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
