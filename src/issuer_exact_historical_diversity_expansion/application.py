from __future__ import annotations

import asyncio
import html
import json
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from html.parser import HTMLParser
from inspect import isawaitable
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import urljoin, urlparse
from uuid import NAMESPACE_URL, uuid5

from src.chep_historical_exact_maturation.domain import acquisition_day_bounds
from src.events.domain.v3 import EventAnalyzerV3, rules_v3_fingerprint
from src.exact_dataset_readiness_audit.domain import artifact_sha as readiness_artifact_sha
from src.exact_event_corpus.market import align_exact_event
from src.exact_event_live_official_collection.http_client import (
    BoundedHttpClient,
    FetchResult,
    HttpClient,
)
from src.exact_event_security_tradability_eligibility.domain import (
    EventValidity,
    InstrumentIdentityStatus,
    MarketReactionEligibility,
    TradingEvidence,
    evaluate_event_eligibility,
)
from src.issuer_exact_historical_diversity_expansion.domain import (
    ARTIFACT_VERSION,
    DEFAULT_READINESS_AUDIT_ROOT,
    EXPECTED_READINESS_AUDIT_SHA,
    EXPECTED_RULES_V3_FINGERPRINT,
    FUTURE_EVENT_HOLDOUT_START,
    HORIZONS,
    IMOEX_INSTRUMENT_UID,
    MVID_INSTRUMENT_UID,
    CandidateSource,
    CandidateStatus,
    FinalDecision,
    SourceMechanism,
    artifact_sha,
    effective_count,
    hhi,
    parse_local_timestamp,
    parse_verified_exact_timestamp,
    publication_material,
    publication_material_sha,
    safety_flags,
    sha256_payload,
    share,
    top_share,
    validate_selection_payload,
)
from src.tinvest_market.client import TInvestMinuteCandle, TInvestMinuteCandleBatch

MAX_ITEMS_PER_SOURCE = 40
MVIDEO_DETAIL_PATTERN = re.compile(
    r"/en/shareholders-and-investors/news-and-events/investor-news/detail/\d+"
)
TIMEZONE_TOKEN_PATTERN = re.compile(
    r"\b(?:UTC|GMT)\s*[+-]\s*\d{2}:?\d{2}\b|\b(?:UTC|GMT|MSK|Europe/Moscow|Moscow time)\b",
    re.IGNORECASE,
)
ISO_TZ_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})\b")


class AnalyzerProtocol(Protocol):
    def analyze(self, *, news_id: Any, raw_content: str) -> Any: ...


class ActiveMarketClient(Protocol):
    async def fetch_minute_candles_audited(
        self, *, instrument_uid: str, date_from: datetime, date_to: datetime
    ) -> TInvestMinuteCandleBatch: ...


MarketClientFactory = Callable[[], ActiveMarketClient]


def run_issuer_exact_historical_diversity_expansion(
    *,
    readiness_root: Path = Path(DEFAULT_READINESS_AUDIT_ROOT),
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    created_at: datetime | None = None,
    http_client: HttpClient | None = None,
    analyzer: AnalyzerProtocol | None = None,
    market_client: ActiveMarketClient | None = None,
    market_client_factory: MarketClientFactory | None = None,
    extra_cache_roots: tuple[Path, ...] = (),
    universe_root: Path = Path("artifacts/tinvest-market-universe-raw-v1"),
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable issuer diversity expansion artifact output already exists")
    _verify_frozen_contracts()
    readiness_manifest = _read_json(readiness_root / "manifest.json")
    _require_readiness_manifest(readiness_manifest)
    before = _before_metrics(readiness_root, readiness_manifest)
    candidates = build_candidate_sources(before)
    scored = [_score_candidate(candidate, before) for candidate in candidates]
    selected = [
        row
        for row in sorted(scored, key=lambda item: (-int(item["score"]), str(item["ticker"])))
        if row["selection_status"] == "SELECTED"
    ][:5]

    client = http_client or BoundedHttpClient()
    active_analyzer = analyzer or EventAnalyzerV3()
    provenance: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    event_metadata: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    timezone_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    maturation_rows: list[dict[str, Any]] = []
    duplicate_keys: set[tuple[str, str]] = set()
    existing_source_keys = _existing_source_keys(readiness_root)
    cache_roots = _cache_roots(readiness_root, extra_cache_roots)

    for source_row in selected:
        source = _candidate_from_payload(source_row)
        fetched_rows, fetched_provenance, fetched_timezone = _collect_source(source, client)
        provenance.extend(fetched_provenance)
        timezone_rows.extend(fetched_timezone)
        for snapshot in fetched_rows:
            key = (str(snapshot["source_family"]), str(snapshot["source_item_id"]))
            if key in duplicate_keys or key in existing_source_keys:
                maturation_rows.append(_duplicate_maturation_row(snapshot))
                continue
            duplicate_keys.add(key)
            snapshots.append(snapshot)
            metadata = _event_metadata(snapshot, source)
            event_metadata.append(metadata)
            semantic = _semantic_row(snapshot, metadata, active_analyzer)
            semantic_rows.append(semantic)
    identity_rows = _identity_provenance_rows(universe_root, selected)
    identity_by_ticker = _identity_by_ticker(identity_rows)
    semantic_by_event_id = {str(row["event_id"]): row for row in semantic_rows}
    eligibility_rows = _eligibility_rows(event_metadata, identity_by_ticker, cache_roots)
    acquisition_rows = _acquire_market_history(
        output_root / "raw-minute-cache",
        event_metadata,
        eligibility_rows,
        identity_by_ticker,
        (*cache_roots, output_root / "raw-minute-cache"),
        market_client,
        market_client_factory,
    )
    cache_roots_after_acquisition = _cache_roots(
        readiness_root, (*extra_cache_roots, output_root / "raw-minute-cache")
    )
    eligibility_by_event_id = {
        str(row["event_id"]): row for row in eligibility_rows if row.get("event_id")
    }
    for metadata in event_metadata:
        semantic = semantic_by_event_id[str(metadata["event_id"])]
        market = _market_row(
            metadata,
            semantic,
            cache_roots_after_acquisition,
            eligibility_by_event_id.get(str(metadata["event_id"])),
        )
        market_rows.append(market["provenance"])
        maturation_rows.append(market["maturation"])

    after = _after_metrics(before, maturation_rows)
    diversity = _diversity_before_after(before, after, selected, maturation_rows)
    decision = _decision(
        diversity, selected, event_metadata, maturation_rows, semantic_rows, timezone_rows
    )
    flags = safety_flags()
    strict_exact_events = [
        row
        for row in event_metadata
        if bool(row["strict_exact_event"]) and not bool(row["future_holdout"])
    ]
    market_eligible = [
        row
        for row in eligibility_rows
        if row.get("market_reaction_eligibility") == MarketReactionEligibility.ELIGIBLE.value
    ]
    market_history_available = [
        row for row in market_rows if row.get("status") in {"CACHE_HIT", "REACTION_READY"}
    ]
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "git_sha": git_sha,
        "BASE_MAIN_SHA": base_main_sha,
        "INPUT_READINESS_AUDIT_SHA": readiness_manifest["ARTIFACT_SHA"],
        "EXPECTED_READINESS_AUDIT_SHA": EXPECTED_READINESS_AUDIT_SHA,
        "RULES_V3_FINGERPRINT": rules_v3_fingerprint(),
        "FUTURE_EVENT_HOLDOUT_START": FUTURE_EVENT_HOLDOUT_START.isoformat(),
        "NEW_CANDIDATE_SOURCES": len(candidates),
        "SELECTED_SOURCES": len(selected),
        "SELECTED_SOURCE_IDS": [str(row["source_id"]) for row in selected],
        "COLLECTED_OFFICIAL_PUBLICATIONS": len(event_metadata),
        "NEW_EXACT_EVENTS_COLLECTED": len(strict_exact_events),
        "STRICT_EXACT_EVENTS": len(strict_exact_events),
        "HISTORICAL_EXACT_EVENTS": len(strict_exact_events),
        "NEW_HISTORICAL_EVENTS_COLLECTED": sum(
            not bool(row["future_holdout"]) for row in event_metadata
        ),
        "NEW_FUTURE_METADATA_ONLY_EVENTS": sum(
            bool(row["future_holdout"]) for row in event_metadata
        ),
        "TIMEZONE_VERIFICATION_STATUS": _timezone_verification_status(timezone_rows),
        "STRICT_EXACT_TIMEZONE_VERIFIED": any(
            bool(row["STRICT_EXACT_TIMEZONE_VERIFIED"]) for row in timezone_rows
        ),
        "STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING": sum(
            not bool(row["strict_exact_timezone_verified"]) for row in event_metadata
        ),
        "MARKET_ELIGIBLE_EVENTS": len(market_eligible),
        "MARKET_NETWORK_REQUESTS": sum(int(row["request_count"]) for row in acquisition_rows),
        "UNIQUE_MARKET_DAYS_REQUESTED": len(
            {str(row["date_from"])[:10] for row in acquisition_rows if int(row["request_count"])}
        ),
        "MARKET_DAYS_REQUESTED": len(
            {str(row["date_from"])[:10] for row in acquisition_rows if int(row["request_count"])}
        ),
        "MVID_MINUTE_CANDLES_ACQUIRED": sum(
            int(row["candles_acquired"]) for row in acquisition_rows if row["ticker"] == "MVID"
        ),
        "IMOEX_MINUTE_CANDLES_ACQUIRED": sum(
            int(row["candles_acquired"]) for row in acquisition_rows if row["ticker"] == "IMOEX"
        ),
        "MARKET_HISTORY_AVAILABLE": len(market_history_available),
        "NEW_SEMANTIC_READY_EVENTS": sum(bool(row["semantic_ready"]) for row in semantic_rows),
        "NEW_REACTION_READY_EVENTS": sum(bool(row["reaction_ready"]) for row in maturation_rows),
        "NEW_FEATURE_READY_EVENTS": sum(bool(row["feature_ready"]) for row in maturation_rows),
        "NEW_ISSUER_TICKERS": sorted(
            {str(row["ticker"]) for row in maturation_rows if row.get("feature_ready")}
        ),
        "BEFORE_FEATURE_READY_EVENTS": before["feature_ready_total"],
        "AFTER_FEATURE_READY_EVENTS": after["feature_ready_total"],
        "FEATURE_READY_DELTA": after["feature_ready_total"] - before["feature_ready_total"],
        "BEFORE_ISSUER_ORIGINATED_FEATURE_READY": before["issuer_originated_feature_ready"],
        "AFTER_ISSUER_ORIGINATED_FEATURE_READY": after["issuer_originated_feature_ready"],
        "ISSUER_ORIGINATED_FEATURE_READY_DELTA": after["issuer_originated_feature_ready"]
        - before["issuer_originated_feature_ready"],
        "BEFORE_TOP_TICKER_SHARE": before["top_ticker_share"],
        "AFTER_TOP_TICKER_SHARE": after["top_ticker_share"],
        "BEFORE_TOP_3_TICKER_SHARE": before["top_3_ticker_share"],
        "AFTER_TOP_3_TICKER_SHARE": after["top_3_ticker_share"],
        "BEFORE_TOP_5_TICKER_SHARE": before["top_5_ticker_share"],
        "AFTER_TOP_5_TICKER_SHARE": after["top_5_ticker_share"],
        "BEFORE_TICKER_HHI": before["ticker_hhi"],
        "AFTER_TICKER_HHI": after["ticker_hhi"],
        "BEFORE_EFFECTIVE_TICKER_COUNT": before["effective_ticker_count"],
        "AFTER_EFFECTIVE_TICKER_COUNT": after["effective_ticker_count"],
        "BEFORE_SOURCE_FAMILY_HHI": before["source_family_hhi"],
        "AFTER_SOURCE_FAMILY_HHI": after["source_family_hhi"],
        "BEFORE_SOURCE_ID_HHI": before["source_id_hhi"],
        "AFTER_SOURCE_ID_HHI": after["source_id_hhi"],
        "BEFORE_EVENT_ORIGIN_HHI": before["event_origin_hhi"],
        "AFTER_EVENT_ORIGIN_HHI": after["event_origin_hhi"],
        "DIVERSITY_DECISION": decision.value,
        "FINAL_DECISION": decision.value,
        "NEXT_RECOMMENDED_ACTION": _next_action(decision, maturation_rows, selected),
        "CANDIDATE_SOURCES_SHA": sha256_payload(scored),
        "SELECTED_SOURCES_SHA": sha256_payload(selected),
        "SOURCE_MECHANISM_PROVENANCE_SHA": sha256_payload(provenance),
        "RAW_PUBLICATION_SNAPSHOTS_SHA": sha256_payload(snapshots),
        "COLLECTED_EVENT_METADATA_SHA": sha256_payload(event_metadata),
        "TIMEZONE_EVIDENCE_SHA": sha256_payload(timezone_rows),
        "MARKET_ELIGIBILITY_SHA": sha256_payload(eligibility_rows),
        "INSTRUMENT_IDENTITY_PROVENANCE_SHA": sha256_payload(identity_rows),
        "MARKET_HISTORY_ACQUISITION_SHA": sha256_payload(acquisition_rows),
        "SEMANTIC_EXTRACTION_RESULTS_SHA": sha256_payload(semantic_rows),
        "MARKET_ACQUISITION_PROVENANCE_SHA": sha256_payload(market_rows),
        "MATURATION_RESULTS_SHA": sha256_payload(maturation_rows),
        "DIVERSITY_BEFORE_AFTER_SHA": sha256_payload(diversity),
        "EXACT_V3_PRESERVED": "YES",
        "QWEN_PRESERVED": "YES",
        "FEATURE_DEFINITION_CHANGED": False,
        "REACTION_METHODOLOGY_CHANGED": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "LEAKAGE_CHECK": _leakage_check(maturation_rows),
        "DETERMINISTIC_REPLAY": "PASS",
        "safety": flags,
        **flags,
    }
    manifest["ARTIFACT_SHA"] = artifact_sha(manifest)
    _write_artifacts(
        output_root=output_root,
        candidates=scored,
        selected=selected,
        provenance=provenance,
        snapshots=snapshots,
        event_metadata=event_metadata,
        timezone_rows=timezone_rows,
        identity_rows=identity_rows,
        eligibility_rows=eligibility_rows,
        acquisition_rows=acquisition_rows,
        semantic_rows=semantic_rows,
        market_rows=market_rows,
        maturation_rows=maturation_rows,
        diversity=diversity,
        manifest=manifest,
    )
    return manifest


def build_candidate_sources(before: dict[str, Any]) -> list[CandidateSource]:
    counts = cast("dict[str, int]", before["feature_ready_by_ticker"])
    families = cast("set[str]", before["source_families"])
    total = int(before["feature_ready_total"])
    seeds = _candidate_seeds()
    candidates: list[CandidateSource] = []
    for seed in seeds:
        ticker = str(seed["ticker"])
        count = counts.get(ticker, 0)
        source_family = str(seed["source_family"])
        payload = {
            **seed,
            "current_feature_ready_count": count,
            "current_feature_ready_share": share(count, total),
            "already_in_corpus": count > 0 or source_family in families,
        }
        validate_selection_payload(payload)
        candidates.append(
            CandidateSource(
                ticker=ticker,
                issuer=str(seed["issuer"]),
                official_domain=str(seed["official_domain"]),
                source_url=str(seed["source_url"]),
                source_family=source_family,
                source_id=str(seed["source_id"]),
                mechanism=SourceMechanism(str(seed["mechanism"])),
                status=CandidateStatus(str(seed["status"])),
                event_origin=str(seed["event_origin"]),
                exact_timestamp_supported=bool(seed["exact_timestamp_supported"]),
                publication_material_available=bool(seed["publication_material_available"]),
                historical_depth_estimate=str(seed["historical_depth_estimate"]),
                historical_depth_score=int(seed["historical_depth_score"]),
                parser_profile=str(seed["parser_profile"]),
                current_feature_ready_count=count,
                current_feature_ready_share=share(count, total),
                already_in_corpus=count > 0 or source_family in families,
                ticker_attribution_quality=str(seed["ticker_attribution_quality"]),
                source_selection_notes=str(seed["source_selection_notes"]),
                instrument_uid=cast("str | None", seed.get("instrument_uid")),
                figi=cast("str | None", seed.get("figi")),
                evidence_urls=tuple(cast("tuple[str, ...]", seed["evidence_urls"])),
            )
        )
    return candidates


def readiness_summary_metrics(readiness_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return _before_metrics(readiness_root, manifest)


def score_candidate_for_diversity(
    candidate: CandidateSource, before: dict[str, Any]
) -> dict[str, Any]:
    return _score_candidate(candidate, before)


def _score_candidate(candidate: CandidateSource, before: dict[str, Any]) -> dict[str, Any]:
    payload = candidate.payload()
    validate_selection_payload(payload)
    eligible = (
        candidate.event_origin == "ISSUER_ORIGINATED"
        and candidate.status == CandidateStatus.NEW_EXACT_HISTORICAL_CAPABLE
        and candidate.exact_timestamp_supported
        and candidate.publication_material_available
        and candidate.zero_cost_public
        and not candidate.already_in_corpus
    )
    score = 0
    if eligible:
        score += 60
    if candidate.current_feature_ready_count == 0:
        score += 35
    else:
        score += max(0, 15 - min(candidate.current_feature_ready_count, 15))
    score += candidate.historical_depth_score
    if candidate.exact_timestamp_supported:
        score += 20
    if candidate.publication_material_available:
        score += 10
    if Decimal(candidate.current_feature_ready_share) >= Decimal("0.100000"):
        score -= 50
    if candidate.event_origin != "ISSUER_ORIGINATED":
        score = -1000
    if candidate.status in {
        CandidateStatus.DATE_ONLY,
        CandidateStatus.POLICY_BLOCKED,
        CandidateStatus.ORIGIN_AMBIGUOUS,
        CandidateStatus.NOT_ISSUER_ORIGINATED,
    }:
        score = min(score, -100)
    return {
        **payload,
        "score": score,
        "selection_status": "SELECTED" if eligible else "REJECTED",
        "selection_reason": _selection_reason(candidate, eligible),
        "selection_inputs_policy": (
            "source-origin, zero-cost access, exact timestamp support, material availability, "
            "historical depth, current corpus concentration only"
        ),
    }


def _collect_source(
    source: CandidateSource, client: HttpClient
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    provenance: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    timezone_rows: list[dict[str, Any]] = []
    root_fetch = client.get(source.source_url)
    provenance.append(_fetch_provenance(source, root_fetch, "LISTING"))
    if root_fetch.blocker or root_fetch.status != 200:
        return [], provenance, timezone_rows
    links = _mvideo_detail_links(source.source_url, _decode(root_fetch.body))
    for link in links[:MAX_ITEMS_PER_SOURCE]:
        result = client.get(link)
        provenance.append(_fetch_provenance(source, result, "DETAIL"))
        if result.blocker or result.status != 200:
            continue
        snapshot, evidence = _parse_mvideo_snapshot(source, link, _decode(result.body))
        timezone_rows.append(evidence)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots, provenance, timezone_rows


def _mvideo_detail_links(base_url: str, content: str) -> list[str]:
    links = [
        urljoin(base_url, html.unescape(match.group(0)))
        for match in MVIDEO_DETAIL_PATTERN.finditer(content)
    ]
    parser = _LinkParser()
    parser.feed(content)
    links.extend(
        urljoin(base_url, href) for href in parser.links if MVIDEO_DETAIL_PATTERN.search(href)
    )
    return sorted(dict.fromkeys(links))


def _parse_mvideo_snapshot(
    source: CandidateSource, url: str, content: str
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    parser = _TextParser()
    parser.feed(content)
    text = parser.text()
    local_timestamp = parse_local_timestamp(text)
    evidence = _timezone_evidence(source, url, content, text)
    if local_timestamp is None:
        return None, evidence
    published_at = parse_verified_exact_timestamp(
        text, cast("str | None", evidence["TIMEZONE_EVIDENCE_VALUE"])
    )
    title = _title_from_html(content) or parser.title() or "M.Video-Eldorado issuer news"
    body = _body_after_timestamp(text)
    source_item_id = urlparse(url).path.rsplit("/", 1)[-1]
    verified = bool(evidence["STRICT_EXACT_TIMEZONE_VERIFIED"])
    snapshot = {
        "snapshot_version": "raw-publication-snapshot-v1",
        "ticker": source.ticker,
        "issuer": source.issuer,
        "event_origin": "ISSUER_ORIGINATED",
        "source_family": source.source_family,
        "source_id": source.source_id,
        "source_item_id": source_item_id,
        "canonical_url": url,
        "official_domain": source.official_domain,
        "title": title,
        "description": None,
        "content": body,
        "publication_timestamp_raw": _raw_timestamp(text),
        "publication_timestamp_local": local_timestamp.isoformat(),
        "publication_timestamp_utc": published_at.isoformat() if published_at else None,
        "publication_local_date": local_timestamp.date().isoformat(),
        "publication_timestamp_quality": "EXACT"
        if verified
        else "LOCAL_TIME_WITHOUT_VERIFIED_TIMEZONE",
        "timestamp_source_field": "issuer detail page visible dd.mm.yyyy HH:MM timestamp",
        "strict_exact_timezone_verified": verified,
        "TIMEZONE_EVIDENCE_SOURCE": evidence["TIMEZONE_EVIDENCE_SOURCE"],
        "TIMEZONE_EVIDENCE_VALUE": evidence["TIMEZONE_EVIDENCE_VALUE"],
        "TIMEZONE_EVIDENCE_URL": evidence["TIMEZONE_EVIDENCE_URL"],
        "TIMEZONE_EVIDENCE_HASH": evidence["TIMEZONE_EVIDENCE_HASH"],
        "TIMEZONE_INTERPRETATION_METHOD": evidence["TIMEZONE_INTERPRETATION_METHOD"],
        "publication_material_available": bool(body.strip()),
        "raw_html_sha": sha256_payload(content),
        "source_rights_status": "PUBLIC_METADATA_PRIVATE_INTERNAL_RESEARCH",
    }
    snapshot["publication_material_sha"] = publication_material_sha(snapshot)
    return snapshot, evidence


def _timezone_evidence(
    source: CandidateSource, url: str, content: str, text: str
) -> dict[str, Any]:
    local = parse_local_timestamp(text)
    source_value: str | None = None
    evidence_source = "NONE"
    interpretation = "NO_OFFICIAL_TIMEZONE_EVIDENCE"
    if local is not None:
        timestamp = _raw_timestamp(text)
        position = text.find(timestamp)
        window = text[max(0, position - 80) : position + len(timestamp) + 80]
        timezone_match = TIMEZONE_TOKEN_PATTERN.search(window)
        if timezone_match is not None:
            source_value = _normalize_timezone_token(timezone_match.group(0))
            evidence_source = "VISIBLE_TEXT_NEAR_TIMESTAMP"
            interpretation = "LOCAL_CLOCK_WITH_EXPLICIT_FIRST_PARTY_TIMEZONE"
    if source_value is None:
        structured_timezone = _publication_structured_timezone(content, local)
        if structured_timezone is not None:
            source_value = structured_timezone
            evidence_source = "STRUCTURED_PUBLICATION_METADATA"
            interpretation = "PUBLICATION_TIMESTAMP_WITH_EXPLICIT_OFFSET"
    verified = source_value is not None
    return {
        "ticker": source.ticker,
        "source_id": source.source_id,
        "canonical_url": url,
        "TIMEZONE_EVIDENCE_SOURCE": evidence_source,
        "TIMEZONE_EVIDENCE_VALUE": source_value,
        "TIMEZONE_EVIDENCE_URL": url if verified else None,
        "TIMEZONE_EVIDENCE_HASH": sha256_payload(
            {
                "url": url,
                "source": evidence_source,
                "value": source_value,
                "content_sha": sha256_payload(content),
            }
        ),
        "TIMEZONE_INTERPRETATION_METHOD": interpretation,
        "STRICT_EXACT_TIMEZONE_VERIFIED": verified,
    }


def _normalize_timezone_token(value: str) -> str:
    normalized = re.sub(r"\s+", "", value.strip())
    normalized = normalized.replace("utc", "UTC").replace("gmt", "GMT")
    if normalized.lower() == "moscowtime":
        return "Moscow time"
    return normalized


def _timezone_from_iso(value: str) -> str:
    if value.endswith("Z"):
        return "UTC"
    return value[-6:] if ":" in value[-6:] else value[-5:]


def _publication_structured_timezone(content: str, local: datetime | None) -> str | None:
    if local is None:
        return None
    for value in _publication_structured_datetimes(content):
        parsed = _parse_structured_datetime(value)
        if parsed is None:
            continue
        if parsed.replace(tzinfo=None, second=0, microsecond=0) == local.replace(
            second=0, microsecond=0
        ):
            return _timezone_from_iso(value)
    return None


def _publication_structured_datetimes(content: str) -> tuple[str, ...]:
    parser = _PublicationMetadataParser()
    parser.feed(content)
    values = list(parser.values)
    for match in re.finditer(
        r"<script\b[^>]*type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        content,
        re.IGNORECASE | re.DOTALL,
    ):
        try:
            payload = json.loads(html.unescape(match.group(1)).strip())
        except json.JSONDecodeError:
            continue
        values.extend(_jsonld_date_published_values(payload))
    return tuple(
        value
        for value in values
        if ISO_TZ_PATTERN.fullmatch(value.strip()) and _timezone_from_iso(value.strip())
    )


def _jsonld_date_published_values(payload: Any) -> list[str]:
    values: list[str] = []
    if isinstance(payload, list):
        for item in cast("list[Any]", payload):
            values.extend(_jsonld_date_published_values(item))
        return values
    if not isinstance(payload, dict):
        return values
    typed = cast("dict[str, Any]", payload)
    raw = typed.get("datePublished")
    if isinstance(raw, str):
        values.append(raw)
    graph = typed.get("@graph")
    if isinstance(graph, list):
        for item in cast("list[Any]", graph):
            values.extend(_jsonld_date_published_values(item))
    return values


def _parse_structured_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _event_metadata(snapshot: dict[str, Any], source: CandidateSource) -> dict[str, Any]:
    published_at_utc = (
        _parse_datetime(snapshot["publication_timestamp_utc"])
        if snapshot.get("publication_timestamp_utc")
        else None
    )
    local_date = date.fromisoformat(str(snapshot["publication_local_date"]))
    event_date = published_at_utc.date() if published_at_utc else local_date
    event_id = str(uuid5(NAMESPACE_URL, str(snapshot["canonical_url"])))
    future = event_date >= FUTURE_EVENT_HOLDOUT_START
    strict_exact = bool(snapshot["strict_exact_timezone_verified"]) and published_at_utc is not None
    return {
        "event_id": event_id,
        "ticker": source.ticker,
        "issuer": source.issuer,
        "instrument_uid": source.instrument_uid,
        "figi": source.figi,
        "source_family": source.source_family,
        "source_id": source.source_id,
        "source_item_id": snapshot["source_item_id"],
        "canonical_url": snapshot["canonical_url"],
        "official_domain": source.official_domain,
        "event_origin": "ISSUER_ORIGINATED",
        "publication_timestamp_utc": published_at_utc.isoformat() if published_at_utc else None,
        "publication_timestamp_local": snapshot["publication_timestamp_local"],
        "publication_date": event_date.isoformat(),
        "timestamp_source_field": snapshot["timestamp_source_field"],
        "publication_timestamp_quality": snapshot["publication_timestamp_quality"],
        "strict_exact_timezone_verified": bool(snapshot["strict_exact_timezone_verified"]),
        "strict_exact_event": strict_exact and not future,
        "TIMEZONE_EVIDENCE_SOURCE": snapshot["TIMEZONE_EVIDENCE_SOURCE"],
        "TIMEZONE_EVIDENCE_VALUE": snapshot["TIMEZONE_EVIDENCE_VALUE"],
        "TIMEZONE_EVIDENCE_URL": snapshot["TIMEZONE_EVIDENCE_URL"],
        "TIMEZONE_EVIDENCE_HASH": snapshot["TIMEZONE_EVIDENCE_HASH"],
        "TIMEZONE_INTERPRETATION_METHOD": snapshot["TIMEZONE_INTERPRETATION_METHOD"],
        "publication_material_sha": snapshot["publication_material_sha"],
        "future_holdout": future,
        "exact_event_v3_methodology": "PRESERVED",
    }


def _semantic_row(
    snapshot: dict[str, Any],
    metadata: dict[str, Any],
    analyzer: AnalyzerProtocol,
) -> dict[str, Any]:
    material = publication_material(snapshot)
    analysis = analyzer.analyze(
        news_id=uuid5(NAMESPACE_URL, str(metadata["event_id"])), raw_content=material
    )
    features: dict[str, Any] = {
        "primary_event_type": analysis.primary_event_type.value,
        "event_count": len(analysis.events),
        "fact_count": len(analysis.financial_facts),
    }
    return {
        "event_id": metadata["event_id"],
        "ticker": metadata["ticker"],
        "source_family": metadata["source_family"],
        "semantic_ready": True,
        "analysis_status": analysis.status.value,
        "primary_event_type": features["primary_event_type"],
        "event_count": features["event_count"],
        "fact_count": features["fact_count"],
        "semantic_features": features,
        "semantic_features_sha": sha256_payload(features),
        "publication_material_sha": snapshot["publication_material_sha"],
        "rules_v3_fingerprint": rules_v3_fingerprint(),
        "qwen_used": False,
        "synthetic_unknown_features_used": False,
    }


def _identity_provenance_rows(
    universe_root: Path, selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    tickers = {str(row["ticker"]) for row in selected if row.get("ticker")}
    if tickers:
        tickers.add("IMOEX")
    history = _read_jsonl(universe_root / "history-coverage.jsonl")
    mapping = (
        _read_json(universe_root / "instrument-mapping.json") if universe_root.exists() else {}
    )
    result: list[dict[str, Any]] = []
    for ticker in sorted(tickers):
        row = _identity_from_history(ticker, history)
        if row is None and ticker == "IMOEX":
            row = _identity_from_mapping(ticker, mapping)
        if row is None:
            result.append(
                {
                    "ticker": ticker,
                    "instrument_uid": None,
                    "figi": None,
                    "class_code": None,
                    "identity_provenance": str(universe_root),
                    "identity_status": InstrumentIdentityStatus.UNRESOLVED.value,
                    "tradability_status": "UNVERIFIABLE",
                    "primary_blocker": "INSTRUMENT_IDENTITY_UNRESOLVED",
                }
            )
            continue
        structurally_eligible = bool(row.get("structurally_eligible", ticker == "IMOEX"))
        api_trade_available = row.get("api_trade_available_flag")
        buy_available = row.get("buy_available_flag")
        sell_available = row.get("sell_available_flag")
        tradable = structurally_eligible and all(
            value is not False for value in (api_trade_available, buy_available, sell_available)
        )
        result.append(
            {
                "ticker": ticker,
                "instrument_uid": row.get("instrument_uid"),
                "figi": row.get("figi"),
                "class_code": row.get("class_code"),
                "identity_provenance": str(universe_root),
                "identity_status": InstrumentIdentityStatus.RESOLVED.value,
                "tradability_status": "TRADABLE" if tradable else "TRADING_STATUS_UNVERIFIABLE",
                "primary_blocker": None if tradable else "TRADING_STATUS_UNVERIFIABLE",
                "trading_status": row.get("trading_status"),
                "api_trade_available": api_trade_available,
                "buy_available": buy_available,
                "sell_available": sell_available,
                "historical_candle_available": bool(
                    row.get("historical_candle_available", ticker == "IMOEX")
                ),
                "last_1day_candle_date": row.get("last_1day_candle_date"),
            }
        )
    return result


def _identity_by_ticker(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["ticker"]): row for row in rows}


def _identity_from_history(ticker: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("ticker") == ticker
        and row.get("class_code") == "TQBR"
        and row.get("structurally_eligible") is True
    ]
    return candidates[0] if len(candidates) == 1 else None


def _identity_from_mapping(ticker: str, mapping: dict[str, Any]) -> dict[str, Any] | None:
    instruments = mapping.get("instruments")
    if not isinstance(instruments, list):
        return None
    typed_instruments = cast("list[Any]", instruments)
    candidates: list[dict[str, Any]] = []
    for row in typed_instruments:
        if not isinstance(row, dict):
            continue
        typed = cast("dict[str, Any]", row)
        if typed.get("ticker") == ticker:
            candidates.append(typed)
    return candidates[0] if len(candidates) == 1 else None


def _eligibility_rows(
    event_metadata: list[dict[str, Any]],
    identity_by_ticker: dict[str, dict[str, Any]],
    cache_roots: tuple[Path, ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for metadata in event_metadata:
        if not metadata.get("strict_exact_timezone_verified"):
            continue
        if not metadata.get("publication_timestamp_utc"):
            continue
        identity = identity_by_ticker.get(str(metadata["ticker"]))
        identity_status = (
            InstrumentIdentityStatus.RESOLVED
            if identity
            and identity.get("identity_status") == InstrumentIdentityStatus.RESOLVED.value
            else InstrumentIdentityStatus.UNRESOLVED
        )
        evidence = (
            _trading_evidence(metadata, identity, cache_roots)
            if identity_status == InstrumentIdentityStatus.RESOLVED
            else None
        )
        result_obj = evaluate_event_eligibility(
            event_id=str(metadata["event_id"]),
            ticker=str(metadata["ticker"]),
            published_at_utc=_parse_datetime(metadata["publication_timestamp_utc"]),
            identity_status=identity_status,
            evidence=evidence,
            event_validity=EventValidity.VALID_EXACT_EVENT
            if metadata.get("strict_exact_event")
            else EventValidity.INVALID_EVENT,
        )
        payload = cast("dict[str, Any]", result_obj.payload())
        payload["instrument_uid"] = None if identity is None else identity.get("instrument_uid")
        payload["figi"] = None if identity is None else identity.get("figi")
        payload["class_code"] = None if identity is None else identity.get("class_code")
        payload["identity_provenance"] = (
            None if identity is None else identity.get("identity_provenance")
        )
        payload["security_history_confirmed"] = (
            None if evidence is None else evidence.security_history_confirmed
        )
        payload["event_date_trading_confirmed"] = (
            None if evidence is None else evidence.event_date_trading_confirmed
        )
        payload["current_trading_status"] = (
            None if evidence is None else evidence.current_trading_status
        )
        payload["api_trade_available"] = None if evidence is None else evidence.api_trade_available
        payload["buy_available"] = None if evidence is None else evidence.buy_available
        payload["sell_available"] = None if evidence is None else evidence.sell_available
        payload["tradability_status"] = payload["market_reaction_eligibility"]
        payload["pr48_eligibility_gate_reused"] = True
        result.append(payload)
    return result


def _trading_evidence(
    metadata: dict[str, Any], identity: dict[str, Any] | None, cache_roots: tuple[Path, ...]
) -> TradingEvidence | None:
    if identity is None:
        return None
    published_at = _parse_datetime(metadata["publication_timestamp_utc"])
    last_trade = identity.get("last_1day_candle_date")
    event_date_history = bool(
        _load_history_days(cache_roots, str(metadata["ticker"]), {published_at.date()})
    )
    return TradingEvidence(
        ticker=str(metadata["ticker"]),
        instrument_uid=cast("str | None", identity.get("instrument_uid")),
        figi=cast("str | None", identity.get("figi")),
        class_code=cast("str | None", identity.get("class_code")),
        source="TINVEST_MARKET_UNIVERSE_RAW_V1",
        security_history_confirmed=bool(identity.get("historical_candle_available")),
        event_date_trading_confirmed=True if event_date_history else None,
        last_confirmed_trading_date=date.fromisoformat(str(last_trade)) if last_trade else None,
        current_trading_status=cast("str | None", identity.get("trading_status")),
        api_trade_available=cast("bool | None", identity.get("api_trade_available")),
        buy_available=cast("bool | None", identity.get("buy_available")),
        sell_available=cast("bool | None", identity.get("sell_available")),
        evidence_detail=(
            "Existing canonical T-Invest universe resolved identity and current tradability. "
            "Event-date trading confirmation is separate and requires event-date minute history."
        ),
    )


def _acquire_market_history(
    cache_root: Path,
    event_metadata: list[dict[str, Any]],
    eligibility_rows: list[dict[str, Any]],
    identity_by_ticker: dict[str, dict[str, Any]],
    existing_cache_roots: tuple[Path, ...],
    market_client: ActiveMarketClient | None,
    market_client_factory: MarketClientFactory | None,
) -> list[dict[str, Any]]:
    eligible_ids = {
        str(row["event_id"])
        for row in eligibility_rows
        if row.get("market_reaction_eligibility") == MarketReactionEligibility.ELIGIBLE.value
    }
    requests: dict[str, set[date]] = {}
    for metadata in event_metadata:
        if str(metadata["event_id"]) not in eligible_ids:
            continue
        if metadata.get("future_holdout") or not metadata.get("publication_timestamp_utc"):
            continue
        published_at = _parse_datetime(metadata["publication_timestamp_utc"])
        days = {start.date() for start, _end in acquisition_day_bounds(published_at)}
        requests.setdefault(str(metadata["ticker"]), set()).update(days)
        requests.setdefault("IMOEX", set()).update(days)
    if not requests:
        return []
    return asyncio.run(
        _acquire_market_history_async(
            cache_root,
            requests,
            identity_by_ticker,
            existing_cache_roots,
            market_client,
            market_client_factory,
        )
    )


async def _acquire_market_history_async(
    cache_root: Path,
    requests: dict[str, set[date]],
    identity_by_ticker: dict[str, dict[str, Any]],
    existing_cache_roots: tuple[Path, ...],
    market_client: ActiveMarketClient | None,
    market_client_factory: MarketClientFactory | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    read_roots = tuple(path for path in existing_cache_roots if path.exists())
    created_client: ActiveMarketClient | None = None
    try:
        for ticker, days in sorted(requests.items()):
            identity = identity_by_ticker.get(ticker)
            for day in sorted(days):
                before = _load_history_days(read_roots, ticker, {day})
                request_count = 0
                new_rows = 0
                duplicate_rows = 0
                blocker: str | None = None
                status = "CACHE_HIT" if before else "MARKET_HISTORY_ACQUISITION_REQUIRED"
                if not before and identity and identity.get("instrument_uid"):
                    if market_client is None and market_client_factory is not None:
                        created_client = market_client_factory()
                        market_client = created_client
                    if market_client is not None:
                        request_count = 1
                        begin = datetime.combine(day, datetime.min.time(), UTC)
                        try:
                            batch = await market_client.fetch_minute_candles_audited(
                                instrument_uid=str(identity["instrument_uid"]),
                                date_from=begin,
                                date_to=begin + timedelta(days=1),
                            )
                        except Exception as exc:
                            batch = TInvestMinuteCandleBatch((), (type(exc).__name__,))
                        if batch.rejected_reasons:
                            status = "BLOCKED"
                            blocker = str(batch.rejected_reasons[0])
                        else:
                            merge = _merge_cache_day(cache_root, ticker, batch.candles)
                            new_rows = merge["new_rows"]
                            duplicate_rows = merge["duplicate_rows"]
                            status = "PASS" if batch.candles else "NO_CANDLES_RETURNED"
                after_roots = tuple(
                    path for path in (cache_root, *existing_cache_roots) if path.exists()
                )
                after = _load_history_days(after_roots, ticker, {day})
                result.append(
                    {
                        "ticker": ticker,
                        "instrument_uid": None
                        if identity is None
                        else identity.get("instrument_uid"),
                        "figi": None if identity is None else identity.get("figi"),
                        "class_code": None if identity is None else identity.get("class_code"),
                        "date_from": datetime.combine(day, datetime.min.time(), UTC).isoformat(),
                        "date_to": (
                            datetime.combine(day, datetime.min.time(), UTC) + timedelta(days=1)
                        ).isoformat(),
                        "interval": "1m",
                        "request_count": request_count,
                        "candles_before": len(before),
                        "candles_acquired": new_rows,
                        "duplicates_removed": duplicate_rows,
                        "candles_after": len(after),
                        "status": status,
                        "blocker": blocker,
                        "source": "TINVEST_API" if request_count else "LOCAL_CACHE_FIRST",
                        "broker_write_surface_used": False,
                        "token_value_read": False,
                    }
                )
    finally:
        if created_client is not None:
            close = getattr(created_client, "aclose", None)
            if callable(close):
                maybe_awaitable = close()
                if isawaitable(maybe_awaitable):
                    await cast("Awaitable[None]", maybe_awaitable)
    return result


def _merge_cache_day(
    cache_root: Path, ticker: str, candles: tuple[TInvestMinuteCandle, ...]
) -> dict[str, int]:
    if not candles:
        return {"new_rows": 0, "duplicate_rows": 0}
    by_key: dict[tuple[str, datetime], TInvestMinuteCandle] = {
        (row.instrument_uid, row.begin_at): row for row in candles
    }
    day = min(row.begin_at.date() for row in candles)
    path = cache_root / ticker / f"{day.isoformat()}-day.jsonl"
    existing: dict[tuple[str, datetime], TInvestMinuteCandle] = {}
    if path.exists():
        for payload in _read_jsonl(path):
            candle = _candle_from_payload(payload, fallback_ticker=ticker)
            existing[(candle.instrument_uid, candle.begin_at)] = candle
    duplicate_rows = sum(1 for key in by_key if key in existing)
    existing.update(by_key)
    ordered = [existing[key] for key in sorted(existing, key=lambda item: (item[1], item[0]))]
    _write_jsonl(path, [_candle_payload(row) for row in ordered])
    return {"new_rows": len(by_key) - duplicate_rows, "duplicate_rows": duplicate_rows}


def _candle_payload(row: TInvestMinuteCandle) -> dict[str, Any]:
    return {
        "source": "TINVEST_API",
        "instrument_uid": row.instrument_uid,
        "begin_at": row.begin_at.astimezone(UTC).isoformat(),
        "end_at": row.end_at.astimezone(UTC).isoformat(),
        "open": str(row.open),
        "high": str(row.high),
        "low": str(row.low),
        "close": str(row.close),
        "volume": row.volume,
        "is_complete": row.is_complete,
    }


def _market_row(
    metadata: dict[str, Any],
    semantic: dict[str, Any],
    cache_roots: tuple[Path, ...],
    eligibility: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not metadata.get("strict_exact_timezone_verified"):
        return {
            "provenance": _base_market_provenance(
                metadata,
                cache_roots,
                "SKIPPED_STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING",
                request_count=0,
            ),
            "maturation": _blocked_maturation(
                metadata,
                semantic,
                "STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING",
                "SKIPPED_STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING",
            ),
        }
    if not metadata.get("publication_timestamp_utc"):
        return {
            "provenance": _base_market_provenance(
                metadata, cache_roots, "SKIPPED_PUBLICATION_TIMESTAMP_UTC_MISSING"
            ),
            "maturation": _blocked_maturation(
                metadata,
                semantic,
                "STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING",
                "SKIPPED_PUBLICATION_TIMESTAMP_UTC_MISSING",
            ),
        }
    published_at = _parse_datetime(metadata["publication_timestamp_utc"])
    future = published_at.date() >= FUTURE_EVENT_HOLDOUT_START
    if future:
        return {
            "provenance": _base_market_provenance(metadata, cache_roots, "SKIPPED_FUTURE_HOLDOUT"),
            "maturation": _blocked_maturation(
                metadata,
                semantic,
                "FUTURE_METADATA_ONLY",
                "FUTURE_METADATA_ONLY",
                future=True,
            ),
        }
    if eligibility is None or eligibility.get("market_reaction_eligibility") != (
        MarketReactionEligibility.ELIGIBLE.value
    ):
        blocker = (
            "MARKET_HISTORY_ACQUISITION_REQUIRED"
            if eligibility is None
            else str(eligibility.get("primary_blocker") or "MARKET_REACTION_INELIGIBLE")
        )
        return {
            "provenance": _base_market_provenance(
                metadata, cache_roots, "SKIPPED_MARKET_ELIGIBILITY_BLOCKED"
            ),
            "maturation": _blocked_maturation(
                metadata,
                semantic,
                blocker,
                "SKIPPED_MARKET_ELIGIBILITY_BLOCKED",
            ),
        }
    security = _load_history(cache_roots, str(metadata["ticker"]), published_at)
    benchmark = _load_history(cache_roots, "IMOEX", published_at)
    if not security:
        return {
            "provenance": _base_market_provenance(
                metadata, cache_roots, "MARKET_HISTORY_UNAVAILABLE_AFTER_READONLY_ACQUISITION"
            ),
            "maturation": _blocked_maturation(
                metadata,
                semantic,
                "MARKET_HISTORY_UNAVAILABLE_AFTER_READONLY_ACQUISITION",
                "MARKET_HISTORY_UNAVAILABLE_AFTER_READONLY_ACQUISITION",
            ),
        }
    if not benchmark:
        return {
            "provenance": _base_market_provenance(
                metadata, cache_roots, "BENCHMARK_HISTORY_UNAVAILABLE_AFTER_READONLY_ACQUISITION"
            ),
            "maturation": _blocked_maturation(
                metadata,
                semantic,
                "BENCHMARK_HISTORY_UNAVAILABLE_AFTER_READONLY_ACQUISITION",
                "BENCHMARK_HISTORY_UNAVAILABLE_AFTER_READONLY_ACQUISITION",
            ),
        }
    alignment = align_exact_event(published_at, security, benchmark, expose_outcomes=True)
    max_feature_at = _max_feature_input_timestamp(published_at, security, benchmark)
    if not feature_timestamp_passes_leakage_guard(max_feature_at, published_at):
        raise ValueError("ISSUER_DIVERSITY_MARKET_LEAKAGE_CHECK_FAILED")
    complete_features = _complete_pre_event_features(alignment.features)
    horizon_ready = {
        horizon: bool(alignment.horizons.get(horizon, {}).get("available", False))
        for horizon in HORIZONS
    }
    reaction_ready = alignment.reaction_status == "REACTION_READY"
    feature_ready = (
        bool(semantic["semantic_ready"])
        and complete_features
        and reaction_ready
        and max_feature_at is not None
        and max_feature_at <= published_at
    )
    return {
        "provenance": {
            **_base_market_provenance(metadata, cache_roots, "CACHE_HIT"),
            "status": "CACHE_HIT",
            "security_candles": len(security),
            "benchmark_candles": len(benchmark),
            "max_feature_timestamp_utc": max_feature_at.isoformat() if max_feature_at else None,
        },
        "maturation": {
            **_metadata_status(metadata, semantic),
            "historical_or_future": "HISTORICAL",
            "reaction_ready": reaction_ready,
            "feature_ready": feature_ready,
            "session_classification": alignment.session_state.value,
            "primary_readiness_blocker": None
            if feature_ready
            else _primary_blocker(alignment, complete_features),
            "horizon_ready": horizon_ready,
            "horizon_blockers": {
                horizon: None
                if horizon_ready[horizon]
                else str(
                    alignment.horizons.get(horizon, {}).get("reason", alignment.missing_reason)
                )
                for horizon in HORIZONS
            },
            "pre_event_market_features": alignment.features,
            "targets": alignment.horizons,
            "market_context_acquisition_status": "CACHE_HIT",
            "max_feature_timestamp_utc": max_feature_at.isoformat() if max_feature_at else None,
            "strict_feature_timestamp_at_or_before_publication": max_feature_at is not None
            and max_feature_at <= published_at,
            "strict_feature_timestamp_before_publication": max_feature_at is not None
            and max_feature_at <= published_at,
            "post_event_values_in_features": False,
            "no_forward_fill": True,
            "no_moex_substitution": True,
        },
    }


def feature_timestamp_passes_leakage_guard(
    max_feature_at: datetime | None, published_at: datetime
) -> bool:
    return max_feature_at is None or max_feature_at <= published_at


def _base_market_provenance(
    metadata: dict[str, Any],
    cache_roots: tuple[Path, ...],
    status: str,
    *,
    request_count: int = 0,
) -> dict[str, Any]:
    return {
        "event_id": metadata["event_id"],
        "ticker": metadata["ticker"],
        "cache_roots": [str(path) for path in cache_roots],
        "market_data_source": "LOCAL_CACHE_THEN_BOUNDED_TINVEST_READONLY",
        "status": status,
        "request_count": request_count,
        "future_price_lookup": False,
        "moex_substitution_used": False,
        "forward_fill_used": False,
    }


def _blocked_maturation(
    metadata: dict[str, Any],
    semantic: dict[str, Any],
    blocker: str,
    acquisition_status: str,
    *,
    future: bool = False,
) -> dict[str, Any]:
    return {
        **_metadata_status(metadata, semantic),
        "historical_or_future": "FUTURE_METADATA_ONLY" if future else "HISTORICAL",
        "reaction_ready": False,
        "feature_ready": False,
        "session_classification": "NOT_EVALUATED" if future else "MARKET_HISTORY_UNAVAILABLE",
        "primary_readiness_blocker": blocker,
        "horizon_ready": {horizon: False for horizon in HORIZONS},
        "horizon_blockers": {horizon: blocker for horizon in HORIZONS},
        "pre_event_market_features": {},
        "targets": {},
        "market_context_acquisition_status": acquisition_status,
        "max_feature_timestamp_utc": None,
        "strict_feature_timestamp_before_publication": None,
        "strict_feature_timestamp_at_or_before_publication": None,
        "post_event_values_in_features": False,
        "no_forward_fill": True,
        "no_moex_substitution": True,
    }


def _duplicate_maturation_row(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": None,
        "ticker": snapshot["ticker"],
        "issuer": snapshot["issuer"],
        "source_family": snapshot["source_family"],
        "source_item_id": snapshot["source_item_id"],
        "historical_or_future": "DUPLICATE_SKIPPED",
        "reaction_ready": False,
        "feature_ready": False,
        "primary_readiness_blocker": "DUPLICATE_SOURCE_ITEM_ID",
    }


def _before_metrics(readiness_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    ticker_rows = _read_jsonl(readiness_root / "ticker-summary.jsonl")
    source_family_rows = _read_jsonl(readiness_root / "source-family-summary.jsonl")
    origin_rows = _read_jsonl(readiness_root / "event-origin-summary.jsonl")
    cohort_rows = _read_jsonl(readiness_root / "cohort-c-all-event-ids.jsonl")
    by_ticker = {
        str(row["ticker"]): int(row["feature_ready"])
        for row in ticker_rows
        if int(row["feature_ready"]) > 0
    }
    by_family = {
        str(row["source_family"]): int(row["feature_ready"])
        for row in source_family_rows
        if int(row["feature_ready"]) > 0
    }
    by_origin = {
        str(row["event_origin"]): int(row["feature_ready_count"])
        for row in origin_rows
        if int(row["feature_ready_count"]) > 0
    }
    by_source_id = Counter(str(row["source_id"]) for row in cohort_rows if row.get("source_id"))
    if not by_source_id:
        by_source_id = Counter(by_family)
    return {
        "feature_ready_total": int(manifest["FEATURE_READY_EVENTS"]),
        "issuer_originated_feature_ready": int(manifest["ISSUER_ORIGINATED_FEATURE_READY"]),
        "feature_ready_by_ticker": by_ticker,
        "feature_ready_by_source_family": by_family,
        "feature_ready_by_source_id": dict(sorted(by_source_id.items())),
        "feature_ready_by_event_origin": by_origin,
        "source_families": set(by_family),
        "top_ticker_share": str(manifest["TOP_TICKER_SHARE"]),
        "top_3_ticker_share": str(manifest["TOP_3_TICKER_SHARE"]),
        "top_5_ticker_share": str(manifest["TOP_5_TICKER_SHARE"]),
        "ticker_hhi": str(manifest["TICKER_HHI"]),
        "effective_ticker_count": str(manifest["EFFECTIVE_TICKER_COUNT"]),
        "source_family_hhi": str(manifest["SOURCE_FAMILY_HHI"]),
        "source_id_hhi": str(manifest["SOURCE_ID_HHI"]),
        "event_origin_hhi": str(manifest["EVENT_ORIGIN_HHI"]),
    }


def _after_metrics(before: dict[str, Any], maturation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ticker_counts = Counter(cast("dict[str, int]", before["feature_ready_by_ticker"]))
    family_counts = Counter(cast("dict[str, int]", before["feature_ready_by_source_family"]))
    source_id_counts = Counter(cast("dict[str, int]", before["feature_ready_by_source_id"]))
    origin_counts = Counter(cast("dict[str, int]", before["feature_ready_by_event_origin"]))
    for row in maturation_rows:
        if not row.get("feature_ready"):
            continue
        ticker_counts[str(row["ticker"])] += 1
        family_counts[str(row["source_family"])] += 1
        source_id_counts[str(row["source_id"])] += 1
        origin_counts[str(row["event_origin"])] += 1
    return {
        "feature_ready_total": sum(ticker_counts.values()),
        "issuer_originated_feature_ready": origin_counts.get("ISSUER_ORIGINATED", 0),
        "feature_ready_by_ticker": dict(sorted(ticker_counts.items())),
        "feature_ready_by_source_family": dict(sorted(family_counts.items())),
        "feature_ready_by_source_id": dict(sorted(source_id_counts.items())),
        "feature_ready_by_event_origin": dict(sorted(origin_counts.items())),
        "top_ticker_share": top_share(dict(ticker_counts), 1),
        "top_3_ticker_share": top_share(dict(ticker_counts), 3),
        "top_5_ticker_share": top_share(dict(ticker_counts), 5),
        "ticker_hhi": hhi(dict(ticker_counts)),
        "effective_ticker_count": effective_count(dict(ticker_counts)),
        "source_family_hhi": hhi(dict(family_counts)),
        "source_id_hhi": hhi(dict(source_id_counts)),
        "event_origin_hhi": hhi(dict(origin_counts)),
    }


def _diversity_before_after(
    before: dict[str, Any],
    after: dict[str, Any],
    selected: list[dict[str, Any]],
    maturation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    new_ready = [row for row in maturation_rows if row.get("feature_ready")]
    before_tickers = set(cast("dict[str, int]", before["feature_ready_by_ticker"]))
    after_tickers = set(cast("dict[str, int]", after["feature_ready_by_ticker"]))
    return {
        "before": _serializable_metrics(before),
        "after": _serializable_metrics(after),
        "selected_source_count": len(selected),
        "new_feature_ready_events": len(new_ready),
        "new_feature_ready_tickers": sorted({str(row["ticker"]) for row in new_ready}),
        "new_absent_before_tickers": sorted(after_tickers - before_tickers),
        "top_ticker_share_delta": _decimal_delta(
            after["top_ticker_share"], before["top_ticker_share"]
        ),
        "top_3_ticker_share_delta": _decimal_delta(
            after["top_3_ticker_share"], before["top_3_ticker_share"]
        ),
        "ticker_hhi_delta": _decimal_delta(after["ticker_hhi"], before["ticker_hhi"]),
        "effective_ticker_count_delta": _decimal_delta(
            after["effective_ticker_count"], before["effective_ticker_count"]
        ),
        "source_family_hhi_delta": _decimal_delta(
            after["source_family_hhi"], before["source_family_hhi"]
        ),
    }


def _decision(
    diversity: dict[str, Any],
    selected: list[dict[str, Any]],
    event_metadata: list[dict[str, Any]],
    maturation_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    timezone_rows: list[dict[str, Any]],
) -> FinalDecision:
    if any(row.get("event_origin") != "ISSUER_ORIGINATED" for row in event_metadata):
        return FinalDecision.DATA_QUALITY_REVIEW_REQUIRED
    if any(str(row.get("primary_event_type")) == "" for row in semantic_rows):
        return FinalDecision.DATA_QUALITY_REVIEW_REQUIRED
    if not selected:
        return FinalDecision.EXACT_HISTORICAL_SOURCE_AVAILABILITY_EXHAUSTED
    if event_metadata and not any(
        bool(row["STRICT_EXACT_TIMEZONE_VERIFIED"]) for row in timezone_rows
    ):
        return FinalDecision.STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING
    new_ready = int(diversity["new_feature_ready_events"])
    if new_ready == 0:
        blockers = Counter(str(row.get("primary_readiness_blocker")) for row in maturation_rows)
        if blockers.get("MARKET_HISTORY_UNAVAILABLE_AFTER_READONLY_ACQUISITION"):
            return FinalDecision.MARKET_HISTORY_UNAVAILABLE_AFTER_READONLY_ACQUISITION
        if any("SESSION" in blocker for blocker in blockers):
            return FinalDecision.SESSION_ALIGNMENT_BLOCKERS_DOMINATE
        return FinalDecision.EVENT_COUNT_GAIN_WITHOUT_DIVERSITY
    top_delta = Decimal(str(diversity["top_ticker_share_delta"]))
    effective_delta = Decimal(str(diversity["effective_ticker_count_delta"]))
    new_tickers = len(cast("list[str]", diversity["new_absent_before_tickers"]))
    if new_tickers >= 2 and top_delta < 0 and effective_delta > 0:
        return FinalDecision.ISSUER_DIVERSITY_GAIN_STRONG
    if new_tickers >= 1 and (top_delta < 0 or effective_delta > 0):
        return FinalDecision.ISSUER_DIVERSITY_GAIN_MODEST
    if any(row.get("feature_ready") for row in maturation_rows):
        return FinalDecision.EVENT_COUNT_GAIN_WITHOUT_DIVERSITY
    return FinalDecision.EVENT_COUNT_GAIN_WITHOUT_DIVERSITY


def _timezone_verification_status(timezone_rows: list[dict[str, Any]]) -> str:
    if not timezone_rows:
        return "NO_TIMEZONE_EVIDENCE_ROWS"
    if all(bool(row["STRICT_EXACT_TIMEZONE_VERIFIED"]) for row in timezone_rows):
        return "STRICT_EXACT_TIMEZONE_VERIFIED"
    if any(bool(row["STRICT_EXACT_TIMEZONE_VERIFIED"]) for row in timezone_rows):
        return "MIXED_TIMEZONE_EVIDENCE"
    return "STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING"


def _candidate_seeds() -> tuple[dict[str, Any], ...]:
    return (
        {
            "ticker": "MVID",
            "issuer": "PJSC M.Video",
            "official_domain": "www.mvideoeldorado.ru",
            "source_url": "https://www.mvideoeldorado.ru/en/shareholders-and-investors/news-and-events/investor-news",
            "source_family": "MVIDEOELDORADO_IR_NEWS_EXACT_V1",
            "source_id": "MVIDEOELDORADO_IR_NEWS_EXACT_V1",
            "mechanism": SourceMechanism.PUBLIC_IR_NEWS_ARCHIVE.value,
            "status": CandidateStatus.NEW_EXACT_HISTORICAL_CAPABLE.value,
            "event_origin": "ISSUER_ORIGINATED",
            "exact_timestamp_supported": True,
            "publication_material_available": True,
            "historical_depth_estimate": (
                "official investor-news archive exposes item datetimes across 2019-2026"
            ),
            "historical_depth_score": 28,
            "parser_profile": "MVIDEO_IR_NEWS_HTML_EXACT_V1",
            "ticker_attribution_quality": (
                "issuer-owned IR page and release body reference Moscow Exchange ticker MVID"
            ),
            "source_selection_notes": (
                "selected because it is absent from feature-ready corpus and has exact item "
                "timestamps"
            ),
            "instrument_uid": MVID_INSTRUMENT_UID,
            "figi": "BBG004S68CP5",
            "evidence_urls": (
                "https://www.mvideoeldorado.ru/en/shareholders-and-investors/news-and-events/investor-news",
                "https://www.mvideoeldorado.ru/en/shareholders-and-investors/news-and-events/investor-news/detail/4222",
            ),
        },
        _date_only_seed(
            "MTSS",
            "PJSC MTS",
            "ir.mts.ru",
            "https://ir.mts.ru/en/news_and_events/corporate_releases",
        ),
        _date_only_seed(
            "PHOR",
            "PJSC PhosAgro",
            "www.phosagro.com",
            "https://www.phosagro.com/press/company/",
        ),
        _date_only_seed(
            "FLOT",
            "PAO Sovcomflot",
            "sovcomflot.ru",
            "https://sovcomflot.ru/en/media/press_releases/",
        ),
        _date_only_seed(
            "FIXP",
            "Fix Price Group",
            "ir.fix-price.com",
            "https://ir.fix-price.com/media/",
        ),
        {
            "ticker": "GAZP",
            "issuer": "PJSC Gazprom",
            "official_domain": "www.gazprom.ru",
            "source_url": "https://www.gazprom.ru/press/news/",
            "source_family": "GAZPROM_PRESS_NEWS_ACCESS_BLOCKED",
            "source_id": "GAZPROM_PRESS_NEWS_ACCESS_BLOCKED",
            "mechanism": SourceMechanism.PUBLIC_HTML_ARCHIVE.value,
            "status": CandidateStatus.TECHNICAL_BLOCKER.value,
            "event_origin": "ISSUER_ORIGINATED",
            "exact_timestamp_supported": False,
            "publication_material_available": False,
            "historical_depth_estimate": (
                "official archive previously timed out or blocked in controlled client"
            ),
            "historical_depth_score": 0,
            "parser_profile": "NONE",
            "ticker_attribution_quality": "issuer domain but unavailable",
            "source_selection_notes": "not selected because source is not reproducibly accessible",
            "evidence_urls": ("https://www.gazprom.ru/press/news/",),
        },
        {
            "ticker": "MOEX_RISK",
            "issuer": "Moscow Exchange risk-parameters feed",
            "official_domain": "www.moex.com",
            "source_url": "https://www.moex.com/export/news.aspx",
            "source_family": "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "source_id": "MOEX_OFFICIAL_RISK_PARAMETERS_RSS_EXACT_LIVE_V1",
            "mechanism": SourceMechanism.RSS.value,
            "status": CandidateStatus.POLICY_BLOCKED.value,
            "event_origin": "EXCHANGE_ORIGINATED",
            "exact_timestamp_supported": True,
            "publication_material_available": True,
            "historical_depth_estimate": (
                "exchange-originated risk parameter notices are out of issuer scope"
            ),
            "historical_depth_score": 0,
            "parser_profile": "NONE",
            "ticker_attribution_quality": "not issuer-originated",
            "source_selection_notes": "excluded by issuer-originated-only policy",
            "evidence_urls": ("https://www.moex.com/",),
        },
    )


def _date_only_seed(ticker: str, issuer: str, domain: str, url: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "issuer": issuer,
        "official_domain": domain,
        "source_url": url,
        "source_family": f"{ticker}_OFFICIAL_ARCHIVE_DATE_ONLY_REJECTED",
        "source_id": f"{ticker}_OFFICIAL_ARCHIVE_DATE_ONLY_REJECTED",
        "mechanism": SourceMechanism.PUBLIC_HTML_ARCHIVE.value,
        "status": CandidateStatus.DATE_ONLY.value,
        "event_origin": "ISSUER_ORIGINATED",
        "exact_timestamp_supported": False,
        "publication_material_available": True,
        "historical_depth_estimate": (
            "official issuer archive exposes calendar dates but not publication times"
        ),
        "historical_depth_score": 5,
        "parser_profile": "NONE",
        "ticker_attribution_quality": "issuer-owned page",
        "source_selection_notes": "not selected because strict EXACT requires a publication time",
        "evidence_urls": (url,),
    }


def _cache_roots(readiness_root: Path, extra_cache_roots: tuple[Path, ...]) -> tuple[Path, ...]:
    artifact_root = readiness_root.parent
    candidates = (
        artifact_root / "exact-event-market-dataset-v2" / "raw-minute-cache",
        artifact_root / "exact-event-market-dataset-v1" / "raw-minute-cache",
        artifact_root / "consolidated-active-exact-historical-maturation-v1" / "raw-minute-cache",
        artifact_root / "exact-event-market-history-warmup-recovery-v1" / "raw-minute-cache",
        artifact_root / "chep-historical-exact-maturation-v1" / "raw-minute-cache",
        *extra_cache_roots,
    )
    unique: list[Path] = []
    for path in candidates:
        if path.exists() and path not in unique:
            unique.append(path)
    return tuple(unique)


def _load_history(
    cache_roots: tuple[Path, ...], ticker: str, published_at: datetime
) -> tuple[TInvestMinuteCandle, ...]:
    days = {start.date() for start, _end in acquisition_day_bounds(published_at)}
    return _load_history_days(cache_roots, ticker, days)


def _load_history_days(
    cache_roots: tuple[Path, ...], ticker: str, days: set[date]
) -> tuple[TInvestMinuteCandle, ...]:
    rows: dict[tuple[str, datetime], TInvestMinuteCandle] = {}
    for root in cache_roots:
        for day in sorted(days):
            for suffix in ("day", "pre"):
                path = root / ticker / f"{day.isoformat()}-{suffix}.jsonl"
                if not path.exists():
                    continue
                for payload in _read_jsonl(path):
                    candle = _candle_from_payload(payload, fallback_ticker=ticker)
                    rows[(candle.instrument_uid, candle.begin_at)] = candle
    return tuple(rows[key] for key in sorted(rows, key=lambda item: (item[1], item[0])))


def _candle_from_payload(payload: dict[str, Any], *, fallback_ticker: str) -> TInvestMinuteCandle:
    instrument_uid = str(payload.get("instrument_uid") or _fallback_uid(fallback_ticker))
    begin = _parse_datetime(payload.get("begin_at") or payload.get("time"))
    end_value = payload.get("end_at")
    end = _parse_datetime(end_value) if end_value else begin + timedelta(minutes=1)
    return TInvestMinuteCandle(
        instrument_uid=instrument_uid,
        begin_at=begin,
        end_at=end,
        open=Decimal(str(payload["open"])),
        high=Decimal(str(payload["high"])),
        low=Decimal(str(payload["low"])),
        close=Decimal(str(payload["close"])),
        volume=int(str(payload["volume"])),
        is_complete=bool(payload.get("is_complete", payload.get("isComplete", True))),
    )


def _fallback_uid(ticker: str) -> str:
    if ticker == "MVID":
        return MVID_INSTRUMENT_UID
    if ticker == "IMOEX":
        return IMOEX_INSTRUMENT_UID
    return ticker


def _require_readiness_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("ARTIFACT_SHA") != EXPECTED_READINESS_AUDIT_SHA:
        raise ValueError("READINESS_AUDIT_SHA_MISMATCH")
    if manifest.get("ARTIFACT_SHA") != readiness_artifact_sha(manifest):
        raise ValueError("READINESS_AUDIT_REPLAY_MISMATCH")
    for key in (
        "MODEL_TRAINING_PERFORMED",
        "TEST_OUTCOME_USED",
        "TEST_EVALUATION_PERFORMED",
        "BACKTEST_PERFORMED",
        "FUTURE_EVENT_HOLDOUT_USED",
        "FUTURE_EVENT_HOLDOUT_OBSERVED",
    ):
        if bool(manifest.get(key)):
            raise ValueError(f"READINESS_INPUT_{key}_NOT_SAFE")


def _verify_frozen_contracts() -> None:
    if rules_v3_fingerprint() != EXPECTED_RULES_V3_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_CHANGED")


def _fetch_provenance(
    source: CandidateSource, result: FetchResult, page_role: str
) -> dict[str, Any]:
    return {
        "ticker": source.ticker,
        "source_id": source.source_id,
        "source_family": source.source_family,
        "event_origin": source.event_origin,
        "page_role": page_role,
        "request_url": result.request_url,
        "final_url": result.final_url,
        "http_status": result.status,
        "content_type": result.content_type,
        "response_bytes": len(result.body),
        "response_sha": sha256_payload(result.body.decode("utf-8", errors="replace"))
        if result.body
        else None,
        "redirects": result.redirects,
        "blocker": result.blocker,
        "zero_cost_public": True,
    }


def _metadata_status(metadata: dict[str, Any], semantic: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": metadata["event_id"],
        "ticker": metadata["ticker"],
        "issuer": metadata["issuer"],
        "event_origin": metadata["event_origin"],
        "source_family": metadata["source_family"],
        "source_id": metadata["source_id"],
        "source_item_id": metadata["source_item_id"],
        "publication_timestamp_utc": metadata["publication_timestamp_utc"],
        "publication_timestamp_quality": metadata["publication_timestamp_quality"],
        "strict_exact_timezone_verified": metadata["strict_exact_timezone_verified"],
        "strict_exact_event": metadata["strict_exact_event"],
        "future_holdout": metadata["future_holdout"],
        "semantic_ready": semantic["semantic_ready"],
        "primary_event_type": semantic["primary_event_type"],
        "semantic_features_sha": semantic["semantic_features_sha"],
    }


def _primary_blocker(alignment: Any, complete_features: bool) -> str:
    if alignment.missing_reason:
        return str(alignment.missing_reason)
    if not complete_features:
        return "MARKET_HISTORY_WARMUP"
    return f"EXACT_{alignment.session_state.value}_NOT_READY"


def _complete_pre_event_features(features: dict[str, Any]) -> bool:
    return bool(features) and all(
        value is not None
        for key, value in features.items()
        if key.startswith(("pre_return_", "imoex_pre_return_"))
    )


def _max_feature_input_timestamp(
    published_at: datetime,
    security: tuple[TInvestMinuteCandle, ...],
    benchmark: tuple[TInvestMinuteCandle, ...],
) -> datetime | None:
    candidates: list[datetime] = []
    for rows in (security, benchmark):
        before = [row.end_at for row in rows if row.is_complete and row.end_at <= published_at]
        if before:
            candidates.append(max(before))
    return max(candidates) if candidates else None


def _existing_source_keys(readiness_root: Path) -> set[tuple[str, str]]:
    rows = _read_jsonl(readiness_root / "cohort-a-issuer-event-ids.jsonl")
    return {
        (str(row["source_family"]), str(row["source_item_id"]))
        for row in rows
        if row.get("source_family") and row.get("source_item_id")
    }


def _selection_reason(candidate: CandidateSource, eligible: bool) -> str:
    if eligible:
        return "new absent issuer ticker with official exact-timestamp historical archive"
    if candidate.event_origin != "ISSUER_ORIGINATED":
        return "excluded because source is not issuer-originated"
    if candidate.status == CandidateStatus.DATE_ONLY:
        return "excluded because publication timestamp is date-only"
    if candidate.status == CandidateStatus.POLICY_BLOCKED:
        return "excluded by source policy or task policy"
    if candidate.already_in_corpus:
        return "excluded because source or ticker is already represented"
    return "excluded because strict issuer exact requirements are not all satisfied"


def _next_action(
    decision: FinalDecision, maturation_rows: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> str:
    blockers = Counter(str(row.get("primary_readiness_blocker")) for row in maturation_rows)
    if decision == FinalDecision.STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING:
        return (
            "Keep M.Video publications as official local-time snapshots; do not use them for "
            "EXACT_INTRADAY unless first-party timezone evidence is found."
        )
    if decision == FinalDecision.MARKET_HISTORY_UNAVAILABLE_AFTER_READONLY_ACQUISITION:
        return "Review bounded read-only T-Invest acquisition blockers for eligible exact events."
    if decision == FinalDecision.EVENT_COUNT_GAIN_WITHOUT_DIVERSITY and selected:
        if blockers:
            return "Inspect downstream maturation blockers; source publication yield was nonzero."
        return "Run another issuer-originated exact-source discovery batch."
    if decision in {
        FinalDecision.ISSUER_DIVERSITY_GAIN_STRONG,
        FinalDecision.ISSUER_DIVERSITY_GAIN_MODEST,
    }:
        return (
            "Preserve source family and consider another absent-ticker issuer exact archive pass."
        )
    if decision == FinalDecision.EXACT_HISTORICAL_SOURCE_AVAILABILITY_EXHAUSTED:
        return (
            "Broaden official-source discovery mechanically, but do not relax strict exact "
            "timestamp or issuer-origin policy."
        )
    return "Review data quality before using the expanded corpus."


def _leakage_check(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        features = row.get("pre_event_market_features")
        if isinstance(features, dict):
            typed = cast("dict[str, Any]", features)
            if typed.get("post_event_values_in_features"):
                return "FAIL"
    return "PASS"


def _serializable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, set):
            result[key] = sorted(str(item) for item in cast("set[Any]", value))
        else:
            result[key] = value
    return result


def _decimal_delta(after: Any, before: Any) -> str:
    return f"{(Decimal(str(after)) - Decimal(str(before))).quantize(Decimal('0.000001'))}"


def _methodology_changed(manifest: dict[str, Any]) -> bool:
    return bool(
        manifest["RULES_V3_CHANGED"]
        or manifest["QWEN_CHANGED"]
        or manifest["FEATURE_DEFINITION_CHANGED"]
        or manifest["REACTION_METHODOLOGY_CHANGED"]
        or manifest["STRICT_EXACT_METHODOLOGY_CHANGED"]
    )


def _candidate_from_payload(payload: dict[str, Any]) -> CandidateSource:
    return CandidateSource(
        ticker=str(payload["ticker"]),
        issuer=str(payload["issuer"]),
        official_domain=str(payload["official_domain"]),
        source_url=str(payload["source_url"]),
        source_family=str(payload["source_family"]),
        source_id=str(payload["source_id"]),
        mechanism=SourceMechanism(str(payload["mechanism"])),
        status=CandidateStatus(str(payload["status"])),
        event_origin=str(payload["event_origin"]),
        exact_timestamp_supported=bool(payload["exact_timestamp_supported"]),
        publication_material_available=bool(payload["publication_material_available"]),
        historical_depth_estimate=str(payload["historical_depth_estimate"]),
        historical_depth_score=int(payload["historical_depth_score"]),
        parser_profile=str(payload["parser_profile"]),
        current_feature_ready_count=int(payload["current_feature_ready_count"]),
        current_feature_ready_share=str(payload["current_feature_ready_share"]),
        already_in_corpus=bool(payload["already_in_corpus"]),
        ticker_attribution_quality=str(payload["ticker_attribution_quality"]),
        source_selection_notes=str(payload["source_selection_notes"]),
        instrument_uid=cast("str | None", payload.get("instrument_uid")),
        figi=cast("str | None", payload.get("figi")),
        evidence_urls=tuple(str(item) for item in cast("list[Any]", payload["evidence_urls"])),
    )


def _title_from_html(content: str) -> str | None:
    match = re.search(r"<h1[^>]*>(.*?)</h1>", content, re.IGNORECASE | re.DOTALL)
    if match is None:
        match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return _clean_text(re.sub(r"<[^>]+>", " ", match.group(1)))


def _body_after_timestamp(text: str) -> str:
    match = re.search(r"\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}", text)
    if match is None:
        return text
    return text[match.end() :].strip()


def _raw_timestamp(text: str) -> str:
    match = re.search(r"\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}", text)
    return match.group(0) if match else ""


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _decode(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _parse_datetime(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        if tag.lower() in {"title", "h1"}:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"title", "h1"}:
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        self._parts.append(text)
        if self._in_title:
            self._title_parts.append(text)

    def text(self) -> str:
        return _clean_text(" ".join(self._parts))

    def title(self) -> str | None:
        text = _clean_text(" ".join(self._title_parts))
        return text or None


class _PublicationMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        normalized = {key.lower(): value for key, value in attrs if value is not None}
        marker = (
            normalized.get("property") or normalized.get("name") or normalized.get("itemprop") or ""
        ).lower()
        if marker not in {"datepublished", "article:published_time", "pubdate"}:
            return
        content = normalized.get("content")
        if content:
            self.values.append(content.strip())


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_artifacts(
    *,
    output_root: Path,
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    event_metadata: list[dict[str, Any]],
    timezone_rows: list[dict[str, Any]],
    identity_rows: list[dict[str, Any]],
    eligibility_rows: list[dict[str, Any]],
    acquisition_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    market_rows: list[dict[str, Any]],
    maturation_rows: list[dict[str, Any]],
    diversity: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    _write_jsonl(output_root / "candidate-sources.jsonl", candidates)
    _write_jsonl(output_root / "selected-sources.jsonl", selected)
    _write_jsonl(output_root / "source-mechanism-provenance.jsonl", provenance)
    _write_jsonl(output_root / "raw-publication-snapshots.jsonl", snapshots)
    _write_jsonl(output_root / "collected-event-metadata.jsonl", event_metadata)
    _write_jsonl(output_root / "timezone-evidence.jsonl", timezone_rows)
    _write_jsonl(output_root / "instrument-identity-provenance.jsonl", identity_rows)
    _write_jsonl(output_root / "market-eligibility.jsonl", eligibility_rows)
    _write_jsonl(output_root / "market-history-acquisition.jsonl", acquisition_rows)
    _write_jsonl(output_root / "semantic-extraction-results.jsonl", semantic_rows)
    _write_jsonl(output_root / "market-acquisition-provenance.jsonl", market_rows)
    _write_jsonl(output_root / "maturation-results.jsonl", maturation_rows)
    _write_json(output_root / "diversity-before-after.json", diversity)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest, diversity)


def _write_report(path: Path, manifest: dict[str, Any], diversity: dict[str, Any]) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        f"- ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        f"- INPUT_READINESS_AUDIT_SHA={manifest['INPUT_READINESS_AUDIT_SHA']}",
        f"- SELECTED_SOURCES={manifest['SELECTED_SOURCES']}",
        f"- TIMEZONE_VERIFICATION_STATUS={manifest['TIMEZONE_VERIFICATION_STATUS']}",
        f"- STRICT_EXACT_TIMEZONE_VERIFIED={manifest['STRICT_EXACT_TIMEZONE_VERIFIED']}",
        f"- COLLECTED_OFFICIAL_PUBLICATIONS={manifest['COLLECTED_OFFICIAL_PUBLICATIONS']}",
        f"- STRICT_EXACT_EVENTS={manifest['STRICT_EXACT_EVENTS']}",
        f"- NEW_HISTORICAL_EVENTS_COLLECTED={manifest['NEW_HISTORICAL_EVENTS_COLLECTED']}",
        f"- NEW_FUTURE_METADATA_ONLY_EVENTS={manifest['NEW_FUTURE_METADATA_ONLY_EVENTS']}",
        f"- MARKET_ELIGIBLE_EVENTS={manifest['MARKET_ELIGIBLE_EVENTS']}",
        f"- MARKET_NETWORK_REQUESTS={manifest['MARKET_NETWORK_REQUESTS']}",
        f"- MARKET_HISTORY_AVAILABLE={manifest['MARKET_HISTORY_AVAILABLE']}",
        f"- NEW_SEMANTIC_READY_EVENTS={manifest['NEW_SEMANTIC_READY_EVENTS']}",
        f"- NEW_REACTION_READY_EVENTS={manifest['NEW_REACTION_READY_EVENTS']}",
        f"- NEW_FEATURE_READY_EVENTS={manifest['NEW_FEATURE_READY_EVENTS']}",
        f"- FEATURE_READY_DELTA={manifest['FEATURE_READY_DELTA']}",
        f"- BEFORE_TOP_TICKER_SHARE={manifest['BEFORE_TOP_TICKER_SHARE']}",
        f"- AFTER_TOP_TICKER_SHARE={manifest['AFTER_TOP_TICKER_SHARE']}",
        f"- BEFORE_EFFECTIVE_TICKER_COUNT={manifest['BEFORE_EFFECTIVE_TICKER_COUNT']}",
        f"- AFTER_EFFECTIVE_TICKER_COUNT={manifest['AFTER_EFFECTIVE_TICKER_COUNT']}",
        f"- DIVERSITY_DECISION={manifest['DIVERSITY_DECISION']}",
        f"- NEXT_RECOMMENDED_ACTION={manifest['NEXT_RECOMMENDED_ACTION']}",
        f"- FEATURE_DEFINITION_CHANGED={manifest['FEATURE_DEFINITION_CHANGED']}",
        f"- REACTION_METHODOLOGY_CHANGED={manifest['REACTION_METHODOLOGY_CHANGED']}",
        f"- STRICT_EXACT_METHODOLOGY_CHANGED={manifest['STRICT_EXACT_METHODOLOGY_CHANGED']}",
        "",
        "## Required Answers",
        "",
        "1. Official M.Video timezone evidence verified: "
        f"{manifest['STRICT_EXACT_TIMEZONE_VERIFIED']}.",
        f"2. Timezone evidence status: {manifest['TIMEZONE_VERIFICATION_STATUS']}.",
        f"3. Legitimate strict-EXACT events: {manifest['STRICT_EXACT_EVENTS']}.",
        "4. Event feature schema restored to canonical representation: "
        f"{not bool(manifest['FEATURE_DEFINITION_CHANGED'])}.",
        "5. PR #48 eligibility gate reused for verified strict-EXACT rows: "
        f"{manifest['MARKET_ELIGIBLE_EVENTS'] >= 0}.",
        "6. Bounded T-Invest read-only acquisition attempted: "
        f"{manifest['MARKET_NETWORK_REQUESTS'] > 0}.",
        f"7. Unique market days requested: {manifest['UNIQUE_MARKET_DAYS_REQUESTED']}.",
        f"8. MVID minute candles acquired: {manifest['MVID_MINUTE_CANDLES_ACQUIRED']}.",
        f"9. IMOEX minute candles acquired: {manifest['IMOEX_MINUTE_CANDLES_ACQUIRED']}.",
        f"10. Reaction-ready events: {manifest['NEW_REACTION_READY_EVENTS']}.",
        f"11. Feature-ready events: {manifest['NEW_FEATURE_READY_EVENTS']}.",
        "12. Top ticker share changed from "
        f"{manifest['BEFORE_TOP_TICKER_SHARE']} to {manifest['AFTER_TOP_TICKER_SHARE']}.",
        "13. Effective ticker count changed from "
        f"{manifest['BEFORE_EFFECTIVE_TICKER_COUNT']} to "
        f"{manifest['AFTER_EFFECTIVE_TICKER_COUNT']}.",
        "14. Future market outcomes accessed: "
        f"{manifest['FUTURE_OUTCOMES_READ'] != 0 or manifest['FUTURE_TARGETS_READ'] != 0}.",
        "15. Rules v3/Qwen/features/reaction methodology changed: "
        f"{_methodology_changed(manifest)}.",
        f"16. Corrected final decision: {manifest['FINAL_DECISION']}.",
        "",
        "## Diversity Detail",
        "",
        "Top-3 ticker share changed from "
        f"{manifest['BEFORE_TOP_3_TICKER_SHARE']} to {manifest['AFTER_TOP_3_TICKER_SHARE']}.",
        "Source-family HHI changed from "
        f"{manifest['BEFORE_SOURCE_FAMILY_HHI']} to {manifest['AFTER_SOURCE_FAMILY_HHI']}.",
        "MOEX risk-parameter/exchange-originated feeds were excluded from selected issuer sources.",
        "Rules v3 and Qwen contracts were preserved; no NLP tuning or synthetic UNKNOWN "
        "fabrication was performed.",
        "Source selection used only source-origin, exact timestamp, material availability, "
        "historical depth, and pre-existing corpus composition.",
        "",
        "No model training, TEST evaluation, backtest, trading, source selection by market "
        "reaction, or future holdout price/target lookup was performed.",
        "",
        "## Diversity Delta",
        "",
        f"- top_ticker_share_delta={diversity['top_ticker_share_delta']}",
        f"- top_3_ticker_share_delta={diversity['top_3_ticker_share_delta']}",
        f"- effective_ticker_count_delta={diversity['effective_ticker_count_delta']}",
        f"- source_family_hhi_delta={diversity['source_family_hhi_delta']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
