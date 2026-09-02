from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid5

from src.events.domain.entities import ExtractedFinancialFact
from src.events.domain.v3 import EventAnalyzerV3, rules_v3_fingerprint
from src.exact_event_live_official_collection.http_client import (
    BoundedHttpClient,
    FetchResult,
    HttpClient,
)
from src.free_live_issuer_accumulation.domain import (
    ARTIFACT_VERSION,
    DEFAULT_SOURCE_REGISTRY_PATH,
    EXPECTED_RULES_V3_FINGERPRINT,
    FUTURE_HOLDOUT_START,
    MAX_ITEMS_PER_SOURCE,
    MAX_SOURCES_PER_SMOKE,
    PARSER_VERSION,
    RAW_SNAPSHOT_VERSION,
    SHADOW_CORPUS_VERSION,
    SOURCE_REGISTRY_VERSION,
    LiveEventStatus,
    LiveIssuerSource,
    ParsedPublication,
    SourceQualificationStatus,
    assert_market_query_upper_bound,
    live_accumulation_safety_flags,
    parse_publication_timestamp,
    sha256_payload,
    sha256_text,
)
from src.instruments.infrastructure.seed import SEED_INSTRUMENTS

EVENT_NAMESPACE = UUID("ab8316e9-52d1-4fec-8f42-6a4c982c6be4")
DEFAULT_HISTORICAL_TICKER_SUMMARY_PATH = Path(
    "artifacts/ml-v2-readiness-audit-v1/ticker-summary.jsonl"
)
SUPPORTED_DISCOVERY_CONTENT_TYPES = frozenset(
    {
        "application/rss+xml",
        "application/xml",
        "text/xml",
    }
)


@dataclass(frozen=True, slots=True)
class Registry:
    historical_frozen_issuer_tickers: tuple[str, ...]
    sources: tuple[LiveIssuerSource, ...]
    milestone: dict[str, Any]


def read_registry(
    path: Path = Path(DEFAULT_SOURCE_REGISTRY_PATH),
    *,
    historical_ticker_summary_path: Path = DEFAULT_HISTORICAL_TICKER_SUMMARY_PATH,
) -> Registry:
    payload = _read_json(path)
    if payload.get("source_registry_version") != SOURCE_REGISTRY_VERSION:
        raise ValueError("LIVE_ISSUER_SOURCE_REGISTRY_VERSION_MISMATCH")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("LIVE_ISSUER_SOURCE_REGISTRY_SOURCES_MISSING")
    historical = load_historical_issuer_tickers(
        historical_ticker_summary_path,
        fallback=payload.get("historical_frozen_issuer_tickers"),
    )
    milestone = payload.get("milestone")
    if not isinstance(milestone, dict):
        raise ValueError("LIVE_DIVERSITY_MILESTONE_MISSING")
    source_rows = cast("list[object]", raw_sources)
    return Registry(
        historical_frozen_issuer_tickers=historical,
        sources=tuple(
            LiveIssuerSource.from_payload(cast("dict[str, Any]", row)) for row in source_rows
        ),
        milestone=cast("dict[str, Any]", milestone),
    )


def load_historical_issuer_tickers(
    path: Path = DEFAULT_HISTORICAL_TICKER_SUMMARY_PATH,
    *,
    fallback: object | None = None,
) -> tuple[str, ...]:
    if path.exists():
        tickers: set[str] = set()
        for row in _read_jsonl(path):
            origins = row.get("event_origins")
            if not isinstance(origins, dict):
                continue
            origin_counts = cast("dict[str, object]", origins)
            if _int_value(origin_counts.get("ISSUER_ORIGINATED")) > 0:
                tickers.add(str(row["ticker"]))
        if tickers:
            return tuple(sorted(tickers))
    if not isinstance(fallback, list):
        raise ValueError("HISTORICAL_FROZEN_ISSUER_TICKERS_MISSING")
    return tuple(sorted(str(item) for item in cast("list[object]", fallback)))


def audit_live_issuer_sources(
    *,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    registry_path: Path = Path(DEFAULT_SOURCE_REGISTRY_PATH),
    historical_ticker_summary_path: Path = DEFAULT_HISTORICAL_TICKER_SUMMARY_PATH,
    client: HttpClient | None = None,
    network_check: bool = False,
    max_sources: int = MAX_SOURCES_PER_SMOKE,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable free live issuer audit artifact output already exists")
    now = created_at or datetime.now(UTC)
    registry = read_registry(
        registry_path,
        historical_ticker_summary_path=historical_ticker_summary_path,
    )
    rows = [
        _source_audit_row(source, registry.historical_frozen_issuer_tickers)
        for source in registry.sources
    ]
    ready_sources = [
        source
        for source in registry.sources
        if source.source_status == SourceQualificationStatus.LIVE_STRICT_EXACT_READY
    ]
    ready_tickers = sorted({source.ticker for source in ready_sources})
    new_ready_tickers = sorted(
        ticker
        for ticker in ready_tickers
        if ticker not in registry.historical_frozen_issuer_tickers
    )
    candidate_universe = _candidate_universe_tickers(registry.historical_frozen_issuer_tickers)
    paid_out_of_scope = [
        source
        for source in registry.sources
        if source.source_status == SourceQualificationStatus.OUT_OF_SCOPE_PAID_SOURCE
    ]
    network_rows: list[dict[str, Any]] = []
    if network_check:
        http = client or BoundedHttpClient()
        audited = [
            source
            for source in registry.sources
            if source.source_status != SourceQualificationStatus.OUT_OF_SCOPE_PAID_SOURCE
        ][:max_sources]
        network_rows = [_audit_source_current_response(source, http, now) for source in audited]
    safety = live_accumulation_safety_flags()
    stats = shadow_corpus_stats(
        [],
        [source.source_id for source in ready_sources],
        historical_tickers=registry.historical_frozen_issuer_tickers,
        ready_tickers=ready_tickers,
    )
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": now.isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "SOURCE_REGISTRY_VERSION": SOURCE_REGISTRY_VERSION,
        "SOURCE_REGISTRY_SHA": sha256_payload(_registry_contract_payload(registry)),
        "HISTORICAL_ISSUER_TICKERS": list(registry.historical_frozen_issuer_tickers),
        "CANDIDATE_UNIVERSE_TICKERS": candidate_universe,
        "NETWORK_AUDIT_PERFORMED": network_check,
        "BOUNDED_HTTP_REQUESTS": len(network_rows),
        "BOUNDED_RETRIES_ENABLED": True,
        "FREE_OFFICIAL_SOURCES_AUDITED": len(
            [
                source
                for source in registry.sources
                if source.source_status != SourceQualificationStatus.OUT_OF_SCOPE_PAID_SOURCE
            ]
        ),
        "PAID_OUT_OF_SCOPE_SOURCES": len(paid_out_of_scope),
        "PAID_SOURCES_USED": False,
        "LIVE_STRICT_EXACT_READY_SOURCES": len(ready_sources),
        "LIVE_STRICT_EXACT_READY_SOURCE_IDS": [source.source_id for source in ready_sources],
        "UNIQUE_ISSUER_TICKERS_COVERED": len(ready_tickers),
        "UNIQUE_ISSUER_TICKERS": ready_tickers,
        "READY_ISSUER_TICKERS": ready_tickers,
        "NEW_TICKERS_RELATIVE_TO_HISTORICAL_7": new_ready_tickers,
        "SOURCE_READY": len(new_ready_tickers)
        >= int(registry.milestone.get("minimum_new_issuer_tickers", 3)),
        "NEW_ITEM_OBSERVED": False,
        "SOURCES_WITH_EXPLICIT_TIMEZONE": sum(
            1 for source in ready_sources if _source_has_explicit_timezone(source)
        ),
        "SOURCES_WITH_EXPLICIT_OR_DOCUMENTED_TIMEZONE": sum(
            1 for source in ready_sources if _source_has_explicit_or_documented_timezone(source)
        ),
        "SOURCES_REJECTED_FOR_TIMEZONE": sum(
            1
            for source in registry.sources
            if source.source_status
            in {
                SourceQualificationStatus.LIVE_TIMESTAMP_UNVERIFIED,
                SourceQualificationStatus.LIVE_DATE_ONLY,
                SourceQualificationStatus.LIVE_CLOCK_WITHOUT_TIMEZONE,
            }
        ),
        "SOURCES_WITH_STABLE_IDENTITY": sum(
            1
            for source in registry.sources
            if source.source_status != SourceQualificationStatus.OUT_OF_SCOPE_PAID_SOURCE
            and source.stable_identity
        ),
        "LIVE_DIVERSITY_MILESTONE": registry.milestone.get("name"),
        "LIVE_DIVERSITY_STATUS": _live_diversity_status(
            historical=registry.historical_frozen_issuer_tickers,
            live_tickers=ready_tickers,
            new_tickers=new_ready_tickers,
            milestone=registry.milestone,
        ),
        "LIVE_DIVERSITY_ACCUMULATION_STATUS": stats["live_diversity_accumulation_status"],
        "STRICT_ANSWER": "YES" if len(new_ready_tickers) >= 3 else "NO",
        "FREE_BLOCKER": (
            "insufficient issuer-originated free strict-EXACT sources for at least "
            "3 new MOEX tickers"
        )
        if len(new_ready_tickers) < 3
        else None,
        "safety": safety,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = sha256_payload(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"ARTIFACT_SHA", "created_at", "git_sha"}
        }
    )
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "source-registry.json", _registry_contract_payload(registry))
    _write_jsonl(output_root / "source-audit.jsonl", rows)
    _write_jsonl(output_root / "network-provenance.jsonl", network_rows)
    _write_json(output_root / "ticker-coverage.json", _ticker_coverage(registry, ready_tickers))
    _write_json(output_root / "shadow-corpus-stats.json", stats)
    _write_json(output_root / "timestamp-contracts.json", _timestamp_contracts(registry))
    _write_jsonl(output_root / "rejections.jsonl", _source_rejections(registry))
    _write_json(output_root / "safety.json", safety)
    _write_report(output_root / "report.md", manifest)
    return manifest


def collect_live_issuer_news(
    *,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    registry_path: Path = Path(DEFAULT_SOURCE_REGISTRY_PATH),
    historical_ticker_summary_path: Path = DEFAULT_HISTORICAL_TICKER_SUMMARY_PATH,
    state_path: Path | None = None,
    client: HttpClient | None = None,
    created_at: datetime | None = None,
    max_sources: int = MAX_SOURCES_PER_SMOKE,
    source_id: str | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable live issuer shadow corpus output already exists")
    now = created_at or datetime.now(UTC)
    if rules_v3_fingerprint() != EXPECTED_RULES_V3_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_CHANGED")
    registry = read_registry(
        registry_path,
        historical_ticker_summary_path=historical_ticker_summary_path,
    )
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=False)
    raw_root = output_root / "raw-snapshots"
    raw_root.mkdir()

    enabled_sources = [
        source
        for source in registry.sources
        if source.enabled
        and source.source_status == SourceQualificationStatus.LIVE_STRICT_EXACT_READY
        and (source_id is None or source.source_id == source_id)
    ]
    if source_id is not None and not enabled_sources:
        raise ValueError("LIVE_ISSUER_SOURCE_NOT_ENABLED_OR_NOT_READY")
    enabled_sources = enabled_sources[:max_sources]
    ready_tickers = sorted({source.ticker for source in enabled_sources})
    new_ready_tickers = sorted(
        ticker
        for ticker in ready_tickers
        if ticker not in registry.historical_frozen_issuer_tickers
    )
    candidate_universe = _candidate_universe_tickers(registry.historical_frozen_issuer_tickers)
    http = client or BoundedHttpClient()
    state = _read_state(state_path)
    seen_item_ids = set(cast("list[str]", state.get("seen_source_item_ids", [])))
    seen_urls = set(cast("list[str]", state.get("seen_canonical_urls", [])))
    seen_content_by_identity = cast("dict[str, str]", state.get("content_sha_by_identity", {}))

    network_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    revision_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    metrics: Counter[str] = Counter()

    for source in enabled_sources:
        source_started_at = datetime.now(UTC)
        result = _fetch_with_bounded_retry(source, http)
        latency_ms = int((datetime.now(UTC) - source_started_at).total_seconds() * 1000)
        network_rows.append(_network_row(source, result, now, latency_ms))
        if result.blocker is not None or result.status is None or result.status >= 400:
            metrics["source_failures"] += 1
            source_rows.append(
                _source_poll_row(source, "ENVIRONMENT_UNAVAILABLE", 0, 0, 0, latency_ms)
            )
            continue
        unsupported_content_type = _unsupported_content_type(result.content_type)
        if unsupported_content_type is not None:
            metrics["source_failures"] += 1
            invalid_rows.append(
                {
                    "source_id": source.source_id,
                    "content_type": result.content_type,
                    "blocker": unsupported_content_type,
                }
            )
            source_rows.append(
                _source_poll_row(source, unsupported_content_type, 0, 0, 0, latency_ms)
            )
            continue
        try:
            publications = parse_rss_publications(source, result.body)
        except ValueError as exc:
            metrics["source_failures"] += 1
            invalid_rows.append({"source_id": source.source_id, "blocker": str(exc)})
            source_rows.append(_source_poll_row(source, str(exc), 0, 0, 0, latency_ms))
            continue
        raw_response_path = raw_root / f"{source.source_id}.xml"
        raw_response_path.write_bytes(result.body)
        source_new = source_duplicates = source_revisions = 0
        sampled_publications = _latest_publications(publications)
        for publication in sampled_publications:
            metrics["items_discovered"] += 1
            material = publication.material()
            if material is None:
                metrics["rejected_items"] += 1
                invalid_rows.append(
                    {
                        "source_id": source.source_id,
                        "source_item_id": publication.source_item_id,
                        "blocker": "PUBLICATION_MATERIAL_MISSING",
                    }
                )
                continue
            if publication.publication_timestamp_utc < FUTURE_HOLDOUT_START:
                metrics["rejected_items"] += 1
                invalid_rows.append(
                    {
                        "source_id": source.source_id,
                        "source_item_id": publication.source_item_id,
                        "published_at": publication.publication_timestamp_utc.isoformat(),
                        "blocker": "BEFORE_LIVE_SHADOW_CUTOFF",
                    }
                )
                continue
            identity = _identity(source, publication.source_item_id)
            content_sha = sha256_text(publication.raw_item)
            duplicate_by_id = identity in seen_item_ids
            duplicate_by_url = publication.canonical_url in seen_urls
            if duplicate_by_id or duplicate_by_url:
                metrics["duplicates"] += 1
                source_duplicates += 1
                if duplicate_by_id and seen_content_by_identity.get(identity) not in {
                    None,
                    content_sha,
                }:
                    revision = _revision_row(source, publication, content_sha, now)
                    revision_rows.append(revision)
                    metrics["revisions"] += 1
                    source_revisions += 1
                    seen_content_by_identity[identity] = content_sha
                continue
            event_id = str(uuid5(EVENT_NAMESPACE, identity))
            snapshot = _snapshot_row(source, publication, event_id, result, now)
            snapshot_path = raw_root / f"{snapshot['raw_snapshot_sha']}.json"
            _write_json(snapshot_path, snapshot)
            semantic = _semantic_payload(event_id, material)
            shadow = _shadow_row(source, publication, event_id, snapshot, semantic, now)
            seen_item_ids.add(identity)
            seen_urls.add(publication.canonical_url)
            seen_content_by_identity[identity] = content_sha
            snapshot_rows.append(snapshot)
            shadow_rows.append(shadow)
            metrics["new_items"] += 1
            metrics["raw_snapshots_frozen"] += 1
            metrics["semantic_ready"] += 1
            if semantic["semantic_unknown"]:
                metrics["semantic_unknown"] += 1
            source_new += 1
        source_rows.append(
            _source_poll_row(
                source,
                "SUCCESS",
                len(sampled_publications),
                source_new,
                source_duplicates,
                latency_ms,
                revisions=source_revisions,
            )
        )

    dedupe_state = {
        "dedupe_state_version": "live-issuer-shadow-dedupe-state-v1",
        "seen_source_item_ids": sorted(seen_item_ids),
        "seen_canonical_urls": sorted(seen_urls),
        "content_sha_by_identity": dict(sorted(seen_content_by_identity.items())),
    }
    stats = shadow_corpus_stats(
        shadow_rows,
        [source.source_id for source in enabled_sources],
        historical_tickers=registry.historical_frozen_issuer_tickers,
        ready_tickers=ready_tickers,
    )
    safety = live_accumulation_safety_flags()
    strict_answer = "YES" if len(new_ready_tickers) >= 3 else "NO"
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "SHADOW_CORPUS_VERSION": SHADOW_CORPUS_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": now.isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "SOURCE_REGISTRY_VERSION": SOURCE_REGISTRY_VERSION,
        "SOURCE_REGISTRY_SHA": sha256_payload(_registry_contract_payload(registry)),
        "HISTORICAL_ISSUER_TICKERS": list(registry.historical_frozen_issuer_tickers),
        "CANDIDATE_UNIVERSE_TICKERS": candidate_universe,
        "PARSER_VERSION": PARSER_VERSION,
        "RULES_V3_FINGERPRINT": rules_v3_fingerprint(),
        "RULES_V3_CHANGED": False,
        "ENABLED_SOURCES": len(enabled_sources),
        "LIVE_STRICT_EXACT_READY_SOURCE_IDS": [source.source_id for source in enabled_sources],
        "UNIQUE_READY_ISSUER_TICKERS": len(ready_tickers),
        "READY_ISSUER_TICKERS": ready_tickers,
        "NEW_TICKERS_RELATIVE_TO_HISTORICAL_7": new_ready_tickers,
        "SOURCE_READY": len(new_ready_tickers)
        >= int(registry.milestone.get("minimum_new_issuer_tickers", 3)),
        "NEW_ITEM_OBSERVED": bool(shadow_rows),
        "BOUNDED_HTTP_REQUESTS": len(network_rows),
        "BOUNDED_RETRIES_ENABLED": True,
        "FREE_OFFICIAL_SOURCES_AUDITED": len(
            [
                source
                for source in registry.sources
                if source.source_status != SourceQualificationStatus.OUT_OF_SCOPE_PAID_SOURCE
            ]
        ),
        "LIVE_STRICT_EXACT_READY_SOURCES": len(enabled_sources),
        "SOURCES_WITH_EXPLICIT_OR_DOCUMENTED_TIMEZONE": sum(
            1 for source in enabled_sources if _source_has_explicit_or_documented_timezone(source)
        ),
        "SOURCES_REJECTED_FOR_TIMEZONE": sum(
            1
            for source in registry.sources
            if source.source_status
            in {
                SourceQualificationStatus.LIVE_TIMESTAMP_UNVERIFIED,
                SourceQualificationStatus.LIVE_DATE_ONLY,
                SourceQualificationStatus.LIVE_CLOCK_WITHOUT_TIMEZONE,
            }
        ),
        "SOURCES_WITH_STABLE_IDENTITY": sum(
            1
            for source in registry.sources
            if source.source_status != SourceQualificationStatus.OUT_OF_SCOPE_PAID_SOURCE
            and source.stable_identity
        ),
        "EVENTS_COLLECTED": len(shadow_rows),
        "RAW_SNAPSHOTS_FROZEN": metrics["raw_snapshots_frozen"],
        "DUPLICATES_ENCOUNTERED": metrics["duplicates"],
        "REVISIONS_CREATED": metrics["revisions"],
        "SEMANTIC_READY_EVENTS": metrics["semantic_ready"],
        "UNKNOWN_EVENTS": metrics["semantic_unknown"],
        "UNKNOWN_RATE": _rate(metrics["semantic_unknown"], metrics["semantic_ready"]),
        "PRE_EVENT_FEATURE_READY_EVENTS": 0,
        "TARGET_STATUS": "SEALED",
        "OLD_FUTURE_HOLDOUT_OPENED": False,
        "OLD_FUTURE_HOLDOUT_STATUS": "SEALED",
        "LIVE_DIVERSITY_STATUS": stats["live_diversity_status"],
        "LIVE_DIVERSITY_ACCUMULATION_STATUS": stats["live_diversity_accumulation_status"],
        "STRICT_ANSWER": strict_answer,
        "FREE_BLOCKER": (
            "insufficient issuer-originated free strict-EXACT sources for at least "
            "3 new MOEX tickers"
        )
        if strict_answer == "NO"
        else None,
        "metrics": {
            "source_polled": len(enabled_sources),
            "items_discovered": metrics["items_discovered"],
            "new_items": metrics["new_items"],
            "duplicates": metrics["duplicates"],
            "rejected_items": metrics["rejected_items"],
            "timestamp_failures": metrics["timestamp_failures"],
            "parser_failures": metrics["parser_failures"],
            "source_failures": metrics["source_failures"],
            "revisions": metrics["revisions"],
        },
        "dashboard": stats,
        "safety": safety,
        **safety,
    }
    manifest["NETWORK_PROVENANCE_SHA"] = sha256_payload(network_rows)
    manifest["RAW_PUBLICATION_SNAPSHOT_SHA"] = sha256_payload(snapshot_rows)
    manifest["SHADOW_CORPUS_SHA"] = sha256_payload(shadow_rows)
    manifest["REVISION_LOG_SHA"] = sha256_payload(revision_rows)
    manifest["DEDUPE_STATE_SHA"] = sha256_payload(dedupe_state)
    manifest["ARTIFACT_SHA"] = sha256_payload(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"ARTIFACT_SHA", "created_at", "git_sha"}
        }
    )
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "source-registry.json", _registry_contract_payload(registry))
    _write_jsonl(output_root / "network-provenance.jsonl", network_rows)
    _write_jsonl(output_root / "source-polls.jsonl", source_rows)
    _write_jsonl(output_root / "raw-publication-snapshots.jsonl", snapshot_rows)
    _write_jsonl(output_root / "live-shadow-corpus.jsonl", shadow_rows)
    _write_jsonl(output_root / "revision-log.jsonl", revision_rows)
    _write_jsonl(output_root / "invalid-items.jsonl", invalid_rows)
    _write_jsonl(output_root / "rejections.jsonl", invalid_rows + _source_rejections(registry))
    _write_json(output_root / "dedupe-state.json", dedupe_state)
    _write_json(output_root / "shadow-corpus-stats.json", stats)
    _write_json(output_root / "ticker-coverage.json", _ticker_coverage(registry, ready_tickers))
    _write_json(output_root / "timestamp-contracts.json", _timestamp_contracts(registry))
    _write_json(output_root / "safety.json", safety)
    _write_report(output_root / "report.md", manifest)
    return manifest


def parse_rss_publications(source: LiveIssuerSource, body: bytes) -> list[ParsedPublication]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ValueError("INVALID_RSS") from exc
    if _local(root.tag) != "rss":
        raise ValueError("INVALID_RSS")
    publications: list[ParsedPublication] = []
    for item in (element for element in root.iter() if _local(element.tag) == "item"):
        title = _text(item, "title")
        description = _text(item, "description")
        content = _text(item, "encoded")
        link = _text(item, "link")
        guid = _text(item, "guid")
        pubdate = _text(item, "pubDate")
        if not pubdate:
            raise ValueError("MISSING_EXACT_TIMESTAMP")
        try:
            published_at = parse_publication_timestamp(pubdate, source.timestamp_contract)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        source_item_id = guid or link
        if not source_item_id:
            raise ValueError("LIVE_NO_STABLE_ID")
        canonical_url = link or f"{source.discovery_url}#{source_item_id}"
        publications.append(
            ParsedPublication(
                source_item_id=source_item_id,
                canonical_url=canonical_url,
                title=title,
                description=description,
                content=content,
                publication_timestamp_raw=pubdate,
                publication_timestamp_utc=published_at,
                raw_payload={
                    "title": title or None,
                    "description": description or None,
                    "content:encoded": content or None,
                    "pubDate": pubdate,
                    "link": link or None,
                    "guid": guid or None,
                },
                raw_item=ET.tostring(item, encoding="unicode"),
            )
        )
    return sorted(
        publications, key=lambda item: (item.publication_timestamp_utc, item.source_item_id)
    )


def _latest_publications(publications: list[ParsedPublication]) -> list[ParsedPublication]:
    return publications[-MAX_ITEMS_PER_SOURCE:]


def shadow_corpus_stats(
    rows: list[dict[str, Any]],
    accepted_source_ids: list[str],
    *,
    historical_tickers: tuple[str, ...] = (),
    ready_tickers: list[str] | None = None,
) -> dict[str, Any]:
    by_ticker = Counter(str(row["ticker"]) for row in rows)
    by_source_family = Counter(str(row["source_id"]) for row in rows)
    by_day = Counter(str(row["published_at"])[:10] for row in rows)
    unknown = sum(
        1 for row in rows if cast("dict[str, Any]", row["semantic_output"])["semantic_unknown"]
    )
    total = len(rows)
    ready = sorted(set(ready_tickers or by_ticker.keys()))
    historical = set(historical_tickers)
    new_ready = sorted(ticker for ticker in ready if ticker not in historical)
    observed_tickers = sorted(by_ticker)
    new_observed = sorted(ticker for ticker in observed_tickers if ticker not in historical)
    accumulation_status = _live_diversity_status(
        historical=tuple(sorted(historical)),
        live_tickers=ready,
        new_tickers=new_ready,
        milestone={"minimum_new_issuer_tickers": 3},
        observed_events=total,
    )
    return {
        "accepted_live_sources": len(accepted_source_ids),
        "accepted_source_ids": accepted_source_ids,
        "source_ready_tickers": ready,
        "new_source_ready_tickers": new_ready,
        "new_item_observed_tickers": new_observed,
        "source_ready": len(new_ready) >= 3,
        "new_item_observed": total > 0,
        "unique_live_issuer_tickers": len(by_ticker),
        "events_collected": total,
        "strict_timestamp_pass": total,
        "ticker_mapping_pass": total,
        "raw_snapshot_pass": total,
        "semantic_ready_count": total,
        "semantic_unknown_count": unknown,
        "semantic_unknown_rate": _rate(unknown, total),
        "pre_event_feature_ready_count": 0,
        "top_1_ticker_share": _top_share(by_ticker, 1),
        "top_3_ticker_share": _top_share(by_ticker, 3),
        "ticker_hhi": _hhi(by_ticker),
        "source_family_hhi": _hhi(by_source_family),
        "events_per_ticker": dict(sorted(by_ticker.items())),
        "events_per_day": dict(sorted(by_day.items())),
        "events_per_week": _events_per_week(rows),
        "target_metrics_included": False,
        "live_diversity_status": accumulation_status,
        "live_diversity_accumulation_status": accumulation_status,
    }


def load_shadow_corpus_stats(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path / "live-shadow-corpus.jsonl")
    source_rows = _read_jsonl(path / "source-polls.jsonl")
    accepted = [str(row["source_id"]) for row in source_rows if row.get("status") == "SUCCESS"]
    manifest = _read_json(path / "manifest.json")
    return shadow_corpus_stats(
        rows,
        accepted,
        historical_tickers=tuple(cast("list[str]", manifest.get("HISTORICAL_ISSUER_TICKERS", []))),
        ready_tickers=cast("list[str]", manifest.get("READY_ISSUER_TICKERS", [])),
    )


def verify_sealed_live_epoch(path: Path) -> dict[str, Any]:
    manifest = _read_json(path / "manifest.json")
    rows = _read_jsonl(path / "live-shadow-corpus.jsonl")
    violations = [
        row
        for row in rows
        if row.get("TARGET_STATUS") != "SEALED"
        or row.get("epoch") != "LIVE_SHADOW_CORPUS"
        or "target" in row
        or "targets" in row
        or "abnormal_return" in row
        or "model_prediction" in row
    ]
    counters_ok = (
        manifest.get("LIVE_POST_EVENT_PRICE_READS") == 0
        and manifest.get("LIVE_TARGETS_COMPUTED") == 0
        and manifest.get("LIVE_OUTCOMES_READ") == 0
        and manifest.get("LIVE_MODEL_PREDICTIONS") == 0
        and manifest.get("OLD_FUTURE_HOLDOUT_OPENED") is False
    )
    return {
        "sealed_epoch_verified": not violations and counters_ok,
        "violations": len(violations),
        "LIVE_POST_EVENT_PRICE_READS": manifest.get("LIVE_POST_EVENT_PRICE_READS"),
        "LIVE_TARGETS_COMPUTED": manifest.get("LIVE_TARGETS_COMPUTED"),
        "LIVE_OUTCOMES_READ": manifest.get("LIVE_OUTCOMES_READ"),
        "LIVE_MODEL_PREDICTIONS": manifest.get("LIVE_MODEL_PREDICTIONS"),
        "OLD_FUTURE_HOLDOUT_OPENED": manifest.get("OLD_FUTURE_HOLDOUT_OPENED"),
    }


def _source_audit_row(
    source: LiveIssuerSource,
    historical_tickers: tuple[str, ...],
) -> dict[str, Any]:
    return {
        **source.payload(),
        "source_contract_sha": source.contract_sha(),
        "new_vs_historical_frozen_7": source.ticker not in historical_tickers,
        "collector_eligible": source.enabled
        and source.source_status == SourceQualificationStatus.LIVE_STRICT_EXACT_READY,
        "TIMESTAMP_EVIDENCE_TYPE": source.timestamp_contract.get("evidence_type"),
        "TIMESTAMP_EVIDENCE_VALUE": source.timestamp_contract.get("evidence_value"),
        "TIMESTAMP_CONTRACT_SHA": source.contract_sha(),
    }


def _snapshot_row(
    source: LiveIssuerSource,
    publication: ParsedPublication,
    event_id: str,
    result: FetchResult,
    observed_at: datetime,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "snapshot_version": RAW_SNAPSHOT_VERSION,
        "event_id": event_id,
        "source_id": source.source_id,
        "source_item_id": publication.source_item_id,
        "canonical_url": publication.canonical_url,
        "issuer": source.issuer,
        "ticker": source.ticker,
        "published_at": publication.publication_timestamp_utc.isoformat(),
        "publication_timestamp_raw": publication.publication_timestamp_raw,
        "timezone_evidence_type": source.timestamp_contract.get("evidence_type"),
        "timezone_evidence_value": source.timestamp_contract.get("evidence_value"),
        "source_contract_sha": source.contract_sha(),
        "parser_version": source.parser,
        "first_observed_at": observed_at.isoformat(),
        "fetched_at": observed_at.isoformat(),
        "title": publication.title or None,
        "description": publication.description or None,
        "content": publication.content or None,
        "raw_payload": publication.raw_payload,
        "raw_item_sha": sha256_text(publication.raw_item),
        "response_sha": sha256_text(result.body.decode("utf-8", errors="replace")),
        "content_type": result.content_type,
    }
    payload["raw_snapshot_sha"] = sha256_payload(payload)
    return payload


def _shadow_row(
    source: LiveIssuerSource,
    publication: ParsedPublication,
    event_id: str,
    snapshot: dict[str, Any],
    semantic: dict[str, Any],
    observed_at: datetime,
) -> dict[str, Any]:
    assert_market_query_upper_bound(
        end_at=publication.publication_timestamp_utc,
        published_at=publication.publication_timestamp_utc,
    )
    status_chain = [
        LiveEventStatus.DISCOVERED.value,
        LiveEventStatus.TIMESTAMP_VERIFIED.value,
        LiveEventStatus.TICKER_RESOLVED.value,
        LiveEventStatus.RAW_SNAPSHOT_FROZEN.value,
        LiveEventStatus.SEMANTIC_READY.value,
        LiveEventStatus.SHADOW_READY.value,
    ]
    return {
        "shadow_corpus_version": SHADOW_CORPUS_VERSION,
        "epoch": "LIVE_SHADOW_CORPUS",
        "event_id": event_id,
        "source_id": source.source_id,
        "source_item_id": publication.source_item_id,
        "canonical_url": publication.canonical_url,
        "issuer_id": source.source_id,
        "issuer": source.issuer,
        "ticker": source.ticker,
        "published_at": publication.publication_timestamp_utc.isoformat(),
        "first_observed_at": observed_at.isoformat(),
        "timezone_contract": source.timestamp_contract,
        "timestamp_contract_sha": source.contract_sha(),
        "raw_snapshot_sha": snapshot["raw_snapshot_sha"],
        "semantic_output": semantic,
        "pre_event_feature_availability": {
            "available": False,
            "reason": "no market-data query performed in bounded smoke mode",
            "upper_bound": publication.publication_timestamp_utc.isoformat(),
        },
        "source_contract_sha": source.contract_sha(),
        "rules_fingerprint": rules_v3_fingerprint(),
        "event_status": LiveEventStatus.SHADOW_READY.value,
        "status_chain": status_chain,
        "TARGET_STATUS": "SEALED",
    }


def _semantic_payload(event_id: str, material: str) -> dict[str, Any]:
    analysis = EventAnalyzerV3().analyze(news_id=UUID(event_id), raw_content=material)
    fact_count = len(analysis.financial_facts)
    event_types = sorted({event.event_type.value for event in analysis.events})
    semantic_unknown = analysis.primary_event_type.value == "UNKNOWN" or not event_types
    return {
        "analysis_version": analysis.analysis_version,
        "primary_event_type": analysis.primary_event_type.value,
        "status": analysis.status.value,
        "direction": "UNKNOWN",
        "importance": "UNKNOWN",
        "event_types": event_types,
        "fact_count": fact_count,
        "facts": [_fact_payload(fact) for fact in analysis.financial_facts],
        "facts_extracted": fact_count > 0,
        "zero_fact_publication": fact_count == 0,
        "semantic_unknown": semantic_unknown,
        "unknown_breakdown": {
            "unsupported_semantic_class": semantic_unknown,
            "informational_or_non_corporate": semantic_unknown and fact_count == 0,
            "zero_fact_publication": fact_count == 0,
        },
        "rules_fingerprint": rules_v3_fingerprint(),
        "uses_market_data": False,
        "uses_reaction_data": False,
        "uses_target_data": False,
    }


def _fact_payload(fact: ExtractedFinancialFact) -> dict[str, Any]:
    return {
        "metric": fact.metric.value,
        "raw_value": str(fact.raw_value),
        "normalized_value": str(fact.normalized_value),
        "unit": fact.unit.value,
        "currency": fact.currency.value,
        "scale": fact.scale.value,
        "period_type": fact.period_type.value,
        "year": fact.year,
        "quarter": fact.quarter,
        "month": fact.month,
        "date_from": None if fact.date_from is None else fact.date_from.isoformat(),
        "date_to": None if fact.date_to is None else fact.date_to.isoformat(),
        "raw_period": fact.raw_period,
        "comparison_type": fact.comparison_type.value,
        "fact_role": fact.fact_role.value,
        "change_direction": fact.change_direction.value,
        "change_value": None if fact.change_value is None else str(fact.change_value),
        "change_unit": None if fact.change_unit is None else fact.change_unit.value,
        "confidence": str(fact.confidence),
        "rule_id": fact.rule_id,
        "evidence_text": fact.evidence_text,
        "extractor_version": fact.extractor_version,
        "matched_rule": fact.matched_rule,
    }


def _network_row(
    source: LiveIssuerSource,
    result: FetchResult,
    fetched_at: datetime,
    latency_ms: int,
) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "ticker": source.ticker,
        "issuer": source.issuer,
        "request_url": result.request_url,
        "final_url": result.final_url,
        "http_status": result.status,
        "content_type": result.content_type,
        "bytes_received": len(result.body),
        "redirects": result.redirects,
        "blocker": result.blocker,
        "fetched_at": fetched_at.isoformat(),
        "latency_ms": latency_ms,
        "bounded_retries": int(source.polling_policy.get("bounded_retries", 1)),
        "max_items_per_poll": int(source.polling_policy.get("max_items_per_poll", 0)),
    }


def _source_poll_row(
    source: LiveIssuerSource,
    status: str,
    items_discovered: int,
    new_items: int,
    duplicates: int,
    latency_ms: int,
    *,
    revisions: int = 0,
) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "ticker": source.ticker,
        "issuer": source.issuer,
        "status": status,
        "items_discovered": items_discovered,
        "new_items": new_items,
        "duplicates": duplicates,
        "revisions": revisions,
        "latest_successful_poll": None if status != "SUCCESS" else datetime.now(UTC).isoformat(),
        "source_latency_ms": latency_ms,
    }


def _revision_row(
    source: LiveIssuerSource,
    publication: ParsedPublication,
    content_sha: str,
    observed_at: datetime,
) -> dict[str, Any]:
    return {
        "revision_version": "live-issuer-publication-revision-v1",
        "source_id": source.source_id,
        "source_item_id": publication.source_item_id,
        "canonical_url": publication.canonical_url,
        "observed_at": observed_at.isoformat(),
        "new_raw_item_sha": content_sha,
        "reason": "stable identity observed with changed raw publication material",
    }


def _fetch_with_bounded_retry(source: LiveIssuerSource, client: HttpClient) -> FetchResult:
    retries = max(0, int(source.polling_policy.get("bounded_retries", 1)))
    max_attempts = retries + 1
    result: FetchResult | None = None
    for attempt in range(max_attempts):
        result = client.get(source.discovery_url)
        if not _retryable(result) or attempt == max_attempts - 1:
            return result
        time.sleep(min(0.25 * (2**attempt), 1.0))
    if result is None:
        raise RuntimeError("HTTP_CLIENT_RETURNED_NO_RESULT")
    return result


def _retryable(result: FetchResult) -> bool:
    if result.status in {429, 500, 502, 503, 504}:
        return True
    return result.blocker in {"TIMEOUT", "HTTP_FAILURE", "RATE_LIMITED", "TECHNICAL_FAILURE"}


def _audit_source_current_response(
    source: LiveIssuerSource,
    client: HttpClient,
    fetched_at: datetime,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    result = _fetch_with_bounded_retry(source, client)
    latency_ms = int((datetime.now(UTC) - started_at).total_seconds() * 1000)
    content_type_rejection = (
        None
        if result.blocker is not None or result.status is None or result.status >= 400
        else _unsupported_content_type(result.content_type)
    )
    parser_status = "NOT_ATTEMPTED"
    item_count = 0
    latest_publication: str | None = None
    if content_type_rejection is None and result.blocker is None and result.status is not None:
        try:
            publications = parse_rss_publications(source, result.body)
            parser_status = "SUCCESS"
            item_count = len(_latest_publications(publications))
            if publications:
                latest_publication = publications[-1].publication_timestamp_utc.isoformat()
        except ValueError as exc:
            parser_status = str(exc)
    return {
        **_network_row(source, result, fetched_at, latency_ms),
        "source_status": source.source_status.value,
        "collector_eligible": source.enabled
        and source.source_status == SourceQualificationStatus.LIVE_STRICT_EXACT_READY,
        "current_response_status": content_type_rejection
        or result.blocker
        or (
            "HTTP_FAILURE" if result.status is not None and result.status >= 400 else parser_status
        ),
        "items_parseable": item_count,
        "latest_publication": latest_publication,
    }


def _unsupported_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return "CONTENT_TYPE_MISSING"
    normalized = content_type.split(";", 1)[0].strip().lower()
    if normalized not in SUPPORTED_DISCOVERY_CONTENT_TYPES:
        return "UNSUPPORTED_CONTENT_TYPE"
    return None


def _candidate_universe_tickers(historical_tickers: tuple[str, ...]) -> list[str]:
    historical = set(historical_tickers)
    return sorted(
        {
            instrument.ticker
            for instrument in SEED_INSTRUMENTS
            if instrument.ticker not in historical
        }
    )


def _ticker_coverage(registry: Registry, ready_tickers: list[str]) -> dict[str, Any]:
    historical = set(registry.historical_frozen_issuer_tickers)
    ready = set(ready_tickers)
    return {
        "historical_frozen_issuer_tickers": list(registry.historical_frozen_issuer_tickers),
        "candidate_universe_tickers": _candidate_universe_tickers(
            registry.historical_frozen_issuer_tickers
        ),
        "ready_issuer_tickers": sorted(ready),
        "new_tickers_relative_to_historical_7": sorted(ready - historical),
        "historical_ready_overlap_tickers": sorted(ready & historical),
    }


def _timestamp_contracts(registry: Registry) -> dict[str, Any]:
    return {
        source.source_id: {
            "ticker": source.ticker,
            "TIMESTAMP_EVIDENCE_TYPE": source.timestamp_contract.get("evidence_type"),
            "TIMESTAMP_EVIDENCE_VALUE": source.timestamp_contract.get("evidence_value"),
            "TIMESTAMP_CONTRACT_SHA": source.contract_sha(),
            "timestamp_field": source.timestamp_path,
            "timezone_contract": source.timestamp_contract,
        }
        for source in registry.sources
    }


def _source_rejections(registry: Registry) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in registry.sources:
        if source.source_status == SourceQualificationStatus.LIVE_STRICT_EXACT_READY:
            continue
        rows.append(
            {
                "source_id": source.source_id,
                "ticker": source.ticker,
                "issuer": source.issuer,
                "source_status": source.source_status.value,
                "source_origin": source.source_origin,
                "collector_eligible": False,
                "reason": source.timestamp_contract.get("policy"),
            }
        )
    return rows


def _source_has_explicit_timezone(source: LiveIssuerSource) -> bool:
    return "EXPLICIT_OFFSET" in str(source.timestamp_contract.get("evidence_type"))


def _source_has_explicit_or_documented_timezone(source: LiveIssuerSource) -> bool:
    evidence_type = str(source.timestamp_contract.get("evidence_type") or "")
    evidence_value = str(source.timestamp_contract.get("evidence_value") or "")
    return (
        "EXPLICIT_OFFSET" in evidence_type
        or "DOCUMENTED_TIMEZONE" in evidence_type
        or "UTC_FIELD" in evidence_type
        or "UTC" in evidence_value.upper()
    )


def _identity(source: LiveIssuerSource, source_item_id: str) -> str:
    return f"{source.source_id}|{source_item_id}"


def _registry_contract_payload(registry: Registry) -> dict[str, Any]:
    return {
        "source_registry_version": SOURCE_REGISTRY_VERSION,
        "historical_frozen_issuer_tickers": list(registry.historical_frozen_issuer_tickers),
        "milestone": registry.milestone,
        "sources": [
            source.payload() | {"source_contract_sha": source.contract_sha()}
            for source in registry.sources
        ],
    }


def _live_diversity_status(
    *,
    historical: tuple[str, ...],
    live_tickers: list[str],
    new_tickers: list[str],
    milestone: dict[str, Any],
    observed_events: int | None = None,
) -> str:
    min_new = int(milestone.get("minimum_new_issuer_tickers", 3))
    if not live_tickers and not new_tickers and not observed_events:
        return "NOT_STARTED"
    if len(new_tickers) >= min_new:
        return "LIVE_DIVERSITY_ACCUMULATION_WORKING"
    if new_tickers or observed_events:
        return "PARTIAL"
    return "INSUFFICIENT_FREE_SOURCE_COVERAGE"


def _events_per_week(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        published = datetime.fromisoformat(str(row["published_at"]))
        iso = published.isocalendar()
        counts[f"{iso.year}-W{iso.week:02d}"] += 1
    return dict(sorted(counts.items()))


def _top_share(counter: Counter[str], n: int) -> str:
    total = sum(counter.values())
    if total == 0:
        return "0.000000"
    return f"{sum(count for _key, count in counter.most_common(n)) / total:.6f}"


def _hhi(counter: Counter[str]) -> str:
    total = sum(counter.values())
    if total == 0:
        return "0.000000"
    return f"{sum((count / total) ** 2 for count in counter.values()):.6f}"


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000000"
    return f"{numerator / denominator:.6f}"


def _int_value(value: object | None) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return 0


def _events_after_cutoff(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if datetime.fromisoformat(str(row["published_at"])) >= FUTURE_HOLDOUT_START
    ]


def _text(item: ET.Element, tag: str) -> str:
    value = next(
        (child.text for child in item if _local(child.tag) == tag and child.text is not None),
        None,
    )
    return " ".join(value.split()) if value else ""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


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


def _read_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "seen_source_item_ids": [],
            "seen_canonical_urls": [],
            "content_sha_by_identity": {},
        }
    return _read_json(path)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    next_action = (
        "Continue free first-party live source discovery for at least 3 new MOEX issuer "
        "tickers with strict timestamp/timezone contracts; keep LIVE_SHADOW_CORPUS sealed."
    )
    metrics = [
        ("BASE_MAIN_SHA", manifest["BASE_MAIN_SHA"]),
        ("HEAD_SHA", manifest["git_sha"]),
        ("artifact SHA", manifest["ARTIFACT_SHA"]),
        ("official free sources audited", manifest["FREE_OFFICIAL_SOURCES_AUDITED"]),
        ("paid sources used", manifest["PAID_SOURCES_USED"]),
        ("LIVE_STRICT_EXACT_READY_SOURCES", manifest["LIVE_STRICT_EXACT_READY_SOURCES"]),
        ("unique ready issuer tickers", manifest.get("READY_ISSUER_TICKERS", [])),
        ("historical issuer tickers", manifest.get("HISTORICAL_ISSUER_TICKERS", [])),
        ("NEW issuer tickers", manifest.get("NEW_TICKERS_RELATIVE_TO_HISTORICAL_7", [])),
        (
            "sources with explicit/documented timezone",
            manifest.get("SOURCES_WITH_EXPLICIT_OR_DOCUMENTED_TIMEZONE", 0),
        ),
        ("timezone-rejected sources", manifest.get("SOURCES_REJECTED_FOR_TIMEZONE", 0)),
        ("sources with stable identity", manifest.get("SOURCES_WITH_STABLE_IDENTITY", 0)),
        ("bounded HTTP requests", manifest.get("BOUNDED_HTTP_REQUESTS", 0)),
        ("newly observed publications", manifest.get("EVENTS_COLLECTED", 0)),
        ("raw snapshots frozen", manifest.get("RAW_SNAPSHOTS_FROZEN", 0)),
        ("duplicate publications", manifest.get("DUPLICATES_ENCOUNTERED", 0)),
        ("semantic-ready events", manifest.get("SEMANTIC_READY_EVENTS", 0)),
        (
            "UNKNOWN count/rate",
            {
                "count": manifest.get("UNKNOWN_EVENTS", 0),
                "rate": manifest.get("UNKNOWN_RATE", "0.000000"),
            },
        ),
        ("pre-event-feature-ready events", manifest.get("PRE_EVENT_FEATURE_READY_EVENTS", 0)),
        ("live outcomes read", manifest["LIVE_OUTCOMES_READ"]),
        ("targets computed", manifest["LIVE_TARGETS_COMPUTED"]),
        ("post-event price reads", manifest["LIVE_POST_EVENT_PRICE_READS"]),
        ("Rules v3 changed?", manifest["RULES_V3_CHANGED"]),
        ("model trained?", manifest["MODEL_TRAINING_PERFORMED"]),
        ("paid API calls", manifest["PAID_API_CALLS"]),
        ("live diversity status", manifest["LIVE_DIVERSITY_ACCUMULATION_STATUS"]),
        ("strict answer YES/NO", manifest["STRICT_ANSWER"]),
        ("exact next action", next_action),
    ]
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        f"ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        "",
        "## Metrics",
        "",
        *[
            f"{index}. {key}={json.dumps(value, ensure_ascii=False, sort_keys=True)}"
            for index, (key, value) in enumerate(metrics, start=1)
        ],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
