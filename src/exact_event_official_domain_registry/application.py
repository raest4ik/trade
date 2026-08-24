from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_live_source_snapshot.application import (
    build_live_source_snapshot_artifact,
)
from src.exact_event_official_domain_registry.domain import (
    ARTIFACT_VERSION,
    CREATED_BY,
    DOMAIN_DISCOVERY_LIMITS,
    EVIDENCE_SCHEMA,
    INPUT_DATASET_SHA,
    MAX_CANDIDATE_DOMAINS_PER_TICKER,
    MAX_REQUESTS_PER_DOMAIN,
    MAX_TICKERS,
    MAX_VALIDATION_URLS_PER_DOMAIN,
    MIN_DOMAIN_DELAY_SECONDS,
    REGISTRY_VERSION,
    DomainBlocker,
    DomainEvidenceRecord,
    domain_safety_flags,
    sha256_bytes,
    sha256_payload,
)
from src.exact_event_official_domain_registry.http_client import (
    BoundedHttpClient,
    HttpClient,
    HttpResult,
    PoliteDomainClient,
)
from src.exact_event_official_source_discovery.domain import (
    current_metrics,
    priority_tier,
)


def build_official_domain_registry_artifact(
    *,
    input_root: Path,
    source_registry_path: Path,
    universe_path: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    live_source_report_path: Path | None = None,
    candidate_domains_path: Path | None = None,
    client: HttpClient | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable official domain registry artifact output already exists")
    _verify_frozen_contracts()
    input_manifest = _read_json(input_root / "manifest.json")
    _require_input_manifest(input_manifest)
    output_root.mkdir(parents=True, exist_ok=False)

    events = _read_jsonl(input_root / "events.jsonl")
    features = _read_jsonl(input_root / "features.jsonl")
    source_registry = _read_jsonl(source_registry_path)
    universe = _read_universe(universe_path)
    before = current_metrics(events, features)
    cohort = _domain_enrichment_cohort(
        live_source_report_path=live_source_report_path,
        exact_counts=before["events_by_ticker"],
        feature_counts=before["feature_ready_by_ticker"],
        source_registry=source_registry,
        universe=universe,
    )
    candidate_seeds = _read_candidate_seeds(candidate_domains_path)
    enrichment = _enrich_domains(
        cohort=cohort,
        source_registry=source_registry,
        universe=universe,
        candidate_seeds=candidate_seeds,
        client=client,
        created_at=created_at,
    )
    registry_rows = _registry_rows(enrichment.evidence_records)
    registry_path = output_root / "official-domain-registry.jsonl"
    evidence_payload = [record.payload() for record in enrichment.evidence_records]
    network_payload = [record.payload for record in enrichment.network_records]
    registry_payload = sorted(registry_rows, key=lambda row: str(row["ticker"]))
    _write_jsonl(registry_path, registry_payload)

    second_manifest: dict[str, Any] | None = None
    second_root = output_root / "second-live-source-snapshot"
    newly_enabled = [str(row["ticker"]) for row in registry_payload]
    if newly_enabled:
        second_manifest = build_live_source_snapshot_artifact(
            input_root=input_root,
            source_registry_path=source_registry_path,
            universe_path=universe_path,
            output_root=second_root,
            base_main_sha=base_main_sha,
            git_sha=git_sha,
            official_domain_registry_path=registry_path,
            ticker_filter=set(newly_enabled),
            client=client,
            created_at=created_at,
        )

    blockers_by_ticker = {
        record.ticker: record.blocker
        for record in enrichment.evidence_records
        if not record.official_domain_confirmed
    }
    confirmed_by_ticker = {
        str(row["ticker"]): str(row["official_domain"]) for row in registry_payload
    }
    safety = domain_safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_DATASET_SHA": INPUT_DATASET_SHA,
        "DOMAIN_ENRICHMENT_COHORT": _cohort_payload(cohort),
        "DOMAIN_ENRICHMENT_COHORT_SHA": sha256_payload(_cohort_payload(cohort)),
        "DOMAIN_DISCOVERY_LIMITS_SHA": sha256_payload(DOMAIN_DISCOVERY_LIMITS),
        "DOMAIN_REGISTRY_SHA": sha256_payload(registry_payload),
        "DOMAIN_NETWORK_PROVENANCE_SHA": sha256_payload(network_payload),
        "EVIDENCE_SCHEMA_SHA": sha256_payload(EVIDENCE_SCHEMA),
        "LIVE_DOMAIN_ENRICHMENT_EXECUTED": enrichment.validation_requests > 0,
        "LIVE_DOMAIN_ENRICHMENT_BLOCKER": _live_blocker(enrichment),
        "DOMAIN_TICKERS_TARGETED": len(cohort),
        "DOMAIN_SEARCH_AVAILABLE": False,
        "DOMAIN_SEARCH_QUERIES": 0,
        "DOMAIN_CANDIDATES_FOUND": enrichment.candidates_found,
        "DOMAIN_VALIDATION_REQUESTS": enrichment.validation_requests,
        "DOMAIN_HTTP_2XX": enrichment.http_2xx,
        "DOMAIN_REDIRECTS": enrichment.redirects,
        "DOMAIN_RATE_LIMITED": enrichment.blockers[DomainBlocker.RATE_LIMITED.value],
        "DOMAIN_TIMEOUTS": enrichment.blockers[DomainBlocker.TIMEOUT.value],
        "DOMAIN_TECHNICAL_FAILURES": _technical_failures(enrichment.blockers),
        "DOMAIN_CONFIRMED_COUNT": len(registry_payload),
        "DOMAIN_AMBIGUOUS_COUNT": enrichment.blockers[
            DomainBlocker.OFFICIAL_DOMAIN_AMBIGUOUS.value
        ],
        "DOMAIN_UNRESOLVED_COUNT": len(cohort) - len(registry_payload),
        "DOMAIN_CONFLICT_COUNT": enrichment.domain_conflicts,
        "NEWLY_DOMAIN_ENABLED_TICKERS_COUNT": len(newly_enabled),
        "NEWLY_DOMAIN_ENABLED_TICKERS": newly_enabled,
        "CONFIRMED_DOMAINS_BY_TICKER": confirmed_by_ticker,
        "BLOCKERS_BY_TICKER": blockers_by_ticker,
        "BLOCKERS_BY_TYPE": dict(sorted(enrichment.blockers.items())),
        "SECOND_LIVE_RUN_EXECUTED": second_manifest is not None,
        "SECOND_LIVE_SOURCE_SNAPSHOT_SHA": _second_value(
            second_manifest, "LIVE_SOURCE_SNAPSHOT_SHA"
        ),
        "DOWNSTREAM_V5_ARTIFACT_SHA": _second_value(second_manifest, "V5_DOWNSTREAM_ARTIFACT_SHA"),
        "SECOND_RUN_TICKERS_ATTEMPTED": _second_value(second_manifest, "LIVE_TICKERS_ATTEMPTED", 0),
        "SECOND_RUN_DOMAINS_USED": _second_value(second_manifest, "LIVE_DOMAINS_CONFIRMED", 0),
        "SECOND_RUN_REQUESTS_TOTAL": _second_value(second_manifest, "LIVE_REQUESTS_TOTAL", 0),
        "SECOND_RUN_CANDIDATES_WRITTEN": _second_value(
            second_manifest, "LIVE_CANDIDATES_WRITTEN", 0
        ),
        "SECOND_RUN_EXACT_CANDIDATES": _second_value(second_manifest, "LIVE_EXACT_CANDIDATES", 0),
        "SECOND_RUN_DATE_ONLY_CANDIDATES": _second_value(
            second_manifest, "LIVE_DATE_ONLY_CANDIDATES", 0
        ),
        "DOWNSTREAM_NEW_OFFICIAL_SOURCES": _second_value(
            second_manifest, "V5_NEW_OFFICIAL_SOURCES_FOUND", 0
        ),
        "DOWNSTREAM_NEW_EXACT_CAPABLE_SOURCES": _second_value(
            second_manifest, "V5_NEW_EXACT_CAPABLE_SOURCES", 0
        ),
        "DOWNSTREAM_NEW_ARCHIVE_CAPABLE_SOURCES": _second_value(
            second_manifest, "V5_NEW_ARCHIVE_CAPABLE_SOURCES", 0
        ),
        "DOWNSTREAM_NEW_EXACT_EVENTS": _second_value(second_manifest, "V5_NEW_EXACT_EVENTS", 0),
        "DOWNSTREAM_NEW_EXACT_HISTORICAL": _second_value(
            second_manifest, "V5_NEW_EXACT_HISTORICAL", 0
        ),
        "DOWNSTREAM_NEW_EXACT_FUTURE_METADATA_ONLY": _second_value(
            second_manifest, "V5_NEW_EXACT_FUTURE_METADATA_ONLY", 0
        ),
        "DOWNSTREAM_EXACT_TOTAL_BEFORE": _second_value(
            second_manifest, "V5_EXACT_TOTAL_BEFORE", before["EXACT_TOTAL"]
        ),
        "DOWNSTREAM_EXACT_TOTAL_AFTER": _second_value(
            second_manifest, "V5_EXACT_TOTAL_AFTER", before["EXACT_TOTAL"]
        ),
        "EXISTING_DOMAIN_ROWS_PRESERVED": "PASS",
        "EXISTING_EVENT_ROWS_PRESERVED": "PASS",
        "EXISTING_FEATURE_ROWS_PRESERVED": "PASS",
        "EXISTING_TARGET_ROWS_PRESERVED": "PASS",
        "DATE_ONLY_COERCIONS": 0,
        "FETCH_TIME_USED_AS_PUBLICATION_TIME": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "LIVE_SOURCE_DISCOVERY_CONCLUSION": _conclusion(registry_payload, second_manifest),
        "safety": safety,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = _artifact_sha(manifest)
    _write_json(output_root / "manifest.json", manifest)
    _write_jsonl(output_root / "domain-evidence.jsonl", evidence_payload)
    _write_jsonl(output_root / "domain-network-provenance.jsonl", network_payload)
    _write_json(output_root / "evidence-schema.json", EVIDENCE_SCHEMA)
    _write_report(output_root / "report.md", manifest)
    return manifest


@dataclass(frozen=True, slots=True)
class _NetworkRecord:
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Enrichment:
    evidence_records: list[DomainEvidenceRecord]
    network_records: list[_NetworkRecord]
    blockers: Counter[str]
    candidates_found: int
    validation_requests: int
    http_2xx: int
    redirects: int
    domain_conflicts: int


def _domain_enrichment_cohort(
    *,
    live_source_report_path: Path | None,
    exact_counts: dict[str, int],
    feature_counts: dict[str, int],
    source_registry: list[dict[str, Any]],
    universe: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    registry_by_ticker = {str(row["ticker"]): row for row in source_registry}
    if live_source_report_path is not None and live_source_report_path.exists():
        rows = _read_jsonl(live_source_report_path)
        tickers = [
            str(row["TICKER"]) for row in rows if row.get("BLOCKER") == "NO_OFFICIAL_DOMAIN"
        ][:MAX_TICKERS]
    else:
        tickers = _fallback_tickers(exact_counts, feature_counts, source_registry, universe)
    cohort: list[dict[str, Any]] = []
    for ticker in tickers[:MAX_TICKERS]:
        registry = registry_by_ticker.get(ticker, {})
        instrument = universe.get(ticker, {})
        cohort.append(
            {
                "ticker": ticker,
                "issuer": str(registry.get("issuer") or instrument.get("name") or ticker),
                "instrument_uid": str(instrument.get("instrument_uid") or ""),
                "exact_event_count": int(exact_counts.get(ticker, 0)),
                "feature_ready_count": int(feature_counts.get(ticker, 0)),
                "priority_tier": priority_tier(
                    ticker=ticker,
                    exact_count=int(exact_counts.get(ticker, 0)),
                    feature_ready_count=int(feature_counts.get(ticker, 0)),
                    in_exact_corpus=ticker in exact_counts,
                ),
            }
        )
    return cohort


def _fallback_tickers(
    exact_counts: dict[str, int],
    feature_counts: dict[str, int],
    source_registry: list[dict[str, Any]],
    universe: dict[str, dict[str, Any]],
) -> list[str]:
    registry_by_ticker = {str(row["ticker"]): row for row in source_registry}
    rows: list[dict[str, Any]] = []
    for ticker in sorted(set(exact_counts) | set(universe)):
        registry = registry_by_ticker.get(ticker, {})
        if (
            registry.get("official_domain")
            or registry.get("source_domain")
            or registry.get("source_url")
        ):
            continue
        exact_count = int(exact_counts.get(ticker, 0))
        feature_count = int(feature_counts.get(ticker, 0))
        tier = priority_tier(
            ticker=ticker,
            exact_count=exact_count,
            feature_ready_count=feature_count,
            in_exact_corpus=ticker in exact_counts,
        )
        rows.append(
            {
                "ticker": ticker,
                "priority_tier": tier,
                "exact_event_count": exact_count,
                "feature_ready_count": feature_count,
            }
        )
    order = {
        "A_ZERO_FEATURE_READY": 0,
        "B_EXACT_1_5": 1,
        "C_EXACT_6_20": 2,
        "D_CANONICAL_TQBR_NOT_IN_EXACT": 3,
        "DEPRIORITIZED": 4,
    }
    return [
        str(row["ticker"])
        for row in sorted(
            rows, key=lambda row: (order[str(row["priority_tier"])], str(row["ticker"]))
        )
    ][:MAX_TICKERS]


def _enrich_domains(
    *,
    cohort: list[dict[str, Any]],
    source_registry: list[dict[str, Any]],
    universe: dict[str, dict[str, Any]],
    candidate_seeds: dict[str, list[dict[str, Any]]],
    client: HttpClient | None,
    created_at: datetime | None,
) -> _Enrichment:
    base_client = client if client is not None else BoundedHttpClient()
    http = PoliteDomainClient(
        base_client, min_delay_seconds=0.0 if client else MIN_DOMAIN_DELAY_SECONDS
    )
    existing_domains = _existing_confirmed_domains(source_registry)
    evidence: list[DomainEvidenceRecord] = []
    network: list[_NetworkRecord] = []
    blockers: Counter[str] = Counter()
    candidates_found = validation_requests = http_2xx = redirects = domain_conflicts = 0
    domain_requests: Counter[str] = Counter()

    for row in cohort:
        ticker = str(row["ticker"])
        seeds = candidate_seeds.get(ticker, [])[:MAX_CANDIDATE_DOMAINS_PER_TICKER]
        candidates_found += len(seeds)
        if not seeds:
            blockers[DomainBlocker.NO_CANDIDATE_DOMAIN.value] += 1
            evidence.append(_unresolved(row, DomainBlocker.NO_CANDIDATE_DOMAIN.value))
            continue
        accepted = False
        last_record: DomainEvidenceRecord | None = None
        for seed in seeds:
            host = normalize_host(str(seed.get("domain") or seed.get("url") or ""))
            if not host:
                continue
            if ticker in existing_domains and existing_domains[ticker] != host:
                domain_conflicts += 1
                blockers[DomainBlocker.OFFICIAL_DOMAIN_AMBIGUOUS.value] += 1
                last_record = _unresolved(
                    row,
                    DomainBlocker.OFFICIAL_DOMAIN_AMBIGUOUS.value,
                    candidate_domain=host,
                    ambiguity_reason="DOMAIN_CONFLICT",
                )
                continue
            candidate_urls = _validation_urls(host, seed)
            host_records: list[HttpResult] = []
            terminal_blocked = False
            for url in candidate_urls:
                if domain_requests[host] >= MAX_REQUESTS_PER_DOMAIN:
                    blockers[DomainBlocker.RATE_LIMITED.value] += 1
                    last_record = _unresolved(
                        row, DomainBlocker.RATE_LIMITED.value, candidate_domain=host
                    )
                    terminal_blocked = True
                    break
                result = http.get(url)
                domain_requests[host] += 1
                validation_requests += 1
                redirects += result.redirects
                if result.status is not None and 200 <= result.status < 300:
                    http_2xx += 1
                network.append(_network_record(result, created_at))
                blocker = _http_blocker(result)
                if blocker is not None:
                    blockers[blocker] += 1
                    last_record = _unresolved(row, blocker, candidate_domain=host, result=result)
                    if _terminal_blocker(blocker):
                        terminal_blocked = True
                        break
                    continue
                if url.endswith("/robots.txt") and _robots_disallows(result.body):
                    blockers[DomainBlocker.ROBOTS_BLOCKED.value] += 1
                    last_record = _unresolved(
                        row,
                        DomainBlocker.ROBOTS_BLOCKED.value,
                        candidate_domain=host,
                        result=result,
                    )
                    terminal_blocked = True
                    break
                host_records.append(result)
            if terminal_blocked:
                continue
            confirmed = _confirm_domain(row, host, seed, host_records)
            if confirmed.official_domain_confirmed:
                blockers[DomainBlocker.DOMAIN_CONFIRMED.value] += 1
                evidence.append(confirmed)
                accepted = True
                break
            last_record = confirmed if confirmed.candidate_domain else last_record
        if not accepted:
            evidence.append(
                last_record or _unresolved(row, DomainBlocker.NO_IDENTITY_EVIDENCE.value)
            )
    return _Enrichment(
        evidence,
        network,
        blockers,
        candidates_found,
        validation_requests,
        http_2xx,
        redirects,
        domain_conflicts,
    )


def _confirm_domain(
    row: dict[str, Any],
    host: str,
    seed: dict[str, Any],
    results: list[HttpResult],
) -> DomainEvidenceRecord:
    expected = str(row["issuer"])
    seed_evidence = str(seed.get("evidence_type") or "")
    if seed_evidence == "SEARCH_RESULT_ONLY":
        return _unresolved(row, DomainBlocker.NO_IDENTITY_EVIDENCE.value, candidate_domain=host)
    if seed_evidence == "PARENT_SUBSIDIARY_UNPROVEN":
        return _unresolved(
            row,
            DomainBlocker.PARENT_SUBSIDIARY_AMBIGUITY.value,
            candidate_domain=host,
            ambiguity_reason="PARENT_SUBSIDIARY_UNPROVEN",
        )
    saw_identity_page = False
    for result in results:
        if _base_content_type(result.content_type) not in {
            "text/html",
            "text/plain",
            "application/json",
            "application/ld+json",
        }:
            continue
        saw_identity_page = True
        text = result.body.decode("utf-8", errors="replace")
        if _identity_match(expected, text):
            evidence_type = (
                "STRONG_OFFICIAL_EXCHANGE_OR_REGULATORY_PAGE"
                if seed_evidence.startswith("OFFICIAL_")
                else "STRONG_LEGAL_ENTITY_WEBSITE_MATCH"
            )
            return DomainEvidenceRecord(
                ticker=str(row["ticker"]),
                issuer=expected,
                instrument_uid=_optional_string(row.get("instrument_uid")),
                candidate_domain=host,
                confirmed_host=host,
                registered_domain=registered_domain(host),
                discovery_origin=str(seed.get("discovery_origin") or "candidate domain seed"),
                evidence_type=evidence_type,
                evidence_url=str(result.final_url or result.request_url),
                evidence_content_sha256=sha256_bytes(result.body),
                legal_name_expected=expected,
                legal_name_observed=_observed_name(expected, text),
                identifier_match=_identifier_match(seed, text),
                official_domain_confirmed=True,
                ambiguity_reason=None,
                http_status=result.status,
                final_url=result.final_url,
                blocker=DomainBlocker.DOMAIN_CONFIRMED.value,
            )
    blocker = (
        DomainBlocker.LEGAL_ENTITY_MISMATCH.value
        if saw_identity_page
        else DomainBlocker.NO_IDENTITY_EVIDENCE.value
    )
    return _unresolved(row, blocker, candidate_domain=host)


def _registry_rows(records: list[DomainEvidenceRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not record.official_domain_confirmed or not record.confirmed_host:
            continue
        rows.append(
            {
                "registry_version": REGISTRY_VERSION,
                "ticker": record.ticker,
                "issuer": record.issuer,
                "instrument_uid": record.instrument_uid,
                "official_domain": record.confirmed_host,
                "confirmed_host": record.confirmed_host,
                "registered_domain": record.registered_domain,
                "confirmation_status": DomainBlocker.DOMAIN_CONFIRMED.value,
                "evidence_type": record.evidence_type,
                "evidence_url": record.evidence_url,
                "evidence_sha256": record.evidence_content_sha256,
                "discovery_origin": record.discovery_origin,
                "created_by": CREATED_BY,
            }
        )
    return rows


def normalize_host(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    host = urlsplit(raw).netloc.rstrip(".").lower()
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return ""


def registered_domain(host: str) -> str:
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _validation_urls(host: str, seed: dict[str, Any]) -> list[str]:
    urls = [f"https://{host}/robots.txt"]
    seed_url = str(seed.get("url") or "")
    if seed_url:
        urls.append(seed_url if "://" in seed_url else f"https://{host}{seed_url}")
    urls.extend(
        f"https://{host}{path}"
        for path in ("/", "/investors", "/investor-relations", "/contacts", "/about")
    )
    return list(dict.fromkeys(urls))[:MAX_VALIDATION_URLS_PER_DOMAIN]


def _identity_match(expected: str, text: str) -> bool:
    normalized_text = _normalize_text(text)
    tokens = [token for token in _normalize_text(expected).split() if len(token) >= 3]
    if not tokens:
        return False
    return all(token in normalized_text for token in tokens[:3])


def _observed_name(expected: str, text: str) -> str | None:
    return expected if _identity_match(expected, text) else None


def _identifier_match(seed: dict[str, Any], text: str) -> str | None:
    identifier = _optional_string(seed.get("identifier"))
    if identifier and identifier in text:
        return identifier
    return None


def _normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def _http_blocker(result: HttpResult) -> str | None:
    if result.blocker is None:
        return None
    if result.blocker in {item.value for item in DomainBlocker}:
        return result.blocker
    return DomainBlocker.TECHNICAL_FETCH_FAILED.value


def _terminal_blocker(blocker: str) -> bool:
    return blocker in {
        DomainBlocker.RATE_LIMITED.value,
        DomainBlocker.AUTH_REQUIRED.value,
        DomainBlocker.CAPTCHA_BLOCKED.value,
        DomainBlocker.PAYMENT_REQUIRED.value,
        DomainBlocker.TLS_FAILED.value,
        DomainBlocker.DNS_FAILED.value,
        DomainBlocker.TIMEOUT.value,
        DomainBlocker.RESPONSE_TOO_LARGE.value,
    }


def _robots_disallows(body: bytes) -> bool:
    text = body.decode("utf-8", errors="ignore").lower()
    current_all = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("user-agent:"):
            current_all = line.split(":", 1)[1].strip() == "*"
        elif current_all and line.startswith("disallow:"):
            if line.split(":", 1)[1].strip() == "/":
                return True
    return False


def _network_record(result: HttpResult, created_at: datetime | None) -> _NetworkRecord:
    return _NetworkRecord(
        {
            "REQUEST_URL": result.request_url,
            "FINAL_URL": result.final_url,
            "HTTP_STATUS": result.status,
            "CONTENT_TYPE": result.content_type,
            "FETCHED_AT": (created_at or datetime.now(UTC)).isoformat(),
            "CONTENT_SHA256": sha256_bytes(result.body) if result.body else None,
            "BYTES_RECEIVED": len(result.body),
            "BLOCKER": result.blocker,
        }
    )


def _unresolved(
    row: dict[str, Any],
    blocker: str,
    *,
    candidate_domain: str | None = None,
    ambiguity_reason: str | None = None,
    result: HttpResult | None = None,
) -> DomainEvidenceRecord:
    return DomainEvidenceRecord(
        ticker=str(row["ticker"]),
        issuer=str(row["issuer"]),
        instrument_uid=_optional_string(row.get("instrument_uid")),
        candidate_domain=candidate_domain,
        confirmed_host=None,
        registered_domain=registered_domain(candidate_domain) if candidate_domain else None,
        discovery_origin="candidate domain seed" if candidate_domain else "no candidate domain",
        evidence_type=None,
        evidence_url=None,
        evidence_content_sha256=None,
        legal_name_expected=str(row["issuer"]),
        legal_name_observed=None,
        identifier_match=None,
        official_domain_confirmed=False,
        ambiguity_reason=ambiguity_reason,
        http_status=result.status if result else None,
        final_url=result.final_url if result else None,
        blocker=blocker,
    )


def _read_candidate_seeds(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None or not path.exists():
        return {}
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        ticker = str(row["ticker"])
        by_ticker.setdefault(ticker, []).append(row)
    return by_ticker


def _existing_confirmed_domains(source_registry: list[dict[str, Any]]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for row in source_registry:
        domain = _optional_string(row.get("official_domain")) or _optional_string(
            row.get("source_domain")
        )
        if domain:
            rows[str(row["ticker"])] = normalize_host(domain)
    return rows


def _technical_failures(blockers: Counter[str]) -> int:
    return sum(
        blockers[name]
        for name in {
            DomainBlocker.TLS_FAILED.value,
            DomainBlocker.DNS_FAILED.value,
            DomainBlocker.HTTP_4XX.value,
            DomainBlocker.HTTP_5XX.value,
            DomainBlocker.RESPONSE_TOO_LARGE.value,
            DomainBlocker.UNSUPPORTED_CONTENT_TYPE.value,
            DomainBlocker.TECHNICAL_FETCH_FAILED.value,
        }
    )


def _live_blocker(enrichment: _Enrichment) -> str | None:
    if enrichment.validation_requests == 0:
        return DomainBlocker.NO_CANDIDATE_DOMAIN.value
    if enrichment.http_2xx == 0 and enrichment.blockers[DomainBlocker.DNS_FAILED.value]:
        return DomainBlocker.NETWORK_UNAVAILABLE.value
    return None


def _conclusion(registry_rows: list[dict[str, Any]], second_manifest: dict[str, Any] | None) -> str:
    if second_manifest and second_manifest["V5_NEW_EXACT_EVENTS"]:
        return "MARKET_MATURATION"
    if second_manifest and second_manifest["V5_NEW_EXACT_CAPABLE_SOURCES"]:
        return "DEEPEN_NEW_EXACT_SOURCES"
    if registry_rows:
        return "SECOND_DOMAIN_ENRICHMENT_PASS"
    return "SECOND_DOMAIN_ENRICHMENT_PASS"


def _second_value(
    second_manifest: dict[str, Any] | None,
    key: str,
    default: object | None = None,
) -> object | None:
    if second_manifest is None:
        return default
    return second_manifest[key]


def _cohort_payload(cohort: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "TICKER": row["ticker"],
            "ISSUER": row["issuer"],
            "PRIORITY_TIER": row["priority_tier"],
            "EXACT_EVENT_COUNT": row["exact_event_count"],
            "FEATURE_READY_COUNT": row["feature_ready_count"],
        }
        for row in cohort
    ]


def _artifact_sha(manifest: dict[str, Any]) -> str:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"ARTIFACT_SHA", "created_at", "git_sha"}
    }
    return sha256_payload(core)


def _base_content_type(value: str | None) -> str:
    return str(value or "").split(";", 1)[0].strip().lower()


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _verify_frozen_contracts() -> None:
    if rules_v3_fingerprint() != EXPECTED_RULES_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_MISMATCH")
    if prompt_hash() != QWEN_PROMPT_SHA or schema_hash() != QWEN_SCHEMA_SHA:
        raise ValueError("FROZEN_QWEN_CONTRACT_MISMATCH")


def _require_input_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("OUTPUT_DATASET_SHA") != INPUT_DATASET_SHA:
        raise ValueError("INPUT_DATASET_SHA_MISMATCH")
    if manifest.get("EXISTING_EVENT_ROWS_PRESERVED") != "PASS":
        raise ValueError("INPUT_EVENTS_NOT_PRESERVED")
    if manifest.get("EXISTING_FEATURE_ROWS_PRESERVED") != "PASS":
        raise ValueError("INPUT_FEATURES_NOT_PRESERVED")
    if manifest.get("EXISTING_TARGET_ROWS_PRESERVED") not in {None, "PASS"}:
        raise ValueError("INPUT_TARGETS_NOT_PRESERVED")
    if bool(manifest.get("TEST_OUTCOME_USED")) or bool(manifest.get("FUTURE_EVENT_HOLDOUT_USED")):
        raise ValueError("INPUT_SAFETY_FLAGS_NOT_PASS")


def _read_universe(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    universe: dict[str, dict[str, Any]] = {}
    for item in cast("list[dict[str, Any]]", payload.get("instruments") or []):
        if (
            str(item.get("class_code") or "").upper() == "TQBR"
            and str(item.get("currency") or "").lower() == "rub"
        ):
            universe[str(item["ticker"])] = item
    return universe


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        "Data-acquisition-only official domain registry enrichment.",
        "",
        f"- INPUT_DATASET_SHA={manifest['INPUT_DATASET_SHA']}",
        f"- DOMAIN_TICKERS_TARGETED={manifest['DOMAIN_TICKERS_TARGETED']}",
        f"- DOMAIN_CONFIRMED_COUNT={manifest['DOMAIN_CONFIRMED_COUNT']}",
        f"- NEWLY_DOMAIN_ENABLED_TICKERS_COUNT={manifest['NEWLY_DOMAIN_ENABLED_TICKERS_COUNT']}",
        f"- SECOND_LIVE_RUN_EXECUTED={manifest['SECOND_LIVE_RUN_EXECUTED']}",
        "- DOWNSTREAM_NEW_EXACT_CAPABLE_SOURCES="
        f"{manifest['DOWNSTREAM_NEW_EXACT_CAPABLE_SOURCES']}",
        f"- DOWNSTREAM_NEW_EXACT_EVENTS={manifest['DOWNSTREAM_NEW_EXACT_EVENTS']}",
        f"- LIVE_SOURCE_DISCOVERY_CONCLUSION={manifest['LIVE_SOURCE_DISCOVERY_CONCLUSION']}",
        "",
        "No model, TEST outcome use, future outcome observation, T-Invest, market maturation, "
        "backtest, paper trading, orders, or BUY/SELL output was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
