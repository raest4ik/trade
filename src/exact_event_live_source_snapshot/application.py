from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin, urlsplit, urlunsplit

from src.exact_event_live_source_snapshot.domain import (
    ARTIFACT_VERSION,
    CACHE_SCHEMA,
    INPUT_DATASET_SHA,
    NETWORK_LIMITS,
    STANDARD_PATH_PROBES,
    SUPPORTED_CONTENT_TYPES,
    LiveBlocker,
    NetworkProvenance,
    SourceReport,
    live_safety_flags,
    sha256_bytes,
    sha256_payload,
)
from src.exact_event_live_source_snapshot.http_client import (
    BoundedHttpClient,
    HttpClient,
    HttpResult,
    PoliteDomainClient,
)
from src.exact_event_official_source_discovery.application import (
    build_official_source_discovery_artifact,
)
from src.exact_event_official_source_discovery.domain import (
    MAX_ITEMS_PER_SOURCE,
    MAX_PAGES_PER_SOURCE,
    MAX_REQUESTS_PER_DOMAIN,
    MAX_TICKERS,
    MAX_URLS_PER_TICKER,
    current_metrics,
    parse_exact_timestamp,
    priority_tier,
)


def build_live_source_snapshot_artifact(
    *,
    input_root: Path,
    source_registry_path: Path,
    universe_path: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    client: HttpClient | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable live source snapshot artifact output already exists")
    output_root.mkdir(parents=True, exist_ok=False)
    snapshot_root = output_root / "live-source-snapshot-cache"
    downstream_root = output_root / "v5-downstream"
    snapshot_root.mkdir()

    events = _read_jsonl(input_root / "events.jsonl")
    features = _read_jsonl(input_root / "features.jsonl")
    registry = _read_jsonl(source_registry_path)
    universe = _read_universe(universe_path)
    before = current_metrics(events, features)
    priority_rows = _v5_priority_rows(
        before["events_by_ticker"],
        before["feature_ready_by_ticker"],
        registry,
        universe,
    )[:MAX_TICKERS]
    priority_payload = [
        {
            "TICKER": row["ticker"],
            "ISSUER": row["issuer"],
            "PRIORITY_TIER": row["priority_tier"],
            "EXACT_EVENT_COUNT": row["exact_event_count"],
            "FEATURE_READY_COUNT": row["feature_ready_count"],
        }
        for row in priority_rows
    ]
    acquisition = _acquire_snapshot(
        priority_rows=priority_rows,
        snapshot_root=snapshot_root,
        client=client,
        created_at=created_at,
    )
    downstream_manifest = build_official_source_discovery_artifact(
        input_root=input_root,
        source_registry_path=source_registry_path,
        universe_path=universe_path,
        output_root=downstream_root,
        base_main_sha=base_main_sha,
        git_sha=git_sha,
        discovery_cache_root=snapshot_root,
        created_at=created_at,
    )
    canonical_candidates = _read_snapshot_candidates(snapshot_root)
    network_payload = [row.payload() for row in acquisition.network_provenance]
    source_report = [row.payload() for row in acquisition.source_reports]
    safety = live_safety_flags()
    live_discovery_blocker = _live_discovery_blocker(acquisition)
    live_discovery_executed = (
        acquisition.requests_total > 0
        and live_discovery_blocker != LiveBlocker.NETWORK_UNAVAILABLE.value
    )
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_DATASET_SHA": INPUT_DATASET_SHA,
        "OUTPUT_DATASET_SHA": downstream_manifest["OUTPUT_DATASET_SHA"],
        "LIVE_SOURCE_SNAPSHOT_SHA": sha256_payload(canonical_candidates),
        "NETWORK_PROVENANCE_SHA": sha256_payload(network_payload),
        "CACHE_SCHEMA_SHA": sha256_payload(CACHE_SCHEMA),
        "STANDARD_PATH_PROBES_SHA": sha256_payload(STANDARD_PATH_PROBES),
        "NETWORK_LIMITS_SHA": sha256_payload(NETWORK_LIMITS),
        "PRIORITY_COHORT_SHA": sha256_payload(priority_payload),
        "PRIORITY_TICKERS": [str(row["ticker"]) for row in priority_rows],
        "LIVE_DISCOVERY_EXECUTED": live_discovery_executed,
        "LIVE_DISCOVERY_BLOCKER": live_discovery_blocker,
        "LIVE_TICKERS_ATTEMPTED": acquisition.tickers_attempted,
        "LIVE_DOMAINS_CONFIRMED": acquisition.domains_confirmed,
        "LIVE_DOMAINS_AMBIGUOUS": acquisition.domains_ambiguous,
        "LIVE_REQUESTS_TOTAL": acquisition.requests_total,
        "LIVE_HTTP_2XX": acquisition.http_2xx,
        "LIVE_REDIRECTS": acquisition.redirects,
        "LIVE_ROBOTS_BLOCKED": acquisition.blockers[LiveBlocker.ROBOTS_BLOCKED.value],
        "LIVE_RATE_LIMITED": acquisition.blockers[LiveBlocker.RATE_LIMITED.value],
        "LIVE_AUTH_BLOCKED": acquisition.blockers[LiveBlocker.AUTH_REQUIRED.value],
        "LIVE_CAPTCHA_BLOCKED": acquisition.blockers[LiveBlocker.CAPTCHA_BLOCKED.value],
        "LIVE_TIMEOUTS": acquisition.blockers[LiveBlocker.TIMEOUT.value],
        "LIVE_TECHNICAL_FAILURES": sum(
            acquisition.blockers[name]
            for name in {
                LiveBlocker.TECHNICAL_FETCH_FAILED.value,
                LiveBlocker.TLS_FAILED.value,
                LiveBlocker.DNS_FAILED.value,
                LiveBlocker.HTTP_4XX.value,
                LiveBlocker.HTTP_5XX.value,
                LiveBlocker.UNSUPPORTED_CONTENT_TYPE.value,
                LiveBlocker.RESPONSE_TOO_LARGE.value,
            }
        ),
        "LIVE_PAGES_PARSED": acquisition.pages_parsed,
        "LIVE_FEEDS_FOUND": acquisition.feeds_found,
        "LIVE_SITEMAPS_FOUND": acquisition.sitemaps_found,
        "LIVE_JSONLD_SOURCES_FOUND": acquisition.jsonld_sources_found,
        "LIVE_CANDIDATES_WRITTEN": len(canonical_candidates),
        "LIVE_EXACT_CANDIDATES": sum(
            1 for row in canonical_candidates if row["timestamp_capability"] == "EXACT"
        ),
        "LIVE_DATE_ONLY_CANDIDATES": sum(
            1 for row in canonical_candidates if row["timestamp_capability"] == "DATE_ONLY"
        ),
        "PER_SOURCE_REPORT": source_report,
        "NETWORK_PROVENANCE": network_payload,
        "V5_DOWNSTREAM_ARTIFACT_SHA": downstream_manifest["ARTIFACT_SHA"],
        "V5_SOURCES_AUDITED": downstream_manifest["SOURCES_AUDITED"],
        "V5_NEW_OFFICIAL_SOURCES_FOUND": downstream_manifest["NEW_OFFICIAL_SOURCES_FOUND"],
        "V5_NEW_EXACT_CAPABLE_SOURCES": downstream_manifest["NEW_EXACT_CAPABLE_SOURCES"],
        "V5_NEW_ARCHIVE_CAPABLE_SOURCES": downstream_manifest["NEW_ARCHIVE_CAPABLE_SOURCES"],
        "V5_NEW_EXACT_EVENTS": downstream_manifest["NEW_EXACT_EVENTS"],
        "V5_NEW_EXACT_HISTORICAL": downstream_manifest["NEW_EXACT_HISTORICAL"],
        "V5_NEW_EXACT_FUTURE_METADATA_ONLY": downstream_manifest["NEW_EXACT_FUTURE_METADATA_ONLY"],
        "V5_NEW_EXACT_TICKERS": downstream_manifest["NEW_EXACT_TICKERS"],
        "V5_NEW_EXACT_ISSUERS": downstream_manifest["NEW_EXACT_ISSUERS"],
        "V5_EXACT_TOTAL_BEFORE": downstream_manifest["EXACT_TOTAL_BEFORE"],
        "V5_EXACT_TOTAL_AFTER": downstream_manifest["EXACT_TOTAL_AFTER"],
        "SOURCE_CANDIDATE_DEDUPE": "PASS",
        "EXISTING_EVENT_ROWS_PRESERVED": downstream_manifest["EXISTING_EVENT_ROWS_PRESERVED"],
        "EXISTING_FEATURE_ROWS_PRESERVED": downstream_manifest["EXISTING_FEATURE_ROWS_PRESERVED"],
        "EXISTING_TARGET_ROWS_PRESERVED": downstream_manifest["EXISTING_TARGET_ROWS_PRESERVED"],
        "DATE_ONLY_COERCIONS": 0,
        "FETCH_TIME_USED_AS_PUBLICATION_TIME": False,
        "STRICT_EXACT_METHODOLOGY_CHANGED": False,
        "SOURCE_ABSENCE_CONCLUSION_ALLOWED": live_discovery_executed,
        "LIVE_SOURCE_DISCOVERY_CONCLUSION": _conclusion(acquisition, downstream_manifest),
        "safety": safety,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = _artifact_sha(manifest)
    _write_json(output_root / "manifest.json", manifest)
    _write_jsonl(output_root / "network-provenance.jsonl", network_payload)
    _write_jsonl(output_root / "source-report.jsonl", source_report)
    _write_json(output_root / "cache-schema.json", CACHE_SCHEMA)
    _write_report(output_root / "report.md", manifest)
    return manifest


@dataclass(slots=True)
class _Acquisition:
    tickers_attempted: int
    domains_confirmed: int
    domains_ambiguous: int
    requests_total: int
    http_2xx: int
    redirects: int
    pages_parsed: int
    feeds_found: int
    sitemaps_found: int
    jsonld_sources_found: int
    blockers: Counter[str]
    source_reports: list[SourceReport]
    network_provenance: list[NetworkProvenance]


def _acquire_snapshot(
    *,
    priority_rows: list[dict[str, Any]],
    snapshot_root: Path,
    client: HttpClient | None,
    created_at: datetime | None,
) -> _Acquisition:
    base_client = client if client is not None else BoundedHttpClient()
    http = PoliteDomainClient(base_client, min_delay_seconds=0.0 if client is not None else 0.5)
    domain_requests: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    reports: list[SourceReport] = []
    provenance: list[NetworkProvenance] = []
    confirmed_domains: set[str] = set()
    ambiguous_domains = 0
    requests_total = http_2xx = redirects = pages_parsed = 0
    feeds_found = sitemaps_found = jsonld_sources_found = 0
    candidates_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for row in priority_rows:
        ticker = str(row["ticker"])
        issuer = str(row["issuer"])
        domain, seed_url, evidence = _official_domain(row)
        if domain is None or seed_url is None:
            blockers[LiveBlocker.NO_OFFICIAL_DOMAIN.value] += 1
            reports.append(
                _report(
                    row,
                    source_url=None,
                    source_domain=None,
                    evidence=None,
                    blocker=LiveBlocker.NO_OFFICIAL_DOMAIN.value,
                )
            )
            continue
        confirmed_domains.add(domain)
        robots = _request(http, _robots_url(domain), domain_requests, created_at)
        provenance.append(robots.provenance)
        requests_total += 1
        redirects += robots.result.redirects
        if robots.result.status is not None and 200 <= robots.result.status < 300:
            http_2xx += 1
        robots_blocker = _robots_fetch_blocker(robots.result)
        if robots_blocker is not None:
            blockers[robots_blocker] += 1
            reports.append(
                _report(
                    row,
                    source_url=seed_url,
                    source_domain=domain,
                    evidence=evidence,
                    blocker=robots_blocker,
                )
            )
            continue
        if _robots_disallows(robots.result.body):
            blockers[LiveBlocker.ROBOTS_BLOCKED.value] += 1
            reports.append(
                _report(
                    row,
                    source_url=seed_url,
                    source_domain=domain,
                    evidence=evidence,
                    blocker=LiveBlocker.ROBOTS_BLOCKED.value,
                )
            )
            continue

        pending_urls = _probe_urls(seed_url, domain)
        visited: set[str] = set()
        ticker_candidates: list[dict[str, Any]] = []
        last_blocker: str | None = None
        while pending_urls and len(visited) < MAX_PAGES_PER_SOURCE:
            url = pending_urls.pop(0)
            if domain_requests[domain] >= MAX_REQUESTS_PER_DOMAIN:
                blockers[LiveBlocker.RATE_LIMITED.value] += 1
                last_blocker = LiveBlocker.RATE_LIMITED.value
                break
            canonical_url = _canonical_url(url)
            if canonical_url in visited or urlsplit(canonical_url).netloc.lower() != domain:
                continue
            visited.add(canonical_url)
            retrieval = _request(http, canonical_url, domain_requests, created_at)
            provenance.append(retrieval.provenance)
            requests_total += 1
            redirects += retrieval.result.redirects
            if retrieval.result.status is not None and 200 <= retrieval.result.status < 300:
                http_2xx += 1
            if retrieval.result.blocker is not None:
                blockers[retrieval.result.blocker] += 1
                last_blocker = retrieval.result.blocker
                continue
            content_type = _base_content_type(retrieval.result.content_type)
            if content_type not in SUPPORTED_CONTENT_TYPES:
                blockers[LiveBlocker.UNSUPPORTED_CONTENT_TYPE.value] += 1
                last_blocker = LiveBlocker.UNSUPPORTED_CONTENT_TYPE.value
                continue
            pages_parsed += 1
            parsed = _parse_source(
                retrieval.result,
                ticker=ticker,
                issuer=issuer,
                domain=domain,
                evidence=evidence or "existing source registry",
            )
            feeds_found += parsed.feeds_found
            sitemaps_found += parsed.sitemaps_found
            jsonld_sources_found += parsed.jsonld_sources_found
            new_follow_urls = [
                _canonical_url(follow_url)
                for follow_url in parsed.follow_urls
                if _canonical_url(follow_url) not in visited
            ]
            pending_urls = [
                *new_follow_urls[:MAX_URLS_PER_TICKER],
                *pending_urls,
            ][: MAX_URLS_PER_TICKER * MAX_PAGES_PER_SOURCE]
            for candidate in parsed.candidates:
                key = (str(candidate["source_family"]), str(candidate["source_url"]))
                if key not in candidates_by_key:
                    candidates_by_key[key] = candidate
                    ticker_candidates.append(candidate)
        if ticker_candidates:
            _write_candidates(snapshot_root, ticker, ticker_candidates)
            for candidate in ticker_candidates:
                reports.append(_candidate_report(row, candidate))
        else:
            blocker = last_blocker or LiveBlocker.NO_DISCOVERY_LINKS.value
            blockers[blocker] += 1
            reports.append(
                _report(
                    row,
                    source_url=seed_url,
                    source_domain=domain,
                    evidence=evidence,
                    blocker=blocker,
                )
            )

    return _Acquisition(
        tickers_attempted=len(priority_rows),
        domains_confirmed=len(confirmed_domains),
        domains_ambiguous=ambiguous_domains,
        requests_total=requests_total,
        http_2xx=http_2xx,
        redirects=redirects,
        pages_parsed=pages_parsed,
        feeds_found=feeds_found,
        sitemaps_found=sitemaps_found,
        jsonld_sources_found=jsonld_sources_found,
        blockers=blockers,
        source_reports=reports,
        network_provenance=provenance,
    )


def _v5_priority_rows(
    exact_counts: dict[str, int],
    feature_counts: dict[str, int],
    registry_rows: list[dict[str, Any]],
    universe: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    registry_by_ticker = {str(row["ticker"]): row for row in registry_rows}
    rows: list[dict[str, Any]] = []
    for ticker in sorted(set(exact_counts) | set(universe)):
        exact_count = int(exact_counts.get(ticker, 0))
        feature_count = int(feature_counts.get(ticker, 0))
        registry = registry_by_ticker.get(ticker, {})
        instrument = universe.get(ticker, {})
        tier = priority_tier(
            ticker=ticker,
            exact_count=exact_count,
            feature_ready_count=feature_count,
            in_exact_corpus=ticker in exact_counts,
        )
        rows.append(
            {
                "ticker": ticker,
                "issuer": str(registry.get("issuer") or instrument.get("name") or ticker),
                "exact_event_count": exact_count,
                "feature_ready_count": feature_count,
                "priority_tier": tier,
                "registry": registry,
                "instrument": instrument,
                "existing_source_unknown": int(
                    not bool(registry.get("source_url"))
                    or str(registry.get("timestamp_capability")) == "UNKNOWN"
                ),
            }
        )
    order = {
        "A_ZERO_FEATURE_READY": 0,
        "B_EXACT_1_5": 1,
        "C_EXACT_6_20": 2,
        "D_CANONICAL_TQBR_NOT_IN_EXACT": 3,
        "DEPRIORITIZED": 4,
    }
    return sorted(
        rows,
        key=lambda row: (
            order[str(row["priority_tier"])],
            -int(row["existing_source_unknown"]),
            str(row["ticker"]),
        ),
    )[:MAX_TICKERS]


@dataclass(frozen=True, slots=True)
class _Retrieval:
    result: HttpResult
    provenance: NetworkProvenance


def _request(
    client: HttpClient,
    url: str,
    domain_requests: Counter[str],
    created_at: datetime | None,
) -> _Retrieval:
    domain = urlsplit(url).netloc.lower()
    domain_requests[domain] += 1
    fetched_at = (created_at or datetime.now(UTC)).isoformat()
    result = client.get(url)
    blocker = result.blocker
    content_sha = sha256_bytes(result.body) if result.body else None
    return _Retrieval(
        result=result,
        provenance=NetworkProvenance(
            request_url=url,
            final_url=result.final_url,
            http_status=result.status,
            content_type=result.content_type,
            fetched_at=fetched_at,
            content_sha256=content_sha,
            bytes_received=len(result.body),
            robots_status="CHECKED" if url.endswith("/robots.txt") else "APPLIED",
            policy_status="FAIL_CLOSED" if blocker else "PUBLIC_HTTP_GET",
            blocker=blocker,
        ),
    )


@dataclass(frozen=True, slots=True)
class _ParsedSource:
    candidates: list[dict[str, Any]]
    follow_urls: list[str]
    feeds_found: int
    sitemaps_found: int
    jsonld_sources_found: int


def _parse_source(
    result: HttpResult,
    *,
    ticker: str,
    issuer: str,
    domain: str,
    evidence: str,
) -> _ParsedSource:
    content_type = _base_content_type(result.content_type)
    url = str(result.final_url or result.request_url)
    text = result.body.decode("utf-8", errors="replace")
    if content_type in {
        "application/rss+xml",
        "application/atom+xml",
        "application/xml",
        "text/xml",
    }:
        return _parse_xml(
            text, source_url=url, ticker=ticker, issuer=issuer, domain=domain, evidence=evidence
        )
    if content_type in {"text/html", "application/xhtml+xml"}:
        return _parse_html(
            text, source_url=url, ticker=ticker, issuer=issuer, domain=domain, evidence=evidence
        )
    if content_type in {"application/json", "application/ld+json"}:
        return _parse_json(
            text, source_url=url, ticker=ticker, issuer=issuer, domain=domain, evidence=evidence
        )
    return _ParsedSource([], [], 0, 0, 0)


def _parse_xml(
    text: str,
    *,
    source_url: str,
    ticker: str,
    issuer: str,
    domain: str,
    evidence: str,
) -> _ParsedSource:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return _ParsedSource([], [], 0, 0, 0)
    tag = _local_name(root.tag)
    if tag == "urlset":
        locs = [
            loc.text.strip()
            for loc in root.findall(".//{*}loc")
            if loc.text and _semantic_url(loc.text)
        ]
        return _ParsedSource([], locs[:MAX_URLS_PER_TICKER], 0, 1, 0)
    items: list[dict[str, Any]] = []
    date_only = 0
    exact = 0
    for item in list(root.findall(".//item"))[:MAX_ITEMS_PER_SOURCE]:
        title = item.findtext("title") or ""
        link = item.findtext("link") or source_url
        raw = item.findtext("pubDate") or ""
        exact_ok = _timestamp_exact(raw)
        exact += int(exact_ok)
        date_only += int(bool(raw) and not exact_ok)
        if exact_ok:
            items.append(_item_payload(ticker, title, raw, link, "RSS pubDate"))
    for entry in list(root.findall(".//{http://www.w3.org/2005/Atom}entry"))[:MAX_ITEMS_PER_SOURCE]:
        title = entry.findtext("{http://www.w3.org/2005/Atom}title") or ""
        raw = entry.findtext("{http://www.w3.org/2005/Atom}published") or ""
        link = source_url
        for link_node in entry.findall("{http://www.w3.org/2005/Atom}link"):
            href = link_node.attrib.get("href")
            if href:
                link = urljoin(source_url, href)
                break
        exact_ok = _timestamp_exact(raw)
        exact += int(exact_ok)
        date_only += int(bool(raw) and not exact_ok)
        if exact_ok:
            items.append(_item_payload(ticker, title, raw, link, "Atom published"))
    if exact:
        return _ParsedSource(
            [
                _candidate(
                    ticker, issuer, domain, source_url, "OFFICIAL_FEED", "EXACT", items, evidence
                )
            ],
            [],
            1,
            0,
            0,
        )
    if date_only:
        return _ParsedSource(
            [
                _candidate(
                    ticker, issuer, domain, source_url, "OFFICIAL_FEED", "DATE_ONLY", [], evidence
                )
            ],
            [],
            1,
            0,
            0,
        )
    return _ParsedSource([], [], 1 if tag in {"rss", "feed"} else 0, 0, 0)


def _parse_html(
    text: str,
    *,
    source_url: str,
    ticker: str,
    issuer: str,
    domain: str,
    evidence: str,
) -> _ParsedSource:
    parser = _DiscoveryHtmlParser()
    parser.feed(text)
    follow_urls = [
        urljoin(source_url, href)
        for href in [*parser.alternate_links, *parser.semantic_links]
        if _same_domain_or_relative(source_url, href)
    ][:MAX_URLS_PER_TICKER]
    candidates: list[dict[str, Any]] = []
    jsonld_items: list[dict[str, Any]] = []
    date_only = 0
    for raw_json in parser.jsonld_blocks:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        for item in _iter_jsonld(payload):
            raw = str(item.get("datePublished") or "")
            title = str(item.get("headline") or item.get("name") or "")
            link = str(item.get("url") or source_url)
            if _timestamp_exact(raw):
                jsonld_items.append(
                    _item_payload(
                        ticker, title, raw, urljoin(source_url, link), "JSON-LD datePublished"
                    )
                )
            elif raw:
                date_only += 1
    if jsonld_items:
        candidates.append(
            _candidate(
                ticker,
                issuer,
                domain,
                source_url,
                "ISSUER_HTML_JSONLD",
                "EXACT",
                jsonld_items,
                evidence,
            )
        )
    elif date_only:
        candidates.append(
            _candidate(
                ticker, issuer, domain, source_url, "ISSUER_HTML_JSONLD", "DATE_ONLY", [], evidence
            )
        )
    return _ParsedSource(
        candidates,
        follow_urls,
        sum(1 for link in parser.alternate_links if _feed_like(link)),
        0,
        1 if candidates else 0,
    )


def _parse_json(
    text: str,
    *,
    source_url: str,
    ticker: str,
    issuer: str,
    domain: str,
    evidence: str,
) -> _ParsedSource:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _ParsedSource([], [], 0, 0, 0)
    items: list[dict[str, Any]] = []
    date_only = 0
    for item in _iter_items(payload):
        raw = str(
            item.get("published_at") or item.get("datePublished") or item.get("published") or ""
        )
        title = str(item.get("title") or item.get("headline") or item.get("name") or "")
        link = str(item.get("canonical_url") or item.get("url") or source_url)
        if _timestamp_exact(raw):
            items.append(
                _item_payload(
                    ticker, title, raw, urljoin(source_url, link), "official JSON published_at"
                )
            )
        elif raw:
            date_only += 1
    if items:
        return _ParsedSource(
            [
                _candidate(
                    ticker,
                    issuer,
                    domain,
                    source_url,
                    "ISSUER_OFFICIAL_JSON",
                    "EXACT",
                    items,
                    evidence,
                )
            ],
            [],
            0,
            0,
            0,
        )
    if date_only:
        return _ParsedSource(
            [
                _candidate(
                    ticker,
                    issuer,
                    domain,
                    source_url,
                    "ISSUER_OFFICIAL_JSON",
                    "DATE_ONLY",
                    [],
                    evidence,
                )
            ],
            [],
            0,
            0,
            0,
        )
    return _ParsedSource([], [], 0, 0, 0)


class _DiscoveryHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.alternate_links: list[str] = []
        self.semantic_links: list[str] = []
        self.jsonld_blocks: list[str] = []
        self._in_jsonld = False
        self._jsonld_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        if tag == "link" and values.get("rel", "").lower() == "alternate":
            kind = values.get("type", "").lower()
            if kind in {"application/rss+xml", "application/atom+xml"} and values.get("href"):
                self.alternate_links.append(values["href"])
        if tag == "a" and values.get("href") and _semantic_url(values["href"]):
            self.semantic_links.append(values["href"])
        if tag == "script" and values.get("type", "").lower() == "application/ld+json":
            self._in_jsonld = True
            self._jsonld_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_jsonld:
            self._jsonld_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_jsonld:
            self.jsonld_blocks.append("".join(self._jsonld_parts))
            self._in_jsonld = False


def _candidate(
    ticker: str,
    issuer: str,
    domain: str,
    source_url: str,
    source_type: str,
    timestamp_capability: str,
    items: list[dict[str, Any]],
    evidence: str,
) -> dict[str, Any]:
    source_family = f"{ticker}_{source_type}_LIVE_V1"
    return {
        "ticker": ticker,
        "issuer": issuer,
        "source_url": _canonical_url(source_url),
        "source_domain": domain,
        "source_type": source_type,
        "source_family": source_family,
        "official_source_confirmed": True,
        "official_identity_evidence": evidence,
        "timestamp_capability": timestamp_capability,
        "timestamp_field": _timestamp_field(source_type),
        "timezone_provenance": "EXPLICIT_SOURCE_OFFSET"
        if timestamp_capability == "EXACT"
        else None,
        "archive_capability": bool(items),
        "discovery_method": "bounded live official static source discovery",
        "policy_status": "OFFICIAL_PUBLIC_ZERO_COST_VERIFIED",
        "technical_status": "SOURCE_READY",
        "public_access": True,
        "auth_required": False,
        "captcha_required": False,
        "payment_required": False,
        "items": items[:MAX_ITEMS_PER_SOURCE],
    }


def _item_payload(
    ticker: str, title: str, raw: str, canonical_url: str, timestamp_field: str
) -> dict[str, str]:
    return {
        "ticker": ticker,
        "title": " ".join(title.split()) or "Official source item",
        "published_at": raw,
        "canonical_url": _canonical_url(canonical_url),
        "source_item_id": _canonical_url(canonical_url),
        "timestamp_field": timestamp_field,
    }


def _write_candidates(snapshot_root: Path, ticker: str, candidates: list[dict[str, Any]]) -> None:
    ticker_root = snapshot_root / ticker
    ticker_root.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(candidates):
        _write_json(ticker_root / f"candidate-{index:03d}.json", candidate)


def _candidate_report(row: dict[str, Any], candidate: dict[str, Any]) -> SourceReport:
    items = cast("list[dict[str, Any]]", candidate.get("items") or [])
    exact = len(items) if candidate.get("timestamp_capability") == "EXACT" else 0
    date_only = 1 if candidate.get("timestamp_capability") == "DATE_ONLY" else 0
    return SourceReport(
        ticker=str(row["ticker"]),
        issuer=str(row["issuer"]),
        source_url=str(candidate["source_url"]),
        source_domain=str(candidate["source_domain"]),
        official_source_confirmed=True,
        official_identity_evidence=str(candidate.get("official_identity_evidence") or ""),
        source_type=str(candidate["source_type"]),
        source_family=str(candidate["source_family"]),
        discovery_method=str(candidate["discovery_method"]),
        timestamp_capability=str(candidate["timestamp_capability"]),
        timestamp_field=cast("str | None", candidate.get("timestamp_field")),
        timezone_provenance=cast("str | None", candidate.get("timezone_provenance")),
        archive_capability=bool(candidate.get("archive_capability")),
        items_discovered=len(items) or date_only,
        exact_items_discovered=exact,
        date_only_items_discovered=date_only,
        policy_status=str(candidate["policy_status"]),
        technical_status=str(candidate["technical_status"]),
        blocker=None if exact else LiveBlocker.DATE_ONLY_SOURCE.value,
    )


def _report(
    row: dict[str, Any],
    *,
    source_url: str | None,
    source_domain: str | None,
    evidence: str | None,
    blocker: str,
) -> SourceReport:
    return SourceReport(
        ticker=str(row["ticker"]),
        issuer=str(row["issuer"]),
        source_url=source_url,
        source_domain=source_domain,
        official_source_confirmed=source_domain is not None,
        official_identity_evidence=evidence,
        source_type=None,
        source_family=None,
        discovery_method="bounded live official source discovery",
        timestamp_capability="UNKNOWN",
        timestamp_field=None,
        timezone_provenance=None,
        archive_capability=False,
        items_discovered=0,
        exact_items_discovered=0,
        date_only_items_discovered=0,
        policy_status="FAIL_CLOSED" if blocker else "OFFICIAL_PUBLIC_ZERO_COST_VERIFIED",
        technical_status=blocker,
        blocker=blocker,
    )


def _official_domain(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    registry = cast("dict[str, Any]", row.get("registry") or {})
    source_url = _optional_string(registry.get("source_url"))
    official_domain = _optional_string(registry.get("official_domain")) or _optional_string(
        registry.get("source_domain")
    )
    if source_url:
        parsed = urlsplit(source_url)
        if parsed.scheme == "https" and parsed.netloc:
            return parsed.netloc.lower(), _canonical_url(source_url), "existing source registry URL"
    if official_domain:
        domain = official_domain.lower()
        return domain, f"https://{domain}/", "existing source registry official domain"
    return None, None, None


def _probe_urls(seed_url: str, domain: str) -> list[str]:
    root = f"https://{domain}/"
    urls = [seed_url, root]
    urls.extend(urljoin(root, path) for path in STANDARD_PATH_PROBES)
    return list(
        dict.fromkeys(_canonical_url(url) for url in urls if urlsplit(url).scheme == "https")
    )


def _robots_url(domain: str) -> str:
    return f"https://{domain}/robots.txt"


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
            value = line.split(":", 1)[1].strip()
            if value == "/":
                return True
    return False


def _robots_fetch_blocker(result: HttpResult) -> str | None:
    if result.blocker is None:
        return None
    if result.blocker == LiveBlocker.HTTP_4XX.value and result.status == 404:
        return None
    return result.blocker


def _timestamp_exact(value: str) -> bool:
    try:
        parse_exact_timestamp(value)
    except ValueError:
        return False
    return True


def _timestamp_field(source_type: str) -> str:
    if source_type == "OFFICIAL_FEED":
        return "RSS pubDate / Atom published"
    if source_type == "ISSUER_HTML_JSONLD":
        return "JSON-LD datePublished"
    return "official JSON published_at"


def _semantic_url(value: str) -> bool:
    lowered = value.casefold()
    return any(
        token in lowered
        for token in (
            "news",
            "press",
            "media",
            "investor",
            "investors",
            "ir",
            "disclosure",
            "release",
            "rss",
            "feed",
            "sitemap",
        )
    )


def _feed_like(value: str) -> bool:
    lowered = value.casefold()
    return "rss" in lowered or "feed" in lowered or "atom" in lowered


def _same_domain_or_relative(base_url: str, value: str) -> bool:
    parsed = urlsplit(value)
    return not parsed.netloc or parsed.netloc.lower() == urlsplit(base_url).netloc.lower()


def _iter_jsonld(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        typed = cast("dict[str, Any]", payload)
        graph = typed.get("@graph")
        rows: list[dict[str, Any]] = [typed]
        if isinstance(graph, list):
            for item_obj in cast("list[Any]", graph):
                if isinstance(item_obj, dict):
                    rows.append(cast("dict[str, Any]", item_obj))
        return rows
    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for item_obj in cast("list[Any]", payload):
            if isinstance(item_obj, dict):
                rows.append(cast("dict[str, Any]", item_obj))
        return rows
    return []


def _iter_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for item_obj in cast("list[Any]", payload):
            if isinstance(item_obj, dict):
                rows.append(cast("dict[str, Any]", item_obj))
        return rows
    if isinstance(payload, dict):
        typed = cast("dict[str, Any]", payload)
        for key in ("items", "data", "results", "news"):
            value = typed.get(key)
            if isinstance(value, list):
                rows: list[dict[str, Any]] = []
                for item_obj in cast("list[Any]", value):
                    if isinstance(item_obj, dict):
                        rows.append(cast("dict[str, Any]", item_obj))
                return rows
        return [typed]
    return []


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _base_content_type(value: str | None) -> str:
    return str(value or "").split(";", 1)[0].strip().lower()


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def _read_snapshot_candidates(snapshot_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(snapshot_root.glob("*/*.json")):
        payload = _read_json(path)
        payload.pop("fetched_at", None)
        rows.append(payload)
    return rows


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


def _live_discovery_blocker(acquisition: _Acquisition) -> str | None:
    if acquisition.requests_total == 0:
        return LiveBlocker.NO_OFFICIAL_DOMAIN.value
    if acquisition.http_2xx == 0 and acquisition.blockers[LiveBlocker.DNS_FAILED.value]:
        return LiveBlocker.NETWORK_UNAVAILABLE.value
    return None


def _conclusion(acquisition: _Acquisition, downstream_manifest: dict[str, Any]) -> str:
    if downstream_manifest[
        "V5_NEW_EXACT_EVENTS"
        if "V5_NEW_EXACT_EVENTS" in downstream_manifest
        else "NEW_EXACT_EVENTS"
    ]:
        return "MARKET_MATURATION"
    if downstream_manifest["NEW_EXACT_CAPABLE_SOURCES"]:
        return "DEEPEN_NEW_EXACT_SOURCES"
    if acquisition.requests_total == 0:
        return "MORE_OFFICIAL_DOMAIN_DISCOVERY"
    if acquisition.http_2xx == 0:
        return "LIVE_COLLECTION_ONLY"
    return "MORE_OFFICIAL_DOMAIN_DISCOVERY"


def _artifact_sha(manifest: dict[str, Any]) -> str:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"ARTIFACT_SHA", "created_at", "git_sha", "NETWORK_PROVENANCE"}
    }
    return sha256_payload(core)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


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
        "Bounded live official source snapshot acquisition.",
        "",
        f"- INPUT_DATASET_SHA={manifest['INPUT_DATASET_SHA']}",
        f"- OUTPUT_DATASET_SHA={manifest['OUTPUT_DATASET_SHA']}",
        f"- LIVE_DISCOVERY_EXECUTED={manifest['LIVE_DISCOVERY_EXECUTED']}",
        f"- LIVE_DISCOVERY_BLOCKER={manifest['LIVE_DISCOVERY_BLOCKER']}",
        f"- LIVE_TICKERS_ATTEMPTED={manifest['LIVE_TICKERS_ATTEMPTED']}",
        f"- LIVE_CANDIDATES_WRITTEN={manifest['LIVE_CANDIDATES_WRITTEN']}",
        f"- V5_NEW_EXACT_CAPABLE_SOURCES={manifest['V5_NEW_EXACT_CAPABLE_SOURCES']}",
        f"- V5_NEW_EXACT_EVENTS={manifest['V5_NEW_EXACT_EVENTS']}",
        f"- LIVE_SOURCE_DISCOVERY_CONCLUSION={manifest['LIVE_SOURCE_DISCOVERY_CONCLUSION']}",
        "",
        "No model, TEST outcome use, future outcome observation, T-Invest, market maturation, "
        "backtest, paper trading, orders, or BUY/SELL output was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
