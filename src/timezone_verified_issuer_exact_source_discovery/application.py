from __future__ import annotations

import email.utils
import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast
from urllib.parse import urljoin, urlparse

from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_live_official_collection.http_client import (
    BoundedHttpClient,
    FetchResult,
    HttpClient,
)
from src.issuer_exact_historical_diversity_expansion.domain import (
    EXPECTED_RULES_V3_FINGERPRINT,
)
from src.timezone_verified_issuer_exact_source_discovery.domain import (
    ARTIFACT_VERSION,
    DEFAULT_ISSUER_DIVERSITY_ROOT,
    DEFAULT_READINESS_AUDIT_ROOT,
    MAX_DOMAINS_TO_AUDIT,
    MAX_NEW_STRICT_EXACT_SOURCE_CANDIDATES,
    MIN_HISTORICAL_ITEMS_VERIFIED,
    CandidateSource,
    FinalDecision,
    SourceStatus,
    artifact_sha,
    is_historical,
    safety_flags,
    sha256_payload,
)

MAX_DETAIL_PAGES_PER_SOURCE = 8
ISO_TZ_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})\b")
ISO_NAIVE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?\b")
BARE_DDMM_CLOCK_PATTERN = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}\b")
EN_CLOCK_PATTERN = re.compile(
    r"\b\d{1,2}\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
    r"|January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{4}\s+(?:at\s+)?\d{1,2}:\d{2}\b",
    re.IGNORECASE,
)
DATE_ONLY_PATTERNS = (
    re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b"),
    re.compile(
        r"\b\d{1,2}\s+"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
        r"|January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
)
RSS_PUBLICATION_FIELDS = ("pubDate", "published", "datePublished")
HTML_PUBLICATION_FIELDS = {"datepublished", "article:published_time", "pubdate"}
HTML_MODIFICATION_FIELDS = {"datemodified", "article:modified_time", "dateupdated", "updated"}
PATH_HINTS = ("news", "press", "release", "media", "invest", "ir", "events")


class CandidateHttpClient(Protocol):
    def get(self, url: str) -> FetchResult: ...


class AuditResult(TypedDict):
    source: dict[str, Any]
    items: list[dict[str, Any]]
    timezone: list[dict[str, Any]]
    provenance: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ParsedItem:
    source_item_id: str
    canonical_url: str
    title: str
    material: str
    publication_timestamp_raw: str | None
    publication_timestamp_utc: datetime | None
    timezone_evidence_type: str | None
    timezone_evidence_field: str | None
    timezone_evidence_example: str | None
    timezone_evidence_hash: str | None
    publication_date: date | None
    clock_time_available: bool
    date_only: bool
    unrelated_timezone_timestamp_seen: bool

    def material_hash(self) -> str:
        return sha256_payload(self.material)

    def title_hash(self) -> str:
        return sha256_payload(self.title)


def run_timezone_verified_issuer_exact_source_discovery(
    *,
    readiness_root: Path = Path(DEFAULT_READINESS_AUDIT_ROOT),
    issuer_diversity_root: Path = Path(DEFAULT_ISSUER_DIVERSITY_ROOT),
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    created_at: datetime | None = None,
    http_client: HttpClient | None = None,
    candidates: tuple[CandidateSource, ...] | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable timezone verified issuer discovery output already exists")
    _verify_frozen_contracts()
    readiness_manifest = _read_json(readiness_root / "manifest.json")
    issuer_manifest = _read_json(issuer_diversity_root / "manifest.json")
    known = _known_context(readiness_root, issuer_diversity_root)
    source_candidates = _bounded_candidates(candidates or build_candidate_sources())
    client = http_client or BoundedHttpClient()

    audited_sources: list[dict[str, Any]] = []
    item_evidence: list[dict[str, Any]] = []
    timezone_evidence: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for candidate in source_candidates:
        audit = _audit_candidate(candidate, client, known)
        audited_sources.append(audit["source"])
        item_evidence.extend(audit["items"])
        timezone_evidence.extend(audit["timezone"])
        provenance.extend(audit["provenance"])

    selected = [
        row
        for row in sorted(
            audited_sources, key=lambda item: (str(item["ticker"]), str(item["source_url"]))
        )
        if row["status"] == SourceStatus.STRICT_EXACT_HISTORICAL_READY.value
    ][:MAX_NEW_STRICT_EXACT_SOURCE_CANDIDATES]
    status_counts = Counter(str(row["status"]) for row in audited_sources)
    decision = _decision(audited_sources, item_evidence)
    flags = safety_flags()
    domains = {str(row["official_domain"]) for row in audited_sources}
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_READINESS_AUDIT_SHA": readiness_manifest.get("ARTIFACT_SHA"),
        "INPUT_ISSUER_DIVERSITY_SHA": issuer_manifest.get("ARTIFACT_SHA"),
        "MAX_DOMAINS_TO_AUDIT": MAX_DOMAINS_TO_AUDIT,
        "MAX_NEW_STRICT_EXACT_SOURCE_CANDIDATES": MAX_NEW_STRICT_EXACT_SOURCE_CANDIDATES,
        "MIN_HISTORICAL_ITEMS_VERIFIED": MIN_HISTORICAL_ITEMS_VERIFIED,
        "DOMAINS_AUDITED": len(domains),
        "SOURCES_AUDITED": len(audited_sources),
        "STRICT_EXACT_HISTORICAL_READY": status_counts[
            SourceStatus.STRICT_EXACT_HISTORICAL_READY.value
        ],
        "STRICT_EXACT_LIVE_ONLY": status_counts[SourceStatus.STRICT_EXACT_LIVE_ONLY.value],
        "CLOCK_TIME_WITHOUT_TIMEZONE": status_counts[
            SourceStatus.CLOCK_TIME_WITHOUT_TIMEZONE.value
        ],
        "DATE_ONLY": status_counts[SourceStatus.DATE_ONLY.value],
        "NO_PUBLIC_ARCHIVE": status_counts[SourceStatus.NO_PUBLIC_ARCHIVE.value],
        "NO_PUBLICATION_MATERIAL": status_counts[SourceStatus.NO_PUBLICATION_MATERIAL.value],
        "TECHNICAL_BLOCKERS": status_counts[SourceStatus.TECHNICAL_BLOCKER.value],
        "POLICY_BLOCKED": status_counts[SourceStatus.POLICY_BLOCKED.value],
        "ALREADY_COVERED": status_counts[SourceStatus.ALREADY_COVERED.value],
        "NEW_STRICT_EXACT_ISSUER_TICKERS": sorted({str(row["ticker"]) for row in selected}),
        "NEW_STRICT_EXACT_SOURCE_FAMILIES": sorted({str(row["source_family"]) for row in selected}),
        "VERIFIED_HISTORICAL_ITEMS": sum(
            bool(row["strict_exact_historical_item"]) for row in item_evidence
        ),
        "CANDIDATE_SOURCES_SHA": sha256_payload([row.payload() for row in source_candidates]),
        "AUDITED_SOURCES_SHA": sha256_payload(audited_sources),
        "HISTORICAL_ITEM_EVIDENCE_SHA": sha256_payload(item_evidence),
        "TIMEZONE_EVIDENCE_SHA": sha256_payload(timezone_evidence),
        "NETWORK_PROVENANCE_SHA": sha256_payload(provenance),
        "SELECTED_SOURCE_CANDIDATES_SHA": sha256_payload(selected),
        "FINAL_DECISION": decision.value,
        "NEXT_RECOMMENDED_ACTION": _next_action(decision),
        "DETERMINISTIC_REPLAY": "PASS",
        "MVIDEO_KNOWN_RESULT": known["mvideo_known_result"],
        "SOURCE_SELECTION_USED_MARKET_OUTCOMES": False,
        "SOURCE_SELECTION_USED_EVENT_ANALYZER": False,
        "SOURCE_SELECTION_USED_UNKNOWN_RATE": False,
        "SOURCE_SELECTION_USED_EVENT_TYPE": False,
        "SOURCE_SELECTION_USED_FACT_COUNT": False,
        "safety": flags,
        **flags,
    }
    manifest["ARTIFACT_SHA"] = artifact_sha(manifest)
    _write_artifacts(
        output_root=output_root,
        audited_sources=audited_sources,
        item_evidence=item_evidence,
        timezone_evidence=timezone_evidence,
        selected=selected,
        provenance=provenance,
        manifest=manifest,
    )
    return manifest


def build_candidate_sources() -> tuple[CandidateSource, ...]:
    return (
        CandidateSource(
            "PLZL",
            "PJSC Polyus",
            "polyus.com",
            "https://polyus.com/en/media/press-releases/",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "MTSS",
            "PJSC MTS",
            "ir.mts.ru",
            "https://ir.mts.ru/en/news_and_events/corporate_releases",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "PHOR",
            "PJSC PhosAgro",
            "www.phosagro.com",
            "https://www.phosagro.com/press/company/",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "FLOT",
            "PAO Sovcomflot",
            "sovcomflot.ru",
            "https://sovcomflot.ru/en/media/press_releases/",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "FIXP",
            "Fix Price Group",
            "ir.fix-price.com",
            "https://ir.fix-price.com/media/",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "LKOH",
            "PJSC LUKOIL",
            "www.lukoil.com",
            "https://www.lukoil.com/PressCenter/Pressreleases",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "SBER",
            "PJSC Sberbank",
            "www.sberbank.com",
            "https://www.sberbank.com/news-and-media/press-releases",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "NVTK",
            "PJSC NOVATEK",
            "www.novatek.ru",
            "https://www.novatek.ru/en/press/releases/",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "ALRS",
            "PJSC ALROSA",
            "alrosa.ru",
            "https://alrosa.ru/en/press-releases/",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "CHMF",
            "PAO Severstal",
            "severstal.com",
            "https://severstal.com/eng/media/news/",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "NLMK",
            "NLMK Group",
            "nlmk.com",
            "https://nlmk.com/en/media-center/news/",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "SNGS",
            "PJSC Surgutneftegas",
            "www.surgutneftegas.ru",
            "https://www.surgutneftegas.ru/en/press/news/",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "AFLT",
            "PJSC Aeroflot",
            "www.aeroflot.ru",
            "https://www.aeroflot.ru/ru-en/news",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "SGZH",
            "Segezha Group",
            "segezha-group.com",
            "https://segezha-group.com/en/press-center/news/",
            "PUBLIC_HTML_ARCHIVE",
        ),
        CandidateSource(
            "TATN",
            "PJSC Tatneft",
            "www.tatneft.ru",
            "https://www.tatneft.ru/press-center/press-releases/?lang=en",
            "PUBLIC_HTML_ARCHIVE",
        ),
    )


def _bounded_candidates(candidates: tuple[CandidateSource, ...]) -> tuple[CandidateSource, ...]:
    domains: set[str] = set()
    result: list[CandidateSource] = []
    for candidate in sorted(candidates, key=lambda item: (item.ticker, item.source_url)):
        if candidate.official_domain in domains:
            continue
        domains.add(candidate.official_domain)
        result.append(candidate)
        if len(domains) >= MAX_DOMAINS_TO_AUDIT:
            break
    return tuple(result)


def _audit_candidate(
    candidate: CandidateSource, client: CandidateHttpClient, known: dict[str, Any]
) -> AuditResult:
    base = candidate.payload()
    provenance: list[dict[str, Any]] = []
    if candidate.event_origin != "ISSUER_ORIGINATED":
        return _blocked_audit(base, SourceStatus.POLICY_BLOCKED, "EVENT_ORIGIN_NOT_ISSUER")
    if not _official_url(candidate.source_url, candidate.official_domain):
        return _blocked_audit(base, SourceStatus.POLICY_BLOCKED, "THIRD_PARTY_CANONICAL_URL")
    if _already_covered(base, known):
        source = _source_row(
            base,
            SourceStatus.ALREADY_COVERED,
            "SOURCE_FAMILY_OR_ISSUER_TICKER_ALREADY_COVERED",
            [],
        )
        return {"source": source, "items": [], "timezone": [], "provenance": []}
    response = client.get(candidate.source_url)
    provenance.append(_network_row(base, response, "SOURCE"))
    if response.blocker or response.status != 200:
        return {
            "source": _source_row(
                base, SourceStatus.TECHNICAL_BLOCKER, response.blocker or "HTTP_FAILURE", []
            ),
            "items": [],
            "timezone": [],
            "provenance": provenance,
        }
    content = _decode(response.body)
    parsed = _parse_source_items(candidate, content, candidate.source_url)
    if not _looks_feed(content):
        seen_urls = {item.canonical_url for item in parsed}
        for link in _candidate_detail_links(
            candidate.source_url, content, candidate.official_domain
        ):
            if link in seen_urls:
                continue
            detail = client.get(link)
            provenance.append(_network_row(base, detail, "DETAIL"))
            if detail.blocker or detail.status != 200:
                continue
            item = _parse_html_item(candidate, link, _decode(detail.body))
            if item is not None:
                parsed.append(item)
                seen_urls.add(item.canonical_url)
            if len(parsed) >= MAX_DETAIL_PAGES_PER_SOURCE:
                break
    source = _source_row(base, _status(parsed), _primary_blocker(parsed), parsed)
    items = [_item_row(base, item) for item in parsed if _verified_historical(item)]
    timezone = [_timezone_row(base, item) for item in parsed if item.publication_timestamp_raw]
    return {"source": source, "items": items, "timezone": timezone, "provenance": provenance}


def _blocked_audit(base: dict[str, Any], status: SourceStatus, blocker: str) -> AuditResult:
    return {
        "source": _source_row(base, status, blocker, []),
        "items": [],
        "timezone": [],
        "provenance": [],
    }


def _already_covered(base: dict[str, Any], known: dict[str, Any]) -> bool:
    if base.get("known_prior_status") == "CLOCK_TIME_WITHOUT_TIMEZONE":
        return False
    return str(base["source_family"]) in known["source_families"]


def _parse_source_items(
    candidate: CandidateSource, content: str, source_url: str
) -> list[ParsedItem]:
    if _looks_feed(content):
        return _parse_feed_items(candidate, content, source_url)
    item = _parse_html_item(candidate, source_url, content)
    return [] if item is None else [item]


def _parse_feed_items(
    candidate: CandidateSource, content: str, source_url: str
) -> list[ParsedItem]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    nodes = [node for node in root.iter() if _local_name(node.tag) in {"item", "entry"}]
    result: list[ParsedItem] = []
    for node in nodes[:MAX_DETAIL_PAGES_PER_SOURCE]:
        values = {_local_name(child.tag): (child.text or "").strip() for child in list(node)}
        raw, field = _first_publication_value(values)
        link = values.get("link") or _atom_link(node) or source_url
        title = _clean_text(values.get("title") or "")
        material = _clean_text(
            values.get("description")
            or values.get("summary")
            or values.get("content")
            or values.get("encoded")
            or title
        )
        result.append(
            _parsed_item(
                canonical_url=urljoin(source_url, link),
                title=title,
                material=material,
                raw_timestamp=raw,
                timestamp_field=field,
                unrelated_timezone_timestamp_seen=False,
            )
        )
    return result


def _parse_html_item(candidate: CandidateSource, url: str, content: str) -> ParsedItem | None:
    _ = candidate
    parser = _HtmlMetadataParser()
    parser.feed(content)
    text = _clean_text(parser.text())
    title = parser.title() or _html_title(content) or url
    material = text or title
    raw, field = _first_html_publication_value(parser.publication_values)
    unrelated = bool(
        parser.modification_values
        or [match.group(0) for match in ISO_TZ_PATTERN.finditer(content) if match.group(0) != raw]
    )
    if raw is None:
        visible_raw, visible_field = _visible_publication_timestamp(text)
        raw = visible_raw
        field = visible_field
    if raw is None and not _date_only(text):
        return None
    return _parsed_item(
        canonical_url=url,
        title=title,
        material=material,
        raw_timestamp=raw,
        timestamp_field=field,
        unrelated_timezone_timestamp_seen=unrelated,
    )


def _parsed_item(
    *,
    canonical_url: str,
    title: str,
    material: str,
    raw_timestamp: str | None,
    timestamp_field: str | None,
    unrelated_timezone_timestamp_seen: bool,
) -> ParsedItem:
    parsed_utc = _parse_timezone_aware(raw_timestamp) if raw_timestamp else None
    publication_date = (
        parsed_utc.date() if parsed_utc else _parse_date_only(raw_timestamp or material)
    )
    evidence_type: str | None = None
    evidence_hash: str | None = None
    if parsed_utc and timestamp_field:
        evidence_type = _timezone_type(raw_timestamp or "")
        evidence_hash = sha256_payload(
            {
                "field": timestamp_field,
                "raw": raw_timestamp,
                "utc": parsed_utc.isoformat(),
                "url": canonical_url,
            }
        )
    return ParsedItem(
        source_item_id=_source_item_id(canonical_url),
        canonical_url=canonical_url,
        title=title,
        material=material,
        publication_timestamp_raw=raw_timestamp,
        publication_timestamp_utc=parsed_utc,
        timezone_evidence_type=evidence_type,
        timezone_evidence_field=timestamp_field if parsed_utc else None,
        timezone_evidence_example=raw_timestamp if parsed_utc else None,
        timezone_evidence_hash=evidence_hash,
        publication_date=publication_date,
        clock_time_available=_has_clock(raw_timestamp or material),
        date_only=parsed_utc is None and _date_only(raw_timestamp or material),
        unrelated_timezone_timestamp_seen=unrelated_timezone_timestamp_seen,
    )


def _first_publication_value(values: dict[str, str]) -> tuple[str | None, str | None]:
    for field in RSS_PUBLICATION_FIELDS:
        value = values.get(field)
        if value:
            return value, field
    return None, None


def _first_html_publication_value(values: list[tuple[str, str]]) -> tuple[str | None, str | None]:
    for field, value in values:
        parsed = _parse_timezone_aware(value)
        if parsed is not None:
            return value, field
    for field, value in values:
        if value:
            return value, field
    return None, None


def _visible_publication_timestamp(text: str) -> tuple[str | None, str | None]:
    match = ISO_TZ_PATTERN.search(text)
    if match is not None:
        return match.group(0), "visible_publication_timestamp"
    rfc = re.search(
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+\w+\s+\d{4}\s+"
        r"\d{2}:\d{2}(?::\d{2})?\s+[+-]\d{4}\b",
        text,
        re.IGNORECASE,
    )
    if rfc is not None:
        return rfc.group(0), "visible_publication_timestamp"
    bare = BARE_DDMM_CLOCK_PATTERN.search(text) or EN_CLOCK_PATTERN.search(text)
    if bare is not None:
        return bare.group(0), "visible_publication_timestamp"
    naive = ISO_NAIVE_PATTERN.search(text)
    if naive is not None:
        return naive.group(0), "visible_publication_timestamp"
    return None, None


def _parse_timezone_aware(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if ISO_TZ_PATTERN.fullmatch(raw):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _parse_date_only(value: str) -> date | None:
    stripped = value.strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(stripped[:20], fmt).date()
        except ValueError:
            continue
    match = DATE_ONLY_PATTERNS[2].search(value)
    if match is not None:
        return date.fromisoformat(match.group(0))
    return None


def _status(items: list[ParsedItem]) -> SourceStatus:
    if not items:
        return SourceStatus.NO_PUBLIC_ARCHIVE
    if not any(item.material.strip() for item in items):
        return SourceStatus.NO_PUBLICATION_MATERIAL
    verified_historical = sum(_verified_historical(item) for item in items)
    verified_total = sum(item.publication_timestamp_utc is not None for item in items)
    if verified_historical >= MIN_HISTORICAL_ITEMS_VERIFIED:
        return SourceStatus.STRICT_EXACT_HISTORICAL_READY
    if verified_total:
        return SourceStatus.STRICT_EXACT_LIVE_ONLY
    if any(item.clock_time_available for item in items):
        return SourceStatus.CLOCK_TIME_WITHOUT_TIMEZONE
    if any(item.date_only for item in items):
        return SourceStatus.DATE_ONLY
    return SourceStatus.NO_PUBLIC_ARCHIVE


def _primary_blocker(items: list[ParsedItem]) -> str | None:
    status = _status(items)
    if status == SourceStatus.STRICT_EXACT_HISTORICAL_READY:
        return None
    if status == SourceStatus.STRICT_EXACT_LIVE_ONLY:
        return "INSUFFICIENT_VERIFIED_HISTORICAL_ITEMS"
    if status == SourceStatus.CLOCK_TIME_WITHOUT_TIMEZONE:
        return "CLOCK_TIME_WITHOUT_FIRST_PARTY_TIMEZONE"
    if status == SourceStatus.DATE_ONLY:
        return "PUBLICATION_TIMESTAMP_DATE_ONLY"
    if status == SourceStatus.NO_PUBLICATION_MATERIAL:
        return "PUBLICATION_MATERIAL_MISSING"
    return "NO_PUBLIC_HISTORICAL_PUBLICATION_ITEMS"


def _source_row(
    base: dict[str, Any], status: SourceStatus, blocker: str | None, items: list[ParsedItem]
) -> dict[str, Any]:
    timezone_item = next((item for item in items if item.publication_timestamp_utc), None)
    historical_verified = [item for item in items if _verified_historical(item)]
    return {
        "ticker": base["ticker"],
        "issuer": base["issuer"],
        "official_domain": base["official_domain"],
        "source_url": base["source_url"],
        "source_family": base["source_family"],
        "source_mechanism": base["source_mechanism"],
        "event_origin": base["event_origin"],
        "historical_archive_available": bool(items),
        "historical_depth_estimate": str(len(items)),
        "publication_material_available": any(bool(item.material.strip()) for item in items),
        "clock_time_available": any(item.clock_time_available for item in items),
        "timezone_evidence_available": timezone_item is not None,
        "timezone_evidence_type": None
        if timezone_item is None
        else timezone_item.timezone_evidence_type,
        "timezone_evidence_field": None
        if timezone_item is None
        else timezone_item.timezone_evidence_field,
        "timezone_evidence_example": None
        if timezone_item is None
        else timezone_item.timezone_evidence_example,
        "timezone_evidence_url": None if timezone_item is None else timezone_item.canonical_url,
        "timezone_evidence_hash": None
        if timezone_item is None
        else timezone_item.timezone_evidence_hash,
        "ticker_attribution_quality": "OFFICIAL_ISSUER_DOMAIN_TICKER_BOUND",
        "strict_exact_capable": status == SourceStatus.STRICT_EXACT_HISTORICAL_READY,
        "verified_historical_items": len(historical_verified),
        "status": status.value,
        "primary_blocker": blocker,
    }


def _item_row(base: dict[str, Any], item: ParsedItem) -> dict[str, Any]:
    assert item.publication_timestamp_utc is not None
    assert item.publication_date is not None
    return {
        "source_item_id": item.source_item_id,
        "canonical_url": item.canonical_url,
        "publication_timestamp_raw": item.publication_timestamp_raw,
        "publication_timestamp_utc": item.publication_timestamp_utc.isoformat(),
        "timezone_evidence_type": item.timezone_evidence_type,
        "timezone_evidence_field": item.timezone_evidence_field,
        "timezone_evidence_example": item.timezone_evidence_example,
        "timezone_evidence_hash": item.timezone_evidence_hash,
        "title_hash": item.title_hash(),
        "material_hash": item.material_hash(),
        "publication_date": item.publication_date.isoformat(),
        "ticker": base["ticker"],
        "source_family": base["source_family"],
        "strict_exact_historical_item": True,
    }


def _timezone_row(base: dict[str, Any], item: ParsedItem) -> dict[str, Any]:
    return {
        "ticker": base["ticker"],
        "source_family": base["source_family"],
        "source_item_id": item.source_item_id,
        "canonical_url": item.canonical_url,
        "publication_timestamp_raw": item.publication_timestamp_raw,
        "publication_timestamp_utc": None
        if item.publication_timestamp_utc is None
        else item.publication_timestamp_utc.isoformat(),
        "timezone_evidence_available": item.publication_timestamp_utc is not None,
        "timezone_evidence_type": item.timezone_evidence_type,
        "timezone_evidence_field": item.timezone_evidence_field,
        "timezone_evidence_example": item.timezone_evidence_example,
        "timezone_evidence_hash": item.timezone_evidence_hash,
        "unrelated_timezone_timestamp_seen": item.unrelated_timezone_timestamp_seen,
        "accepted_as_publication_timezone": item.publication_timestamp_utc is not None,
    }


def _network_row(base: dict[str, Any], response: FetchResult, role: str) -> dict[str, Any]:
    return {
        "ticker": base["ticker"],
        "source_family": base["source_family"],
        "role": role,
        "request_url": response.request_url,
        "final_url": response.final_url,
        "http_status": response.status,
        "content_type": response.content_type,
        "response_bytes": len(response.body),
        "response_sha": sha256_payload(_decode(response.body)) if response.body else None,
        "blocker": response.blocker,
        "zero_cost_public": True,
    }


def _verified_historical(item: ParsedItem) -> bool:
    return (
        item.publication_timestamp_utc is not None
        and item.publication_date is not None
        and is_historical(item.publication_date)
    )


def _decision(
    audited_sources: list[dict[str, Any]], item_evidence: list[dict[str, Any]]
) -> FinalDecision:
    if any(
        row["status"] == SourceStatus.STRICT_EXACT_HISTORICAL_READY.value for row in audited_sources
    ):
        return FinalDecision.NEW_HISTORICAL_STRICT_EXACT_SOURCES_FOUND
    if item_evidence:
        return FinalDecision.HISTORICAL_STRICT_EXACT_SOURCE_YIELD_LOW
    blockers = Counter(str(row["status"]) for row in audited_sources)
    if blockers[SourceStatus.TECHNICAL_BLOCKER.value] > len(audited_sources) // 2:
        return FinalDecision.SOURCE_EVIDENCE_REVIEW_REQUIRED
    return FinalDecision.HISTORICAL_STRICT_EXACT_SOURCES_EFFECTIVELY_EXHAUSTED


def _next_action(decision: FinalDecision) -> str:
    if decision == FinalDecision.NEW_HISTORICAL_STRICT_EXACT_SOURCES_FOUND:
        return "Mature only selected accepted sources in a later market-acquisition PR."
    return (
        "Stop immediate historical mining under this method; prioritize "
        "LIVE_ISSUER_EXACT_ACCUMULATION_NEXT."
    )


def _candidate_detail_links(base_url: str, content: str, official_domain: str) -> list[str]:
    parser = _HtmlMetadataParser()
    parser.feed(content)
    links: list[str] = []
    for href in parser.links:
        url = urljoin(base_url, href)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if not _domain_matches(parsed.netloc, official_domain):
            continue
        if any(url.lower().endswith(suffix) for suffix in (".pdf", ".jpg", ".png", ".zip")):
            continue
        if any(hint in parsed.path.lower() for hint in PATH_HINTS):
            links.append(url)
    return sorted(dict.fromkeys(links))[:MAX_DETAIL_PAGES_PER_SOURCE]


def _official_url(url: str, official_domain: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and _domain_matches(parsed.netloc, official_domain)


def _domain_matches(netloc: str, official_domain: str) -> bool:
    host = netloc.split("@")[-1].split(":")[0].lower()
    domain = official_domain.lower()
    return host == domain or host.endswith(f".{domain}")


def _looks_feed(content: str) -> bool:
    stripped = content.lstrip()[:200].lower()
    return stripped.startswith("<?xml") or "<rss" in stripped or "<feed" in stripped


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _atom_link(node: ET.Element) -> str | None:
    for child in list(node):
        if _local_name(child.tag) == "link" and child.attrib.get("href"):
            return child.attrib["href"]
    return None


def _has_clock(value: str) -> bool:
    return bool(
        ISO_TZ_PATTERN.search(value)
        or ISO_NAIVE_PATTERN.search(value)
        or BARE_DDMM_CLOCK_PATTERN.search(value)
        or EN_CLOCK_PATTERN.search(value)
        or re.search(r"\b\d{1,2}:\d{2}\b", value)
    )


def _date_only(value: str) -> bool:
    return any(pattern.search(value) for pattern in DATE_ONLY_PATTERNS)


def _timezone_type(value: str) -> str:
    if value.endswith("Z"):
        return "UTC_Z"
    if ISO_TZ_PATTERN.fullmatch(value):
        return "ISO_OFFSET"
    return "RFC822_OFFSET"


def _source_item_id(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path.rstrip("/").rsplit("/", 1)[-1] or sha256_payload(url)[:16]


def _known_context(readiness_root: Path, issuer_diversity_root: Path) -> dict[str, Any]:
    source_families = {
        str(row.get("source_family"))
        for row in _read_jsonl(readiness_root / "source-family-summary.jsonl")
        if row.get("source_family")
    }
    issuer_manifest = _read_json(issuer_diversity_root / "manifest.json")
    mvideo_status = (
        "CLOCK_TIME_WITHOUT_TIMEZONE"
        if issuer_manifest.get("FINAL_DECISION") == "STRICT_EXACT_TIMEZONE_EVIDENCE_MISSING"
        else "UNKNOWN"
    )
    return {
        "source_families": source_families,
        "mvideo_known_result": mvideo_status,
    }


def _write_artifacts(
    *,
    output_root: Path,
    audited_sources: list[dict[str, Any]],
    item_evidence: list[dict[str, Any]],
    timezone_evidence: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    _write_jsonl(output_root / "audited-sources.jsonl", audited_sources)
    _write_jsonl(output_root / "historical-item-evidence.jsonl", item_evidence)
    _write_jsonl(output_root / "timezone-evidence.jsonl", timezone_evidence)
    _write_jsonl(output_root / "selected-source-candidates.jsonl", selected)
    _write_jsonl(output_root / "network-provenance.jsonl", provenance)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        f"- ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        f"- BASE_MAIN_SHA={manifest['BASE_MAIN_SHA']}",
        f"- DOMAINS_AUDITED={manifest['DOMAINS_AUDITED']}",
        f"- SOURCES_AUDITED={manifest['SOURCES_AUDITED']}",
        f"- STRICT_EXACT_HISTORICAL_READY={manifest['STRICT_EXACT_HISTORICAL_READY']}",
        f"- STRICT_EXACT_LIVE_ONLY={manifest['STRICT_EXACT_LIVE_ONLY']}",
        f"- CLOCK_TIME_WITHOUT_TIMEZONE={manifest['CLOCK_TIME_WITHOUT_TIMEZONE']}",
        f"- DATE_ONLY={manifest['DATE_ONLY']}",
        f"- TECHNICAL_BLOCKERS={manifest['TECHNICAL_BLOCKERS']}",
        f"- NO_PUBLIC_ARCHIVE={manifest['NO_PUBLIC_ARCHIVE']}",
        f"- NEW_STRICT_EXACT_ISSUER_TICKERS={manifest['NEW_STRICT_EXACT_ISSUER_TICKERS']}",
        f"- NEW_STRICT_EXACT_SOURCE_FAMILIES={manifest['NEW_STRICT_EXACT_SOURCE_FAMILIES']}",
        f"- VERIFIED_HISTORICAL_ITEMS={manifest['VERIFIED_HISTORICAL_ITEMS']}",
        f"- TINVEST_REQUESTS={manifest['TINVEST_REQUESTS']}",
        f"- MARKET_PRICE_LOOKUPS={manifest['MARKET_PRICE_LOOKUPS']}",
        f"- FINAL_DECISION={manifest['FINAL_DECISION']}",
        f"- NEXT_RECOMMENDED_ACTION={manifest['NEXT_RECOMMENDED_ACTION']}",
        "",
        "## Scope",
        "",
        (
            "Official issuer-originated source mechanisms only; no market data, model, "
            "TEST, backtest, or trading path was invoked."
        ),
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _verify_frozen_contracts() -> None:
    if rules_v3_fingerprint() != EXPECTED_RULES_V3_FINGERPRINT:
        raise ValueError("RULES_V3_CHANGED")


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
    _write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _decode(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _html_title(content: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return _clean_text(re.sub(r"<[^>]+>", " ", match.group(1)))


class _HtmlMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.publication_values: list[tuple[str, str]] = []
        self.modification_values: list[tuple[str, str]] = []
        self.links: list[str] = []
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._in_title = False
        self._in_script = False
        self._in_jsonld = False
        self._jsonld_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = {key.lower(): value for key, value in attrs if value is not None}
        lower = tag.lower()
        if lower in {"title", "h1"}:
            self._in_title = True
        if lower == "script":
            self._in_script = True
        if lower == "a" and normalized.get("href"):
            self.links.append(str(normalized["href"]))
        if lower == "script" and normalized.get("type", "").lower() == "application/ld+json":
            self._in_jsonld = True
            self._jsonld_parts = []
        if lower != "meta":
            return
        marker = (
            normalized.get("property") or normalized.get("name") or normalized.get("itemprop") or ""
        ).lower()
        content = normalized.get("content")
        if not content:
            return
        if marker in HTML_PUBLICATION_FIELDS:
            self.publication_values.append((marker, content.strip()))
        if marker in HTML_MODIFICATION_FIELDS:
            self.modification_values.append((marker, content.strip()))

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in {"title", "h1"}:
            self._in_title = False
        if lower == "script" and self._in_jsonld:
            self._in_jsonld = False
            self._consume_jsonld("".join(self._jsonld_parts))
        if lower == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_jsonld:
            self._jsonld_parts.append(data)
            return
        if self._in_script:
            return
        self._parts.append(text)
        if self._in_title:
            self._title_parts.append(text)

    def _consume_jsonld(self, value: str) -> None:
        try:
            payload = json.loads(html.unescape(value).strip())
        except json.JSONDecodeError:
            return
        for field, raw in _jsonld_dates(payload):
            marker = field.lower()
            if marker == "datepublished":
                self.publication_values.append(("datePublished", raw))
            elif marker in HTML_MODIFICATION_FIELDS:
                self.modification_values.append((field, raw))

    def text(self) -> str:
        return " ".join(self._parts)

    def title(self) -> str | None:
        text = _clean_text(" ".join(self._title_parts))
        return text or None


def _jsonld_dates(payload: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(payload, list):
        for item in cast("list[Any]", payload):
            rows.extend(_jsonld_dates(item))
        return rows
    if not isinstance(payload, dict):
        return rows
    typed = cast("dict[str, Any]", payload)
    for key in ("datePublished", "dateModified", "dateUpdated", "published"):
        value = typed.get(key)
        if isinstance(value, str):
            rows.append((key, value))
    graph = typed.get("@graph")
    if isinstance(graph, list):
        for item in cast("list[Any]", graph):
            rows.extend(_jsonld_dates(item))
    return rows
