from __future__ import annotations

import html
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from src.exact_event_live_official_collection.http_client import (
    BoundedHttpClient,
    FetchResult,
    HttpClient,
)
from src.free_live_issuer_accumulation.domain import (
    live_accumulation_safety_flags,
    parse_publication_timestamp,
    sha256_payload,
)
from src.free_live_issuer_accumulation.operation import (
    build_operation_status,
    verify_operation_seal,
)
from src.free_live_operational_burnin_and_onboarding_v3.application import (
    DEFAULT_INSTRUMENT_MAPPING_PATH,
    distinct_new_target_eligible_legal_issuers,
    diversity_eligibility_payload,
    evaluate_target_eligibility,
    load_instrument_mapping_rows,
)
from src.moex_target_source_discovery_v5.application import (
    DEFAULT_OPERATION_ROOT,
    build_candidate_universe,
    canonical_registry_from_mapping,
    default_v5_source_configs,
    mapping_payload,
)
from src.moex_target_source_discovery_v5.application import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_V5_ROOT,
)

ARTIFACT_VERSION = "moex-issuer-controlled-channel-discovery-v6"
SOURCE_CLASS = "ISSUER_CONTROLLED_PLATFORM_HOSTED_PUBLIC_SOURCE"
DEFAULT_OUTPUT_ROOT = Path(f"artifacts/{ARTIFACT_VERSION}")
LIVE_EPOCH_START = datetime(2026, 8, 11, tzinfo=UTC)
MAX_CHANNELS_PER_ISSUER = 2
MAX_POSTS_PER_CHANNEL = 3
TELEGRAM_LINK_RE = re.compile(
    r"https?://t\.me/(?:s/|\+)?(?P<channel>[A-Za-z0-9_]{4,})", re.IGNORECASE
)
VK_LINK_RE = re.compile(r"https?://vk\.com/(?P<page>[A-Za-z0-9_.-]{4,})", re.IGNORECASE)
TG_TIME_RE = re.compile(r"<time\b[^>]*datetime=[\"'](?P<datetime>[^\"']+)[\"']", re.I)
TG_POST_RE = re.compile(r"data-post=[\"'](?P<post>[A-Za-z0-9_]+/\d+)[\"']", re.I)
RELATIVE_TIME_RE = re.compile(r"\b(today|yesterday|сегодня|вчера)\b", re.I)


@dataclass(frozen=True, slots=True)
class ChannelCandidate:
    ticker: str
    legal_issuer: str
    official_url: str
    official_domain: str
    instrument: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OwnershipProof:
    ticker: str
    legal_issuer: str
    platform: str
    channel_id: str
    channel_url: str
    official_url: str
    official_domain: str
    proof_level: str
    proof_ready: bool
    blocker: str | None

    def payload(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "legal_issuer": self.legal_issuer,
            "platform": self.platform,
            "channel_id": self.channel_id,
            "channel_url": self.channel_url,
            "official_url": self.official_url,
            "official_domain": self.official_domain,
            "proof_level": self.proof_level,
            "ISSUER_CONTROL_PROOF_READY": self.proof_ready,
            "blocker": self.blocker,
        }


@dataclass(frozen=True, slots=True)
class PlatformPost:
    source_item_id: str
    canonical_url: str
    published_at: datetime
    published_raw: str
    content: str

    def payload(self) -> dict[str, Any]:
        return {
            "source_item_id": self.source_item_id,
            "canonical_url": self.canonical_url,
            "published_at": self.published_at.isoformat(),
            "published_raw": self.published_raw,
            "content_sha": sha256_payload({"content": self.content}),
        }


@dataclass(frozen=True, slots=True)
class ChannelProbe:
    proof: OwnershipProof
    source_id: str
    public_access_ready: bool
    timestamp_ready: bool
    identity_ready: bool
    source_ready: bool
    status: str
    blocker: str | None
    response: FetchResult | None
    posts: tuple[PlatformPost, ...]

    def payload(self) -> dict[str, Any]:
        first = self.posts[0] if self.posts else None
        return {
            **self.proof.payload(),
            "source_id": self.source_id,
            "CHANNEL_DISCOVERED": bool(self.proof.channel_id),
            "ISSUER_CONTROL_PROVEN": self.proof.proof_ready,
            "PUBLIC_ACCESS_READY": self.public_access_ready,
            "TIMESTAMP_READY": self.timestamp_ready,
            "IDENTITY_READY": self.identity_ready,
            "SOURCE_READY": self.source_ready,
            "FREE_PUBLIC_SOURCE_READY": self.source_ready,
            "STRICT_EXACT_TIMESTAMP_READY": self.timestamp_ready,
            "STABLE_IDENTITY_READY": self.identity_ready,
            "current_status": self.status,
            "blocker": self.blocker,
            "http_status": None if self.response is None else self.response.status,
            "final_url": None if self.response is None else self.response.final_url,
            "content_type": None if self.response is None else self.response.content_type,
            "posts_observed": len(self.posts),
            "observed_post": None if first is None else first.payload(),
        }


def run_moex_issuer_controlled_channel_discovery_v6(
    *,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    operation_root: Path = DEFAULT_OPERATION_ROOT,
    instrument_mapping_path: Path = DEFAULT_INSTRUMENT_MAPPING_PATH,
    v5_root: Path = DEFAULT_V5_ROOT,
    client: HttpClient | None = None,
    network_check: bool = True,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable v6 output already exists")
    output_root.mkdir(parents=True, exist_ok=False)
    now = created_at or datetime.now(UTC)
    mapping_rows = load_instrument_mapping_rows(instrument_mapping_path)
    registry = canonical_registry_from_mapping(mapping_rows)
    universe, excluded = build_candidate_universe(
        mapping_rows=mapping_rows,
        canonical_registry=registry,
        previous_v4_rejections={},
    )
    v5_tickers = load_v5_canonical_tickers(v5_root)
    candidates = [
        ChannelCandidate(
            ticker=candidate.ticker,
            legal_issuer=candidate.legal_issuer,
            official_url=default_v5_source_configs()[candidate.ticker].url,
            official_domain=default_v5_source_configs()[candidate.ticker].official_domain,
            instrument=mapping_payload(candidate),
        )
        for candidate in universe
        if candidate.ticker in v5_tickers and candidate.ticker in default_v5_source_configs()
    ]
    http = client or BoundedHttpClient(
        timeout_seconds=8.0, redirect_limit=3, max_response_bytes=512_000
    )
    discovery_rows: list[dict[str, Any]] = []
    proofs: list[OwnershipProof] = []
    probes: list[ChannelProbe] = []
    rejections: list[dict[str, Any]] = []
    if network_check:
        for candidate in candidates:
            proof_rows, candidate_rejections = discover_issuer_controlled_channels(
                candidate, client=http
            )
            proofs.extend(proof_rows)
            rejections.extend(candidate_rejections)
            discovery_rows.append(
                {
                    "ticker": candidate.ticker,
                    "legal_issuer": candidate.legal_issuer,
                    "official_url": candidate.official_url,
                    "official_domain": candidate.official_domain,
                    "channels_discovered": len(proof_rows),
                    "issuer_control_proofs_found": sum(proof.proof_ready for proof in proof_rows),
                    "blockers": [row["blocker"] for row in candidate_rejections],
                }
            )
            for proof in proof_rows[:MAX_CHANNELS_PER_ISSUER]:
                probes.append(probe_platform_channel(proof, client=http))
    else:
        discovery_rows = [
            {
                "ticker": candidate.ticker,
                "legal_issuer": candidate.legal_issuer,
                "official_url": candidate.official_url,
                "official_domain": candidate.official_domain,
                "channels_discovered": 0,
                "issuer_control_proofs_found": 0,
                "blockers": ["PENDING"],
            }
            for candidate in candidates
        ]

    accepted_sources = [accepted_source_payload(probe) for probe in probes if probe.source_ready]
    rejected_sources = [
        *rejections,
        *[rejected_probe_payload(probe) for probe in probes if not probe.source_ready],
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
    blockers = Counter(row["blocker"] for row in rejected_sources if row.get("blocker"))
    live_shadow_posts = [
        post.payload()
        for probe in probes
        if probe.source_ready
        for post in probe.posts
        if post.published_at >= LIVE_EPOCH_START
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
        "AUTH_BYPASS_ATTEMPTS": 0,
        "PRIVATE_CHANNEL_READS": 0,
        "CANONICAL_ISSUERS_INSPECTED": len(candidates),
        "OFFICIAL_PLATFORM_CHANNELS_DISCOVERED": len(proofs),
        "ISSUER_OWNERSHIP_PROOFS_FOUND": sum(proof.proof_ready for proof in proofs),
        "PUBLIC_CHANNELS_ACCESSIBLE": sum(probe.public_access_ready for probe in probes),
        "STRICT_TIMESTAMP_READY_CHANNELS": sum(probe.timestamp_ready for probe in probes),
        "STABLE_IDENTITY_READY_CHANNELS": sum(probe.identity_ready for probe in probes),
        "SOURCE_READY_CHANNELS": sum(probe.source_ready for probe in probes),
        "TARGET_ELIGIBLE_SOURCES": diversity["NEW_TARGET_ELIGIBLE_SOURCE_COUNT"],
        "FEATURE_COMPATIBLE_SOURCES": diversity["FEATURE_PIPELINE_COMPATIBLE_SOURCE_COUNT"],
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
        "LIVE_SHADOW_POSTS_ELIGIBLE": len(live_shadow_posts),
        "ML_V2_DATASET_STATUS": "NOT_OPENED_BY_V6_CHANNEL_DISCOVERY",
        "SOURCE_READY_DOES_NOT_IMPLY_ML_DIVERSITY_ELIGIBLE": True,
        "ISSUER_ORIGINATED_OFFICIAL_DOMAIN_PRESERVED": True,
        "NEW_SOURCE_CLASS": SOURCE_CLASS,
        "NEW_SOURCE_ORIGIN": SOURCE_CLASS,
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
    _write_jsonl(output_root / "issuer-channel-discovery.jsonl", discovery_rows)
    _write_jsonl(output_root / "ownership-proof.jsonl", [proof.payload() for proof in proofs])
    _write_jsonl(output_root / "channel-probes.jsonl", [probe.payload() for probe in probes])
    _write_json(output_root / "accepted-sources.json", {"sources": accepted_sources})
    _write_jsonl(output_root / "rejected-sources.jsonl", rejected_sources)
    _write_jsonl(
        output_root / "target-mapping.jsonl", [candidate.instrument for candidate in candidates]
    )
    _write_json(output_root / "diversity-status.json", diversity)
    _write_json(
        output_root / "candidate-backlog.json",
        {
            "next_recheck_date": (now.date() + timedelta(days=7)).isoformat(),
            "candidates": candidate_backlog(candidates, proofs, probes, now.date()),
            "excluded_before_channel_discovery": excluded,
        },
    )
    _write_json(output_root / "safety.json", safety)
    _write_report(output_root / "report.md", manifest)
    return manifest


def load_v5_canonical_tickers(v5_root: Path) -> set[str]:
    manifest_path = v5_root / "manifest.json"
    if not manifest_path.exists():
        return set(default_v5_source_configs())
    manifest = cast("dict[str, Any]", json.loads(manifest_path.read_text(encoding="utf-8")))
    rows = manifest.get("CANONICAL_TARGET_TICKERS_CONSIDERED", [])
    return {str(ticker).upper() for ticker in rows if str(ticker).strip()}


def discover_issuer_controlled_channels(
    candidate: ChannelCandidate, *, client: HttpClient
) -> tuple[list[OwnershipProof], list[dict[str, Any]]]:
    if not official_first_party(candidate.official_url, candidate.official_domain):
        return [], [discovery_rejection(candidate, "POLICY_BLOCKED")]
    response = client.get(candidate.official_url)
    if response.blocker or response.status is None or response.status >= 400:
        return [], [
            discovery_rejection(candidate, response.blocker or "TECHNICAL_FAILURE", response)
        ]
    text = response.body.decode("utf-8", errors="replace")
    telegram_channels = list(dict.fromkeys(TELEGRAM_LINK_RE.findall(text)))[
        :MAX_CHANNELS_PER_ISSUER
    ]
    vk_pages = list(dict.fromkeys(VK_LINK_RE.findall(text)))[:MAX_CHANNELS_PER_ISSUER]
    proofs: list[OwnershipProof] = []
    for channel in telegram_channels:
        proofs.append(
            OwnershipProof(
                candidate.ticker,
                candidate.legal_issuer,
                "telegram",
                channel,
                f"https://t.me/{channel}",
                candidate.official_url,
                candidate.official_domain,
                "LEVEL_A_OFFICIAL_SITE_OUTBOUND_LINK",
                True,
                None,
            )
        )
    for page in vk_pages:
        proofs.append(
            OwnershipProof(
                candidate.ticker,
                candidate.legal_issuer,
                "vk",
                page,
                f"https://vk.com/{page}",
                candidate.official_url,
                candidate.official_domain,
                "LEVEL_A_OFFICIAL_SITE_OUTBOUND_LINK",
                True,
                None,
            )
        )
    if not proofs:
        return [], [discovery_rejection(candidate, "NO_OFFICIAL_CHANNEL_REFERENCE", response)]
    return proofs, []


def probe_platform_channel(proof: OwnershipProof, *, client: HttpClient) -> ChannelProbe:
    if not proof.proof_ready:
        return channel_probe(proof, "ISSUER_CONTROL_UNPROVEN", None, ())
    if proof.platform == "telegram":
        probe_url = f"https://t.me/s/{proof.channel_id}"
        response = client.get(probe_url)
        if response.blocker or response.status is None or response.status >= 400:
            blocker = response.blocker or "PUBLIC_ACCESS_FAILURE"
            if blocker == "AUTH_REQUIRED":
                return channel_probe(proof, "AUTH_REQUIRED", response, ())
            return channel_probe(proof, blocker, response, ())
        text = response.body.decode("utf-8", errors="replace")
        if "tgme_channel_join_telegram" in text or "private channel" in text.lower():
            return channel_probe(proof, "PRIVATE_CHANNEL", response, ())
        posts = parse_telegram_posts(text, proof.channel_id)
        if not posts and RELATIVE_TIME_RE.search(text):
            return channel_probe(proof, "TIMESTAMP_UNVERIFIED", response, ())
        if not posts:
            return channel_probe(proof, "STABLE_IDENTITY_UNVERIFIED", response, ())
        return ChannelProbe(
            proof,
            source_id=f"{proof.ticker}_{proof.channel_id.upper()}_TELEGRAM_V6",
            public_access_ready=True,
            timestamp_ready=True,
            identity_ready=True,
            source_ready=True,
            status="STRICT_EXACT_READY",
            blocker=None,
            response=response,
            posts=tuple(posts[:MAX_POSTS_PER_CHANNEL]),
        )
    return channel_probe(proof, "POLICY_BLOCKED", None, ())


def parse_telegram_posts(text: str, channel_id: str) -> list[PlatformPost]:
    post_ids = TG_POST_RE.findall(text)
    timestamps = TG_TIME_RE.findall(text)
    posts: list[PlatformPost] = []
    for raw_post_id, raw_timestamp in zip(post_ids, timestamps, strict=False):
        if not raw_post_id.startswith(f"{channel_id}/"):
            continue
        try:
            published = parse_publication_timestamp(raw_timestamp, None)
        except ValueError:
            continue
        message_id = raw_post_id.rsplit("/", 1)[-1]
        canonical_url = f"https://t.me/{channel_id}/{message_id}"
        posts.append(
            PlatformPost(
                source_item_id=canonical_url,
                canonical_url=canonical_url,
                published_at=published,
                published_raw=raw_timestamp,
                content=html.unescape(strip_tags(text[:4000])),
            )
        )
    return posts


def channel_probe(
    proof: OwnershipProof,
    blocker: str,
    response: FetchResult | None,
    posts: tuple[PlatformPost, ...],
) -> ChannelProbe:
    timestamp_ready = blocker not in {"TIMESTAMP_UNVERIFIED"} and bool(posts)
    identity_ready = blocker not in {"STABLE_IDENTITY_UNVERIFIED"} and bool(posts)
    public_ready = blocker not in {
        "AUTH_REQUIRED",
        "PRIVATE_CHANNEL",
        "PUBLIC_ACCESS_FAILURE",
        "TECHNICAL_FAILURE",
    }
    return ChannelProbe(
        proof,
        source_id=f"{proof.ticker}_{proof.channel_id.upper()}_{proof.platform.upper()}_V6",
        public_access_ready=public_ready and bool(posts),
        timestamp_ready=timestamp_ready,
        identity_ready=identity_ready,
        source_ready=False,
        status=blocker,
        blocker=blocker,
        response=response,
        posts=posts,
    )


def accepted_source_payload(probe: ChannelProbe) -> dict[str, Any]:
    first = probe.posts[0]
    payload = {
        "source_id": probe.source_id,
        "ticker": probe.proof.ticker,
        "legal_issuer": probe.proof.legal_issuer,
        "domain": urlparse(probe.proof.channel_url).netloc,
        "discovery_url": probe.proof.channel_url,
        "mechanism": f"{probe.proof.platform}_issuer_controlled_public_channel",
        "source_class": SOURCE_CLASS,
        "source_origin": SOURCE_CLASS,
        "ISSUER_CONTROL_PROOF_READY": True,
        "FREE_PUBLIC_SOURCE_READY": True,
        "STRICT_EXACT_TIMESTAMP_READY": True,
        "STABLE_IDENTITY_READY": True,
        "FREE_OFFICIAL_SOURCE_READY": True,
        "SOURCE_READY": True,
        "timestamp_field": "telegram_web.time.datetime",
        "timezone_evidence": "LEVEL_A_EXPLICIT_OFFSET_OR_UTC",
        "identity_mechanism": "canonical public message URL",
        "parser_version": "telegram-public-web-message-v6",
        "real_item_observed": True,
        "first_item": first.payload(),
        "live_shadow_candidates": [
            post.payload() for post in probe.posts if post.published_at >= LIVE_EPOCH_START
        ],
    }
    return payload | {"contract_sha": sha256_payload(payload)}


def rejected_probe_payload(probe: ChannelProbe) -> dict[str, Any]:
    return {
        "source_id": probe.source_id,
        "ticker": probe.proof.ticker,
        "legal_issuer": probe.proof.legal_issuer,
        "platform": probe.proof.platform,
        "channel_url": probe.proof.channel_url,
        "status": probe.status,
        "blocker": probe.blocker,
        "ISSUER_CONTROL_PROOF_READY": probe.proof.proof_ready,
        "PUBLIC_ACCESS_READY": probe.public_access_ready,
        "TIMESTAMP_READY": probe.timestamp_ready,
        "IDENTITY_READY": probe.identity_ready,
        "SOURCE_READY": probe.source_ready,
        "paid_fallback_considered": False,
    }


def discovery_rejection(
    candidate: ChannelCandidate, blocker: str, response: FetchResult | None = None
) -> dict[str, Any]:
    mapped = blocker if blocker in BLOCKERS else "TECHNICAL_FAILURE"
    return {
        "ticker": candidate.ticker,
        "legal_issuer": candidate.legal_issuer,
        "official_url": candidate.official_url,
        "official_domain": candidate.official_domain,
        "status": mapped,
        "blocker": mapped,
        "http_status": None if response is None else response.status,
        "ISSUER_CONTROL_PROOF_READY": False,
        "SOURCE_READY": False,
        "paid_fallback_considered": False,
    }


BLOCKERS = {
    "NO_OFFICIAL_CHANNEL_REFERENCE",
    "ISSUER_CONTROL_UNPROVEN",
    "AUTH_REQUIRED",
    "PRIVATE_CHANNEL",
    "TIMESTAMP_UNVERIFIED",
    "STABLE_IDENTITY_UNVERIFIED",
    "PUBLIC_ACCESS_FAILURE",
    "POLICY_BLOCKED",
    "TARGET_INELIGIBLE",
    "FEATURE_INCOMPATIBLE",
    "TECHNICAL_FAILURE",
}


def candidate_backlog(
    candidates: Sequence[ChannelCandidate],
    proofs: Sequence[OwnershipProof],
    probes: Sequence[ChannelProbe],
    checked_on: date,
) -> list[dict[str, Any]]:
    proofs_by_ticker: dict[str, list[OwnershipProof]] = {}
    for proof in proofs:
        proofs_by_ticker.setdefault(proof.ticker, []).append(proof)
    probe_by_source = {probe.source_id: probe for probe in probes}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_proofs = proofs_by_ticker.get(candidate.ticker, [])
        attempted = [proof.channel_url for proof in candidate_proofs]
        current_status = "NO_OFFICIAL_CHANNEL_REFERENCE"
        if candidate_proofs:
            current_status = "ISSUER_CONTROL_PROVEN"
            for proof in candidate_proofs:
                source_id = f"{proof.ticker}_{proof.channel_id.upper()}_{proof.platform.upper()}_V6"
                if source_id in probe_by_source:
                    current_status = probe_by_source[source_id].status
                    break
        rows.append(
            {
                "legal_issuer": candidate.legal_issuer,
                "canonical_tickers": [candidate.ticker],
                "instrument_uid": candidate.instrument.get("instrument_uid"),
                "board": candidate.instrument.get("board"),
                "market_data_compatible": candidate.instrument.get("market_data_mapping_ready"),
                "official_domains": [candidate.official_domain],
                "mechanism_attempted": "official-domain outbound Telegram/VK discovery",
                "platform_channels_attempted": attempted,
                "current_source_status": current_status,
                "next_free_hypothesis": (
                    "Search another official-domain page that names public channels explicitly."
                ),
                "next_recheck_date": (checked_on + timedelta(days=7)).isoformat(),
            }
        )
    return rows


def safety_payload(status: dict[str, Any]) -> dict[str, Any]:
    counters = cast("dict[str, Any]", status.get("outcome_counters", {}))
    return {
        **live_accumulation_safety_flags(),
        "FREE_SOURCES_ONLY": True,
        "PAID_SOURCES_USED": False,
        "PAID_API_CALLS": 0,
        "AUTH_BYPASS_ATTEMPTS": 0,
        "PRIVATE_CHANNEL_READS": 0,
        "LIVE_OUTCOMES_READ": int(counters.get("LIVE_OUTCOMES_READ", 0)),
        "LIVE_TARGETS_COMPUTED": int(counters.get("LIVE_TARGETS_COMPUTED", 0)),
        "LIVE_POST_EVENT_PRICE_READS": int(counters.get("LIVE_POST_EVENT_PRICE_READS", 0)),
        "LIVE_MODEL_PREDICTIONS": 0,
        "MODEL_TRAINING_PERFORMED": False,
        "BACKTEST_PERFORMED": False,
        "OLD_FUTURE_HOLDOUT_OPENED": False,
        "BROKER_MUTATIONS": int(counters.get("BROKER_MUTATIONS", 0)),
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
    passed = (
        live_status["LIVE_RESEARCH_OPERATION_STATUS"] == "READY"
        and live_seal["sealed_epoch_verified"] is True
        and int(live_status.get("timestamp_rejections", 0)) == 0
        and int(live_status.get("sealed_violations", 0)) == 0
        and safety_zero
    )
    return {
        "OPERATIONAL_BURN_IN": "PASS" if passed else "FAIL",
        "OPERATION": "YES" if passed else "NO",
    }


def official_first_party(url: str, official_domain: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.netloc.lower() == official_domain.lower()


def strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


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


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# MOEX issuer-controlled channel discovery v6",
        "",
        f"- BASE_MAIN_SHA: {manifest['BASE_MAIN_SHA']}",
        f"- HEAD_SHA: {manifest['HEAD_SHA']}",
        f"- ARTIFACT_SHA: {manifest['ARTIFACT_SHA']}",
        f"- Canonical issuers inspected: {manifest['CANONICAL_ISSUERS_INSPECTED']}",
        (
            "- Official platform channels discovered: "
            f"{manifest['OFFICIAL_PLATFORM_CHANNELS_DISCOVERED']}"
        ),
        f"- Issuer ownership proofs found: {manifest['ISSUER_OWNERSHIP_PROOFS_FOUND']}",
        f"- Public channels accessible: {manifest['PUBLIC_CHANNELS_ACCESSIBLE']}",
        f"- Strict timestamp-ready channels: {manifest['STRICT_TIMESTAMP_READY_CHANNELS']}",
        f"- Stable-identity-ready channels: {manifest['STABLE_IDENTITY_READY_CHANNELS']}",
        f"- Source-ready channels: {manifest['SOURCE_READY_CHANNELS']}",
        f"- Target-eligible sources: {manifest['TARGET_ELIGIBLE_SOURCES']}",
        f"- Feature-compatible sources: {manifest['FEATURE_COMPATIBLE_SOURCES']}",
        (
            "- New target-eligible distinct legal issuers: "
            f"{manifest['NEW_TARGET_ELIGIBLE_DISTINCT_LEGAL_ISSUERS']}"
        ),
        f"- Blockers by category: {manifest['BLOCKERS_BY_CATEGORY']}",
        f"- TARGET_DIVERSITY: {manifest['TARGET_DIVERSITY']}",
        f"- LIVE_RESEARCH_OPERATION_STATUS: {manifest['LIVE_RESEARCH_OPERATION_STATUS']}",
        f"- OPERATIONAL_BURN_IN: {manifest['OPERATIONAL_BURN_IN']}",
        "",
        (
            "No auth bypass, private channel reads, outcomes, targets, model training, "
            "backtest, or broker mutation."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
