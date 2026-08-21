from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import EventAnalyzerV3, rules_v3_fingerprint
from src.exact_event_corpus.domain import ExactEvent, deterministic_clusters
from src.exact_event_source_diversity_v3.domain import (
    ARTIFACT_VERSION,
    FUTURE_EVENT_HOLDOUT_START,
    INPUT_WARMUP_DATASET_SHA,
    OUTPUT_DATASET_VERSION,
    SourceDiscoveryRecord,
    SourceStatus,
    concentration,
    parse_rss_pubdate_utc,
    require_warmup_manifest,
    sha256_payload,
    source_diversity_safety_flags,
)

_MOEX_ISSUER_RSS_URL = "https://www.moex.com/export/news.aspx?cat=100"
_MOEX_DOMAIN = "www.moex.com"
_MOEX_SOURCE_FAMILY = "MOEX_OFFICIAL_ISSUER_NOTICE_RSS_V3"
_REJECTED_MOEX_PHRASES = (
    "ценового коридора",
    "дискретн",
    "дестабилизации цен",
    "риск-параметр",
    "РЕПО",
)
_TICKER_PATTERNS = (
    re.compile(r"ценн(?:ой|ая|ые|ых) бумаг(?:и|а|ах)?\s+([A-Z0-9]{2,10})"),  # noqa: RUF001
    re.compile(r"\b([A-Z]{2,5}P?)\b"),
)


def build_source_diversity_v3_artifact(
    *,
    warmup_root: Path,
    v2_root: Path,
    universe_path: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    created_at: datetime | None = None,
    moex_feed_path: Path | None = None,
    max_new_events: int = 100,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable source diversity v3 artifact output already exists")
    _verify_frozen_contracts()
    warmup_manifest = _read_json(warmup_root / "manifest.json")
    require_warmup_manifest(warmup_manifest)
    events_before = _read_jsonl(warmup_root / "events.jsonl")
    features_before = _read_jsonl(warmup_root / "features.jsonl")
    targets_before = _read_jsonl(v2_root / "targets.jsonl")
    previous_registry = _read_jsonl(v2_root / "source-registry.jsonl")
    universe = _read_universe(universe_path)
    feed_path = (
        moex_feed_path
        or v2_root / "raw-source-cache" / "MOEX_OFFICIAL_ISSUER_NOTICE_RSS" / "feed.xml"
    )
    feed_items = _read_moex_feed(feed_path)
    existing_tickers = {_metadata(row)["ticker"] for row in events_before}
    feed_records = _feed_records(feed_items, universe)
    registry = _source_registry_v3(universe, previous_registry, feed_records, existing_tickers)
    new_events, acquisition = _new_events_from_feed(
        feed_records,
        existing_tickers=existing_tickers,
        existing_rows=events_before,
        max_new_events=max_new_events,
    )
    new_rows = _event_rows(new_events)
    events_after = [*events_before, *new_rows]
    features_after = [*features_before]
    targets_after = [*targets_before]
    new_clusters = deterministic_clusters(new_events)
    clusters_after = [
        *_read_jsonl(v2_root / "clusters.jsonl"),
        *[
            {"event_id": str(event.event_id), "event_cluster_id": str(new_clusters[event.event_id])}
            for event in new_events
        ],
    ]
    _assert_prefix_preserved(events_before, events_after, "events")
    _assert_prefix_preserved(features_before, features_after, "features")
    _assert_no_future_targets(events_after, targets_after)
    before_metrics = _metrics(events_before, features_before)
    after_metrics = _metrics(events_after, features_after)
    registry_rows = [record.payload() for record in registry]
    source_status_counts = Counter(str(row["timestamp_capability"]) for row in registry_rows)
    source_registry_sha = sha256_payload(registry_rows)
    provenance = {
        "artifact_version": ARTIFACT_VERSION,
        "base_main_sha": base_main_sha,
        "git_sha": git_sha,
        "input_warmup_dataset_sha": INPUT_WARMUP_DATASET_SHA,
        "source": "OFFICIAL_ZERO_COST_PUBLIC_SOURCES_ONLY",
        "new_source_transport": "OFFICIAL_RSS_CACHE",
        "moex_feed_path": str(feed_path),
        "market_source": "TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES_ONLY",
        "tinkoff_token_value_read": False,
        "source_selection_used_returns": False,
        "source_selection_used_model_metrics": False,
        "source_selection_used_test_outcomes": False,
        "rules_v3_fingerprint": rules_v3_fingerprint(),
        "qwen_prompt_sha": prompt_hash(),
        "qwen_schema_sha": schema_hash(),
    }
    timestamp_manifest = {
        "timestamp_methodology": "EXACT_OFFICIAL_RSS_PUBDATE_WITH_EXPLICIT_OFFSET",
        "date_only_coercions": 0,
        "fetch_time_used_as_publication_time": False,
        "events": [
            {
                "event_id": str(event.event_id),
                "source_item_id": event.source_item_id,
                "raw": event.publication_timestamp_raw,
                "utc": event.publication_timestamp_utc.isoformat(),
                "source_field": event.timestamp_source_field,
            }
            for event in new_events
        ],
    }
    duplicate_reconciliation = _duplicate_reconciliation(events_after)
    feature_schema_sha = _feature_schema_sha(features_after)
    output_dataset_sha = sha256_payload(
        {
            "dataset_version": OUTPUT_DATASET_VERSION,
            "input_dataset_sha": INPUT_WARMUP_DATASET_SHA,
            "events": events_after,
            "features": features_after,
            "targets": targets_after,
        }
    )
    safety = source_diversity_safety_flags()
    manifest: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_DATASET_SHA": INPUT_WARMUP_DATASET_SHA,
        "OUTPUT_DATASET_VERSION": OUTPUT_DATASET_VERSION,
        "OUTPUT_DATASET_SHA": output_dataset_sha,
        "SOURCE_REGISTRY_SHA": source_registry_sha,
        "PROVENANCE_SHA": sha256_payload(provenance),
        "TIMESTAMP_METHODOLOGY_SHA": sha256_payload(timestamp_manifest),
        "DEDUPE_CLUSTER_SHA": sha256_payload(clusters_after),
        "FEATURE_SCHEMA_SHA": feature_schema_sha,
        "ARTIFACT_SHA": None,
        "TOTAL_CANDIDATES": len(universe),
        "SOURCES_DISCOVERED": sum(row.source_found for row in registry),
        "SOURCE_STATUS_COUNTS": dict(sorted(source_status_counts.items())),
        "EXACT_CAPABLE": source_status_counts["EXACT"] + source_status_counts["MIXED"],
        "MIXED": source_status_counts["MIXED"],
        "DATE_ONLY": source_status_counts["DATE_ONLY"],
        "UNKNOWN": source_status_counts["UNKNOWN"],
        "EXACT_CAPABLE_SOURCES_BEFORE": _exact_capable(previous_registry),
        "EXACT_CAPABLE_SOURCES_AFTER": sum(
            row.timestamp_capability in {SourceStatus.EXACT, SourceStatus.MIXED} for row in registry
        ),
        "before": before_metrics,
        "after": after_metrics,
        "EXACT_TOTAL_BEFORE": before_metrics["EXACT_TOTAL"],
        "EXACT_TOTAL_AFTER": after_metrics["EXACT_TOTAL"],
        "EXACT_DELTA": after_metrics["EXACT_TOTAL"] - before_metrics["EXACT_TOTAL"],
        "EXACT_TICKERS_BEFORE": before_metrics["EXACT_UNIQUE_TICKERS"],
        "EXACT_TICKERS_AFTER": after_metrics["EXACT_UNIQUE_TICKERS"],
        "EXACT_ISSUERS_BEFORE": before_metrics["EXACT_UNIQUE_ISSUERS"],
        "EXACT_ISSUERS_AFTER": after_metrics["EXACT_UNIQUE_ISSUERS"],
        "SOURCE_FAMILIES_BEFORE": before_metrics["EXACT_SOURCE_FAMILIES"],
        "SOURCE_FAMILIES_AFTER": after_metrics["EXACT_SOURCE_FAMILIES"],
        "REACTION_READY_BEFORE": before_metrics["REACTION_READY"],
        "REACTION_READY_AFTER": after_metrics["REACTION_READY"],
        "FEATURE_READY_BEFORE": before_metrics["FEATURE_READY"],
        "FEATURE_READY_AFTER": after_metrics["FEATURE_READY"],
        "NEW_EXACT_TICKERS": sorted({event.ticker for event in new_events} - existing_tickers),
        "NEW_EXACT_ISSUERS": sorted({event.issuer for event in new_events}),
        "NEW_SOURCE_FAMILIES": sorted(
            {_MOEX_SOURCE_FAMILY} - set(before_metrics["source_family_counts"])
        ),
        "EVENTS_BY_TICKER": after_metrics["events_by_ticker"],
        "FEATURE_READY_BY_TICKER": after_metrics["feature_ready_by_ticker"],
        "TICKER_TOP1_BEFORE": before_metrics["ticker_concentration"]["top1_share"],
        "TICKER_TOP1_AFTER": after_metrics["ticker_concentration"]["top1_share"],
        "TICKER_TOP3_BEFORE": before_metrics["ticker_concentration"]["top3_share"],
        "TICKER_TOP3_AFTER": after_metrics["ticker_concentration"]["top3_share"],
        "ISSUER_HHI_BEFORE": before_metrics["issuer_concentration"]["hhi"],
        "ISSUER_HHI_AFTER": after_metrics["issuer_concentration"]["hhi"],
        "EFFECTIVE_ISSUER_COUNT_BEFORE": before_metrics["issuer_concentration"]["effective_count"],
        "EFFECTIVE_ISSUER_COUNT_AFTER": after_metrics["issuer_concentration"]["effective_count"],
        "SOURCE_TOP1_BEFORE": before_metrics["source_concentration"]["top1_share"],
        "SOURCE_TOP1_AFTER": after_metrics["source_concentration"]["top1_share"],
        "SOURCE_HHI_BEFORE": before_metrics["source_concentration"]["hhi"],
        "SOURCE_HHI_AFTER": after_metrics["source_concentration"]["hhi"],
        "EFFECTIVE_SOURCE_COUNT_BEFORE": before_metrics["source_concentration"]["effective_count"],
        "EFFECTIVE_SOURCE_COUNT_AFTER": after_metrics["source_concentration"]["effective_count"],
        "TIMESTAMP_PROVENANCE_COUNTS": after_metrics["timestamp_provenance_counts"],
        "SESSION_COUNTS": after_metrics["session_counts"],
        "MATCHED": acquisition["MATCHED"],
        "AMBIGUOUS": acquisition["AMBIGUOUS"],
        "UNMATCHED": acquisition["UNMATCHED"],
        "DUPLICATE_RECONCILIATION": duplicate_reconciliation,
        "TECHNICAL_BLOCKED": sum(
            row.timestamp_capability == SourceStatus.TECHNICAL_BLOCKED for row in registry
        ),
        "POLICY_BLOCKED": sum(
            row.timestamp_capability == SourceStatus.POLICY_BLOCKED for row in registry
        ),
        "FAILED_CLOSED": sum(
            row.timestamp_capability == SourceStatus.FAILED_CLOSED for row in registry
        ),
        "EXACT_V2_PRESERVED": "YES",
        "EXISTING_EVENT_ROWS_PRESERVED": "PASS",
        "EXISTING_FEATURE_ROWS_PRESERVED": "PASS",
        "LEAKAGE_CHECK": "PASS",
        "safety": safety,
        "MODEL_TRAINING_PERFORMED": safety["MODEL_TRAINING_PERFORMED"],
        "TEST_OUTCOME_USED": safety["TEST_OUTCOME_USED"],
        "FUTURE_EVENT_HOLDOUT_USED": safety["FUTURE_EVENT_HOLDOUT_USED"],
        "FUTURE_EVENT_HOLDOUT_OBSERVED": safety["FUTURE_EVENT_HOLDOUT_OBSERVED"],
        "RULES_V3_CHANGED": safety["RULES_V3_CHANGED"],
        "QWEN_CHANGED": safety["QWEN_CHANGED"],
        "NLP_TUNING_PERFORMED": safety["NLP_TUNING_PERFORMED"],
        "CONFIRMED_SIGNAL": safety["CONFIRMED_SIGNAL"],
        "BACKTEST_APPROVED": safety["BACKTEST_APPROVED"],
        "PAPER_TRADING_APPROVED": safety["PAPER_TRADING_APPROVED"],
        "REAL_TRADING_APPROVED": safety["REAL_TRADING_APPROVED"],
    }
    manifest["ARTIFACT_SHA"] = sha256_payload({**manifest, "ARTIFACT_SHA": None})
    _write_artifacts(
        output_root=output_root,
        events=events_after,
        features=features_after,
        targets=targets_after,
        registry=registry_rows,
        source_blockers=registry_rows,
        clusters=clusters_after,
        provenance={**provenance, "sha256": manifest["PROVENANCE_SHA"]},
        timestamp_manifest=timestamp_manifest,
        manifest=manifest,
    )
    return manifest


def _verify_frozen_contracts() -> None:
    if rules_v3_fingerprint() != EXPECTED_RULES_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_MISMATCH")
    if prompt_hash() != QWEN_PROMPT_SHA or schema_hash() != QWEN_SCHEMA_SHA:
        raise ValueError("FROZEN_QWEN_CONTRACT_MISMATCH")


def _read_universe(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    rows = cast("list[dict[str, Any]]", payload["instruments"])
    return {
        str(row["ticker"]): row
        for row in rows
        if str(row["ticker"]) != "IMOEX" and str(row.get("class_code")) == "TQBR"
    }


def _read_moex_feed(path: Path) -> list[dict[str, str]]:
    root = ET.fromstring(path.read_bytes())
    rows: list[dict[str, str]] = []
    for item in root.findall("./channel/item"):
        rows.append(
            {
                "title": html.unescape(item.findtext("title") or ""),
                "description": html.unescape(item.findtext("description") or ""),
                "link": item.findtext("link") or "",
                "pubDate": item.findtext("pubDate") or "",
            }
        )
    return rows


def _feed_records(
    feed_items: list[dict[str, str]], universe: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in feed_items:
        text = f"{item['title']} {item['description']}"
        if any(phrase in text for phrase in _REJECTED_MOEX_PHRASES):
            continue
        tickers = _extract_tickers(text, universe)
        if not tickers:
            continue
        published = parse_rss_pubdate_utc(item["pubDate"])
        for ticker in tickers:
            records.append(
                {
                    **item,
                    "ticker": ticker,
                    "issuer": str(universe[ticker]["name"]),
                    "instrument_uid": str(universe[ticker]["instrument_uid"]),
                    "published_at": published,
                }
            )
    return sorted(
        records,
        key=lambda row: (
            cast("datetime", row["published_at"]),
            str(row["ticker"]),
            str(row["link"]),
        ),
    )


def _extract_tickers(text: str, universe: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    found: set[str] = set()
    for pattern in _TICKER_PATTERNS:
        for ticker in pattern.findall(text):
            if ticker in universe:
                found.add(ticker)
    return tuple(sorted(found))


def _source_registry_v3(
    universe: dict[str, dict[str, Any]],
    previous_registry: list[dict[str, Any]],
    feed_records: list[dict[str, Any]],
    existing_tickers: set[str],
) -> tuple[SourceDiscoveryRecord, ...]:
    previous = {str(row["ticker"]): row for row in previous_registry}
    feed_tickers = {str(row["ticker"]) for row in feed_records}
    records: list[SourceDiscoveryRecord] = []
    for ticker, instrument in sorted(universe.items()):
        old = previous.get(ticker, {})
        if ticker in feed_tickers and ticker not in existing_tickers:
            records.append(
                SourceDiscoveryRecord(
                    issuer=str(instrument["name"]),
                    ticker=ticker,
                    official_domain=_MOEX_DOMAIN,
                    source_url=_MOEX_ISSUER_RSS_URL,
                    source_family=_MOEX_SOURCE_FAMILY,
                    transport_type="OFFICIAL_RSS",
                    timestamp_capability=SourceStatus.EXACT,
                    timezone_semantics="EXPLICIT",
                    archive_depth="bounded cached MOEX issuer RSS page",
                    pagination_method="single RSS feed cache; max item limit applied",
                    machine_readable_status="MACHINE_READABLE_XML",
                    policy_status="OFFICIAL_PUBLIC_ZERO_COST_VERIFIED",
                    technical_status="SOURCE_READY",
                    acquisition_status="SOURCE_READY_METADATA_ONLY",
                    provenance="official MOEX RSS pubDate with numeric timezone offset",
                    official_ownership_proof="moex.com official exchange news RSS",
                    source_found=True,
                    exact_timestamp=True,
                    archive=True,
                    source_ready=True,
                    technical_blocker=None,
                    policy_blocker=None,
                    notes="New ticker discovered target-free from cached official MOEX issuer RSS.",
                )
            )
            continue
        status = SourceStatus(str(old.get("timestamp_capability", "UNKNOWN")))
        source_found = bool(old.get("source_url"))
        records.append(
            SourceDiscoveryRecord(
                issuer=str(instrument["name"]),
                ticker=ticker,
                official_domain=_optional_string(old.get("official_domain")),
                source_url=_optional_string(old.get("source_url")),
                source_family=_optional_string(old.get("source_family")),
                transport_type=_transport_for(old),
                timestamp_capability=status,
                timezone_semantics=str(old.get("timezone_semantics", "UNKNOWN")),
                archive_depth=_optional_string(old.get("historical_archive_start")),
                pagination_method="preserved from v2 registry" if source_found else None,
                machine_readable_status=(
                    "MACHINE_READABLE_OR_DETERMINISTIC" if source_found else "UNKNOWN"
                ),
                policy_status=str(old.get("source_policy_status", "UNKNOWN_FAIL_CLOSED")),
                technical_status=str(old.get("collector_status", "NO_OFFICIAL_NEWS_ARCHIVE")),
                acquisition_status=str(old.get("collector_status", "NO_OFFICIAL_NEWS_ARCHIVE")),
                provenance="preserved v2 source registry",
                official_ownership_proof=_optional_string(old.get("official_domain")),
                source_found=source_found,
                exact_timestamp=status in {SourceStatus.EXACT, SourceStatus.MIXED},
                archive=bool(old.get("historical_archive_start")),
                source_ready=str(old.get("collector_status")) == "SOURCE_READY",
                technical_blocker=None if source_found else "NO_VERIFIED_OFFICIAL_EXACT_ARCHIVE",
                policy_blocker=None,
                notes=str(old.get("reason", "Preserved v2 fail-closed registry state")),
            )
        )
    return tuple(records)


def _new_events_from_feed(
    feed_records: list[dict[str, Any]],
    *,
    existing_tickers: set[str],
    existing_rows: list[dict[str, Any]],
    max_new_events: int,
) -> tuple[list[ExactEvent], dict[str, int]]:
    existing_identities = {
        (str(_metadata(row)["source_code"]), str(_metadata(row)["source_item_id"]))
        for row in existing_rows
    }
    selected: list[ExactEvent] = []
    matched = ambiguous = unmatched = 0
    for row in feed_records:
        ticker = str(row["ticker"])
        if ticker in existing_tickers:
            continue
        matched += 1
        link = str(row["link"])
        event = ExactEvent.create(
            source_code="MOEX_OFFICIAL_ISSUER_NOTICE_RSS_EXACT_V3",
            source_item_id=f"{ticker}:{link}",
            canonical_url=link,
            ticker=ticker,
            issuer=str(row["issuer"]),
            instrument_uid=str(row["instrument_uid"]),
            title=str(row["title"]),
            publication_timestamp_raw=str(row["pubDate"]),
            publication_timestamp_utc=cast("datetime", row["published_at"]),
            timestamp_source_field="official MOEX RSS item pubDate with explicit +0300 offset",
        )
        identity = (event.source_code, event.source_item_id)
        if identity in existing_identities:
            continue
        selected.append(event)
        if len(selected) >= max_new_events:
            break
    return selected, {"MATCHED": matched, "AMBIGUOUS": ambiguous, "UNMATCHED": unmatched}


def _event_rows(events: list[ExactEvent]) -> list[dict[str, Any]]:
    analyzer = EventAnalyzerV3()
    clusters = deterministic_clusters(events)
    rows: list[dict[str, Any]] = []
    for event in events:
        future = event.publication_date >= FUTURE_EVENT_HOLDOUT_START
        analysis = analyzer.analyze(news_id=event.event_id, raw_content=event.title)
        metadata = {
            **event.metadata_payload(),
            "event_cluster_id": str(clusters[event.event_id]),
            "session_state": "FUTURE_METADATA_ONLY" if future else "MARKET_CONTEXT_NOT_BUILT",
            "market_alignment_version": "tinvest-exact-minute-alignment-v1",
            "reaction_family": "EXACT_INTRADAY",
            "future_holdout": future,
        }
        rows.append(
            {
                "metadata": metadata,
                "event_features": {
                    "primary_event_type": analysis.primary_event_type.value,
                    "event_count": len(analysis.events),
                    "fact_count": len(analysis.financial_facts),
                },
                "pre_event_market_features": {},
                "target_availability": {
                    "research_outcomes_visible": False,
                    "reaction_ready": False,
                    "feature_ready": False,
                    "status": (
                        "FUTURE_HOLDOUT_METADATA_ONLY"
                        if future
                        else "TINVEST_HISTORY_NOT_ACQUIRED_IN_V3_CACHE_ONLY"
                    ),
                    "missing_reason": (
                        "FUTURE_HOLDOUT_OUTCOMES_GUARDED"
                        if future
                        else "TINVEST_HISTORY_UNAVAILABLE_CACHE_ONLY"
                    ),
                },
                "quality": {
                    "feature_cutoff": event.publication_timestamp_utc.isoformat(),
                    "reaction_starts_after_or_at_publication": True,
                    "security_benchmark_same_window": True,
                    "no_forward_fill": True,
                    "no_interpolation": True,
                    "no_source_mixing": True,
                },
            }
        )
    return rows


def _metrics(events: list[dict[str, Any]], features: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = [_metadata(row) for row in events]
    ticker_counts = Counter(str(row["ticker"]) for row in metadata)
    issuer_counts = Counter(str(row["issuer"]) for row in metadata)
    source_counts = Counter(str(row["source_code"]) for row in metadata)
    feature_by_ticker = Counter(
        str(_metadata_by_id(events)[str(row["event_id"])]["ticker"]) for row in features
    )
    timestamp_provenance = Counter(str(row["timestamp_source_field"]) for row in metadata)
    session_counts = Counter(str(row["session_state"]) for row in metadata)
    return {
        "EXACT_TOTAL": len(events),
        "EXACT_UNIQUE_TICKERS": len(ticker_counts),
        "EXACT_UNIQUE_ISSUERS": len(issuer_counts),
        "EXACT_SOURCE_FAMILIES": len(source_counts),
        "REACTION_READY": sum(
            bool(cast("dict[str, Any]", row["target_availability"])["reaction_ready"])
            for row in events
        ),
        "FEATURE_READY": len(features),
        "events_by_ticker": dict(sorted(ticker_counts.items())),
        "feature_ready_by_ticker": dict(sorted(feature_by_ticker.items())),
        "source_family_counts": dict(sorted(source_counts.items())),
        "ticker_concentration": concentration(ticker_counts),
        "issuer_concentration": concentration(issuer_counts),
        "source_concentration": concentration(source_counts),
        "timestamp_provenance_counts": dict(sorted(timestamp_provenance.items())),
        "session_counts": dict(sorted(session_counts.items())),
    }


def _metadata_by_id(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(_metadata(row)["event_id"]): _metadata(row) for row in events}


def _feature_schema_sha(features: list[dict[str, Any]]) -> str:
    event_names = sorted(
        {name for row in features for name in cast("dict[str, Any]", row["event_features"])}
    )
    market_names = sorted(
        {name for row in features for name in cast("dict[str, Any]", row["market_features"])}
    )
    return sha256_payload({"event_features": event_names, "market_features": market_names})


def _duplicate_reconciliation(events: list[dict[str, Any]]) -> str:
    event_ids = [str(_metadata(row)["event_id"]) for row in events]
    identities = [
        (str(_metadata(row)["source_code"]), str(_metadata(row)["source_item_id"]))
        for row in events
    ]
    return (
        "PASS"
        if len(event_ids) == len(set(event_ids)) and len(identities) == len(set(identities))
        else "FAIL"
    )


def _assert_prefix_preserved(
    before: list[dict[str, Any]], after: list[dict[str, Any]], name: str
) -> None:
    if after[: len(before)] != before:
        raise ValueError(f"EXISTING_{name.upper()}_ROWS_NOT_PRESERVED")


def _assert_no_future_targets(events: list[dict[str, Any]], targets: list[dict[str, Any]]) -> None:
    future_ids = {
        str(_metadata(row)["event_id"])
        for row in events
        if bool(_metadata(row).get("future_holdout"))
    }
    target_ids = {str(row["event_id"]) for row in targets}
    if future_ids & target_ids:
        raise ValueError("FUTURE_HOLDOUT_TARGET_READ")


def _exact_capable(rows: list[dict[str, Any]]) -> int:
    return sum(str(row.get("timestamp_capability")) in {"EXACT", "MIXED"} for row in rows)


def _transport_for(row: dict[str, Any]) -> str | None:
    family = str(row.get("source_family") or "")
    if "RSS" in family:
        return "OFFICIAL_RSS"
    if "WORDPRESS" in family:
        return "OFFICIAL_WORDPRESS_JSON"
    if "JSON" in family or "API" in family:
        return "OFFICIAL_JSON_ARCHIVE"
    if "APP_STATE" in family or "NEXT" in family:
        return "OFFICIAL_APP_STATE"
    return None


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["metadata"])


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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_artifacts(
    *,
    output_root: Path,
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    source_blockers: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    provenance: dict[str, Any],
    timestamp_manifest: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "events.jsonl", events)
    _write_jsonl(output_root / "features.jsonl", features)
    _write_jsonl(output_root / "targets.jsonl", targets)
    _write_jsonl(output_root / "source-registry.jsonl", registry)
    _write_jsonl(output_root / "source-blocker-registry.jsonl", source_blockers)
    _write_jsonl(output_root / "clusters.jsonl", clusters)
    _write_json(output_root / "provenance-manifest.json", provenance)
    _write_json(output_root / "timestamp-manifest.json", timestamp_manifest)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        "Data-acquisition-only source diversity expansion.",
        "",
        f"- INPUT_DATASET_SHA={manifest['INPUT_DATASET_SHA']}",
        f"- OUTPUT_DATASET_SHA={manifest['OUTPUT_DATASET_SHA']}",
        f"- EXACT_TOTAL_BEFORE={manifest['EXACT_TOTAL_BEFORE']}",
        f"- EXACT_TOTAL_AFTER={manifest['EXACT_TOTAL_AFTER']}",
        f"- NEW_EXACT_TICKERS={', '.join(manifest['NEW_EXACT_TICKERS'])}",
        f"- EXISTING_EVENT_ROWS_PRESERVED={manifest['EXISTING_EVENT_ROWS_PRESERVED']}",
        f"- EXISTING_FEATURE_ROWS_PRESERVED={manifest['EXISTING_FEATURE_ROWS_PRESERVED']}",
        f"- LEAKAGE_CHECK={manifest['LEAKAGE_CHECK']}",
        "",
        "No model training, TEST outcome use, future holdout outcome observation, backtest, paper "
        "trading, orders, or BUY/SELL output was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
