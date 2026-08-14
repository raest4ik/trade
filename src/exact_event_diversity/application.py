from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, cast

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_corpus.application import build_exact_dataset
from src.exact_event_corpus.domain import (
    FUTURE_EVENT_HOLDOUT_START,
    ExactEvent,
    ExactSourceRegistryEntry,
    TimestampCapability,
    TimezoneSemantics,
    sha256_payload,
)
from src.exact_event_diversity.domain import (
    DATASET_VERSION,
    FROZEN_V1_COUNTS,
    FROZEN_V1_HASHES,
    PARSER_VERSION,
    SOURCE_REGISTRY_VERSION,
    concentration,
    exact_model_data_status,
    feature_ready_gap,
)
from src.tinvest_market.client import TInvestReadOnlyClient

_NEW_SOURCE_OVERRIDES: dict[str, dict[str, object]] = {
    "X5": {
        "official_domain": "www.x5.ru",
        "source_url": "https://www.x5.ru/wp-json/wp/v2/news",
        "source_family": "X5_OFFICIAL_WORDPRESS_REST",
        "timestamp_capability": TimestampCapability.EXACT,
        "timestamp_field_source": "WordPress REST news.date_gmt",
        "timezone_semantics": TimezoneSemantics.EXPLICIT,
        "historical_archive_start": "2021-01-01",
        "incremental_supported": True,
    },
    "VKCO": {
        "official_domain": "vk.company",
        "source_url": "https://vk.company/ru/press/releases/",
        "source_family": "VK_OFFICIAL_NEXT_PUBLIC_STATE",
        "timestamp_capability": TimestampCapability.EXACT,
        "timestamp_field_source": "public __NEXT_DATA__.pageProps.publications.pub_date",
        "timezone_semantics": TimezoneSemantics.EXPLICIT,
        "historical_archive_start": "2014-01-01",
        "incremental_supported": True,
    },
    "T": {
        "official_domain": "cfg.tbank.ru",
        "source_url": "https://cfg.tbank.ru/about/public/api/news/platform/v1/getArticles",
        "source_family": "TBANK_OFFICIAL_PUBLIC_NEWS_API",
        "timestamp_capability": TimestampCapability.EXACT,
        "timestamp_field_source": "public getArticles response.items.publishedAt",
        "timezone_semantics": TimezoneSemantics.EXPLICIT,
        "historical_archive_start": "2022-04-28",
        "incremental_supported": True,
    },
    "BELU": {
        "official_domain": "novabev.com",
        "source_url": "https://novabev.com/en/investors/news/",
        "source_family": "NOVABEV_OFFICIAL_APP_STATE",
        "timestamp_capability": TimestampCapability.MIXED,
        "timestamp_field_source": "embedded App.news.items.activeFrom Unix epoch seconds",
        "timezone_semantics": TimezoneSemantics.EXPLICIT,
        "historical_archive_start": "2025-10-09",
        "incremental_supported": True,
    },
    "MOEX": {
        "official_domain": "www.moex.com",
        "source_url": "https://www.moex.com/export/news.aspx?cat=120",
        "source_family": "MOEX_OFFICIAL_SHAREHOLDER_RSS",
        "timestamp_capability": TimestampCapability.EXACT,
        "timestamp_field_source": "official RSS item pubDate with numeric offset",
        "timezone_semantics": TimezoneSemantics.EXPLICIT,
        "historical_archive_start": "2026-07-01",
        "incremental_supported": True,
    },
    "SMLT": {
        "official_domain": "www.moex.com",
        "source_url": "https://www.moex.com/export/news.aspx?cat=100",
        "source_family": "MOEX_OFFICIAL_ISSUER_NOTICE_RSS",
        "timestamp_capability": TimestampCapability.MIXED,
        "timestamp_field_source": "official RSS item pubDate with numeric offset",
        "timezone_semantics": TimezoneSemantics.EXPLICIT,
        "historical_archive_start": "2026-07-16",
        "incremental_supported": True,
    },
    "VTBR": {
        "official_domain": "www.moex.com",
        "source_url": "https://www.moex.com/export/news.aspx?cat=100",
        "source_family": "MOEX_OFFICIAL_ISSUER_NOTICE_RSS",
        "timestamp_capability": TimestampCapability.MIXED,
        "timestamp_field_source": "official RSS item pubDate with numeric offset",
        "timezone_semantics": TimezoneSemantics.EXPLICIT,
        "historical_archive_start": "2026-07-16",
        "incremental_supported": True,
    },
}


def build_diversity_source_registry(
    mapping_path: Path, previous_registry_path: Path
) -> tuple[ExactSourceRegistryEntry, ...]:
    mapping = cast("dict[str, Any]", json.loads(mapping_path.read_text(encoding="utf-8")))
    instruments = cast("list[dict[str, Any]]", mapping["instruments"])
    previous = {str(row["ticker"]): row for row in _read_jsonl(previous_registry_path)}
    entries: list[ExactSourceRegistryEntry] = []
    for instrument in sorted(
        instruments, key=lambda item: (str(item["ticker"]), str(item["instrument_uid"]))
    ):
        ticker = str(instrument["ticker"])
        if ticker == "IMOEX":
            continue
        old = previous.get(ticker)
        if old is None:
            raise ValueError(f"V1_SOURCE_REGISTRY_ROW_MISSING:{ticker}")
        override = _NEW_SOURCE_OVERRIDES.get(ticker, {})
        capability = TimestampCapability(
            str(override.get("timestamp_capability", old["timestamp_capability"]))
        )
        timezone = TimezoneSemantics(
            str(override.get("timezone_semantics", old["timezone_semantics"]))
        )
        entry = ExactSourceRegistryEntry(
            ticker=ticker,
            issuer=str(instrument["name"]),
            instrument_uid=str(instrument["instrument_uid"]),
            official_domain=_optional_string(
                override.get("official_domain", old.get("official_domain"))
            ),
            source_url=_optional_string(override.get("source_url", old.get("source_url"))),
            source_family=_optional_string(override.get("source_family", old.get("source_family"))),
            parser_version=(
                PARSER_VERSION if override else _optional_string(old.get("parser_version"))
            ),
            timestamp_capability=capability,
            timestamp_field_source=_optional_string(
                override.get("timestamp_field_source", old.get("timestamp_field_source"))
            ),
            timezone_semantics=timezone,
            historical_archive_start=_optional_string(
                override.get("historical_archive_start", old.get("historical_archive_start"))
            ),
            historical_archive_end=date.today().isoformat()
            if override
            else _optional_string(old.get("historical_archive_end")),
            incremental_supported=bool(
                override.get("incremental_supported", old.get("incremental_supported", False))
            ),
            public_access=True if override else bool(old.get("public_access", False)),
            payment_required=False if override else bool(old.get("payment_required", False)),
            auth_required=False if override else bool(old.get("auth_required", False)),
            source_policy_status=(
                "OFFICIAL_PUBLIC_ZERO_COST_VERIFIED"
                if override
                else str(old.get("source_policy_status", "UNKNOWN_FAIL_CLOSED"))
            ),
            collector_status="SOURCE_READY" if override else str(old.get("collector_status")),
            reason=(
                "Official public structured source exposes a verified publication timestamp"
                if override
                else str(old.get("reason", "No verified official exact-time source"))
            ),
        )
        entry.payload()
        entries.append(entry)
    if len(entries) != 315:
        raise ValueError(f"SOURCE_REGISTRY_SIZE_MISMATCH:{len(entries)}")
    return tuple(entries)


def verify_frozen_v1(v1_dir: Path) -> dict[str, object]:
    manifest = _read_json(v1_dir / "manifest.json")
    for key, expected in FROZEN_V1_HASHES.items():
        if manifest.get(key) != expected:
            raise ValueError(f"EXACT_V1_FROZEN_HASH_MISMATCH:{key}")
    event_rows = _read_jsonl(v1_dir / "events.jsonl")
    feature_rows = _read_jsonl(v1_dir / "features.jsonl")
    target_rows = _read_jsonl(v1_dir / "targets.jsonl")
    registry_rows = _read_jsonl(v1_dir / "source-registry.jsonl")
    cluster_rows = _read_jsonl(v1_dir / "clusters.jsonl")
    timestamp_manifest = _read_json(v1_dir / "timestamp-manifest.json")
    reaction_manifest = _read_json(v1_dir / "reaction-manifest.json")
    provenance = _read_json(v1_dir / "provenance-manifest.json")
    provenance_without_sha = {key: value for key, value in provenance.items() if key != "sha256"}
    computed = {
        "exact_dataset_sha": sha256_payload(
            {
                "dataset_version": "exact-event-market-dataset-v1",
                "events": event_rows,
                "features": feature_rows,
                "targets": target_rows,
            }
        ),
        "source_registry_sha": sha256_payload(registry_rows),
        "provenance_sha": sha256_payload(provenance_without_sha),
        "timestamp_manifest_sha": sha256_payload(timestamp_manifest),
        "cluster_manifest_sha": sha256_payload(cluster_rows),
        "reaction_manifest_sha": sha256_payload(reaction_manifest),
    }
    if computed != FROZEN_V1_HASHES:
        raise ValueError("EXACT_V1_ARTIFACT_CONTENT_MISMATCH")
    counts = {
        "exact_timestamp_events": len(event_rows),
        "exact_reaction_ready": sum(
            bool(cast("dict[str, object]", row["target_availability"])["reaction_ready"])
            for row in event_rows
        ),
        "exact_feature_ready": len(feature_rows),
        "exact_unique_tickers": len(
            {cast("dict[str, object]", row["metadata"])["ticker"] for row in event_rows}
        ),
        "exact_unique_issuers": len(
            {cast("dict[str, object]", row["metadata"])["issuer"] for row in event_rows}
        ),
    }
    if counts != FROZEN_V1_COUNTS:
        raise ValueError("EXACT_V1_FROZEN_COUNT_MISMATCH")
    return {"manifest": manifest, "counts": counts, "hashes": computed}


async def build_diversity_dataset(
    *,
    new_events: list[ExactEvent],
    registry: tuple[ExactSourceRegistryEntry, ...],
    client: TInvestReadOnlyClient,
    v1_dir: Path,
    output_dir: Path,
    benchmark_instrument_uid: str,
    git_sha: str,
    source_acquisition_diagnostics: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    _verify_frozen_contracts()
    verify_frozen_v1(v1_dir)
    old_event_rows = _read_jsonl(v1_dir / "events.jsonl")
    old_feature_rows = _read_jsonl(v1_dir / "features.jsonl")
    old_target_rows = _read_jsonl(v1_dir / "targets.jsonl")
    old_exclusions = _read_jsonl(v1_dir / "exclusions.jsonl")
    old_cluster_rows = _read_jsonl(v1_dir / "clusters.jsonl")
    old_timestamp = _read_json(v1_dir / "timestamp-manifest.json")
    old_reaction = _read_json(v1_dir / "reaction-manifest.json")
    incremental_events, cross_duplicates = exclude_v1_duplicates(new_events, old_event_rows)
    incremental_dir = output_dir / "_incremental"
    await build_exact_dataset(
        events=incremental_events,
        registry=registry,
        client=client,
        output_dir=incremental_dir,
        candle_cache_dir=output_dir / "raw-minute-cache",
        benchmark_instrument_uid=benchmark_instrument_uid,
        git_sha=git_sha,
    )
    incremental_event_rows = _read_jsonl(incremental_dir / "events.jsonl")
    event_rows = [*old_event_rows, *incremental_event_rows]
    feature_rows = [*old_feature_rows, *_read_jsonl(incremental_dir / "features.jsonl")]
    target_rows = [*old_target_rows, *_read_jsonl(incremental_dir / "targets.jsonl")]
    exclusions = [
        *old_exclusions,
        *cross_duplicates,
        *_read_jsonl(incremental_dir / "exclusions.jsonl"),
    ]
    cluster_rows = [*old_cluster_rows, *_read_jsonl(incremental_dir / "clusters.jsonl")]
    incremental_timestamp = _read_json(incremental_dir / "timestamp-manifest.json")
    incremental_reaction = _read_json(incremental_dir / "reaction-manifest.json")
    timestamp_manifest: dict[str, object] = {
        "schema_version": SOURCE_REGISTRY_VERSION,
        "exact_definition": "real official publication time with deterministic timezone",
        "guessed_timestamps": 0,
        "timezone_unresolved_accepted": 0,
        "events": [
            *cast("list[dict[str, object]]", old_timestamp["events"]),
            *cast("list[dict[str, object]]", incremental_timestamp["events"]),
        ],
    }
    reaction_manifest: dict[str, object] = {
        **{key: value for key, value in old_reaction.items() if key != "rows"},
        "rows": [
            *cast("list[dict[str, object]]", old_reaction["rows"]),
            *cast("list[dict[str, object]]", incremental_reaction["rows"]),
        ],
    }
    registry_rows = [
        {**entry.payload(), "source_registry_version": SOURCE_REGISTRY_VERSION}
        for entry in registry
    ]
    provenance: dict[str, object] = {
        "dataset_version": DATASET_VERSION,
        "git_sha": git_sha,
        "parent_dataset_version": "exact-event-market-dataset-v1",
        "frozen_v1_hashes": FROZEN_V1_HASHES,
        "news_sources": "OFFICIAL_ZERO_COST_PUBLIC_SOURCES_ONLY",
        "market_source": "TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES",
        "market_source_cost_rub": 0,
        "source_selection_used_future_returns": False,
        "source_selection_used_target_metrics": False,
        "raw_full_text_redistributed": False,
        "rules_v3_fingerprint": rules_v3_fingerprint(),
        "qwen_prompt_sha": prompt_hash(),
        "qwen_schema_sha": schema_hash(),
    }
    assert_v1_preserved(old_event_rows, event_rows, old_feature_rows, feature_rows)
    assert_rows_preserved(old_target_rows, target_rows, artifact="targets")
    assert_rows_preserved(old_cluster_rows, cluster_rows, artifact="clusters")
    assert_holdout_guard(event_rows, target_rows)
    report = _build_report(
        event_rows=event_rows,
        feature_rows=feature_rows,
        target_rows=target_rows,
        exclusions=exclusions,
        cluster_rows=cluster_rows,
        registry_rows=registry_rows,
        timestamp_manifest=timestamp_manifest,
        reaction_manifest=reaction_manifest,
        provenance=provenance,
        git_sha=git_sha,
        source_acquisition_diagnostics=source_acquisition_diagnostics or [],
    )
    await asyncio.to_thread(
        _write_v2_artifacts,
        output_dir,
        event_rows,
        feature_rows,
        target_rows,
        exclusions,
        registry_rows,
        cluster_rows,
        timestamp_manifest,
        reaction_manifest,
        {**provenance, "sha256": report["provenance_sha"]},
        report,
    )
    return report


def _build_report(
    *,
    event_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    exclusions: list[dict[str, object]],
    cluster_rows: list[dict[str, object]],
    registry_rows: list[dict[str, object]],
    timestamp_manifest: dict[str, object],
    reaction_manifest: dict[str, object],
    provenance: dict[str, object],
    git_sha: str,
    source_acquisition_diagnostics: list[dict[str, object]],
) -> dict[str, object]:
    metadata_by_id = {
        str(cast("dict[str, object]", row["metadata"])["event_id"]): cast(
            "dict[str, object]", row["metadata"]
        )
        for row in event_rows
    }
    ticker_counts = Counter(str(row["ticker"]) for row in metadata_by_id.values())
    issuer_counts = Counter(str(row["issuer"]) for row in metadata_by_id.values())
    source_by_ticker = {
        str(row["ticker"]): str(row.get("source_family") or "UNKNOWN") for row in registry_rows
    }
    source_counts = Counter(
        source_by_ticker[str(metadata["ticker"])] for metadata in metadata_by_id.values()
    )
    feature_tickers = Counter(
        str(metadata_by_id[str(row["event_id"])]["ticker"]) for row in feature_rows
    )
    session_counts = Counter(str(row["session_state"]) for row in metadata_by_id.values())
    type_counts: Counter[str] = Counter()
    unknown_by_ticker: Counter[str] = Counter()
    unknown_by_source: Counter[str] = Counter()
    unknown_by_year: Counter[str] = Counter()
    event_type_detail: dict[str, dict[str, set[str] | int]] = {}
    for row in event_rows:
        metadata = cast("dict[str, object]", row["metadata"])
        features = cast("dict[str, object]", row["event_features"])
        event_type = str(features["primary_event_type"])
        type_counts[event_type] += 1
        detail = event_type_detail.setdefault(
            event_type,
            {
                "count": 0,
                "tickers": set(),
                "issuers": set(),
                "source_families": set(),
                "years": set(),
            },
        )
        detail["count"] = cast("int", detail["count"]) + 1
        cast("set[str]", detail["tickers"]).add(str(metadata["ticker"]))
        cast("set[str]", detail["issuers"]).add(str(metadata["issuer"]))
        cast("set[str]", detail["source_families"]).add(source_by_ticker[str(metadata["ticker"])])
        cast("set[str]", detail["years"]).add(str(metadata["publication_date"])[:4])
        if event_type == "UNKNOWN":
            unknown_by_ticker[str(metadata["ticker"])] += 1
            unknown_by_source[source_by_ticker[str(metadata["ticker"])]] += 1
            unknown_by_year[str(metadata["publication_date"])[:4]] += 1
    event_type_diversity = {
        event_type: {
            "exact_count": detail["count"],
            "unique_tickers": len(cast("set[str]", detail["tickers"])),
            "unique_issuers": len(cast("set[str]", detail["issuers"])),
            "source_families": sorted(cast("set[str]", detail["source_families"])),
            "years": sorted(cast("set[str]", detail["years"])),
        }
        for event_type, detail in sorted(event_type_detail.items())
    }
    target_by_id = {str(row["event_id"]): row for row in target_rows}
    horizon_ready = {
        f"reaction_ready_{horizon}m": sum(
            bool(
                cast(
                    "dict[str, object]",
                    cast("dict[str, object]", target_by_id[event_id]["horizons"]).get(
                        f"{horizon}m", {}
                    ),
                ).get("available", False)
            )
            for event_id in target_by_id
        )
        for horizon in (1, 5, 15, 30, 60)
    }
    reaction_ready = sum(
        bool(cast("dict[str, object]", row["target_availability"])["reaction_ready"])
        for row in event_rows
    )
    gap = feature_ready_gap(event_rows)
    future_rows = [
        row
        for row in event_rows
        if bool(cast("dict[str, object]", row["metadata"])["future_holdout"])
    ]
    dataset_sha = sha256_payload(
        {
            "dataset_version": DATASET_VERSION,
            "events": event_rows,
            "features": feature_rows,
            "targets": target_rows,
        }
    )
    registry_sha = sha256_payload(registry_rows)
    provenance_sha = sha256_payload(provenance)
    timestamp_sha = sha256_payload(timestamp_manifest)
    reaction_sha = sha256_payload(reaction_manifest)
    cluster_sha = sha256_payload(cluster_rows)
    exact_capable = [
        row for row in registry_rows if row["timestamp_capability"] in {"EXACT", "MIXED"}
    ]
    exact_source_families = sorted({str(row["source_family"]) for row in exact_capable})
    new_source_families = sorted(
        {str(row["source_family"]) for row in _NEW_SOURCE_OVERRIDES.values()}
    )
    model_status = exact_model_data_status(
        feature_ready=len(feature_rows), feature_ready_by_ticker=feature_tickers
    )
    return {
        "dataset_version": DATASET_VERSION,
        "git_sha": git_sha,
        "old_exact_timestamp_events": FROZEN_V1_COUNTS["exact_timestamp_events"],
        "exact_timestamp_events": len(event_rows),
        "exact_event_growth": len(event_rows) - FROZEN_V1_COUNTS["exact_timestamp_events"],
        "old_exact_reaction_ready": FROZEN_V1_COUNTS["exact_reaction_ready"],
        "exact_reaction_ready": reaction_ready,
        "old_exact_feature_ready": FROZEN_V1_COUNTS["exact_feature_ready"],
        "exact_feature_ready": len(feature_rows),
        "old_exact_unique_tickers": FROZEN_V1_COUNTS["exact_unique_tickers"],
        "exact_unique_tickers": len(ticker_counts),
        "ticker_diversity_growth": len(ticker_counts) - FROZEN_V1_COUNTS["exact_unique_tickers"],
        "old_exact_unique_issuers": FROZEN_V1_COUNTS["exact_unique_issuers"],
        "exact_unique_issuers": len(issuer_counts),
        "new_exact_capable_official_sources": len(exact_capable) - 4,
        "exact_capable_official_sources": len(exact_capable),
        "exact_source_families": exact_source_families,
        "exact_source_family_count": len(exact_source_families),
        "new_exact_source_families": new_source_families,
        "exact_source_registry_summary": {
            "rows": len(registry_rows),
            "timestamp_capability": dict(
                sorted(Counter(str(row["timestamp_capability"]) for row in registry_rows).items())
            ),
            "collector_status": dict(
                sorted(Counter(str(row["collector_status"]) for row in registry_rows).items())
            ),
        },
        "source_acquisition_diagnostics": source_acquisition_diagnostics,
        "events_per_ticker": dict(sorted(ticker_counts.items())),
        "feature_ready_per_ticker": dict(sorted(feature_tickers.items())),
        "tickers_with_at_least_10_feature_ready": sorted(
            ticker for ticker, count in feature_tickers.items() if count >= 10
        ),
        "tickers_with_at_least_25_feature_ready": sorted(
            ticker for ticker, count in feature_tickers.items() if count >= 25
        ),
        "ticker_concentration": concentration(ticker_counts),
        "issuer_concentration": concentration(issuer_counts),
        "source_concentration": concentration(source_counts),
        "PRE_OPEN_exact_events": session_counts["PRE_OPEN"],
        "DURING_SESSION_exact_events": session_counts["DURING_MAIN_SESSION"],
        "AFTER_CLOSE_exact_events": session_counts["AFTER_CLOSE"],
        "NON_TRADING_DAY_exact_events": session_counts["NON_TRADING_DAY"],
        "OTHER_UNKNOWN_session_events": session_counts["OTHER/UNKNOWN"],
        **horizon_ready,
        "reaction_ready_but_not_feature_ready": gap["count"],
        "feature_ready_missing_reasons": gap["reasons"],
        "UNKNOWN_event_count": type_counts["UNKNOWN"],
        "UNKNOWN_event_share": type_counts["UNKNOWN"] / len(event_rows) if event_rows else 0.0,
        "UNKNOWN_by_ticker": dict(sorted(unknown_by_ticker.items())),
        "UNKNOWN_by_source_family": dict(sorted(unknown_by_source.items())),
        "UNKNOWN_by_year": dict(sorted(unknown_by_year.items())),
        "event_type_diversity": event_type_diversity,
        "events_per_event_type": dict(sorted(type_counts.items())),
        "duplicate_update_diagnostics": dict(
            sorted(Counter(str(row["reason"]) for row in exclusions).items())
        ),
        "cluster_count": len({str(row["event_cluster_id"]) for row in cluster_rows}),
        "EXACT_V1_PRESERVED": "YES",
        "frozen_v1_hashes": FROZEN_V1_HASHES,
        "exact_dataset_sha": dataset_sha,
        "source_registry_sha": registry_sha,
        "provenance_sha": provenance_sha,
        "timestamp_manifest_sha": timestamp_sha,
        "reaction_manifest_sha": reaction_sha,
        "cluster_manifest_sha": cluster_sha,
        "EVENT_MARKET_LEAKAGE_CHECK": "PASS",
        "EXACT_VOLUME_STATUS": (
            "EXACT_VOLUME_READY" if len(feature_rows) >= 250 else "EXACT_VOLUME_NOT_READY"
        ),
        "EXACT_DIVERSITY_STATUS": (
            "EXACT_DIVERSITY_READY" if len(ticker_counts) >= 10 else "EXACT_DIVERSITY_NOT_READY"
        ),
        "EXACT_MODEL_DATA_STATUS": model_status,
        "FUTURE_EVENT_HOLDOUT_START": FUTURE_EVENT_HOLDOUT_START.isoformat(),
        "future_holdout_event_count": len(future_rows),
        "future_exact_event_count": len(future_rows),
        "future_unique_tickers": len(
            {cast("dict[str, object]", row["metadata"])["ticker"] for row in future_rows}
        ),
        "future_exact_tickers": sorted(
            {str(cast("dict[str, object]", row["metadata"])["ticker"]) for row in future_rows}
        ),
        "future_unique_issuers": len(
            {cast("dict[str, object]", row["metadata"])["issuer"] for row in future_rows}
        ),
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "holdout_guard": "PASS",
        "outcome_fields_exported_for_future": 0,
        "rules_changed": False,
        "qwen_changed": False,
        "qwen_run": False,
        "NLP_FROZEN": True,
        "model_trained": False,
        "abc_evaluated": False,
        "backtest_executed": False,
        "paper_trading_executed": False,
        "orders_submitted": False,
        "buy_sell_generated": False,
        "real_trading_executed": False,
        "paid_services_used": False,
    }


def exclude_v1_duplicates(
    events: list[ExactEvent], old_rows: list[dict[str, object]]
) -> tuple[list[ExactEvent], list[dict[str, object]]]:
    identities = {
        (
            str(cast("dict[str, object]", row["metadata"])["source_code"]),
            str(cast("dict[str, object]", row["metadata"])["source_item_id"]),
        )
        for row in old_rows
    }
    urls = {str(cast("dict[str, object]", row["metadata"])["canonical_url"]) for row in old_rows}
    selected: list[ExactEvent] = []
    dropped: list[dict[str, object]] = []
    for event in events:
        if (event.source_code, event.source_item_id) in identities or event.canonical_url in urls:
            dropped.append(
                {
                    "event_id": str(event.event_id),
                    "ticker": event.ticker,
                    "reason": "EXACT_V1_DUPLICATE_PRESERVED",
                }
            )
        else:
            selected.append(event)
    return selected, dropped


def assert_v1_preserved(
    old_events: list[dict[str, object]],
    new_events: list[dict[str, object]],
    old_features: list[dict[str, object]],
    new_features: list[dict[str, object]],
) -> None:
    assert_rows_preserved(old_events, new_events, artifact="events")
    assert_rows_preserved(old_features, new_features, artifact="features")


def assert_rows_preserved(
    old_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
    *,
    artifact: str,
) -> None:
    if new_rows[: len(old_rows)] != old_rows:
        raise ValueError(f"EXACT_V1_NOT_PRESERVED:{artifact}")


def assert_holdout_guard(
    event_rows: list[dict[str, object]], target_rows: list[dict[str, object]]
) -> None:
    future_ids = {
        str(cast("dict[str, object]", row["metadata"])["event_id"])
        for row in event_rows
        if bool(cast("dict[str, object]", row["metadata"])["future_holdout"])
    }
    target_ids = {str(row["event_id"]) for row in target_rows}
    if future_ids & target_ids:
        raise ValueError("FUTURE_EVENT_HOLDOUT_READ_ATTEMPT")
    for row in event_rows:
        metadata = cast("dict[str, object]", row["metadata"])
        availability = cast("dict[str, object]", row["target_availability"])
        if metadata["event_id"] in future_ids and availability["research_outcomes_visible"]:
            raise ValueError("FUTURE_EVENT_HOLDOUT_READ_ATTEMPT")


def _verify_frozen_contracts() -> None:
    if rules_v3_fingerprint() != EXPECTED_RULES_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_MISMATCH")
    if prompt_hash() != QWEN_PROMPT_SHA or schema_hash() != QWEN_SCHEMA_SHA:
        raise ValueError("FROZEN_QWEN_CONTRACT_MISMATCH")


def _write_v2_artifacts(
    output_dir: Path,
    event_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    target_rows: list[dict[str, object]],
    exclusions: list[dict[str, object]],
    registry_rows: list[dict[str, object]],
    cluster_rows: list[dict[str, object]],
    timestamp_manifest: dict[str, object],
    reaction_manifest: dict[str, object],
    provenance: dict[str, object],
    report: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "events.jsonl", event_rows)
    _write_jsonl(output_dir / "features.jsonl", feature_rows)
    _write_jsonl(output_dir / "targets.jsonl", target_rows)
    _write_jsonl(output_dir / "exclusions.jsonl", exclusions)
    _write_jsonl(output_dir / "source-registry.jsonl", registry_rows)
    _write_jsonl(output_dir / "clusters.jsonl", cluster_rows)
    _write_json(output_dir / "timestamp-manifest.json", timestamp_manifest)
    _write_json(output_dir / "reaction-manifest.json", reaction_manifest)
    _write_json(output_dir / "provenance-manifest.json", provenance)
    _write_json(
        output_dir / "feature-ready-gap.json",
        {
            "count": report["reaction_ready_but_not_feature_ready"],
            "reasons": report["feature_ready_missing_reasons"],
        },
    )
    _write_json(
        output_dir / "concentration-diagnostics.json",
        {
            "ticker": report["ticker_concentration"],
            "issuer": report["issuer_concentration"],
            "source": report["source_concentration"],
        },
    )
    _write_json(
        output_dir / "unknown-event-diagnostic.json",
        {
            "count": report["UNKNOWN_event_count"],
            "share": report["UNKNOWN_event_share"],
            "by_ticker": report["UNKNOWN_by_ticker"],
            "by_source_family": report["UNKNOWN_by_source_family"],
            "by_year": report["UNKNOWN_by_year"],
            "market_outcomes_used": False,
        },
    )
    _write_json(output_dir / "event-type-diversity.json", report["event_type_diversity"])
    _write_json(
        output_dir / "future-holdout-status.json",
        {
            key: report[key]
            for key in (
                "FUTURE_EVENT_HOLDOUT_START",
                "future_holdout_event_count",
                "future_exact_event_count",
                "future_unique_tickers",
                "future_exact_tickers",
                "future_unique_issuers",
                "FUTURE_EVENT_HOLDOUT_OBSERVED",
                "holdout_guard",
                "outcome_fields_exported_for_future",
            )
        },
    )
    _write_json(output_dir / "manifest.json", report)


def _read_json(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        cast("dict[str, object]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
