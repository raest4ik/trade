from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.domain import (
    DATASET_VERSION,
    EVENT_RULES_VERSION,
    FACTS_VERSION,
    MARKET_CONTEXT_VERSION,
    PREDICTIVE_UNIT,
    QWEN_PROMPT_SHA,
    QWEN_SCHEMA_SHA,
    REACTION_DAILY,
    REACTION_EXACT,
    AcquiredEvent,
    EventMarketRow,
    EventSourceRegistryEntry,
    SourceRegistryStatus,
    classify_reaction,
    deduplicate_events,
    event_feature_names,
    readiness,
    sha256_payload,
)
from src.event_market_dataset.sources import ArchiveSourceConfig, acquire_archive
from src.events.domain.enums import EventType
from src.events.domain.v3 import EventAnalyzerV3, rules_v3_fingerprint
from src.news.domain.enums import PublicationTimestampQuality

EXPECTED_RULES_FINGERPRINT = "3510511d1f7b3ce02a4efa245816b9422e6014088f1595b0339dcfd5be9e7f06"
EXPECTED_FEATURE_SCHEMA_SHA = "f7a60ecf55d7d0f7d455035810312224a30ee637a3bab2dfede231ca9dc0bb45"


def build_source_registry(
    mapping_path: Path,
    *,
    checked_on: date,
) -> tuple[EventSourceRegistryEntry, ...]:
    payload = cast("dict[str, Any]", json.loads(mapping_path.read_text(encoding="utf-8")))
    instruments = cast("list[dict[str, Any]]", payload["instruments"])
    known = {
        "ROSN": {
            "url": "https://www.rosneft.com/press/releases/rss/",
            "name": "Rosneft Press Releases RSS",
            "owner": "Rosneft Oil Company",
            "type": "ISSUER_RSS",
            "method": "single bounded RSS page",
            "history": "YES",
            "range": "current 20-item feed plus official dated archive",
            "live": True,
        },
        "YDEX": {
            "url": "https://ir.yandex.ru/press-releases",
            "name": "Yandex Investor Relations Press Releases",
            "owner": "Yandex",
            "type": "ISSUER_RSS_AND_ARCHIVE",
            "method": "single RSS page plus explicit bounded year pages",
            "history": "YES",
            "range": "YDEX identity accepted from 2024-07-24; older YNDX rows excluded",
            "live": True,
        },
        "NVTK": {
            "url": "https://www.novatek.ru/en/press/releases/",
            "name": "NOVATEK Press Releases and Events",
            "owner": "PAO NOVATEK",
            "type": "ISSUER_ARCHIVE",
            "method": "bounded numbered archive pages",
            "history": "YES",
            "range": "multi-year official archive",
            "live": False,
        },
    }
    entries: list[EventSourceRegistryEntry] = []
    ordered = sorted(
        instruments,
        key=lambda value: (str(value["ticker"]), str(value["instrument_uid"])),
    )
    for item in ordered:
        ticker = str(item["ticker"])
        if ticker == "IMOEX":
            continue
        source = known.get(ticker)
        if source is None:
            entries.append(
                EventSourceRegistryEntry(
                    ticker=ticker,
                    issuer_name=str(item["name"]),
                    instrument_uid=str(item["instrument_uid"]),
                    figi=_optional(item.get("figi")),
                    official_source_url=None,
                    source_name=None,
                    source_type=None,
                    official_owner=None,
                    collection_method=None,
                    history_available="UNKNOWN",
                    historical_range=None,
                    live_supported=False,
                    status=SourceRegistryStatus.NO_OFFICIAL_SOURCE_FOUND,
                    reason=(
                        "No verified official free source URL is registered; no URL was guessed"
                    ),
                    public_access=False,
                    payment_required=False,
                    authentication_required=False,
                    robots_rate_limit_notes=(
                        "Not checked because no official source contract is known"
                    ),
                    redistribution_status="UNKNOWN_FAIL_CLOSED",
                    internal_research_use_status="NOT_APPROVED",
                    first_seen=checked_on.isoformat(),
                    last_checked=checked_on.isoformat(),
                )
            )
            continue
        entries.append(
            EventSourceRegistryEntry(
                ticker=ticker,
                issuer_name=str(item["name"]),
                instrument_uid=str(item["instrument_uid"]),
                figi=_optional(item.get("figi")),
                official_source_url=str(source["url"]),
                source_name=str(source["name"]),
                source_type=str(source["type"]),
                official_owner=str(source["owner"]),
                collection_method=str(source["method"]),
                history_available=str(source["history"]),
                historical_range=str(source["range"]),
                live_supported=bool(source["live"]),
                status=SourceRegistryStatus.SOURCE_READY,
                reason="Official issuer-owned public source with bounded zero-cost access",
                public_access=True,
                payment_required=False,
                authentication_required=False,
                robots_rate_limit_notes=("Bounded requests, normal rate, no access-control bypass"),
                redistribution_status="RAW_FULL_TEXT_NOT_REDISTRIBUTED",
                internal_research_use_status="PRIVATE_INTERNAL_RESEARCH_APPROVED",
                first_seen=checked_on.isoformat(),
                last_checked=checked_on.isoformat(),
            )
        )
    return tuple(entries)


async def acquire_new_events(
    registry: tuple[EventSourceRegistryEntry, ...],
    *,
    date_from: date,
    date_to: date,
    per_source_limit: int,
    cache_dir: Path,
    allow_partial_sources: bool = False,
) -> tuple[list[AcquiredEvent], list[dict[str, object]]]:
    by_ticker = {item.ticker: item for item in registry}
    configs = (
        ArchiveSourceConfig(
            source_code="YANDEX_IR_ARCHIVE_DATE_ONLY",
            source_name="Yandex Investor Relations Press Releases",
            official_owner="Yandex",
            ticker="YDEX",
            issuer_name=by_ticker["YDEX"].issuer_name,
            instrument_uid=by_ticker["YDEX"].instrument_uid,
            figi=by_ticker["YDEX"].figi,
            url_template="https://ir.yandex.ru/press-releases?year={page}",
            page_values=tuple(range(max(date_from.year, 2024), date_to.year + 1)),
            source_type="ISSUER_ARCHIVE",
            collection_method="explicit bounded year pages",
            historical_range=f"{date_from.year}-{date_to.year}",
            live_supported=False,
        ),
        ArchiveSourceConfig(
            source_code="NOVATEK_PRESS_RELEASE_ARCHIVE_DATE_ONLY",
            source_name="NOVATEK Press Releases and Events",
            official_owner="PAO NOVATEK",
            ticker="NVTK",
            issuer_name=by_ticker["NVTK"].issuer_name,
            instrument_uid=by_ticker["NVTK"].instrument_uid,
            figi=by_ticker["NVTK"].figi,
            url_template="https://www.novatek.ru/en/press/releases/?from_4={page}",
            page_values=tuple(range(1, 13)),
            source_type="ISSUER_ARCHIVE",
            collection_method="bounded numbered archive pages",
            historical_range=f"{date_from.isoformat()}..{date_to.isoformat()}",
            live_supported=False,
        ),
    )
    acquired: list[AcquiredEvent] = []
    errors: list[dict[str, object]] = []
    for config in configs:
        try:
            source_date_from = date_from
            if config.source_code == "YANDEX_IR_ARCHIVE_DATE_ONLY":
                source_date_from = max(date_from, date(2024, 7, 24))
                if date_from < source_date_from:
                    errors.append(
                        {
                            "source_code": config.source_code,
                            "reason": "TICKER_IDENTITY_RANGE_EXCLUDED",
                            "excluded_date_from": date_from.isoformat(),
                            "excluded_date_to": "2024-07-23",
                            "official_mapping_basis": (
                                "YDEX Moscow Exchange trading began 2024-07-24"
                            ),
                        }
                    )
            acquired.extend(
                await acquire_archive(
                    config,
                    date_from=source_date_from,
                    date_to=date_to,
                    limit=per_source_limit,
                    cache_dir=cache_dir / config.source_code,
                )
            )
        except RuntimeError as exc:
            error: dict[str, object] = {
                "source_code": config.source_code,
                "reason": str(exc),
            }
            errors.append(error)
            if not allow_partial_sources:
                raise RuntimeError(f"SOURCE_ACQUISITION_FAILED:{config.source_code}:{exc}") from exc
    return acquired, errors


def build_dataset(
    *,
    acquired_events: list[AcquiredEvent],
    existing_events: list[AcquiredEvent],
    existing_texts: dict[str, str],
    old_corpus_path: Path,
    old_manifest_path: Path,
    market_feature_path: Path,
    market_manifest_path: Path,
    raw_series_dir: Path,
    instrument_mapping_path: Path,
    output_dir: Path,
    source_registry: tuple[EventSourceRegistryEntry, ...],
    source_errors: list[dict[str, object]],
    git_sha: str,
) -> dict[str, object]:
    if rules_v3_fingerprint() != EXPECTED_RULES_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_MISMATCH")
    if prompt_hash() != QWEN_PROMPT_SHA or schema_hash() != QWEN_SCHEMA_SHA:
        raise ValueError("FROZEN_QWEN_CONTRACT_MISMATCH")
    market_manifest = _read_json(market_manifest_path)
    if market_manifest.get("feature_schema_sha") != EXPECTED_FEATURE_SCHEMA_SHA:
        raise ValueError("MARKET_FEATURE_SCHEMA_MISMATCH")
    old_manifest = _read_json(old_manifest_path)
    market = _load_market_features(market_feature_path)
    identities = _load_instrument_identities(instrument_mapping_path)
    analyzer = EventAnalyzerV3()
    old_features, old_targets = _load_old_exact(
        old_corpus_path,
        market=market,
        identities=identities,
        existing_texts=existing_texts,
        analyzer=analyzer,
    )
    new_events, dedupe_drops = deduplicate_events(
        acquired_events,
        existing_events=existing_events,
    )
    series = _load_series(raw_series_dir, {"YDEX", "NVTK", "IMOEX"})
    benchmark = series["IMOEX"]
    built_rows: list[EventMarketRow] = []
    targets: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = list(dedupe_drops)
    for event in new_events:
        context = market_context(market, event.ticker, event.publication_date)
        if context is None:
            exclusions.append(_exclude(event, "PRE_EVENT_MARKET_CONTEXT_MISSING"))
            continue
        security = series.get(event.ticker)
        if security is None:
            exclusions.append(_exclude(event, "SECURITY_DAILY_SERIES_MISSING"))
            continue
        target = daily_target(event, security, benchmark)
        if target is None:
            exclusions.append(_exclude(event, "DATE_SAFE_REACTION_WINDOW_MISSING"))
            continue
        analysis = analyzer.analyze(news_id=event.event_id, raw_content=event.title)
        event_features: dict[str, object] = {
            "primary_event_type": analysis.primary_event_type.value,
            "event_count": len(analysis.events),
            "fact_count": len(analysis.financial_facts),
        }
        event_features.update(
            {
                f"event_type_{item.value.lower()}": any(
                    detected.event_type == item for detected in analysis.events
                )
                for item in EventType
            }
        )
        cutoff, market_values = context
        built_rows.append(
            EventMarketRow(
                event=event,
                reaction_family=REACTION_DAILY,
                market_context_cutoff=cutoff,
                event_features=event_features,
                market_features=market_values,
                semantics={
                    "analysis_status": analysis.status.value,
                    "primary_event_type": analysis.primary_event_type.value,
                    "rule_version": EVENT_RULES_VERSION,
                    "financial_facts_version": FACTS_VERSION,
                    "qwen_used": False,
                },
            )
        )
        targets.append(target)

    feature_payloads = [*old_features, *(item.feature_payload() for item in built_rows)]
    target_payloads = [*old_targets, *targets]
    _reconcile_features_targets(feature_payloads, target_payloads)
    if not all(leakage_pass(item) for item in feature_payloads):
        raise ValueError("EVENT_MARKET_LEAKAGE_CHECK_FAILED")

    registry_payload = [item.payload() for item in source_registry]
    source_registry_sha = sha256_payload(registry_payload)
    feature_schema = _feature_schema(feature_payloads)
    feature_schema_sha = sha256_payload(feature_schema)
    dataset_sha = sha256_payload(
        {
            "dataset_version": DATASET_VERSION,
            "features": feature_payloads,
            "targets": target_payloads,
        }
    )
    old_total = int(old_manifest["REAL_discovered"])
    old_matched = int(old_manifest["matched"])
    old_exact_ready = int(old_manifest["reaction_ready"])
    old_feature_ready = int(old_manifest["feature_ready"])
    total_real = old_total + len(new_events)
    matched_total = old_matched + len(new_events)
    exact_total = int(old_manifest["REAL_EXACT"])
    date_only_total = len(new_events)
    date_safe_ready = len(built_rows)
    feature_ready = len(feature_payloads)
    ticker_counts = Counter(cast("dict[str, int]", old_manifest["ticker_distribution"]))
    ticker_counts.update(event.ticker for event in new_events)
    source_counts = Counter(cast("dict[str, int]", old_manifest["source_distribution"]))
    source_counts.update(event.source_code for event in new_events)
    year_counts = Counter(
        month[:4]
        for month, count in cast("dict[str, int]", old_manifest["month_distribution"]).items()
        for _ in range(count)
    )
    year_counts.update(str(event.publication_date.year) for event in new_events)
    all_dates = [event.publication_date for event in existing_events]
    all_dates.extend(event.publication_date for event in new_events)
    ready = readiness(feature_ready, len(ticker_counts))
    provenance = {
        "dataset_version": DATASET_VERSION,
        "git_sha": git_sha,
        "source_registry_sha": source_registry_sha,
        "market_dataset_sha": market_manifest["raw_dataset_sha"],
        "market_feature_sha": market_manifest["feature_sha"],
        "market_feature_schema_sha": market_manifest["feature_schema_sha"],
        "old_corpus_sha": sha256_payload(old_manifest),
        "source_errors": source_errors,
        "external_paid_data_cost_rub": 0,
        "raw_full_text_redistributed": False,
        "moex_dataset_used": False,
        "tinvest_source_usage": "PRIVATE_INTERNAL_USE_CONFIRMED",
        "retrieval_policy": "OFFICIAL_ISSUER_OWNED_BOUNDED_ONLY",
    }
    provenance_sha = sha256_payload(provenance)
    report: dict[str, object] = {
        "dataset_version": DATASET_VERSION,
        "git_sha": git_sha,
        "issuer_universe_considered": len(source_registry),
        "official_sources_discovered": sum(
            item.official_source_url is not None for item in source_registry
        ),
        "source_ready_count": sum(
            item.status == SourceRegistryStatus.SOURCE_READY for item in source_registry
        ),
        "source_families_implemented": [
            "ROSNEFT_PRESS_RELEASES_RSS",
            "YANDEX_IR_PRESS_RELEASES_RSS_AND_ARCHIVE",
            "NOVATEK_PRESS_RELEASE_ARCHIVE_DATE_ONLY",
        ],
        "source_policy_blocked_count": sum(
            item.status == SourceRegistryStatus.BLOCKED_BY_SOURCE_POLICY for item in source_registry
        ),
        "old_total_real_events": old_total,
        "old_reaction_ready": old_exact_ready,
        "old_feature_ready": old_feature_ready,
        "old_unique_tickers": len(cast("dict[str, int]", old_manifest["ticker_distribution"])),
        "new_total_real_events": total_real,
        "event_corpus_growth_absolute": total_real - old_total,
        "event_corpus_growth_percent": (total_real - old_total) / old_total * 100.0,
        "exact_timestamp_events": exact_total,
        "date_only_events": date_only_total,
        "unverified_events": 0,
        "ticker_matched_events": matched_total,
        "ticker_ambiguous_unresolved": int(old_manifest["ambiguous"])
        + int(old_manifest["unmatched"]),
        "reaction_ready_exact": old_exact_ready,
        "reaction_ready_date_safe": date_safe_ready,
        "event_market_feature_ready": feature_ready,
        "unique_tickers": len(ticker_counts),
        "unique_issuers": len(ticker_counts),
        "event_date_from": min(all_dates).isoformat() if all_dates else None,
        "event_date_to": max(all_dates).isoformat() if all_dates else None,
        "events_per_ticker": dict(sorted(ticker_counts.items())),
        "events_per_year": dict(sorted(year_counts.items())),
        "events_per_source": dict(sorted(source_counts.items())),
        "event_market_dataset_sha": dataset_sha,
        "source_registry_sha": source_registry_sha,
        "provenance_manifest_sha": provenance_sha,
        "feature_schema_sha": feature_schema_sha,
        "event_market_leakage_check": "PASS",
        "predictive_unit": PREDICTIVE_UNIT,
        "market_only_daily_rows_as_event_examples": False,
        **ready,
        "market_only_baseline_status": "FROZEN_NEGATIVE_BASELINE",
        "rules_v3_fingerprint": rules_v3_fingerprint(),
        "rules_changed": False,
        "qwen_changed": False,
        "qwen_prompt_sha": prompt_hash(),
        "qwen_schema_sha": schema_hash(),
        "live_collector_preserved": True,
        "model_trained": False,
        "model_selection": False,
        "hyperparameter_tuning": False,
        "observed_market_test_used": False,
        "future_holdout_evaluated": False,
        "backtest_executed": False,
        "paper_trading_executed": False,
        "production_order_executed": False,
        "sandbox_order_executed": False,
        "buy_sell_generated": False,
        "real_trading_allowed": False,
        "paid_services_used": False,
        "exclusion_count": len(exclusions),
        "exclusions_by_reason": dict(
            sorted(Counter(str(item["reason"]) for item in exclusions).items())
        ),
        "grouping_diagnostics": _clusters(feature_payloads),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "features.jsonl", feature_payloads)
    _write_jsonl(output_dir / "targets.jsonl", target_payloads)
    _write_jsonl(output_dir / "exclusions.jsonl", exclusions)
    _write_jsonl(output_dir / "source-registry.jsonl", registry_payload)
    _write_json(output_dir / "feature-schema.json", feature_schema)
    _write_json(
        output_dir / "provenance-manifest.json",
        {**provenance, "provenance_manifest_sha": provenance_sha},
    )
    _write_json(output_dir / "manifest.json", report)
    return report


def _load_old_exact(
    path: Path,
    *,
    market: dict[str, list[tuple[date, datetime, dict[str, float]]]],
    identities: dict[str, dict[str, str | None]],
    existing_texts: dict[str, str],
    analyzer: EventAnalyzerV3,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    features: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = cast("dict[str, Any]", json.loads(line))
        metadata = cast("dict[str, Any]", row["metadata"])
        quality = cast("dict[str, Any]", row["quality"])
        event_id = str(metadata["news_id"])
        published = datetime.fromisoformat(str(metadata["published_at"]).replace(" ", "T"))
        ticker = str(metadata["ticker"])
        context = market_context(market, ticker, published.date())
        if context is None:
            raise ValueError(f"OLD_EVENT_DAILY_CONTEXT_MISSING:{event_id}")
        _daily_cutoff, daily_features = context
        observation_ends = [
            datetime.fromisoformat(str(quality[key]).replace(" ", "T"))
            for key in ("security_observation_end_at", "benchmark_observation_end_at")
        ]
        cutoff = max(observation_ends)
        if not cutoff < published:
            raise ValueError(f"OLD_EVENT_POINT_IN_TIME_VIOLATION:{event_id}")
        text = existing_texts.get(event_id)
        if text is None:
            raise ValueError(f"OLD_EVENT_TEXT_MISSING:{event_id}")
        analysis = analyzer.analyze(news_id=_uuid(event_id), raw_content=text)
        event_features: dict[str, object] = {
            "primary_event_type": analysis.primary_event_type.value,
            "event_count": len(analysis.events),
            "fact_count": len(analysis.financial_facts),
        }
        event_features.update(
            {
                f"event_type_{item.value.lower()}": any(
                    detected.event_type == item for detected in analysis.events
                )
                for item in EventType
            }
        )
        old_values = cast("dict[str, object]", row["features_available_at_publication"])
        intraday_features = {
            key: _numeric(value)
            for key, value in old_values.items()
            if key.startswith(("pre_", "imoex_pre_", "realized_", "volume_")) and value is not None
        }
        instrument = identities[ticker]
        feature: dict[str, object] = {
            "metadata": {
                "event_id": event_id,
                "source_code": metadata["source"],
                "source_item_id": metadata["source_item_id"],
                "canonical_url": metadata["source_item_id"],
                "ticker": ticker,
                "issuer_name": instrument["name"],
                "instrument_uid": instrument["instrument_uid"],
                "figi": instrument["figi"],
                "publication_date": published.date().isoformat(),
                "published_at": published.isoformat(),
                "publication_time_quality": PublicationTimestampQuality.EXACT.value,
                "storage_policy": "EXCERPT_ALLOWED",
                "source_rights_status": "PRIVATE_INTERNAL_RESEARCH_ONLY",
                "title_hash": None,
                "dataset_version": DATASET_VERSION,
                "predictive_unit": PREDICTIVE_UNIT,
                "market_only_daily_rows_as_event_examples": False,
                "reaction_family": REACTION_EXACT,
                "market_context_version": MARKET_CONTEXT_VERSION,
                "market_context_cutoff": cutoff.isoformat(),
                "event_rules_version": EVENT_RULES_VERSION,
                "financial_facts_version": FACTS_VERSION,
            },
            "event_features": event_features,
            "market_features": {**daily_features, **intraday_features},
            "event_semantics": {
                "analysis_status": analysis.status.value,
                "primary_event_type": analysis.primary_event_type.value,
                "rule_version": EVENT_RULES_VERSION,
                "financial_facts_version": FACTS_VERSION,
                "qwen_used": False,
            },
            "quality": {
                "event_available_at_cutoff": True,
                "market_context_available_at_cutoff": True,
                "post_event_values_in_features": False,
            },
        }
        target: dict[str, object] = {
            "event_id": event_id,
            "reaction_family": REACTION_EXACT,
            "horizons": row["labels"],
        }
        features.append(feature)
        targets.append(target)
    return features, targets


def _load_market_features(
    path: Path,
) -> dict[str, list[tuple[date, datetime, dict[str, float]]]]:
    result: dict[str, list[tuple[date, datetime, dict[str, float]]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = cast("dict[str, Any]", json.loads(line))
        trade_date = date.fromisoformat(str(row["trade_date"]))
        feature_as_of = date.fromisoformat(str(row["feature_as_of"]))
        cutoff = datetime.combine(feature_as_of, time(23, 59, 59), UTC)
        values = {
            key: float(cast("float | int", value))
            for key, value in cast("dict[str, object]", row["features"]).items()
        }
        result.setdefault(str(row["ticker"]), []).append((trade_date, cutoff, values))
    for values in result.values():
        values.sort(key=lambda item: (item[1], item[0]))
    return result


def market_context(
    rows: dict[str, list[tuple[date, datetime, dict[str, float]]]],
    ticker: str,
    publication_date: date,
) -> tuple[datetime, dict[str, float]] | None:
    candidates = [
        (cutoff, values)
        for _trade_date, cutoff, values in rows.get(ticker, [])
        if cutoff.date() < publication_date
    ]
    return max(candidates, key=lambda item: item[0]) if candidates else None


def _load_series(path: Path, tickers: set[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for ticker in sorted(tickers):
        file = path / f"{ticker}.jsonl"
        if not file.exists():
            raise ValueError(f"RAW_DAILY_SERIES_MISSING:{ticker}")
        result[ticker] = [
            cast("dict[str, Any]", json.loads(line))
            for line in file.read_text(encoding="utf-8").splitlines()
        ]
    return result


def daily_target(
    event: AcquiredEvent,
    security: list[dict[str, Any]],
    benchmark: list[dict[str, Any]],
) -> dict[str, object] | None:
    security_by_date = {date.fromisoformat(str(item["trade_date"])): item for item in security}
    benchmark_by_date = {date.fromisoformat(str(item["trade_date"])): item for item in benchmark}
    common = sorted(security_by_date.keys() & benchmark_by_date.keys())
    before = [item for item in common if item < event.publication_date]
    after = [item for item in common if item > event.publication_date]
    if not before or not after:
        return None
    baseline, target = before[-1], after[0]
    security_return = (
        Decimal(str(security_by_date[target]["close"]))
        / Decimal(str(security_by_date[baseline]["close"]))
        - 1
    )
    benchmark_return = (
        Decimal(str(benchmark_by_date[target]["close"]))
        / Decimal(str(benchmark_by_date[baseline]["close"]))
        - 1
    )
    abnormal = security_return - benchmark_return
    return {
        "event_id": str(event.event_id),
        "reaction_family": REACTION_DAILY,
        "baseline_session_date": baseline.isoformat(),
        "target_session_date": target.isoformat(),
        "security_return": str(security_return),
        "benchmark_return": str(benchmark_return),
        "abnormal_return": str(abnormal),
        "classification": classify_reaction(abnormal),
        "classification_threshold": "0.002",
    }


def leakage_pass(row: dict[str, object]) -> bool:
    metadata = cast("dict[str, Any]", row["metadata"])
    cutoff = datetime.fromisoformat(str(metadata["market_context_cutoff"]))
    quality = str(metadata["publication_time_quality"])
    if quality == PublicationTimestampQuality.EXACT.value:
        published = datetime.fromisoformat(str(metadata["published_at"]))
        if cutoff >= published:
            return False
    elif cutoff.date() >= date.fromisoformat(str(metadata["publication_date"])):
        return False
    forbidden = {
        "security_return",
        "benchmark_return",
        "abnormal_return",
        "target_close",
        "future_price",
        "future_volume",
    }
    return not (forbidden & cast("dict[str, object]", row["market_features"]).keys())


def _clusters(rows: list[dict[str, object]]) -> dict[str, object]:
    per_date: Counter[str] = Counter()
    per_issuer_date: Counter[str] = Counter()
    for row in rows:
        metadata = cast("dict[str, Any]", row["metadata"])
        day = str(metadata["publication_date"])
        ticker = str(metadata["ticker"])
        per_date[day] += 1
        per_issuer_date[f"{ticker}:{day}"] += 1
    return {
        "events_per_date": dict(sorted(per_date.items())),
        "events_per_issuer_date": dict(sorted(per_issuer_date.items())),
        "issuer_date_clusters_gt_1": sum(value > 1 for value in per_issuer_date.values()),
        "same_story_cluster_key": "ticker + publication_date + title_hash",
        "future_split_grouping_key": "ticker + publication_date",
        "grouped_temporal_split_required": True,
    }


def _feature_schema(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "event_features": list(event_feature_names()),
        "market_features": sorted(
            {key for item in rows for key in cast("dict[str, object]", item["market_features"])}
        ),
        "targets_stored_separately": True,
        "predictive_unit": PREDICTIVE_UNIT,
    }


def _reconcile_features_targets(
    features: list[dict[str, object]], targets: list[dict[str, object]]
) -> None:
    feature_ids = [str(cast("dict[str, Any]", item["metadata"])["event_id"]) for item in features]
    target_ids = [str(item["event_id"]) for item in targets]
    if feature_ids != target_ids or len(feature_ids) != len(set(feature_ids)):
        raise ValueError("FEATURE_TARGET_IDENTITY_RECONCILIATION_FAILED")


def _load_instrument_identities(path: Path) -> dict[str, dict[str, str | None]]:
    payload = _read_json(path)
    return {
        str(item["ticker"]): {
            "name": str(item["name"]),
            "instrument_uid": str(item["instrument_uid"]),
            "figi": _optional(item.get("figi")),
        }
        for item in cast("list[dict[str, Any]]", payload["instruments"])
    }


def _exclude(event: AcquiredEvent, reason: str) -> dict[str, object]:
    return {
        "event_id": str(event.event_id),
        "source_code": event.source_code,
        "ticker": event.ticker,
        "publication_date": event.publication_date.isoformat(),
        "reason": reason,
    }


def _optional(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _numeric(value: object) -> float:
    if isinstance(value, (str, int, float)):
        return float(value)
    raise TypeError("market feature must be numeric")


def _uuid(value: str):
    from uuid import UUID

    return UUID(value)


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


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
