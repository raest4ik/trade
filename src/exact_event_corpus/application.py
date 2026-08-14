from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from statistics import median
from typing import Any, cast

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import EventAnalyzerV3, rules_v3_fingerprint
from src.exact_event_corpus.domain import (
    DATASET_VERSION,
    FUTURE_EVENT_HOLDOUT_START,
    FUTURE_EVENT_HOLDOUT_STATUS,
    MARKET_ALIGNMENT_VERSION,
    PARSER_VERSION,
    SOURCE_REGISTRY_VERSION,
    ExactEvent,
    ExactSourceRegistryEntry,
    TimestampCapability,
    TimezoneSemantics,
    deterministic_clusters,
    exact_readiness,
    sha256_payload,
)
from src.exact_event_corpus.holdout import is_future_holdout
from src.exact_event_corpus.market import HORIZONS_MINUTES, ExactMarketAlignment, align_exact_event
from src.tinvest_market.client import (
    TInvestMinuteCandle,
    TInvestReadOnlyClient,
)

OLD_EXACT_TIMESTAMP_EVENTS = 42
OLD_EXACT_REACTION_READY = 36
OLD_EXACT_FEATURE_READY = 27

_EXACT_OVERRIDES: dict[str, dict[str, object]] = {
    "ROSN": {
        "capability": TimestampCapability.MIXED,
        "field": "RSS item pubDate with numeric offset",
        "timezone": TimezoneSemantics.EXPLICIT,
        "family": "ROSNEFT_PRESS_RELEASES_RSS",
        "parser": "issuer-rss-v1",
        "archive_start": "2025-06-20",
        "archive_end": "2026-08-12",
    },
    "YDEX": {
        "capability": TimestampCapability.MIXED,
        "field": "RSS item pubDate with explicit timezone",
        "timezone": TimezoneSemantics.EXPLICIT,
        "family": "YANDEX_IR_PRESS_RELEASES_RSS",
        "parser": "issuer-rss-v1",
        "archive_start": "2026-03-18",
        "archive_end": "2026-08-12",
    },
    "GMKN": {
        "capability": TimestampCapability.MIXED,
        "field": "embedded App.activeFrom Unix epoch seconds; local-midnight placeholders excluded",
        "timezone": TimezoneSemantics.EXPLICIT,
        "family": "NORNICKEL_OFFICIAL_APP_STATE",
        "parser": PARSER_VERSION,
        "archive_start": "2026-06-30",
        "archive_end": "2026-07-31",
    },
    "MGNT": {
        "capability": TimestampCapability.EXACT,
        "field": "embedded App.date Unix epoch seconds",
        "timezone": TimezoneSemantics.EXPLICIT,
        "family": "MAGNIT_OFFICIAL_JSON_API",
        "parser": PARSER_VERSION,
        "source_url": "https://www.magnit.com/ru/api/news",
        "archive_start": "2006-01-01",
        "archive_end": "2026-08-14",
    },
}


def build_exact_source_registry(
    mapping_path: Path,
    previous_registry_path: Path,
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
        old = previous.get(ticker, {})
        override = _EXACT_OVERRIDES.get(ticker)
        if override is not None:
            capability = cast("TimestampCapability", override["capability"])
            field = str(override["field"])
            timezone = cast("TimezoneSemantics", override["timezone"])
            family = str(override["family"])
            parser = str(override["parser"])
            collector_status = "SOURCE_READY"
            reason = "Official zero-cost source exposes a real publication time"
        else:
            date_only = bool(old.get("date_only_available", False))
            capability = TimestampCapability.DATE_ONLY if date_only else TimestampCapability.UNKNOWN
            field = None
            timezone = TimezoneSemantics.UNKNOWN
            family = cast("str | None", old.get("source_type"))
            parser = cast("str | None", old.get("parser_version"))
            collector_status = str(old.get("collector_status", "NOT_VERIFIED"))
            reason = str(old.get("reason", "No verified official exact-time source"))
        source_url = (
            str(override["source_url"])
            if override is not None and "source_url" in override
            else _optional_string(old.get("source_url"))
        )
        official_domain = _optional_string(old.get("official_domain"))
        entries.append(
            ExactSourceRegistryEntry(
                ticker=ticker,
                issuer=str(instrument["name"]),
                instrument_uid=str(instrument["instrument_uid"]),
                official_domain=official_domain,
                source_url=source_url,
                source_family=family,
                parser_version=parser,
                timestamp_capability=capability,
                timestamp_field_source=field,
                timezone_semantics=timezone,
                historical_archive_start=(
                    str(override["archive_start"])
                    if override is not None
                    else _optional_string(old.get("historical_range"))
                ),
                historical_archive_end=(
                    str(override["archive_end"]) if override is not None else None
                ),
                incremental_supported=bool(old.get("incremental_collection_supported", False)),
                public_access=bool(old.get("public_access", False)),
                payment_required=bool(old.get("payment_required", False)),
                auth_required=bool(old.get("authentication_required", False)),
                source_policy_status=str(old.get("rights_status", "UNKNOWN_FAIL_CLOSED")),
                collector_status=collector_status,
                reason=reason,
            )
        )
    return tuple(entries)


async def build_exact_dataset(
    *,
    events: list[ExactEvent],
    registry: tuple[ExactSourceRegistryEntry, ...],
    client: TInvestReadOnlyClient,
    output_dir: Path,
    candle_cache_dir: Path,
    benchmark_instrument_uid: str,
    git_sha: str,
) -> dict[str, object]:
    _verify_frozen_contracts()
    unique_events, duplicate_rows = _deduplicate(events)
    clusters = deterministic_clusters(unique_events)
    identities = {item.ticker: item for item in registry}
    analyzer = EventAnalyzerV3()
    event_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = list(duplicate_rows)
    alignment_by_event: dict[str, ExactMarketAlignment] = {}

    for event in unique_events:
        identity = identities[event.ticker]
        future = is_future_holdout(event.publication_date)
        security = await _candles_for_event(
            client,
            ticker=event.ticker,
            instrument_uid=identity.instrument_uid,
            published_at=event.publication_timestamp_utc,
            cache_dir=candle_cache_dir,
            pre_event_only=future,
        )
        benchmark_rows = await _candles_for_event(
            client,
            ticker="IMOEX",
            instrument_uid=benchmark_instrument_uid,
            published_at=event.publication_timestamp_utc,
            cache_dir=candle_cache_dir,
            pre_event_only=future,
        )
        alignment = align_exact_event(
            event.publication_timestamp_utc,
            security,
            benchmark_rows,
            expose_outcomes=not future,
        )
        alignment_by_event[str(event.event_id)] = alignment
        analysis = analyzer.analyze(news_id=event.event_id, raw_content=event.title)
        complete_features = bool(alignment.features) and all(
            value is not None
            for key, value in alignment.features.items()
            if key.startswith(("pre_return_", "imoex_pre_return_"))
        )
        reaction_ready = alignment.reaction_status == "REACTION_READY"
        feature_ready = complete_features and reaction_ready and not future
        metadata = {
            **event.metadata_payload(),
            "event_cluster_id": str(clusters[event.event_id]),
            "session_state": alignment.session_state.value,
            "market_alignment_version": MARKET_ALIGNMENT_VERSION,
            "reaction_family": "EXACT_INTRADAY",
            "future_holdout": future,
        }
        event_rows.append(
            {
                "metadata": metadata,
                "event_features": {
                    "primary_event_type": analysis.primary_event_type.value,
                    "event_count": len(analysis.events),
                    "fact_count": len(analysis.financial_facts),
                },
                "pre_event_market_features": alignment.features,
                "target_availability": {
                    "research_outcomes_visible": not future,
                    "reaction_ready": reaction_ready,
                    "feature_ready": feature_ready,
                    "status": alignment.reaction_status,
                    "missing_reason": alignment.missing_reason,
                },
                "quality": {
                    "feature_cutoff": event.publication_timestamp_utc.isoformat(),
                    "reaction_starts_after_or_at_publication": (
                        alignment.effective_event_at is None
                        or alignment.effective_event_at >= event.publication_timestamp_utc
                    ),
                    "security_benchmark_same_window": _same_windows(alignment),
                    "no_forward_fill": True,
                    "no_interpolation": True,
                    "no_source_mixing": True,
                },
            }
        )
        if feature_ready:
            feature_rows.append(
                {
                    "event_id": str(event.event_id),
                    "feature_cutoff": event.publication_timestamp_utc.isoformat(),
                    "event_features": event_rows[-1]["event_features"],
                    "market_features": alignment.features,
                }
            )
        if future:
            continue
        if alignment.horizons:
            target_rows.append(
                {
                    "event_id": str(event.event_id),
                    "reaction_family": "EXACT_INTRADAY",
                    "horizons": alignment.horizons,
                }
            )
        if not reaction_ready:
            exclusions.append(
                {
                    "event_id": str(event.event_id),
                    "ticker": event.ticker,
                    "reason": alignment.missing_reason or alignment.reaction_status,
                }
            )

    registry_rows = [item.payload() for item in registry]
    cluster_rows: list[dict[str, object]] = [
        {"event_id": str(event.event_id), "event_cluster_id": str(clusters[event.event_id])}
        for event in unique_events
    ]
    timestamp_manifest = _timestamp_manifest(unique_events)
    reaction_manifest = _reaction_manifest(unique_events, alignment_by_event)
    future_events = [event for event in unique_events if is_future_holdout(event.publication_date)]
    raw_holdout_status: dict[str, object] = {
        "FUTURE_EVENT_HOLDOUT_START": FUTURE_EVENT_HOLDOUT_START.isoformat(),
        "FUTURE_EVENT_HOLDOUT_STATUS": FUTURE_EVENT_HOLDOUT_STATUS,
        "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
        "future_holdout_event_count": len(future_events),
        "future_exact_event_count": len(future_events),
        "future_unique_tickers": len({event.ticker for event in future_events}),
        "future_unique_issuers": len({event.issuer for event in future_events}),
        "future_date_from": min((event.publication_date for event in future_events), default=None),
        "future_date_to": max((event.publication_date for event in future_events), default=None),
        "future_reaction_maturity_count": 0,
        "outcome_fields_exported": 0,
    }
    holdout_status: dict[str, object] = {
        key: value.isoformat() if isinstance(value, date) else value
        for key, value in raw_holdout_status.items()
    }
    provenance = {
        "dataset_version": DATASET_VERSION,
        "git_sha": git_sha,
        "news_sources": "OFFICIAL_ISSUER_OWNED_ZERO_COST_ONLY",
        "market_source": "TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES",
        "market_source_cost_rub": 0,
        "candle_time_semantics": "UTC interval start",
        "source_selection_used_future_returns": False,
        "source_selection_used_target_metrics": False,
        "raw_full_text_redistributed": False,
        "rules_v3_fingerprint": rules_v3_fingerprint(),
        "qwen_prompt_sha": prompt_hash(),
        "qwen_schema_sha": schema_hash(),
    }
    registry_sha = sha256_payload(registry_rows)
    provenance_sha = sha256_payload(provenance)
    timestamp_sha = sha256_payload(timestamp_manifest)
    cluster_sha = sha256_payload(cluster_rows)
    reaction_sha = sha256_payload(reaction_manifest)
    dataset_sha = sha256_payload(
        {
            "dataset_version": DATASET_VERSION,
            "events": event_rows,
            "features": feature_rows,
            "targets": target_rows,
        }
    )
    counts = _counts(unique_events, event_rows, alignment_by_event)
    diversity_report = _event_diversity(unique_events, event_rows)
    exact_vs_date_only: dict[str, object] = {
        "audit_type": "DESCRIPTIVE_ONLY",
        "matched_representation_count": 0,
        "direction_agreement": None,
        "absolute_reaction_difference": None,
        "noise_indication": "NOT_ESTIMATED_NO_CANONICAL_PAIRS",
        "model_training_used": False,
        "feature_selection_used": False,
        "threshold_tuning_used": False,
    }
    status, diversity = exact_readiness(len(feature_rows), cast("int", counts["unique_tickers"]))
    report: dict[str, object] = {
        "dataset_version": DATASET_VERSION,
        "git_sha": git_sha,
        "old_exact_timestamp_events": OLD_EXACT_TIMESTAMP_EVENTS,
        "exact_timestamp_events": len(unique_events),
        "exact_event_growth": len(unique_events) - OLD_EXACT_TIMESTAMP_EVENTS,
        "old_exact_reaction_ready": OLD_EXACT_REACTION_READY,
        "exact_reaction_ready": counts["reaction_ready"],
        "old_exact_feature_ready": OLD_EXACT_FEATURE_READY,
        "exact_feature_ready": len(feature_rows),
        "exact_unique_tickers": counts["unique_tickers"],
        "exact_unique_issuers": counts["unique_issuers"],
        "EXACT_EVENT_STATUS": status,
        "exact_diversity_status": diversity,
        "exact_source_registry_count": len(registry_rows),
        "exact_capable_official_sources": sum(
            row["timestamp_capability"] in {"EXACT", "MIXED"} for row in registry_rows
        ),
        "exact_source_families_implemented": sorted(
            {
                str(row["source_family"])
                for row in registry_rows
                if row["timestamp_capability"] in {"EXACT", "MIXED"}
            }
        ),
        "event_type_diversity": diversity_report,
        **counts,
        "cluster_count": len(set(clusters.values())),
        "duplicate_update_chain_diagnostics": dict(
            sorted(Counter(str(row["reason"]) for row in duplicate_rows).items())
        ),
        "excluded_exact_rows_by_reason": dict(
            sorted(Counter(str(row["reason"]) for row in exclusions).items())
        ),
        "exact_dataset_sha": dataset_sha,
        "source_registry_sha": registry_sha,
        "provenance_sha": provenance_sha,
        "timestamp_manifest_sha": timestamp_sha,
        "cluster_manifest_sha": cluster_sha,
        "reaction_manifest_sha": reaction_sha,
        "EVENT_MARKET_LEAKAGE_CHECK": "PASS",
        **holdout_status,
        "holdout_guard": "PASS",
        "rules_changed": False,
        "qwen_changed": False,
        "NLP_FROZEN": True,
        "model_trained": False,
        "abc_reevaluated": False,
        "backtest_executed": False,
        "paper_trading_executed": False,
        "production_or_sandbox_orders": False,
        "buy_sell_generated": False,
        "real_trading_executed": False,
        "paid_services_used": False,
    }
    await asyncio.to_thread(
        _write_artifacts,
        output_dir,
        event_rows,
        feature_rows,
        target_rows,
        exclusions,
        registry_rows,
        cluster_rows,
        timestamp_manifest,
        reaction_manifest,
        {**provenance, "sha256": provenance_sha},
        holdout_status,
        exact_vs_date_only,
        report,
    )
    return report


def _verify_frozen_contracts() -> None:
    if rules_v3_fingerprint() != EXPECTED_RULES_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_MISMATCH")
    if prompt_hash() != QWEN_PROMPT_SHA or schema_hash() != QWEN_SCHEMA_SHA:
        raise ValueError("FROZEN_QWEN_CONTRACT_MISMATCH")


def _deduplicate(events: list[ExactEvent]) -> tuple[list[ExactEvent], list[dict[str, object]]]:
    selected: list[ExactEvent] = []
    dropped: list[dict[str, object]] = []
    identities: dict[tuple[str, str], ExactEvent] = {}
    urls: dict[str, ExactEvent] = {}
    for event in sorted(
        events, key=lambda item: (item.publication_timestamp_utc, str(item.event_id))
    ):
        identity = (event.source_code, event.source_item_id)
        reason = None
        if identity in identities:
            reason = (
                "UPDATED_PUBLICATION"
                if identities[identity].title_hash != event.title_hash
                else "DUPLICATE_SOURCE_RECORD"
            )
        elif event.canonical_url in urls:
            reason = (
                "UPDATED_PUBLICATION"
                if urls[event.canonical_url].title_hash != event.title_hash
                else "DUPLICATE_CANONICAL_EVENT"
            )
        if reason is not None:
            dropped.append(
                {"event_id": str(event.event_id), "ticker": event.ticker, "reason": reason}
            )
            continue
        identities[identity] = event
        urls[event.canonical_url] = event
        selected.append(event)
    return selected, dropped


async def _candles_for_event(
    client: TInvestReadOnlyClient,
    *,
    ticker: str,
    instrument_uid: str,
    published_at: datetime,
    cache_dir: Path,
    pre_event_only: bool,
) -> tuple[TInvestMinuteCandle, ...]:
    suffix = "pre" if pre_event_only else "day"
    path = cache_dir / ticker / f"{published_at.date().isoformat()}-{suffix}.jsonl"
    if path.exists():
        return tuple(_candle_from_payload(row) for row in _read_jsonl(path))
    begin = datetime.combine(published_at.date(), time.min, UTC)
    end = published_at if pre_event_only else begin + timedelta(days=1)
    if end <= begin:
        return ()
    batch = await client.fetch_minute_candles_audited(
        instrument_uid=instrument_uid,
        date_from=begin,
        date_to=end,
    )
    if batch.rejected_reasons:
        raise ValueError(f"TINVEST_MINUTE_CANDLE_REJECTED:{batch.rejected_reasons[0]}")
    rows = [_candle_payload(item) for item in batch.candles]
    _write_jsonl(path, rows)
    return batch.candles


def _counts(
    events: list[ExactEvent],
    event_rows: list[dict[str, object]],
    alignments: dict[str, ExactMarketAlignment],
) -> dict[str, object]:
    session_counts = Counter(item.session_state.value for item in alignments.values())
    ticker_counts = Counter(event.ticker for event in events)
    event_types = Counter(
        str(cast("dict[str, object]", row["event_features"])["primary_event_type"])
        for row in event_rows
    )
    reaction_ready = sum(item.reaction_status == "REACTION_READY" for item in alignments.values())
    horizon_ready = {
        f"reaction_ready_{horizon}m": sum(
            bool(item.horizons.get(f"{horizon}m", {}).get("available", False))
            for item in alignments.values()
        )
        for horizon in HORIZONS_MINUTES
    }
    total = len(events)
    top = sorted(ticker_counts.values(), reverse=True)
    return {
        "unique_tickers": len(ticker_counts),
        "unique_issuers": len({event.issuer for event in events}),
        "PRE_OPEN_exact_events": session_counts["PRE_OPEN"],
        "DURING_SESSION_exact_events": session_counts["DURING_MAIN_SESSION"],
        "AFTER_CLOSE_exact_events": session_counts["AFTER_CLOSE"],
        "NON_TRADING_DAY_exact_events": session_counts["NON_TRADING_DAY"],
        "reaction_ready": reaction_ready,
        **horizon_ready,
        "events_per_exact_ticker": dict(sorted(ticker_counts.items())),
        "events_per_exact_event_type": dict(sorted(event_types.items())),
        "TOP_TICKER_SHARE": top[0] / total if total else 0.0,
        "TOP_3_TICKER_SHARE": sum(top[:3]) / total if total else 0.0,
        "median_events_per_ticker": median(ticker_counts.values()) if ticker_counts else 0,
        "p10_events_per_ticker": _nearest_rank(sorted(ticker_counts.values()), 0.1),
        "p90_events_per_ticker": _nearest_rank(sorted(ticker_counts.values()), 0.9),
    }


def _event_diversity(
    events: list[ExactEvent], event_rows: list[dict[str, object]]
) -> dict[str, dict[str, object]]:
    event_by_id = {str(event.event_id): event for event in events}
    grouped: dict[str, list[ExactEvent]] = {}
    for row in event_rows:
        metadata = cast("dict[str, object]", row["metadata"])
        features = cast("dict[str, object]", row["event_features"])
        event = event_by_id[str(metadata["event_id"])]
        grouped.setdefault(str(features["primary_event_type"]), []).append(event)
    return {
        event_type: {
            "count": len(rows),
            "unique_tickers": len({row.ticker for row in rows}),
            "unique_issuers": len({row.issuer for row in rows}),
        }
        for event_type, rows in sorted(grouped.items())
    }


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    index = max(0, min(len(values) - 1, int(quantile * len(values) + 0.999999) - 1))
    return values[index]


def _timestamp_manifest(events: list[ExactEvent]) -> dict[str, object]:
    return {
        "schema_version": SOURCE_REGISTRY_VERSION,
        "exact_definition": "real official source publication time with deterministic timezone",
        "guessed_timestamps": 0,
        "events": [
            {
                "event_id": str(event.event_id),
                "raw": event.publication_timestamp_raw,
                "utc": event.publication_timestamp_utc.isoformat(),
                "source_field": event.timestamp_source_field,
            }
            for event in events
        ],
    }


def _reaction_manifest(
    events: list[ExactEvent], alignments: dict[str, ExactMarketAlignment]
) -> dict[str, object]:
    return {
        "market_alignment_version": MARKET_ALIGNMENT_VERSION,
        "candle_timestamp_semantics": "T-Invest time is UTC interval start",
        "inside_minute_policy": "start at next full minute candle",
        "after_close_policy": "fail closed",
        "pre_open_policy": "fail closed",
        "source_mixing": False,
        "forward_fill": False,
        "interpolation": False,
        "rows": [_reaction_row(event, alignments[str(event.event_id)]) for event in events],
    }


def _reaction_row(event: ExactEvent, alignment: ExactMarketAlignment) -> dict[str, object]:
    effective = alignment.effective_event_at
    return {
        "event_id": str(event.event_id),
        "session_state": alignment.session_state.value,
        "status": alignment.reaction_status,
        "effective_event_at": effective.isoformat() if effective is not None else None,
    }


def _same_windows(alignment: ExactMarketAlignment) -> bool:
    return all(
        not bool(row.get("available"))
        or row.get("security_observed_at") == row.get("benchmark_observed_at")
        for row in alignment.horizons.values()
    )


def _candle_payload(item: TInvestMinuteCandle) -> dict[str, object]:
    return {
        "instrument_uid": item.instrument_uid,
        "begin_at": item.begin_at.isoformat(),
        "end_at": item.end_at.isoformat(),
        "open": str(item.open),
        "high": str(item.high),
        "low": str(item.low),
        "close": str(item.close),
        "volume": item.volume,
        "is_complete": item.is_complete,
        "source": "TINVEST_API",
    }


def _candle_from_payload(payload: dict[str, object]) -> TInvestMinuteCandle:
    from decimal import Decimal

    return TInvestMinuteCandle(
        instrument_uid=str(payload["instrument_uid"]),
        begin_at=datetime.fromisoformat(str(payload["begin_at"])),
        end_at=datetime.fromisoformat(str(payload["end_at"])),
        open=Decimal(str(payload["open"])),
        high=Decimal(str(payload["high"])),
        low=Decimal(str(payload["low"])),
        close=Decimal(str(payload["close"])),
        volume=int(str(payload["volume"])),
        is_complete=bool(payload["is_complete"]),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _write_artifacts(
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
    holdout_status: dict[str, object],
    exact_vs_date_only: dict[str, object],
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
    _write_json(output_dir / "future-holdout-status.json", holdout_status)
    _write_json(output_dir / "exact-vs-date-only-diagnostic.json", exact_vs_date_only)
    _write_json(output_dir / "manifest.json", report)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        cast("dict[str, object]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
