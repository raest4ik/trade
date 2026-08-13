from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any, cast

from src.tinvest_market.application import acquire_instrument_series, date_chunks, resolve_universe
from src.tinvest_market.client import TInvestClientError, TInvestReadOnlyClient
from src.tinvest_market.domain import DailyBar, ResolvedInstrument, build_dataset, sha256_payload
from src.tinvest_market.policy import execution_safety, source_policy
from src.tinvest_market_universe.domain import (
    FEATURE_VERSION,
    MEMBERSHIP_MODE,
    ORIGINAL_TICKERS,
    PRICE_ADJUSTMENT_STATUS,
    RAW_VERSION,
    REPORT_VERSION,
    SOURCE_USAGE_READINESS,
    SURVIVORSHIP_RISK,
    DiscoveryResult,
    ExpandedFeatureRow,
    discover,
    enhance_features,
    feature_schema_sha,
    feature_summary,
    history_tier,
)


@dataclass(frozen=True, slots=True)
class ExpansionResult:
    report: dict[str, object]
    raw_manifest: dict[str, object]
    feature_manifest: dict[str, object]


async def expand_universe(
    client: TInvestReadOnlyClient,
    *,
    raw_dir: Path,
    feature_dir: Path,
    baseline_raw_dir: Path,
    date_from: date,
    date_to: date,
    git_sha: str,
    discovered_at: datetime | None = None,
) -> ExpansionResult:
    if date_to < date_from:
        raise ValueError("invalid date range")
    timestamp = discovered_at or datetime.now(UTC)
    with corpus_lock(raw_dir / "universe.lock"):
        shares = await client.list_shares()
        discovery = discover(shares)
        await asyncio.to_thread(_write_discovery, raw_dir, discovery, timestamp)
        acquired, instruments, failed, cache_hits = await _acquire_candidates(
            client,
            discovery=discovery,
            raw_dir=raw_dir,
            date_from=date_from,
            date_to=date_to,
            resolved_at=timestamp,
        )
        benchmark, benchmark_instrument, benchmark_error = await _acquire_benchmark(
            client,
            raw_dir=raw_dir,
            date_from=date_from,
            date_to=date_to,
            resolved_at=timestamp,
        )
        if benchmark_instrument is not None:
            instruments = (*instruments, benchmark_instrument)
        await asyncio.to_thread(
            _write_history_coverage,
            raw_dir,
            discovery,
            acquired,
            failed,
            timestamp,
        )
        all_rows = [row for rows in acquired.values() for row in rows]
        dataset_sha = sha256_payload(
            [
                row.payload()
                for row in sorted(all_rows, key=lambda item: (item.ticker, item.trade_date))
            ]
        )
        universe_identity = [
            _catalog_payload(item, timestamp=None)
            for item in sorted(
                discovery.eligible, key=lambda value: (value.ticker, value.instrument_uid)
            )
        ]
        universe_manifest_sha = sha256_payload(universe_identity)
        quality = _quality(acquired, raw_dir / "checkpoints")
        preservation = _compare_baseline(acquired, baseline_raw_dir)
        raw_manifest: dict[str, object] = {
            "dataset_version": RAW_VERSION,
            "created_at": timestamp.isoformat(),
            "git_sha": git_sha,
            "source_policy": source_policy(),
            "discovery": discovery.diagnostics,
            "requested_date_from": date_from.isoformat(),
            "requested_date_to": date_to.isoformat(),
            "downloaded_instrument_count": len(acquired),
            "failed_instrument_count": len(failed),
            "failed_instruments": failed,
            "cache_hits": sorted(cache_hits),
            "benchmark": "IMOEX",
            "benchmark_source": "TINVEST_API" if benchmark else None,
            "benchmark_error": benchmark_error,
            "benchmark_rows": len(benchmark or ()),
            "raw_rows": len(all_rows),
            "rows_per_ticker": dict(sorted(Counter(row.ticker for row in all_rows).items())),
            "earliest_date": min((row.trade_date for row in all_rows), default=None),
            "latest_date": max((row.trade_date for row in all_rows), default=None),
            "quality": quality,
            "original_ticker_preservation": preservation,
            "dataset_sha": dataset_sha,
            "universe_manifest_sha": universe_manifest_sha,
            "universe_membership_mode": MEMBERSHIP_MODE,
            "historical_membership_point_in_time_verified": False,
            "survivorship_bias_risk": SURVIVORSHIP_RISK,
            "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
            "source_usage_readiness": SOURCE_USAGE_READINESS,
            **_safety_metadata(),
        }
        await asyncio.to_thread(_write_json, raw_dir / "dataset-manifest.json", raw_manifest)
        await asyncio.to_thread(
            _write_json,
            raw_dir / "instrument-mapping.json",
            {"instruments": [item.payload() for item in instruments]},
        )

        built = build_dataset(acquired, benchmark)
        uid_by_ticker = {
            item.ticker: item.instrument_uid for item in instruments if item.ticker != "IMOEX"
        }
        feature_rows, feature_names = enhance_features(built.features, uid_by_ticker)
        feature_manifest = await asyncio.to_thread(
            _write_features,
            feature_dir,
            feature_rows,
            feature_names,
            raw_manifest,
            git_sha,
            timestamp,
        )
        report = _build_report(raw_manifest, feature_manifest, acquired)
        await asyncio.to_thread(
            _write_json, feature_dir / "market-universe-expansion-v1.json", report
        )
        return ExpansionResult(report, raw_manifest, feature_manifest)


async def _acquire_candidates(
    client: TInvestReadOnlyClient,
    *,
    discovery: DiscoveryResult,
    raw_dir: Path,
    date_from: date,
    date_to: date,
    resolved_at: datetime,
) -> tuple[
    dict[str, tuple[DailyBar, ...]],
    tuple[ResolvedInstrument, ...],
    list[dict[str, str]],
    list[str],
]:
    counts = Counter(item.ticker for item in discovery.eligible)
    acquired: dict[str, tuple[DailyBar, ...]] = {}
    instruments: list[ResolvedInstrument] = []
    failed: list[dict[str, str]] = []
    cache_hits: list[str] = []
    for item in discovery.eligible:
        if counts[item.ticker] > 1:
            failed.append(
                {
                    "ticker": item.ticker,
                    "uid": item.instrument_uid,
                    "reason": "AMBIGUOUS_TICKER_IDENTITY",
                }
            )
            continue
        instrument = _resolved(item, resolved_at)
        discovered_start = item.first_1day_candle_date
        if discovered_start is None:
            discovered_start = await _probe_earliest_date(
                client,
                instrument_uid=item.instrument_uid,
                date_from=date_from,
                date_to=date_to,
            )
        start = max(date_from, discovered_start or date_to)
        try:
            rows, cache_hit = await acquire_instrument_series(
                client,
                instrument=instrument,
                date_from=start,
                date_to=date_to,
                raw_dir=raw_dir,
                allow_rejected_candles=True,
            )
        except (TInvestClientError, ValueError) as exc:
            failed.append({"ticker": item.ticker, "uid": item.instrument_uid, "reason": str(exc)})
            continue
        acquired[item.ticker] = rows
        instruments.append(instrument)
        if cache_hit:
            cache_hits.append(item.ticker)
    return acquired, tuple(instruments), failed, cache_hits


async def _probe_earliest_date(
    client: TInvestReadOnlyClient,
    *,
    instrument_uid: str,
    date_from: date,
    date_to: date,
) -> date | None:
    earliest: date | None = None
    for chunk_from, chunk_to in reversed(date_chunks(date_from, date_to)):
        try:
            batch = await client.fetch_daily_candles_audited(
                instrument_uid=instrument_uid,
                date_from=chunk_from,
                date_to=chunk_to,
            )
        except TInvestClientError:
            continue
        if batch.candles:
            earliest = min(item.trade_date for item in batch.candles)
    return earliest


async def _acquire_benchmark(
    client: TInvestReadOnlyClient,
    *,
    raw_dir: Path,
    date_from: date,
    date_to: date,
    resolved_at: datetime,
) -> tuple[tuple[DailyBar, ...] | None, ResolvedInstrument | None, str | None]:
    try:
        resolution = await resolve_universe(client, tickers=(), resolved_at=resolved_at)
        instrument = next(item for item in resolution.instruments if item.ticker == "IMOEX")
        rows, _ = await acquire_instrument_series(
            client,
            instrument=instrument,
            date_from=max(date_from, instrument.first_1day_candle_date or date_from),
            date_to=date_to,
            raw_dir=raw_dir,
        )
        return rows, instrument, None
    except (StopIteration, TInvestClientError, ValueError) as exc:
        return None, None, str(exc)


def _resolved(item: Any, timestamp: datetime) -> ResolvedInstrument:
    return ResolvedInstrument(
        ticker=item.ticker,
        class_code=item.class_code,
        instrument_uid=item.instrument_uid,
        figi=item.figi,
        instrument_type=item.instrument_type,
        first_1day_candle_date=item.first_1day_candle_date,
        name=item.name,
        exchange=item.exchange,
        currency=item.currency,
        resolved_at=timestamp,
    )


def _write_discovery(raw_dir: Path, discovery: DiscoveryResult, discovered_at: datetime) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = [_catalog_payload(item, timestamp=discovered_at) for item in discovery.shares]
    _write_jsonl(raw_dir / "discovery-shares.jsonl", rows)
    _write_json(raw_dir / "discovery-summary.json", discovery.diagnostics)


def _write_history_coverage(
    raw_dir: Path,
    discovery: DiscoveryResult,
    acquired: dict[str, tuple[DailyBar, ...]],
    failed: list[dict[str, str]],
    discovered_at: datetime,
) -> None:
    failures = {item["uid"]: item["reason"] for item in failed}
    rows: list[dict[str, object]] = []
    for item in discovery.eligible:
        candles = acquired.get(item.ticker, ())
        rows.append(
            {
                **_catalog_payload(item, timestamp=discovered_at),
                "historical_candle_available": bool(candles),
                "earliest_available_date": (
                    min(row.trade_date for row in candles).isoformat() if candles else None
                ),
                "last_1day_candle_date": (
                    max(row.trade_date for row in candles).isoformat() if candles else None
                ),
                "historical_row_count": len(candles),
                "history_tier": history_tier(len(candles)),
                "acquisition_error": failures.get(item.instrument_uid),
            }
        )
    _write_jsonl(raw_dir / "history-coverage.jsonl", rows)


def _catalog_payload(item: Any, timestamp: datetime | None) -> dict[str, object]:
    payload = item.payload()
    if timestamp is not None:
        payload["discovered_at"] = timestamp.isoformat()
    payload["structurally_eligible"] = (
        item.instrument_type.upper() in {"SHARE", "INSTRUMENT_TYPE_SHARE"}
        and (item.currency or "").lower() == "rub"
        and item.class_code.upper() == "TQBR"
    )
    return payload


def _write_features(
    output_dir: Path,
    rows: tuple[ExpandedFeatureRow, ...],
    names: tuple[str, ...],
    raw_manifest: dict[str, object],
    git_sha: str,
    created_at: datetime,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [item.payload() for item in rows]
    _write_jsonl(output_dir / "features.jsonl", payloads)
    summary = feature_summary(rows)
    summary = {
        key: value.isoformat() if isinstance(value, date) else value
        for key, value in summary.items()
    }
    feature_sha = sha256_payload(payloads)
    manifest: dict[str, object] = {
        "dataset_version": FEATURE_VERSION,
        "report_version": REPORT_VERSION,
        "created_at": created_at.isoformat(),
        "git_sha": git_sha,
        "raw_dataset_sha": raw_manifest["dataset_sha"],
        "universe_manifest_sha": raw_manifest["universe_manifest_sha"],
        "feature_sha": feature_sha,
        "feature_schema_sha": feature_schema_sha(names),
        "feature_names": list(names),
        "feature_count": len(names),
        "rolling_window_ends_at": "t-1",
        "target_values_persisted": False,
        **summary,
        **_safety_metadata(),
    }
    _write_json(output_dir / "dataset-manifest.json", manifest)
    return manifest


def _quality(acquired: dict[str, tuple[DailyBar, ...]], checkpoint_dir: Path) -> dict[str, object]:
    per_ticker: dict[str, object] = {}
    observations: list[dict[str, object]] = []
    counts = {"gt_10pct": 0, "gt_20pct": 0, "gt_50pct": 0}
    rejected_total = 0
    for ticker, rows in sorted(acquired.items()):
        ordered = sorted(rows, key=lambda item: item.trade_date)
        if len({item.trade_date for item in ordered}) != len(ordered):
            raise ValueError(f"duplicate ticker/date: {ticker}")
        returns: list[tuple[date, float]] = []
        for previous, current in pairwise(ordered):
            value = current.close / previous.close - 1.0
            returns.append((current.trade_date, value))
            counts["gt_10pct"] += abs(value) > 0.10
            counts["gt_20pct"] += abs(value) > 0.20
            counts["gt_50pct"] += abs(value) > 0.50
        largest = sorted(returns, key=lambda item: abs(item[1]), reverse=True)[:5]
        observations.extend(
            {"ticker": ticker, "trade_date": day.isoformat(), "return": value}
            for day, value in largest
        )
        gaps = [(right.trade_date - left.trade_date).days for left, right in pairwise(ordered)]
        checkpoint_path = checkpoint_dir / f"{ticker}.json"
        checkpoint = (
            cast(
                "dict[str, object]",
                json.loads(checkpoint_path.read_text(encoding="utf-8")),
            )
            if checkpoint_path.exists()
            else {}
        )
        rejected = int(str(checkpoint.get("rejected_candle_count", 0)))
        rejected_total += rejected
        per_ticker[ticker] = {
            "row_count": len(ordered),
            "earliest_date": ordered[0].trade_date.isoformat() if ordered else None,
            "latest_date": ordered[-1].trade_date.isoformat() if ordered else None,
            "history_tier": history_tier(len(ordered)),
            "max_calendar_gap_days": max(gaps) if gaps else None,
            "calendar_gaps_gt_4_days": sum(value > 4 for value in gaps),
            "abs_return_gt_10pct": sum(abs(value) > 0.10 for _, value in returns),
            "abs_return_gt_20pct": sum(abs(value) > 0.20 for _, value in returns),
            "abs_return_gt_50pct": sum(abs(value) > 0.50 for _, value in returns),
            "rejected_invalid_candle_rows": rejected,
            "rejected_invalid_candle_reasons": checkpoint.get("rejected_candle_reasons", {}),
        }
    return {
        "ticker_date_unique": True,
        "ohlc_valid": True,
        "volume_nonnegative": True,
        "rows_forward_filled": 0,
        "rows_interpolated": 0,
        "rows_fabricated": 0,
        "rows_clipped": 0,
        "rows_winsorized": 0,
        "rejected_invalid_candle_rows": rejected_total,
        "extreme_return_counts": counts,
        "top_extreme_observations": sorted(
            observations, key=lambda item: abs(_numeric(item["return"])), reverse=True
        )[:50],
        "per_ticker": per_ticker,
    }


def _compare_baseline(
    acquired: dict[str, tuple[DailyBar, ...]], baseline_dir: Path
) -> dict[str, object]:
    results: dict[str, object] = {}
    for ticker in ORIGINAL_TICKERS:
        path = baseline_dir / "series" / f"{ticker}.jsonl"
        current = {item.trade_date.isoformat(): item.payload() for item in acquired.get(ticker, ())}
        if not path.exists() or not current:
            results[ticker] = {"preserved": False, "reason": "MISSING_SERIES"}
            continue
        baseline_rows = [
            cast("dict[str, object]", json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        baseline = {str(item["trade_date"]): item for item in baseline_rows}
        overlap = sorted(set(current) & set(baseline))
        fields = ("instrument_uid", "open", "high", "low", "close", "volume", "is_complete")
        mismatches = [
            day
            for day in overlap
            if any(current[day].get(field) != baseline[day].get(field) for field in fields)
        ]
        results[ticker] = {
            "preserved": bool(overlap) and not mismatches,
            "overlap_rows": len(overlap),
            "mismatch_rows": len(mismatches),
            "mismatch_sample": mismatches[:10],
            "baseline_overlap_sha": sha256_payload(
                [
                    {field: baseline[day].get(field) for field in fields} | {"trade_date": day}
                    for day in overlap
                ]
            ),
            "expanded_overlap_sha": sha256_payload(
                [
                    {field: current[day].get(field) for field in fields} | {"trade_date": day}
                    for day in overlap
                ]
            ),
        }
    return {
        "tickers": results,
        "all_original_10_present": all(ticker in acquired for ticker in ORIGINAL_TICKERS),
        "all_original_10_preserved": all(
            cast("dict[str, object]", results[ticker])["preserved"] is True
            for ticker in ORIGINAL_TICKERS
        ),
    }


def _build_report(
    raw: dict[str, object], features: dict[str, object], acquired: dict[str, tuple[DailyBar, ...]]
) -> dict[str, object]:
    row_counts = [len(rows) for rows in acquired.values()]
    tiers = Counter(history_tier(value) for value in row_counts)
    discovery = cast("dict[str, object]", raw["discovery"])
    preservation = cast("dict[str, object]", raw["original_ticker_preservation"])
    quality = cast("dict[str, object]", raw["quality"])
    return {
        "report_version": REPORT_VERSION,
        "discovered_shares_count": discovery["discovered_shares_count"],
        "instrument_status_all_confirmed": True,
        "class_code_distribution": discovery["class_code_distribution"],
        "tqbr_rub_candidate_count": discovery["tqbr_rub_candidate_count"],
        "downloaded_instrument_count": raw["downloaded_instrument_count"],
        "failed_instrument_count": raw["failed_instrument_count"],
        "failed_instruments": raw["failed_instruments"],
        "catalog_active_inactive_distribution": discovery["availability_distribution"],
        "active_inactive_distribution": discovery["candidate_availability_distribution"],
        "original_10_preserved": preservation["all_original_10_preserved"],
        "raw_rows": raw["raw_rows"],
        "feature_ready_rows": features["feature_ready_rows"],
        "earliest_date": raw["earliest_date"],
        "latest_date": raw["latest_date"],
        "partition_counts": features["partition_counts"],
        "future_holdout_session_count": features["future_holdout_session_count"],
        "median_history_sessions": float(median(row_counts)) if row_counts else 0.0,
        "history_tier_distribution": dict(sorted(tiers.items())),
        "tickers_252_plus": sum(value >= 252 for value in row_counts),
        "tickers_756_plus": sum(value >= 756 for value in row_counts),
        "tickers_1260_plus": sum(value >= 1260 for value in row_counts),
        "extreme_return_counts": quality["extreme_return_counts"],
        "rejected_invalid_candle_rows": quality["rejected_invalid_candle_rows"],
        "dataset_sha": raw["dataset_sha"],
        "universe_manifest_sha": raw["universe_manifest_sha"],
        "feature_sha": features["feature_sha"],
        "feature_schema_sha": features["feature_schema_sha"],
        "universe_membership_mode": MEMBERSHIP_MODE,
        "survivorship_bias_risk": SURVIVORSHIP_RISK,
        "price_adjustment_status": PRICE_ADJUSTMENT_STATUS,
        "source_usage_readiness": SOURCE_USAGE_READINESS,
        "data_status": "UNIVERSE_EXPANDED"
        if len(cast("dict[str, object]", features["features_per_ticker"])) > len(ORIGINAL_TICKERS)
        else "UNIVERSE_EXPANSION_LIMITED",
        **_safety_metadata(),
    }


def _safety_metadata() -> dict[str, object]:
    return {
        "model_trained": False,
        "model_selection_performed": False,
        "observed_test_used": False,
        "future_holdout_evaluated": False,
        "backtest_executed": False,
        "paper_trading_executed": False,
        "production_order_executed": False,
        "sandbox_order_executed": False,
        "buy_sell_generated": False,
        "paid_services_used": False,
        "moex_iss_data_used": False,
        "execution_safety": execution_safety(),
    }


@contextmanager
def corpus_lock(path: Path) -> Generator[None, None, None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("TINVEST_UNIVERSE_COLLECTION_ALREADY_RUNNING") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        if path.exists():
            path.unlink()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _numeric(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("expected numeric value")
    return float(value)
