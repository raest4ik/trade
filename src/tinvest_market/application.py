from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from src.tinvest_market.client import TInvestContractError, TInvestReadOnlyClient
from src.tinvest_market.domain import (
    BENCHMARK_TICKER,
    RAW_DATASET_VERSION,
    SECURITY_TICKERS,
    DailyBar,
    ResolvedInstrument,
    map_candle,
    resolve_instrument,
    sha256_payload,
)
from src.tinvest_market.policy import SOURCE, source_policy

MAX_CHUNK_DAYS = 1800


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    security_bars: dict[str, tuple[DailyBar, ...]]
    benchmark_bars: tuple[DailyBar, ...] | None
    instruments: tuple[ResolvedInstrument, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class UniverseResolution:
    instruments: tuple[ResolvedInstrument, ...]
    diagnostics: tuple[dict[str, object], ...]


async def resolve_universe(
    client: TInvestReadOnlyClient,
    *,
    tickers: tuple[str, ...] = SECURITY_TICKERS,
    resolved_at: datetime | None = None,
) -> UniverseResolution:
    timestamp = resolved_at or datetime.now(UTC)
    resolved: list[ResolvedInstrument] = []
    diagnostics: list[dict[str, object]] = []
    normalized = tuple(dict.fromkeys(item.strip().upper() for item in tickers if item.strip()))
    for ticker in normalized:
        candidates = await client.find_instruments(ticker, instrument_kind="INSTRUMENT_TYPE_SHARE")
        try:
            short = resolve_instrument(ticker, candidates, resolved_at=timestamp)
            metadata = await client.get_instrument_by_uid(short.instrument_uid)
            resolved.append(
                _verified_metadata(ticker, short, metadata, timestamp, expected_class_code="TQBR")
            )
        except (TInvestContractError, ValueError) as exc:
            diagnostics.append(_resolution_diagnostic(ticker, candidates, exc))
    try:
        indicatives = await client.list_indicatives()
        imoex_candidates = tuple(item for item in indicatives if item.ticker == BENCHMARK_TICKER)
        short = resolve_instrument(
            BENCHMARK_TICKER,
            imoex_candidates,
            resolved_at=timestamp,
            expected_class_code=None,
        )
        metadata = await client.get_instrument_by_uid(short.instrument_uid)
        resolved.append(
            _verified_metadata(
                BENCHMARK_TICKER,
                short,
                metadata,
                timestamp,
                expected_class_code=None,
            )
        )
    except (TInvestContractError, ValueError) as exc:
        diagnostics.append(_resolution_diagnostic(BENCHMARK_TICKER, (), exc))
    return UniverseResolution(tuple(resolved), tuple(diagnostics))


def _verified_metadata(
    ticker: str,
    short: ResolvedInstrument,
    metadata: object,
    resolved_at: datetime,
    *,
    expected_class_code: str | None,
) -> ResolvedInstrument:
    from src.tinvest_market.client import TInvestInstrument

    if not isinstance(metadata, TInvestInstrument):
        raise TypeError("unexpected instrument metadata")
    if metadata.ticker != ticker or metadata.instrument_uid != short.instrument_uid:
        raise ValueError(f"INSTRUMENT_METADATA_IDENTITY_MISMATCH:{ticker}")
    resolved = resolve_instrument(
        ticker,
        (metadata,),
        resolved_at=resolved_at,
        expected_class_code=expected_class_code,
    )
    return ResolvedInstrument(
        ticker=resolved.ticker,
        class_code=resolved.class_code,
        instrument_uid=resolved.instrument_uid,
        figi=resolved.figi,
        instrument_type=resolved.instrument_type,
        first_1day_candle_date=(resolved.first_1day_candle_date or short.first_1day_candle_date),
        name=resolved.name,
        exchange=resolved.exchange,
        currency=resolved.currency,
        resolved_at=resolved_at,
    )


def _resolution_diagnostic(
    ticker: str, candidates: tuple[object, ...], error: Exception
) -> dict[str, object]:
    from src.tinvest_market.client import TInvestInstrument

    typed = [item for item in candidates if isinstance(item, TInvestInstrument)]
    return {
        "ticker": ticker,
        "status": "UNRESOLVED_FAIL_CLOSED",
        "reason": str(error),
        "exact_ticker_candidates": sum(item.ticker == ticker for item in typed),
        "exact_tqbr_candidates": sum(
            item.ticker == ticker and item.class_code == "TQBR" for item in typed
        ),
    }


async def acquire_history(
    client: TInvestReadOnlyClient,
    *,
    raw_dir: Path,
    date_from: date,
    date_to: date,
    tickers: tuple[str, ...] = SECURITY_TICKERS,
    git_sha: str = "UNKNOWN",
    resolved_at: datetime | None = None,
) -> AcquisitionResult:
    if date_to < date_from:
        raise ValueError("date_to must not be before date_from")
    normalized_tickers = tuple(
        dict.fromkeys(item.strip().upper() for item in tickers if item.strip())
    )
    if not normalized_tickers:
        raise ValueError("at least one security ticker is required")
    await asyncio.to_thread(_prepare_directories, raw_dir)
    series_dir = raw_dir / "series"
    checkpoint_dir = raw_dir / "checkpoints"
    resolution = await resolve_universe(client, tickers=normalized_tickers, resolved_at=resolved_at)
    instruments = resolution.instruments
    mapping = {item.ticker: item for item in instruments}
    missing = sorted(set(normalized_tickers) - set(mapping))
    resolved_tickers = tuple(ticker for ticker in normalized_tickers if ticker in mapping)
    if not resolved_tickers:
        raise ValueError("no securities resolved")

    all_bars: dict[str, tuple[DailyBar, ...]] = {}
    cache_hits: list[str] = []
    for ticker in (*resolved_tickers, BENCHMARK_TICKER):
        instrument = mapping.get(ticker)
        if instrument is None:
            continue
        start = max(date_from, instrument.first_1day_candle_date or date_from)
        bars, cache_hit = await _acquire_series(
            client,
            instrument=instrument,
            date_from=start,
            date_to=date_to,
            series_path=series_dir / f"{ticker}.jsonl",
            checkpoint_path=checkpoint_dir / f"{ticker}.json",
        )
        all_bars[ticker] = bars
        if cache_hit:
            cache_hits.append(ticker)

    security_bars = {ticker: all_bars[ticker] for ticker in resolved_tickers}
    benchmark_bars = all_bars.get(BENCHMARK_TICKER)
    rows = [bar for ticker_rows in all_bars.values() for bar in ticker_rows]
    dates = [bar.trade_date for bar in rows]
    ticker_distribution = Counter(bar.ticker for bar in rows)
    year_distribution = Counter(str(bar.trade_date.year) for bar in rows)
    mapping_payload = [item.payload() for item in sorted(instruments, key=lambda item: item.ticker)]
    mapping_identity_payload = [
        {key: value for key, value in item.items() if key != "resolved_at"}
        for item in mapping_payload
    ]
    dataset_sha = sha256_payload(
        [bar.payload() for bar in sorted(rows, key=lambda item: (item.ticker, item.trade_date))]
    )
    manifest = {
        "dataset_version": RAW_DATASET_VERSION,
        "source": SOURCE,
        "source_policy": source_policy(),
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "instrument_mapping_sha": sha256_payload(mapping_identity_payload),
        "tickers": sorted(ticker_distribution),
        "date_from": min(dates).isoformat() if dates else None,
        "date_to": max(dates).isoformat() if dates else None,
        "requested_date_from": date_from.isoformat(),
        "requested_date_to": date_to.isoformat(),
        "row_count": len(rows),
        "ticker_distribution": dict(sorted(ticker_distribution.items())),
        "year_distribution": dict(sorted(year_distribution.items())),
        "dataset_sha": dataset_sha,
        "cache_hits": sorted(cache_hits),
        "imoex_resolved": BENCHMARK_TICKER in mapping,
        "resolved_securities_count": len(resolved_tickers),
        "resolved_security_tickers": list(resolved_tickers),
        "unresolved_security_tickers": missing,
        "resolution_diagnostics": list(resolution.diagnostics),
        "gap_diagnostics": {
            ticker: _gap_diagnostic(ticker_rows) for ticker, ticker_rows in sorted(all_bars.items())
        },
        "readonly_token_detected": True,
        "tls_production": "OK",
        "readonly_auth": "AUTH_OK",
        "paid_services": False,
        "model_trained": False,
    }
    await asyncio.to_thread(_write_acquisition_metadata, raw_dir, mapping_payload, manifest)
    return AcquisitionResult(security_bars, benchmark_bars, instruments, manifest)


async def validate_connectivity(client: TInvestReadOnlyClient) -> dict[str, object]:
    candidates = await client.find_instruments("SBER", instrument_kind="INSTRUMENT_TYPE_SHARE")
    resolved = resolve_instrument("SBER", candidates, resolved_at=datetime.now(UTC))
    sample_to = date.today() - timedelta(days=1)
    sample_from = sample_to - timedelta(days=14)
    candles = await client.fetch_daily_candles(
        instrument_uid=resolved.instrument_uid,
        date_from=sample_from,
        date_to=sample_to,
    )
    return {
        "auth": "AUTH_OK",
        "instrument_metadata_read": True,
        "sample_candle_rows": len(candles),
        "account_mutation": False,
    }


async def _acquire_series(
    client: TInvestReadOnlyClient,
    *,
    instrument: ResolvedInstrument,
    date_from: date,
    date_to: date,
    series_path: Path,
    checkpoint_path: Path,
    allow_rejected_candles: bool = False,
) -> tuple[tuple[DailyBar, ...], bool]:
    expected: dict[str, object] = {
        "ticker": instrument.ticker,
        "instrument_uid": instrument.instrument_uid,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "complete": True,
        "storage_policy": "COMPLETE_CANDLES_ONLY_V1",
    }
    cached = await asyncio.to_thread(
        _load_cached_series,
        checkpoint_path,
        series_path,
        expected,
        instrument.ticker,
        instrument.instrument_uid,
    )
    if cached is not None:
        return cached, True

    existing = await asyncio.to_thread(
        _load_extendable_series,
        checkpoint_path,
        series_path,
        expected,
        instrument.ticker,
        instrument.instrument_uid,
    )
    by_date: dict[date, DailyBar] = {
        item.trade_date: item for item in (existing[0] if existing is not None else ())
    }
    fetch_from = existing[1] + timedelta(days=1) if existing is not None else date_from
    incomplete_seen = False
    rejected_candle_count = 0
    rejected_reasons: Counter[str] = Counter()
    for chunk_from, chunk_to in date_chunks(fetch_from, date_to):
        if allow_rejected_candles:
            batch = await client.fetch_daily_candles_audited(
                instrument_uid=instrument.instrument_uid,
                date_from=chunk_from,
                date_to=chunk_to,
            )
            candles = batch.candles
            rejected_candle_count += len(batch.rejected_reasons)
            rejected_reasons.update(batch.rejected_reasons)
        else:
            candles = await client.fetch_daily_candles(
                instrument_uid=instrument.instrument_uid,
                date_from=chunk_from,
                date_to=chunk_to,
            )
        for candle in candles:
            if not candle.is_complete:
                incomplete_seen = True
                continue
            bar = map_candle(instrument.ticker, candle)
            bar.validate()
            if not chunk_from <= bar.trade_date <= chunk_to:
                raise ValueError(
                    f"candle outside requested range for {instrument.ticker}/{bar.trade_date}"
                )
            existing = by_date.get(bar.trade_date)
            if existing is not None:
                raise ValueError(f"duplicate candle for {instrument.ticker}/{bar.trade_date}")
            by_date[bar.trade_date] = bar
    bars = tuple(by_date[key] for key in sorted(by_date))
    await asyncio.to_thread(
        _write_series_and_checkpoint,
        series_path,
        checkpoint_path,
        bars,
        {
            **expected,
            "complete": not incomplete_seen,
            "rejected_candle_count": rejected_candle_count,
            "rejected_candle_reasons": dict(sorted(rejected_reasons.items())),
        },
    )
    return bars, False


async def acquire_instrument_series(
    client: TInvestReadOnlyClient,
    *,
    instrument: ResolvedInstrument,
    date_from: date,
    date_to: date,
    raw_dir: Path,
    allow_rejected_candles: bool = False,
) -> tuple[tuple[DailyBar, ...], bool]:
    """Acquire one UID-bound series through the shared bounded read-only path."""
    await asyncio.to_thread(_prepare_directories, raw_dir)
    return await _acquire_series(
        client,
        instrument=instrument,
        date_from=date_from,
        date_to=date_to,
        series_path=raw_dir / "series" / f"{instrument.ticker}.jsonl",
        checkpoint_path=raw_dir / "checkpoints" / f"{instrument.ticker}.json",
        allow_rejected_candles=allow_rejected_candles,
    )


def date_chunks(date_from: date, date_to: date) -> tuple[tuple[date, date], ...]:
    result: list[tuple[date, date]] = []
    current = date_from
    while current <= date_to:
        end = min(current + timedelta(days=MAX_CHUNK_DAYS), date_to)
        result.append((current, end))
        current = end + timedelta(days=1)
    return tuple(result)


def _gap_diagnostic(rows: tuple[DailyBar, ...]) -> dict[str, int | None]:
    ordered = sorted(rows, key=lambda item: item.trade_date)
    gaps = [(right.trade_date - left.trade_date).days for left, right in pairwise(ordered)]
    return {
        "row_count": len(ordered),
        "max_calendar_gap_days": max(gaps) if gaps else None,
        "calendar_gaps_gt_4_days": sum(item > 4 for item in gaps),
        "rows_synthesized": 0,
    }


def _load_series(path: Path, ticker: str, instrument_uid: str) -> tuple[DailyBar, ...]:
    rows: list[DailyBar] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = cast("dict[str, object]", json.loads(line))
        if payload.get("ticker") != ticker or payload.get("instrument_uid") != instrument_uid:
            raise ValueError("cached series identity mismatch")
        rows.append(
            DailyBar(
                ticker=ticker,
                instrument_uid=instrument_uid,
                trade_date=date.fromisoformat(str(payload["trade_date"])),
                open=_as_float(payload["open"]),
                high=_as_float(payload["high"]),
                low=_as_float(payload["low"]),
                close=_as_float(payload["close"]),
                volume=_as_float(payload["volume"]),
                is_complete=bool(payload["is_complete"]),
            )
        )
    return tuple(rows)


def _prepare_directories(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "series").mkdir(exist_ok=True)
    (raw_dir / "checkpoints").mkdir(exist_ok=True)


def _load_cached_series(
    checkpoint_path: Path,
    series_path: Path,
    expected: dict[str, object],
    ticker: str,
    instrument_uid: str,
) -> tuple[DailyBar, ...] | None:
    if not checkpoint_path.exists() or not series_path.exists():
        return None
    checkpoint = cast("dict[str, object]", json.loads(checkpoint_path.read_text(encoding="utf-8")))
    if not all(checkpoint.get(key) == value for key, value in expected.items()):
        return None
    return _load_series(series_path, ticker, instrument_uid)


def _load_extendable_series(
    checkpoint_path: Path,
    series_path: Path,
    expected: dict[str, object],
    ticker: str,
    instrument_uid: str,
) -> tuple[tuple[DailyBar, ...], date] | None:
    if not checkpoint_path.exists() or not series_path.exists():
        return None
    checkpoint = cast("dict[str, object]", json.loads(checkpoint_path.read_text(encoding="utf-8")))
    stable = ("ticker", "instrument_uid", "date_from", "storage_policy")
    if not all(checkpoint.get(key) == expected.get(key) for key in stable):
        return None
    if checkpoint.get("complete") is not True:
        return None
    previous_to = date.fromisoformat(str(checkpoint.get("date_to")))
    requested_to = date.fromisoformat(str(expected["date_to"]))
    if previous_to >= requested_to:
        return None
    return _load_series(series_path, ticker, instrument_uid), previous_to


def _write_series_and_checkpoint(
    series_path: Path,
    checkpoint_path: Path,
    bars: tuple[DailyBar, ...],
    expected: dict[str, object],
) -> None:
    payloads = [item.payload() for item in bars]
    _write_jsonl(series_path, payloads)
    _write_json(
        checkpoint_path,
        {
            **expected,
            "row_count": len(bars),
            "series_sha": sha256_payload(payloads),
        },
    )


def _write_acquisition_metadata(
    raw_dir: Path,
    mapping_payload: list[dict[str, object]],
    manifest: dict[str, Any],
) -> None:
    _write_json(raw_dir / "instrument-mapping.json", {"instruments": mapping_payload})
    _write_json(raw_dir / "dataset-manifest.json", manifest)


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        raise ValueError("cached numeric value is invalid")
    return float(value)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
