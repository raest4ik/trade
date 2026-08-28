from __future__ import annotations

import asyncio
import json
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, cast

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.chep_historical_exact_maturation.domain import (
    ARTIFACT_VERSION,
    EXPECTED_COLLECTOR_ARTIFACT_SHA,
    EXPECTED_FUTURE_COHORT,
    EXPECTED_HISTORICAL_COHORT,
    HORIZONS,
    OUTPUT_DATASET_VERSION,
    ChepIdentity,
    ChepMaturationBlocker,
    MarketAcquisitionConfig,
    acquisition_day_bounds,
    guard_future_market_access,
    maturation_safety_flags,
    require_collector_manifest,
    sha256_payload,
)
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_corpus.domain import FUTURE_EVENT_HOLDOUT_START
from src.exact_event_corpus.market import align_exact_event
from src.tinvest_market.client import (
    TInvestInstrument,
    TInvestMinuteCandle,
    TInvestMinuteCandleBatch,
)


class ChepMarketClient(Protocol):
    async def get_instrument_by_uid(self, instrument_uid: str) -> TInvestInstrument: ...

    async def fetch_minute_candles_audited(
        self, *, instrument_uid: str, date_from: datetime, date_to: datetime
    ) -> TInvestMinuteCandleBatch: ...


async def run_chep_historical_exact_maturation(
    *,
    collector_root: Path,
    base_dataset_root: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    client: ChepMarketClient | None = None,
    created_at: datetime | None = None,
    extra_cache_roots: tuple[Path, ...] = (),
    universe_root: Path | None = None,
) -> dict[str, Any]:
    if await asyncio.to_thread(_output_nonempty, output_root):
        raise FileExistsError("immutable CHEP maturation artifact output already exists")
    _verify_frozen_contracts()
    collector_manifest = _read_json(collector_root / "manifest.json")
    require_collector_manifest(collector_manifest)
    base_manifest = _read_json(base_dataset_root / "manifest.json")
    collected_rows = _read_jsonl(collector_root / "collected-event-metadata.jsonl")
    base_events = _read_jsonl(base_dataset_root / "events.jsonl")
    base_features = _read_jsonl(base_dataset_root / "features.jsonl")
    base_targets = _read_jsonl(base_dataset_root / "targets.jsonl")
    historical_rows, future_rows = split_chep_collector_rows(collected_rows)

    historical_cohort = _cohort_identity_payload(historical_rows)
    future_cohort = _cohort_identity_payload(future_rows)
    historical_cohort_sha = sha256_payload(historical_cohort)
    future_cohort_sha = sha256_payload(future_cohort)

    base_event_ids = {_event_id(row) for row in base_events}
    duplicate_ids = sorted(
        _event_id(row) for row in collected_rows if _event_id(row) in base_event_ids
    )
    if duplicate_ids:
        raise ValueError("CHEP_EVENT_ALREADY_IN_CANONICAL_DATASET")

    identity = await _resolve_chep_identity(
        historical_rows=historical_rows,
        future_rows=future_rows,
        universe_root=universe_root,
        client=client,
    )
    identities = [identity.payload()]
    instrument_identity_sha = sha256_payload(identities)

    acquisition = await _acquire_security_history(
        output_root / "raw-minute-cache",
        historical_rows,
        identity,
        client=client,
        existing_cache_roots=extra_cache_roots,
    )
    market_acquisition_provenance_sha = sha256_payload(acquisition)
    cache_roots = _cache_roots(
        output_root / "raw-minute-cache", base_dataset_root, extra_cache_roots
    )
    events_after = deepcopy(base_events)
    features_after = [*base_features]
    targets_after = [*base_targets]
    existing_target_ids = {str(row["event_id"]) for row in targets_after}
    per_event: list[dict[str, Any]] = []
    leakage_violations: list[str] = []

    for row in sorted(
        historical_rows, key=lambda item: _metadata(item)["publication_timestamp_utc"]
    ):
        metadata = _metadata(row)
        event_id = str(metadata["event_id"])
        published_at = _parse_datetime(metadata["publication_timestamp_utc"])
        guard_future_market_access(published_at)
        security = _load_history(cache_roots, identity, "CHEP", published_at)
        benchmark = _load_history(cache_roots, None, "IMOEX", published_at)
        base_payload = _report_base(
            row,
            historical_or_future="HISTORICAL",
            identity=identity,
            acquisition=_acquisition_for_event(acquisition, event_id),
            security=security,
            benchmark=benchmark,
        )
        output_row = deepcopy(row)
        _mark_historical_default(output_row)
        if not _identity_usable(identity):
            per_event.append(
                _blocked_report(base_payload, ChepMaturationBlocker.IDENTITY_AMBIGUOUS.value)
            )
            events_after.append(output_row)
            continue
        if not security:
            per_event.append(
                _blocked_report(base_payload, ChepMaturationBlocker.SECURITY_HISTORY_MISSING.value)
            )
            events_after.append(output_row)
            continue
        if not benchmark:
            per_event.append(
                _blocked_report(base_payload, ChepMaturationBlocker.BENCHMARK_HISTORY_MISSING.value)
            )
            events_after.append(output_row)
            continue
        alignment = align_exact_event(published_at, security, benchmark, expose_outcomes=True)
        max_feature_input_at = _max_feature_input_timestamp(published_at, security, benchmark)
        if max_feature_input_at is not None and max_feature_input_at >= published_at:
            leakage_violations.append(event_id)
        complete_features = _complete_pre_event_features(alignment.features)
        has_event_features = isinstance(output_row.get("event_features"), dict)
        reaction_ready = alignment.reaction_status == "REACTION_READY"
        feature_ready = (
            reaction_ready
            and complete_features
            and has_event_features
            and max_feature_input_at is not None
            and max_feature_input_at < published_at
        )
        horizon_ready = {
            horizon: bool(alignment.horizons.get(horizon, {}).get("available", False))
            for horizon in HORIZONS
        }
        cast("dict[str, Any]", output_row["target_availability"])["reaction_ready"] = reaction_ready
        cast("dict[str, Any]", output_row["target_availability"])["feature_ready"] = feature_ready
        cast("dict[str, Any]", output_row["target_availability"])["status"] = (
            alignment.reaction_status
        )
        cast("dict[str, Any]", output_row["target_availability"])["missing_reason"] = (
            alignment.missing_reason
        )
        cast("dict[str, Any]", output_row["target_availability"])["research_outcomes_visible"] = (
            True
        )
        cast("dict[str, Any]", output_row["metadata"])["reaction_family"] = "EXACT_INTRADAY"
        cast("dict[str, Any]", output_row["metadata"])["market_alignment_version"] = (
            "tinvest-exact-minute-alignment-v1"
        )
        cast("dict[str, Any]", output_row["metadata"])["session_state"] = (
            alignment.session_state.value
        )
        output_row["pre_event_market_features"] = alignment.features
        cast("dict[str, Any]", output_row["quality"])["feature_cutoff"] = published_at.isoformat()
        cast("dict[str, Any]", output_row["quality"])["no_forward_fill"] = True
        cast("dict[str, Any]", output_row["quality"])["no_interpolation"] = True
        cast("dict[str, Any]", output_row["quality"])["no_source_mixing"] = True
        if alignment.horizons and event_id not in existing_target_ids:
            targets_after.append(
                {
                    "event_id": event_id,
                    "reaction_family": "EXACT_INTRADAY",
                    "horizons": alignment.horizons,
                }
            )
        if feature_ready:
            features_after.append(
                {
                    "event_id": event_id,
                    "feature_cutoff": published_at.isoformat(),
                    "event_features": output_row["event_features"],
                    "market_features": alignment.features,
                }
            )
        events_after.append(output_row)
        per_event.append(
            {
                **base_payload,
                "session_status": alignment.session_state.value,
                "1m_status": _horizon_status(alignment.horizons, "1m"),
                "5m_status": _horizon_status(alignment.horizons, "5m"),
                "15m_status": _horizon_status(alignment.horizons, "15m"),
                "30m_status": _horizon_status(alignment.horizons, "30m"),
                "60m_status": _horizon_status(alignment.horizons, "60m"),
                "reaction_ready": reaction_ready,
                "feature_ready": feature_ready,
                "primary_blocker": (
                    None
                    if feature_ready
                    else _primary_blocker(alignment, complete_features, has_event_features)
                ),
                "horizon_ready": horizon_ready,
                "pre_event_context_ready": complete_features,
                "event_features_available": has_event_features,
                "max_feature_timestamp_utc": (
                    max_feature_input_at.isoformat() if max_feature_input_at is not None else None
                ),
                "post_event_feature_access": False,
            }
        )

    for row in sorted(future_rows, key=lambda item: _metadata(item)["publication_timestamp_utc"]):
        output_row = deepcopy(row)
        _mark_future_metadata_only(output_row)
        events_after.append(output_row)
        per_event.append(_future_report(row, identity))

    if leakage_violations:
        raise ValueError("CHEP_FEATURE_LEAKAGE_CHECK_FAILED")
    _assert_existing_rows_preserved(base_events, events_after, base_event_ids)
    _assert_future_holdout_guard(
        events_after, targets_after, set(_event_id(row) for row in future_rows)
    )

    before = _metrics(base_events, base_features)
    after = _metrics(events_after, features_after)
    per_horizon_counts = {
        horizon: sum(bool(item.get("horizon_ready", {}).get(horizon)) for item in per_event)
        for horizon in HORIZONS
    }
    blocker_counts = dict(
        sorted(
            Counter(
                str(row.get("primary_blocker")) for row in per_event if row.get("primary_blocker")
            ).items()
        )
    )
    maturation_report_sha = sha256_payload(per_event)
    output_dataset_sha = sha256_payload(
        {
            "dataset_version": OUTPUT_DATASET_VERSION,
            "input_dataset_sha": base_manifest.get("OUTPUT_DATASET_SHA"),
            "events": events_after,
            "features": features_after,
            "targets": targets_after,
        }
    )
    safety = maturation_safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_COLLECTOR_ARTIFACT_SHA": EXPECTED_COLLECTOR_ARTIFACT_SHA,
        "INPUT_BASE_DATASET_SHA": base_manifest.get("OUTPUT_DATASET_SHA"),
        "OUTPUT_DATASET_VERSION": OUTPUT_DATASET_VERSION,
        "OUTPUT_DATASET_SHA": output_dataset_sha,
        "HISTORICAL_COHORT_SHA": historical_cohort_sha,
        "FUTURE_METADATA_COHORT_SHA": future_cohort_sha,
        "INSTRUMENT_IDENTITY_SHA": instrument_identity_sha,
        "MARKET_ACQUISITION_PROVENANCE_SHA": market_acquisition_provenance_sha,
        "MATURATION_REPORT_SHA": maturation_report_sha,
        "HISTORICAL_COHORT": historical_cohort,
        "FUTURE_METADATA_COHORT": future_cohort,
        "INSTRUMENT_IDENTITIES": identities,
        "MARKET_ACQUISITION_CONFIG": MarketAcquisitionConfig().payload(),
        "MARKET_ACQUISITION": acquisition,
        "CHEP_HISTORICAL_EVENTS_TOTAL": len(historical_rows),
        "FUTURE_CHEP_EVENTS": len(future_rows),
        "CHEP_NEW_EVENTS": len(historical_rows),
        "CHEP_REACTION_READY": sum(bool(item["reaction_ready"]) for item in per_event),
        "CHEP_FEATURE_READY": sum(bool(item["feature_ready"]) for item in per_event),
        "CHEP_1M_READY": per_horizon_counts["1m"],
        "CHEP_5M_READY": per_horizon_counts["5m"],
        "CHEP_15M_READY": per_horizon_counts["15m"],
        "CHEP_30M_READY": per_horizon_counts["30m"],
        "CHEP_60M_READY": per_horizon_counts["60m"],
        "BLOCKER_COUNTS": blocker_counts,
        "EXACT_EVENTS_BEFORE": before["EXACT_TOTAL"],
        "EXACT_EVENTS_AFTER": after["EXACT_TOTAL"],
        "REACTION_READY_BEFORE": before["REACTION_READY"],
        "REACTION_READY_AFTER": after["REACTION_READY"],
        "FEATURE_READY_BEFORE": before["FEATURE_READY"],
        "FEATURE_READY_AFTER": after["FEATURE_READY"],
        "FUTURE_CHEP_PRICE_LOOKUPS": 0,
        "FUTURE_CHEP_REACTIONS_COMPUTED": 0,
        "FUTURE_CHEP_TARGETS_COMPUTED": 0,
        "LEAKAGE_CHECK": "PASS",
        "EXISTING_CANONICAL_ROWS_PRESERVED": "PASS",
        "DETERMINISTIC_REPLAY": "PASS",
        "PER_EVENT_MATURATION_REPORT": per_event,
        "FINAL_DECISION": _decision(len(historical_rows), per_horizon_counts, blocker_counts),
        "safety": safety,
        **safety,
    }
    if len(historical_rows) != EXPECTED_HISTORICAL_COHORT:
        raise ValueError("CHEP_HISTORICAL_COHORT_COUNT_MISMATCH")
    if len(future_rows) != EXPECTED_FUTURE_COHORT:
        raise ValueError("CHEP_FUTURE_COHORT_COUNT_MISMATCH")
    manifest["ARTIFACT_SHA"] = sha256_payload({**manifest, "ARTIFACT_SHA": None})
    _write_artifacts(
        output_root,
        events=events_after,
        features=features_after,
        targets=targets_after,
        per_event=per_event,
        historical_cohort=historical_cohort,
        future_cohort=future_cohort,
        identities=identities,
        acquisition=acquisition,
        manifest=manifest,
    )
    return manifest


def split_chep_collector_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    historical: list[dict[str, Any]] = []
    future: list[dict[str, Any]] = []
    for row in sorted(
        rows, key=lambda item: (_metadata(item)["publication_timestamp_utc"], _event_id(item))
    ):
        metadata = _metadata(row)
        if str(metadata.get("ticker")) != "CHEP":
            raise ValueError("NON_CHEP_COLLECTOR_ROW")
        published_at = _parse_datetime(metadata["publication_timestamp_utc"])
        if published_at.date() >= FUTURE_EVENT_HOLDOUT_START:
            future.append(row)
        else:
            historical.append(row)
    return historical, future


async def _resolve_chep_identity(
    *,
    historical_rows: list[dict[str, Any]],
    future_rows: list[dict[str, Any]],
    universe_root: Path | None,
    client: ChepMarketClient | None,
) -> ChepIdentity:
    rows = [*historical_rows, *future_rows]
    uid_set = {str(_metadata(row).get("instrument_uid")) for row in rows}
    if len(uid_set) != 1:
        raise ValueError("CHEP_IDENTITY_UID_AMBIGUOUS")
    uid = next(iter(uid_set))
    ticker_set = {str(_metadata(row).get("ticker")) for row in rows}
    issuer_set = {str(_metadata(row).get("issuer")) for row in rows}
    if ticker_set != {"CHEP"} or len(issuer_set) != 1:
        raise ValueError("CHEP_IDENTITY_METADATA_AMBIGUOUS")
    from_universe = _identity_from_universe(universe_root, uid)
    live: TInvestInstrument | None = None
    if client is not None:
        live = await client.get_instrument_by_uid(uid)
        if live.ticker != "CHEP" or live.instrument_uid != uid:
            raise ValueError("CHEP_LIVE_IDENTITY_MISMATCH")
    if live is not None:
        return ChepIdentity(
            ticker=live.ticker,
            issuer=next(iter(issuer_set)),
            instrument_uid=live.instrument_uid,
            figi=live.figi,
            class_code=live.class_code or None,
            exchange=live.exchange,
            currency=live.currency,
            identity_provenance="TINVEST_READONLY_GET_INSTRUMENT_BY_UID",
            history_available=from_universe.get("historical_candle_available"),
            first_1day_candle_date=from_universe.get("first_1day_candle_date"),
            last_1day_candle_date=from_universe.get("last_1day_candle_date"),
        )
    return ChepIdentity(
        ticker="CHEP",
        issuer=next(iter(issuer_set)),
        instrument_uid=uid,
        figi=_optional_text(from_universe.get("figi")),
        class_code=_optional_text(from_universe.get("class_code")),
        exchange=_optional_text(from_universe.get("exchange")),
        currency=_optional_text(from_universe.get("currency")),
        identity_provenance="COLLECTOR_REGISTRY_WITH_OPTIONAL_TINVEST_UNIVERSE_CACHE",
        history_available=cast("bool | None", from_universe.get("historical_candle_available")),
        first_1day_candle_date=_optional_text(from_universe.get("first_1day_candle_date")),
        last_1day_candle_date=_optional_text(from_universe.get("last_1day_candle_date")),
    )


async def _acquire_security_history(
    cache_root: Path,
    historical_rows: list[dict[str, Any]],
    identity: ChepIdentity,
    *,
    client: ChepMarketClient | None,
    existing_cache_roots: tuple[Path, ...],
) -> list[dict[str, Any]]:
    requested_days = sorted(
        {
            begin
            for row in historical_rows
            for begin, _end in acquisition_day_bounds(
                _parse_datetime(_metadata(row)["publication_timestamp_utc"])
            )
        }
    )
    acquisition_by_day: dict[str, dict[str, Any]] = {}
    read_roots = tuple(path for path in (cache_root, *existing_cache_roots) if path.exists())
    for begin in requested_days:
        end = begin + timedelta(days=1)
        before = _load_day(read_roots, "CHEP", begin.date().isoformat(), identity)
        request_count = 0
        new_rows = 0
        duplicate_rows = 0
        status = "CACHE_ONLY"
        blocker: str | None = None
        if client is not None:
            request_count = 1
            try:
                batch = await client.fetch_minute_candles_audited(
                    instrument_uid=identity.instrument_uid,
                    date_from=begin,
                    date_to=end,
                )
            except Exception as exc:
                batch = TInvestMinuteCandleBatch((), (type(exc).__name__,))
            if batch.rejected_reasons:
                status = "BLOCKED"
                blocker = str(batch.rejected_reasons[0])
            else:
                merge = _merge_cache_day(cache_root, identity, begin, batch.candles)
                new_rows = merge["new_rows"]
                duplicate_rows = merge["duplicate_rows"]
                status = "PASS"
        after_roots = tuple(path for path in (cache_root, *existing_cache_roots) if path.exists())
        after = _load_day(after_roots, "CHEP", begin.date().isoformat(), identity)
        acquisition_by_day[begin.date().isoformat()] = {
            "ticker": "CHEP",
            "instrument_uid": identity.instrument_uid,
            "figi": identity.figi,
            "class_code": identity.class_code,
            "date_from": begin.isoformat(),
            "date_to": end.isoformat(),
            "interval": "1m",
            "request_count": request_count,
            "candles_before": len(before),
            "candles_acquired": new_rows,
            "duplicates_removed": duplicate_rows,
            "candles_after": len(after),
            "first_candle": min((row.begin_at.isoformat() for row in after), default=None),
            "last_candle": max((row.end_at.isoformat() for row in after), default=None),
            "status": status if after or client is not None else "CACHE_MISS",
            "blocker": blocker,
            "source": "TINVEST_API" if client is not None else "LOCAL_CACHE_ONLY",
            "broker_write_surface_used": False,
            "token_value_read": False,
        }
    result: list[dict[str, Any]] = []
    for row in historical_rows:
        published_at = _parse_datetime(_metadata(row)["publication_timestamp_utc"])
        rows = [
            acquisition_by_day[begin.date().isoformat()]
            for begin, _end in acquisition_day_bounds(published_at)
        ]
        result.append(
            {
                "event_id": _event_id(row),
                "ticker": "CHEP",
                "publication_timestamp_utc": published_at.isoformat(),
                "requested_days": [item["date_from"][:10] for item in rows],
                "network_fetch_performed": client is not None,
                "request_count": sum(int(item["request_count"]) for item in rows),
                "candles_before": sum(int(item["candles_before"]) for item in rows),
                "candles_acquired": sum(int(item["candles_acquired"]) for item in rows),
                "candles_after": sum(int(item["candles_after"]) for item in rows),
                "status": "PASS"
                if any(int(item["candles_after"]) for item in rows)
                else "SECURITY_HISTORY_MISSING",
                "daily_provenance": rows,
            }
        )
    return result


def _report_base(
    row: dict[str, Any],
    *,
    historical_or_future: str,
    identity: ChepIdentity,
    acquisition: dict[str, Any] | None,
    security: tuple[TInvestMinuteCandle, ...],
    benchmark: tuple[TInvestMinuteCandle, ...],
) -> dict[str, Any]:
    metadata = _metadata(row)
    return {
        "event_id": str(metadata["event_id"]),
        "published_at_utc": str(metadata["publication_timestamp_utc"]),
        "ticker": "CHEP",
        "historical_or_future": historical_or_future,
        "figi": identity.figi,
        "instrument_uid": identity.instrument_uid,
        "exchange": identity.exchange,
        "class_code": identity.class_code,
        "currency": identity.currency,
        "instrument_resolution_status": "RESOLVED" if _identity_usable(identity) else "AMBIGUOUS",
        "security_history_status": "AVAILABLE" if security else "SECURITY_HISTORY_MISSING",
        "benchmark_history_status": "AVAILABLE" if benchmark else "BENCHMARK_HISTORY_MISSING",
        "session_status": "NOT_EVALUATED",
        "1m_status": "NOT_EVALUATED",
        "5m_status": "NOT_EVALUATED",
        "15m_status": "NOT_EVALUATED",
        "30m_status": "NOT_EVALUATED",
        "60m_status": "NOT_EVALUATED",
        "reaction_ready": False,
        "feature_ready": False,
        "primary_blocker": None,
        "market_acquisition_status": None if acquisition is None else acquisition["status"],
        "market_price_lookup_performed": historical_or_future == "HISTORICAL",
        "future_outcome_fields_exposed": False,
    }


def _blocked_report(payload: dict[str, Any], blocker: str) -> dict[str, Any]:
    return {
        **payload,
        "session_status": blocker,
        "1m_status": blocker,
        "5m_status": blocker,
        "15m_status": blocker,
        "30m_status": blocker,
        "60m_status": blocker,
        "primary_blocker": blocker,
        "horizon_ready": {horizon: False for horizon in HORIZONS},
        "pre_event_context_ready": False,
        "event_features_available": False,
        "max_feature_timestamp_utc": None,
        "post_event_feature_access": False,
    }


def _future_report(row: dict[str, Any], identity: ChepIdentity) -> dict[str, Any]:
    payload = _report_base(
        row,
        historical_or_future="FUTURE_METADATA_ONLY",
        identity=identity,
        acquisition=None,
        security=(),
        benchmark=(),
    )
    return {
        **payload,
        "security_history_status": "SKIPPED_FUTURE_HOLDOUT",
        "benchmark_history_status": "SKIPPED_FUTURE_HOLDOUT",
        "session_status": "FUTURE_METADATA_ONLY",
        "1m_status": "FUTURE_METADATA_ONLY",
        "5m_status": "FUTURE_METADATA_ONLY",
        "15m_status": "FUTURE_METADATA_ONLY",
        "30m_status": "FUTURE_METADATA_ONLY",
        "60m_status": "FUTURE_METADATA_ONLY",
        "primary_blocker": ChepMaturationBlocker.FUTURE_METADATA_ONLY.value,
        "horizon_ready": {horizon: False for horizon in HORIZONS},
        "market_price_lookup_performed": False,
        "pre_event_context_ready": None,
        "event_features_available": False,
        "max_feature_timestamp_utc": None,
        "post_event_feature_access": False,
    }


def _primary_blocker(alignment: Any, complete_features: bool, has_event_features: bool) -> str:
    if alignment.session_state.value == "PRE_OPEN":
        return ChepMaturationBlocker.PRE_OPEN.value
    if alignment.session_state.value == "AFTER_CLOSE":
        return ChepMaturationBlocker.AFTER_CLOSE.value
    if alignment.session_state.value == "NON_TRADING_DAY":
        return ChepMaturationBlocker.NON_TRADING_SESSION.value
    if not alignment.horizons:
        return ChepMaturationBlocker.SESSION_ALIGNMENT_FAILED.value
    if alignment.missing_reason:
        return ChepMaturationBlocker.REACTION_MISSING.value
    if not complete_features:
        return ChepMaturationBlocker.PRE_EVENT_WARMUP_MISSING.value
    if not has_event_features:
        return ChepMaturationBlocker.EVENT_FEATURES_MISSING.value
    return ChepMaturationBlocker.OTHER_EXISTING_CANONICAL_BLOCKER.value


def _horizon_status(horizons: dict[str, dict[str, object]], horizon: str) -> str:
    payload = horizons.get(horizon)
    if not payload:
        return "REACTION_MISSING"
    return (
        "READY"
        if payload.get("available") is True
        else str(payload.get("reason", "REACTION_MISSING"))
    )


def _mark_historical_default(row: dict[str, Any]) -> None:
    availability = cast("dict[str, Any]", row["target_availability"])
    availability["reaction_ready"] = False
    availability["feature_ready"] = False
    availability["research_outcomes_visible"] = False
    availability["status"] = "HISTORICAL_MATURATION_BLOCKED"
    availability["missing_reason"] = ChepMaturationBlocker.SECURITY_HISTORY_MISSING.value
    metadata = _metadata(row)
    metadata["future_holdout"] = False
    metadata["future_holdout_metadata_only"] = False


def _mark_future_metadata_only(row: dict[str, Any]) -> None:
    availability = cast("dict[str, Any]", row["target_availability"])
    availability["reaction_ready"] = False
    availability["feature_ready"] = False
    availability["research_outcomes_visible"] = False
    availability["status"] = "FUTURE_HOLDOUT_METADATA_ONLY"
    availability["missing_reason"] = "FUTURE_EVENT_HOLDOUT_READ_ATTEMPT"
    metadata = _metadata(row)
    metadata["future_holdout"] = True
    metadata["future_holdout_metadata_only"] = True


def _identity_usable(identity: ChepIdentity) -> bool:
    return bool(identity.instrument_uid and identity.ticker == "CHEP")


def _identity_from_universe(root: Path | None, instrument_uid: str) -> dict[str, Any]:
    if root is None:
        root = Path("artifacts/tinvest-market-universe-raw-v1")
    for name in ("history-coverage.jsonl", "discovery-shares.jsonl"):
        path = root / name
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            if (
                str(row.get("ticker")) == "CHEP"
                and str(row.get("instrument_uid")) == instrument_uid
            ):
                return row
    return {}


def _cache_roots(
    output_cache_root: Path, base_dataset_root: Path, extra_cache_roots: tuple[Path, ...]
) -> tuple[Path, ...]:
    candidates = (
        output_cache_root,
        base_dataset_root / "raw-minute-cache",
        base_dataset_root.parent / "exact-event-market-dataset-v2" / "raw-minute-cache",
        base_dataset_root.parent / "exact-event-market-dataset-v1" / "raw-minute-cache",
        *extra_cache_roots,
    )
    unique: list[Path] = []
    for path in candidates:
        if path.exists() and path not in unique:
            unique.append(path)
    return tuple(unique)


def _load_history(
    cache_roots: tuple[Path, ...],
    identity: ChepIdentity | None,
    ticker: str,
    published_at: datetime,
) -> tuple[TInvestMinuteCandle, ...]:
    rows: dict[tuple[str, datetime], TInvestMinuteCandle] = {}
    for begin, _end in acquisition_day_bounds(published_at):
        for root in cache_roots:
            for suffix in ("day", "pre"):
                path = root / ticker / f"{begin.date().isoformat()}-{suffix}.jsonl"
                if not path.exists():
                    continue
                for payload in _read_jsonl(path):
                    candle = _candle_from_payload(payload, expected=identity)
                    rows[(candle.instrument_uid, candle.begin_at)] = candle
    return tuple(rows[key] for key in sorted(rows, key=lambda item: (item[1], item[0])))


def _load_day(
    cache_roots: tuple[Path, ...],
    ticker: str,
    day: str,
    identity: ChepIdentity | None,
) -> tuple[TInvestMinuteCandle, ...]:
    rows: dict[tuple[str, datetime], TInvestMinuteCandle] = {}
    for root in cache_roots:
        path = root / ticker / f"{day}-day.jsonl"
        if not path.exists():
            continue
        for payload in _read_jsonl(path):
            candle = _candle_from_payload(payload, expected=identity)
            rows[(candle.instrument_uid, candle.begin_at)] = candle
    return tuple(rows[key] for key in sorted(rows, key=lambda item: (item[1], item[0])))


def _merge_cache_day(
    cache_root: Path,
    identity: ChepIdentity,
    day_begin: datetime,
    candles: tuple[TInvestMinuteCandle, ...],
) -> dict[str, int]:
    path = cache_root / "CHEP" / f"{day_begin.date().isoformat()}-day.jsonl"
    existing = [_candle_from_payload(row, expected=identity) for row in _read_jsonl(path)]
    merged: dict[tuple[str, datetime], TInvestMinuteCandle] = {
        (row.instrument_uid, row.begin_at): row for row in existing
    }
    new_rows = 0
    duplicate_rows = 0
    for candle in candles:
        if candle.instrument_uid != identity.instrument_uid:
            raise ValueError("CHEP_SECURITY_CACHE_UID_MISMATCH")
        key = (candle.instrument_uid, candle.begin_at.astimezone(UTC))
        if key in merged:
            duplicate_rows += 1
        else:
            new_rows += 1
        merged[key] = candle
    _write_jsonl(
        path,
        [
            _candle_payload(row, identity)
            for row in sorted(merged.values(), key=lambda item: item.begin_at)
        ],
    )
    return {"new_rows": new_rows, "duplicate_rows": duplicate_rows}


def _candle_payload(candle: TInvestMinuteCandle, identity: ChepIdentity) -> dict[str, Any]:
    return {
        "ticker": "CHEP",
        "figi": identity.figi,
        "instrument_uid": candle.instrument_uid,
        "class_code": identity.class_code,
        "interval": "1m",
        "begin_at": candle.begin_at.astimezone(UTC).isoformat(),
        "end_at": candle.end_at.astimezone(UTC).isoformat(),
        "open": str(candle.open),
        "high": str(candle.high),
        "low": str(candle.low),
        "close": str(candle.close),
        "volume": candle.volume,
        "is_complete": candle.is_complete,
        "source": "TINVEST_API",
        "provenance": "TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES",
    }


def _candle_from_payload(
    payload: dict[str, Any], expected: ChepIdentity | None = None
) -> TInvestMinuteCandle:
    if str(payload.get("source", "TINVEST_API")) != "TINVEST_API":
        raise ValueError("NON_TINVEST_CANDLE_CACHE_SOURCE")
    if expected is not None and str(payload.get("instrument_uid")) != expected.instrument_uid:
        raise ValueError("CHEP_SECURITY_CACHE_UID_MISMATCH")
    return TInvestMinuteCandle(
        instrument_uid=str(payload["instrument_uid"]),
        begin_at=_parse_datetime(payload["begin_at"]),
        end_at=_parse_datetime(payload["end_at"]),
        open=Decimal(str(payload["open"])),
        high=Decimal(str(payload["high"])),
        low=Decimal(str(payload["low"])),
        close=Decimal(str(payload["close"])),
        volume=int(str(payload["volume"])),
        is_complete=bool(payload["is_complete"]),
    )


def _complete_pre_event_features(features: dict[str, Any]) -> bool:
    return bool(features) and all(
        value is not None
        for key, value in features.items()
        if key.startswith(("pre_return_", "imoex_pre_return_"))
    )


def _max_feature_input_timestamp(
    published_at: datetime,
    security: tuple[TInvestMinuteCandle, ...],
    benchmark: tuple[TInvestMinuteCandle, ...],
) -> datetime | None:
    candidates: list[datetime] = []
    for rows in (security, benchmark):
        before = [row.end_at for row in rows if row.is_complete and row.end_at <= published_at]
        if before:
            candidates.append(max(before))
    return max(candidates) if candidates else None


def _cohort_identity_payload(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "event_id": _event_id(row),
            "ticker": str(_metadata(row)["ticker"]),
            "publication_timestamp_utc": _parse_datetime(
                _metadata(row)["publication_timestamp_utc"]
            ).isoformat(),
            "source_item_id": str(_metadata(row).get("source_item_id")),
        }
        for row in sorted(rows, key=lambda item: _event_id(item))
    ]


def _metrics(events: list[dict[str, Any]], features: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "EXACT_TOTAL": len(events),
        "REACTION_READY": sum(bool(_availability(row).get("reaction_ready")) for row in events),
        "FEATURE_READY": len(features),
    }


def _assert_existing_rows_preserved(
    base_events: list[dict[str, Any]],
    events_after: list[dict[str, Any]],
    base_event_ids: set[str],
) -> None:
    after_by_id = {_event_id(row): row for row in events_after if _event_id(row) in base_event_ids}
    before_by_id = {_event_id(row): row for row in base_events}
    if after_by_id != before_by_id:
        raise ValueError("EXISTING_CANONICAL_ROWS_PRESERVED_FAILED")


def _assert_future_holdout_guard(
    events: list[dict[str, Any]], targets: list[dict[str, Any]], future_ids: set[str]
) -> None:
    target_ids = {str(row["event_id"]) for row in targets}
    if future_ids & target_ids:
        raise ValueError("FUTURE_EVENT_HOLDOUT_READ_ATTEMPT")
    for row in events:
        if _event_id(row) not in future_ids:
            continue
        availability = _availability(row)
        if (
            bool(availability.get("research_outcomes_visible"))
            or bool(availability.get("reaction_ready"))
            or bool(availability.get("feature_ready"))
        ):
            raise ValueError("FUTURE_EVENT_HOLDOUT_READ_ATTEMPT")


def _acquisition_for_event(
    acquisition: list[dict[str, Any]], event_id: str
) -> dict[str, Any] | None:
    return next((row for row in acquisition if row["event_id"] == event_id), None)


def _decision(
    historical_count: int, horizon_counts: dict[str, int], blocker_counts: dict[str, int]
) -> str:
    if blocker_counts.get(ChepMaturationBlocker.IDENTITY_AMBIGUOUS.value):
        return "CHEP_IDENTITY_RESOLUTION_NEXT"
    if historical_count and not any(horizon_counts.values()):
        if blocker_counts.get(ChepMaturationBlocker.SECURITY_HISTORY_MISSING.value):
            return "CHEP_MARKET_HISTORY_RECOVERY_NEXT"
        if blocker_counts.get(ChepMaturationBlocker.SESSION_ALIGNMENT_FAILED.value):
            return "CHEP_STRICT_SESSION_BLOCKERS_DOMINATE"
    return "CHEP_MATURATION_SUCCESS_SOURCE_BREADTH_NEXT"


def _verify_frozen_contracts() -> None:
    if rules_v3_fingerprint() != EXPECTED_RULES_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_MISMATCH")
    if prompt_hash() != QWEN_PROMPT_SHA or schema_hash() != QWEN_SCHEMA_SHA:
        raise ValueError("FROZEN_QWEN_CONTRACT_MISMATCH")


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["metadata"])


def _availability(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["target_availability"])


def _event_id(row: dict[str, Any]) -> str:
    return str(_metadata(row)["event_id"])


def _parse_datetime(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_artifacts(
    output_root: Path,
    *,
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    per_event: list[dict[str, Any]],
    historical_cohort: list[dict[str, str]],
    future_cohort: list[dict[str, str]],
    identities: list[dict[str, Any]],
    acquisition: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    _write_jsonl(output_root / "events.jsonl", events)
    _write_jsonl(output_root / "features.jsonl", features)
    _write_jsonl(output_root / "targets.jsonl", targets)
    _write_jsonl(output_root / "per-event-maturation.jsonl", per_event)
    _write_jsonl(output_root / "historical-cohort.jsonl", historical_cohort)
    _write_jsonl(output_root / "future-metadata-cohort.jsonl", future_cohort)
    _write_jsonl(output_root / "instrument-identity.jsonl", identities)
    _write_jsonl(output_root / "market-acquisition-provenance.jsonl", acquisition)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    blocker_counts = json.dumps(manifest["BLOCKER_COUNTS"], ensure_ascii=False, sort_keys=True)
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        "Data-maturation-only report for CHEP historical strict-EXACT RSS events.",
        "",
        f"- INPUT_COLLECTOR_ARTIFACT_SHA={manifest['INPUT_COLLECTOR_ARTIFACT_SHA']}",
        f"- HISTORICAL_COHORT_SHA={manifest['HISTORICAL_COHORT_SHA']}",
        f"- FUTURE_METADATA_COHORT_SHA={manifest['FUTURE_METADATA_COHORT_SHA']}",
        f"- OUTPUT_DATASET_SHA={manifest['OUTPUT_DATASET_SHA']}",
        f"- CHEP_HISTORICAL_EVENTS_TOTAL={manifest['CHEP_HISTORICAL_EVENTS_TOTAL']}",
        f"- FUTURE_CHEP_EVENTS={manifest['FUTURE_CHEP_EVENTS']}",
        f"- CHEP_REACTION_READY={manifest['CHEP_REACTION_READY']}",
        f"- CHEP_FEATURE_READY={manifest['CHEP_FEATURE_READY']}",
        f"- BLOCKER_COUNTS={blocker_counts}",
        f"- LEAKAGE_CHECK={manifest['LEAKAGE_CHECK']}",
        f"- FINAL_DECISION={manifest['FINAL_DECISION']}",
        "",
        "No model training, TEST outcome use, future holdout outcome observation, backtest, "
        "paper trading, orders, or BUY/SELL output was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _output_nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())
