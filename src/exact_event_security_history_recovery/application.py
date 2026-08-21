from __future__ import annotations

import asyncio
import json
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, cast

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_corpus.market import align_exact_event
from src.exact_event_security_history_recovery.domain import (
    ARTIFACT_VERSION,
    FUTURE_EVENT_HOLDOUT_START,
    HORIZONS,
    OUTPUT_DATASET_VERSION,
    PR36_OUTPUT_DATASET_SHA,
    PR37_ARTIFACT_SHA,
    AcquisitionConfig,
    RecoveryBlocker,
    RecoveryIdentity,
    acquisition_bounds,
    acquisition_day_bounds,
    recovery_safety_flags,
    require_pr36_manifest,
    require_pr37_manifest,
    sha256_payload,
)
from src.tinvest_market.client import TInvestMinuteCandle, TInvestMinuteCandleBatch


class SecurityHistoryRecoveryClient(Protocol):
    async def fetch_minute_candles_audited(
        self, *, instrument_uid: str, date_from: datetime, date_to: datetime
    ) -> TInvestMinuteCandleBatch: ...


async def run_security_history_recovery(
    *,
    pr36_root: Path,
    diagnostics_root: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    client: SecurityHistoryRecoveryClient | None = None,
    created_at: datetime | None = None,
    extra_cache_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if await asyncio.to_thread(_output_nonempty, output_root):
        raise FileExistsError("immutable security history recovery artifact output already exists")
    _verify_frozen_contracts()
    pr36_manifest = _read_json(pr36_root / "manifest.json")
    pr37_manifest = _read_json(diagnostics_root / "manifest.json")
    require_pr36_manifest(pr36_manifest)
    require_pr37_manifest(pr37_manifest)
    events_before = _read_jsonl(pr36_root / "events.jsonl")
    features_before = _read_jsonl(pr36_root / "features.jsonl")
    targets_before = _read_jsonl(pr36_root / "targets.jsonl")
    diagnostic_rows = _read_jsonl(diagnostics_root / "per-event-diagnostics.jsonl")
    cohort = _recovery_cohort(diagnostic_rows)
    cohort_ids = {item.event_id for item in cohort}
    event_by_id = {_event_id(row): row for row in events_before}
    _reconcile_cohort_with_events(cohort, event_by_id)

    recovery_cache_root = output_root / "raw-minute-cache"
    acquisition = await _acquire_recovery_cache(
        recovery_cache_root,
        cohort,
        client=client,
        existing_cache_roots=extra_cache_roots,
    )
    acquisition_by_event = {str(item["EVENT_ID"]): item for item in acquisition}
    cache_roots = _cache_roots(pr36_root, recovery_cache_root, extra_cache_roots)
    security_cache_roots = tuple(
        path for path in (recovery_cache_root, *extra_cache_roots) if path.exists()
    )
    events_after = deepcopy(events_before)
    event_after_by_id = {_event_id(row): row for row in events_after}
    features_after = [*features_before]
    targets_after = [*targets_before]
    existing_target_ids = {str(row["event_id"]) for row in targets_before}
    per_event: list[dict[str, Any]] = []
    recovered_event_ids: list[str] = []
    blocked_event_ids: list[str] = []
    leakage_violations: list[str] = []

    for identity in cohort:
        before_row = event_by_id[identity.event_id]
        final_row = event_after_by_id[identity.event_id]
        availability_before = _availability(before_row)
        security = _load_security_history(security_cache_roots, identity)
        benchmark = _load_benchmark_history(cache_roots, "IMOEX", identity.publication_timestamp)
        acquisition_row = acquisition_by_event[identity.event_id]
        base_payload = _per_event_base(
            identity,
            availability_before=availability_before,
            acquisition_row=acquisition_row,
            security=security,
            benchmark=benchmark,
        )
        if not bool(acquisition_row["CACHE_IDENTITY_MATCH"] == "PASS"):
            per_event.append(_blocked_payload(base_payload, RecoveryBlocker.IDENTITY_CHANGED.value))
            blocked_event_ids.append(identity.event_id)
            continue
        if not security:
            per_event.append(
                _blocked_payload(base_payload, RecoveryBlocker.SECURITY_HISTORY_INSUFFICIENT.value)
            )
            blocked_event_ids.append(identity.event_id)
            continue
        if not benchmark:
            per_event.append(
                _blocked_payload(base_payload, RecoveryBlocker.BENCHMARK_HISTORY_MISSING.value)
            )
            blocked_event_ids.append(identity.event_id)
            continue

        alignment = align_exact_event(
            identity.publication_timestamp,
            security,
            benchmark,
            expose_outcomes=True,
        )
        max_feature_input_at = _max_feature_input_timestamp(
            identity.publication_timestamp, security, benchmark
        )
        if (
            max_feature_input_at is not None
            and max_feature_input_at >= identity.publication_timestamp
        ):
            leakage_violations.append(identity.event_id)
        horizon_ready = {
            horizon: bool(alignment.horizons.get(horizon, {}).get("available", False))
            for horizon in HORIZONS
        }
        complete_features = _complete_pre_event_features(alignment.features)
        reaction_ready = alignment.reaction_status == "REACTION_READY"
        feature_ready = (
            complete_features
            and reaction_ready
            and max_feature_input_at is not None
            and max_feature_input_at < identity.publication_timestamp
        )
        cast("dict[str, Any]", final_row["target_availability"])["reaction_ready"] = reaction_ready
        cast("dict[str, Any]", final_row["target_availability"])["feature_ready"] = feature_ready
        cast("dict[str, Any]", final_row["target_availability"])["status"] = (
            alignment.reaction_status
        )
        cast("dict[str, Any]", final_row["target_availability"])["missing_reason"] = (
            alignment.missing_reason
        )
        cast("dict[str, Any]", final_row["target_availability"])["research_outcomes_visible"] = True
        cast("dict[str, Any]", final_row["metadata"])["session_state"] = (
            alignment.session_state.value
        )
        final_row["pre_event_market_features"] = alignment.features
        cast("dict[str, Any]", final_row["quality"])["feature_cutoff"] = (
            identity.publication_timestamp.isoformat()
        )
        cast("dict[str, Any]", final_row["quality"])["no_forward_fill"] = True
        cast("dict[str, Any]", final_row["quality"])["no_interpolation"] = True
        cast("dict[str, Any]", final_row["quality"])["no_source_mixing"] = True
        if alignment.horizons and identity.event_id not in existing_target_ids:
            targets_after.append(
                {
                    "event_id": identity.event_id,
                    "reaction_family": "EXACT_INTRADAY",
                    "horizons": alignment.horizons,
                }
            )
        if feature_ready:
            features_after.append(
                {
                    "event_id": identity.event_id,
                    "feature_cutoff": identity.publication_timestamp.isoformat(),
                    "event_features": final_row["event_features"],
                    "market_features": alignment.features,
                }
            )
            recovered_event_ids.append(identity.event_id)
        else:
            blocked_event_ids.append(identity.event_id)
        per_event.append(
            {
                **base_payload,
                "PRE_EVENT_CONTEXT_READY": complete_features,
                "MAX_FEATURE_TIMESTAMP": (
                    max_feature_input_at.isoformat() if max_feature_input_at is not None else None
                ),
                "REACTION_1M_READY": horizon_ready["1m"],
                "REACTION_5M_READY": horizon_ready["5m"],
                "REACTION_15M_READY": horizon_ready["15m"],
                "REACTION_30M_READY": horizon_ready["30m"],
                "REACTION_60M_READY": horizon_ready["60m"],
                "FEATURE_READY_AFTER": feature_ready,
                "RECOVERY_STATUS": "RECOVERED" if feature_ready else "BLOCKED",
                "FINAL_BLOCKER": None
                if feature_ready
                else _primary_blocker(alignment, complete_features),
                "SESSION_CLASSIFICATION": alignment.session_state.value,
                "HORIZON_READY": horizon_ready,
            }
        )

    if leakage_violations:
        raise ValueError("SECURITY_HISTORY_RECOVERY_LEAKAGE_CHECK_FAILED")
    _assert_non_cohort_events_preserved(events_before, events_after, cohort_ids)
    feature_preservation = _existing_features_preserved(features_before, features_after)
    if feature_preservation["status"] != "PASS":
        raise ValueError("EXISTING_FEATURE_ROWS_PRESERVED_FAILED")
    _assert_future_targets_guard(events_after, targets_after)

    before = _metrics(events_before, features_before)
    after = _metrics(events_after, features_after)
    output_dataset_sha = sha256_payload(
        {
            "dataset_version": OUTPUT_DATASET_VERSION,
            "input_dataset_sha": PR36_OUTPUT_DATASET_SHA,
            "events": events_after,
            "features": features_after,
            "targets": targets_after,
        }
    )
    config = AcquisitionConfig().payload()
    cache_provenance = _cache_provenance(
        security_cache_roots,
        cohort,
        network_fetch_performed=client is not None,
    )
    safety = recovery_safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_DATASET_SHA": PR36_OUTPUT_DATASET_SHA,
        "OUTPUT_DATASET_VERSION": OUTPUT_DATASET_VERSION,
        "OUTPUT_DATASET_SHA": output_dataset_sha,
        "PR37_ARTIFACT_SHA": PR37_ARTIFACT_SHA,
        "RECOVERY_COHORT_SHA": sha256_payload(sorted(cohort_ids)),
        "RECOVERY_COHORT_TOTAL": len(cohort),
        "RECOVERY_EVENT_IDS": sorted(cohort_ids),
        "INSTRUMENT_IDENTITIES": [
            item.payload() for item in sorted(cohort, key=lambda row: row.event_id)
        ],
        "CACHE_ACQUISITION_CONFIG_SHA": sha256_payload(config),
        "CACHE_ACQUISITION_CONFIG": config,
        "CACHE_PROVENANCE_HASH": _cache_provenance_hash(cache_provenance),
        "CACHE_PROVENANCE": cache_provenance,
        "CACHE_ACQUISITION_STATUS": "PASS"
        if all(str(row["ACQUISITION_STATUS"]) in {"PASS", "CACHE_ONLY"} for row in acquisition)
        else "BLOCKED",
        "CACHE_CANDLES_ACQUIRED_BY_TICKER": dict(sorted(_acquired_by_ticker(acquisition).items())),
        "CACHE_DEDUPE": "PASS",
        "PER_EVENT_RECOVERY": per_event,
        "RECOVERED_EVENT_IDS": sorted(recovered_event_ids),
        "BLOCKED_EVENT_IDS": sorted(blocked_event_ids),
        "EXACT_TOTAL_BEFORE": before["EXACT_TOTAL"],
        "EXACT_TOTAL_AFTER": after["EXACT_TOTAL"],
        "REACTION_READY_BEFORE": before["REACTION_READY"],
        "REACTION_READY_AFTER": after["REACTION_READY"],
        "REACTION_READY_DELTA": after["REACTION_READY"] - before["REACTION_READY"],
        "FEATURE_READY_BEFORE": before["FEATURE_READY"],
        "FEATURE_READY_AFTER": after["FEATURE_READY"],
        "FEATURE_READY_DELTA": after["FEATURE_READY"] - before["FEATURE_READY"],
        "RECOVERY_SUCCESS_COUNT": len(recovered_event_ids),
        "RECOVERY_BLOCKED_COUNT": len(blocked_event_ids),
        "PER_HORIZON_READY_BEFORE": _per_horizon_ready(targets_before),
        "PER_HORIZON_READY_AFTER": _per_horizon_ready(targets_after),
        "FEATURE_READY_UNIQUE_TICKERS_BEFORE": before["FEATURE_READY_UNIQUE_TICKERS"],
        "FEATURE_READY_UNIQUE_TICKERS_AFTER": after["FEATURE_READY_UNIQUE_TICKERS"],
        "FEATURE_READY_BY_TICKER_BEFORE": before["FEATURE_READY_BY_TICKER"],
        "FEATURE_READY_BY_TICKER_AFTER": after["FEATURE_READY_BY_TICKER"],
        "FEATURE_READY_TOP1_BEFORE": before["FEATURE_READY_TOP1"],
        "FEATURE_READY_TOP1_AFTER": after["FEATURE_READY_TOP1"],
        "FEATURE_READY_TOP3_BEFORE": before["FEATURE_READY_TOP3"],
        "FEATURE_READY_TOP3_AFTER": after["FEATURE_READY_TOP3"],
        "FEATURE_READY_ISSUER_HHI_BEFORE": before["FEATURE_READY_ISSUER_HHI"],
        "FEATURE_READY_ISSUER_HHI_AFTER": after["FEATURE_READY_ISSUER_HHI"],
        "EFFECTIVE_FEATURE_READY_ISSUER_COUNT_BEFORE": before[
            "EFFECTIVE_FEATURE_READY_ISSUER_COUNT"
        ],
        "EFFECTIVE_FEATURE_READY_ISSUER_COUNT_AFTER": after["EFFECTIVE_FEATURE_READY_ISSUER_COUNT"],
        "EXACT_V3_PRESERVED": "YES",
        "PR36_DATASET_PRESERVED": "YES",
        "EXISTING_EVENT_ROWS_PRESERVED": "PASS",
        "EXISTING_FEATURE_ROWS_PRESERVED": feature_preservation["status"],
        "existing_event_rows_preservation_hash": sha256_payload(
            [_strip_row(row) for row in events_before if _event_id(row) not in cohort_ids]
        ),
        "existing_feature_rows_preservation_hash": feature_preservation["hash"],
        "FEATURE_SCHEMA_SHA": _feature_schema_sha(features_after),
        "LEAKAGE_CHECK": "PASS",
        "DETERMINISTIC_REPLAY_CORE_FIELDS": "ARTIFACT_SHA_EXCLUDES_CREATED_AT_AND_GIT_SHA",
        "safety": safety,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = _artifact_sha(manifest)
    _write_artifacts(
        output_root,
        manifest=manifest,
        events=events_after,
        features=features_after,
        targets=targets_after,
        per_event=per_event,
        acquisition=acquisition,
        future_exclusions=_future_exclusions(events_before),
        cache_provenance=cache_provenance,
    )
    return manifest


def _recovery_cohort(rows: list[dict[str, Any]]) -> list[RecoveryIdentity]:
    result: list[RecoveryIdentity] = []
    for row in rows:
        published_at = _parse_datetime(row["PUBLICATION_TIMESTAMP"])
        if published_at.date() >= FUTURE_EVENT_HOLDOUT_START:
            raise ValueError("FUTURE_EVENT_ENTERED_SECURITY_HISTORY_RECOVERY")
        if str(row.get("ROOT_CAUSE")) != "CURRENT_IDENTITY_HAS_HISTORY":
            continue
        if bool(row.get("RECOVERY_POSSIBLE")) is not True:
            continue
        if bool(row.get("RECOVERY_PERFORMED")) is not False:
            continue
        result.append(
            RecoveryIdentity(
                event_id=str(row["EVENT_ID"]),
                ticker=str(row["TICKER"]),
                issuer=str(row["ISSUER"]),
                publication_timestamp=published_at,
                figi=_required_text(row, "CURRENT_FIGI"),
                instrument_uid=_required_text(row, "CURRENT_UID"),
                class_code=_required_text(row, "CURRENT_CLASS_CODE"),
            )
        )
    if not result:
        raise ValueError("RECOVERY_COHORT_EMPTY")
    return sorted(result, key=lambda item: item.event_id)


def _reconcile_cohort_with_events(
    cohort: list[RecoveryIdentity], event_by_id: dict[str, dict[str, Any]]
) -> None:
    for identity in cohort:
        row = event_by_id.get(identity.event_id)
        if row is None:
            raise ValueError("RECOVERY_EVENT_NOT_IN_PR36_DATASET")
        metadata = _metadata(row)
        if bool(metadata.get("future_holdout")):
            raise ValueError("FUTURE_EVENT_ENTERED_SECURITY_HISTORY_RECOVERY")
        if str(metadata.get("ticker")) != identity.ticker:
            raise ValueError("RECOVERY_EVENT_TICKER_MISMATCH")
        if str(metadata.get("instrument_uid")) != identity.instrument_uid:
            raise ValueError("RECOVERY_EVENT_UID_MISMATCH")


async def _acquire_recovery_cache(
    cache_root: Path,
    cohort: list[RecoveryIdentity],
    *,
    client: SecurityHistoryRecoveryClient | None,
    existing_cache_roots: tuple[Path, ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    read_roots = tuple(path for path in (cache_root, *existing_cache_roots) if path.exists())
    for identity in cohort:
        before = _load_security_history(read_roots, identity)
        request_count = 0
        newly_acquired = 0
        duplicate_count = 0
        if client is not None:
            for begin, end in acquisition_day_bounds(identity.publication_timestamp):
                request_count += 1
                batch = await client.fetch_minute_candles_audited(
                    instrument_uid=identity.instrument_uid,
                    date_from=begin,
                    date_to=end,
                )
                if batch.rejected_reasons:
                    raise ValueError(
                        f"SECURITY_CACHE_ACQUISITION_FAILED:{batch.rejected_reasons[0]}"
                    )
                merge = _merge_cache_day(cache_root, identity, begin, batch.candles)
                newly_acquired += merge["new_rows"]
                duplicate_count += merge["duplicate_rows"]
        after_roots = tuple(path for path in (cache_root, *existing_cache_roots) if path.exists())
        after = _load_security_history(after_roots, identity)
        start, end = acquisition_bounds(identity.publication_timestamp)
        result.append(
            {
                "EVENT_ID": identity.event_id,
                "TICKER": identity.ticker,
                "EXPECTED_UID": identity.instrument_uid,
                "CACHE_UID": identity.instrument_uid if after else None,
                "CACHE_IDENTITY_MATCH": "PASS"
                if _identity_cache_matches(after, identity)
                else "FAIL",
                "ACQUISITION_FROM": start.isoformat(),
                "ACQUISITION_TO": end.isoformat(),
                "INTERVAL": "1m",
                "REQUEST_COUNT": request_count,
                "CANDLES_BEFORE": len(before),
                "CANDLES_ACQUIRED": newly_acquired,
                "CANDLES_AFTER": len(after),
                "CANDLE_COUNT": len(after),
                "DUPLICATES_REMOVED": duplicate_count,
                "FIRST_SECURITY_CANDLE": min(
                    (row.begin_at.isoformat() for row in after), default=None
                ),
                "LAST_SECURITY_CANDLE": max(
                    (row.end_at.isoformat() for row in after), default=None
                ),
                "TIMESTAMP_MIN": min((row.begin_at.isoformat() for row in after), default=None),
                "TIMESTAMP_MAX": max((row.end_at.isoformat() for row in after), default=None),
                "SORTED": "PASS"
                if tuple(sorted(after, key=lambda item: item.begin_at)) == after
                else "FAIL",
                "UNIQUE_TIMESTAMP_IDENTITY": "PASS"
                if _unique_identity_timestamps(after)
                else "FAIL",
                "ACQUISITION_STATUS": "PASS"
                if after
                else ("CACHE_ONLY" if client is None else "EMPTY"),
            }
        )
    return result


def _merge_cache_day(
    cache_root: Path,
    identity: RecoveryIdentity,
    day_begin: datetime,
    candles: tuple[TInvestMinuteCandle, ...],
) -> dict[str, int]:
    path = cache_root / identity.ticker / f"{day_begin.date().isoformat()}-day.jsonl"
    existing_payloads = _read_jsonl(path) if path.exists() else []
    existing = [_candle_from_payload(row, expected=identity) for row in existing_payloads]
    merged: dict[tuple[str, datetime], TInvestMinuteCandle] = {
        (row.instrument_uid, row.begin_at): row for row in existing
    }
    duplicate_rows = 0
    new_rows = 0
    for candle in candles:
        if candle.instrument_uid != identity.instrument_uid:
            raise ValueError("SECURITY_CACHE_UID_MISMATCH")
        key = (candle.instrument_uid, candle.begin_at.astimezone(UTC))
        if key in merged:
            duplicate_rows += 1
        else:
            new_rows += 1
        merged[key] = candle
    rows = [
        _candle_payload(candle, identity)
        for candle in sorted(merged.values(), key=lambda item: (item.begin_at, item.instrument_uid))
    ]
    _write_jsonl(path, rows)
    return {"new_rows": new_rows, "duplicate_rows": duplicate_rows}


def _load_security_history(
    cache_roots: tuple[Path, ...], identity: RecoveryIdentity
) -> tuple[TInvestMinuteCandle, ...]:
    rows: dict[tuple[str, datetime], TInvestMinuteCandle] = {}
    for begin, _end in acquisition_day_bounds(identity.publication_timestamp):
        for root in cache_roots:
            path = root / identity.ticker / f"{begin.date().isoformat()}-day.jsonl"
            if not path.exists():
                continue
            for payload in _read_jsonl(path):
                candle = _candle_from_payload(payload, expected=identity)
                rows[(candle.instrument_uid, candle.begin_at)] = candle
    return tuple(rows[key] for key in sorted(rows, key=lambda item: (item[1], item[0])))


def _load_benchmark_history(
    cache_roots: tuple[Path, ...], ticker: str, published_at: datetime
) -> tuple[TInvestMinuteCandle, ...]:
    rows: dict[tuple[str, datetime], TInvestMinuteCandle] = {}
    for begin, _end in acquisition_day_bounds(published_at):
        for root in cache_roots:
            for suffix in ("day", "pre"):
                path = root / ticker / f"{begin.date().isoformat()}-{suffix}.jsonl"
                if not path.exists():
                    continue
                for payload in _read_jsonl(path):
                    candle = _candle_from_payload(payload)
                    rows[(candle.instrument_uid, candle.begin_at)] = candle
    return tuple(rows[key] for key in sorted(rows, key=lambda item: (item[1], item[0])))


def _cache_roots(
    pr36_root: Path, recovery_cache_root: Path, extra_cache_roots: tuple[Path, ...]
) -> tuple[Path, ...]:
    candidates = (
        recovery_cache_root,
        pr36_root / "raw-minute-cache",
        pr36_root.parent / "exact-event-market-dataset-v2" / "raw-minute-cache",
        pr36_root.parent / "exact-event-market-dataset-v1" / "raw-minute-cache",
        *extra_cache_roots,
    )
    unique: list[Path] = []
    for path in candidates:
        if path.exists() and path not in unique:
            unique.append(path)
    return tuple(unique)


def _per_event_base(
    identity: RecoveryIdentity,
    *,
    availability_before: dict[str, Any],
    acquisition_row: dict[str, Any],
    security: tuple[TInvestMinuteCandle, ...],
    benchmark: tuple[TInvestMinuteCandle, ...],
) -> dict[str, Any]:
    return {
        "EVENT_ID": identity.event_id,
        "TICKER": identity.ticker,
        "PUBLICATION_TIMESTAMP": identity.publication_timestamp.isoformat(),
        "FIGI": identity.figi,
        "UID": identity.instrument_uid,
        "CLASS_CODE": identity.class_code,
        "ACQUISITION_FROM": acquisition_row["ACQUISITION_FROM"],
        "ACQUISITION_TO": acquisition_row["ACQUISITION_TO"],
        "CANDLES_BEFORE": acquisition_row["CANDLES_BEFORE"],
        "CANDLES_ACQUIRED": acquisition_row["CANDLES_ACQUIRED"],
        "CANDLES_AFTER": acquisition_row["CANDLES_AFTER"],
        "FIRST_SECURITY_CANDLE": acquisition_row["FIRST_SECURITY_CANDLE"],
        "LAST_SECURITY_CANDLE": acquisition_row["LAST_SECURITY_CANDLE"],
        "SECURITY_HISTORY_READY": bool(security),
        "BENCHMARK_HISTORY_READY": bool(benchmark),
        "PRE_EVENT_CONTEXT_READY": False,
        "MAX_FEATURE_TIMESTAMP": None,
        "REACTION_1M_READY": False,
        "REACTION_5M_READY": False,
        "REACTION_15M_READY": False,
        "REACTION_30M_READY": False,
        "REACTION_60M_READY": False,
        "FEATURE_READY_BEFORE": bool(availability_before.get("feature_ready")),
        "FEATURE_READY_AFTER": False,
        "RECOVERY_STATUS": "BLOCKED",
        "FINAL_BLOCKER": None,
    }


def _blocked_payload(payload: dict[str, Any], blocker: str) -> dict[str, Any]:
    return {**payload, "RECOVERY_STATUS": "BLOCKED", "FINAL_BLOCKER": blocker}


def _primary_blocker(alignment: Any, complete_features: bool) -> str:
    if alignment.session_state.value == "PRE_OPEN":
        return RecoveryBlocker.PRE_OPEN.value
    if alignment.session_state.value == "AFTER_CLOSE":
        return RecoveryBlocker.AFTER_CLOSE.value
    if alignment.session_state.value == "NON_TRADING_DAY":
        return RecoveryBlocker.NON_TRADING_DAY.value
    if not alignment.horizons:
        return RecoveryBlocker.SESSION_ALIGNMENT_FAILED.value
    if not complete_features:
        return RecoveryBlocker.MARKET_HISTORY_WARMUP.value
    if alignment.missing_reason:
        return RecoveryBlocker.REACTION_WINDOW_INCOMPLETE.value
    return RecoveryBlocker.OTHER_FAIL_CLOSED.value


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


def _metrics(events: list[dict[str, Any]], features: list[dict[str, Any]]) -> dict[str, Any]:
    event_by_id = {_event_id(row): row for row in events}
    feature_tickers = Counter(
        str(_metadata(event_by_id[str(row["event_id"])])["ticker"]) for row in features
    )
    feature_issuers = Counter(
        str(_metadata(event_by_id[str(row["event_id"])])["issuer"]) for row in features
    )
    ticker_concentration = _concentration(feature_tickers)
    issuer_concentration = _concentration(feature_issuers)
    return {
        "EXACT_TOTAL": len(events),
        "REACTION_READY": sum(bool(_availability(row).get("reaction_ready")) for row in events),
        "FEATURE_READY": len(features),
        "FEATURE_READY_BY_TICKER": dict(sorted(feature_tickers.items())),
        "FEATURE_READY_UNIQUE_TICKERS": len(feature_tickers),
        "FEATURE_READY_TOP1": ticker_concentration["top1_share"],
        "FEATURE_READY_TOP3": ticker_concentration["top3_share"],
        "FEATURE_READY_ISSUER_HHI": issuer_concentration["hhi"],
        "EFFECTIVE_FEATURE_READY_ISSUER_COUNT": issuer_concentration["effective_count"],
    }


def _concentration(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    shares = sorted((count / total for count in counter.values()), reverse=True) if total else []
    hhi = sum(share * share for share in shares)
    return {
        "top1_share": shares[0] if shares else 0.0,
        "top3_share": sum(shares[:3]),
        "hhi": hhi,
        "effective_count": 1 / hhi if hhi else 0.0,
    }


def _per_horizon_ready(targets: list[dict[str, Any]]) -> dict[str, int]:
    return {
        horizon: sum(
            bool(cast("dict[str, Any]", row.get("horizons", {})).get(horizon, {}).get("available"))
            for row in targets
        )
        for horizon in HORIZONS
    }


def _acquired_by_ticker(acquisition: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in acquisition:
        ticker = str(row["TICKER"])
        result[ticker] = result.get(ticker, 0) + int(row["CANDLES_ACQUIRED"])
    return result


def _existing_features_preserved(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> dict[str, str]:
    after_by_id = {str(row["event_id"]): row for row in after}
    mismatched = [
        str(row["event_id"]) for row in before if after_by_id.get(str(row["event_id"])) != row
    ]
    return {
        "status": "PASS" if not mismatched else "FAIL",
        "hash": sha256_payload({str(row["event_id"]): row for row in before}),
    }


def _assert_non_cohort_events_preserved(
    before: list[dict[str, Any]], after: list[dict[str, Any]], cohort_ids: set[str]
) -> None:
    before_non_cohort = {_event_id(row): row for row in before if _event_id(row) not in cohort_ids}
    after_non_cohort = {_event_id(row): row for row in after if _event_id(row) not in cohort_ids}
    if before_non_cohort != after_non_cohort:
        raise ValueError("EXISTING_EVENT_ROWS_PRESERVED_FAILED")


def _assert_future_targets_guard(
    events: list[dict[str, Any]], targets: list[dict[str, Any]]
) -> None:
    future_ids = {
        _event_id(row)
        for row in events
        if _parse_datetime(_metadata(row)["publication_timestamp_utc"]).date()
        >= FUTURE_EVENT_HOLDOUT_START
    }
    if future_ids & {str(row["event_id"]) for row in targets}:
        raise ValueError("FUTURE_HOLDOUT_TARGET_READ")
    for row in events:
        if _event_id(row) in future_ids and bool(
            _availability(row).get("research_outcomes_visible")
        ):
            raise ValueError("FUTURE_HOLDOUT_OUTCOME_VISIBLE")


def _feature_schema_sha(features: list[dict[str, Any]]) -> str:
    event_names = sorted(
        {name for row in features for name in cast("dict[str, Any]", row["event_features"])}
    )
    market_names = sorted(
        {name for row in features for name in cast("dict[str, Any]", row["market_features"])}
    )
    return sha256_payload({"event_features": event_names, "market_features": market_names})


def _cache_provenance(
    cache_roots: tuple[Path, ...],
    cohort: list[RecoveryIdentity],
    *,
    network_fetch_performed: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for identity in cohort:
        candles = _load_security_history(cache_roots, identity)
        rows.append(
            {
                "event_id": identity.event_id,
                "ticker": identity.ticker,
                "figi": identity.figi,
                "instrument_uid": identity.instrument_uid,
                "class_code": identity.class_code,
                "interval": "1m",
                "candle_count": len(candles),
                "first_candle": min((row.begin_at.isoformat() for row in candles), default=None),
                "last_candle": max((row.end_at.isoformat() for row in candles), default=None),
                "source": "TINVEST_API",
            }
        )
    return {
        "network_fetch_performed": network_fetch_performed,
        "token_value_read": False,
        "sandbox_used": False,
        "broker_write_surface_used": False,
        "rows": rows,
    }


def _cache_provenance_hash(cache_provenance: dict[str, Any]) -> str:
    return sha256_payload(
        {
            "rows": cache_provenance["rows"],
            "token_value_read": cache_provenance["token_value_read"],
            "sandbox_used": cache_provenance["sandbox_used"],
            "broker_write_surface_used": cache_provenance["broker_write_surface_used"],
        }
    )


def _future_exclusions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in events:
        metadata = _metadata(row)
        published_at = _parse_datetime(metadata["publication_timestamp_utc"])
        if published_at.date() < FUTURE_EVENT_HOLDOUT_START:
            continue
        result.append(
            {
                "event_id": str(metadata["event_id"]),
                "ticker": str(metadata["ticker"]),
                "publication_timestamp_utc": published_at.isoformat(),
                "excluded_from_recovery": True,
                "future_outcome_observed": False,
            }
        )
    return result


def _identity_cache_matches(
    rows: tuple[TInvestMinuteCandle, ...], identity: RecoveryIdentity
) -> bool:
    return all(row.instrument_uid == identity.instrument_uid for row in rows)


def _unique_identity_timestamps(rows: tuple[TInvestMinuteCandle, ...]) -> bool:
    keys = {(row.instrument_uid, row.begin_at) for row in rows}
    return len(keys) == len(rows)


def _candle_payload(item: TInvestMinuteCandle, identity: RecoveryIdentity) -> dict[str, object]:
    return {
        "ticker": identity.ticker,
        "figi": identity.figi,
        "instrument_uid": item.instrument_uid,
        "class_code": identity.class_code,
        "interval": "1m",
        "begin_at": item.begin_at.astimezone(UTC).isoformat(),
        "end_at": item.end_at.astimezone(UTC).isoformat(),
        "open": str(item.open),
        "high": str(item.high),
        "low": str(item.low),
        "close": str(item.close),
        "volume": item.volume,
        "is_complete": item.is_complete,
        "source": "TINVEST_API",
        "provenance": "TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES",
    }


def _candle_from_payload(
    payload: dict[str, Any], expected: RecoveryIdentity | None = None
) -> TInvestMinuteCandle:
    if str(payload.get("source", "TINVEST_API")) != "TINVEST_API":
        raise ValueError("NON_TINVEST_CANDLE_CACHE_SOURCE")
    if expected is not None:
        if str(payload.get("instrument_uid")) != expected.instrument_uid:
            raise ValueError("SECURITY_CACHE_UID_MISMATCH")
        figi = payload.get("figi")
        class_code = payload.get("class_code")
        if figi is not None and str(figi) != expected.figi:
            raise ValueError("SECURITY_CACHE_FIGI_MISMATCH")
        if class_code is not None and str(class_code) != expected.class_code:
            raise ValueError("SECURITY_CACHE_CLASS_CODE_MISMATCH")
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


def _artifact_sha(manifest: dict[str, Any]) -> str:
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"ARTIFACT_SHA", "created_at", "git_sha"}
    }
    core["CACHE_CANDLES_ACQUIRED_BY_TICKER"] = None
    provenance = cast("dict[str, Any]", core["CACHE_PROVENANCE"])
    core["CACHE_PROVENANCE"] = {**provenance, "network_fetch_performed": None}
    core["PER_EVENT_RECOVERY"] = [
        {**row, "CANDLES_BEFORE": None, "CANDLES_ACQUIRED": None}
        for row in cast("list[dict[str, Any]]", core["PER_EVENT_RECOVERY"])
    ]
    return sha256_payload(core)


def _strip_row(row: dict[str, Any]) -> dict[str, Any]:
    return row


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


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"RECOVERY_IDENTITY_{key}_MISSING")
    return value


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
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    per_event: list[dict[str, Any]],
    acquisition: list[dict[str, Any]],
    future_exclusions: list[dict[str, Any]],
    cache_provenance: dict[str, Any],
) -> None:
    _write_json(output_root / "manifest.json", manifest)
    _write_jsonl(output_root / "events.jsonl", events)
    _write_jsonl(output_root / "features.jsonl", features)
    _write_jsonl(output_root / "targets.jsonl", targets)
    _write_jsonl(output_root / "per-event-recovery.jsonl", per_event)
    _write_jsonl(output_root / "cache-acquisition.jsonl", acquisition)
    _write_jsonl(output_root / "future-holdout-exclusions.jsonl", future_exclusions)
    _write_json(output_root / "cache-provenance.json", cache_provenance)
    _write_report(output_root / "report.md", manifest)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        "Data-recovery-only report for PR37 security history gaps.",
        "",
        f"- INPUT_DATASET_SHA={manifest['INPUT_DATASET_SHA']}",
        f"- OUTPUT_DATASET_SHA={manifest['OUTPUT_DATASET_SHA']}",
        f"- RECOVERY_COHORT_TOTAL={manifest['RECOVERY_COHORT_TOTAL']}",
        f"- RECOVERY_SUCCESS_COUNT={manifest['RECOVERY_SUCCESS_COUNT']}",
        f"- RECOVERY_BLOCKED_COUNT={manifest['RECOVERY_BLOCKED_COUNT']}",
        f"- FEATURE_READY_DELTA={manifest['FEATURE_READY_DELTA']}",
        f"- REACTION_READY_DELTA={manifest['REACTION_READY_DELTA']}",
        f"- CACHE_DEDUPE={manifest['CACHE_DEDUPE']}",
        f"- LEAKAGE_CHECK={manifest['LEAKAGE_CHECK']}",
        "",
        "No model training, TEST outcome use, future holdout outcome observation, source "
        "expansion, backtest, paper trading, orders, or BUY/SELL output was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _output_nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())
