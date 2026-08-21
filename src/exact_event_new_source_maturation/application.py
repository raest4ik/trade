from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_corpus.market import align_exact_event
from src.exact_event_new_source_maturation.domain import (
    ARTIFACT_VERSION,
    FUTURE_EVENT_HOLDOUT_START,
    HORIZONS,
    INPUT_DATASET_SHA,
    OUTPUT_DATASET_VERSION,
    PREVIOUS_DATASET_SHA,
    concentration,
    maturation_safety_flags,
    require_input_manifests,
    sha256_payload,
)
from src.exact_event_warmup_recovery.domain import acquisition_dates
from src.tinvest_market.client import TInvestMinuteCandle


def run_new_source_maturation(
    *,
    previous_root: Path,
    current_root: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    created_at: datetime | None = None,
    extra_cache_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable new source maturation artifact output already exists")
    _verify_frozen_contracts()
    previous_manifest = _read_json(previous_root / "manifest.json")
    current_manifest = _read_json(current_root / "manifest.json")
    require_input_manifests(previous_manifest, current_manifest)
    previous_events = _read_jsonl(previous_root / "events.jsonl")
    current_events = _read_jsonl(current_root / "events.jsonl")
    current_features = _read_jsonl(current_root / "features.jsonl")
    current_targets = _read_jsonl(current_root / "targets.jsonl")
    old_ids = {_event_id(row) for row in previous_events}
    cohort = [row for row in current_events if _event_id(row) not in old_ids]
    cohort_ids = [_event_id(row) for row in cohort]
    if not cohort:
        raise ValueError("NEW_EVENT_COHORT_EMPTY")

    events_after = deepcopy(current_events)
    event_by_id = {_event_id(row): row for row in events_after}
    features_after = [*current_features]
    targets_after = [*current_targets]
    cache_roots = _cache_roots(current_root, previous_root, extra_cache_roots)
    per_event: list[dict[str, Any]] = []
    recovered_event_ids: list[str] = []
    blocked_event_ids: list[str] = []
    leakage_violations: list[str] = []

    for row in cohort:
        metadata = _metadata(row)
        event_id = str(metadata["event_id"])
        published_at = _parse_datetime(metadata["publication_timestamp_utc"])
        original_availability = cast("dict[str, Any]", row["target_availability"])
        future = published_at.date() >= FUTURE_EVENT_HOLDOUT_START
        if future:
            status = _future_status(row)
            per_event.append(status)
            blocked_event_ids.append(event_id)
            continue
        security = _load_history(cache_roots, str(metadata["ticker"]), published_at)
        benchmark = _load_history(cache_roots, "IMOEX", published_at)
        if not security:
            status = _blocked_status(row, "MARKET_HISTORY_MISSING", security, benchmark)
            per_event.append(status)
            blocked_event_ids.append(event_id)
            continue
        if not benchmark:
            status = _blocked_status(row, "BENCHMARK_HISTORY_MISSING", security, benchmark)
            per_event.append(status)
            blocked_event_ids.append(event_id)
            continue
        alignment = align_exact_event(published_at, security, benchmark, expose_outcomes=True)
        max_feature_input_at = _max_feature_input_timestamp(published_at, security, benchmark)
        if max_feature_input_at is not None and max_feature_input_at >= published_at:
            leakage_violations.append(event_id)
        horizon_ready = {
            horizon: bool(alignment.horizons.get(horizon, {}).get("available", False))
            for horizon in HORIZONS
        }
        horizon_blockers = {
            horizon: (
                None
                if horizon_ready[horizon]
                else str(
                    alignment.horizons.get(horizon, {}).get("reason", "REACTION_WINDOW_INCOMPLETE")
                )
            )
            for horizon in HORIZONS
        }
        complete_features = _complete_pre_event_features(alignment.features)
        reaction_ready = alignment.reaction_status == "REACTION_READY"
        feature_ready = (
            complete_features
            and reaction_ready
            and max_feature_input_at is not None
            and max_feature_input_at < published_at
        )
        final_row = event_by_id[event_id]
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
        cast("dict[str, Any]", final_row["quality"])["feature_cutoff"] = published_at.isoformat()
        cast("dict[str, Any]", final_row["quality"])["no_forward_fill"] = True
        cast("dict[str, Any]", final_row["quality"])["no_interpolation"] = True
        cast("dict[str, Any]", final_row["quality"])["no_source_mixing"] = True
        if alignment.horizons:
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
                    "event_features": final_row["event_features"],
                    "market_features": alignment.features,
                }
            )
            recovered_event_ids.append(event_id)
        else:
            blocked_event_ids.append(event_id)
        per_event.append(
            {
                **_event_metadata_status(row),
                "historical_or_future": "HISTORICAL",
                "original_reaction_ready": bool(original_availability["reaction_ready"]),
                "original_feature_ready": bool(original_availability["feature_ready"]),
                "final_reaction_ready": reaction_ready,
                "final_feature_ready": feature_ready,
                "session_classification": alignment.session_state.value,
                "primary_readiness_blocker": (
                    None if feature_ready else _primary_blocker(alignment, complete_features)
                ),
                "horizon_ready": horizon_ready,
                "horizon_blockers": horizon_blockers,
                "market_context_acquisition_status": "CACHE_HIT",
                "security_history_available": bool(security),
                "benchmark_history_available": bool(benchmark),
                "sufficient_warmup": complete_features,
                "max_feature_timestamp_utc": (
                    max_feature_input_at.isoformat() if max_feature_input_at is not None else None
                ),
                "no_forward_fill": True,
                "no_moex_substitution": True,
                "post_event_feature_access": False,
                "strict_feature_timestamp_before_publication": (
                    max_feature_input_at is not None and max_feature_input_at < published_at
                ),
            }
        )

    if leakage_violations:
        raise ValueError("NEW_SOURCE_MATURATION_LEAKAGE_CHECK_FAILED")
    _assert_non_cohort_events_preserved(current_events, events_after, set(cohort_ids))
    preserved_features = _existing_features_preserved(current_features, features_after)
    if preserved_features["status"] != "PASS":
        raise ValueError("EXISTING_FEATURE_ROWS_PRESERVED_FAILED")
    _assert_future_targets_guard(events_after, targets_after)

    before = _metrics(current_events, current_features)
    after = _metrics(events_after, features_after)
    per_horizon_ready_counts = {
        horizon: sum(bool(item["horizon_ready"].get(horizon)) for item in per_event)
        for horizon in HORIZONS
    }
    output_dataset_sha = sha256_payload(
        {
            "dataset_version": OUTPUT_DATASET_VERSION,
            "input_dataset_sha": INPUT_DATASET_SHA,
            "events": events_after,
            "features": features_after,
            "targets": targets_after,
        }
    )
    existing_event_hash = sha256_payload(
        [_strip_cohort(row) for row in current_events if _event_id(row) not in set(cohort_ids)]
    )
    feature_schema_sha = _feature_schema_sha(features_after)
    safety = maturation_safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_DATASET_SHA": INPUT_DATASET_SHA,
        "PREVIOUS_DATASET_SHA": PREVIOUS_DATASET_SHA,
        "OUTPUT_DATASET_VERSION": OUTPUT_DATASET_VERSION,
        "OUTPUT_DATASET_SHA": output_dataset_sha,
        "INPUT_NEW_EVENT_COHORT_SHA": sha256_payload(sorted(cohort_ids)),
        "NEW_EVENT_IDS": sorted(cohort_ids),
        "NEW_EVENTS_TOTAL": len(cohort),
        "NEW_EVENTS_HISTORICAL": sum(
            item["historical_or_future"] == "HISTORICAL" for item in per_event
        ),
        "NEW_EVENTS_FUTURE_METADATA_ONLY": sum(
            item["historical_or_future"] == "FUTURE_METADATA_ONLY" for item in per_event
        ),
        "PER_EVENT_STATUS": per_event,
        "RECOVERED_EVENT_IDS": sorted(recovered_event_ids),
        "BLOCKED_EVENT_IDS": sorted(blocked_event_ids),
        "PER_HORIZON_READY_COUNTS": per_horizon_ready_counts,
        "EXACT_TOTAL_BEFORE": before["EXACT_TOTAL"],
        "EXACT_TOTAL_AFTER": after["EXACT_TOTAL"],
        "REACTION_READY_BEFORE": before["REACTION_READY"],
        "REACTION_READY_AFTER": after["REACTION_READY"],
        "REACTION_READY_DELTA": after["REACTION_READY"] - before["REACTION_READY"],
        "FEATURE_READY_BEFORE": before["FEATURE_READY"],
        "FEATURE_READY_AFTER": after["FEATURE_READY"],
        "FEATURE_READY_DELTA": after["FEATURE_READY"] - before["FEATURE_READY"],
        "NEW_EVENTS_REACTION_READY_BEFORE": _cohort_ready(cohort, "reaction_ready"),
        "NEW_EVENTS_REACTION_READY_AFTER": _cohort_ready(
            [event_by_id[event_id] for event_id in cohort_ids], "reaction_ready"
        ),
        "NEW_EVENTS_FEATURE_READY_BEFORE": _cohort_ready(cohort, "feature_ready"),
        "NEW_EVENTS_FEATURE_READY_AFTER": _cohort_ready(
            [event_by_id[event_id] for event_id in cohort_ids], "feature_ready"
        ),
        "FEATURE_READY_BY_TICKER_BEFORE": before["FEATURE_READY_BY_TICKER"],
        "FEATURE_READY_BY_TICKER_AFTER": after["FEATURE_READY_BY_TICKER"],
        "FEATURE_READY_UNIQUE_TICKERS_BEFORE": before["FEATURE_READY_UNIQUE_TICKERS"],
        "FEATURE_READY_UNIQUE_TICKERS_AFTER": after["FEATURE_READY_UNIQUE_TICKERS"],
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
        "EXISTING_EVENT_ROWS_PRESERVED": "PASS",
        "EXISTING_FEATURE_ROWS_PRESERVED": preserved_features["status"],
        "existing_event_rows_preservation_hash": existing_event_hash,
        "existing_feature_rows_preservation_hash": preserved_features["hash"],
        "FEATURE_SCHEMA_SHA": feature_schema_sha,
        "LEAKAGE_CHECK": "PASS",
        "safety": safety,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = sha256_payload({**manifest, "ARTIFACT_SHA": None})
    _write_artifacts(output_root, events_after, features_after, targets_after, per_event, manifest)
    return manifest


def _future_status(row: dict[str, Any]) -> dict[str, Any]:
    availability = cast("dict[str, Any]", row["target_availability"])
    return {
        **_event_metadata_status(row),
        "historical_or_future": "FUTURE_METADATA_ONLY",
        "original_reaction_ready": bool(availability["reaction_ready"]),
        "original_feature_ready": bool(availability["feature_ready"]),
        "final_reaction_ready": False,
        "final_feature_ready": False,
        "session_classification": str(_metadata(row).get("session_state")),
        "primary_readiness_blocker": "FUTURE_METADATA_ONLY",
        "horizon_ready": {horizon: False for horizon in HORIZONS},
        "horizon_blockers": {horizon: "FUTURE_METADATA_ONLY" for horizon in HORIZONS},
        "market_context_acquisition_status": "SKIPPED_FUTURE_HOLDOUT",
        "security_history_available": None,
        "benchmark_history_available": None,
        "sufficient_warmup": None,
        "max_feature_timestamp_utc": None,
        "no_forward_fill": True,
        "no_moex_substitution": True,
        "post_event_feature_access": False,
    }


def _blocked_status(
    row: dict[str, Any],
    blocker: str,
    security: tuple[TInvestMinuteCandle, ...],
    benchmark: tuple[TInvestMinuteCandle, ...],
) -> dict[str, Any]:
    availability = cast("dict[str, Any]", row["target_availability"])
    return {
        **_event_metadata_status(row),
        "historical_or_future": "HISTORICAL",
        "original_reaction_ready": bool(availability["reaction_ready"]),
        "original_feature_ready": bool(availability["feature_ready"]),
        "final_reaction_ready": False,
        "final_feature_ready": False,
        "session_classification": "MARKET_HISTORY_UNAVAILABLE",
        "primary_readiness_blocker": blocker,
        "horizon_ready": {horizon: False for horizon in HORIZONS},
        "horizon_blockers": {horizon: blocker for horizon in HORIZONS},
        "market_context_acquisition_status": blocker,
        "security_history_available": bool(security),
        "benchmark_history_available": bool(benchmark),
        "sufficient_warmup": False,
        "max_feature_timestamp_utc": None,
        "no_forward_fill": True,
        "no_moex_substitution": True,
        "post_event_feature_access": False,
    }


def _event_metadata_status(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(row)
    return {
        "event_id": str(metadata["event_id"]),
        "ticker": str(metadata["ticker"]),
        "issuer": str(metadata["issuer"]),
        "source": str(metadata["source_code"]),
        "publication_timestamp_utc": str(metadata["publication_timestamp_utc"]),
        "timestamp_provenance": str(metadata["timestamp_source_field"]),
        "future_holdout": bool(metadata.get("future_holdout")),
    }


def _load_history(
    cache_roots: tuple[Path, ...], ticker: str, published_at: datetime
) -> tuple[TInvestMinuteCandle, ...]:
    rows: dict[tuple[str, datetime], TInvestMinuteCandle] = {}
    for day in acquisition_dates(published_at, max_history_days=7):
        for root in cache_roots:
            for suffix in ("day", "pre"):
                path = root / ticker / f"{day.isoformat()}-{suffix}.jsonl"
                if not path.exists():
                    continue
                for payload in _read_jsonl(path):
                    candle = _candle_from_payload(payload)
                    rows[(candle.instrument_uid, candle.begin_at)] = candle
    return tuple(rows[key] for key in sorted(rows, key=lambda item: (item[1], item[0])))


def _cache_roots(
    current_root: Path, previous_root: Path, extra_cache_roots: tuple[Path, ...]
) -> tuple[Path, ...]:
    candidates = (
        current_root / "raw-minute-cache",
        current_root.parent / "exact-event-market-dataset-v2" / "raw-minute-cache",
        previous_root.parent / "exact-event-market-dataset-v2" / "raw-minute-cache",
        previous_root.parent / "exact-event-market-dataset-v1" / "raw-minute-cache",
        *extra_cache_roots,
    )
    unique: list[Path] = []
    for path in candidates:
        if path.exists() and path not in unique:
            unique.append(path)
    return tuple(unique)


def _primary_blocker(alignment: Any, complete_features: bool) -> str:
    if alignment.session_state.value == "PRE_OPEN":
        return "PRE_OPEN"
    if alignment.session_state.value == "AFTER_CLOSE":
        return "AFTER_CLOSE"
    if alignment.session_state.value == "NON_TRADING_DAY":
        return "NON_TRADING_DAY"
    if not complete_features:
        return "MARKET_HISTORY_WARMUP"
    if alignment.missing_reason:
        return str(alignment.missing_reason)
    return "OTHER_FAIL_CLOSED"


def _metrics(events: list[dict[str, Any]], features: list[dict[str, Any]]) -> dict[str, Any]:
    event_by_id = {_event_id(row): row for row in events}
    feature_tickers = Counter(
        str(_metadata(event_by_id[str(row["event_id"])])["ticker"]) for row in features
    )
    feature_issuers = Counter(
        str(_metadata(event_by_id[str(row["event_id"])])["issuer"]) for row in features
    )
    ticker_concentration = concentration(feature_tickers)
    issuer_concentration = concentration(feature_issuers)
    return {
        "EXACT_TOTAL": len(events),
        "REACTION_READY": sum(
            bool(cast("dict[str, Any]", row["target_availability"]).get("reaction_ready"))
            for row in events
        ),
        "FEATURE_READY": len(features),
        "FEATURE_READY_BY_TICKER": dict(sorted(feature_tickers.items())),
        "FEATURE_READY_UNIQUE_TICKERS": len(feature_tickers),
        "FEATURE_READY_TOP1": ticker_concentration["top1_share"],
        "FEATURE_READY_TOP3": ticker_concentration["top3_share"],
        "FEATURE_READY_ISSUER_HHI": issuer_concentration["hhi"],
        "EFFECTIVE_FEATURE_READY_ISSUER_COUNT": issuer_concentration["effective_count"],
    }


def _cohort_ready(rows: list[dict[str, Any]], field: str) -> int:
    return sum(bool(cast("dict[str, Any]", row["target_availability"]).get(field)) for row in rows)


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
    before_non_cohort = {
        str(_event_id(row)): row for row in before if _event_id(row) not in cohort_ids
    }
    after_non_cohort = {
        str(_event_id(row)): row for row in after if _event_id(row) not in cohort_ids
    }
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
    target_ids = {str(row["event_id"]) for row in targets}
    if future_ids & target_ids:
        raise ValueError("FUTURE_HOLDOUT_TARGET_READ")
    for row in events:
        if _event_id(row) in future_ids and cast("dict[str, Any]", row["target_availability"]).get(
            "research_outcomes_visible"
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


def _strip_cohort(row: dict[str, Any]) -> dict[str, Any]:
    return row


def _verify_frozen_contracts() -> None:
    if rules_v3_fingerprint() != EXPECTED_RULES_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_MISMATCH")
    if prompt_hash() != QWEN_PROMPT_SHA or schema_hash() != QWEN_SCHEMA_SHA:
        raise ValueError("FROZEN_QWEN_CONTRACT_MISMATCH")


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["metadata"])


def _event_id(row: dict[str, Any]) -> str:
    return str(_metadata(row)["event_id"])


def _parse_datetime(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _candle_from_payload(payload: dict[str, Any]) -> TInvestMinuteCandle:
    if str(payload.get("source", "TINVEST_API")) != "TINVEST_API":
        raise ValueError("NON_TINVEST_CANDLE_CACHE_SOURCE")
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
    output_root: Path,
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    per_event: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "events.jsonl", events)
    _write_jsonl(output_root / "features.jsonl", features)
    _write_jsonl(output_root / "targets.jsonl", targets)
    _write_jsonl(output_root / "per-event-status.jsonl", per_event)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        "Data-maturation-only report for PR35 new exact source events.",
        "",
        f"- INPUT_DATASET_SHA={manifest['INPUT_DATASET_SHA']}",
        f"- OUTPUT_DATASET_SHA={manifest['OUTPUT_DATASET_SHA']}",
        f"- NEW_EVENTS_TOTAL={manifest['NEW_EVENTS_TOTAL']}",
        f"- NEW_EVENTS_HISTORICAL={manifest['NEW_EVENTS_HISTORICAL']}",
        f"- NEW_EVENTS_FUTURE_METADATA_ONLY={manifest['NEW_EVENTS_FUTURE_METADATA_ONLY']}",
        f"- FEATURE_READY_DELTA={manifest['FEATURE_READY_DELTA']}",
        f"- REACTION_READY_DELTA={manifest['REACTION_READY_DELTA']}",
        f"- EXACT_V3_PRESERVED={manifest['EXACT_V3_PRESERVED']}",
        f"- EXISTING_EVENT_ROWS_PRESERVED={manifest['EXISTING_EVENT_ROWS_PRESERVED']}",
        f"- EXISTING_FEATURE_ROWS_PRESERVED={manifest['EXISTING_FEATURE_ROWS_PRESERVED']}",
        f"- LEAKAGE_CHECK={manifest['LEAKAGE_CHECK']}",
        "",
        (
            "No model training, TEST outcome use, future holdout outcome observation, source "
            "expansion, backtest, paper trading, orders, or BUY/SELL output was performed."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
