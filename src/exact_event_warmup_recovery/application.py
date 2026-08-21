from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from src.exact_event_corpus.market import align_exact_event
from src.exact_event_warmup_recovery.domain import (
    ARTIFACT_VERSION,
    EXPECTED_INPUT_DATASET_SHA,
    OUTPUT_DATASET_VERSION,
    RecoveryConfig,
    WarmupEventRootCause,
    acquisition_dates,
    earliest_required_timestamp,
    recovery_safety_flags,
    require_input_manifest,
    sha256_file,
    sha256_payload,
)
from src.tinvest_market.client import TInvestMinuteCandle


def run_warmup_recovery(
    dataset_root: Path,
    output_root: Path,
    *,
    base_main_sha: str,
    git_sha: str,
    baseline_root: Path | None = None,
    v1_dataset_root: Path | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable warmup recovery artifact output already exists")
    manifest = _read_json(dataset_root / "manifest.json")
    require_input_manifest(manifest)
    events_before = _read_jsonl(dataset_root / "events.jsonl")
    features_before = _read_jsonl(dataset_root / "features.jsonl")
    features_before_by_id = {str(row["event_id"]): row for row in features_before}
    split_assignments = _split_assignments(baseline_root) if baseline_root is not None else {}
    config = RecoveryConfig(base_main_sha=base_main_sha)
    cache_roots = _cache_roots(dataset_root, v1_dataset_root)
    affected = _warmup_events(events_before)
    if len(events_before) != 706 or len(features_before) != 408 or len(affected) != 157:
        raise ValueError("BASELINE_WARMUP_ACCOUNTING_MISMATCH")

    root_causes = [
        _root_cause(
            row, _load_history(cache_roots, str(_metadata(row)["ticker"]), row), cache_roots
        )
        for row in affected
    ]
    events_after = deepcopy(events_before)
    event_after_by_id = {_event_id(row): row for row in events_after}
    recovered_features: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    leakage_violations: list[str] = []

    for row in affected:
        metadata = _metadata(row)
        event_id = str(metadata["event_id"])
        published_at = _parse_datetime(metadata["publication_timestamp_utc"])
        ticker = str(metadata["ticker"])
        security = _load_history(cache_roots, ticker, row)
        benchmark = _load_history(cache_roots, "IMOEX", row)
        alignment = align_exact_event(published_at, security, benchmark, expose_outcomes=False)
        complete = _complete_pre_event_features(alignment.features)
        max_input_at = _max_feature_input_timestamp(published_at, security, benchmark)
        if max_input_at is not None and max_input_at > published_at:
            leakage_violations.append(event_id)
        if complete and max_input_at is not None and max_input_at <= published_at:
            updated = event_after_by_id[event_id]
            cast("dict[str, Any]", updated["target_availability"])["feature_ready"] = True
            updated["pre_event_market_features"] = alignment.features
            cast("dict[str, Any]", updated["quality"])["feature_cutoff"] = published_at.isoformat()
            recovered_features.append(
                {
                    "event_id": event_id,
                    "feature_cutoff": published_at.isoformat(),
                    "event_features": updated["event_features"],
                    "market_features": alignment.features,
                }
            )
        else:
            remaining.append(_remaining_payload(row, alignment.features, security, benchmark))

    feature_rows_after = [*features_before, *recovered_features]
    preserved = _existing_rows_preserved(features_before_by_id, feature_rows_after)
    if preserved["status"] != "PASS":
        raise ValueError("EXISTING_FEATURE_ROWS_PRESERVED_FAILED")
    if leakage_violations:
        raise ValueError("RECOVERY_LEAKAGE_CHECK_FAILED")

    train_val_last_date = _train_val_last_date(events_before, split_assignments)
    train_val_warmup_ids = {
        _event_id(row)
        for row in affected
        if train_val_last_date is not None
        and str(_metadata(row).get("publication_date", "")) <= train_val_last_date
    }
    recovered_ids = {str(row["event_id"]) for row in recovered_features}
    remaining_ids = {str(row["event_id"]) for row in remaining}
    affected_ids = {_event_id(row) for row in affected}
    warmup_recovered = len(recovered_features)
    warmup_remaining = len(remaining)
    before_counts = _counts(events_before, features_before)
    if warmup_recovered + warmup_remaining != len(affected):
        raise ValueError("WARMUP_RECONCILIATION_FAILED")
    concentration_before = _concentration_scopes(events_before, features_before, split_assignments)
    concentration_after = _concentration_scopes(events_after, feature_rows_after, split_assignments)
    output_dataset_sha = sha256_payload(
        {
            "dataset_version": OUTPUT_DATASET_VERSION,
            "input_dataset_sha": EXPECTED_INPUT_DATASET_SHA,
            "events": events_after,
            "features": feature_rows_after,
            "targets_file_sha256": sha256_file(dataset_root / "targets.jsonl"),
        }
    )
    generated_at = (created_at or datetime.now(UTC)).isoformat()
    recovery_config = config.payload()
    recovery_config_sha = sha256_payload(recovery_config)
    artifact_manifest: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": generated_at,
        "git_sha": git_sha,
        "BASE_MAIN_SHA": base_main_sha,
        "INPUT_DATASET_SHA": EXPECTED_INPUT_DATASET_SHA,
        "OUTPUT_DATASET_VERSION": OUTPUT_DATASET_VERSION,
        "OUTPUT_DATASET_SHA": output_dataset_sha,
        "input_feature_schema_sha": _feature_schema_sha(features_before),
        "RECOVERY_CONFIG_SHA": recovery_config_sha,
        "recovery_config": recovery_config,
        "AFFECTED_COHORT_SHA": sha256_payload(sorted(affected_ids)),
        "recovered_ids_hash": sha256_payload(sorted(recovered_ids)),
        "remaining_ids_hash": sha256_payload(sorted(remaining_ids)),
        "T_INVEST_PROVENANCE": {
            "source": "TINVEST_READONLY_PRODUCTION_EXCHANGE_CANDLES",
            "cache_roots": [str(path) for path in cache_roots],
            "source_cost_rub": 0,
            "network_fetch_performed": False,
            "token_value_read": False,
        },
        "EXACT_TOTAL": len(events_before),
        "REACTION_READY": before_counts["reaction_ready"],
        "FEATURE_READY_BEFORE": len(features_before),
        "FEATURE_READY_AFTER": len(feature_rows_after),
        "FEATURE_READY_DELTA": len(feature_rows_after) - len(features_before),
        "WARMUP_LOST_BEFORE": len(affected),
        "WARMUP_RECOVERED": warmup_recovered,
        "WARMUP_REMAINING": warmup_remaining,
        "WARMUP_RECONCILIATION": "PASS",
        "TRAIN_VAL_WARMUP_LOST_BEFORE": len(train_val_warmup_ids),
        "TRAIN_VAL_WARMUP_RECOVERED": len(recovered_ids & train_val_warmup_ids),
        "TRAIN_VAL_WARMUP_REMAINING": len(remaining_ids & train_val_warmup_ids),
        "RECOVERED_BY_TICKER": dict(
            sorted(
                Counter(
                    _ticker_for(event_after_by_id[event_id]) for event_id in recovered_ids
                ).items()
            )
        ),
        "REMAINING_BY_REASON": dict(
            sorted(Counter(str(row["reason"]) for row in remaining).items())
        ),
        "remaining_by_ticker": dict(
            sorted(
                Counter(
                    _ticker_for(event_after_by_id[event_id]) for event_id in remaining_ids
                ).items()
            )
        ),
        "per_source_recovery": _per_source_recovery(
            event_after_by_id, recovered_ids, remaining_ids
        ),
        "per_year_recovery": _per_year_recovery(event_after_by_id, recovered_ids, remaining_ids),
        "EXISTING_FEATURE_ROWS_PRESERVED": preserved["status"],
        "existing_feature_rows_preservation_hash": preserved["hash"],
        "concentration_before": concentration_before,
        "concentration_after": concentration_after,
        "TINVEST_SOURCE_ONLY": True,
        "MOEX_SUBSTITUTION_USED": False,
        "FORWARD_FILL_USED": False,
        "LEAKAGE_CHECK": "PASS",
        "target_methodology_changed": False,
        "targets_jsonl_read_as_structured_data": False,
        "rules_changed": False,
        "qwen_changed": False,
        "qwen_run": False,
        "safety": recovery_safety_flags(),
    }
    artifact_manifest["ARTIFACT_SHA"] = sha256_payload({**artifact_manifest, "ARTIFACT_SHA": None})
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "manifest.json", artifact_manifest)
    _write_jsonl(output_root / "before-root-cause.jsonl", [item.payload() for item in root_causes])
    _write_jsonl(output_root / "remaining-events.jsonl", remaining)
    _write_jsonl(output_root / "recovered-features.jsonl", recovered_features)
    _write_jsonl(output_root / "events.jsonl", events_after)
    _write_jsonl(output_root / "features.jsonl", feature_rows_after)
    _write_report(output_root / "report.md", artifact_manifest)
    return artifact_manifest


def _warmup_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in events:
        availability = _availability(row)
        if not availability.get("reaction_ready") or availability.get("feature_ready"):
            continue
        market = cast("dict[str, Any]", row.get("pre_event_market_features", {}))
        if _blocking_features(market):
            result.append(row)
    return result


def _root_cause(
    row: dict[str, Any],
    security: tuple[TInvestMinuteCandle, ...],
    cache_roots: tuple[Path, ...],
) -> WarmupEventRootCause:
    metadata = _metadata(row)
    published_at = _parse_datetime(metadata["publication_timestamp_utc"])
    earliest = earliest_required_timestamp(published_at)
    start = min((item.begin_at for item in security), default=None)
    missing_minutes = (
        max(0, int((earliest - start).total_seconds() // 60)) if start is not None else None
    )
    benchmark = _load_history(cache_roots, "IMOEX", row)
    benchmark_start = min((item.begin_at for item in benchmark), default=None)
    return WarmupEventRootCause(
        event_id=str(metadata["event_id"]),
        ticker=str(metadata["ticker"]),
        issuer=str(metadata["issuer"]),
        source=str(metadata["source_code"]),
        publication_timestamp_utc=published_at.isoformat(),
        required_lookback_minutes=60,
        earliest_required_timestamp_utc=earliest.isoformat(),
        available_security_history_start_utc=start.isoformat() if start is not None else None,
        available_benchmark_history_start_utc=(
            benchmark_start.isoformat() if benchmark_start is not None else None
        ),
        missing_history_amount_minutes=missing_minutes,
        blocking_features_before=tuple(
            _blocking_features(cast("dict[str, Any]", row.get("pre_event_market_features", {})))
        ),
    )


def _load_history(
    cache_roots: tuple[Path, ...], ticker: str, event_row: dict[str, Any]
) -> tuple[TInvestMinuteCandle, ...]:
    metadata = _metadata(event_row)
    published_at = _parse_datetime(metadata["publication_timestamp_utc"])
    rows: dict[tuple[str, datetime], TInvestMinuteCandle] = {}
    for day in acquisition_dates(published_at, max_history_days=7):
        for root in cache_roots:
            path = root / ticker / f"{day.isoformat()}-day.jsonl"
            if not path.exists():
                continue
            for payload in _read_jsonl(path):
                candle = _candle_from_payload(payload)
                rows[(candle.instrument_uid, candle.begin_at)] = candle
    return tuple(rows[key] for key in sorted(rows, key=lambda item: (item[1], item[0])))


def _cache_roots(dataset_root: Path, v1_dataset_root: Path | None) -> tuple[Path, ...]:
    roots = [dataset_root / "raw-minute-cache"]
    if v1_dataset_root is not None:
        roots.append(v1_dataset_root / "raw-minute-cache")
    else:
        candidate = dataset_root.parent / "exact-event-market-dataset-v1" / "raw-minute-cache"
        roots.append(candidate)
    existing = tuple(path for path in roots if path.exists())
    if not existing:
        raise ValueError("TINVEST_MINUTE_CACHE_MISSING")
    return existing


def _complete_pre_event_features(features: dict[str, Any]) -> bool:
    return bool(features) and not _blocking_features(features)


def _blocking_features(features: dict[str, Any]) -> list[str]:
    return [
        key
        for key, value in sorted(features.items())
        if key.startswith(("pre_return_", "imoex_pre_return_")) and value is None
    ]


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


def _remaining_payload(
    row: dict[str, Any],
    recovered_features: dict[str, Any],
    security: tuple[TInvestMinuteCandle, ...],
    benchmark: tuple[TInvestMinuteCandle, ...],
) -> dict[str, Any]:
    metadata = _metadata(row)
    missing = _blocking_features(recovered_features)
    if not security or not benchmark:
        reason = "TINVEST_HISTORY_UNAVAILABLE"
    elif missing:
        reason = "INSUFFICIENT_REQUIRED_LOOKBACK"
    else:
        reason = "UNRESOLVED_FAIL_CLOSED"
    return {
        "event_id": str(metadata["event_id"]),
        "ticker": str(metadata["ticker"]),
        "issuer": str(metadata["issuer"]),
        "source": str(metadata["source_code"]),
        "publication_timestamp_utc": str(metadata["publication_timestamp_utc"]),
        "reason": reason,
        "blocking_features_after_attempt": missing,
    }


def _counts(events: list[dict[str, Any]], features: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "events": len(events),
        "reaction_ready": sum(bool(_availability(row).get("reaction_ready")) for row in events),
        "feature_ready": len(features),
    }


def _existing_rows_preserved(
    before_by_id: dict[str, dict[str, Any]], after_features: list[dict[str, Any]]
) -> dict[str, str]:
    after_by_id = {str(row["event_id"]): row for row in after_features}
    mismatched = [
        event_id
        for event_id, before in sorted(before_by_id.items())
        if after_by_id.get(event_id) != before
    ]
    return {
        "status": "PASS" if not mismatched else "FAIL",
        "hash": sha256_payload(
            {event_id: before_by_id[event_id] for event_id in sorted(before_by_id)}
        ),
    }


def _concentration_scopes(
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
    assignments: dict[str, str],
) -> dict[str, Any]:
    event_by_id = {_event_id(row): row for row in events}
    feature_ids = {str(row["event_id"]) for row in features}
    historical_ids = {
        event_id
        for event_id, row in event_by_id.items()
        if str(_metadata(row).get("publication_date", "")) <= "2026-08-10"
    }
    scopes = {
        "ALL_HISTORICAL_THROUGH_2026_08_10": feature_ids & historical_ids,
        "TRAIN": {event_id for event_id, split in assignments.items() if split == "TRAIN"}
        & feature_ids,
        "VALIDATION": {event_id for event_id, split in assignments.items() if split == "VALIDATION"}
        & feature_ids,
        "TRAIN_VALIDATION": {
            event_id for event_id, split in assignments.items() if split in {"TRAIN", "VALIDATION"}
        }
        & feature_ids,
    }
    return {
        name: {
            "ticker": _concentration(
                Counter(_ticker_for(event_by_id[event_id]) for event_id in ids)
            ),
            "issuer": _concentration(
                Counter(_issuer_for(event_by_id[event_id]) for event_id in ids)
            ),
            "source": _concentration(
                Counter(_source_for(event_by_id[event_id]) for event_id in ids)
            ),
        }
        for name, ids in scopes.items()
    }


def _concentration(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    shares = sorted((count / total for count in counter.values()), reverse=True) if total else []
    hhi = sum(share * share for share in shares)
    return {
        "rows": total,
        "counts": dict(sorted(counter.items())),
        "top1_share": shares[0] if shares else 0.0,
        "top3_share": sum(shares[:3]),
        "hhi": hhi,
        "effective_count": 1 / hhi if hhi else 0.0,
    }


def _per_source_recovery(
    event_by_id: dict[str, dict[str, Any]], recovered_ids: set[str], remaining_ids: set[str]
) -> dict[str, dict[str, int]]:
    sources = sorted(
        {_source_for(event_by_id[event_id]) for event_id in recovered_ids | remaining_ids}
    )
    return {
        source: {
            "recovered": sum(
                _source_for(event_by_id[event_id]) == source for event_id in recovered_ids
            ),
            "remaining": sum(
                _source_for(event_by_id[event_id]) == source for event_id in remaining_ids
            ),
        }
        for source in sources
    }


def _per_year_recovery(
    event_by_id: dict[str, dict[str, Any]], recovered_ids: set[str], remaining_ids: set[str]
) -> dict[str, dict[str, int]]:
    years = sorted(
        {
            str(_metadata(event_by_id[event_id]).get("publication_date", ""))[:4]
            for event_id in recovered_ids | remaining_ids
        }
    )
    return {
        year: {
            "recovered": sum(
                str(_metadata(event_by_id[event_id]).get("publication_date", ""))[:4] == year
                for event_id in recovered_ids
            ),
            "remaining": sum(
                str(_metadata(event_by_id[event_id]).get("publication_date", ""))[:4] == year
                for event_id in remaining_ids
            ),
        }
        for year in years
    }


def _feature_schema_sha(features: list[dict[str, Any]]) -> str:
    event_feature_names = sorted(
        {name for row in features for name in cast("dict[str, Any]", row["event_features"])}
    )
    market_feature_names = sorted(
        {name for row in features for name in cast("dict[str, Any]", row["market_features"])}
    )
    return sha256_payload(
        {"event_features": event_feature_names, "market_features": market_feature_names}
    )


def _split_assignments(baseline_root: Path | None) -> dict[str, str]:
    if baseline_root is None:
        return {}
    path = baseline_root / "15m-split-manifest.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    return {
        str(item["event_id"]): str(item["split"])
        for item in cast("list[dict[str, Any]]", payload["assignments"])
    }


def _train_val_last_date(events: list[dict[str, Any]], assignments: dict[str, str]) -> str | None:
    event_by_id = {_event_id(row): row for row in events}
    dates = [
        str(_metadata(event_by_id[event_id]).get("publication_date", ""))
        for event_id, split in assignments.items()
        if split in {"TRAIN", "VALIDATION"} and event_id in event_by_id
    ]
    return max(dates) if dates else None


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["metadata"])


def _availability(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["target_availability"])


def _event_id(row: dict[str, Any]) -> str:
    return str(_metadata(row)["event_id"])


def _ticker_for(row: dict[str, Any]) -> str:
    return str(_metadata(row)["ticker"])


def _issuer_for(row: dict[str, Any]) -> str:
    return str(_metadata(row)["issuer"])


def _source_for(row: dict[str, Any]) -> str:
    return str(_metadata(row)["source_code"])


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
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(cast("dict[str, Any]", json.loads(line)))
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    safety = cast("dict[str, Any]", manifest["safety"])
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        "Data-recovery-only EXACT event market history warmup report.",
        "",
        f"- INPUT_DATASET_SHA={manifest['INPUT_DATASET_SHA']}",
        f"- OUTPUT_DATASET_SHA={manifest['OUTPUT_DATASET_SHA']}",
        f"- WARMUP_LOST_BEFORE={manifest['WARMUP_LOST_BEFORE']}",
        f"- WARMUP_RECOVERED={manifest['WARMUP_RECOVERED']}",
        f"- WARMUP_REMAINING={manifest['WARMUP_REMAINING']}",
        f"- EXISTING_FEATURE_ROWS_PRESERVED={manifest['EXISTING_FEATURE_ROWS_PRESERVED']}",
        f"- LEAKAGE_CHECK={manifest['LEAKAGE_CHECK']}",
        "",
        "## Safety",
        "",
        *[f"- {key}={str(value).lower()}" for key, value in safety.items()],
        "",
        "No model training, TEST outcome use, future holdout observation, feature tuning, "
        "NLP tuning, backtest, paper trading, orders, or BUY/SELL output was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
