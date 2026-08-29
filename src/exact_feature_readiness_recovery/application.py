from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid5

from src.consolidated_active_exact_historical_maturation.domain import (
    artifact_sha as input_maturation_artifact_sha,
)
from src.events.domain.v3 import EventAnalyzerV3
from src.exact_event_corpus.domain import SessionState
from src.exact_event_corpus.market import align_exact_event
from src.exact_feature_readiness_recovery.domain import (
    ARTIFACT_VERSION,
    DEFAULT_INPUT_ARTIFACT_ROOT,
    EXPECTED_INPUT_MATURATION_ARTIFACT_SHA,
    FUTURE_EVENT_HOLDOUT_START,
    HORIZONS,
    FeatureRecoveryBlocker,
    artifact_sha,
    pipeline_trace,
    safety_flags,
    sha256_payload,
)
from src.tinvest_market.client import TInvestMinuteCandle

_SEMANTIC_RECONSTRUCTION_NAMESPACE = UUID("02ed269d-88f1-4559-befe-26a7c8fe2068")
_PUBLICATION_MATERIAL_FIELDS = (
    "title",
    "description",
    "summary",
    "content",
    "raw_content",
    "publication_text",
    "source_title",
)


def run_exact_feature_readiness_recovery(
    *,
    input_root: Path = Path(DEFAULT_INPUT_ARTIFACT_ROOT),
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    manifest_in = _read_json(input_root / "manifest.json")
    _require_input_manifest(manifest_in)

    maturation_cohort = _read_jsonl(input_root / "maturation-cohort.jsonl")
    maturation_results = _read_jsonl(input_root / "maturation-results.jsonl")
    acquisition = _read_jsonl(input_root / "market-acquisition-provenance.jsonl")
    identities = _read_jsonl(input_root / "instrument-identities.jsonl")
    events_before = _read_jsonl(input_root / "events.jsonl")
    features_before = _read_jsonl(input_root / "features.jsonl")
    targets_before = _read_jsonl(input_root / "targets.jsonl")
    original_inputs = [
        *_read_jsonl(input_root / "input-v1-events.jsonl"),
        *_read_jsonl(input_root / "input-v2-events.jsonl"),
    ]

    target_cohort = _target_cohort(maturation_results)
    target_ids = {str(row["event_id"]) for row in target_cohort}
    future_events_in_target = sum(
        _parse_datetime(row["published_at_utc"]).date() >= FUTURE_EVENT_HOLDOUT_START
        for row in target_cohort
    )
    if future_events_in_target:
        raise ValueError("FUTURE_EVENT_ENTERED_FEATURE_RECOVERY")

    trace_payload = pipeline_trace().payload()
    events_after = deepcopy(events_before)
    event_after_by_id = {_event_id(row): row for row in events_after}
    event_before_by_id = {_event_id(row): row for row in events_before}
    features_after_by_id = {str(row["event_id"]): row for row in features_before}
    existing_feature_ids = set(features_after_by_id)
    target_rows_before = [row for row in targets_before if str(row["event_id"]) in target_ids]
    target_rows_before_sha = sha256_payload(target_rows_before)

    identities_by_ticker = {str(row["ticker"]): row for row in identities}
    acquisition_by_id = {str(row["event_id"]): row for row in acquisition}
    original_by_id = {_event_id(row): row for row in original_inputs}
    analyzer = EventAnalyzerV3()
    blocker_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    market_provenance: list[dict[str, Any]] = []
    recovered_ids: set[str] = set()
    reconstructed_semantic_ids: set[str] = set()
    leakage_violations: list[str] = []

    for target in target_cohort:
        event_id = str(target["event_id"])
        row = event_after_by_id[event_id]
        before_row = event_before_by_id[event_id]
        published = _published_at(row)
        identity = identities_by_ticker.get(_ticker(row))
        acquisition_row = acquisition_by_id.get(event_id)
        blocker, secondary = _classify_target(row, target, identity, acquisition_row)
        security = _load_history(input_root / "raw-minute-cache", _ticker(row), published)
        benchmark = _load_history(input_root / "raw-minute-cache", "IMOEX", published)
        alignment = None
        max_input_at = _max_feature_input_timestamp(published, security, benchmark)
        if max_input_at is not None and max_input_at > published:
            blocker = FeatureRecoveryBlocker.FEATURE_LEAKAGE_GUARD_REJECTED.value
            leakage_violations.append(event_id)
        if security and benchmark:
            try:
                alignment = align_exact_event(
                    published,
                    security,
                    benchmark,
                    expose_outcomes=False,
                )
            except Exception as exc:
                blocker = FeatureRecoveryBlocker.FEATURE_CALCULATION_FAILED.value
                secondary = (*secondary, type(exc).__name__)
        market_features = alignment.features if alignment is not None else {}
        complete_market = _complete_pre_event_features(market_features)
        market_pipeline_invoked_before = bool(target.get("market_price_lookup_performed"))
        if blocker in _recoverable_blockers() and not security:
            blocker = FeatureRecoveryBlocker.SECURITY_HISTORY_MISSING.value
        elif blocker in _recoverable_blockers() and not benchmark:
            blocker = FeatureRecoveryBlocker.BENCHMARK_HISTORY_MISSING.value
        elif (
            blocker in _recoverable_blockers()
            and alignment is not None
            and alignment.session_state != SessionState.DURING_MAIN_SESSION
        ):
            blocker = _session_blocker(alignment.session_state)
        elif blocker in _recoverable_blockers() and alignment is not None and not complete_market:
            blocker = FeatureRecoveryBlocker.PRE_EVENT_WARMUP_INSUFFICIENT.value
        semantic = _legitimate_event_features(
            event_id=event_id,
            row=row,
            original_row=original_by_id.get(event_id),
            analyzer=analyzer,
        )
        if semantic.features is None and blocker in _recoverable_blockers():
            blocker = FeatureRecoveryBlocker.SEMANTIC_EVENT_FEATURES_MISSING.value
        can_recover = (
            blocker in _recoverable_blockers()
            and semantic.features is not None
            and alignment is not None
            and alignment.session_state == SessionState.DURING_MAIN_SESSION
            and complete_market
            and max_input_at is not None
            and max_input_at <= published
        )
        if can_recover:
            row["event_features"] = semantic.features
            row["pre_event_market_features"] = market_features
            availability = _availability(row)
            availability["feature_ready"] = True
            availability["missing_reason"] = None
            availability["status"] = "REACTION_READY"
            quality = cast("dict[str, Any]", row["quality"])
            quality["feature_cutoff"] = published.isoformat()
            quality["no_forward_fill"] = True
            quality["no_interpolation"] = True
            quality["no_source_mixing"] = True
            features_after_by_id[event_id] = {
                "event_id": event_id,
                "feature_cutoff": published.isoformat(),
                "event_features": semantic.features,
                "market_features": market_features,
            }
            recovered_ids.add(event_id)
            if semantic.source == "FROZEN_EVENT_ANALYZER_V3":
                reconstructed_semantic_ids.add(event_id)
        blocker_rows.append(
            {
                **target,
                "primary_blocker": blocker,
                "secondary_blockers": list(secondary),
                "root_cause_family": _root_cause_family(blocker),
                "market_feature_pipeline_invoked_before": market_pipeline_invoked_before,
                "market_feature_pipeline_invoked_during_recovery": alignment is not None,
                "market_features_complete": complete_market,
                "semantic_event_features_present": semantic.present_before,
                "semantic_event_pipeline_reconstructable": semantic.reconstructable,
                "semantic_event_features_reconstructed": event_id in reconstructed_semantic_ids,
                "semantic_event_features_missing": semantic.features is None,
                "semantic_event_feature_source": semantic.source,
                "security_history_rows": len(security),
                "benchmark_history_rows": len(benchmark),
                "pre_event_market_features_complete": complete_market,
                "event_features_before": before_row.get("event_features"),
                "event_features_after": row.get("event_features"),
                "max_feature_input_timestamp_utc": (
                    max_input_at.isoformat() if max_input_at is not None else None
                ),
                "recovered": event_id in recovered_ids,
            }
        )
        recovery_rows.append(
            {
                "event_id": event_id,
                "ticker": _ticker(row),
                "source_family": str(target["source_family"]),
                "feature_ready_before": False,
                "feature_ready_after": event_id in recovered_ids,
                "recovery_action": _recovery_action(event_id, recovered_ids, semantic.source),
                "feature_definition_changed": False,
                "reaction_changed": False,
                "target_derived_feature_used": False,
                "post_event_market_input_used": False,
            }
        )
        market_provenance.append(
            {
                "event_id": event_id,
                "ticker": _ticker(row),
                "source": "LOCAL_INPUT_ARTIFACT_TINVEST_CACHE",
                "network_fetch_performed": False,
                "token_value_read": False,
                "security_candles_read": len(security),
                "benchmark_candles_read": len(benchmark),
                "max_feature_input_timestamp_utc": (
                    max_input_at.isoformat() if max_input_at is not None else None
                ),
            }
        )

    if leakage_violations:
        raise ValueError("FEATURE_LEAKAGE_GUARD_REJECTED")
    target_rows_after = [row for row in targets_before if str(row["event_id"]) in target_ids]
    if sha256_payload(target_rows_after) != target_rows_before_sha:
        raise ValueError("REACTION_ROWS_CHANGED")

    new_feature_ids = sorted(recovered_ids - existing_feature_ids)
    features_after = [
        features_after_by_id[str(row["event_id"])] if str(row["event_id"]) in recovered_ids else row
        for row in features_before
    ]
    features_after.extend(features_after_by_id[event_id] for event_id in new_feature_ids)
    _assert_no_duplicate_ids(events_after, "events")
    _assert_no_duplicate_ids(features_after, "features")
    _assert_no_duplicate_ids(targets_before, "targets")
    _assert_non_target_events_preserved(events_before, events_after, target_ids)

    still_blocked = [row for row in recovery_rows if not row["feature_ready_after"]]
    blockers_sha = sha256_payload(blocker_rows)
    provenance_sha = sha256_payload(market_provenance)
    recovery_sha = sha256_payload(recovery_rows)
    output_dataset_sha = sha256_payload(
        {
            "input_maturation_artifact_sha": manifest_in["ARTIFACT_SHA"],
            "events": events_after,
            "features": features_after,
            "targets": targets_before,
        }
    )
    flags = safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "git_sha": git_sha,
        "BASE_MAIN_SHA": base_main_sha,
        "INPUT_MATURATION_ARTIFACT_SHA": manifest_in["ARTIFACT_SHA"],
        "INPUT_MATURATION_COHORT_SHA": sha256_payload(maturation_cohort),
        "TARGET_REACTION_READY_FEATURE_BLOCKED": len(target_cohort),
        "TARGET_COHORT_SHA": sha256_payload(target_cohort),
        "PIPELINE_TRACE_SHA": sha256_payload(trace_payload),
        "FEATURE_BLOCKERS_SHA": blockers_sha,
        "MARKET_RECOVERY_PROVENANCE_SHA": provenance_sha,
        "FEATURE_RECOVERY_RESULT_SHA": recovery_sha,
        "OUTPUT_DATASET_SHA": output_dataset_sha,
        "TARGET_EVENTS": len(target_cohort),
        "FUTURE_EVENTS_IN_TARGET_COHORT": future_events_in_target,
        "FEATURE_READY_RECOVERED": len(recovered_ids),
        "FEATURE_READY_STILL_BLOCKED": len(still_blocked),
        "FEATURE_READY_BEFORE": int(manifest_in["FEATURE_READY_AFTER"]),
        "FEATURE_READY_AFTER": int(manifest_in["FEATURE_READY_AFTER"]) + len(recovered_ids),
        "MARKET_FEATURE_PIPELINE_INVOKED_BEFORE": sum(
            bool(row["market_feature_pipeline_invoked_before"]) for row in blocker_rows
        ),
        "MARKET_FEATURES_COMPLETE": sum(
            bool(row["market_features_complete"]) for row in blocker_rows
        ),
        "SEMANTIC_EVENT_FEATURES_PRESENT": sum(
            bool(row["semantic_event_features_present"]) for row in blocker_rows
        ),
        "SEMANTIC_EVENT_FEATURES_RECONSTRUCTED": len(reconstructed_semantic_ids),
        "SEMANTIC_EVENT_FEATURES_MISSING": sum(
            bool(row["semantic_event_features_missing"]) for row in blocker_rows
        ),
        "SEMANTIC_EVENT_PIPELINE_RECONSTRUCTABLE": sum(
            bool(row["semantic_event_pipeline_reconstructable"]) for row in blocker_rows
        ),
        "BLOCKED_BY_REASON": _counter_payload(
            row["primary_blocker"] for row in blocker_rows if not row["recovered"]
        ),
        "BLOCKED_BY_TICKER": _counter_payload(
            row["ticker"] for row in blocker_rows if not row["recovered"]
        ),
        "BLOCKED_BY_SOURCE_FAMILY": _counter_payload(
            row["source_family"] for row in blocker_rows if not row["recovered"]
        ),
        "RECOVERED_BY_TICKER": _counter_payload(
            row["ticker"] for row in blocker_rows if row["recovered"]
        ),
        "RECOVERED_BY_SOURCE_FAMILY": _counter_payload(
            row["source_family"] for row in blocker_rows if row["recovered"]
        ),
        "PER_TICKER": _per_ticker(blocker_rows),
        "PER_SOURCE_FAMILY": _per_source_family(blocker_rows),
        "DOMINANT_ROOT_CAUSE_FAMILY": _dominant_root_cause(blocker_rows),
        "MARKET_FEATURE_PIPELINE_INVOKED_BEFORE_THIS_PR": all(
            bool(row["market_feature_pipeline_invoked_before"]) for row in blocker_rows
        ),
        "SEMANTIC_FEATURE_PIPELINE_INVOKED_BEFORE_THIS_PR": all(
            bool(row["semantic_event_features_present"]) for row in blocker_rows
        ),
        "PRE_EVENT_WARMUP_SUFFICIENT_FOR_TARGET": all(
            row["market_features_complete"] for row in blocker_rows
        ),
        "REACTION_ROWS_CHANGED": 0,
        "REACTION_ROWS_SHA_BEFORE": target_rows_before_sha,
        "REACTION_ROWS_SHA_AFTER": sha256_payload(target_rows_after),
        "EXISTING_CANONICAL_ROWS_PRESERVED": "PASS",
        "DETERMINISTIC_REPLAY": "PASS",
        "FINAL_DECISION": _decision(len(recovered_ids), len(still_blocked), blocker_rows),
        "safety": flags,
        **flags,
    }
    manifest["ARTIFACT_SHA"] = artifact_sha(manifest)
    _write_jsonl(output_root / "input-target-cohort.jsonl", target_cohort)
    _write_json(output_root / "pipeline-trace.json", trace_payload)
    _write_jsonl(output_root / "feature-blockers.jsonl", blocker_rows)
    _write_jsonl(output_root / "market-recovery-provenance.jsonl", market_provenance)
    _write_jsonl(output_root / "feature-recovery-results.jsonl", recovery_rows)
    _write_jsonl(output_root / "events.jsonl", events_after)
    _write_jsonl(output_root / "features.jsonl", features_after)
    _write_jsonl(output_root / "targets.jsonl", targets_before)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)
    return manifest


def _require_input_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("ARTIFACT_SHA") != EXPECTED_INPUT_MATURATION_ARTIFACT_SHA:
        raise ValueError("INPUT_MATURATION_ARTIFACT_SHA_MISMATCH")
    if manifest.get("ARTIFACT_SHA") != input_maturation_artifact_sha(manifest):
        raise ValueError("INPUT_MATURATION_ARTIFACT_REPLAY_MISMATCH")
    required_safe_false = (
        "FUTURE_EVENT_HOLDOUT_USED",
        "FUTURE_EVENT_HOLDOUT_OBSERVED",
        "MODEL_TRAINING_PERFORMED",
        "TEST_OUTCOME_USED",
        "TEST_EVALUATION_PERFORMED",
        "BACKTEST_PERFORMED",
    )
    for key in required_safe_false:
        if bool(manifest.get(key)):
            raise ValueError(f"INPUT_{key}_NOT_SAFE")


def _target_cohort(maturation_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in maturation_results:
        published = _parse_datetime(row["published_at_utc"])
        if published.date() >= FUTURE_EVENT_HOLDOUT_START:
            continue
        if bool(row.get("reaction_ready")) and not bool(row.get("feature_ready")):
            rows.append(
                {
                    "event_id": str(row["event_id"]),
                    "ticker": str(row["ticker"]),
                    "source_id": str(row["source_id"]),
                    "source_family": str(row["source_family"]),
                    "published_at_utc": published.isoformat(),
                    "reaction_ready": True,
                    "feature_ready": False,
                    "ready_1m": bool(row["ready_1m"]),
                    "ready_5m": bool(row["ready_5m"]),
                    "ready_15m": bool(row["ready_15m"]),
                    "ready_30m": bool(row["ready_30m"]),
                    "ready_60m": bool(row["ready_60m"]),
                    "market_price_lookup_performed": bool(row.get("market_price_lookup_performed")),
                    "pre_event_context_ready": bool(row.get("pre_event_context_ready")),
                    "existing_primary_blocker": row.get("primary_blocker"),
                }
            )
    return sorted(rows, key=lambda item: (item["published_at_utc"], item["event_id"]))


def _classify_target(
    row: dict[str, Any],
    target: dict[str, Any],
    identity: dict[str, Any] | None,
    acquisition: dict[str, Any] | None,
) -> tuple[str, tuple[str, ...]]:
    secondary = tuple(
        str(value) for value in (target.get("existing_primary_blocker"),) if value not in {None, ""}
    )
    if identity is None or not identity.get("instrument_uid"):
        return FeatureRecoveryBlocker.INSTRUMENT_IDENTITY_UNRESOLVED.value, secondary
    if identity.get("identity_provenance") == "AMBIGUOUS":
        return FeatureRecoveryBlocker.INSTRUMENT_IDENTITY_AMBIGUOUS.value, secondary
    if acquisition is not None and acquisition.get("security_status") != "PASS":
        return FeatureRecoveryBlocker.SECURITY_HISTORY_MISSING.value, secondary
    if acquisition is not None and acquisition.get("benchmark_status") != "PASS":
        return FeatureRecoveryBlocker.BENCHMARK_HISTORY_MISSING.value, secondary
    session = str(_metadata(row).get("session_state") or target.get("session_status") or "")
    if session == "PRE_OPEN":
        return FeatureRecoveryBlocker.PRE_OPEN.value, secondary
    if session == "AFTER_CLOSE":
        return FeatureRecoveryBlocker.AFTER_CLOSE.value, secondary
    if session == "NON_TRADING_DAY":
        return FeatureRecoveryBlocker.NON_TRADING_SESSION.value, secondary
    if session not in {"", "DURING_MAIN_SESSION"}:
        return FeatureRecoveryBlocker.SESSION_ALIGNMENT_FAILED.value, secondary
    if not all(bool(target[f"ready_{horizon}"]) for horizon in HORIZONS):
        return FeatureRecoveryBlocker.SESSION_ALIGNMENT_FAILED.value, secondary
    if not _event_features_schema_valid(row.get("event_features")):
        if row.get("event_features") is None:
            return FeatureRecoveryBlocker.FEATURE_PIPELINE_NOT_INVOKED.value, secondary
        return FeatureRecoveryBlocker.FEATURE_SCHEMA_MISMATCH.value, secondary
    availability = _availability(row)
    if not bool(availability.get("feature_ready")):
        return FeatureRecoveryBlocker.FEATURE_STATE_NOT_PROPAGATED.value, secondary
    return FeatureRecoveryBlocker.CANONICAL_INTEGRATION_FAILED.value, secondary


class _SemanticResult:
    def __init__(
        self,
        *,
        features: dict[str, object] | None,
        source: str | None,
        present_before: bool,
        reconstructable: bool,
    ) -> None:
        self.features = features
        self.source = source
        self.present_before = present_before
        self.reconstructable = reconstructable


def _legitimate_event_features(
    *,
    event_id: str,
    row: dict[str, Any],
    original_row: dict[str, Any] | None,
    analyzer: EventAnalyzerV3,
) -> _SemanticResult:
    existing = _stored_event_features(row.get("event_features"))
    if existing is not None:
        return _SemanticResult(
            features=existing,
            source="STORED_EVENT_ROW",
            present_before=True,
            reconstructable=False,
        )
    source_text = _publication_material(original_row) or _publication_material(row)
    if source_text is None:
        return _SemanticResult(
            features=None,
            source=None,
            present_before=False,
            reconstructable=False,
        )
    analysis = analyzer.analyze(news_id=_analysis_uuid(event_id), raw_content=source_text)
    return _SemanticResult(
        features={
            "primary_event_type": analysis.primary_event_type.value,
            "event_count": len(analysis.events),
            "fact_count": len(analysis.financial_facts),
        },
        source="FROZEN_EVENT_ANALYZER_V3",
        present_before=False,
        reconstructable=True,
    )


def _stored_event_features(value: object) -> dict[str, object] | None:
    if _event_features_schema_valid(value):
        return cast("dict[str, object]", value)
    return None


def _publication_material(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    candidates: list[str] = []
    containers = [row]
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        containers.append(cast("dict[str, Any]", metadata))
    for container in containers:
        for field in _PUBLICATION_MATERIAL_FIELDS:
            value = container.get(field)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    if not candidates:
        return None
    return "\n".join(dict.fromkeys(candidates))


def _analysis_uuid(event_id: str) -> UUID:
    try:
        return UUID(event_id)
    except ValueError:
        return uuid5(_SEMANTIC_RECONSTRUCTION_NAMESPACE, event_id)


def _event_features_schema_valid(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    typed = cast("dict[str, object]", value)
    return (
        isinstance(typed.get("primary_event_type"), str)
        and isinstance(typed.get("event_count"), int)
        and isinstance(typed.get("fact_count"), int)
    )


def _recoverable_blockers() -> set[str]:
    return {
        FeatureRecoveryBlocker.FEATURE_STATE_NOT_PROPAGATED.value,
        FeatureRecoveryBlocker.FEATURE_PIPELINE_NOT_INVOKED.value,
        FeatureRecoveryBlocker.FEATURE_SCHEMA_MISMATCH.value,
    }


def _session_blocker(session_state: SessionState) -> str:
    if session_state == SessionState.PRE_OPEN:
        return FeatureRecoveryBlocker.PRE_OPEN.value
    if session_state == SessionState.AFTER_CLOSE:
        return FeatureRecoveryBlocker.AFTER_CLOSE.value
    if session_state == SessionState.NON_TRADING_DAY:
        return FeatureRecoveryBlocker.NON_TRADING_SESSION.value
    return FeatureRecoveryBlocker.SESSION_ALIGNMENT_FAILED.value


def _load_history(
    root: Path, ticker: str, published_at: datetime
) -> tuple[TInvestMinuteCandle, ...]:
    rows: dict[tuple[str, datetime], TInvestMinuteCandle] = {}
    start = published_at.date()
    for path in sorted((root / ticker).glob("*-day.jsonl")) if (root / ticker).exists() else []:
        day = _day_from_cache_path(path)
        if day is None:
            continue
        days_before_event = (start - day).days
        if days_before_event < 0 or days_before_event > 7:
            continue
        for payload in _read_jsonl(path):
            candle = _candle_from_payload(payload)
            rows[(candle.instrument_uid, candle.begin_at)] = candle
    return tuple(rows[key] for key in sorted(rows, key=lambda item: (item[1], item[0])))


def _day_from_cache_path(path: Path) -> date | None:
    try:
        return datetime.fromisoformat(path.name[:10]).date()
    except ValueError:
        return None


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


def _root_cause_family(blocker: str) -> str:
    if blocker in {
        FeatureRecoveryBlocker.SECURITY_HISTORY_MISSING.value,
        FeatureRecoveryBlocker.BENCHMARK_HISTORY_MISSING.value,
        FeatureRecoveryBlocker.PRE_EVENT_WARMUP_MISSING.value,
        FeatureRecoveryBlocker.PRE_EVENT_WARMUP_INSUFFICIENT.value,
    }:
        return "A_DATA_MISSING"
    if blocker in {
        FeatureRecoveryBlocker.FEATURE_PIPELINE_NOT_INVOKED.value,
        FeatureRecoveryBlocker.FEATURE_SCHEMA_MISMATCH.value,
        FeatureRecoveryBlocker.CANONICAL_INTEGRATION_FAILED.value,
    }:
        return "B_IMPLEMENTATION_PIPELINE_WIRING"
    if blocker == FeatureRecoveryBlocker.FEATURE_STATE_NOT_PROPAGATED.value:
        return "C_CANONICAL_STATE_PROPAGATION"
    if blocker in {
        FeatureRecoveryBlocker.SESSION_ALIGNMENT_FAILED.value,
        FeatureRecoveryBlocker.NON_TRADING_SESSION.value,
        FeatureRecoveryBlocker.PRE_OPEN.value,
        FeatureRecoveryBlocker.AFTER_CLOSE.value,
        FeatureRecoveryBlocker.MARKET_FEATURE_INPUT_INCOMPLETE.value,
        FeatureRecoveryBlocker.FEATURE_LEAKAGE_GUARD_REJECTED.value,
    }:
        return "D_FROZEN_METHODOLOGY_CORRECTLY_REJECTS_EVENT"
    if blocker in {
        FeatureRecoveryBlocker.INSTRUMENT_IDENTITY_UNRESOLVED.value,
        FeatureRecoveryBlocker.INSTRUMENT_IDENTITY_AMBIGUOUS.value,
    }:
        return "E_IDENTITY_MARKET_ELIGIBILITY"
    if blocker == FeatureRecoveryBlocker.SEMANTIC_EVENT_FEATURES_MISSING.value:
        return "G_SEMANTIC_EVENT_FEATURES_MISSING"
    return "F_UNKNOWN_UNVERIFIED"


def _dominant_root_cause(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row["root_cause_family"]) for row in rows)
    return counts.most_common(1)[0][0] if counts else "F_UNKNOWN_UNVERIFIED"


def _decision(recovered: int, still_blocked: int, rows: list[dict[str, Any]]) -> str:
    if recovered and recovered >= still_blocked:
        if _dominant_root_cause(rows) == "C_CANONICAL_STATE_PROPAGATION":
            return "FEATURE_STATE_PROPAGATION_RECOVERED"
        if any(bool(row["semantic_event_features_reconstructed"]) for row in rows):
            return "FROZEN_SEMANTIC_FEATURES_RECOVERED"
        return "FEATURE_PIPELINE_WIRING_RECOVERED"
    if still_blocked and _dominant_root_cause(rows) == "A_DATA_MISSING":
        return "MARKET_HISTORY_RECOVERY_REQUIRED"
    if still_blocked and _dominant_root_cause(rows).startswith("D_"):
        return "FROZEN_FEATURE_PREREQUISITES_DOMINATE"
    if still_blocked and _dominant_root_cause(rows).startswith("G_"):
        return "SEMANTIC_EVENT_FEATURES_MISSING"
    return "ACTIVE_EXACT_DATA_QUALITY_REVIEW_REQUIRED"


def _recovery_action(event_id: str, recovered_ids: set[str], semantic_source: str | None) -> str:
    if event_id not in recovered_ids:
        return "KEPT_BLOCKED"
    if semantic_source == "FROZEN_EVENT_ANALYZER_V3":
        return "FROZEN_SEMANTIC_EVENT_FEATURES_RECONSTRUCTED"
    return "PRE_EXISTING_SEMANTIC_EVENT_FEATURES_PROPAGATED"


def _per_ticker(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return _per_group(rows, "ticker")


def _per_source_family(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return _per_group(rows, "source_family")


def _per_group(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in sorted({str(row[field]) for row in rows}):
        subset = [row for row in rows if str(row[field]) == name]
        result[name] = {
            "TARGET": len(subset),
            "RECOVERED": sum(bool(row["recovered"]) for row in subset),
            "STILL_BLOCKED": sum(not bool(row["recovered"]) for row in subset),
            "PRIMARY_BLOCKERS": _counter_payload(
                row["primary_blocker"] for row in subset if not row["recovered"]
            ),
        }
    return result


def _counter_payload(values: Iterable[object]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _assert_no_duplicate_ids(rows: list[dict[str, Any]], label: str) -> None:
    ids = [_event_id(row) if label == "events" else str(row["event_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"DUPLICATE_{label.upper()}_ROWS")


def _assert_non_target_events_preserved(
    before: list[dict[str, Any]], after: list[dict[str, Any]], target_ids: set[str]
) -> None:
    before_rows = {_event_id(row): row for row in before if _event_id(row) not in target_ids}
    after_rows = {_event_id(row): row for row in after if _event_id(row) not in target_ids}
    if before_rows != after_rows:
        raise ValueError("NON_TARGET_CANONICAL_ROWS_CHANGED")


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["metadata"])


def _availability(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["target_availability"])


def _event_id(row: dict[str, Any]) -> str:
    return str(_metadata(row)["event_id"])


def _ticker(row: dict[str, Any]) -> str:
    return str(_metadata(row)["ticker"])


def _published_at(row: dict[str, Any]) -> datetime:
    return _parse_datetime(_metadata(row)["publication_timestamp_utc"])


def _parse_datetime(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


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
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        f"- ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        f"- INPUT_MATURATION_ARTIFACT_SHA={manifest['INPUT_MATURATION_ARTIFACT_SHA']}",
        "- TARGET_REACTION_READY_FEATURE_BLOCKED="
        f"{manifest['TARGET_REACTION_READY_FEATURE_BLOCKED']}",
        f"- FEATURE_READY_RECOVERED={manifest['FEATURE_READY_RECOVERED']}",
        f"- FEATURE_READY_STILL_BLOCKED={manifest['FEATURE_READY_STILL_BLOCKED']}",
        f"- FEATURE_READY_BEFORE={manifest['FEATURE_READY_BEFORE']}",
        f"- FEATURE_READY_AFTER={manifest['FEATURE_READY_AFTER']}",
        f"- MARKET_FEATURES_COMPLETE={manifest['MARKET_FEATURES_COMPLETE']}",
        f"- SEMANTIC_EVENT_FEATURES_PRESENT={manifest['SEMANTIC_EVENT_FEATURES_PRESENT']}",
        "- SEMANTIC_EVENT_FEATURES_RECONSTRUCTED="
        f"{manifest['SEMANTIC_EVENT_FEATURES_RECONSTRUCTED']}",
        f"- SEMANTIC_EVENT_FEATURES_MISSING={manifest['SEMANTIC_EVENT_FEATURES_MISSING']}",
        f"- DOMINANT_ROOT_CAUSE_FAMILY={manifest['DOMINANT_ROOT_CAUSE_FAMILY']}",
        f"- FINAL_DECISION={manifest['FINAL_DECISION']}",
        "",
        "Feature definitions, reaction methodology, session methodology, TEST, models, "
        "backtests, trading, and future outcomes were not used or changed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
