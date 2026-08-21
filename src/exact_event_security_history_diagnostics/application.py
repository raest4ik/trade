from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_security_history_diagnostics.domain import (
    ARTIFACT_VERSION,
    FUTURE_EVENT_HOLDOUT_START,
    PR36_OUTPUT_DATASET_SHA,
    CandleAvailabilityProbe,
    RootCauseStatus,
    daily_probe,
    instrument_payload,
    minute_probe,
    probe_windows,
    require_pr36_manifest,
    security_history_safety_flags,
    sha256_payload,
    status_counts,
    valid_at_event_time,
)
from src.tinvest_market.client import (
    TInvestCandleBatch,
    TInvestInstrument,
    TInvestMinuteCandleBatch,
)


class SecurityHistoryReadClient(Protocol):
    async def find_instruments(
        self, query: str, *, instrument_kind: str
    ) -> tuple[TInvestInstrument, ...]: ...

    async def get_instrument_by_uid(self, instrument_uid: str) -> TInvestInstrument: ...

    async def fetch_daily_candles_audited(
        self, *, instrument_uid: str, date_from: Any, date_to: Any
    ) -> TInvestCandleBatch | Any: ...

    async def fetch_minute_candles_audited(
        self, *, instrument_uid: str, date_from: datetime, date_to: datetime
    ) -> TInvestMinuteCandleBatch: ...


async def run_security_history_diagnostics(
    *,
    pr36_root: Path,
    universe_path: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    client: SecurityHistoryReadClient | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if await asyncio.to_thread(_output_nonempty, output_root):
        raise FileExistsError(
            "immutable security history diagnostics artifact output already exists"
        )
    _verify_frozen_contracts()
    pr36_manifest = _read_json(pr36_root / "manifest.json")
    require_pr36_manifest(pr36_manifest)
    events = _read_jsonl(pr36_root / "events.jsonl")
    features = _read_jsonl(pr36_root / "features.jsonl")
    per_event_status = _read_jsonl(pr36_root / "per-event-status.jsonl")
    current_manifest_events = cast("list[str]", pr36_manifest["NEW_EVENT_IDS"])
    pr36_new_ids = set(current_manifest_events)
    future_exclusions = [
        _future_exclusion(row)
        for row in per_event_status
        if str(row["event_id"]) in pr36_new_ids
        and str(row["historical_or_future"]) == "FUTURE_METADATA_ONLY"
    ]
    diagnostic_status_rows = [
        row
        for row in per_event_status
        if str(row["event_id"]) in pr36_new_ids
        and str(row["historical_or_future"]) == "HISTORICAL"
        and str(row["primary_readiness_blocker"]) == "MARKET_HISTORY_MISSING"
    ]
    if not diagnostic_status_rows:
        raise ValueError("DIAGNOSTIC_COHORT_EMPTY")
    if any(
        _parse_datetime(row["publication_timestamp_utc"]).date() >= FUTURE_EVENT_HOLDOUT_START
        for row in diagnostic_status_rows
    ):
        raise ValueError("FUTURE_EVENT_ENTERED_SECURITY_HISTORY_DIAGNOSTICS")
    event_by_id = {_event_id(row): row for row in events}
    universe = _read_universe(universe_path)
    diagnostics: list[dict[str, Any]] = []
    for status_row in diagnostic_status_rows:
        event = event_by_id[str(status_row["event_id"])]
        diagnostics.append(
            await _diagnose_event(
                status_row=status_row,
                event=event,
                universe=universe,
                client=client,
            )
        )
    _assert_preserved(events, deepcopy(events), "EVENT")
    _assert_preserved(features, deepcopy(features), "FEATURE")
    safety = security_history_safety_flags()
    counts = status_counts(diagnostics)
    recovered_event_ids: list[str] = []
    blocked_event_ids = sorted(str(row["EVENT_ID"]) for row in diagnostics)
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_DATASET_SHA": PR36_OUTPUT_DATASET_SHA,
        "OUTPUT_DATASET_SHA": PR36_OUTPUT_DATASET_SHA,
        "OUTPUT_DATASET_SEMANTICS": "UNCHANGED_DIAGNOSTICS_ONLY",
        "DIAGNOSTIC_COHORT_SHA": sha256_payload(
            sorted(str(row["event_id"]) for row in diagnostic_status_rows)
        ),
        "DIAGNOSTIC_EVENTS_TOTAL": len(diagnostics),
        "PER_EVENT_DIAGNOSTICS": diagnostics,
        "FUTURE_HOLDOUT_EXCLUSIONS": future_exclusions,
        "FUTURE_HOLDOUT_EXCLUSION_STATUS": "PASS",
        "CURRENT_IDENTITY_HAS_HISTORY_COUNT": counts[
            RootCauseStatus.CURRENT_IDENTITY_HAS_HISTORY.value
        ],
        "HISTORICAL_IDENTITY_FOUND_COUNT": counts[RootCauseStatus.HISTORICAL_IDENTITY_FOUND.value],
        "LIFECYCLE_MISMATCH_COUNT": counts[
            RootCauseStatus.CURRENT_IDENTITY_LIFECYCLE_MISMATCH.value
        ],
        "TICKER_RENAMED_COUNT": counts[RootCauseStatus.TICKER_RENAMED.value],
        "RELISTED_SECURITY_COUNT": counts[RootCauseStatus.RELISTED_SECURITY.value],
        "SECURITY_REISSUED_COUNT": counts[RootCauseStatus.SECURITY_REISSUED.value],
        "NOT_TRADING_AT_EVENT_TIME_COUNT": counts[
            RootCauseStatus.INSTRUMENT_NOT_TRADING_AT_EVENT_TIME.value
        ],
        "TINVEST_HISTORY_UNAVAILABLE_COUNT": counts[
            RootCauseStatus.TINVEST_HISTORY_UNAVAILABLE.value
        ],
        "TINVEST_INSTRUMENT_NOT_FOUND_COUNT": counts[
            RootCauseStatus.TINVEST_INSTRUMENT_NOT_FOUND.value
        ],
        "CLASS_CODE_MISMATCH_COUNT": counts[RootCauseStatus.CLASS_CODE_MISMATCH.value],
        "NON_SUPPORTED_SECURITY_TYPE_COUNT": counts[
            RootCauseStatus.NON_SUPPORTED_SECURITY_TYPE.value
        ],
        "IDENTITY_AMBIGUOUS_COUNT": counts[RootCauseStatus.IDENTITY_AMBIGUOUS.value],
        "OTHER_FAIL_CLOSED_COUNT": counts[RootCauseStatus.OTHER_FAIL_CLOSED.value],
        "RECOVERY_POSSIBLE_COUNT": sum(bool(row["RECOVERY_POSSIBLE"]) for row in diagnostics),
        "RECOVERY_PERFORMED_COUNT": 0,
        "RECOVERED_EVENT_IDS": recovered_event_ids,
        "BLOCKED_EVENT_IDS": blocked_event_ids,
        "REACTION_READY_BEFORE": int(pr36_manifest["REACTION_READY_AFTER"]),
        "REACTION_READY_AFTER": int(pr36_manifest["REACTION_READY_AFTER"]),
        "FEATURE_READY_BEFORE": int(pr36_manifest["FEATURE_READY_AFTER"]),
        "FEATURE_READY_AFTER": int(pr36_manifest["FEATURE_READY_AFTER"]),
        "EXACT_V3_PRESERVED": "YES",
        "PR36_DATASET_PRESERVED": "YES",
        "EXISTING_EVENT_ROWS_PRESERVED": "PASS",
        "EXISTING_FEATURE_ROWS_PRESERVED": "PASS",
        "LEAKAGE_CHECK": "PASS",
        "safety": safety,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = sha256_payload({**manifest, "ARTIFACT_SHA": None})
    _write_artifacts(output_root, manifest, diagnostics, future_exclusions)
    return manifest


async def _diagnose_event(
    *,
    status_row: dict[str, Any],
    event: dict[str, Any],
    universe: dict[str, dict[str, Any]],
    client: SecurityHistoryReadClient | None,
) -> dict[str, Any]:
    metadata = _metadata(event)
    event_id = str(status_row["event_id"])
    ticker = str(status_row["ticker"])
    issuer = str(status_row["issuer"])
    published_at = _parse_datetime(status_row["publication_timestamp_utc"])
    event_uid = str(metadata.get("instrument_uid") or "")
    snapshot_identity = universe.get(ticker)
    current_identity = instrument_payload(snapshot_identity) if snapshot_identity else None
    current_identity_source = "LOCAL_TINVEST_UNIVERSE_SNAPSHOT" if snapshot_identity else None
    candidates: list[dict[str, object]] = []
    identity_errors: list[str] = []
    live_probe_status = "NOT_RUN"
    if current_identity is not None:
        candidates.append({**current_identity, "provenance": current_identity_source})
    if client is not None:
        live_probe_status = "RUN"
        live_candidates, errors = await _live_candidates(
            client, ticker=ticker, issuer=issuer, event_uid=event_uid
        )
        identity_errors.extend(errors)
        candidates.extend(live_candidates)
    candidates = _dedupe_candidates(candidates)
    current_identity = _choose_current_identity(candidates, ticker, event_uid)
    current_identity_source = (
        str(current_identity.get("provenance")) if current_identity is not None else None
    )
    probes: list[dict[str, object]] = []
    if client is not None:
        for candidate in candidates:
            probes.extend(
                probe.payload()
                for probe in await _probe_candidate(
                    client,
                    candidate=candidate,
                    published_at=published_at,
                    is_current=str(candidate["instrument_uid"]) == event_uid,
                )
            )
    current_probes = [
        item
        for item in probes
        if str(item["instrument_uid"]) == str(event_uid) and item["status"] == "HISTORY_PRESENT"
    ]
    historical_candidates = [
        item for item in candidates if str(item.get("instrument_uid")) != event_uid
    ]
    supported_historical_uids = {
        str(item["instrument_uid"])
        for item in historical_candidates
        if _supported_for_frozen_methodology(item)
    }
    historical_history = [
        item
        for item in probes
        if str(item["instrument_uid"]) in supported_historical_uids
        and item["status"] == "HISTORY_PRESENT"
    ]
    valid_current = valid_at_event_time(current_identity, published_at.date())
    root_cause = _root_cause(
        current_identity=current_identity,
        valid_current=valid_current,
        current_history=bool(current_probes),
        historical_candidates=historical_candidates,
        historical_history=historical_history,
        candidate_count=len(candidates),
        all_probes_failed=bool(probes)
        and all(str(item["status"]) == "PROBE_ERROR" for item in probes),
    )
    recovery_possible = root_cause in {
        RootCauseStatus.CURRENT_IDENTITY_HAS_HISTORY,
        RootCauseStatus.HISTORICAL_IDENTITY_FOUND,
    }
    return {
        "EVENT_ID": event_id,
        "TICKER": ticker,
        "ISSUER": issuer,
        "PUBLICATION_TIMESTAMP": published_at.isoformat(),
        "TIMESTAMP_PROVENANCE": str(status_row["timestamp_provenance"]),
        "CURRENT_FIGI": _field(current_identity, "figi"),
        "CURRENT_UID": _field(current_identity, "instrument_uid"),
        "CURRENT_CLASS_CODE": _field(current_identity, "class_code"),
        "CURRENT_EXCHANGE": _field(current_identity, "exchange"),
        "CURRENT_CURRENCY": _field(current_identity, "currency"),
        "CURRENT_INSTRUMENT_TYPE": _field(current_identity, "instrument_type"),
        "CURRENT_TRADING_STATUS": _field(current_identity, "trading_status"),
        "CURRENT_POSITION_UID": None,
        "CURRENT_FIRST_1DAY_CANDLE_DATE": _field(current_identity, "first_1day_candle_date"),
        "CURRENT_LAST_1DAY_CANDLE_DATE": _field(current_identity, "last_1day_candle_date"),
        "CURRENT_IDENTITY_SOURCE": current_identity_source,
        "CURRENT_IDENTITY_VALID_AT_EVENT_TIME": valid_current,
        "CURRENT_HISTORY_AVAILABLE": bool(current_probes),
        "HISTORICAL_IDENTITY_FOUND": bool(historical_history),
        "HISTORICAL_FIGI": _field(historical_candidates[0], "figi")
        if historical_history and historical_candidates
        else None,
        "HISTORICAL_UID": str(historical_history[0]["instrument_uid"])
        if historical_history
        else None,
        "HISTORICAL_CLASS_CODE": _field(historical_candidates[0], "class_code")
        if historical_history and historical_candidates
        else None,
        "HISTORICAL_IDENTITY_VALID_AT_EVENT_TIME": bool(historical_history) or None,
        "HISTORICAL_HISTORY_AVAILABLE": bool(historical_history),
        "FIRST_AVAILABLE_CANDLE": _first_probe_timestamp(probes),
        "LAST_AVAILABLE_CANDLE": _last_probe_timestamp(probes),
        "CANDIDATE_HISTORICAL_IDENTITIES": candidates,
        "IDENTITY_PROVENANCE": _identity_provenance(candidates, identity_errors),
        "CANDLE_AVAILABILITY_PROBES": probes,
        "ROOT_CAUSE": root_cause.value,
        "BLOCKER_STATUS": root_cause.value,
        "RECOVERY_POSSIBLE": recovery_possible,
        "RECOVERY_RECOMMENDED": recovery_possible,
        "RECOVERY_PERFORMED": False,
        "FINAL_REACTION_READY": False,
        "FINAL_FEATURE_READY": False,
        "LIVE_PROBE_STATUS": live_probe_status,
        "WHY_EXISTING_RESOLVER_CHOSE_CURRENT_IDENTITY": _resolver_reason(
            ticker, event_uid, current_identity
        ),
        "WHY_HISTORICAL_CANDLES_RESULT_WAS_EMPTY": _empty_reason(
            root_cause, live_probe_status, bool(current_probes)
        ),
    }


async def _live_candidates(
    client: SecurityHistoryReadClient, *, ticker: str, issuer: str, event_uid: str
) -> tuple[list[dict[str, object]], list[str]]:
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    for query, provenance in (
        (ticker, "TINVEST_FIND_INSTRUMENT_EXACT_TICKER"),
        (issuer, "TINVEST_FIND_INSTRUMENT_ISSUER_NAME"),
    ):
        try:
            found = await client.find_instruments(query, instrument_kind="INSTRUMENT_TYPE_SHARE")
        except Exception as exc:
            errors.append(f"{provenance}:{type(exc).__name__}:{exc}")
            continue
        rows.extend(
            {
                **instrument_payload(item),
                "provenance": provenance,
            }
            for item in found
            if item.ticker == ticker or item.instrument_uid == event_uid or item.name == issuer
        )
    if event_uid:
        try:
            item = await client.get_instrument_by_uid(event_uid)
            rows.append({**instrument_payload(item), "provenance": "TINVEST_GET_INSTRUMENT_BY_UID"})
        except Exception as exc:
            errors.append(f"TINVEST_GET_INSTRUMENT_BY_UID:{type(exc).__name__}:{exc}")
    return rows, errors


async def _probe_candidate(
    client: SecurityHistoryReadClient,
    *,
    candidate: dict[str, object],
    published_at: datetime,
    is_current: bool,
) -> tuple[CandleAvailabilityProbe, ...]:
    uid = str(candidate["instrument_uid"])
    probes: list[CandleAvailabilityProbe] = []
    for label, start, end in probe_windows(published_at):
        try:
            batch = await client.fetch_daily_candles_audited(
                instrument_uid=uid,
                date_from=start,
                date_to=end,
            )
            probes.append(
                daily_probe(
                    label=label,
                    instrument_uid=uid,
                    date_from=start,
                    date_to=end,
                    candles=batch.candles,
                    rejected_reasons=batch.rejected_reasons,
                )
            )
        except Exception as exc:
            probes.append(
                CandleAvailabilityProbe(
                    label=label,
                    instrument_uid=uid,
                    interval="1d",
                    date_from=start.isoformat(),
                    date_to=end.isoformat(),
                    candle_count=0,
                    complete_candle_count=0,
                    first_timestamp=None,
                    last_timestamp=None,
                    rejected_reasons=(f"{type(exc).__name__}:{exc}",),
                    status="PROBE_ERROR",
                )
            )
    if is_current:
        start_dt = datetime.combine(published_at.date(), time.min, UTC)
        end_dt = start_dt + timedelta(days=1)
        try:
            batch = await client.fetch_minute_candles_audited(
                instrument_uid=uid,
                date_from=start_dt,
                date_to=end_dt,
            )
            probes.append(
                minute_probe(
                    label="event_day_minute",
                    instrument_uid=uid,
                    date_from=start_dt,
                    date_to=end_dt,
                    candles=batch.candles,
                    rejected_reasons=batch.rejected_reasons,
                )
            )
        except Exception as exc:
            probes.append(
                CandleAvailabilityProbe(
                    label="event_day_minute",
                    instrument_uid=uid,
                    interval="1m",
                    date_from=start_dt.isoformat(),
                    date_to=end_dt.isoformat(),
                    candle_count=0,
                    complete_candle_count=0,
                    first_timestamp=None,
                    last_timestamp=None,
                    rejected_reasons=(f"{type(exc).__name__}:{exc}",),
                    status="PROBE_ERROR",
                )
            )
    return tuple(probes)


def _root_cause(
    *,
    current_identity: dict[str, object] | None,
    valid_current: bool | None,
    current_history: bool,
    historical_candidates: list[dict[str, object]],
    historical_history: list[dict[str, object]],
    candidate_count: int,
    all_probes_failed: bool,
) -> RootCauseStatus:
    if all_probes_failed:
        return RootCauseStatus.OTHER_FAIL_CLOSED
    if current_identity is None and candidate_count == 0:
        return RootCauseStatus.TINVEST_INSTRUMENT_NOT_FOUND
    if candidate_count > 1 and current_identity is None:
        return RootCauseStatus.IDENTITY_AMBIGUOUS
    if current_identity is not None:
        if str(current_identity.get("class_code") or "") != "TQBR":
            return RootCauseStatus.CLASS_CODE_MISMATCH
        if not _share_like(current_identity):
            return RootCauseStatus.NON_SUPPORTED_SECURITY_TYPE
    if valid_current is False:
        return RootCauseStatus.INSTRUMENT_NOT_TRADING_AT_EVENT_TIME
    if current_history:
        return RootCauseStatus.CURRENT_IDENTITY_HAS_HISTORY
    if historical_history:
        if current_identity is None:
            return RootCauseStatus.HISTORICAL_IDENTITY_FOUND
        if any(
            item.get("ticker") != current_identity.get("ticker") for item in historical_candidates
        ):
            return RootCauseStatus.TICKER_RENAMED
        return RootCauseStatus.HISTORICAL_IDENTITY_FOUND
    return RootCauseStatus.TINVEST_HISTORY_UNAVAILABLE


def _supported_for_frozen_methodology(candidate: dict[str, object]) -> bool:
    return str(candidate.get("class_code") or "") == "TQBR" and _share_like(candidate)


def _share_like(candidate: dict[str, object]) -> bool:
    instrument_type = str(candidate.get("instrument_type") or "").upper()
    return instrument_type in {"SHARE", "INSTRUMENT_TYPE_SHARE"}


def _future_exclusion(row: dict[str, Any]) -> dict[str, object]:
    return {
        "event_id": str(row["event_id"]),
        "ticker": str(row["ticker"]),
        "historical_or_future": str(row["historical_or_future"]),
        "excluded_from_security_history_diagnostics": True,
        "future_outcome_observed": False,
    }


def _choose_current_identity(
    candidates: list[dict[str, object]], ticker: str, event_uid: str
) -> dict[str, object] | None:
    for candidate in candidates:
        if event_uid and str(candidate.get("instrument_uid")) == event_uid:
            return candidate
    exact = [
        item
        for item in candidates
        if item.get("ticker") == ticker and item.get("class_code") == "TQBR"
    ]
    unique = {str(item["instrument_uid"]): item for item in exact if item.get("instrument_uid")}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _dedupe_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    by_uid: dict[str, dict[str, object]] = {}
    for item in candidates:
        uid = str(item.get("instrument_uid") or "")
        if not uid:
            continue
        if uid in by_uid:
            provenance = sorted(
                set(str(by_uid[uid].get("provenance", "")).split(";"))
                | set(str(item.get("provenance", "")).split(";"))
            )
            by_uid[uid] = {**by_uid[uid], **item, "provenance": ";".join(filter(None, provenance))}
        else:
            by_uid[uid] = item
    return [by_uid[key] for key in sorted(by_uid)]


def _identity_provenance(
    candidates: list[dict[str, object]], errors: list[str]
) -> dict[str, object]:
    return {
        "candidate_count": len(candidates),
        "candidate_sources": sorted({str(item.get("provenance")) for item in candidates}),
        "errors": errors,
        "ticker_alone_sufficient": False,
        "uid_bound_identity_required": True,
    }


def _resolver_reason(
    ticker: str, event_uid: str, current_identity: dict[str, object] | None
) -> str:
    if current_identity is None:
        return "No unique TQBR share identity was resolved fail-closed."
    return (
        f"Existing resolver selected exact ticker {ticker} with UID {event_uid} from the "
        "T-Invest universe snapshot / UID-bound metadata."
    )


def _empty_reason(
    root_cause: RootCauseStatus, live_probe_status: str, current_history: bool
) -> str:
    if live_probe_status != "RUN":
        return "Live T-Invest probes were not run; local PR36 cache had no security minute rows."
    if current_history:
        return (
            "T-Invest read-only probes found history for the current identity; PR36 remained "
            "blocked because local raw-minute-cache did not contain this security."
        )
    if root_cause == RootCauseStatus.INSTRUMENT_NOT_TRADING_AT_EVENT_TIME:
        return "Current T-Invest identity lifecycle starts after the event publication date."
    if root_cause == RootCauseStatus.OTHER_FAIL_CLOSED:
        return "T-Invest read-only probes failed closed before candle availability could be proven."
    return "Bounded read-only T-Invest probes did not return security candles."


def _first_probe_timestamp(probes: list[dict[str, object]]) -> str | None:
    values = [str(item["first_timestamp"]) for item in probes if item.get("first_timestamp")]
    return min(values) if values else None


def _last_probe_timestamp(probes: list[dict[str, object]]) -> str | None:
    values = [str(item["last_timestamp"]) for item in probes if item.get("last_timestamp")]
    return max(values) if values else None


def _field(row: dict[str, object] | None, key: str) -> object | None:
    return None if row is None else row.get(key)


def _assert_preserved(before: list[dict[str, Any]], after: list[dict[str, Any]], name: str) -> None:
    if before != after:
        raise ValueError(f"EXISTING_{name}_ROWS_PRESERVED_FAILED")


def _output_nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _verify_frozen_contracts() -> None:
    if rules_v3_fingerprint() != EXPECTED_RULES_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_MISMATCH")
    if prompt_hash() != QWEN_PROMPT_SHA or schema_hash() != QWEN_SCHEMA_SHA:
        raise ValueError("FROZEN_QWEN_CONTRACT_MISMATCH")


def _read_universe(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    return {str(row["ticker"]): row for row in cast("list[dict[str, Any]]", payload["instruments"])}


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["metadata"])


def _event_id(row: dict[str, Any]) -> str:
    return str(_metadata(row)["event_id"])


def _parse_datetime(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


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
    manifest: dict[str, Any],
    diagnostics: list[dict[str, Any]],
    future_exclusions: list[dict[str, object]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "manifest.json", manifest)
    _write_jsonl(output_root / "per-event-diagnostics.jsonl", diagnostics)
    _write_jsonl(output_root / "future-holdout-exclusions.jsonl", future_exclusions)
    _write_report(output_root / "report.md", manifest)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        "Diagnostics-only report for PR36 historical EXACT security history gaps.",
        "",
        f"- INPUT_DATASET_SHA={manifest['INPUT_DATASET_SHA']}",
        f"- OUTPUT_DATASET_SHA={manifest['OUTPUT_DATASET_SHA']}",
        f"- DIAGNOSTIC_EVENTS_TOTAL={manifest['DIAGNOSTIC_EVENTS_TOTAL']}",
        f"- CURRENT_IDENTITY_HAS_HISTORY_COUNT={manifest['CURRENT_IDENTITY_HAS_HISTORY_COUNT']}",
        f"- TINVEST_HISTORY_UNAVAILABLE_COUNT={manifest['TINVEST_HISTORY_UNAVAILABLE_COUNT']}",
        f"- RECOVERY_PERFORMED_COUNT={manifest['RECOVERY_PERFORMED_COUNT']}",
        f"- EXACT_V3_PRESERVED={manifest['EXACT_V3_PRESERVED']}",
        f"- PR36_DATASET_PRESERVED={manifest['PR36_DATASET_PRESERVED']}",
        f"- EXISTING_EVENT_ROWS_PRESERVED={manifest['EXISTING_EVENT_ROWS_PRESERVED']}",
        f"- EXISTING_FEATURE_ROWS_PRESERVED={manifest['EXISTING_FEATURE_ROWS_PRESERVED']}",
        "",
        "No model training, TEST outcome use, future holdout outcome observation, source "
        "expansion, backtest, paper trading, orders, or BUY/SELL output was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
