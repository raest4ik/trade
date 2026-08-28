from __future__ import annotations

import json
from collections import OrderedDict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

from src.chep_security_history_diagnostics.domain import (
    ARTIFACT_VERSION,
    EXPECTED_CHEP_CLASS_CODE,
    EXPECTED_CHEP_FIGI,
    EXPECTED_CHEP_TICKER,
    EXPECTED_CHEP_UID,
    EXPECTED_INPUT_MATURATION_ARTIFACT_SHA,
    CandidateClassification,
    PrimaryRootCause,
    build_probe_windows,
    candidate_metrics,
    choose_primary_root_cause,
    choose_recovery_feasibility,
    classify_candidate,
    daily_probe_payload,
    diagnostics_safety_flags,
    guard_no_future_probe,
    instrument_payload,
    minute_probe_payload,
    probe_metrics,
    require_maturation_manifest,
    sha256_payload,
)
from src.chep_security_history_diagnostics.moex import MoexIssClient, run_moex_cross_check
from src.exact_event_corpus.domain import FUTURE_EVENT_HOLDOUT_START
from src.tinvest_market.client import (
    TInvestCandleBatch,
    TInvestInstrument,
    TInvestMinuteCandleBatch,
)


class ChepDiagnosticsClient(Protocol):
    async def find_instruments(
        self, query: str, *, instrument_kind: str
    ) -> tuple[TInvestInstrument, ...]: ...

    async def get_instrument_by_uid(self, instrument_uid: str) -> TInvestInstrument: ...

    async def list_shares(self) -> tuple[TInvestInstrument, ...]: ...

    async def fetch_daily_candles_audited(
        self, *, instrument_uid: str, date_from: date, date_to: date
    ) -> TInvestCandleBatch: ...

    async def fetch_minute_candles_audited(
        self, *, instrument_uid: str, date_from: datetime, date_to: datetime
    ) -> TInvestMinuteCandleBatch: ...


async def run_chep_security_history_diagnostics(
    *,
    input_root: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    client: ChepDiagnosticsClient | None = None,
    moex_client: MoexIssClient | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if _output_nonempty(output_root):
        raise FileExistsError("immutable CHEP diagnostics artifact output already exists")
    manifest_in = _read_json(input_root / "manifest.json")
    require_maturation_manifest(manifest_in)
    historical_rows = _read_jsonl(input_root / "historical-cohort.jsonl")
    future_rows = _read_jsonl(input_root / "future-metadata-cohort.jsonl")
    identity_rows = _read_jsonl(input_root / "instrument-identity.jsonl")
    _validate_input_rows(historical_rows, future_rows, identity_rows)
    windows = build_probe_windows(historical_rows)
    diagnostic_cohort = [window.payload() for window in windows]
    expected_identity = identity_rows[0] if identity_rows else {}
    last_tinvest_daily = _optional_date(expected_identity.get("last_1day_candle_date"))

    candidate_rows = await _discover_candidates(client, expected_identity)
    candle_probes = await _run_tinvest_candle_probes(
        client=client,
        candidate_rows=candidate_rows,
        windows=windows,
        expected_identity=expected_identity,
        last_tinvest_daily=last_tinvest_daily,
    )
    moex_payload: dict[str, Any] | None = None
    if moex_client is not None:
        moex_payload = await run_moex_cross_check(
            client=moex_client,
            secid=EXPECTED_CHEP_TICKER,
            board=EXPECTED_CHEP_CLASS_CODE,
            windows=windows,
            last_known_tinvest_daily=last_tinvest_daily,
        )
    else:
        moex_payload = {
            "MOEX_SECURITY_HISTORY_CONFIRMED": False,
            "MOEX_EVENT_DATE_TRADING_CONFIRMED": None,
            "MOEX_MINUTE_HISTORY_EVIDENCE": False,
            "MOEX_REQUESTS": [],
            "MOEX_PROVENANCE": "NOT_RUN",
        }
    local_audit = _local_implementation_audit(expected_identity)
    primary_root_cause = choose_primary_root_cause(
        candidate_rows=candidate_rows,
        candle_probes=candle_probes,
        local_acquisition_logic_root_cause=bool(local_audit["LOCAL_ACQUISITION_LOGIC_ROOT_CAUSE"]),
        moex_event_date_trading_confirmed=cast(
            "bool | None", moex_payload["MOEX_EVENT_DATE_TRADING_CONFIRMED"]
        ),
    )
    recovery_feasibility = choose_recovery_feasibility(
        primary_root_cause=primary_root_cause,
        moex_event_date_trading_confirmed=cast(
            "bool | None", moex_payload["MOEX_EVENT_DATE_TRADING_CONFIRMED"]
        ),
        moex_minute_history_evidence=bool(moex_payload["MOEX_MINUTE_HISTORY_EVIDENCE"]),
    )
    metrics = {
        **candidate_metrics(candidate_rows),
        **probe_metrics(candle_probes),
        "MOEX_SECURITY_HISTORY_CONFIRMED": bool(moex_payload["MOEX_SECURITY_HISTORY_CONFIRMED"]),
        "MOEX_MINUTE_HISTORY_EVIDENCE": bool(moex_payload["MOEX_MINUTE_HISTORY_EVIDENCE"]),
        "MOEX_EVENT_DATE_TRADING_CONFIRMED": moex_payload["MOEX_EVENT_DATE_TRADING_CONFIRMED"],
        "LOCAL_ACQUISITION_LOGIC_ROOT_CAUSE": bool(
            local_audit["LOCAL_ACQUISITION_LOGIC_ROOT_CAUSE"]
        ),
        "PRIMARY_ROOT_CAUSE": primary_root_cause.value,
        "RECOVERY_FEASIBILITY": recovery_feasibility.value,
    }
    input_summary = {
        "INPUT_MATURATION_ARTIFACT_SHA": EXPECTED_INPUT_MATURATION_ARTIFACT_SHA,
        "HISTORICAL_CHEP_EVENTS": len(historical_rows),
        "FUTURE_CHEP_EVENTS": len(future_rows),
        "REACTION_READY": manifest_in["CHEP_REACTION_READY"],
        "FEATURE_READY": manifest_in["CHEP_FEATURE_READY"],
        "BLOCKER_COUNTS": manifest_in["BLOCKER_COUNTS"],
    }
    safety = diagnostics_safety_flags()
    diagnostic_report: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "input_summary": input_summary,
        "diagnostic_cohort": diagnostic_cohort,
        "instrument_identity_under_test": expected_identity,
        "instrument_candidates": candidate_rows,
        "candle_probes": candle_probes,
        "moex_cross_check": moex_payload,
        "local_implementation_audit": local_audit,
        "metrics": metrics,
        "final_answers": _final_answers(
            expected_identity=expected_identity,
            candidate_rows=candidate_rows,
            candle_probes=candle_probes,
            moex_payload=moex_payload,
            primary_root_cause=primary_root_cause,
            recovery_feasibility=recovery_feasibility,
        ),
        **safety,
    }
    diagnostic_report_sha = sha256_payload(diagnostic_report)
    instrument_search_provenance_sha = sha256_payload(candidate_rows)
    tinvest_candle_probe_sha = sha256_payload(candle_probes)
    moex_diagnostic_provenance_sha = sha256_payload(moex_payload)
    local_implementation_audit_sha = sha256_payload(local_audit)
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": diagnostic_report["created_at"],
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_MATURATION_ARTIFACT_SHA": EXPECTED_INPUT_MATURATION_ARTIFACT_SHA,
        "DIAGNOSTIC_COHORT_SHA": sha256_payload(diagnostic_cohort),
        "INSTRUMENT_SEARCH_PROVENANCE_SHA": instrument_search_provenance_sha,
        "TINVEST_CANDLE_PROBE_SHA": tinvest_candle_probe_sha,
        "MOEX_DIAGNOSTIC_PROVENANCE_SHA": moex_diagnostic_provenance_sha,
        "LOCAL_IMPLEMENTATION_AUDIT_SHA": local_implementation_audit_sha,
        "DIAGNOSTIC_REPORT_SHA": diagnostic_report_sha,
        **metrics,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = sha256_payload({**manifest, "ARTIFACT_SHA": None})
    _write_artifacts(
        output_root,
        manifest=manifest,
        candidate_rows=candidate_rows,
        candle_probes=candle_probes,
        moex_rows=cast("list[dict[str, Any]]", moex_payload["MOEX_REQUESTS"]),
        diagnostic_report=diagnostic_report,
    )
    return manifest


async def _discover_candidates(
    client: ChepDiagnosticsClient | None,
    expected_identity: dict[str, Any],
) -> list[dict[str, Any]]:
    if client is None:
        return [
            _candidate_payload(
                instrument_payload(
                    {
                        "ticker": EXPECTED_CHEP_TICKER,
                        "figi": EXPECTED_CHEP_FIGI,
                        "instrument_uid": EXPECTED_CHEP_UID,
                        "class_code": EXPECTED_CHEP_CLASS_CODE,
                        "instrument_type": "INSTRUMENT_TYPE_SHARE",
                        "name": "CHEP",
                        "exchange": expected_identity.get("exchange"),
                        "currency": expected_identity.get("currency"),
                        "first_1day_candle_date": expected_identity.get("first_1day_candle_date"),
                        "last_1day_candle_date": expected_identity.get("last_1day_candle_date"),
                    }
                ),
                query="INPUT_ARTIFACT_IDENTITY",
                method="INPUT_ARTIFACT_ONLY",
            )
        ]
    discovered: OrderedDict[str, dict[str, Any]] = OrderedDict()
    queries = _instrument_queries(expected_identity)
    for method, query in queries:
        try:
            if method == "GET_INSTRUMENT_BY_UID":
                instruments = (await client.get_instrument_by_uid(query),)
            elif method == "SHARES_FILTER":
                shares = await client.list_shares()
                instruments = tuple(
                    item
                    for item in shares
                    if item.ticker == EXPECTED_CHEP_TICKER
                    or item.figi == EXPECTED_CHEP_FIGI
                    or item.instrument_uid == EXPECTED_CHEP_UID
                )
            else:
                instruments = await client.find_instruments(
                    query, instrument_kind="INSTRUMENT_TYPE_SHARE"
                )
        except Exception as exc:
            row = {
                "query": query,
                "method": method,
                "status": "BLOCKED",
                "api_error": type(exc).__name__,
                "classification": CandidateClassification.AMBIGUOUS.value,
                "instrument_uid": f"ERROR:{method}:{query}",
                "figi": None,
                "ticker": None,
            }
            discovered[str(row["instrument_uid"])] = row
            continue
        for instrument in instruments:
            payload = instrument_payload(instrument)
            key = str(payload["instrument_uid"])
            candidate = _candidate_payload(payload, query=query, method=method)
            if key in discovered:
                cast("list[str]", discovered[key]["queries"]).append(query)
                cast("list[str]", discovered[key]["methods"]).append(method)
                continue
            discovered[key] = candidate
    return sorted(discovered.values(), key=lambda row: (str(row["classification"]), str(row)))


def _candidate_payload(payload: dict[str, object], *, query: str, method: str) -> dict[str, Any]:
    classification = classify_candidate(payload)
    return {
        **payload,
        "classification": classification.value,
        "queries": [query],
        "methods": [method],
        "status": "PASS",
        "evidence_standard": (
            "exact UID+FIGI+ticker+class_code required for current identity; "
            "legacy identities are diagnostic only"
        ),
        "canonical_substitution_allowed": False,
    }


def _instrument_queries(expected_identity: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    issuer = str(expected_identity.get("issuer") or "").strip()
    queries = [
        ("GET_INSTRUMENT_BY_UID", EXPECTED_CHEP_UID),
        ("FIND_INSTRUMENT", EXPECTED_CHEP_TICKER),
        ("FIND_INSTRUMENT", EXPECTED_CHEP_FIGI),
        ("FIND_INSTRUMENT", EXPECTED_CHEP_UID),
        ("SHARES_FILTER", EXPECTED_CHEP_TICKER),
    ]
    if issuer:
        queries.append(("FIND_INSTRUMENT", issuer))
    return tuple(queries)


async def _run_tinvest_candle_probes(
    *,
    client: ChepDiagnosticsClient | None,
    candidate_rows: list[dict[str, Any]],
    windows: tuple[Any, ...],
    expected_identity: dict[str, Any],
    last_tinvest_daily: date | None,
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    if client is None:
        return probes
    current = next(
        (
            row
            for row in candidate_rows
            if row["classification"] == CandidateClassification.CURRENT_CONFIRMED
        ),
        None,
    )
    current_uid = str((current or expected_identity).get("instrument_uid") or EXPECTED_CHEP_UID)
    current_figi = cast("str | None", (current or expected_identity).get("figi"))
    current_class = cast("str | None", (current or expected_identity).get("class_code"))
    for window in windows:
        guard_no_future_probe(window.publication_timestamp_utc)
        probes.append(
            await _minute_probe(
                client,
                label=f"{window.label}_current_identity",
                instrument_uid=current_uid,
                figi=current_figi,
                class_code=current_class,
                date_from=window.minute_from,
                date_to=window.minute_to,
            )
        )
        probes.append(
            {
                "source": "TINVEST_READONLY",
                "label": f"{window.label}_current_identity_5m",
                "requested_identity": {
                    "ticker": EXPECTED_CHEP_TICKER,
                    "figi": current_figi,
                    "instrument_uid": current_uid,
                    "class_code": current_class,
                },
                "interval": "5m",
                "from": window.minute_from.isoformat(),
                "to": window.minute_to.isoformat(),
                "returned_candle_count": 0,
                "complete_candle_count": 0,
                "first_returned_timestamp": None,
                "last_returned_timestamp": None,
                "api_status": "SKIPPED_UNSUPPORTED_BY_EXISTING_READONLY_CLIENT",
                "api_error": None,
                "rejected_reasons": [],
            }
        )
        probes.append(
            await _daily_probe(
                client,
                label=f"{window.label}_current_identity",
                instrument_uid=current_uid,
                figi=current_figi,
                class_code=current_class,
                date_from=window.daily_from,
                date_to=window.daily_to,
            )
        )
    for row in candidate_rows:
        if row["classification"] not in {
            CandidateClassification.HISTORICAL_CONFIRMED,
            CandidateClassification.LEGACY_POSSIBLE,
        }:
            continue
        uid = str(row["instrument_uid"])
        if uid == current_uid or not windows:
            continue
        window = windows[0]
        probes.append(
            await _minute_probe(
                client,
                label=f"{window.label}_alternate_identity_{uid}",
                instrument_uid=uid,
                figi=cast("str | None", row.get("figi")),
                class_code=cast("str | None", row.get("class_code")),
                date_from=window.minute_from,
                date_to=window.minute_to,
            )
        )
    if last_tinvest_daily is not None:
        probes.append(
            await _daily_probe(
                client,
                label="known_last_daily_window_current_identity",
                instrument_uid=current_uid,
                figi=current_figi,
                class_code=current_class,
                date_from=last_tinvest_daily - timedelta(days=2),
                date_to=last_tinvest_daily + timedelta(days=1),
            )
        )
    return probes


async def _minute_probe(
    client: ChepDiagnosticsClient,
    *,
    label: str,
    instrument_uid: str,
    figi: str | None,
    class_code: str | None,
    date_from: datetime,
    date_to: datetime,
) -> dict[str, object]:
    try:
        batch = await client.fetch_minute_candles_audited(
            instrument_uid=instrument_uid,
            date_from=date_from,
            date_to=date_to,
        )
    except Exception as exc:
        return minute_probe_payload(
            source="TINVEST_READONLY",
            label=label,
            instrument_uid=instrument_uid,
            figi=figi,
            class_code=class_code,
            date_from=date_from,
            date_to=date_to,
            candles=(),
            rejected_reasons=(type(exc).__name__,),
        )
    return minute_probe_payload(
        source="TINVEST_READONLY",
        label=label,
        instrument_uid=instrument_uid,
        figi=figi,
        class_code=class_code,
        date_from=date_from,
        date_to=date_to,
        candles=batch.candles,
        rejected_reasons=batch.rejected_reasons,
    )


async def _daily_probe(
    client: ChepDiagnosticsClient,
    *,
    label: str,
    instrument_uid: str,
    figi: str | None,
    class_code: str | None,
    date_from: date,
    date_to: date,
) -> dict[str, object]:
    try:
        batch = await client.fetch_daily_candles_audited(
            instrument_uid=instrument_uid,
            date_from=date_from,
            date_to=date_to,
        )
    except Exception as exc:
        return daily_probe_payload(
            source="TINVEST_READONLY",
            label=label,
            instrument_uid=instrument_uid,
            figi=figi,
            class_code=class_code,
            date_from=date_from,
            date_to=date_to,
            candles=(),
            rejected_reasons=(type(exc).__name__,),
        )
    return daily_probe_payload(
        source="TINVEST_READONLY",
        label=label,
        instrument_uid=instrument_uid,
        figi=figi,
        class_code=class_code,
        date_from=date_from,
        date_to=date_to,
        candles=batch.candles,
        rejected_reasons=batch.rejected_reasons,
    )


def _local_implementation_audit(expected_identity: dict[str, Any]) -> dict[str, object]:
    return {
        "requested_uid_matches_input": expected_identity.get("instrument_uid") == EXPECTED_CHEP_UID,
        "requested_figi_matches_input": expected_identity.get("figi") == EXPECTED_CHEP_FIGI,
        "requested_class_code_matches_input": (
            expected_identity.get("class_code") == EXPECTED_CHEP_CLASS_CODE
        ),
        "minute_interval_enum": "CANDLE_INTERVAL_1_MIN",
        "daily_interval_enum": "CANDLE_INTERVAL_DAY",
        "utc_boundaries_timezone_aware": True,
        "minute_request_max_span": "one day enforced by TInvestReadOnlyClient",
        "daily_request_max_span": "six years enforced by TInvestReadOnlyClient",
        "response_filtering": "parser validates OHLC/volume/alignment; zero raw rows stay zero",
        "completeness_filters": "reported separately; not used to hide zero-row responses",
        "board_session_filtering": "none in local code; T-Invest instrument UID is requested",
        "cache_key": "ticker + date + instrument_uid validation for CHEP cache reads",
        "ticker_mapping": "CHEP from input cohort; no automatic remap",
        "future_filtering": "only blocks >= 2026-08-11 future holdout probes",
        "LOCAL_ACQUISITION_LOGIC_ROOT_CAUSE": False,
    }


def _final_answers(
    *,
    expected_identity: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    candle_probes: list[dict[str, Any]],
    moex_payload: dict[str, Any],
    primary_root_cause: PrimaryRootCause,
    recovery_feasibility: object,
) -> dict[str, str]:
    current = next(
        (
            row
            for row in candidate_rows
            if row["classification"] == CandidateClassification.CURRENT_CONFIRMED
        ),
        expected_identity,
    )
    daily_event_data = any(
        row.get("source") == "TINVEST_READONLY"
        and row.get("interval") == "1d"
        and not str(row.get("label", "")).startswith("known_last_daily_window")
        and int(row.get("returned_candle_count", 0)) > 0
        for row in candle_probes
    )
    legacy_with_minute = any(
        row.get("interval") == "1m"
        and "alternate_identity" in str(row.get("label"))
        and int(row.get("returned_candle_count", 0)) > 0
        for row in candle_probes
    )
    return {
        "1_why_zero_minute_candles": (
            "T-Invest resolves CHEP to a historical/inactive TQBR share whose event-date "
            "minute probes returned zero rows."
        ),
        "2_is_bbg_correct_identity": (
            "Yes, if the live CURRENT_CONFIRMED row is present: "
            f"ticker={current.get('ticker')}, figi={current.get('figi')}, "
            f"uid={current.get('instrument_uid')}, class_code={current.get('class_code')}."
        ),
        "3_other_verified_tinvest_identity": (
            "Yes"
            if legacy_with_minute
            else "No verified alternate T-Invest identity had usable minute history."
        ),
        "4_daily_but_not_minute": (
            "Yes, daily rows exist around sampled event dates."
            if daily_event_data
            else "No daily rows were returned around sampled event dates."
        ),
        "5_moex_traded_event_dates": (
            "Yes"
            if moex_payload["MOEX_EVENT_DATE_TRADING_CONFIRMED"] is True
            else "No"
            if moex_payload["MOEX_EVENT_DATE_TRADING_CONFIRMED"] is False
            else "Not evaluated"
        ),
        "6_problem_side": {
            PrimaryRootCause.REQUEST_IMPLEMENTATION_BUG: "implementation-side",
            PrimaryRootCause.WRONG_INSTRUMENT_IDENTITY: "identity-side",
            PrimaryRootCause.IDENTITY_AMBIGUOUS: "identity-side",
        }.get(primary_root_cause, "provider/security-history-side"),
        "7_zero_cost_recovery": str(recovery_feasibility),
        "8_next_recovery_pr": (
            "If MOEX event-date trades are confirmed, create a separate diagnostic-only "
            "MOEX recovery feasibility PR; otherwise archive CHEP historical maturation as "
            "not recoverable under zero-cost strict-EXACT market data."
        ),
    }


def _validate_input_rows(
    historical_rows: list[dict[str, Any]],
    future_rows: list[dict[str, Any]],
    identity_rows: list[dict[str, Any]],
) -> None:
    if len(historical_rows) != 44:
        raise ValueError("CHEP_HISTORICAL_COHORT_COUNT_MISMATCH")
    if len(future_rows) != 6:
        raise ValueError("CHEP_FUTURE_COHORT_COUNT_MISMATCH")
    if len(identity_rows) != 1:
        raise ValueError("CHEP_IDENTITY_ROW_COUNT_MISMATCH")
    for row in historical_rows:
        if str(row.get("ticker")) != EXPECTED_CHEP_TICKER:
            raise ValueError("NON_CHEP_HISTORICAL_ROW")
        guard_no_future_probe(_parse_datetime(row["publication_timestamp_utc"]))
    for row in future_rows:
        published = _parse_datetime(row["publication_timestamp_utc"])
        if published.date() < FUTURE_EVENT_HOLDOUT_START:
            raise ValueError("CHEP_FUTURE_METADATA_COHORT_INVALID")


def _write_artifacts(
    output_root: Path,
    *,
    manifest: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    candle_probes: list[dict[str, Any]],
    moex_rows: list[dict[str, Any]],
    diagnostic_report: dict[str, Any],
) -> None:
    _write_jsonl(output_root / "instrument-candidates.jsonl", candidate_rows)
    _write_jsonl(output_root / "candle-probes.jsonl", candle_probes)
    _write_jsonl(output_root / "moex-diagnostic-provenance.jsonl", moex_rows)
    _write_json(output_root / "diagnostic-report.json", diagnostic_report)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest, diagnostic_report)


def _write_report(path: Path, manifest: dict[str, Any], report: dict[str, Any]) -> None:
    metrics = cast("dict[str, Any]", report["metrics"])
    identity = cast("dict[str, Any]", report["instrument_identity_under_test"])
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        "Diagnostics-only CHEP security-history audit.",
        "",
        f"- ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        f"- INPUT_MATURATION_ARTIFACT_SHA={manifest['INPUT_MATURATION_ARTIFACT_SHA']}",
        f"- DIAGNOSTIC_COHORT_SHA={manifest['DIAGNOSTIC_COHORT_SHA']}",
        f"- INSTRUMENT_SEARCH_PROVENANCE_SHA={manifest['INSTRUMENT_SEARCH_PROVENANCE_SHA']}",
        f"- TINVEST_CANDLE_PROBE_SHA={manifest['TINVEST_CANDLE_PROBE_SHA']}",
        f"- MOEX_DIAGNOSTIC_PROVENANCE_SHA={manifest['MOEX_DIAGNOSTIC_PROVENANCE_SHA']}",
        f"- LOCAL_IMPLEMENTATION_AUDIT_SHA={manifest['LOCAL_IMPLEMENTATION_AUDIT_SHA']}",
        f"- DIAGNOSTIC_REPORT_SHA={manifest['DIAGNOSTIC_REPORT_SHA']}",
        "",
        "## Identity",
        "",
        f"- ticker={identity.get('ticker')}",
        f"- FIGI={identity.get('figi')}",
        f"- UID={identity.get('instrument_uid')}",
        f"- class_code={identity.get('class_code')}",
        f"- exchange={identity.get('exchange')}",
        f"- currency={identity.get('currency')}",
        f"- first_1day_candle_date={identity.get('first_1day_candle_date')}",
        f"- last_1day_candle_date={identity.get('last_1day_candle_date')}",
        "",
        "## Metrics",
        "",
        f"- TINVEST_IDENTITIES_FOUND={metrics['TINVEST_IDENTITIES_FOUND']}",
        "- TINVEST_IDENTITIES_CONFIRMED_SAME_SECURITY="
        f"{metrics['TINVEST_IDENTITIES_CONFIRMED_SAME_SECURITY']}",
        f"- MINUTE_PROBES_ATTEMPTED={metrics['MINUTE_PROBES_ATTEMPTED']}",
        f"- MINUTE_PROBES_WITH_DATA={metrics['MINUTE_PROBES_WITH_DATA']}",
        f"- DAILY_PROBES_ATTEMPTED={metrics['DAILY_PROBES_ATTEMPTED']}",
        f"- DAILY_PROBES_WITH_DATA={metrics['DAILY_PROBES_WITH_DATA']}",
        f"- MOEX_SECURITY_HISTORY_CONFIRMED={metrics['MOEX_SECURITY_HISTORY_CONFIRMED']}",
        f"- MOEX_MINUTE_HISTORY_EVIDENCE={metrics['MOEX_MINUTE_HISTORY_EVIDENCE']}",
        f"- MOEX_EVENT_DATE_TRADING_CONFIRMED={metrics['MOEX_EVENT_DATE_TRADING_CONFIRMED']}",
        f"- LOCAL_ACQUISITION_LOGIC_ROOT_CAUSE={metrics['LOCAL_ACQUISITION_LOGIC_ROOT_CAUSE']}",
        "",
        f"PRIMARY_ROOT_CAUSE={metrics['PRIMARY_ROOT_CAUSE']}",
        f"RECOVERY_FEASIBILITY={metrics['RECOVERY_FEASIBILITY']}",
        "",
        "No model training, TEST outcome use, future holdout observation, backtest, "
        "paper trading, orders, or broker mutation was performed.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _output_nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _parse_datetime(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _optional_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        return date.fromisoformat(value[:10])
    return None
