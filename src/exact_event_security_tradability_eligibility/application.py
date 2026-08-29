from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

from src.exact_event_corpus.domain import FUTURE_EVENT_HOLDOUT_START
from src.exact_event_security_tradability_eligibility.domain import (
    ARTIFACT_VERSION,
    EXPECTED_CHEP_FIGI,
    EXPECTED_CHEP_HISTORICAL_EVENTS,
    EXPECTED_CHEP_TICKER,
    EXPECTED_CHEP_UID,
    EXPECTED_FUTURE_CHEP_EVENTS,
    EXPECTED_INPUT_DIAGNOSTIC_ARTIFACT_SHA,
    EXPECTED_INPUT_MATURATION_ARTIFACT_SHA,
    EligibilityPolicy,
    EventValidity,
    FinalDecision,
    InstrumentIdentityStatus,
    MarketReactionEligibility,
    TradingEvidence,
    collection_decision_for_ticker,
    eligibility_safety_flags,
    evaluate_event_eligibility,
    parse_datetime,
    require_diagnostic_manifest,
    result_counts,
    sha256_payload,
)


def run_security_tradability_eligibility(
    *,
    diagnostic_root: Path,
    maturation_root: Path,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable tradability eligibility artifact output already exists")
    diagnostic_manifest = _read_json(diagnostic_root / "manifest.json")
    require_diagnostic_manifest(diagnostic_manifest)
    diagnostic_report = _read_json(diagnostic_root / "diagnostic-report.json")
    maturation_manifest = _read_json(maturation_root / "manifest.json")
    if maturation_manifest.get("ARTIFACT_SHA") != EXPECTED_INPUT_MATURATION_ARTIFACT_SHA:
        raise ValueError("INPUT_MATURATION_ARTIFACT_SHA_MISMATCH")
    historical_rows = _read_jsonl(maturation_root / "historical-cohort.jsonl")
    future_rows = _read_jsonl(maturation_root / "future-metadata-cohort.jsonl")
    canonical_events = _read_jsonl(maturation_root / "events.jsonl")
    canonical_features = _read_jsonl(maturation_root / "features.jsonl")
    if len(historical_rows) != EXPECTED_CHEP_HISTORICAL_EVENTS:
        raise ValueError("CHEP_HISTORICAL_COHORT_COUNT_MISMATCH")
    if len(future_rows) != EXPECTED_FUTURE_CHEP_EVENTS:
        raise ValueError("CHEP_FUTURE_COHORT_COUNT_MISMATCH")
    event_hash_before = sha256_payload(canonical_events)
    evidence = _trading_evidence_from_diagnostic(diagnostic_report)
    evidence_rows = [evidence.payload()]
    results = [
        evaluate_event_eligibility(
            event_id=str(row["event_id"]),
            ticker=str(row["ticker"]),
            published_at_utc=parse_datetime(row["publication_timestamp_utc"]),
            identity_status=InstrumentIdentityStatus.RESOLVED,
            evidence=evidence,
            event_validity=EventValidity.VALID_EXACT_EVENT,
        )
        for row in sorted(
            historical_rows, key=lambda item: (item["publication_timestamp_utc"], item["event_id"])
        )
    ]
    future_results = [
        evaluate_event_eligibility(
            event_id=str(row["event_id"]),
            ticker=str(row["ticker"]),
            published_at_utc=parse_datetime(row["publication_timestamp_utc"]),
            identity_status=InstrumentIdentityStatus.NOT_EVALUATED_FUTURE_HOLDOUT,
            evidence=None,
            event_validity=EventValidity.VALID_EXACT_EVENT,
        )
        for row in sorted(
            future_rows, key=lambda item: (item["publication_timestamp_utc"], item["event_id"])
        )
    ]
    all_results = [*results, *future_results]
    result_payloads = [result.payload() for result in all_results]
    policy = EligibilityPolicy()
    policy_payload = policy.payload()
    collection_decision = collection_decision_for_ticker(all_results)
    canonical_exact_total = len(canonical_events)
    future_canonical_count = sum(
        parse_datetime(cast("dict[str, Any]", row["metadata"])["publication_timestamp_utc"]).date()
        >= FUTURE_EVENT_HOLDOUT_START
        for row in canonical_events
    )
    ineligible_count = sum(
        result.market_reaction_eligibility
        == MarketReactionEligibility.SECURITY_NOT_TRADING_AT_EVENT_TIME
        for result in all_results
    )
    dataset_accounting = {
        "CANONICAL_EXACT_EVENTS_TOTAL": canonical_exact_total,
        "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS": canonical_exact_total
        - ineligible_count
        - future_canonical_count,
        "MARKET_REACTION_INELIGIBLE_EXACT_EVENTS": ineligible_count,
        "FUTURE_METADATA_ONLY_EXACT_EVENTS": future_canonical_count,
        "REACTION_READY_EVENTS": sum(
            bool(cast("dict[str, Any]", row["target_availability"]).get("reaction_ready"))
            for row in canonical_events
        ),
        "FEATURE_READY_EVENTS": len(canonical_features),
    }
    counts = result_counts(all_results)
    chep_counts = result_counts(results)
    final_decision = FinalDecision.SOURCE_BREADTH_EXPANSION_NEXT
    safety = eligibility_safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_DIAGNOSTIC_ARTIFACT_SHA": EXPECTED_INPUT_DIAGNOSTIC_ARTIFACT_SHA,
        "INPUT_MATURATION_ARTIFACT_SHA": EXPECTED_INPUT_MATURATION_ARTIFACT_SHA,
        "ELIGIBILITY_POLICY_SHA": sha256_payload(policy_payload),
        "CHEP_COHORT_SHA": sha256_payload(
            {
                "historical": historical_rows,
                "future_metadata_only": future_rows,
            }
        ),
        "TRADING_EVIDENCE_PROVENANCE_SHA": sha256_payload(evidence_rows),
        "ELIGIBILITY_RESULT_SHA": sha256_payload(result_payloads),
        "CANONICAL_EVENT_ROWS_PRESERVED_SHA": event_hash_before,
        **dataset_accounting,
        **counts,
        "CHEP_HISTORICAL_EXACT_EVENTS": len(results),
        "CHEP_EVENT_VALID": sum(
            result.event_validity == EventValidity.VALID_EXACT_EVENT for result in results
        ),
        "CHEP_CANONICAL_EVENTS_PRESERVED": len(results),
        "CHEP_MARKET_REACTION_ELIGIBLE": chep_counts["EVENTS_MARKET_ELIGIBLE"],
        "CHEP_SECURITY_NOT_TRADING": chep_counts["SECURITY_NOT_TRADING_COUNT"],
        "CHEP_REACTION_ATTEMPTS_SKIPPED": chep_counts["REACTION_ATTEMPTS_AVOIDED"],
        "CHEP_FEATURE_ATTEMPTS_SKIPPED": sum(result.feature_attempt_skipped for result in results),
        "FUTURE_CHEP_EVENTS": len(future_results),
        "FUTURE_CHEP_PRICE_LOOKUPS": 0,
        "FUTURE_CHEP_REACTION_ATTEMPTS": 0,
        "FUTURE_CHEP_TARGET_ATTEMPTS": 0,
        "CHEP_COLLECTION_DECISION": collection_decision.value,
        "FINAL_DECISION": final_decision.value,
        "EMPTY_CANDLES_ALONE_PROVE_NON_TRADING": False,
        **safety,
    }
    manifest["ARTIFACT_SHA"] = sha256_payload({**manifest, "ARTIFACT_SHA": None})
    _write_artifacts(
        output_root,
        policy=policy_payload,
        result_payloads=result_payloads,
        evidence_rows=evidence_rows,
        manifest=manifest,
    )
    return manifest


def _trading_evidence_from_diagnostic(report: dict[str, Any]) -> TradingEvidence:
    metrics = cast("dict[str, Any]", report["metrics"])
    if metrics.get("PRIMARY_ROOT_CAUSE") != "HISTORICAL_SECURITY_NOT_SUPPORTED":
        raise ValueError("DIAGNOSTIC_ROOT_CAUSE_NOT_ELIGIBILITY_EVIDENCE")
    if metrics.get("RECOVERY_FEASIBILITY") != "NOT_RECOVERABLE_WITH_ZERO_COST_SOURCES":
        raise ValueError("DIAGNOSTIC_RECOVERY_FEASIBILITY_NOT_ELIGIBILITY_EVIDENCE")
    if metrics.get("MOEX_SECURITY_HISTORY_CONFIRMED") is not True:
        raise ValueError("MOEX_SECURITY_HISTORY_NOT_CONFIRMED")
    if metrics.get("MOEX_EVENT_DATE_TRADING_CONFIRMED") is not False:
        raise ValueError("MOEX_EVENT_DATE_NON_TRADING_NOT_CONFIRMED")
    current = _current_candidate(report)
    last_trade = _last_confirmed_trade_date(report)
    return TradingEvidence(
        ticker=EXPECTED_CHEP_TICKER,
        instrument_uid=str(current.get("instrument_uid") or EXPECTED_CHEP_UID),
        figi=str(current.get("figi") or EXPECTED_CHEP_FIGI),
        class_code=str(current.get("class_code") or "TQBR"),
        source="CHEP_SECURITY_HISTORY_DIAGNOSTICS_V1",
        security_history_confirmed=True,
        event_date_trading_confirmed=False,
        last_confirmed_trading_date=last_trade,
        current_trading_status=cast("str | None", current.get("trading_status")),
        api_trade_available=cast("bool | None", current.get("api_trade_available_flag")),
        buy_available=cast("bool | None", current.get("buy_available_flag")),
        sell_available=cast("bool | None", current.get("sell_available_flag")),
        evidence_detail=(
            "T-Invest current identity is not available for trading; T-Invest and MOEX confirm "
            "historical CHEP rows through 2021-09-21; MOEX event-date probes for sampled 2026 "
            "CHEP dates returned no trading rows."
        ),
    )


def _current_candidate(report: dict[str, Any]) -> dict[str, Any]:
    rows = cast("list[dict[str, Any]]", report["instrument_candidates"])
    matches = [
        row
        for row in rows
        if row.get("classification") == "CURRENT_CONFIRMED"
        and row.get("ticker") == EXPECTED_CHEP_TICKER
        and row.get("figi") == EXPECTED_CHEP_FIGI
        and row.get("instrument_uid") == EXPECTED_CHEP_UID
    ]
    if len(matches) != 1:
        raise ValueError("CURRENT_CHEP_IDENTITY_NOT_CONFIRMED")
    return matches[0]


def _last_confirmed_trade_date(report: dict[str, Any]) -> date:
    probes = cast("list[dict[str, Any]]", report["candle_probes"])
    candidates = [
        str(row["last_returned_timestamp"])[:10]
        for row in probes
        if row.get("label") == "known_last_daily_window_current_identity"
        and row.get("interval") == "1d"
        and int(row.get("returned_candle_count", 0)) > 0
    ]
    moex = cast("dict[str, Any]", report["moex_cross_check"])
    for row in cast("list[dict[str, Any]]", moex["MOEX_REQUESTS"]):
        if (
            row.get("label") == "last_known_tinvest_daily_history"
            and int(row.get("returned_row_count", 0)) > 0
            and row.get("last_returned_timestamp") is not None
        ):
            candidates.append(str(row["last_returned_timestamp"])[:10])
    if not candidates:
        raise ValueError("LAST_CONFIRMED_CHEP_TRADE_DATE_MISSING")
    return date.fromisoformat(max(candidates))


def _write_artifacts(
    output_root: Path,
    *,
    policy: dict[str, object],
    result_payloads: list[dict[str, object]],
    evidence_rows: list[dict[str, object]],
    manifest: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "eligibility-policy.json", policy)
    _write_jsonl(output_root / "event-eligibility.jsonl", result_payloads)
    _write_jsonl(output_root / "trading-evidence.jsonl", evidence_rows)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        "Data-quality gate for exact-event security tradability eligibility.",
        "",
        f"- ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        f"- INPUT_DIAGNOSTIC_ARTIFACT_SHA={manifest['INPUT_DIAGNOSTIC_ARTIFACT_SHA']}",
        f"- ELIGIBILITY_POLICY_SHA={manifest['ELIGIBILITY_POLICY_SHA']}",
        f"- CHEP_COHORT_SHA={manifest['CHEP_COHORT_SHA']}",
        f"- TRADING_EVIDENCE_PROVENANCE_SHA={manifest['TRADING_EVIDENCE_PROVENANCE_SHA']}",
        f"- ELIGIBILITY_RESULT_SHA={manifest['ELIGIBILITY_RESULT_SHA']}",
        "",
        f"- CANONICAL_EXACT_EVENTS_TOTAL={manifest['CANONICAL_EXACT_EVENTS_TOTAL']}",
        "- MARKET_REACTION_ELIGIBLE_EXACT_EVENTS="
        f"{manifest['MARKET_REACTION_ELIGIBLE_EXACT_EVENTS']}",
        "- MARKET_REACTION_INELIGIBLE_EXACT_EVENTS="
        f"{manifest['MARKET_REACTION_INELIGIBLE_EXACT_EVENTS']}",
        f"- REACTION_READY_EVENTS={manifest['REACTION_READY_EVENTS']}",
        f"- FEATURE_READY_EVENTS={manifest['FEATURE_READY_EVENTS']}",
        "",
        f"- CHEP_HISTORICAL_EXACT_EVENTS={manifest['CHEP_HISTORICAL_EXACT_EVENTS']}",
        f"- CHEP_EVENT_VALID={manifest['CHEP_EVENT_VALID']}",
        f"- CHEP_MARKET_REACTION_ELIGIBLE={manifest['CHEP_MARKET_REACTION_ELIGIBLE']}",
        f"- CHEP_SECURITY_NOT_TRADING={manifest['CHEP_SECURITY_NOT_TRADING']}",
        f"- CHEP_REACTION_ATTEMPTS_SKIPPED={manifest['CHEP_REACTION_ATTEMPTS_SKIPPED']}",
        f"- CHEP_COLLECTION_DECISION={manifest['CHEP_COLLECTION_DECISION']}",
        "",
        f"FINAL_DECISION={manifest['FINAL_DECISION']}",
        "",
        "EVENT_VALIDITY and MARKET_REACTION_ELIGIBILITY are intentionally separate.",
        "No canonical events were deleted or rewritten.",
    ]
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
