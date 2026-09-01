from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_dataset_readiness_audit.ml_v2 import (
    EXPECTED_INPUT_ARTIFACT_SHA,
    EXPECTED_RULES_V3_FINGERPRINT,
)
from src.issuer_historical_diversity_recovery.domain import (
    ARTIFACT_VERSION,
    CANONICAL_COHORT,
    DEFAULT_BACKFILL_ROOT,
    DEFAULT_CHEP_MATURATION_ROOT,
    DEFAULT_CONSOLIDATED_MATURATION_ROOT,
    DEFAULT_ISSUER_DIVERSITY_ROOT,
    DEFAULT_ML_V2_READINESS_ROOT,
    DEFAULT_TZ_DISCOVERY_ROOT,
    FUTURE_HOLDOUT_START,
    SOURCE_OPTIONS_VERSION,
    RecoveryDecision,
    SourceOption,
    SourceOptionStatus,
    artifact_sha,
    safety_flags,
    sha256_payload,
)

HORIZONS = ("1m", "5m", "15m", "30m", "60m")
ISSUER_SOURCE_MARKERS = (
    "MAGNIT",
    "NORNICKEL",
    "ROSNEFT",
    "TBANK",
    "VK",
    "X5",
    "YANDEX",
    "CHEP",
    "NOVATEK",
    "LUKOIL",
    "TATNEFT",
    "ALROSA",
    "PHOSAGRO",
    "POLYUS",
    "INTERRAO",
)


def run_historical_issuer_diversity_recovery_audit(
    *,
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    backfill_root: Path = Path(DEFAULT_BACKFILL_ROOT),
    readiness_root: Path = Path(DEFAULT_ML_V2_READINESS_ROOT),
    tz_discovery_root: Path = Path(DEFAULT_TZ_DISCOVERY_ROOT),
    issuer_diversity_root: Path = Path(DEFAULT_ISSUER_DIVERSITY_ROOT),
    consolidated_root: Path = Path(DEFAULT_CONSOLIDATED_MATURATION_ROOT),
    chep_root: Path = Path(DEFAULT_CHEP_MATURATION_ROOT),
    created_at: datetime | None = None,
    env_names: list[str] | None = None,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("immutable historical issuer diversity recovery output exists")
    if rules_v3_fingerprint() != EXPECTED_RULES_V3_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_CHANGED")

    readiness_manifest = _read_json(readiness_root / "manifest.json")
    gate = _read_json(readiness_root / "canonical-gate-criteria.json")
    _require_readiness_contract(readiness_manifest, gate)

    backfill_manifest = _read_json(backfill_root / "manifest.json")
    if backfill_manifest.get("ARTIFACT_SHA") != EXPECTED_INPUT_ARTIFACT_SHA:
        raise ValueError("BACKFILL_ARTIFACT_SHA_MISMATCH")

    events = _read_jsonl(backfill_root / "events.jsonl")
    features = _read_jsonl(backfill_root / "features.jsonl")
    targets = _read_jsonl(backfill_root / "targets.jsonl")
    materials = _read_jsonl(backfill_root / "semantic-material-provenance.jsonl")

    feature_ids = {str(row["event_id"]) for row in features}
    historical_issuer_rows = [
        _event_audit_row(row, feature_ids)
        for row in events
        if _is_historical_strict_exact_issuer(row)
    ]
    issuer_feature_ready = [row for row in historical_issuer_rows if row["feature_ready"]]
    target_availability = _target_availability(targets)
    material_by_id = {str(row.get("event_id")): row for row in materials}

    ticker_gap = _ticker_gap_analysis(
        issuer_feature_ready, historical_issuer_rows, target_availability
    )
    current_gap = _current_gap_analysis(issuer_feature_ready)
    semantic_unknown = _semantic_unknown_analysis(issuer_feature_ready, material_by_id)
    concentration_scenarios = _concentration_scenarios(issuer_feature_ready)

    previous_sources = _previous_source_options(tz_discovery_root, issuer_diversity_root)
    paid_options = _paid_and_authenticated_options(env_names or [])
    cache_recovery = _cache_recovery_audit(consolidated_root, chep_root)
    source_options = sorted(
        [*previous_sources, *paid_options, *cache_recovery["source_options"]],
        key=lambda row: (row["provider"], row["ticker_scope"], row["mechanism"], row["status"]),
    )
    source_summary = _source_option_summary(source_options)
    mechanism_evaluation = _mechanism_evaluation(source_options, env_names or [])
    new_mechanism_evaluation = [
        row
        for row in mechanism_evaluation
        if row["mechanism"] not in {"PUBLIC_HTML_ARCHIVE", "PUBLIC_IR_NEWS_ARCHIVE", "RSS"}
    ]
    protection = {
        "historical_cutoff": FUTURE_HOLDOUT_START,
        "future_metadata_only_allowed_fields": [
            "count metadata",
            "source identity",
            "ticker",
            "timestamp-quality metadata",
        ],
        "FUTURE_OUTCOMES_READ": 0,
        "FUTURE_TARGETS_READ": 0,
        "FUTURE_PRICE_LOOKUPS": 0,
    }
    decision = _decision(source_summary, cache_recovery)
    flags = safety_flags()
    now = created_at or datetime.now(UTC)
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": now.isoformat(),
        "git_sha": git_sha,
        "BASE_MAIN_SHA": base_main_sha,
        "CANONICAL_COHORT": CANONICAL_COHORT,
        "SOURCE_OPTIONS_VERSION": SOURCE_OPTIONS_VERSION,
        "ML_V2_READINESS_ARTIFACT_SHA": readiness_manifest["ARTIFACT_SHA"],
        "BACKFILL_ARTIFACT_SHA": backfill_manifest["ARTIFACT_SHA"],
        "RULES_V3_FINGERPRINT": rules_v3_fingerprint(),
        "READINESS_THRESHOLDS_CHANGED": False,
        "CURRENT_ISSUER_ROWS": current_gap["current_issuer_rows"],
        "CURRENT_ISSUER_TICKERS": current_gap["current_issuer_tickers"],
        "DOMINANT_TICKER": current_gap["dominant_ticker"],
        "DOMINANT_TICKER_ROWS": current_gap["dominant_ticker_rows"],
        "DOMINANT_TICKER_SHARE": current_gap["dominant_ticker_share"],
        "ROWS_REQUIRED_TOP1_LE_50": current_gap["rows_required_top1_le_50"],
        "NON_UNKNOWN_ROWS_REQUIRED_UNKNOWN_LE_50": current_gap[
            "non_unknown_rows_required_unknown_le_50"
        ],
        "EXHAUSTED_HISTORICAL_SOURCE_CANDIDATES": source_summary[
            "exhausted_historical_source_candidates"
        ],
        "NEW_MECHANISMS_EVALUATED": len(new_mechanism_evaluation),
        "ALL_MECHANISMS_TRACKED": len(mechanism_evaluation),
        "NEW_STRICT_EXACT_HISTORICAL_ISSUER_TICKERS_FOUND": 0,
        "NEW_HISTORICAL_EVENTS_FOUND": 0,
        "NEW_FEATURE_READY_EVENTS": 0,
        "RECOVERED_FROM_EXISTING_CACHE_EVENTS": cache_recovery["recovered_feature_ready_events"],
        "PAID_AUTHENTICATED_VIABLE_SOURCES_FOUND": source_summary[
            "paid_authenticated_viable_sources"
        ],
        "FINAL_DECISION": decision.value,
        "STRICT_ANSWER": "NO",
        "MISSING_RESOURCE": (
            "licensing/access to official historical disclosure API with publication timestamp "
            "and timezone provenance"
            if decision == RecoveryDecision.PAID_OR_AUTHENTICATED_SOURCE_REQUIRED
            else "safe historical data availability"
        ),
        "NEXT_RECOMMENDED_ACTION": _next_action(decision),
        "CURRENT_GAP_ANALYSIS_SHA": sha256_payload(current_gap),
        "TICKER_GAP_ANALYSIS_SHA": sha256_payload(ticker_gap),
        "SEMANTIC_UNKNOWN_ANALYSIS_SHA": sha256_payload(semantic_unknown),
        "CONCENTRATION_SCENARIOS_SHA": sha256_payload(concentration_scenarios),
        "SOURCE_OPTIONS_REGISTRY_SHA": sha256_payload(source_options),
        "SOURCE_OPTION_SUMMARY_SHA": sha256_payload(source_summary),
        "MECHANISM_EVALUATION_SHA": sha256_payload(mechanism_evaluation),
        "CACHE_RECOVERY_AUDIT_SHA": sha256_payload(cache_recovery),
        "HISTORICAL_CUTOFF_PROTECTION_SHA": sha256_payload(protection),
        "INPUT_FILE_SHAS": {
            "readiness_manifest_sha": sha256_payload(readiness_manifest),
            "readiness_gate_sha": sha256_payload(gate),
            "backfill_manifest_sha": sha256_payload(backfill_manifest),
            "events_sha": sha256_payload(events),
            "features_sha": sha256_payload(features),
            "targets_availability_sha": sha256_payload(target_availability),
            "materials_sha": sha256_payload(materials),
        },
        "safety": flags,
        **flags,
    }
    manifest["ARTIFACT_SHA"] = artifact_sha(manifest)

    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "current-gap-analysis.json", current_gap)
    _write_jsonl(output_root / "ticker-gap-analysis.jsonl", ticker_gap)
    _write_json(output_root / "semantic-unknown-analysis.json", semantic_unknown)
    _write_json(output_root / "concentration-scenarios.json", concentration_scenarios)
    _write_jsonl(output_root / "source-options-registry.jsonl", source_options)
    _write_json(output_root / "source-option-summary.json", source_summary)
    _write_jsonl(output_root / "mechanism-evaluation.jsonl", mechanism_evaluation)
    _write_json(output_root / "cache-recovery-audit.json", cache_recovery)
    _write_json(output_root / "historical-cutoff-protection.json", protection)
    _write_report(output_root / "report.md", manifest, current_gap, source_summary)
    return manifest


def validate_source_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a machine-readable source-contract decision without using market outcomes."""
    published_at = str(candidate.get("published_at") or "")
    timezone_evidence = str(candidate.get("timezone_evidence") or "")
    ticker = str(candidate.get("ticker") or "")
    attributed = str(candidate.get("ticker_attribution") or "")
    timestamp_field = str(candidate.get("timestamp_field") or "")
    if candidate.get("date_modified_used"):
        return _rejected("DATE_MODIFIED_REJECTED")
    if candidate.get("synthetic_timezone"):
        return _rejected("SYNTHETIC_TIMEZONE_REJECTED")
    if not published_at or published_at[:10] >= FUTURE_HOLDOUT_START:
        return _rejected("FUTURE_OR_MISSING_PUBLICATION_REJECTED")
    if "datePublished" not in timestamp_field and "published" not in timestamp_field.lower():
        return _rejected("PUBLICATION_TIMESTAMP_MISSING")
    if not timezone_evidence or timezone_evidence == "SERVER_TIMEZONE":
        return _rejected("PUBLICATION_SPECIFIC_TIMEZONE_MISSING")
    if not ticker or attributed == "AMBIGUOUS":
        return _rejected("TICKER_ATTRIBUTION_AMBIGUOUS")
    if not candidate.get("publication_material_available"):
        return _rejected("PUBLICATION_MATERIAL_MISSING")
    return {"accepted": True, "status": SourceOptionStatus.STRICT_EXACT_HISTORICAL_CAPABLE.value}


def _rejected(reason: str) -> dict[str, Any]:
    return {"accepted": False, "rejection_reason": reason}


def _require_readiness_contract(manifest: dict[str, Any], gate: dict[str, Any]) -> None:
    if manifest.get("CANONICAL_COHORT") != CANONICAL_COHORT:
        raise ValueError("CANONICAL_ML_V2_COHORT_MISSING")
    criteria = gate.get("criteria", {})
    expected = {
        "issuer_feature_ready_rows": 500,
        "unique_issuer_tickers": 10,
        "issuer_semantic_unknown_rate": "0.50",
        "top_1_ticker_share": "0.50",
        "primary_15m_target_coverage": "0.95",
    }
    for key, value in expected.items():
        if criteria.get(key, {}).get("threshold") != value:
            raise ValueError(f"READINESS_THRESHOLD_CHANGED:{key}")


def _event_audit_row(row: dict[str, Any], feature_ids: set[str]) -> dict[str, Any]:
    metadata = cast("dict[str, Any]", row.get("metadata") or {})
    event_features = cast("dict[str, Any]", row.get("event_features") or {})
    event_id = str(metadata.get("event_id") or row.get("event_id"))
    return {
        "event_id": event_id,
        "ticker": str(metadata.get("ticker") or "UNKNOWN"),
        "issuer": str(metadata.get("issuer") or metadata.get("issuer_name") or "UNKNOWN"),
        "source_family": str(
            metadata.get("source_code") or metadata.get("source_family") or "UNKNOWN"
        ),
        "source_id": str(metadata.get("source_id") or metadata.get("source_code") or "UNKNOWN"),
        "source_item_id": str(
            metadata.get("source_item_id") or metadata.get("canonical_url") or ""
        ),
        "published_at": str(
            metadata.get("publication_timestamp_utc") or metadata.get("published_at") or ""
        ),
        "publication_date": str(metadata.get("publication_date") or "")[:10],
        "timestamp_quality": str(metadata.get("timestamp_quality") or ""),
        "publication_timestamp_mechanism": str(metadata.get("timestamp_source_field") or "UNKNOWN"),
        "publication_timezone": str(metadata.get("publication_timezone") or "UNKNOWN"),
        "feature_ready": event_id in feature_ids,
        "primary_event_type": str(event_features.get("primary_event_type") or "UNKNOWN"),
        "event_count": int(event_features.get("event_count") or 0),
        "fact_count": int(event_features.get("fact_count") or 0),
    }


def _is_historical_strict_exact_issuer(row: dict[str, Any]) -> bool:
    metadata = cast("dict[str, Any]", row.get("metadata") or {})
    published_at = str(
        metadata.get("publication_timestamp_utc") or metadata.get("published_at") or ""
    )
    source_family = str(metadata.get("source_code") or metadata.get("source_family") or "").upper()
    if str(metadata.get("timestamp_quality") or "") != "EXACT":
        return False
    if (
        bool(metadata.get("future_holdout"))
        or not published_at
        or published_at[:10] >= FUTURE_HOLDOUT_START
    ):
        return False
    if "MOEX" in source_family:
        return False
    haystack = " ".join(
        str(metadata.get(key) or "") for key in ("source_code", "source_id", "canonical_url")
    )
    return any(marker in haystack.upper() for marker in ISSUER_SOURCE_MARKERS)


def _current_gap_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ticker_counts = Counter(row["ticker"] for row in rows)
    unknown_rows = sum(1 for row in rows if row["primary_event_type"] == "UNKNOWN")
    total = len(rows)
    dominant_ticker, dominant_rows = ticker_counts.most_common(1)[0]
    rows_required_top1 = max(0, (dominant_rows * 2) - total)
    non_unknown_required = max(0, (unknown_rows * 2) - total)
    return {
        "current_issuer_rows": total,
        "current_issuer_tickers": len(ticker_counts),
        "unknown_rows": unknown_rows,
        "non_unknown_rows": total - unknown_rows,
        "issuer_unknown_rate": _share(unknown_rows, total),
        "dominant_ticker": dominant_ticker,
        "dominant_ticker_rows": dominant_rows,
        "dominant_ticker_share": _share(dominant_rows, total),
        "rows_required_top1_le_50": rows_required_top1,
        "non_unknown_rows_required_unknown_le_50": non_unknown_required,
        "independent_blockers": {
            "NEW_TICKER_DIVERSITY": {"minimum_new_tickers": 3},
            "SEMANTIC_QUALITY": {"minimum_new_non_unknown_rows": non_unknown_required},
            "CONCENTRATION": {"minimum_new_non_dominant_rows": rows_required_top1},
        },
    }


def _ticker_gap_analysis(
    feature_ready_rows: list[dict[str, Any]],
    strict_rows: list[dict[str, Any]],
    target_availability: dict[str, dict[str, bool]],
) -> list[dict[str, Any]]:
    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    strict_by_ticker: Counter[str] = Counter(row["ticker"] for row in strict_rows)
    for row in feature_ready_rows:
        by_ticker[row["ticker"]].append(row)
    total = len(feature_ready_rows)
    result: list[dict[str, Any]] = []
    for ticker in sorted(by_ticker):
        rows = by_ticker[ticker]
        unknown = sum(1 for row in rows if row["primary_event_type"] == "UNKNOWN")
        horizon_coverage = {}
        for horizon in HORIZONS:
            available = sum(
                1
                for row in rows
                if target_availability.get(row["event_id"], {}).get(horizon) is True
            )
            horizon_coverage[horizon] = {
                "available": available,
                "missing": len(rows) - available,
                "coverage": _share(available, len(rows)),
            }
        result.append(
            {
                "ticker": ticker,
                "issuer": _counter_payload(row["issuer"] for row in rows),
                "source_families": _counter_payload(row["source_family"] for row in rows),
                "strict_exact_rows": strict_by_ticker[ticker],
                "feature_ready_rows": len(rows),
                "unknown_rows": unknown,
                "unknown_rate": _share(unknown, len(rows)),
                "share_of_canonical_cohort": _share(len(rows), total),
                "first_publication_date": min(row["published_at"][:10] for row in rows),
                "last_publication_date": max(row["published_at"][:10] for row in rows),
                "primary_event_type_distribution": _counter_payload(
                    row["primary_event_type"] for row in rows
                ),
                "fact_count_distribution": _counter_payload(str(row["fact_count"]) for row in rows),
                "target_coverage": horizon_coverage,
                "publication_timestamp_mechanism": _counter_payload(
                    f"{row['publication_timestamp_mechanism']} / {row['publication_timezone']}"
                    for row in rows
                ),
            }
        )
    return result


def _semantic_unknown_analysis(
    rows: list[dict[str, Any]], material_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    unknown = [row for row in rows if row["primary_event_type"] == "UNKNOWN"]
    by_template = Counter(_template_key(row["source_item_id"]) for row in unknown)
    source_family_unknown = Counter(row["source_family"] for row in unknown)
    source_family_total = Counter(row["source_family"] for row in rows)
    return {
        "unknown_rows": len(unknown),
        "unknown_rate": _share(len(unknown), len(rows)),
        "unknown_by_ticker": _counter_payload(row["ticker"] for row in unknown),
        "unknown_rate_by_source_family": {
            source: _share(source_family_unknown[source], total)
            for source, total in sorted(source_family_total.items())
        },
        "highest_unknown_source_families": [
            {
                "source_family": source,
                "unknown_rows": source_family_unknown[source],
                "total_rows": source_family_total[source],
                "unknown_rate": _share(source_family_unknown[source], source_family_total[source]),
            }
            for source, _count in source_family_unknown.most_common()
        ],
        "unknown_by_publication_template": dict(sorted(by_template.items())),
        "unknown_by_title_material_availability": _counter_payload(
            "material_available"
            if material_by_id.get(row["event_id"], {}).get("publication_material_available")
            else "material_missing"
            for row in unknown
        ),
        "unknown_by_event_origin": {"ISSUER_ORIGINATED": len(unknown)},
        "unknown_with_zero_facts": sum(1 for row in unknown if row["fact_count"] == 0),
        "unknown_with_extracted_facts": sum(1 for row in unknown if row["fact_count"] > 0),
        "unknown_due_unsupported_event_class": sum(1 for row in unknown if row["fact_count"] > 0),
        "unknown_likely_non_corporate_informational_publication": sum(
            1 for row in unknown if row["fact_count"] == 0 and row["event_count"] == 0
        ),
        "composition_signal": (
            "UNKNOWN is concentrated in issuer general-news feeds and rows with zero "
            "Rules v3 facts; diagnosis only, Rules v3 unchanged."
        ),
    }


def _concentration_scenarios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter(str(row["ticker"]) for row in rows)
    dominant_count = counts.most_common(1)[0][1]
    low_rep_counts = dict(counts)
    for ticker in sorted(counts, key=lambda value: counts[value]):
        low_rep_counts[ticker] += 25
    scenarios = {
        "+3_new_tickers_1_each": _projected_concentration(
            {**counts, "NEW1": 1, "NEW2": 1, "NEW3": 1}
        ),
        "+3_new_tickers_30_each": _projected_concentration(
            {**counts, "NEW1": 30, "NEW2": 30, "NEW3": 30}
        ),
        "+5_new_tickers_1_each": _projected_concentration(
            {**counts, "NEW1": 1, "NEW2": 1, "NEW3": 1, "NEW4": 1, "NEW5": 1}
        ),
        "+5_new_tickers_30_each": _projected_concentration(
            {**counts, "NEW1": 30, "NEW2": 30, "NEW3": 30, "NEW4": 30, "NEW5": 30}
        ),
        "low_representation_existing_issuers_plus_25_each": _projected_concentration(
            low_rep_counts
        ),
    }
    return {
        "current": _projected_concentration(counts),
        "dominant_ticker_rows_fixed_at": dominant_count,
        "unknown_projection_policy": (
            "UNKNOWN rate is not projected without factual semantic classification of new rows."
        ),
        "scenarios": scenarios,
    }


def _previous_source_options(tz_root: Path, issuer_diversity_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if (tz_root / "audited-sources.jsonl").exists():
        for row in _read_jsonl(tz_root / "audited-sources.jsonl"):
            status = _map_previous_status(str(row.get("status") or ""))
            rows.append(
                SourceOption(
                    provider=str(row.get("official_domain") or "unknown"),
                    ticker_scope=str(row.get("ticker") or "UNKNOWN"),
                    issuer_scope=str(row.get("issuer") or "UNKNOWN"),
                    mechanism=str(row.get("source_mechanism") or "PUBLIC_HTML_ARCHIVE"),
                    status=status,
                    evidence_source="timezone-verified-issuer-exact-source-discovery-v2",
                    evidence_url=str(row.get("source_url") or ""),
                    timestamp_contract=str(row.get("primary_blocker") or row.get("status") or ""),
                    timezone_contract="publication-specific timezone evidence missing",
                    historical_archive=str(row.get("historical_archive_available")),
                    publication_identity=str(row.get("source_family") or ""),
                    access_status="reused immutable evidence; no re-audit",
                    storage_or_license="zero-cost public candidate",
                    internal_ml_research_status="not accepted for strict-EXACT historical training",
                    no_reaudit_reason="HISTORICAL_STRICT_EXACT_SOURCES_EFFECTIVELY_EXHAUSTED",
                ).payload()
            )
    if (issuer_diversity_root / "candidate-sources.jsonl").exists():
        for row in _read_jsonl(issuer_diversity_root / "candidate-sources.jsonl"):
            status = _map_previous_status(str(row.get("status") or ""))
            if str(row.get("source_id")) == "MVIDEOELDORADO_IR_NEWS_EXACT_V1":
                status = SourceOptionStatus.CLOCK_WITHOUT_TIMEZONE
            rows.append(
                SourceOption(
                    provider=str(row.get("official_domain") or "unknown"),
                    ticker_scope=str(row.get("ticker") or "UNKNOWN"),
                    issuer_scope=str(row.get("issuer") or "UNKNOWN"),
                    mechanism=str(row.get("mechanism") or "PUBLIC_HTML_ARCHIVE"),
                    status=status,
                    evidence_source="issuer-exact-historical-diversity-expansion-v1",
                    evidence_url=str(row.get("source_url") or ""),
                    timestamp_contract=str(row.get("selection_reason") or row.get("status") or ""),
                    timezone_contract=str(row.get("source_selection_notes") or ""),
                    historical_archive=str(row.get("historical_depth_estimate") or ""),
                    publication_identity=str(
                        row.get("source_id") or row.get("source_family") or ""
                    ),
                    access_status="reused immutable evidence; no re-audit",
                    storage_or_license="zero-cost public candidate",
                    internal_ml_research_status="not accepted after timezone/maturation audit",
                    no_reaudit_reason="prior immutable evidence still current for method class",
                ).payload()
            )
    return _dedupe_options(rows)


def _paid_and_authenticated_options(env_names: list[str]) -> list[dict[str, Any]]:
    has_tinvest = "TINVEST_READONLY_TOKEN" in set(env_names)
    options = [
        SourceOption(
            provider="Interfax CRKI e-disclosure Gateway",
            ticker_scope="Russian listed issuers via disclosure subjects",
            issuer_scope="issuer disclosures and publicator events",
            mechanism="REST_JSON_AUTHENTICATED_DISCLOSURE_API",
            status=SourceOptionStatus.PAID_LICENSE_REQUIRED,
            evidence_source="public provider documentation",
            evidence_url="https://e-disclosure.ru/poluchenie-informacii/shlyuz-api",
            timestamp_contract=(
                "DisclosureEvents/publicator-events expose structured publication events"
            ),
            timezone_contract="must be verified in trial response contract before ingestion",
            historical_archive="paid archive option advertised to 2020-07-01",
            publication_identity="Disclosure event/document/message uid",
            access_status="no project credential detected; do not purchase automatically",
            storage_or_license="paid monthly subscription; archive surcharge; contract required",
            internal_ml_research_status=(
                "viable only after license permits storage/internal ML research"
            ),
        ).payload(),
        SourceOption(
            provider="Interfax CRKI e-disclosure FTP",
            ticker_scope="filtered issuers/messages/documents",
            issuer_scope="issuer disclosure messages and documents",
            mechanism="PAID_FTP_XML_DAILY_EXPORT",
            status=SourceOptionStatus.PAID_LICENSE_REQUIRED,
            evidence_source="public provider documentation",
            evidence_url="https://www.e-disclosure.ru/poluchenie-informacii/vygruzka-na-ftp",
            timestamp_contract=(
                "XML metadata for disclosed messages/documents; exact timezone needs "
                "contract sample"
            ),
            timezone_contract="must be verified from sample XML/documentation",
            historical_archive="daily export service, archive depth must be contracted",
            publication_identity="provider XML identity/file uid",
            access_status="no project credential detected; do not purchase automatically",
            storage_or_license="paid FTP delivery contract required",
            internal_ml_research_status="candidate only after license review",
        ).payload(),
        SourceOption(
            provider="MOEX Corporate Information Center",
            ticker_scope="MOEX issuer universe",
            issuer_scope=(
                "corporate actions, issuer profiles, IR calendar and corporate information"
            ),
            mechanism="PAID_MOEX_CKI_API",
            status=SourceOptionStatus.NEW_MECHANISM_NOT_YET_TESTED,
            evidence_source="public provider documentation",
            evidence_url="https://www.moex.com/tsentr-korporativnoj-informatsii",
            timestamp_contract=(
                "structured corporate data via API/FTP; disclosure publication timestamp not proven"
            ),
            timezone_contract=(
                "publication-specific timestamp/timezone must be proven before strict-EXACT use"
            ),
            historical_archive=(
                "historical issuer data advertised for 10-15 years depending on issuer"
            ),
            publication_identity="MOEX CKI record identity",
            access_status="paid/demo access required",
            storage_or_license="license terms required",
            internal_ml_research_status="not yet proven sufficient for strict-EXACT issuer events",
        ).payload(),
        SourceOption(
            provider="T-Invest existing read-only token",
            ticker_scope="market data only",
            issuer_scope="not an issuer disclosure source",
            mechanism="EXISTING_AUTHENTICATED_MARKET_DATA_API",
            status=SourceOptionStatus.POLICY_BLOCKED,
            evidence_source="environment capability scan",
            evidence_url=None,
            timestamp_contract="market candle timestamps, not issuer publication timestamps",
            timezone_contract="not applicable for publication evidence",
            historical_archive="minute market history capability only",
            publication_identity="none for issuer disclosures",
            access_status="credential name present" if has_tinvest else "credential name absent",
            storage_or_license="project market-data path only",
            internal_ml_research_status="cannot close issuer disclosure ticker gap",
        ).payload(),
    ]
    return options


def _cache_recovery_audit(consolidated_root: Path, chep_root: Path) -> dict[str, Any]:
    source_options: list[dict[str, Any]] = []
    per_artifact: list[dict[str, Any]] = []
    recovered = 0
    for root, name in (
        (consolidated_root, "consolidated-active-exact-historical-maturation-v1"),
        (chep_root, "chep-historical-exact-maturation-v1-cache-only"),
    ):
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        feature_ready = int(
            manifest.get("NEW_FEATURE_READY") or manifest.get("NEW_FEATURE_READY_EVENTS") or 0
        )
        recovered += feature_ready
        per_ticker = cast("dict[str, Any]", manifest.get("PER_TICKER") or {})
        tickers = sorted(str(ticker) for ticker in per_ticker)
        if not tickers and "CHEP_HISTORICAL_EVENTS_TOTAL" in manifest:
            tickers = ["CHEP"]
        per_artifact.append(
            {
                "artifact": name,
                "artifact_sha": manifest.get("ARTIFACT_SHA"),
                "candidate_tickers": tickers,
                "historical_events": int(
                    manifest.get("COMBINED_HISTORICAL_INPUT")
                    or manifest.get("CHEP_HISTORICAL_EVENTS_TOTAL")
                    or 0
                ),
                "feature_ready_events": feature_ready,
                "primary_blockers": manifest.get("BLOCKER_COUNTS") or {},
            }
        )
        for ticker in tickers:
            source_options.append(
                SourceOption(
                    provider=name,
                    ticker_scope=ticker,
                    issuer_scope=ticker,
                    mechanism="EXISTING_LOCAL_CACHE_RECOVERY_AUDIT",
                    status=SourceOptionStatus.TECHNICAL_BLOCKER,
                    evidence_source=name,
                    evidence_url=None,
                    timestamp_contract="existing local exact publication snapshots reviewed",
                    timezone_contract="see upstream artifact",
                    historical_archive="local artifact/cache",
                    publication_identity="upstream event_id/source_item_id",
                    access_status="local cache present but not feature-ready recoverable",
                    storage_or_license="local project artifact",
                    internal_ml_research_status="no recovered feature-ready issuer rows",
                    candidate_count=1,
                    no_reaudit_reason="local recovery artifact already computed",
                ).payload()
            )
    return {
        "artifacts_reviewed": per_artifact,
        "recovered_feature_ready_events": recovered,
        "source_options": source_options,
    }


def _source_option_summary(options: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(str(row["status"]) for row in options)
    exhausted = sum(
        1
        for row in options
        if row["status"]
        in {
            SourceOptionStatus.DATE_ONLY.value,
            SourceOptionStatus.CLOCK_WITHOUT_TIMEZONE.value,
            SourceOptionStatus.TECHNICAL_BLOCKER.value,
            SourceOptionStatus.POLICY_BLOCKED.value,
            SourceOptionStatus.ALREADY_EXHAUSTED.value,
        }
        and row["evidence_source"]
        in {
            "timezone-verified-issuer-exact-source-discovery-v2",
            "issuer-exact-historical-diversity-expansion-v1",
        }
    )
    paid_viable = sum(
        1
        for row in options
        if row["provider"] == "Interfax CRKI e-disclosure Gateway"
        and row["status"] == SourceOptionStatus.PAID_LICENSE_REQUIRED.value
    )
    return {
        "total_options": len(options),
        "by_status": dict(sorted(by_status.items())),
        "exhausted_historical_source_candidates": exhausted,
        "strict_exact_historical_capable_zero_cost_sources": by_status[
            SourceOptionStatus.STRICT_EXACT_HISTORICAL_CAPABLE.value
        ],
        "paid_authenticated_viable_sources": paid_viable,
        "no_reaudit_policy": "Prior immutable evidence is reused unless mechanism or URL changed.",
    }


def _mechanism_evaluation(
    options: list[dict[str, Any]], env_names: list[str]
) -> list[dict[str, Any]]:
    providers_by_mechanism: dict[str, set[str]] = defaultdict(set)
    statuses_by_mechanism: dict[str, Counter[str]] = defaultdict(Counter)
    for row in options:
        key = str(row["mechanism"])
        providers_by_mechanism[key].add(str(row["provider"]))
        statuses_by_mechanism[key][str(row["status"])] += 1
    result: list[dict[str, Any]] = []
    for key in sorted(providers_by_mechanism):
        result.append(
            {
                "mechanism": key,
                "providers": sorted(providers_by_mechanism[key]),
                "statuses": dict(sorted(statuses_by_mechanism[key].items())),
                "existing_credential_names_detected": sorted(
                    name for name in env_names if name in {"TINVEST_READONLY_TOKEN"}
                ),
                "network_requests_performed": 0,
                "future_outcomes_read": 0,
                "future_targets_read": 0,
                "future_price_lookups": 0,
            }
        )
    return result


def _decision(summary: dict[str, Any], cache_recovery: dict[str, Any]) -> RecoveryDecision:
    if cache_recovery["recovered_feature_ready_events"] > 0:
        return RecoveryDecision.EXISTING_LOCAL_DATA_RECOVERY_AVAILABLE
    if summary["strict_exact_historical_capable_zero_cost_sources"] >= 3:
        return RecoveryDecision.HISTORICAL_DIVERSITY_RECOVERY_READY
    if summary["paid_authenticated_viable_sources"] > 0:
        return RecoveryDecision.PAID_OR_AUTHENTICATED_SOURCE_REQUIRED
    if summary["strict_exact_historical_capable_zero_cost_sources"] > 0:
        return RecoveryDecision.PARTIAL_DIVERSITY_GAIN_ONLY
    return RecoveryDecision.NO_METHOD_SAFE_HISTORICAL_PATH_FOUND


def _next_action(decision: RecoveryDecision) -> str:
    if decision == RecoveryDecision.PAID_OR_AUTHENTICATED_SOURCE_REQUIRED:
        return (
            "Request licensed/test access to Interfax CRKI e-disclosure Gateway; verify response "
            "fields for publication timestamp timezone, storage/internal-ML rights, and at least "
            "3 pre-2026-08-11 issuer tickers before any ingestion."
        )
    return (
        "Keep live strict-EXACT issuer accumulation and postpone ML v2 until the canonical "
        "gate can be met without old TEST/future holdout reuse."
    )


def _map_previous_status(status: str) -> SourceOptionStatus:
    mapping = {
        "DATE_ONLY": SourceOptionStatus.DATE_ONLY,
        "CLOCK_TIME_WITHOUT_TIMEZONE": SourceOptionStatus.CLOCK_WITHOUT_TIMEZONE,
        "CLOCK_WITHOUT_TIMEZONE": SourceOptionStatus.CLOCK_WITHOUT_TIMEZONE,
        "TECHNICAL_BLOCKER": SourceOptionStatus.TECHNICAL_BLOCKER,
        "POLICY_BLOCKED": SourceOptionStatus.POLICY_BLOCKED,
        "NEW_EXACT_HISTORICAL_CAPABLE": SourceOptionStatus.STRICT_EXACT_HISTORICAL_CAPABLE,
        "STRICT_EXACT_HISTORICAL_READY": SourceOptionStatus.STRICT_EXACT_HISTORICAL_CAPABLE,
    }
    return mapping.get(status, SourceOptionStatus.ALREADY_EXHAUSTED)


def _target_availability(targets: list[dict[str, Any]]) -> dict[str, dict[str, bool]]:
    result: dict[str, dict[str, bool]] = {}
    for row in targets:
        event_id = str(row.get("event_id") or "")
        horizons = cast("dict[str, Any]", row.get("horizons") or {})
        result[event_id] = {
            horizon: bool(cast("dict[str, Any]", horizons.get(horizon) or {}).get("available"))
            for horizon in HORIZONS
        }
    return result


def _projected_concentration(counts: dict[str, int] | Counter[str]) -> dict[str, Any]:
    typed = {str(key): int(value) for key, value in counts.items()}
    total = sum(typed.values())
    return {
        "total_issuer_rows": total,
        "unique_tickers": len(typed),
        "projected_top_1_share": _top_share(typed, 1),
        "projected_top_3_share": _top_share(typed, 3),
        "ticker_hhi": _hhi(typed),
        "effective_ticker_count": _effective_count(typed),
    }


def _top_share(counts: dict[str, int], top_n: int) -> str:
    return _share(sum(sorted(counts.values(), reverse=True)[:top_n]), sum(counts.values()))


def _hhi(counts: dict[str, int]) -> str:
    total = Decimal(sum(counts.values()))
    if total == 0:
        return "0.000000"
    return _fmt(sum(((Decimal(count) / total) ** 2 for count in counts.values()), Decimal("0")))


def _effective_count(counts: dict[str, int]) -> str:
    hhi = Decimal(_hhi(counts))
    if hhi == 0:
        return "0.000000"
    return _fmt(Decimal("1") / hhi)


def _share(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000000"
    return _fmt(Decimal(numerator) / Decimal(denominator))


def _fmt(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.000001'))}"


def _counter_payload(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _template_key(source_item_id: str) -> str:
    if not source_item_id:
        return "UNKNOWN_TEMPLATE"
    head = source_item_id.split("?")[0].strip("/")
    pieces = [piece for piece in head.split("/") if piece]
    if len(pieces) <= 1:
        return head or "ROOT"
    return "/".join(pieces[:3])


def _dedupe_options(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("provider")),
            str(row.get("ticker_scope")),
            str(row.get("mechanism")),
            str(row.get("publication_identity")),
        )
        deduped[key] = row
    return list(deduped.values())


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    current_gap: dict[str, Any],
    source_summary: dict[str, Any],
) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        f"- STRICT_ANSWER={manifest['STRICT_ANSWER']}",
        f"- FINAL_DECISION={manifest['FINAL_DECISION']}",
        f"- CURRENT_ISSUER_ROWS={manifest['CURRENT_ISSUER_ROWS']}",
        f"- CURRENT_ISSUER_TICKERS={manifest['CURRENT_ISSUER_TICKERS']}",
        (
            f"- DOMINANT_TICKER={current_gap['dominant_ticker']} "
            f"{current_gap['dominant_ticker_rows']} / {current_gap['dominant_ticker_share']}"
        ),
        f"- ROWS_REQUIRED_TOP1_LE_50={manifest['ROWS_REQUIRED_TOP1_LE_50']}",
        (
            "- NON_UNKNOWN_ROWS_REQUIRED_UNKNOWN_LE_50="
            f"{manifest['NON_UNKNOWN_ROWS_REQUIRED_UNKNOWN_LE_50']}"
        ),
        (
            "- EXHAUSTED_HISTORICAL_SOURCE_CANDIDATES="
            f"{source_summary['exhausted_historical_source_candidates']}"
        ),
        (
            "- PAID_AUTHENTICATED_VIABLE_SOURCES_FOUND="
            f"{manifest['PAID_AUTHENTICATED_VIABLE_SOURCES_FOUND']}"
        ),
        f"- FUTURE_OUTCOMES_READ={manifest['FUTURE_OUTCOMES_READ']}",
        f"- FUTURE_TARGETS_READ={manifest['FUTURE_TARGETS_READ']}",
        f"- FUTURE_PRICE_LOOKUPS={manifest['FUTURE_PRICE_LOOKUPS']}",
        "",
        "## Decision",
        "",
        manifest["NEXT_RECOMMENDED_ACTION"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
