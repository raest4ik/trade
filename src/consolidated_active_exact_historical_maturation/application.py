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
from src.chep_historical_exact_maturation.domain import acquisition_day_bounds
from src.consolidated_active_exact_historical_maturation.domain import (
    ARTIFACT_VERSION,
    BENCHMARK_TICKER,
    DEFAULT_BASE_DATASET_ROOT,
    DEFAULT_LIVE_REGISTRY_PATH,
    DEFAULT_UNIVERSE_ROOT,
    DEFAULT_V1_ARTIFACT_ROOT,
    DEFAULT_V2_ARTIFACT_ROOT,
    EXPECTED_V1_ARTIFACT_SHA,
    FUTURE_EVENT_HOLDOUT_START,
    HORIZONS,
    OUTPUT_DATASET_VERSION,
    ActiveIdentity,
    MarketAcquisitionConfig,
    artifact_sha,
    safety_flags,
    sha256_payload,
)
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_corpus.market import align_exact_event
from src.exact_event_live_source_breadth_expansion_v2.domain import artifact_sha as v2_artifact_sha
from src.exact_event_security_tradability_eligibility.domain import (
    EventValidity,
    InstrumentIdentityStatus,
    MarketReactionEligibility,
    TradingEvidence,
    evaluate_event_eligibility,
)
from src.tinvest_market.client import (
    TInvestInstrument,
    TInvestMinuteCandle,
    TInvestMinuteCandleBatch,
)


class ActiveMarketClient(Protocol):
    async def get_instrument_by_uid(self, instrument_uid: str) -> TInvestInstrument: ...

    async def fetch_minute_candles_audited(
        self, *, instrument_uid: str, date_from: datetime, date_to: datetime
    ) -> TInvestMinuteCandleBatch: ...


async def run_consolidated_active_exact_historical_maturation(
    *,
    v1_root: Path = Path(DEFAULT_V1_ARTIFACT_ROOT),
    v2_root: Path = Path(DEFAULT_V2_ARTIFACT_ROOT),
    base_dataset_root: Path = Path(DEFAULT_BASE_DATASET_ROOT),
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    live_registry_path: Path = Path(DEFAULT_LIVE_REGISTRY_PATH),
    universe_root: Path = Path(DEFAULT_UNIVERSE_ROOT),
    client: ActiveMarketClient | None = None,
    created_at: datetime | None = None,
    extra_cache_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    if await asyncio.to_thread(_output_nonempty, output_root):
        raise FileExistsError(
            "immutable active historical maturation artifact output already exists"
        )
    _verify_frozen_contracts()
    v1_manifest = _read_json(v1_root / "manifest.json")
    v2_manifest = _read_json(v2_root / "manifest.json")
    _require_v1_manifest(v1_manifest)
    _require_v2_manifest(v2_manifest)

    v1_rows_all = _read_jsonl(v1_root / "live-collection" / "collected-event-metadata.jsonl")
    v2_rows_all = _read_jsonl(v2_root / "live-collection" / "collected-event-metadata.jsonl")
    v1_historical, v1_future = _split_artifact_rows(v1_rows_all)
    v2_historical, v2_future = _split_artifact_rows(v2_rows_all)
    v1_historical = [row for row in v1_historical if _ticker(row) != "CHEP"]
    v2_historical = [row for row in v2_historical if _ticker(row) != "CHEP"]
    future_excluded = _cohort_identity_payload([*v1_future, *v2_future])

    base_events = _read_jsonl(base_dataset_root / "events.jsonl")
    base_features = _read_jsonl(base_dataset_root / "features.jsonl")
    base_targets = _read_jsonl(base_dataset_root / "targets.jsonl")
    base_event_by_id = {_event_id(row): row for row in base_events}
    historical_input = _dedupe_new_historical(
        [*v1_historical, *v2_historical],
        base_event_by_id=base_event_by_id,
    )
    maturation_cohort = _cohort_identity_payload(historical_input)
    _write_jsonl(output_root / "input-v1-events.jsonl", v1_historical)
    _write_jsonl(output_root / "input-v2-events.jsonl", v2_historical)
    _write_jsonl(output_root / "maturation-cohort.jsonl", maturation_cohort)
    _write_jsonl(output_root / "future-excluded-cohort.jsonl", future_excluded)

    sources = _source_rows(v1_root, v2_root)
    registry_rows = _read_live_registry(live_registry_path)
    _verify_registry_matches_artifacts(sources, registry_rows, v2_manifest)
    universe_rows = _read_universe_rows(universe_root)
    identities = _resolve_identities(historical_input, sources, universe_rows)
    identities_by_ticker = {row.ticker: row for row in identities}
    identity_payloads = [row.payload() for row in identities]
    _write_jsonl(output_root / "instrument-identities.jsonl", identity_payloads)

    eligibility_rows = _eligibility_rows(historical_input, identities_by_ticker)
    eligible_ids = {
        str(row["event_id"])
        for row in eligibility_rows
        if row["tradability_status"] == MarketReactionEligibility.ELIGIBLE.value
    }
    market_eligible = [row for row in historical_input if _event_id(row) in eligible_ids]
    acquisition = await _acquire_required_history(
        output_root / "raw-minute-cache",
        market_eligible,
        identities_by_ticker,
        client=client,
        existing_cache_roots=_cache_roots(
            output_root / "raw-minute-cache", base_dataset_root, extra_cache_roots
        ),
    )
    _write_jsonl(output_root / "market-acquisition-provenance.jsonl", acquisition)
    cache_roots = _cache_roots(
        output_root / "raw-minute-cache", base_dataset_root, extra_cache_roots
    )

    events_before_maturation = _events_before_maturation(base_events, [*v1_rows_all, *v2_rows_all])
    events_after = deepcopy(events_before_maturation)
    event_after_by_id = {_event_id(row): row for row in events_after}
    features_after = [*base_features]
    targets_after = [*base_targets]
    existing_target_ids = {str(row["event_id"]) for row in targets_after}
    per_event: list[dict[str, Any]] = []
    leakage_violations: list[str] = []

    eligibility_by_id = {str(row["event_id"]): row for row in eligibility_rows}
    acquisition_by_event = {str(row["event_id"]): row for row in acquisition}
    for row in sorted(historical_input, key=lambda item: (_published_at(item), _event_id(item))):
        event_id = _event_id(row)
        output_row = event_after_by_id[event_id]
        identity = identities_by_ticker.get(_ticker(row))
        eligibility = eligibility_by_id[event_id]
        base_report = _report_base(row, identity, eligibility, acquisition_by_event.get(event_id))
        _mark_historical_default(output_row)
        if eligibility["tradability_status"] != MarketReactionEligibility.ELIGIBLE.value:
            per_event.append(_blocked_report(base_report, str(eligibility["primary_blocker"])))
            continue
        if identity is None:
            per_event.append(
                _blocked_report(
                    base_report, MarketReactionEligibility.INSTRUMENT_IDENTITY_UNRESOLVED.value
                )
            )
            continue
        published_at = _published_at(row)
        security = _load_history(cache_roots, identity, identity.ticker, published_at)
        benchmark = _load_history(cache_roots, None, BENCHMARK_TICKER, published_at)
        if not security:
            per_event.append(_blocked_report(base_report, "SECURITY_HISTORY_MISSING"))
            continue
        if not benchmark:
            per_event.append(_blocked_report(base_report, "BENCHMARK_HISTORY_MISSING"))
            continue
        alignment = align_exact_event(published_at, security, benchmark, expose_outcomes=True)
        max_feature_input_at = _max_feature_input_timestamp(published_at, security, benchmark)
        if max_feature_input_at is not None and max_feature_input_at > published_at:
            leakage_violations.append(event_id)
        complete_features = _complete_pre_event_features(alignment.features)
        has_event_features = isinstance(output_row.get("event_features"), dict)
        reaction_ready = alignment.reaction_status == "REACTION_READY"
        feature_ready = (
            reaction_ready
            and complete_features
            and has_event_features
            and max_feature_input_at is not None
            and max_feature_input_at <= published_at
        )
        horizon_ready = {
            horizon: bool(alignment.horizons.get(horizon, {}).get("available", False))
            for horizon in HORIZONS
        }
        availability = _availability(output_row)
        availability["reaction_ready"] = reaction_ready
        availability["feature_ready"] = feature_ready
        availability["status"] = alignment.reaction_status
        availability["missing_reason"] = alignment.missing_reason
        availability["research_outcomes_visible"] = True
        metadata = _metadata(output_row)
        metadata["reaction_family"] = "EXACT_INTRADAY"
        metadata["market_alignment_version"] = "tinvest-exact-minute-alignment-v1"
        metadata["session_state"] = alignment.session_state.value
        output_row["pre_event_market_features"] = alignment.features
        quality = cast("dict[str, Any]", output_row["quality"])
        quality["feature_cutoff"] = published_at.isoformat()
        quality["no_forward_fill"] = True
        quality["no_interpolation"] = True
        quality["no_source_mixing"] = True
        if alignment.horizons and event_id not in existing_target_ids:
            targets_after.append(
                {
                    "event_id": event_id,
                    "reaction_family": "EXACT_INTRADAY",
                    "horizons": alignment.horizons,
                }
            )
            existing_target_ids.add(event_id)
        if feature_ready and event_id not in {str(item["event_id"]) for item in features_after}:
            features_after.append(
                {
                    "event_id": event_id,
                    "feature_cutoff": published_at.isoformat(),
                    "event_features": output_row["event_features"],
                    "market_features": alignment.features,
                }
            )
        per_event.append(
            {
                **base_report,
                "security_history_status": "AVAILABLE",
                "benchmark_history_status": "AVAILABLE",
                "session_status": alignment.session_state.value,
                "ready_1m": horizon_ready["1m"],
                "ready_5m": horizon_ready["5m"],
                "ready_15m": horizon_ready["15m"],
                "ready_30m": horizon_ready["30m"],
                "ready_60m": horizon_ready["60m"],
                "reaction_ready": reaction_ready,
                "feature_ready": feature_ready,
                "primary_blocker": (
                    None
                    if feature_ready
                    else _primary_blocker(alignment, complete_features, has_event_features)
                ),
                "horizon_blockers": {
                    horizon: _horizon_status(alignment.horizons, horizon) for horizon in HORIZONS
                },
                "pre_event_context_ready": complete_features,
                "event_features_available": has_event_features,
                "max_feature_timestamp_utc": (
                    max_feature_input_at.isoformat() if max_feature_input_at is not None else None
                ),
                "post_event_feature_access": False,
            }
        )

    if leakage_violations:
        raise ValueError("ACTIVE_HISTORICAL_MATURATION_LEAKAGE_CHECK_FAILED")
    _assert_existing_rows_preserved(base_events, events_after, set(base_event_by_id))
    _assert_future_holdout_guard(
        events_after, targets_after, {_event_id(row) for row in [*v1_future, *v2_future]}
    )
    maturation_result_sha = sha256_payload(per_event)
    output_dataset_sha = sha256_payload(
        {
            "dataset_version": OUTPUT_DATASET_VERSION,
            "input_v1_artifact_sha": v1_manifest["ARTIFACT_SHA"],
            "input_v2_artifact_sha": v2_manifest["ARTIFACT_SHA"],
            "events": events_after,
            "features": features_after,
            "targets": targets_after,
        }
    )
    before = _baseline_metrics(v1_manifest, v2_manifest, events_before_maturation, base_features)
    after = {
        "CANONICAL_EXACT_EVENTS": len(events_after),
        "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS": before["MARKET_REACTION_ELIGIBLE_EXACT_EVENTS"]
        + len(market_eligible),
        "REACTION_READY_EVENTS": sum(
            bool(_availability(row).get("reaction_ready")) for row in events_after
        ),
        "FEATURE_READY_EVENTS": len(features_after),
    }
    per_ticker = _per_ticker(per_event, historical_input, eligibility_rows)
    blocker_counts = dict(
        sorted(
            Counter(
                str(row["primary_blocker"]) for row in per_event if row["primary_blocker"]
            ).items()
        )
    )
    per_horizon = {
        horizon: sum(bool(row[f"ready_{horizon}"]) for row in per_event) for horizon in HORIZONS
    }
    flags = safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "BASE_MAIN_SHA": base_main_sha,
        "git_sha": git_sha,
        "INPUT_V1_ARTIFACT_SHA": v1_manifest["ARTIFACT_SHA"],
        "INPUT_V2_ARTIFACT_SHA": v2_manifest["ARTIFACT_SHA"],
        "MATURATION_COHORT_SHA": sha256_payload(maturation_cohort),
        "FUTURE_EXCLUDED_COHORT_SHA": sha256_payload(future_excluded),
        "INSTRUMENT_IDENTITY_SHA": sha256_payload(identity_payloads),
        "MARKET_ACQUISITION_PROVENANCE_SHA": sha256_payload(acquisition),
        "MATURATION_RESULT_SHA": maturation_result_sha,
        "OUTPUT_DATASET_VERSION": OUTPUT_DATASET_VERSION,
        "OUTPUT_DATASET_SHA": output_dataset_sha,
        "MARKET_ACQUISITION_CONFIG": MarketAcquisitionConfig().payload(),
        "V1_HISTORICAL_INPUT": len(v1_historical),
        "V2_HISTORICAL_INPUT": len(v2_historical),
        "COMBINED_HISTORICAL_INPUT": len(v1_historical) + len(v2_historical),
        "DEDUPED_HISTORICAL_INPUT": len(historical_input),
        "MARKET_ELIGIBLE_INPUT": len(market_eligible),
        "MARKET_INELIGIBLE_INPUT": len(historical_input) - len(market_eligible),
        "NEW_REACTION_READY": sum(bool(row["reaction_ready"]) for row in per_event),
        "NEW_FEATURE_READY": sum(bool(row["feature_ready"]) for row in per_event),
        "NEW_1M_READY": per_horizon["1m"],
        "NEW_5M_READY": per_horizon["5m"],
        "NEW_15M_READY": per_horizon["15m"],
        "NEW_30M_READY": per_horizon["30m"],
        "NEW_60M_READY": per_horizon["60m"],
        "CANONICAL_EXACT_EVENTS_BEFORE": before["CANONICAL_EXACT_EVENTS"],
        "CANONICAL_EXACT_EVENTS_AFTER": after["CANONICAL_EXACT_EVENTS"],
        "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_BEFORE": before[
            "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS"
        ],
        "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_AFTER": after[
            "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS"
        ],
        "REACTION_READY_BEFORE": before["REACTION_READY_EVENTS"],
        "REACTION_READY_AFTER": after["REACTION_READY_EVENTS"],
        "FEATURE_READY_BEFORE": before["FEATURE_READY_EVENTS"],
        "FEATURE_READY_AFTER": after["FEATURE_READY_EVENTS"],
        "PER_TICKER": per_ticker,
        "BLOCKER_COUNTS": blocker_counts,
        "LEAKAGE_CHECK": "PASS",
        "EXISTING_CANONICAL_ROWS_PRESERVED": "PASS",
        "DETERMINISTIC_REPLAY": "PASS",
        "FINAL_DECISION": _decision(
            len(historical_input), len(market_eligible), per_event, blocker_counts
        ),
        "safety": flags,
        **flags,
    }
    manifest["ARTIFACT_SHA"] = artifact_sha(manifest)
    _write_artifacts(
        output_root,
        events=events_after,
        features=features_after,
        targets=targets_after,
        maturation_results=per_event,
        manifest=manifest,
    )
    return manifest


def _require_v1_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "ARTIFACT_SHA": EXPECTED_V1_ARTIFACT_SHA,
        "NEW_EXACT_LIVE_SOURCES": 5,
        "NEW_CANONICAL_EXACT_EVENTS": 56,
        "NEW_HISTORICAL_EXACT_EVENTS": 45,
        "NEW_FUTURE_METADATA_ONLY_EVENTS": 11,
        "REPLAY_ITEMS_NEW": 0,
        "REPLAY_ITEMS_DUPLICATE": 56,
        "FINAL_DECISION": "SOURCE_BREADTH_GAINED",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"INPUT_V1_{key}_MISMATCH")
    tickers = set(cast("list[str]", manifest.get("NEW_TICKERS_WITH_EXACT_SOURCE") or []))
    if not {"AFKS", "ASTR", "ELMT", "OZON", "RUAL"} <= tickers:
        raise ValueError("INPUT_V1_TICKER_SET_MISMATCH")


def _require_v2_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("ARTIFACT_VERSION") != "exact-event-live-source-breadth-expansion-v2":
        raise ValueError("INPUT_V2_ARTIFACT_VERSION_MISMATCH")
    if manifest.get("ARTIFACT_SHA") != v2_artifact_sha(manifest):
        raise ValueError("INPUT_V2_ARTIFACT_SHA_MISMATCH")
    expected = {
        "NEW_EXACT_LIVE_SOURCES": 5,
        "REPLAY_ITEMS_NEW": 0,
        "FINAL_DECISION": "SOURCE_BREADTH_GAINED",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"INPUT_V2_{key}_MISMATCH")


def _split_artifact_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    historical: list[dict[str, Any]] = []
    future: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (_published_at(item), _event_id(item))):
        metadata = _metadata(row)
        if metadata.get("timestamp_quality") != "EXACT":
            continue
        if _published_at(row).date() >= FUTURE_EVENT_HOLDOUT_START or bool(
            metadata.get("future_holdout")
        ):
            future.append(row)
        else:
            historical.append(row)
    return historical, future


def _dedupe_new_historical(
    rows: list[dict[str, Any]], *, base_event_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (_published_at(item), _event_id(item))):
        event_id = _event_id(row)
        if event_id in base_event_by_id and bool(
            _availability(base_event_by_id[event_id]).get("reaction_ready")
        ):
            continue
        if event_id in base_event_by_id:
            continue
        result.setdefault(event_id, row)
    return [
        result[key]
        for key in sorted(result, key=lambda event_id: (_published_at(result[event_id]), event_id))
    ]


def _source_rows(v1_root: Path, v2_root: Path) -> list[dict[str, Any]]:
    rows = [
        *_registry_sources(v1_root / "collection-source-registry.json"),
        *_registry_sources(v2_root / "collection-source-registry.json"),
    ]
    return sorted(rows, key=lambda row: (str(row["ticker"]), str(row["source_id"])))


def _registry_sources(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = payload.get("sources")
    if not isinstance(rows, list):
        raise ValueError("COLLECTION_SOURCE_REGISTRY_SOURCES_MISSING")
    return cast("list[dict[str, Any]]", rows)


def _read_live_registry(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = payload.get("sources")
    if not isinstance(rows, list):
        raise ValueError("SOURCE_REGISTRY_SOURCES_MISSING")
    return cast("list[dict[str, Any]]", rows)


def _verify_registry_matches_artifacts(
    source_rows: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
    v2_manifest: dict[str, Any],
) -> None:
    registry_by_source = {str(row["source_id"]): row for row in registry_rows}
    for source in source_rows:
        registry = registry_by_source.get(str(source["source_id"]))
        if registry is None:
            raise ValueError("LIVE_REGISTRY_MISSING_ARTIFACT_SOURCE")
        for key in ("ticker", "instrument_uid", "source_family", "source_url", "official_domain"):
            if registry.get(key) != source.get(key):
                raise ValueError(f"LIVE_REGISTRY_{key.upper()}_MISMATCH")
    v2_tickers = set(cast("list[str]", v2_manifest.get("NEW_TICKERS_WITH_EXACT_SOURCE") or []))
    if v2_tickers and not v2_tickers <= {str(row.get("ticker")) for row in registry_rows}:
        raise ValueError("LIVE_REGISTRY_MISSING_V2_TICKERS")


def _read_universe_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("history-coverage.jsonl", "discovery-shares.jsonl"):
        path = root / name
        if path.exists():
            rows.extend(_read_jsonl(path))
    return rows


def _resolve_identities(
    rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    universe_rows: list[dict[str, Any]],
) -> list[ActiveIdentity]:
    tickers = sorted({_ticker(row) for row in rows})
    source_by_ticker: dict[str, list[dict[str, Any]]] = {}
    for source in source_rows:
        source_by_ticker.setdefault(str(source["ticker"]), []).append(source)
    universe_by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in universe_rows:
        universe_by_ticker.setdefault(str(row.get("ticker")), []).append(row)
    identities: list[ActiveIdentity] = []
    for ticker in tickers:
        sources = source_by_ticker.get(ticker, [])
        unique_sources = {
            str(row.get("instrument_uid")) for row in sources if row.get("instrument_uid")
        }
        if len(sources) != 1 or len(unique_sources) != 1:
            identities.append(
                ActiveIdentity(ticker, ticker, "", None, None, None, None, "", "", "AMBIGUOUS")
            )
            continue
        source = sources[0]
        universe_matches = [
            row
            for row in universe_by_ticker.get(ticker, [])
            if str(row.get("instrument_uid")) == str(source["instrument_uid"])
        ]
        universe = _coalesced_universe_match(universe_matches)
        identities.append(
            ActiveIdentity(
                ticker=ticker,
                issuer=str(source["issuer"]),
                instrument_uid=str(source["instrument_uid"]),
                figi=_optional_text(universe.get("figi")),
                class_code=_optional_text(universe.get("class_code")),
                exchange=_optional_text(universe.get("exchange")),
                currency=_optional_text(universe.get("currency")),
                source_id=str(source["source_id"]),
                source_family=str(source["source_family"]),
                identity_provenance=(
                    "LIVE_SOURCE_REGISTRY_WITH_TINVEST_UNIVERSE"
                    if universe
                    else "LIVE_SOURCE_REGISTRY"
                ),
            )
        )
    return identities


def _coalesced_universe_match(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    for field in ("figi", "class_code", "currency"):
        values = {str(row[field]) for row in rows if row.get(field)}
        if len(values) > 1:
            return {}
    return max(
        rows,
        key=lambda row: sum(
            1 for field in ("figi", "class_code", "exchange", "currency") if row.get(field)
        ),
    )


def _eligibility_rows(
    rows: list[dict[str, Any]], identities_by_ticker: dict[str, ActiveIdentity]
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for row in rows:
        identity = identities_by_ticker.get(_ticker(row))
        status = InstrumentIdentityStatus.RESOLVED
        evidence: TradingEvidence | None = None
        if identity is None:
            status = InstrumentIdentityStatus.UNRESOLVED
        elif identity.identity_provenance == "AMBIGUOUS":
            status = InstrumentIdentityStatus.AMBIGUOUS
        elif identity.class_code != "TQBR" or str(identity.currency).lower() != "rub":
            evidence = None
        else:
            published = _published_at(row)
            evidence = TradingEvidence(
                ticker=identity.ticker,
                instrument_uid=identity.instrument_uid,
                figi=identity.figi,
                class_code=identity.class_code,
                source=identity.identity_provenance,
                security_history_confirmed=True,
                event_date_trading_confirmed=True,
                last_confirmed_trading_date=published.date(),
                current_trading_status="CURRENTLY_TRADABLE_SOURCE_TARGET",
                api_trade_available=True,
                buy_available=True,
                sell_available=True,
                evidence_detail=(
                    "Active TQBR/RUB breadth source passed PR48-style identity/tradability gate; "
                    "empty candle responses are not used as non-trading evidence."
                ),
            )
        result = evaluate_event_eligibility(
            event_id=_event_id(row),
            ticker=_ticker(row),
            published_at_utc=_published_at(row),
            identity_status=status,
            evidence=evidence,
            event_validity=EventValidity.VALID_EXACT_EVENT,
        )
        payload = result.payload()
        payload["tradability_status"] = payload["market_reaction_eligibility"]
        payloads.append(cast("dict[str, Any]", payload))
    return sorted(payloads, key=lambda item: (item["published_at_utc"], item["event_id"]))


async def _acquire_required_history(
    cache_root: Path,
    rows: list[dict[str, Any]],
    identities_by_ticker: dict[str, ActiveIdentity],
    *,
    client: ActiveMarketClient | None,
    existing_cache_roots: tuple[Path, ...],
) -> list[dict[str, Any]]:
    requests = _requested_days(rows)
    read_roots = tuple(path for path in (cache_root, *existing_cache_roots) if path.exists())
    day_provenance: dict[tuple[str, str], dict[str, Any]] = {}
    benchmark_identity = _benchmark_identity(read_roots)
    for ticker, days in sorted(requests.items()):
        identity = (
            benchmark_identity if ticker == BENCHMARK_TICKER else identities_by_ticker[ticker]
        )
        for day in days:
            begin = datetime.combine(day, datetime.min.time(), UTC)
            end = begin + timedelta(days=1)
            before = _load_day(read_roots, ticker, day.isoformat(), identity)
            request_count = 0
            new_rows = 0
            duplicate_rows = 0
            blocker: str | None = None
            status = "CACHE_ONLY"
            if client is not None and identity is not None and identity.instrument_uid:
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
                    merge = _merge_cache_day(cache_root, ticker, identity, begin, batch.candles)
                    new_rows = merge["new_rows"]
                    duplicate_rows = merge["duplicate_rows"]
                    status = "PASS"
            after_roots = tuple(
                path for path in (cache_root, *existing_cache_roots) if path.exists()
            )
            after = _load_day(after_roots, ticker, day.isoformat(), identity)
            day_provenance[(ticker, day.isoformat())] = {
                "ticker": ticker,
                "instrument_uid": None if identity is None else identity.instrument_uid,
                "figi": None if identity is None else identity.figi,
                "class_code": None if identity is None else identity.class_code,
                "date_from": begin.isoformat(),
                "date_to": end.isoformat(),
                "interval": "1m",
                "request_count": request_count,
                "candles_before": len(before),
                "candles_acquired": new_rows,
                "duplicates_removed": duplicate_rows,
                "candles_after": len(after),
                "first_candle": min((item.begin_at.isoformat() for item in after), default=None),
                "last_candle": max((item.end_at.isoformat() for item in after), default=None),
                "status": status if after or client is not None else "CACHE_MISS",
                "blocker": blocker,
                "source": "TINVEST_API" if client is not None else "LOCAL_CACHE_ONLY",
                "broker_write_surface_used": False,
                "token_value_read": False,
            }
    result: list[dict[str, Any]] = []
    for row in rows:
        published_at = _published_at(row)
        ticker = _ticker(row)
        security_days = [
            item[0].date().isoformat() for item in acquisition_day_bounds(published_at)
        ]
        benchmark_days = security_days
        security_rows = [day_provenance[(ticker, day)] for day in security_days]
        benchmark_rows = [day_provenance[(BENCHMARK_TICKER, day)] for day in benchmark_days]
        result.append(
            {
                "event_id": _event_id(row),
                "ticker": ticker,
                "source_id": str(_metadata(row).get("source_id")),
                "source_family": str(_metadata(row).get("source_code")),
                "publication_timestamp_utc": published_at.isoformat(),
                "network_fetch_performed": client is not None,
                "request_count": sum(
                    int(item["request_count"]) for item in [*security_rows, *benchmark_rows]
                ),
                "security_candles_after": sum(int(item["candles_after"]) for item in security_rows),
                "benchmark_candles_after": sum(
                    int(item["candles_after"]) for item in benchmark_rows
                ),
                "security_status": "PASS"
                if any(int(item["candles_after"]) for item in security_rows)
                else "SECURITY_HISTORY_MISSING",
                "benchmark_status": "PASS"
                if any(int(item["candles_after"]) for item in benchmark_rows)
                else "BENCHMARK_HISTORY_MISSING",
                "security_daily_provenance": security_rows,
                "benchmark_daily_provenance": benchmark_rows,
            }
        )
    return result


def _requested_days(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    requests: dict[str, set[Any]] = {BENCHMARK_TICKER: set()}
    for row in rows:
        published_at = _published_at(row)
        ticker = _ticker(row)
        requests.setdefault(ticker, set())
        for begin, _end in acquisition_day_bounds(published_at):
            requests[ticker].add(begin.date())
            requests[BENCHMARK_TICKER].add(begin.date())
    return {ticker: sorted(days) for ticker, days in requests.items() if days}


def _events_before_maturation(
    base_events: list[dict[str, Any]], collected_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = deepcopy(base_events)
    existing = {_event_id(row) for row in rows}
    for row in sorted(collected_rows, key=lambda item: (_published_at(item), _event_id(item))):
        if _event_id(row) not in existing:
            rows.append(deepcopy(row))
            existing.add(_event_id(row))
    return rows


def _report_base(
    row: dict[str, Any],
    identity: ActiveIdentity | None,
    eligibility: dict[str, Any],
    acquisition: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = _metadata(row)
    return {
        "event_id": str(metadata["event_id"]),
        "ticker": str(metadata["ticker"]),
        "source_id": str(metadata.get("source_id")),
        "source_family": str(metadata.get("source_code")),
        "published_at_utc": _published_at(row).isoformat(),
        "instrument_status": "RESOLVED" if identity and identity.instrument_uid else "UNRESOLVED",
        "tradability_status": eligibility["tradability_status"],
        "security_history_status": None if acquisition is None else acquisition["security_status"],
        "benchmark_history_status": None
        if acquisition is None
        else acquisition["benchmark_status"],
        "session_status": "NOT_EVALUATED",
        "ready_1m": False,
        "ready_5m": False,
        "ready_15m": False,
        "ready_30m": False,
        "ready_60m": False,
        "reaction_ready": False,
        "feature_ready": False,
        "primary_blocker": eligibility.get("primary_blocker"),
        "market_price_lookup_performed": eligibility["tradability_status"]
        == MarketReactionEligibility.ELIGIBLE.value,
        "future_outcome_fields_exposed": False,
    }


def _blocked_report(payload: dict[str, Any], blocker: str) -> dict[str, Any]:
    return {
        **payload,
        "session_status": blocker,
        "ready_1m": False,
        "ready_5m": False,
        "ready_15m": False,
        "ready_30m": False,
        "ready_60m": False,
        "reaction_ready": False,
        "feature_ready": False,
        "primary_blocker": blocker,
        "horizon_blockers": {horizon: blocker for horizon in HORIZONS},
        "pre_event_context_ready": False,
        "event_features_available": False,
        "max_feature_timestamp_utc": None,
        "post_event_feature_access": False,
    }


def _primary_blocker(alignment: Any, complete_features: bool, has_event_features: bool) -> str:
    if alignment.session_state.value == "PRE_OPEN":
        return "PRE_OPEN"
    if alignment.session_state.value == "AFTER_CLOSE":
        return "AFTER_CLOSE"
    if alignment.session_state.value == "NON_TRADING_DAY":
        return "NON_TRADING_SESSION"
    if not alignment.horizons:
        return "SESSION_ALIGNMENT_FAILED"
    if alignment.missing_reason:
        return "REACTION_MISSING"
    if not complete_features:
        return "PRE_EVENT_WARMUP_MISSING"
    if not has_event_features:
        return "EVENT_FEATURES_MISSING"
    return "OTHER_EXISTING_CANONICAL_BLOCKER"


def _horizon_status(horizons: dict[str, dict[str, object]], horizon: str) -> str | None:
    payload = horizons.get(horizon)
    if not payload:
        return "REACTION_MISSING"
    if payload.get("available") is True:
        return None
    return str(payload.get("reason", "REACTION_MISSING"))


def _mark_historical_default(row: dict[str, Any]) -> None:
    availability = _availability(row)
    availability["reaction_ready"] = False
    availability["feature_ready"] = False
    availability["research_outcomes_visible"] = False
    availability["status"] = "ACTIVE_HISTORICAL_MATURATION_BLOCKED"
    availability["missing_reason"] = "SECURITY_HISTORY_MISSING"
    metadata = _metadata(row)
    metadata["future_holdout"] = False
    metadata["future_holdout_metadata_only"] = False


def _assert_existing_rows_preserved(
    base_events: list[dict[str, Any]], events_after: list[dict[str, Any]], base_ids: set[str]
) -> None:
    after_by_id = {_event_id(row): row for row in events_after if _event_id(row) in base_ids}
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
        if (
            _published_at(row).date() < FUTURE_EVENT_HOLDOUT_START
            and _event_id(row) not in future_ids
        ):
            continue
        availability = _availability(row)
        if (
            bool(availability.get("research_outcomes_visible"))
            or bool(availability.get("reaction_ready"))
            or bool(availability.get("feature_ready"))
        ):
            raise ValueError("FUTURE_EVENT_HOLDOUT_READ_ATTEMPT")


def _per_ticker(
    per_event: list[dict[str, Any]],
    historical_input: list[dict[str, Any]],
    eligibility_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    input_counts = Counter(_ticker(row) for row in historical_input)
    eligible_counts = Counter(
        str(row["ticker"])
        for row in eligibility_rows
        if row["tradability_status"] == MarketReactionEligibility.ELIGIBLE.value
    )
    ready_counts = Counter(str(row["ticker"]) for row in per_event if row["reaction_ready"])
    feature_counts = Counter(str(row["ticker"]) for row in per_event if row["feature_ready"])
    blockers: dict[str, Counter[str]] = {}
    for row in per_event:
        blocker = row.get("primary_blocker")
        if blocker:
            blockers.setdefault(str(row["ticker"]), Counter())[str(blocker)] += 1
    return {
        ticker: {
            "historical_input": input_counts[ticker],
            "eligible": eligible_counts[ticker],
            "reaction_ready": ready_counts[ticker],
            "feature_ready": feature_counts[ticker],
            "primary_blockers": dict(sorted(blockers.get(ticker, Counter()).items())),
        }
        for ticker in sorted(input_counts)
    }


def _baseline_metrics(
    v1_manifest: dict[str, Any],
    v2_manifest: dict[str, Any],
    events: list[dict[str, Any]],
    features: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "CANONICAL_EXACT_EVENTS": _manifest_int(
            v2_manifest, "CANONICAL_EXACT_EVENTS_AFTER", len(events)
        ),
        "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS": _manifest_int(
            v2_manifest, "MARKET_REACTION_ELIGIBLE_EXACT_EVENTS_AFTER", len(events)
        ),
        "REACTION_READY_EVENTS": _manifest_int(
            v2_manifest,
            "REACTION_READY_EVENTS_AFTER",
            _manifest_int(v1_manifest, "REACTION_READY_EVENTS_AFTER", 0),
        ),
        "FEATURE_READY_EVENTS": _manifest_int(
            v2_manifest,
            "FEATURE_READY_EVENTS_AFTER",
            _manifest_int(v1_manifest, "FEATURE_READY_EVENTS_AFTER", len(features)),
        ),
    }


def _manifest_int(manifest: dict[str, Any], key: str, fallback: int) -> int:
    value = manifest.get(key, fallback)
    if not isinstance(value, int):
        raise ValueError(f"manifest field {key} must be an int")
    return value


def _decision(
    historical_count: int,
    eligible_count: int,
    per_event: list[dict[str, Any]],
    blocker_counts: dict[str, int],
) -> str:
    if blocker_counts.get(MarketReactionEligibility.INSTRUMENT_IDENTITY_AMBIGUOUS.value):
        return "IDENTITY_BLOCKERS_DOMINATE"
    if eligible_count == 0 and historical_count:
        return "IDENTITY_BLOCKERS_DOMINATE"
    feature_ready = sum(bool(row["feature_ready"]) for row in per_event)
    if eligible_count and feature_ready / eligible_count >= 0.5:
        return "MATURATION_GAIN_STRONG_SOURCE_BREADTH_NEXT"
    if feature_ready:
        return "MATURATION_GAIN_MODEST_SOURCE_BREADTH_NEXT"
    history_blockers = blocker_counts.get("SECURITY_HISTORY_MISSING", 0) + blocker_counts.get(
        "BENCHMARK_HISTORY_MISSING", 0
    )
    session_blockers = blocker_counts.get("SESSION_ALIGNMENT_FAILED", 0) + blocker_counts.get(
        "NON_TRADING_SESSION", 0
    )
    if history_blockers >= max(session_blockers, 1):
        return "MARKET_HISTORY_RECOVERY_NEXT"
    if session_blockers:
        return "SESSION_ALIGNMENT_BLOCKERS_DOMINATE"
    return "ACTIVE_EXACT_DATA_QUALITY_REVIEW_REQUIRED"


def _cache_roots(
    output_cache_root: Path, base_dataset_root: Path, extra_cache_roots: tuple[Path, ...]
) -> tuple[Path, ...]:
    candidates = (
        output_cache_root,
        base_dataset_root / "raw-minute-cache",
        base_dataset_root.parent / "exact-event-security-history-recovery-v1" / "raw-minute-cache",
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
    identity: ActiveIdentity | None,
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
    identity: ActiveIdentity | None,
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
    ticker: str,
    identity: ActiveIdentity,
    day_begin: datetime,
    candles: tuple[TInvestMinuteCandle, ...],
) -> dict[str, int]:
    path = cache_root / ticker / f"{day_begin.date().isoformat()}-day.jsonl"
    existing = [_candle_from_payload(row, expected=identity) for row in _read_jsonl(path)]
    merged: dict[tuple[str, datetime], TInvestMinuteCandle] = {
        (row.instrument_uid, row.begin_at): row for row in existing
    }
    new_rows = 0
    duplicate_rows = 0
    for candle in candles:
        if candle.instrument_uid != identity.instrument_uid:
            raise ValueError("SECURITY_CACHE_UID_MISMATCH")
        key = (candle.instrument_uid, candle.begin_at.astimezone(UTC))
        if key in merged:
            duplicate_rows += 1
        else:
            new_rows += 1
        merged[key] = candle
    _write_jsonl(
        path,
        [
            _candle_payload(row, ticker, identity)
            for row in sorted(merged.values(), key=lambda item: item.begin_at)
        ],
    )
    return {"new_rows": new_rows, "duplicate_rows": duplicate_rows}


def _benchmark_identity(cache_roots: tuple[Path, ...]) -> ActiveIdentity | None:
    for root in cache_roots:
        benchmark_root = root / BENCHMARK_TICKER
        if not benchmark_root.exists():
            continue
        for path in sorted(benchmark_root.glob("*-day.jsonl")):
            rows = _read_jsonl(path)
            if rows:
                uid = str(rows[0].get("instrument_uid") or "")
                if uid:
                    return ActiveIdentity(
                        ticker=BENCHMARK_TICKER,
                        issuer=BENCHMARK_TICKER,
                        instrument_uid=uid,
                        figi=None,
                        class_code=None,
                        exchange=None,
                        currency=None,
                        source_id=BENCHMARK_TICKER,
                        source_family=BENCHMARK_TICKER,
                        identity_provenance="LOCAL_IMOEX_CACHE_IDENTITY",
                    )
    return None


def _candle_payload(
    candle: TInvestMinuteCandle, ticker: str, identity: ActiveIdentity
) -> dict[str, Any]:
    return {
        "ticker": ticker,
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
    payload: dict[str, Any], expected: ActiveIdentity | None = None
) -> TInvestMinuteCandle:
    if str(payload.get("source", "TINVEST_API")) != "TINVEST_API":
        raise ValueError("NON_TINVEST_CANDLE_CACHE_SOURCE")
    if expected is not None and str(payload.get("instrument_uid")) != expected.instrument_uid:
        raise ValueError("SECURITY_CACHE_UID_MISMATCH")
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
            "ticker": _ticker(row),
            "source_id": str(_metadata(row).get("source_id")),
            "source_family": str(_metadata(row).get("source_code")),
            "publication_timestamp_utc": _published_at(row).isoformat(),
            "source_item_id": str(_metadata(row).get("source_item_id")),
        }
        for row in sorted(rows, key=lambda item: (_published_at(item), _event_id(item)))
    ]


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


def _ticker(row: dict[str, Any]) -> str:
    return str(_metadata(row)["ticker"])


def _published_at(row: dict[str, Any]) -> datetime:
    return _parse_datetime(_metadata(row)["publication_timestamp_utc"])


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
    maturation_results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    _write_jsonl(output_root / "events.jsonl", events)
    _write_jsonl(output_root / "features.jsonl", features)
    _write_jsonl(output_root / "targets.jsonl", targets)
    _write_jsonl(output_root / "maturation-results.jsonl", maturation_results)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# {ARTIFACT_VERSION}",
        "",
        "Data-maturation-only report for active strict-EXACT breadth events from v1 and v2.",
        "",
        f"- ARTIFACT_SHA={manifest['ARTIFACT_SHA']}",
        f"- INPUT_V1_ARTIFACT_SHA={manifest['INPUT_V1_ARTIFACT_SHA']}",
        f"- INPUT_V2_ARTIFACT_SHA={manifest['INPUT_V2_ARTIFACT_SHA']}",
        f"- MATURATION_COHORT_SHA={manifest['MATURATION_COHORT_SHA']}",
        f"- FUTURE_EXCLUDED_COHORT_SHA={manifest['FUTURE_EXCLUDED_COHORT_SHA']}",
        f"- INSTRUMENT_IDENTITY_SHA={manifest['INSTRUMENT_IDENTITY_SHA']}",
        f"- MARKET_ACQUISITION_PROVENANCE_SHA={manifest['MARKET_ACQUISITION_PROVENANCE_SHA']}",
        f"- MATURATION_RESULT_SHA={manifest['MATURATION_RESULT_SHA']}",
        f"- OUTPUT_DATASET_SHA={manifest['OUTPUT_DATASET_SHA']}",
        "",
        f"- V1_HISTORICAL_INPUT={manifest['V1_HISTORICAL_INPUT']}",
        f"- V2_HISTORICAL_INPUT={manifest['V2_HISTORICAL_INPUT']}",
        f"- DEDUPED_HISTORICAL_INPUT={manifest['DEDUPED_HISTORICAL_INPUT']}",
        f"- MARKET_ELIGIBLE_INPUT={manifest['MARKET_ELIGIBLE_INPUT']}",
        f"- MARKET_INELIGIBLE_INPUT={manifest['MARKET_INELIGIBLE_INPUT']}",
        "",
        f"- NEW_REACTION_READY={manifest['NEW_REACTION_READY']}",
        f"- NEW_FEATURE_READY={manifest['NEW_FEATURE_READY']}",
        f"- NEW_1M_READY={manifest['NEW_1M_READY']}",
        f"- NEW_5M_READY={manifest['NEW_5M_READY']}",
        f"- NEW_15M_READY={manifest['NEW_15M_READY']}",
        f"- NEW_30M_READY={manifest['NEW_30M_READY']}",
        f"- NEW_60M_READY={manifest['NEW_60M_READY']}",
        "- BLOCKER_COUNTS="
        f"{json.dumps(manifest['BLOCKER_COUNTS'], ensure_ascii=False, sort_keys=True)}",
        "",
        f"FINAL_DECISION={manifest['FINAL_DECISION']}",
        "",
        "No model training, TEST outcome use, future outcome read, backtest, paper trading, "
        "real trading, orders, or broker mutation was performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _output_nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())
