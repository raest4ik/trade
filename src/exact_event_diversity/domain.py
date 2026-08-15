from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime
from statistics import median
from typing import cast

DATASET_VERSION = "exact-event-market-dataset-v2"
SOURCE_REGISTRY_VERSION = "exact-event-source-registry-v2"
PARSER_VERSION = "official-structured-exact-v2"

FROZEN_V1_HASHES = {
    "exact_dataset_sha": "6de46b85a5d62975a94250e3e12de068343b74af9647c50b5bb225ed386d4be8",
    "source_registry_sha": "21aca618a8bb71694ef77ceff533ce3652c626d6fd3b8dae9fb2fc1e60fbeb80",
    "provenance_sha": "fa0416889ce3588997161639f6279653956cc09814a5fab47fa9fa9ee175c3f4",
    "timestamp_manifest_sha": "3877b4ae8b4c4e51a4cf1369c5f05afa68149a32cf5e6dae5d6ae4c2be2a4fcd",
    "cluster_manifest_sha": "a86689a52ef4e4da0645916e94a6c2c2721f740e9c795c33045813326b3c0bcf",
    "reaction_manifest_sha": "d6dd0deb1c5ef3ad7785f27ddff615b1e37251507571b3becf523161fb095394",
}
FROZEN_V1_COUNTS = {
    "exact_timestamp_events": 449,
    "exact_reaction_ready": 342,
    "exact_feature_ready": 239,
    "exact_unique_tickers": 4,
    "exact_unique_issuers": 4,
}


def parse_explicit_utc(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("TIMESTAMP_TIMEZONE_UNRESOLVED")
    return parsed.astimezone(UTC)


def concentration(counts: Counter[str]) -> dict[str, object]:
    total = sum(counts.values())
    ordered = sorted(counts.values(), reverse=True)
    shares = [count / total for count in ordered] if total else []
    hhi = sum(share**2 for share in shares)
    values = sorted(counts.values())
    return {
        "counts": dict(sorted(counts.items())),
        "top_share": shares[0] if shares else 0.0,
        "top_3_share": sum(shares[:3]),
        "hhi": hhi,
        "effective_count": 1 / hhi if hhi else 0.0,
        "median": median(values) if values else 0,
        "p10": _nearest_rank(values, 0.1),
        "p90": _nearest_rank(values, 0.9),
    }


def feature_ready_gap(event_rows: list[dict[str, object]]) -> dict[str, object]:
    reasons: Counter[str] = Counter()
    gap = 0
    for row in event_rows:
        availability = cast("dict[str, object]", row["target_availability"])
        if not availability.get("reaction_ready") or availability.get("feature_ready"):
            continue
        gap += 1
        event_features = cast("dict[str, object]", row.get("event_features", {}))
        market = cast("dict[str, object]", row.get("pre_event_market_features", {}))
        metadata = cast("dict[str, object]", row.get("metadata", {}))
        if not event_features:
            reason = "missing_event_features"
        elif not metadata.get("instrument_uid"):
            reason = "ticker_mapping"
        elif not market:
            reason = "missing_pre_event_market_context"
        elif any(
            value is None
            for key, value in market.items()
            if key.startswith(("pre_return_", "imoex_pre_return_"))
        ):
            reason = "market_history_warmup"
        elif metadata.get("event_cluster_id") is None:
            reason = "cluster_exclusion"
        elif not metadata.get("storage_policy"):
            reason = "source_policy_issue"
        else:
            reason = "other"
        reasons[reason] += 1
    if sum(reasons.values()) != gap:
        raise ValueError("FEATURE_READY_REASON_RECONCILIATION_FAILED")
    for name in (
        "missing_event_features",
        "missing_pre_event_market_context",
        "market_history_warmup",
        "ticker_mapping",
        "cluster_exclusion",
        "source_policy_issue",
        "other",
    ):
        reasons.setdefault(name, 0)
    return {"count": gap, "reasons": dict(sorted(reasons.items()))}


def exact_model_data_status(*, feature_ready: int, feature_ready_by_ticker: Counter[str]) -> str:
    eligible_tickers = sum(count >= 10 for count in feature_ready_by_ticker.values())
    if feature_ready >= 100 and len(feature_ready_by_ticker) >= 10 and eligible_tickers >= 3:
        return "READY_FOR_EXACT_BASELINE_EXPERIMENT"
    return "NOT_READY_FOR_EXACT_MODEL"


def _nearest_rank(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    index = max(0, min(len(values) - 1, math.ceil(quantile * len(values)) - 1))
    return values[index]
