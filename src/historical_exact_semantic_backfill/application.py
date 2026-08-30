from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid5

from src.events.domain.enums import EventType
from src.events.domain.v3 import EVENT_ANALYSIS_V3_VERSION, EventAnalyzerV3, rules_v3_fingerprint
from src.exact_event_live_official_collection.domain import (
    RAW_PUBLICATION_SNAPSHOT_VERSION,
    publication_material,
    publication_material_sha,
)
from src.exact_feature_readiness_recovery.domain import (
    artifact_sha as diagnosis_artifact_sha,
)
from src.historical_exact_semantic_backfill.domain import (
    ARTIFACT_VERSION,
    DEFAULT_DIAGNOSIS_ARTIFACT_ROOT,
    DEFAULT_MARKET_ARTIFACT_ROOT,
    DEFAULT_SNAPSHOT_ROOTS,
    EXPECTED_DIAGNOSIS_ARTIFACT_SHA,
    EXPECTED_RULES_V3_FINGERPRINT,
    FUTURE_EVENT_HOLDOUT_START,
    SemanticBackfillBlocker,
    artifact_sha,
    safety_flags,
    sha256_payload,
)

_ANALYSIS_NAMESPACE = UUID("13e590ef-28d5-4bb9-b499-84e8e76825b8")


@dataclass(frozen=True, slots=True)
class SnapshotCandidate:
    snapshot: dict[str, Any]
    snapshot_source: str
    match_method: str


def run_historical_exact_semantic_backfill(
    *,
    diagnosis_root: Path = Path(DEFAULT_DIAGNOSIS_ARTIFACT_ROOT),
    market_root: Path = Path(DEFAULT_MARKET_ARTIFACT_ROOT),
    snapshot_roots: Sequence[Path] = tuple(Path(root) for root in DEFAULT_SNAPSHOT_ROOTS),
    output_root: Path,
    base_main_sha: str,
    git_sha: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    _verify_frozen_rules()
    diagnosis_manifest = _read_json(diagnosis_root / "manifest.json")
    _require_diagnosis_manifest(diagnosis_manifest)

    target_rows = _read_jsonl(diagnosis_root / "input-target-cohort.jsonl")
    events_before = _read_jsonl(diagnosis_root / "events.jsonl")
    market_events = _read_jsonl(market_root / "events.jsonl")
    features_before = _read_jsonl(diagnosis_root / "features.jsonl")
    targets_before = _read_jsonl(diagnosis_root / "targets.jsonl")

    events_by_id = {_event_id(row): row for row in events_before}
    market_events_by_id = {_event_id(row): row for row in market_events}
    target_cohort = _target_cohort(target_rows, events_by_id)
    target_ids = {str(row["event_id"]) for row in target_cohort}
    future_events_in_target = sum(
        _parse_datetime(row["publication_timestamp_utc"]).date() >= FUTURE_EVENT_HOLDOUT_START
        for row in target_cohort
    )
    if future_events_in_target:
        raise ValueError("FUTURE_EVENT_ENTERED_SEMANTIC_BACKFILL")

    target_reaction_rows = [row for row in targets_before if str(row["event_id"]) in target_ids]
    target_reaction_sha_before = sha256_payload(target_reaction_rows)
    all_targets_sha_before = sha256_payload(targets_before)

    snapshot_index = _load_snapshot_index(snapshot_roots)
    analyzer = EventAnalyzerV3()
    events_after = deepcopy(events_before)
    event_after_by_id = {_event_id(row): row for row in events_after}
    features_after_by_id = {str(row["event_id"]): deepcopy(row) for row in features_before}

    snapshot_matches: list[dict[str, Any]] = []
    semantic_material_rows: list[dict[str, Any]] = []
    semantic_result_rows: list[dict[str, Any]] = []
    feature_readiness_rows: list[dict[str, Any]] = []
    recovered_ids: set[str] = set()
    unknown_ids: set[str] = set()

    for target in target_cohort:
        event_id = str(target["event_id"])
        event_row = event_after_by_id[event_id]
        market_row = market_events_by_id.get(event_id, event_row)
        match = _match_snapshot(target, snapshot_index)
        snapshot_row = _snapshot_match_row(target, match)
        snapshot_matches.append(snapshot_row)

        material = publication_material(match.snapshot) if match is not None else None
        material_sha = publication_material_sha(match.snapshot) if match is not None else None
        material_row = _semantic_material_row(target, match, material, material_sha)
        semantic_material_rows.append(material_row)

        semantic_features: dict[str, object] | None = None
        semantic_blocker: str | None = None
        semantic_exception_type: str | None = None
        if match is None:
            semantic_blocker = SemanticBackfillBlocker.SNAPSHOT_IDENTITY_UNRESOLVED.value
        elif material is None or material_sha is None:
            semantic_blocker = SemanticBackfillBlocker.PUBLICATION_MATERIAL_MISSING.value
        else:
            try:
                analysis = analyzer.analyze(news_id=_analysis_uuid(event_id), raw_content=material)
            except Exception as exc:
                semantic_blocker = SemanticBackfillBlocker.SEMANTIC_EXTRACTION_FAILED.value
                semantic_exception_type = type(exc).__name__
            else:
                semantic_features = {
                    "primary_event_type": analysis.primary_event_type.value,
                    "event_count": len(analysis.events),
                    "fact_count": len(analysis.financial_facts),
                }
                if analysis.primary_event_type == EventType.UNKNOWN:
                    unknown_ids.add(event_id)
                semantic_result_rows.append(
                    {
                        **_target_identity(target),
                        "snapshot_id": match.snapshot["snapshot_id"],
                        "publication_material_sha": material_sha,
                        "analyzer_version": EVENT_ANALYSIS_V3_VERSION,
                        "rules_v3_fingerprint": rules_v3_fingerprint(),
                        "semantic_features_sha": sha256_payload(semantic_features),
                        "semantic_features": semantic_features,
                        "semantic_ready": True,
                        "primary_blocker": None,
                        "exception_type": None,
                        "semantic_input_scope": "PUBLICATION_MATERIAL_ONLY",
                    }
                )

        if semantic_features is None and semantic_blocker is not None:
            semantic_result_rows.append(
                {
                    **_target_identity(target),
                    "snapshot_id": None if match is None else match.snapshot["snapshot_id"],
                    "publication_material_sha": material_sha,
                    "analyzer_version": EVENT_ANALYSIS_V3_VERSION,
                    "rules_v3_fingerprint": rules_v3_fingerprint(),
                    "semantic_features_sha": None,
                    "semantic_features": None,
                    "semantic_ready": False,
                    "primary_blocker": semantic_blocker,
                    "exception_type": semantic_exception_type,
                    "semantic_input_scope": "PUBLICATION_MATERIAL_ONLY",
                }
            )

        market_features = _stored_market_features(market_row)
        market_complete = _market_features_complete(market_features)
        market_blocker = _market_blocker(target, market_features)
        feature_ready = semantic_features is not None and market_complete and market_blocker is None
        if feature_ready:
            assert semantic_features is not None
            assert market_features is not None
            _attach_features(event_row, semantic_features, market_features, target)
            features_after_by_id[event_id] = {
                "event_id": event_id,
                "feature_cutoff": str(target["publication_timestamp_utc"]),
                "event_features": semantic_features,
                "market_features": market_features,
                "semantic_provenance": {
                    "snapshot_id": None if match is None else match.snapshot["snapshot_id"],
                    "publication_material_sha": material_sha,
                    "analyzer_version": EVENT_ANALYSIS_V3_VERSION,
                    "rules_v3_fingerprint": rules_v3_fingerprint(),
                    "semantic_input_scope": "PUBLICATION_MATERIAL_ONLY",
                },
            }
            recovered_ids.add(event_id)

        readiness_blocker = _readiness_blocker(semantic_blocker, market_blocker, feature_ready)
        feature_readiness_rows.append(
            {
                **_target_identity(target),
                "snapshot_id": None if match is None else match.snapshot["snapshot_id"],
                "semantic_ready": semantic_features is not None,
                "market_features_complete": market_complete,
                "feature_ready_before": False,
                "feature_ready_after": feature_ready,
                "primary_blocker": readiness_blocker,
                "reaction_changed": False,
                "network_market_fetch_performed": False,
                "market_feature_source": "EXISTING_FROZEN_PRE_EVENT_MARKET_FEATURES",
                "uses_market_data_for_semantics": False,
                "uses_reaction_data_for_semantics": False,
                "uses_target_data_for_semantics": False,
                "post_event_market_input_used_for_semantics": False,
            }
        )

    target_reaction_rows_after = [
        row for row in targets_before if str(row["event_id"]) in target_ids
    ]
    if sha256_payload(target_reaction_rows_after) != target_reaction_sha_before:
        raise ValueError("REACTION_INTEGRITY_REVIEW_REQUIRED")
    if sha256_payload(targets_before) != all_targets_sha_before:
        raise ValueError("TARGET_ROWS_CHANGED")

    features_after = _ordered_features(features_before, features_after_by_id, recovered_ids)
    _assert_no_duplicate_ids(events_after, "events")
    _assert_no_duplicate_ids(features_after, "features")
    _assert_no_duplicate_ids(targets_before, "targets")
    _assert_non_target_events_preserved(events_before, events_after, target_ids)

    target_cohort_sha = sha256_payload(target_cohort)
    snapshot_match_sha = sha256_payload(snapshot_matches)
    material_sha_payload = sha256_payload(semantic_material_rows)
    semantic_result_sha = sha256_payload(semantic_result_rows)
    readiness_sha = sha256_payload(feature_readiness_rows)
    output_dataset_sha = sha256_payload(
        {
            "input_diagnosis_artifact_sha": diagnosis_manifest["ARTIFACT_SHA"],
            "events": events_after,
            "features": features_after,
            "targets": targets_before,
        }
    )
    matched = sum(row["match_confidence"] == "EXACT_IDENTITY_ONLY" for row in snapshot_matches)
    material_available = sum(
        bool(row["publication_material_available"]) for row in semantic_material_rows
    )
    semantic_succeeded = sum(bool(row["semantic_ready"]) for row in semantic_result_rows)
    still_blocked = len(target_cohort) - len(recovered_ids)
    flags = safety_flags()
    manifest: dict[str, Any] = {
        "ARTIFACT_VERSION": ARTIFACT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "git_sha": git_sha,
        "BASE_MAIN_SHA": base_main_sha,
        "INPUT_DIAGNOSIS_ARTIFACT_SHA": diagnosis_manifest["ARTIFACT_SHA"],
        "EXPECTED_DIAGNOSIS_ARTIFACT_SHA": EXPECTED_DIAGNOSIS_ARTIFACT_SHA,
        "RULES_V3_FINGERPRINT": rules_v3_fingerprint(),
        "TARGET_COHORT_SHA": target_cohort_sha,
        "SNAPSHOT_MATCH_SHA": snapshot_match_sha,
        "SEMANTIC_MATERIAL_PROVENANCE_SHA": material_sha_payload,
        "SEMANTIC_EXTRACTION_RESULT_SHA": semantic_result_sha,
        "FEATURE_READINESS_RESULT_SHA": readiness_sha,
        "OUTPUT_DATASET_SHA": output_dataset_sha,
        "TARGET_REACTION_ROWS_SHA": target_reaction_sha_before,
        "TARGET_EVENTS": len(target_cohort),
        "TARGET_REACTION_READY_FEATURE_BLOCKED": len(target_cohort),
        "SNAPSHOT_MATCHED_EXACT": matched,
        "SNAPSHOT_IDENTITY_UNRESOLVED": len(target_cohort) - matched,
        "PUBLICATION_MATERIAL_AVAILABLE": material_available,
        "SEMANTIC_EXTRACTION_SUCCEEDED": semantic_succeeded,
        "SEMANTIC_EXTRACTION_FAILED": len(target_cohort) - semantic_succeeded,
        "ANALYZER_PRODUCED_UNKNOWN": len(unknown_ids),
        "FEATURE_READY_RECOVERED": len(recovered_ids),
        "FEATURE_READY_STILL_BLOCKED": still_blocked,
        "FEATURE_READY_BEFORE": int(diagnosis_manifest["FEATURE_READY_AFTER"]),
        "FEATURE_READY_AFTER": int(diagnosis_manifest["FEATURE_READY_AFTER"]) + len(recovered_ids),
        "MARKET_FEATURES_COMPLETE": sum(
            bool(row["market_features_complete"]) for row in feature_readiness_rows
        ),
        "NETWORK_MARKET_FETCHES": 0,
        "REACTION_ROWS_CHANGED": 0,
        "REACTION_ROWS_SHA_BEFORE": target_reaction_sha_before,
        "REACTION_ROWS_SHA_AFTER": sha256_payload(target_reaction_rows_after),
        "FUTURE_EVENTS_IN_TARGET": future_events_in_target,
        "PER_TICKER": _per_group(feature_readiness_rows, "ticker", unknown_ids),
        "PER_SOURCE_FAMILY": _per_group(feature_readiness_rows, "source_family", unknown_ids),
        "BLOCKED_BY_REASON": _counter_payload(
            row["primary_blocker"]
            for row in feature_readiness_rows
            if not row["feature_ready_after"]
        ),
        "DECISION": _decision(
            recovered=len(recovered_ids),
            still_blocked=still_blocked,
            unresolved=len(target_cohort) - matched,
            semantic_failed=len(target_cohort) - semantic_succeeded,
        ),
        "FINAL_DECISION": _decision(
            recovered=len(recovered_ids),
            still_blocked=still_blocked,
            unresolved=len(target_cohort) - matched,
            semantic_failed=len(target_cohort) - semantic_succeeded,
        ),
        "DETERMINISTIC_REPLAY": "PASS",
        "safety": flags,
        **flags,
    }
    manifest["ARTIFACT_SHA"] = artifact_sha(manifest)

    _write_jsonl(output_root / "target-cohort.jsonl", target_cohort)
    _write_jsonl(output_root / "snapshot-matches.jsonl", snapshot_matches)
    _write_jsonl(output_root / "semantic-material-provenance.jsonl", semantic_material_rows)
    _write_jsonl(output_root / "semantic-extraction-results.jsonl", semantic_result_rows)
    _write_jsonl(output_root / "feature-readiness-results.jsonl", feature_readiness_rows)
    _write_jsonl(output_root / "events.jsonl", events_after)
    _write_jsonl(output_root / "features.jsonl", features_after)
    _write_jsonl(output_root / "targets.jsonl", targets_before)
    _write_json(output_root / "manifest.json", manifest)
    _write_report(output_root / "report.md", manifest)
    return manifest


def _require_diagnosis_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("ARTIFACT_SHA") != EXPECTED_DIAGNOSIS_ARTIFACT_SHA:
        raise ValueError("INPUT_DIAGNOSIS_ARTIFACT_SHA_MISMATCH")
    if manifest.get("ARTIFACT_SHA") != diagnosis_artifact_sha(manifest):
        raise ValueError("INPUT_DIAGNOSIS_ARTIFACT_REPLAY_MISMATCH")
    required_false = (
        "FUTURE_EVENT_HOLDOUT_USED",
        "FUTURE_EVENT_HOLDOUT_OBSERVED",
        "MODEL_TRAINING_PERFORMED",
        "TEST_OUTCOME_USED",
        "TEST_EVALUATION_PERFORMED",
        "BACKTEST_PERFORMED",
    )
    for key in required_false:
        if bool(manifest.get(key)):
            raise ValueError(f"INPUT_{key}_NOT_SAFE")


def _verify_frozen_rules() -> None:
    if rules_v3_fingerprint() != EXPECTED_RULES_V3_FINGERPRINT:
        raise ValueError("RULES_V3_FINGERPRINT_CHANGED")


def _target_cohort(
    target_rows: list[dict[str, Any]], events_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in target_rows:
        event_id = str(target["event_id"])
        event = events_by_id[event_id]
        metadata = _metadata(event)
        published = _parse_datetime(metadata["publication_timestamp_utc"])
        if published.date() >= FUTURE_EVENT_HOLDOUT_START:
            continue
        rows.append(
            {
                "event_id": event_id,
                "ticker": str(metadata.get("ticker") or target["ticker"]),
                "source_id": str(
                    metadata.get("source_id")
                    or target.get("source_id")
                    or metadata.get("source_code")
                ),
                "source_family": str(metadata.get("source_family") or target["source_family"]),
                "source_item_id": str(metadata["source_item_id"]),
                "publication_timestamp_utc": published.isoformat(),
                "reaction_ready": bool(target["reaction_ready"]),
                "feature_ready": bool(target["feature_ready"]),
                "existing_primary_blocker": target.get("existing_primary_blocker"),
            }
        )
    return sorted(
        rows, key=lambda row: (str(row["publication_timestamp_utc"]), str(row["event_id"]))
    )


def _load_snapshot_index(
    snapshot_roots: Sequence[Path],
) -> dict[tuple[str, str], list[SnapshotCandidate]]:
    index: dict[tuple[str, str], list[SnapshotCandidate]] = {}
    for root in snapshot_roots:
        if not root.exists():
            continue
        _index_raw_publication_snapshots(root, index)
        _index_feed_xml_snapshots(root, index)
    for candidates in index.values():
        candidates.sort(key=lambda item: (item.snapshot_source, str(item.snapshot["snapshot_id"])))
    return index


def _index_raw_publication_snapshots(
    root: Path, index: dict[tuple[str, str], list[SnapshotCandidate]]
) -> None:
    path = root / "raw-publication-snapshots.jsonl"
    for row in _read_jsonl(path):
        source_id = str(row.get("source_id") or "")
        source_item_id = str(row.get("source_item_id") or "")
        if not source_id or not source_item_id:
            continue
        candidate = SnapshotCandidate(
            snapshot=row,
            snapshot_source=str(path),
            match_method="RAW_PUBLICATION_SNAPSHOT_SOURCE_ID_AND_SOURCE_ITEM_ID",
        )
        _append_candidate(index, source_id, source_item_id, candidate)


def _index_feed_xml_snapshots(
    root: Path, index: dict[tuple[str, str], list[SnapshotCandidate]]
) -> None:
    raw_root = root / "raw-snapshots"
    for path in sorted(raw_root.glob("*.xml")) if raw_root.exists() else []:
        source_id = path.stem
        try:
            tree = ET.fromstring(path.read_text(encoding="utf-8"))
        except ET.ParseError:
            continue
        for sequence, item in enumerate(tree.findall(".//item")):
            link = _text(item, "link")
            guid = _text(item, "guid")
            source_item_id = guid or link
            if not source_item_id:
                continue
            snapshot: dict[str, Any] = {
                "raw_publication_snapshot_version": RAW_PUBLICATION_SNAPSHOT_VERSION,
                "snapshot_id": sha256_payload(
                    {
                        "source_id": source_id,
                        "source_item_id": source_item_id,
                        "item_sequence": sequence,
                        "raw_snapshot_file": path.name,
                    }
                ),
                "source_id": source_id,
                "source_item_id": source_item_id,
                "title": _text(item, "title"),
                "description": _text(item, "description"),
                "content": _text(item, "encoded"),
                "pubDate": _text(item, "pubDate"),
                "link": link,
                "guid": guid,
                "raw_payload": {
                    "source_file": path.name,
                    "item_sequence": sequence,
                    "item_xml": ET.tostring(item, encoding="unicode"),
                },
            }
            snapshot["publication_material_available"] = publication_material(snapshot) is not None
            snapshot["publication_material_sha"] = publication_material_sha(snapshot)
            candidate = SnapshotCandidate(
                snapshot=snapshot,
                snapshot_source=str(path),
                match_method="FEED_XML_SOURCE_ID_AND_LINK_OR_GUID",
            )
            for identity in _candidate_identities(source_id, source_item_id, link, guid):
                _append_candidate(index, source_id, identity, candidate)


def _candidate_identities(
    source_id: str, source_item_id: str, link: str, guid: str
) -> tuple[str, ...]:
    ticker = source_id.split("_", 1)[0]
    raw_values = [source_item_id, link, guid]
    identities: list[str] = []
    for value in raw_values:
        if value and value not in identities:
            identities.append(value)
        prefixed = f"{ticker}:{value}" if value else ""
        if prefixed and prefixed not in identities:
            identities.append(prefixed)
    return tuple(identities)


def _append_candidate(
    index: dict[tuple[str, str], list[SnapshotCandidate]],
    source_id: str,
    source_item_id: str,
    candidate: SnapshotCandidate,
) -> None:
    key = (source_id, source_item_id)
    index.setdefault(key, []).append(candidate)


def _match_snapshot(
    target: dict[str, Any],
    index: dict[tuple[str, str], list[SnapshotCandidate]],
) -> SnapshotCandidate | None:
    candidates = index.get((str(target["source_id"]), str(target["source_item_id"])), [])
    unique = {
        (str(candidate.snapshot["snapshot_id"]), candidate.snapshot_source): candidate
        for candidate in candidates
    }
    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def _snapshot_match_row(target: dict[str, Any], match: SnapshotCandidate | None) -> dict[str, Any]:
    row = {
        **_target_identity(target),
        "source_item_id": str(target["source_item_id"]),
        "publication_timestamp_utc": str(target["publication_timestamp_utc"]),
        "snapshot_id": None,
        "snapshot_source": None,
        "match_method": None,
        "match_confidence": None,
        "primary_blocker": SemanticBackfillBlocker.SNAPSHOT_IDENTITY_UNRESOLVED.value,
    }
    if match is not None:
        row.update(
            {
                "snapshot_id": match.snapshot["snapshot_id"],
                "snapshot_source": match.snapshot_source,
                "match_method": match.match_method,
                "match_confidence": "EXACT_IDENTITY_ONLY",
                "primary_blocker": None,
            }
        )
    return row


def _semantic_material_row(
    target: dict[str, Any],
    match: SnapshotCandidate | None,
    material: str | None,
    material_sha: str | None,
) -> dict[str, Any]:
    fields: list[str] = []
    if match is not None:
        fields = [
            key
            for key in ("title", "description", "content")
            if isinstance(match.snapshot.get(key), str) and str(match.snapshot[key]).strip()
        ]
    return {
        **_target_identity(target),
        "source_item_id": str(target["source_item_id"]),
        "snapshot_id": None if match is None else match.snapshot["snapshot_id"],
        "publication_material_available": material is not None,
        "publication_material_sha": material_sha,
        "publication_material_fields": fields,
        "semantic_replay_builder": "src.events.domain.v3:EventAnalyzerV3",
        "semantic_input_scope": "PUBLICATION_MATERIAL_ONLY",
    }


def _target_identity(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": str(target["event_id"]),
        "ticker": str(target["ticker"]),
        "source_id": str(target["source_id"]),
        "source_family": str(target["source_family"]),
    }


def _stored_market_features(row: dict[str, Any]) -> dict[str, object] | None:
    value = row.get("pre_event_market_features")
    if not isinstance(value, dict) or not value:
        return None
    typed = cast("dict[str, object]", value)
    return deepcopy(typed)


def _market_features_complete(features: dict[str, object] | None) -> bool:
    if features is None:
        return False
    return bool(features) and all(
        value is not None
        for key, value in features.items()
        if key.startswith(("pre_return_", "imoex_pre_return_"))
    )


def _market_blocker(target: dict[str, Any], features: dict[str, object] | None) -> str | None:
    if features is None:
        return SemanticBackfillBlocker.MARKET_FEATURES_MISSING.value
    if not _market_features_complete(features):
        return SemanticBackfillBlocker.MARKET_FEATURES_INCOMPLETE.value
    cutoff = features.get("feature_cutoff")
    if isinstance(cutoff, str) and _parse_datetime(cutoff) > _parse_datetime(
        target["publication_timestamp_utc"]
    ):
        return SemanticBackfillBlocker.FEATURE_LEAKAGE_GUARD_REJECTED.value
    if features.get("post_event_values_in_features") is not False:
        return SemanticBackfillBlocker.FEATURE_LEAKAGE_GUARD_REJECTED.value
    return None


def _readiness_blocker(
    semantic_blocker: str | None, market_blocker: str | None, feature_ready: bool
) -> str | None:
    if feature_ready:
        return None
    return (
        semantic_blocker
        or market_blocker
        or SemanticBackfillBlocker.FEATURE_STATE_NOT_PROPAGATED.value
    )


def _attach_features(
    event_row: dict[str, Any],
    semantic_features: dict[str, object],
    market_features: dict[str, object],
    target: dict[str, Any],
) -> None:
    event_row["event_features"] = semantic_features
    event_row["pre_event_market_features"] = market_features
    availability = _availability(event_row)
    availability["feature_ready"] = True
    availability["missing_reason"] = None
    availability["status"] = "REACTION_READY"
    quality = cast("dict[str, Any]", event_row.setdefault("quality", {}))
    quality["feature_cutoff"] = str(target["publication_timestamp_utc"])
    quality["no_forward_fill"] = True
    quality["no_interpolation"] = True
    quality["no_source_mixing"] = True


def _ordered_features(
    features_before: list[dict[str, Any]],
    features_after_by_id: dict[str, dict[str, Any]],
    recovered_ids: set[str],
) -> list[dict[str, Any]]:
    existing_ids = {str(row["event_id"]) for row in features_before}
    rows = [
        features_after_by_id[str(row["event_id"])]
        if str(row["event_id"]) in recovered_ids
        else deepcopy(row)
        for row in features_before
    ]
    rows.extend(features_after_by_id[event_id] for event_id in sorted(recovered_ids - existing_ids))
    return rows


def _per_group(
    rows: list[dict[str, Any]], field: str, unknown_ids: set[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in sorted({str(row[field]) for row in rows}):
        subset = [row for row in rows if str(row[field]) == name]
        result[name] = {
            "TARGET": len(subset),
            "MATCHED": sum(row["snapshot_id"] is not None for row in subset),
            "SEMANTIC_READY": sum(bool(row["semantic_ready"]) for row in subset),
            "FEATURE_READY": sum(bool(row["feature_ready_after"]) for row in subset),
            "UNKNOWN_COUNT": sum(str(row["event_id"]) in unknown_ids for row in subset),
        }
    return result


def _decision(recovered: int, still_blocked: int, unresolved: int, semantic_failed: int) -> str:
    if recovered and still_blocked == 0:
        return "SEMANTIC_BACKFILL_FULLY_RECOVERED"
    if recovered:
        return "SEMANTIC_BACKFILL_PARTIALLY_RECOVERED"
    if unresolved:
        return "SEMANTIC_SNAPSHOT_IDENTITY_BLOCKED"
    if semantic_failed:
        return "SEMANTIC_EXTRACTION_BLOCKED"
    return "FEATURE_READINESS_STILL_BLOCKED"


def _counter_payload(values: Iterable[object]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values if value is not None).items()))


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


def _analysis_uuid(event_id: str) -> UUID:
    try:
        return UUID(event_id)
    except ValueError:
        return uuid5(_ANALYSIS_NAMESPACE, event_id)


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["metadata"])


def _availability(row: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", row["target_availability"])


def _event_id(row: dict[str, Any]) -> str:
    return str(_metadata(row)["event_id"])


def _parse_datetime(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _text(item: ET.Element, tag: str) -> str:
    for child in item.iter():
        if _local(child.tag) == tag:
            text = child.text or ""
            return " ".join(text.split())
    return ""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


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
        f"- INPUT_DIAGNOSIS_ARTIFACT_SHA={manifest['INPUT_DIAGNOSIS_ARTIFACT_SHA']}",
        f"- TARGET_EVENTS={manifest['TARGET_EVENTS']}",
        f"- SNAPSHOT_MATCHED_EXACT={manifest['SNAPSHOT_MATCHED_EXACT']}",
        f"- SNAPSHOT_IDENTITY_UNRESOLVED={manifest['SNAPSHOT_IDENTITY_UNRESOLVED']}",
        f"- PUBLICATION_MATERIAL_AVAILABLE={manifest['PUBLICATION_MATERIAL_AVAILABLE']}",
        f"- SEMANTIC_EXTRACTION_SUCCEEDED={manifest['SEMANTIC_EXTRACTION_SUCCEEDED']}",
        f"- ANALYZER_PRODUCED_UNKNOWN={manifest['ANALYZER_PRODUCED_UNKNOWN']}",
        f"- FEATURE_READY_RECOVERED={manifest['FEATURE_READY_RECOVERED']}",
        f"- FEATURE_READY_STILL_BLOCKED={manifest['FEATURE_READY_STILL_BLOCKED']}",
        f"- FEATURE_READY_BEFORE={manifest['FEATURE_READY_BEFORE']}",
        f"- FEATURE_READY_AFTER={manifest['FEATURE_READY_AFTER']}",
        f"- REACTION_ROWS_CHANGED={manifest['REACTION_ROWS_CHANGED']}",
        f"- NETWORK_MARKET_FETCHES={manifest['NETWORK_MARKET_FETCHES']}",
        f"- DECISION={manifest['DECISION']}",
        "",
        "Semantic inputs were restricted to official publication material. Rules v3, Qwen, "
        "feature definitions, reaction methodology, TEST, models, backtests, trading, and future "
        "holdout outcomes were not used or changed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
