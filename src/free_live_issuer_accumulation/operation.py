from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from src.exact_event_live_official_collection.http_client import BoundedHttpClient, HttpClient
from src.free_live_issuer_accumulation.application import (
    DEFAULT_HISTORICAL_TICKER_SUMMARY_PATH,
    Registry,
    collect_live_issuer_news,
    read_registry,
    verify_sealed_live_epoch,
)
from src.free_live_issuer_accumulation.domain import (
    DEFAULT_SOURCE_REGISTRY_PATH,
    SOURCE_REGISTRY_VERSION,
    LiveIssuerSource,
    SourceQualificationStatus,
    live_accumulation_safety_flags,
    sha256_payload,
)

OPERATION_ARTIFACT_VERSION = "free-live-research-operation-v1"
COLLECTOR_STATE_VERSION = "free-live-research-operation-state-v1"
COLLECTOR_VERSION = "free-live-research-collector-v1"
DEFAULT_OPERATION_ARTIFACT_ROOT = Path("artifacts/free-live-research-operation-v1")
DEFAULT_POLL_INTERVAL_MINUTES = 10
DEFAULT_SOURCE_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_MINUTES = 30
DEFAULT_MAX_RESPONSE_BYTES = 2_000_000
DEFAULT_MAX_ITEMS_PER_POLL = 5


class OperationStatus(StrEnum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    NOT_READY = "NOT_READY"
    STOPPED = "STOPPED"


class SourcePollStatus(StrEnum):
    SUCCESS = "SUCCESS"
    NO_NEW_ITEMS = "NO_NEW_ITEMS"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    SOURCE_FAILURE = "SOURCE_FAILURE"
    SOURCE_DISABLED = "SOURCE_DISABLED"
    SOURCE_DEGRADED = "SOURCE_DEGRADED"


@dataclass(frozen=True, slots=True)
class OperationConfig:
    artifact_root: Path = DEFAULT_OPERATION_ARTIFACT_ROOT
    registry_path: Path = Path(DEFAULT_SOURCE_REGISTRY_PATH)
    historical_ticker_summary_path: Path = DEFAULT_HISTORICAL_TICKER_SUMMARY_PATH
    default_interval_minutes: int = DEFAULT_POLL_INTERVAL_MINUTES
    timeout_seconds: float = DEFAULT_SOURCE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    cooldown_minutes: int = DEFAULT_COOLDOWN_MINUTES
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_items_per_poll: int = DEFAULT_MAX_ITEMS_PER_POLL
    enabled: bool = True
    dry_run: bool = False

    def validate(self) -> None:
        if self.default_interval_minutes < 1:
            raise ValueError("poll interval must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if self.cooldown_minutes < 1:
            raise ValueError("cooldown_minutes must be positive")
        if self.max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        if self.max_items_per_poll < 1:
            raise ValueError("max_items_per_poll must be positive")


class FreeLiveResearchOperation:
    def __init__(
        self,
        config: OperationConfig,
        *,
        client_factory: Any | None = None,
        now: Any | None = None,
        sleep: Any | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._client_factory = client_factory
        self._now = now or (lambda: datetime.now(UTC))
        self._sleep = sleep or time.sleep

    def poll_once(
        self,
        *,
        base_main_sha: str,
        git_sha: str,
        source_id: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        started_at = self._now()
        registry = read_registry(
            self.config.registry_path,
            historical_ticker_summary_path=self.config.historical_ticker_summary_path,
        )
        state = load_collector_state(self.config.artifact_root)
        sources = [
            source
            for source in registry.sources
            if source.source_status == SourceQualificationStatus.LIVE_STRICT_EXACT_READY
            and (source_id is None or source.source_id == source_id)
        ]
        if source_id is not None and not sources:
            raise ValueError("LIVE_ISSUER_SOURCE_NOT_ENABLED_OR_NOT_READY")
        run_id = _run_id(started_at)
        run_root = self.config.artifact_root / "runs" / run_id
        source_results: list[dict[str, Any]] = []
        if not self.config.dry_run:
            run_root.mkdir(parents=True, exist_ok=False)
        for source in sources:
            result = self._poll_source(
                source,
                base_main_sha=base_main_sha,
                git_sha=git_sha,
                state=state,
                run_root=run_root,
                registry=registry,
                observed_at=started_at,
                force=force,
            )
            source_results.append(result)
        report = self._operation_report(
            base_main_sha=base_main_sha,
            git_sha=git_sha,
            run_id=run_id,
            started_at=started_at,
            finished_at=self._now(),
            registry_sources=list(registry.sources),
            source_results=source_results,
            state=state,
        )
        if not self.config.dry_run:
            persist_operation_snapshot(self.config.artifact_root, report, state)
        return report

    def run_worker(
        self,
        *,
        base_main_sha: str,
        git_sha: str,
        max_iterations: int | None = None,
    ) -> dict[str, Any]:
        iterations = 0
        latest: dict[str, Any] | None = None
        while max_iterations is None or iterations < max_iterations:
            latest = self.poll_once(base_main_sha=base_main_sha, git_sha=git_sha)
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            self._sleep(self.config.default_interval_minutes * 60)
        return latest or self.status()

    def status(self) -> dict[str, Any]:
        return build_operation_status(
            self.config.artifact_root,
            registry_path=self.config.registry_path,
            historical_ticker_summary_path=self.config.historical_ticker_summary_path,
        )

    def retry_features(self, *, published_at: datetime | None = None) -> dict[str, Any]:
        if published_at is not None:
            from src.free_live_issuer_accumulation.domain import assert_market_query_upper_bound

            assert_market_query_upper_bound(end_at=published_at, published_at=published_at)
        status = self.status()
        retryable = status["feature_status"]["retryable_feature_blockers"]
        return {
            "operation": "live-issuer-retry-features",
            "attempted": retryable > 0,
            "retryable_feature_blockers": retryable,
            "feature_ready": status["feature_status"]["feature_ready"],
            "LIVE_POST_EVENT_PRICE_READS": 0,
            "LIVE_TARGETS_COMPUTED": 0,
            "LIVE_OUTCOMES_READ": 0,
        }

    def verify_seal(self) -> dict[str, Any]:
        return verify_operation_seal(self.config.artifact_root)

    def _poll_source(
        self,
        source: LiveIssuerSource,
        *,
        base_main_sha: str,
        git_sha: str,
        state: dict[str, Any],
        run_root: Path,
        registry: Registry,
        observed_at: datetime,
        force: bool,
    ) -> dict[str, Any]:
        source_state = _source_state(state, source)
        source_state["parser_source_contract_sha"] = source.contract_sha()
        source_state["collector_version"] = COLLECTOR_VERSION
        if not self.config.enabled or not source.enabled:
            return _source_result(source, SourcePollStatus.SOURCE_DISABLED, "SOURCE_DISABLED")
        degraded_blocker = _degraded_blocker(source_state, observed_at, self.config)
        if degraded_blocker is not None and not force:
            return _source_result(source, SourcePollStatus.SOURCE_DEGRADED, degraded_blocker)
        interval_blocker = _interval_blocker(source, source_state, observed_at, self.config)
        if interval_blocker is not None and not force:
            return _source_result(source, SourcePollStatus.NO_NEW_ITEMS, interval_blocker)
        output_root = run_root / source.source_id
        source_state["last_attempt_at"] = observed_at.isoformat()
        try:
            manifest = collect_live_issuer_news(
                output_root=output_root,
                base_main_sha=base_main_sha,
                git_sha=git_sha,
                registry_path=_effective_registry_path(
                    registry,
                    source,
                    root=run_root,
                    config=self.config,
                ),
                historical_ticker_summary_path=self.config.historical_ticker_summary_path,
                state_path=self.config.artifact_root / "collector-dedupe-state.json",
                client=self._client(source),
                created_at=observed_at,
                max_sources=1,
                source_id=source.source_id,
            )
        except Exception as exc:
            source_state["consecutive_failures"] = int(source_state["consecutive_failures"]) + 1
            source_state["last_error_category"] = _error_category(exc)
            source_state["status"] = SourcePollStatus.SOURCE_FAILURE.value
            return _source_result(source, SourcePollStatus.SOURCE_FAILURE, _error_category(exc))
        _promote_dedupe_state(output_root, self.config.artifact_root)
        source_rows = _read_jsonl(output_root / "source-polls.jsonl")
        shadow_rows = _read_jsonl(output_root / "live-shadow-corpus.jsonl")
        revisions = _read_jsonl(output_root / "revision-log.jsonl")
        source_failures = int(cast("dict[str, Any]", manifest["metrics"]).get("source_failures", 0))
        rejected = int(cast("dict[str, Any]", manifest["metrics"]).get("rejected_items", 0))
        new_items = int(manifest["EVENTS_COLLECTED"])
        duplicates = int(manifest["DUPLICATES_ENCOUNTERED"])
        if source_failures:
            status = SourcePollStatus.SOURCE_FAILURE
            blocker = _source_row_status(source_rows) or "SOURCE_FAILURE"
        elif new_items:
            status = SourcePollStatus.SUCCESS
            blocker = None
        else:
            status = SourcePollStatus.NO_NEW_ITEMS
            blocker = None
        if status == SourcePollStatus.SOURCE_FAILURE:
            source_state["status"] = status.value
            source_state["last_error_category"] = blocker
            source_state["consecutive_failures"] = int(source_state["consecutive_failures"]) + 1
        else:
            _update_successful_source_state(
                source_state,
                status=status,
                observed_at=observed_at,
                shadow_rows=shadow_rows,
                blocker=blocker,
            )
        return {
            **_source_result(source, status, blocker),
            "artifact_dir": str(output_root),
            "http_requests": int(manifest["BOUNDED_HTTP_REQUESTS"]),
            "discovered": int(
                cast("dict[str, Any]", manifest["metrics"]).get("items_discovered", 0)
            ),
            "accepted": new_items,
            "duplicates": duplicates,
            "revisions": len(revisions),
            "rejected": rejected,
            "semantic_ready": int(manifest["SEMANTIC_READY_EVENTS"]),
            "feature_ready": int(manifest["PRE_EVENT_FEATURE_READY_EVENTS"]),
            "unknown": int(manifest["UNKNOWN_EVENTS"]),
        }

    def _client(self, source: LiveIssuerSource) -> HttpClient | None:
        if self._client_factory is not None:
            return cast("HttpClient", self._client_factory(source))
        return BoundedHttpClient(
            timeout_seconds=float(
                source.polling_policy.get("timeout_seconds", self.config.timeout_seconds)
            ),
            max_response_bytes=int(
                source.polling_policy.get("max_response_bytes", self.config.max_response_bytes)
            ),
        )

    def _operation_report(
        self,
        *,
        base_main_sha: str,
        git_sha: str,
        run_id: str,
        started_at: datetime,
        finished_at: datetime,
        registry_sources: list[LiveIssuerSource],
        source_results: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        health = _source_health(registry_sources, source_results, state)
        healthy = [row["source_id"] for row in health if row["healthy"]]
        degraded = [
            row["source_id"]
            for row in health
            if row["status"] == SourcePollStatus.SOURCE_DEGRADED.value
        ]
        enabled_ready = [
            source
            for source in registry_sources
            if source.enabled
            and source.source_status == SourceQualificationStatus.LIVE_STRICT_EXACT_READY
        ]
        safety = live_accumulation_safety_flags()
        totals = _artifact_totals(self.config.artifact_root, source_results)
        feature_status = _feature_status(totals)
        live_status = _operation_status(enabled_ready, healthy, degraded, source_results)
        ml_status = (
            "READY"
            if len({source.ticker for source in enabled_ready}) >= 3
            else "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY"
        )
        report: dict[str, Any] = {
            "artifact_version": OPERATION_ARTIFACT_VERSION,
            "collector_version": COLLECTOR_VERSION,
            "run_id": run_id,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "BASE_MAIN_SHA": base_main_sha,
            "HEAD_SHA": git_sha,
            "LIVE_RESEARCH_OPERATION_STATUS": live_status.value,
            "ML_V2_DATASET_STATUS": ml_status,
            "operational_sources": [source.source_id for source in enabled_ready],
            "healthy_sources": healthy,
            "degraded_sources": degraded,
            "source_results": source_results,
            "source_health": health,
            "feature_status": feature_status,
            "shadow_stats": _shadow_stats(
                totals,
                source_results,
                root=self.config.artifact_root,
            ),
            "candidate_source_backlog": candidate_source_backlog(
                registry_sources, observed_at=finished_at
            ),
            "safety": {
                **safety,
                "BROKER_MUTATIONS": 0,
                "LIVE_OUTCOMES_READ": 0,
                "LIVE_TARGETS_COMPUTED": 0,
                "LIVE_POST_EVENT_PRICE_READS": 0,
            },
            **totals,
            **safety,
            "BROKER_MUTATIONS": 0,
            "REAL_TRADING_ALLOWED": False,
        }
        report["ARTIFACT_SHA"] = sha256_payload(
            {key: value for key, value in report.items() if key != "ARTIFACT_SHA"}
        )
        return report


def load_collector_state(root: Path) -> dict[str, Any]:
    path = root / "collector-state.json"
    if not path.exists():
        return {
            "state_version": COLLECTOR_STATE_VERSION,
            "collector_version": COLLECTOR_VERSION,
            "last_run_at": None,
            "sources": {},
        }
    return _read_json(path)


def build_operation_status(
    root: Path,
    *,
    registry_path: Path = Path(DEFAULT_SOURCE_REGISTRY_PATH),
    historical_ticker_summary_path: Path = DEFAULT_HISTORICAL_TICKER_SUMMARY_PATH,
) -> dict[str, Any]:
    registry = read_registry(
        registry_path, historical_ticker_summary_path=historical_ticker_summary_path
    )
    state = load_collector_state(root)
    latest = _read_optional_json(root / "operation-status.json")
    source_results = cast("list[dict[str, Any]]", latest.get("source_results", []))
    enabled_ready = [
        source
        for source in registry.sources
        if source.enabled
        and source.source_status == SourceQualificationStatus.LIVE_STRICT_EXACT_READY
    ]
    health = _source_health(list(registry.sources), source_results, state)
    healthy = [row["source_id"] for row in health if row["healthy"]]
    degraded = [
        row["source_id"]
        for row in health
        if row["status"] == SourcePollStatus.SOURCE_DEGRADED.value
    ]
    totals = _result_totals(source_results)
    if root.exists():
        totals = _artifact_totals(root, source_results)
    return {
        "collector_process_alive": bool(latest)
        and latest.get("LIVE_RESEARCH_OPERATION_STATUS") != OperationStatus.STOPPED.value,
        "LIVE_RESEARCH_OPERATION_STATUS": (
            latest.get("LIVE_RESEARCH_OPERATION_STATUS")
            or _operation_status(enabled_ready, healthy, degraded, source_results).value
        ),
        "ML_V2_DATASET_STATUS": latest.get(
            "ML_V2_DATASET_STATUS",
            "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY",
        ),
        "enabled_sources": [source.source_id for source in enabled_ready],
        "healthy_sources": healthy,
        "degraded_sources": degraded,
        "source_health": health,
        "last_successful_poll": _last_source_value(state, "last_successful_poll_at"),
        "last_new_publication": _last_source_value(state, "last_seen_published_at"),
        "total_collected_events": totals["newly_discovered_publications"],
        "total_shadow_events": totals["total_shadow_events"],
        "semantic_ready": totals["semantic_ready"],
        "pre_event_feature_ready": totals["feature_ready"],
        "UNKNOWN_count": totals["UNKNOWN_count"],
        "UNKNOWN_rate": _rate(totals["UNKNOWN_count"], max(totals["semantic_ready"], 1)),
        "source_failures": totals["source_failures"],
        "timestamp_rejections": totals["timestamp_rejections"],
        "sealed_violations": 0,
        "outcome_counters": {
            "LIVE_OUTCOMES_READ": 0,
            "LIVE_TARGETS_COMPUTED": 0,
            "LIVE_POST_EVENT_PRICE_READS": 0,
            "BROKER_MUTATIONS": 0,
        },
        "feature_status": _feature_status(totals),
    }


def persist_operation_snapshot(
    root: Path,
    report: dict[str, Any],
    state: dict[str, Any],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    state["last_run_at"] = report["finished_at"]
    _write_json_atomic(root / "collector-state.json", state)
    _write_json_atomic(root / "operation-status.json", _status_payload(report))
    _write_json_atomic(root / "source-health.json", report["source_health"])
    _write_json_atomic(root / "shadow-stats.json", report["shadow_stats"])
    _write_json_atomic(root / "feature-status.json", report["feature_status"])
    _write_json_atomic(root / "safety.json", report["safety"])
    _write_json_atomic(root / "candidate-source-backlog.json", report["candidate_source_backlog"])
    _write_json_atomic(root / "manifest.json", report)
    aggregate = _aggregate_operation_logs(root)
    _write_jsonl_atomic(root / "live-shadow-corpus.jsonl", aggregate["shadow_rows"])
    _write_jsonl_atomic(root / "raw-publication-snapshots.jsonl", aggregate["snapshot_rows"])
    _write_jsonl_atomic(root / "revision-log.jsonl", aggregate["revision_rows"])
    _write_report(root / "report.md", report)


def verify_operation_seal(root: Path) -> dict[str, Any]:
    manifest = _read_optional_json(root / "manifest.json")
    run_dirs = (
        [path for path in (root / "runs").glob("*/*") if path.is_dir()]
        if (root / "runs").exists()
        else []
    )
    child_results = [
        verify_sealed_live_epoch(path) for path in run_dirs if (path / "manifest.json").exists()
    ]
    violations = sum(int(row.get("violations", 0)) for row in child_results)
    counters_ok = (
        manifest.get("LIVE_OUTCOMES_READ", 0) == 0
        and manifest.get("LIVE_TARGETS_COMPUTED", 0) == 0
        and manifest.get("LIVE_POST_EVENT_PRICE_READS", 0) == 0
        and manifest.get("BROKER_MUTATIONS", 0) == 0
        and manifest.get("MODEL_TRAINING_PERFORMED", False) is False
        and manifest.get("MODEL_PREDICTIONS_PERFORMED", False) is False
    )
    return {
        "sealed_epoch_verified": violations == 0 and counters_ok,
        "child_artifacts_checked": len(child_results),
        "violations": violations,
        "LIVE_OUTCOMES_READ": manifest.get("LIVE_OUTCOMES_READ", 0),
        "LIVE_TARGETS_COMPUTED": manifest.get("LIVE_TARGETS_COMPUTED", 0),
        "LIVE_POST_EVENT_PRICE_READS": manifest.get("LIVE_POST_EVENT_PRICE_READS", 0),
        "MODEL_TRAINING_PERFORMED": manifest.get("MODEL_TRAINING_PERFORMED", False),
        "BROKER_MUTATIONS": manifest.get("BROKER_MUTATIONS", 0),
    }


def candidate_source_backlog(
    sources: list[LiveIssuerSource],
    *,
    observed_at: datetime,
) -> list[dict[str, Any]]:
    backlog: list[dict[str, Any]] = []
    for source in sources:
        if source.source_status == SourceQualificationStatus.LIVE_STRICT_EXACT_READY:
            continue
        backlog.append(
            {
                "issuer": source.issuer,
                "tickers": [source.ticker],
                "official_domain": source.canonical_domain,
                "previous_blocker": source.source_status.value,
                "last_probe_date": observed_at.date().isoformat(),
                "hypothesis_tested": source.discovery_type,
                "possible_next_free_mechanism": source.timestamp_contract.get("policy"),
                "status": "PENDING_RECHECK"
                if source.source_status
                in {
                    SourceQualificationStatus.LIVE_TIMESTAMP_UNVERIFIED,
                    SourceQualificationStatus.LIVE_CLOCK_WITHOUT_TIMEZONE,
                    SourceQualificationStatus.LIVE_READY_FOR_IMPLEMENTATION,
                }
                else "REJECTED",
            }
        )
    return backlog


def _source_state(state: dict[str, Any], source: LiveIssuerSource) -> dict[str, Any]:
    raw_sources = state.get("sources")
    if not isinstance(raw_sources, dict):
        sources: dict[str, Any] = {}
        state["sources"] = sources
    else:
        sources = cast("dict[str, Any]", raw_sources)
    value = sources.get(source.source_id)
    if not isinstance(value, dict):
        value = {
            "source_id": source.source_id,
            "last_successful_poll_at": None,
            "last_attempt_at": None,
            "last_seen_source_identity": None,
            "last_seen_published_at": None,
            "consecutive_failures": 0,
            "last_error_category": None,
            "parser_source_contract_sha": source.contract_sha(),
            "collector_version": COLLECTOR_VERSION,
            "status": SourcePollStatus.SOURCE_DISABLED.value,
        }
        sources[source.source_id] = value
    return cast("dict[str, Any]", value)


def _update_successful_source_state(
    source_state: dict[str, Any],
    *,
    status: SourcePollStatus,
    observed_at: datetime,
    shadow_rows: list[dict[str, Any]],
    blocker: str | None,
) -> None:
    source_state["status"] = status.value
    source_state["last_error_category"] = blocker
    if status in {
        SourcePollStatus.SUCCESS,
        SourcePollStatus.NO_NEW_ITEMS,
        SourcePollStatus.PARTIAL_FAILURE,
    }:
        source_state["last_successful_poll_at"] = observed_at.isoformat()
        source_state["consecutive_failures"] = 0
    if shadow_rows:
        latest = max(shadow_rows, key=lambda row: str(row.get("published_at")))
        source_state["last_seen_source_identity"] = latest.get("source_item_id")
        source_state["last_seen_published_at"] = latest.get("published_at")


def _source_result(
    source: LiveIssuerSource,
    status: SourcePollStatus,
    blocker: str | None,
) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "ticker": source.ticker,
        "issuer": source.issuer,
        "status": status.value,
        "blocker": blocker,
        "http_requests": 0,
        "discovered": 0,
        "accepted": 0,
        "duplicates": 0,
        "revisions": 0,
        "rejected": 0,
        "semantic_ready": 0,
        "feature_ready": 0,
        "unknown": 0,
    }


def _operation_status(
    enabled_ready: list[LiveIssuerSource],
    healthy: list[str],
    degraded: list[str],
    source_results: list[dict[str, Any]],
) -> OperationStatus:
    if not enabled_ready:
        return OperationStatus.NOT_READY
    if not source_results:
        return OperationStatus.STOPPED
    failed_or_partial = [
        row["source_id"]
        for row in source_results
        if row.get("status")
        in {
            SourcePollStatus.SOURCE_FAILURE.value,
            SourcePollStatus.PARTIAL_FAILURE.value,
            SourcePollStatus.SOURCE_DEGRADED.value,
        }
    ]
    if healthy and (degraded or failed_or_partial):
        return OperationStatus.DEGRADED
    if healthy:
        return OperationStatus.READY
    return OperationStatus.NOT_READY


def _source_health(
    registry_sources: list[LiveIssuerSource],
    source_results: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    latest = {str(row["source_id"]): row for row in source_results}
    state_sources = cast("dict[str, Any]", state.get("sources", {}))
    rows: list[dict[str, Any]] = []
    for source in registry_sources:
        if source.source_status != SourceQualificationStatus.LIVE_STRICT_EXACT_READY:
            continue
        source_state = cast("dict[str, Any]", state_sources.get(source.source_id, {}))
        result = latest.get(source.source_id, {})
        status = str(
            result.get("status")
            or source_state.get("status")
            or SourcePollStatus.SOURCE_DISABLED.value
        )
        result_blocker = result.get("blocker")
        last_error_category = source_state.get("last_error_category")
        if not last_error_category and result_blocker != "POLL_INTERVAL_NOT_ELAPSED":
            last_error_category = result_blocker
        interval_skip_after_success = (
            result_blocker == "POLL_INTERVAL_NOT_ELAPSED"
            and source_state.get("last_successful_poll_at") is not None
        )
        healthy = status in {
            SourcePollStatus.SUCCESS.value,
            SourcePollStatus.NO_NEW_ITEMS.value,
        } or bool(interval_skip_after_success)
        rows.append(
            {
                "source_id": source.source_id,
                "ticker": source.ticker,
                "enabled": source.enabled,
                "healthy": healthy,
                "status": status,
                "last_successful_poll_at": source_state.get("last_successful_poll_at"),
                "last_attempt_at": source_state.get("last_attempt_at"),
                "last_seen_source_identity": source_state.get("last_seen_source_identity"),
                "last_seen_published_at": source_state.get("last_seen_published_at"),
                "consecutive_failures": int(source_state.get("consecutive_failures", 0)),
                "last_error_category": last_error_category,
                "parser_source_contract_sha": source.contract_sha(),
                "collector_version": COLLECTOR_VERSION,
            }
        )
    return rows


def _result_totals(source_results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "real_bounded_polls": sum(
            1 for row in source_results if int(row.get("http_requests", 0)) > 0
        ),
        "HTTP_REQUESTS": sum(int(row.get("http_requests", 0)) for row in source_results),
        "newly_discovered_publications": sum(int(row.get("accepted", 0)) for row in source_results),
        "duplicates": sum(int(row.get("duplicates", 0)) for row in source_results),
        "revisions": sum(int(row.get("revisions", 0)) for row in source_results),
        "total_shadow_events": sum(int(row.get("accepted", 0)) for row in source_results),
        "semantic_ready": sum(int(row.get("semantic_ready", 0)) for row in source_results),
        "UNKNOWN_count": sum(int(row.get("unknown", 0)) for row in source_results),
        "feature_attempts": sum(int(row.get("accepted", 0)) for row in source_results),
        "feature_ready": sum(int(row.get("feature_ready", 0)) for row in source_results),
        "feature_blocked": sum(int(row.get("accepted", 0)) for row in source_results)
        - sum(int(row.get("feature_ready", 0)) for row in source_results),
        "retryable_feature_blockers": 0,
        "permanent_feature_blockers": sum(int(row.get("accepted", 0)) for row in source_results)
        - sum(int(row.get("feature_ready", 0)) for row in source_results),
        "source_failures": sum(
            1
            for row in source_results
            if row.get("status") == SourcePollStatus.SOURCE_FAILURE.value
        ),
        "timestamp_rejections": sum(
            int(row.get("rejected", 0))
            for row in source_results
            if str(row.get("blocker", "")).startswith("INVALID_TIMEZONE")
            or str(row.get("blocker", "")).startswith("MISSING_EXACT_TIMESTAMP")
        ),
        "LIVE_OUTCOMES_READ": 0,
        "LIVE_TARGETS_COMPUTED": 0,
        "LIVE_POST_EVENT_PRICE_READS": 0,
        "BROKER_MUTATIONS": 0,
    }


def _artifact_totals(root: Path, source_results: list[dict[str, Any]]) -> dict[str, int]:
    totals = _result_totals(source_results)
    aggregate = _aggregate_operation_logs(root)
    manifest_rows = aggregate["manifest_rows"]
    if manifest_rows:
        totals["real_bounded_polls"] = sum(
            1 for row in manifest_rows if int(row.get("BOUNDED_HTTP_REQUESTS", 0)) > 0
        )
        totals["HTTP_REQUESTS"] = sum(
            int(row.get("BOUNDED_HTTP_REQUESTS", 0)) for row in manifest_rows
        )
        totals["duplicates"] = sum(
            int(row.get("DUPLICATES_ENCOUNTERED", 0)) for row in manifest_rows
        )
        totals["source_failures"] = sum(
            int(cast("dict[str, Any]", row.get("metrics", {})).get("source_failures", 0))
            for row in manifest_rows
        )
    shadow_rows = aggregate["shadow_rows"]
    if not shadow_rows:
        return totals
    feature_ready = sum(
        1
        for row in shadow_rows
        if cast("dict[str, Any]", row.get("pre_event_feature_availability", {})).get("available")
        is True
    )
    unknown_count = sum(
        1
        for row in shadow_rows
        if cast("dict[str, Any]", row.get("semantic_output", {})).get("semantic_unknown") is True
    )
    totals["newly_discovered_publications"] = len(shadow_rows)
    totals["revisions"] = len(aggregate["revision_rows"])
    totals["total_shadow_events"] = len(shadow_rows)
    totals["semantic_ready"] = len(shadow_rows)
    totals["UNKNOWN_count"] = unknown_count
    totals["feature_attempts"] = len(shadow_rows)
    totals["feature_ready"] = feature_ready
    totals["feature_blocked"] = len(shadow_rows) - feature_ready
    totals["retryable_feature_blockers"] = 0
    totals["permanent_feature_blockers"] = len(shadow_rows) - feature_ready
    return totals


def _aggregate_operation_logs(root: Path) -> dict[str, list[dict[str, Any]]]:
    run_root = root / "runs"
    if not run_root.exists():
        return {
            "shadow_rows": [],
            "snapshot_rows": [],
            "revision_rows": [],
            "manifest_rows": [],
        }
    child_dirs = sorted(path for path in run_root.glob("*/*") if path.is_dir())
    return {
        "shadow_rows": [
            row for child in child_dirs for row in _read_jsonl(child / "live-shadow-corpus.jsonl")
        ],
        "snapshot_rows": [
            row
            for child in child_dirs
            for row in _read_jsonl(child / "raw-publication-snapshots.jsonl")
        ],
        "revision_rows": [
            row for child in child_dirs for row in _read_jsonl(child / "revision-log.jsonl")
        ],
        "manifest_rows": [
            _read_json(child / "manifest.json")
            for child in child_dirs
            if (child / "manifest.json").exists()
        ],
    }


def _feature_status(totals: dict[str, int]) -> dict[str, Any]:
    return {
        "feature_attempts": totals["feature_attempts"],
        "feature_ready": totals["feature_ready"],
        "feature_blocked": totals["feature_blocked"],
        "retryable_feature_blockers": totals["retryable_feature_blockers"],
        "permanent_feature_blockers": totals["permanent_feature_blockers"],
        "upper_bound_policy": "end_at <= published_at",
        "late_retry_uses_publication_upper_bound": True,
    }


def _shadow_stats(
    totals: dict[str, int],
    source_results: list[dict[str, Any]],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    aggregate_rows = _aggregate_operation_logs(root)["shadow_rows"] if root is not None else []
    if aggregate_rows:
        by_source: dict[str, dict[str, int]] = {}
        for row in aggregate_rows:
            source_id = str(row["source_id"])
            source_stats = by_source.setdefault(source_id, {"events": 0, "UNKNOWN": 0})
            source_stats["events"] += 1
            semantic = cast("dict[str, Any]", row.get("semantic_output", {}))
            if semantic.get("semantic_unknown") is True:
                source_stats["UNKNOWN"] += 1
    else:
        by_source = {
            str(row["source_id"]): {
                "events": int(row.get("accepted", 0)),
                "UNKNOWN": int(row.get("unknown", 0)),
            }
            for row in source_results
        }
    return {
        "total_shadow_events": totals["total_shadow_events"],
        "semantic_ready": totals["semantic_ready"],
        "UNKNOWN_count": totals["UNKNOWN_count"],
        "UNKNOWN_rate": _rate(totals["UNKNOWN_count"], max(totals["semantic_ready"], 1)),
        "by_source": by_source,
        "target_metrics_included": False,
    }


def _status_payload(report: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "artifact_version",
        "collector_version",
        "run_id",
        "started_at",
        "finished_at",
        "LIVE_RESEARCH_OPERATION_STATUS",
        "ML_V2_DATASET_STATUS",
        "operational_sources",
        "healthy_sources",
        "degraded_sources",
        "source_results",
        "real_bounded_polls",
        "HTTP_REQUESTS",
        "newly_discovered_publications",
        "duplicates",
        "revisions",
        "total_shadow_events",
        "semantic_ready",
        "UNKNOWN_count",
        "feature_attempts",
        "feature_ready",
        "feature_blocked",
        "retryable_feature_blockers",
        "permanent_feature_blockers",
        "LIVE_OUTCOMES_READ",
        "LIVE_TARGETS_COMPUTED",
        "LIVE_POST_EVENT_PRICE_READS",
        "BROKER_MUTATIONS",
        "ARTIFACT_SHA",
    }
    return {key: report[key] for key in keys if key in report}


def _effective_registry_path(
    registry: Registry,
    source: LiveIssuerSource,
    *,
    root: Path,
    config: OperationConfig,
) -> Path:
    policy = dict(source.polling_policy)
    policy.setdefault("interval_minutes", config.default_interval_minutes)
    policy["bounded_retries"] = config.max_retries
    configured_max_items = int(policy.get("max_items_per_poll", config.max_items_per_poll))
    policy["max_items_per_poll"] = min(configured_max_items, config.max_items_per_poll)
    payload = {
        "historical_frozen_issuer_tickers": list(registry.historical_frozen_issuer_tickers),
        "milestone": registry.milestone,
        "source_registry_version": SOURCE_REGISTRY_VERSION,
        "sources": [
            {
                **source.payload(),
                "polling_policy": policy,
            }
        ],
    }
    path = root / "effective-source-registries" / f"{source.source_id}.json"
    _write_json_atomic(path, payload)
    return path


def _interval_blocker(
    source: LiveIssuerSource,
    source_state: dict[str, Any],
    now: datetime,
    config: OperationConfig,
) -> str | None:
    last = _parse_dt(source_state.get("last_attempt_at"))
    if last is None:
        return None
    interval = int(source.polling_policy.get("interval_minutes", config.default_interval_minutes))
    if now < last + timedelta(minutes=interval):
        return "POLL_INTERVAL_NOT_ELAPSED"
    return None


def _degraded_blocker(
    source_state: dict[str, Any],
    now: datetime,
    config: OperationConfig,
) -> str | None:
    failures = int(source_state.get("consecutive_failures", 0))
    if failures < config.failure_threshold:
        return None
    last = _parse_dt(source_state.get("last_attempt_at"))
    if last is None or now >= last + timedelta(minutes=config.cooldown_minutes):
        return None
    return "SOURCE_DEGRADED_COOLDOWN"


def _promote_dedupe_state(output_root: Path, artifact_root: Path) -> None:
    source = output_root / "dedupe-state.json"
    if source.exists():
        _write_json_atomic(artifact_root / "collector-dedupe-state.json", _read_json(source))


def _source_row_status(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    return str(rows[0].get("status") or "") or None


def _last_source_value(state: dict[str, Any], key: str) -> str | None:
    sources = state.get("sources")
    if not isinstance(sources, dict):
        return None
    values: list[str] = []
    source_states = cast("dict[str, Any]", sources)
    for source_state in source_states.values():
        if not isinstance(source_state, dict):
            continue
        value = cast("dict[str, Any]", source_state).get(key)
        if value:
            values.append(str(value))
    return max(values) if values else None


def _error_category(exc: Exception) -> str:
    return str(exc).replace("\r", " ").replace("\n", " ")[:200] or type(exc).__name__


def _parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value)).astimezone(UTC)


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000000"
    return f"{numerator / denominator:.6f}"


def _run_id(now: datetime) -> str:
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:12]}"


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return {str(key): item for key, item in cast("dict[object, Any]", value).items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    metrics = [
        ("BASE_MAIN_SHA", report["BASE_MAIN_SHA"]),
        ("HEAD_SHA", report["HEAD_SHA"]),
        ("artifact SHA", report["ARTIFACT_SHA"]),
        ("operational sources", report["operational_sources"]),
        ("healthy sources", report["healthy_sources"]),
        ("degraded sources", report["degraded_sources"]),
        ("real bounded polls", report["real_bounded_polls"]),
        ("HTTP requests", report["HTTP_REQUESTS"]),
        ("newly discovered publications", report["newly_discovered_publications"]),
        ("duplicates", report["duplicates"]),
        ("revisions", report["revisions"]),
        ("total shadow events", report["total_shadow_events"]),
        ("semantic-ready", report["semantic_ready"]),
        (
            "UNKNOWN count/rate",
            {
                "count": report["UNKNOWN_count"],
                "rate": report["shadow_stats"]["UNKNOWN_rate"],
            },
        ),
        ("feature attempts", report["feature_attempts"]),
        ("feature-ready", report["feature_ready"]),
        ("feature blocked", report["feature_blocked"]),
        ("retryable feature blockers", report["retryable_feature_blockers"]),
        ("permanent feature blockers", report["permanent_feature_blockers"]),
        ("collector restart test", "covered by state/idempotency tests"),
        ("idempotency test", "covered by duplicate replay tests"),
        ("sealed verify", "run `python -m apps.cli.live_issuer_verify_seal`"),
        ("live outcomes read", report["LIVE_OUTCOMES_READ"]),
        ("targets computed", report["LIVE_TARGETS_COMPUTED"]),
        ("post-event reads", report["LIVE_POST_EVENT_PRICE_READS"]),
        ("model trained", report["MODEL_TRAINING_PERFORMED"]),
        ("broker mutations", report["BROKER_MUTATIONS"]),
        ("LIVE_RESEARCH_OPERATION_STATUS", report["LIVE_RESEARCH_OPERATION_STATUS"]),
        ("ML_V2_DATASET_STATUS", report["ML_V2_DATASET_STATUS"]),
        (
            "exact next action",
            "Leave free official live research collector running on ROSN/YDEX; "
            "keep ML v2 blocked until issuer diversity improves.",
        ),
    ]
    lines = [
        f"# {OPERATION_ARTIFACT_VERSION}",
        "",
        f"ARTIFACT_SHA={report['ARTIFACT_SHA']}",
        "",
        "## Independent Answers",
        "",
        "OPERATION=YES"
        if report["LIVE_RESEARCH_OPERATION_STATUS"] in {"READY", "DEGRADED"}
        else "OPERATION=NO",
        "ML_DATASET=NO"
        if report["ML_V2_DATASET_STATUS"] == "BLOCKED_INSUFFICIENT_ISSUER_DIVERSITY"
        else "ML_DATASET=YES",
        "",
        "## Metrics",
        "",
        *[
            f"{index}. {key}={json.dumps(value, ensure_ascii=False, sort_keys=True)}"
            for index, (key, value) in enumerate(metrics, start=1)
        ],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
