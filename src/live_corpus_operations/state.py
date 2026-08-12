from __future__ import annotations

import json
import os
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from src.live_corpus_operations.domain import (
    ACCEPTED_SOURCE_CODES,
    DEFAULT_LOG_RETENTION,
    SOURCE_TICKERS,
    LiveRunReport,
)

_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|password|token|secret)(\s*[:=]\s*)([^\s,;]+)"
)


def sanitize_error(value: object) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")[:1000]
    return _SECRET_PATTERN.sub(r"\1\2[REDACTED]", text)


@dataclass(slots=True)
class FileJobLock(AbstractContextManager["FileJobLock"]):
    path: Path
    stale_after: timedelta = timedelta(hours=6)
    acquired: bool = False
    token: str = ""

    def try_acquire(self, *, now: datetime | None = None) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        current = now or datetime.now(UTC)
        self.token = uuid4().hex
        payload = {
            "pid": os.getpid(),
            "token": self.token,
            "started_at": current.isoformat(),
        }
        for _ in range(2):
            try:
                with self.path.open("x", encoding="utf-8", newline="\n") as output:
                    output.write(json.dumps(payload, sort_keys=True) + "\n")
                self.acquired = True
                return True
            except FileExistsError:
                if not self._stale(current):
                    return False
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
        return False

    def _stale(self, now: datetime) -> bool:
        try:
            payload = _read_object(self.path)
            started_at = datetime.fromisoformat(str(payload["started_at"]))
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return True
        return now - started_at > self.stale_after

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            payload = _read_object(self.path)
            if payload.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        finally:
            self.acquired = False

    def __enter__(self) -> FileJobLock:
        if not self.try_acquire():
            raise RuntimeError("live corpus job is already running")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class OperationsStateStore:
    def __init__(self, root: Path, *, log_retention: int = DEFAULT_LOG_RETENTION) -> None:
        self.root = root
        self.checkpoints_path = root / "checkpoints.json"
        self.health_path = root / "health.json"
        self.history_path = root / "growth-history.jsonl"
        self.logs_path = root / "logs"
        self.log_retention = log_retention

    def checkpoints(self) -> dict[str, Any]:
        if not self.checkpoints_path.exists():
            return {
                "schema_version": "live-corpus-checkpoints-v1",
                "last_run_at": None,
                "last_success_at": None,
                "sources": {code: _empty_source_checkpoint() for code in ACCEPTED_SOURCE_CODES},
            }
        return _read_object(self.checkpoints_path)

    def health(self) -> dict[str, Any]:
        if not self.health_path.exists():
            return {
                "schema_version": "live-corpus-health-v1",
                "last_run_at": None,
                "last_success_at": None,
                "status": "NEVER_RUN",
                "sources": {
                    ticker: _empty_source_health(source_code)
                    for source_code, ticker in SOURCE_TICKERS.items()
                },
                "database_status": "UNKNOWN",
                "moex_status": "UNKNOWN",
            }
        return _read_object(self.health_path)

    def persist(self, report: LiveRunReport, *, successful: bool) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        checkpoints = self.checkpoints()
        health = self.health()
        checkpoints["last_run_at"] = report.finished_at
        health["last_run_at"] = report.finished_at
        health["status"] = report.status.value
        health["database_status"] = report.maturation.database_status
        health["moex_status"] = report.maturation.moex_status
        if successful:
            checkpoints["last_success_at"] = report.finished_at
            health["last_success_at"] = report.finished_at
        checkpoint_sources = _object_dict(checkpoints, "sources")
        health_sources = _object_dict(health, "sources")
        for outcome in report.source_results:
            checkpoint = _object_dict(checkpoint_sources, outcome.source_code)
            checkpoint["last_run_at"] = report.finished_at
            checkpoint["items_seen"] = int(checkpoint.get("items_seen", 0)) + outcome.items_seen
            checkpoint["items_imported"] = (
                int(checkpoint.get("items_imported", 0)) + outcome.items_imported
            )
            if outcome.status == "SUCCEEDED":
                checkpoint["last_success_at"] = report.finished_at
                checkpoint["last_successful_source_item"] = outcome.last_item_id
                checkpoint["last_item_at"] = outcome.last_item_at
            source_health = _object_dict(health_sources, SOURCE_TICKERS[outcome.source_code])
            if outcome.status == "SUCCEEDED":
                source_health["last_success"] = report.finished_at
                source_health["last_item_at"] = outcome.last_item_at
                source_health["consecutive_failures"] = 0
            else:
                source_health["consecutive_failures"] = (
                    int(source_health.get("consecutive_failures", 0)) + 1
                )
        _write_json_atomic(self.checkpoints_path, checkpoints)
        _write_json_atomic(self.health_path, health)
        if successful:
            self._append_history(report)
        self._write_log(report)

    def _append_history(self, report: LiveRunReport) -> None:
        existing = (
            self.history_path.read_text(encoding="utf-8") if self.history_path.exists() else ""
        )
        known = {
            str(item.get("run_id"))
            for item in _jsonl_objects(existing)
            if item.get("run_id") is not None
        }
        if report.run_id in known:
            return
        row = {
            "run_id": report.run_id,
            "date": report.finished_at[:10],
            **report.snapshot.payload(),
        }
        content = existing + json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        _write_text_atomic(self.history_path, content)

    def _write_log(self, report: LiveRunReport) -> None:
        self.logs_path.mkdir(parents=True, exist_ok=True)
        path = (
            self.logs_path
            / f"{report.started_at.replace(':', '').replace('+', '_')}-{report.run_id}.json"
        )
        _write_json_atomic(path, report.payload())
        logs = sorted(self.logs_path.glob("*.json"), key=lambda item: item.name, reverse=True)
        for old in logs[self.log_retention :]:
            old.unlink(missing_ok=True)


def _empty_source_checkpoint() -> dict[str, Any]:
    return {
        "last_run_at": None,
        "last_success_at": None,
        "last_successful_source_item": None,
        "last_item_at": None,
        "items_seen": 0,
        "items_imported": 0,
    }


def _empty_source_health(source_code: str) -> dict[str, Any]:
    return {
        "source_code": source_code,
        "last_success": None,
        "last_item_at": None,
        "consecutive_failures": 0,
    }


def _object_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        value = {}
        payload[key] = value
    return cast("dict[str, Any]", value)


def _read_object(path: Path) -> dict[str, Any]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return {str(key): item for key, item in cast("dict[object, Any]", value).items()}


def _jsonl_objects(content: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in content.splitlines():
        if not line.strip():
            continue
        value = cast("object", json.loads(line))
        if isinstance(value, dict):
            result.append(
                {str(key): item for key, item in cast("dict[object, Any]", value).items()}
            )
    return result


def _write_json_atomic(path: Path, payload: object) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
