from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID

from sqlalchemy import select

from src.free_historical_data.registry import compliant_exact_audits
from src.historical_news.infrastructure.models import (
    HistoricalNewsCandidateRecord,
    HistoricalNewsSourceRecord,
)
from src.live_corpus_operations.domain import (
    ACCEPTED_SOURCE_CODES,
    CorpusSnapshot,
    LiveRunConfig,
    MaturationOutcome,
    MaturationState,
    SourceOutcome,
)
from src.live_corpus_operations.state import sanitize_error
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandExecutor(Protocol):
    async def module(
        self,
        module: str,
        arguments: list[str],
        *,
        accepted_codes: frozenset[int] = frozenset({0}),
        timeout_seconds: float = 900.0,
    ) -> CommandResult: ...


class LocalCommandExecutor:
    async def module(
        self,
        module: str,
        arguments: list[str],
        *,
        accepted_codes: frozenset[int] = frozenset({0}),
        timeout_seconds: float = 900.0,
    ) -> CommandResult:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            module,
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_seconds)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(f"bounded command timed out: {module}") from None
        result = CommandResult(
            returncode=process.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace").strip(),
            stderr=stderr.decode("utf-8", errors="replace").strip(),
        )
        if result.returncode not in accepted_codes:
            detail = result.stderr or result.stdout or f"exit {result.returncode}"
            raise RuntimeError(f"{module} failed: {sanitize_error(detail)}")
        return result


class LocalLiveCorpusBackend:
    def __init__(
        self,
        *,
        executor: CommandExecutor | None = None,
        repo_root: Path,
    ) -> None:
        self._executor = executor or LocalCommandExecutor()
        self._repo_root = repo_root
        self._audits = {item.source_code: item for item in compliant_exact_audits()}
        self._new_publication_times: list[datetime] = []

    def validate_source(self, source_code: str) -> None:
        if source_code not in ACCEPTED_SOURCE_CODES:
            raise ValueError("source is not accepted for REAL ML corpus")
        audit = self._audits.get(source_code)
        if audit is None:
            raise ValueError("accepted source no longer passes exact-source registry validation")
        audit.validate()

    async def ingest_source(
        self,
        source_code: str,
        config: LiveRunConfig,
        checkpoint: dict[str, object],
    ) -> SourceOutcome:
        del checkpoint
        arguments = [
            "--source-code",
            source_code,
            "--from",
            config.date_from.isoformat(),
            "--to",
            config.date_to.isoformat(),
            "--limit",
            str(config.limit),
            "--max-pages",
            "1",
            "--timeout",
            str(config.timeout_seconds),
            "--max-retries",
            str(config.max_retries),
            "--min-request-interval",
            "0.5",
            "--match-instruments",
        ]
        if config.dry_run:
            arguments.append("--dry-run")
        result = await self._executor.module("apps.cli.collect_live_news", arguments)
        payload = _last_json_object(result.stdout)
        last_item_id, last_item_at = (None, None)
        if not config.dry_run:
            last_item_id, last_item_at = await self._latest_source_item(source_code)
            run_id = payload.get("run_id")
            if isinstance(run_id, str) and run_id:
                self._new_publication_times.extend(await self._imported_publications(UUID(run_id)))
        return SourceOutcome(
            source_code=source_code,
            status="SUCCEEDED",
            items_seen=int(payload.get("discovered_count", 0)),
            items_imported=int(payload.get("imported_count", 0)),
            duplicates=int(payload.get("duplicate_count", 0)),
            matched=int(payload.get("matched_news_count", 0)),
            rejected=int(payload.get("rejected_count", 0)),
            last_item_id=last_item_id,
            last_item_at=last_item_at,
        )

    async def mature(self, config: LiveRunConfig) -> MaturationOutcome:
        before = self.snapshot()
        recent_boundary = config.date_to - timedelta(days=7)
        maturation_from = (
            min(self._new_publication_times)
            if self._new_publication_times
            else max(config.date_from, recent_boundary)
        )
        range_arguments = [
            "--from",
            maturation_from.isoformat(),
            "--to",
            config.date_to.isoformat(),
            "--limit",
            str(config.limit),
        ]
        if config.dry_run:
            range_arguments.append("--dry-run")
        prepared_result = await self._executor.module(
            "apps.cli.prepare_reaction_ready_corpus", range_arguments
        )
        prepared = _last_json_object(prepared_result.stdout)
        prepared_counts = _object(prepared, "prepared") if "prepared" in prepared else prepared
        matched = int(prepared_counts.get("matched_count", prepared_counts.get("matched", 0)))
        unmatched = int(prepared_counts.get("unmatched_count", prepared_counts.get("unmatched", 0)))
        if config.dry_run:
            return MaturationOutcome(
                matched=matched,
                unmatched=unmatched,
                waiting_intraday=max(0, matched - before.intraday_reaction_ready),
                waiting_daily=max(0, matched - before.daily_reaction_ready),
                moex_status="DRY_RUN",
                database_status="DRY_RUN",
                state_counts=_state_counts(before, matched),
            )
        errors: list[str] = []
        moex_status = "SUCCEEDED"
        windows = prepared_counts.get("market_backfill_windows", prepared_counts.get("windows", []))
        unique_benchmark_windows: set[tuple[str, str]] = set()
        if isinstance(windows, list):
            for raw_window in cast("list[object]", windows):
                if not isinstance(raw_window, dict):
                    continue
                window = cast("dict[str, Any]", raw_window)
                ticker = str(window.get("ticker", ""))
                from_date = _iso_date(window.get("date_from"))
                till_date = min(_iso_date(window.get("date_to")), date.today())
                if from_date > till_date:
                    continue
                arguments = [
                    "--ticker",
                    ticker,
                    "--from",
                    from_date.isoformat(),
                    "--till",
                    till_date.isoformat(),
                ]
                try:
                    await self._executor.module("apps.cli.backfill_candles", arguments)
                    unique_benchmark_windows.add((from_date.isoformat(), till_date.isoformat()))
                except Exception as exc:
                    moex_status = "PARTIAL"
                    errors.append(sanitize_error(exc))
        for from_text, till_text in sorted(unique_benchmark_windows):
            try:
                await self._executor.module(
                    "apps.cli.backfill_benchmark",
                    ["IMOEX", "--from", from_text, "--till", till_text],
                )
            except Exception as exc:
                moex_status = "PARTIAL"
                errors.append(sanitize_error(exc))
        await self._executor.module(
            "apps.cli.compute_abnormal_reactions",
            ["--all", "--limit", "1000"],
            accepted_codes=frozenset({0, 1}),
        )
        cumulative_from = before.date_from or config.date_from.date().isoformat()
        await self._executor.module(
            "apps.cli.build_official_source_corpus",
            [
                "--from",
                cumulative_from,
                "--to",
                config.date_to.isoformat(),
                "--limit",
                "1000",
                "--output",
                "artifacts/reaction-ready-corpus-v3",
            ],
        )
        await self._executor.module("apps.cli.build_free_daily_historical_corpus", [])
        await self._executor.module("apps.cli.ml_readiness", [])
        after = self.snapshot()
        waiting_intraday = max(0, after.matched_total - after.intraday_reaction_ready)
        waiting_daily = max(0, after.matched_total - after.daily_reaction_ready)
        return MaturationOutcome(
            matched=matched,
            unmatched=unmatched,
            intraday_reactions_created=max(
                0, after.intraday_reaction_ready - before.intraday_reaction_ready
            ),
            daily_reactions_created=max(
                0, after.daily_reaction_ready - before.daily_reaction_ready
            ),
            intraday_features_created=max(
                0, after.intraday_feature_ready - before.intraday_feature_ready
            ),
            daily_features_created=max(0, after.daily_feature_ready - before.daily_feature_ready),
            waiting_intraday=waiting_intraday,
            waiting_daily=waiting_daily,
            moex_status=moex_status,
            database_status="SUCCEEDED",
            state_counts=_state_counts(after, matched),
            errors=tuple(errors),
        )

    def snapshot(self) -> CorpusSnapshot:
        coverage = _read_optional_json(
            self._repo_root / "artifacts/free-daily-historical-v1/coverage.json"
        )
        readiness = _read_optional_json(
            self._repo_root / "artifacts/predictive-baseline-v1/readiness.json"
        )
        intraday = _object(coverage, "intraday")
        per_ticker = {
            str(key): int(value)
            for key, value in _object(coverage, "daily_feature_per_ticker").items()
        }
        return CorpusSnapshot(
            captured_at=datetime.now(UTC).isoformat(),
            real_total=int(coverage.get("provenance_real", coverage.get("total", 0))),
            exact_total=int(_object(coverage, "timestamp_quality").get("EXACT", 0)),
            matched_total=int(coverage.get("matched", 0)),
            intraday_reaction_ready=int(intraday.get("reaction_ready", 0)),
            intraday_feature_ready=int(intraday.get("feature_ready", 0)),
            daily_reaction_ready=int(coverage.get("daily_reaction_ready", 0)),
            daily_feature_ready=int(coverage.get("daily_feature_ready", 0)),
            ticker_count=int(readiness.get("ticker_count", len(per_ticker))),
            source_count=int(readiness.get("source_count", len(_object(coverage, "per_source")))),
            date_from=_optional_text(readiness.get("date_from")),
            date_to=_optional_text(readiness.get("date_to")),
            ticker_distribution=per_ticker,
        )

    async def _latest_source_item(self, source_code: str) -> tuple[str | None, str | None]:
        engine = create_engine(get_settings().database_url)
        try:
            async with create_session_factory(engine)() as session:
                result = await session.execute(
                    select(
                        HistoricalNewsCandidateRecord.source_item_id,
                        HistoricalNewsCandidateRecord.source_published_at,
                    )
                    .join(
                        HistoricalNewsSourceRecord,
                        HistoricalNewsSourceRecord.id == HistoricalNewsCandidateRecord.source_id,
                    )
                    .where(HistoricalNewsSourceRecord.source_code == source_code)
                    .order_by(
                        HistoricalNewsCandidateRecord.source_published_at.desc(),
                        HistoricalNewsCandidateRecord.source_item_id.desc(),
                    )
                    .limit(1)
                )
                row = result.first()
        finally:
            await engine.dispose()
        if row is None:
            return None, None
        return str(row[0]), None if row[1] is None else row[1].isoformat()

    async def _imported_publications(self, run_id: UUID) -> list[datetime]:
        engine = create_engine(get_settings().database_url)
        try:
            async with create_session_factory(engine)() as session:
                result = await session.execute(
                    select(HistoricalNewsCandidateRecord.source_published_at)
                    .where(
                        HistoricalNewsCandidateRecord.ingestion_run_id == run_id,
                        HistoricalNewsCandidateRecord.status == "IMPORTED",
                        HistoricalNewsCandidateRecord.source_published_at.is_not(None),
                    )
                    .order_by(HistoricalNewsCandidateRecord.source_published_at)
                )
                return [item for item in result.scalars() if item is not None]
        finally:
            await engine.dispose()


def _state_counts(snapshot: CorpusSnapshot, matched: int) -> dict[str, int]:
    return {
        MaturationState.INGESTED.value: snapshot.real_total,
        MaturationState.MATCHED.value: snapshot.matched_total,
        MaturationState.WAITING_INTRADAY_TARGET.value: max(
            0, matched - snapshot.intraday_reaction_ready
        ),
        MaturationState.INTRADAY_READY.value: snapshot.intraday_reaction_ready,
        MaturationState.WAITING_DAILY_TARGET.value: max(0, matched - snapshot.daily_reaction_ready),
        MaturationState.DAILY_READY.value: snapshot.daily_reaction_ready,
        MaturationState.FEATURE_READY.value: snapshot.daily_feature_ready,
    }


def _last_json_object(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = cast("object", json.loads(line))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return {str(key): item for key, item in cast("dict[object, Any]", value).items()}
    raise ValueError("command output did not contain a JSON object")


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return {str(key): item for key, item in cast("dict[object, Any]", value).items()}


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _iso_date(value: object) -> date:
    if not isinstance(value, str):
        raise ValueError("market window date is missing")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
