from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from apps.cli.diagnose_exact_event_session_alignment import build_parser
from src.ai_events.domain.prompt import prompt_hash, schema_hash
from src.event_market_dataset.application import EXPECTED_RULES_FINGERPRINT
from src.event_market_dataset.domain import QWEN_PROMPT_SHA, QWEN_SCHEMA_SHA
from src.events.domain.v3 import rules_v3_fingerprint
from src.exact_event_session_alignment_diagnostics.application import (
    run_session_alignment_diagnostics,
)
from src.exact_event_session_alignment_diagnostics.domain import (
    ARTIFACT_VERSION,
    FUTURE_EVENT_HOLDOUT_START,
    INPUT_DATASET_SHA,
    OUTPUT_DATASET_SHA,
    PR38_ARTIFACT_SHA,
    PR38_RECOVERY_COHORT_SHA,
    session_diagnostic_safety_flags,
    sha256_payload,
)


def test_cli_defaults_to_session_alignment_diagnostics_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "4" * 40])
    assert args.pr38_dir == "artifacts/exact-event-security-history-recovery-v1"
    assert args.output_dir == "artifacts/exact-event-session-alignment-diagnostics-v1"


def test_safety_flags_forbid_model_test_future_and_trading() -> None:
    flags = session_diagnostic_safety_flags()
    assert flags["DIAGNOSTICS_ONLY"] is True
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert flags["RULES_V3_CHANGED"] is False
    assert flags["QWEN_CHANGED"] is False
    assert flags["NLP_TUNING_PERFORMED"] is False
    assert flags["ALIGNMENT_METHODOLOGY_CHANGED"] is False
    assert flags["CONFIRMED_SIGNAL"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False
    assert flags["SANDBOX_ORDER_SUBMISSION_ALLOWED"] is False


def test_cohort_filters_pr38_session_alignment_rows_and_excludes_btbr_future(
    tmp_path: Path,
) -> None:
    pr38_root = _write_pr38_fixture(
        tmp_path,
        event_specs=[
            _spec("gemc", "GEMC", "uid-GEMC"),
            _spec("incb", "INCB", "uid-INCB"),
            _spec("btbr", "BTBR", "uid-BTBR", status="RECOVERED", blocker=None),
            _spec("future", "FUTR", "uid-FUTR", published_at="2026-08-13T10:00:30+00:00"),
        ],
    )
    _write_pair_cache(pr38_root, "GEMC", "uid-GEMC", _gap_times())
    _write_pair_cache(pr38_root, "INCB", "uid-INCB", _gap_times())
    manifest = run_session_alignment_diagnostics(
        pr38_root=pr38_root,
        output_root=tmp_path / ARTIFACT_VERSION,
        base_main_sha="4775d0cbbacf5faeac5726b73e120f7299370faa",
        git_sha="5" * 40,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )

    assert manifest["INPUT_DATASET_SHA"] == INPUT_DATASET_SHA
    assert manifest["OUTPUT_DATASET_SHA"] == OUTPUT_DATASET_SHA
    assert manifest["PR38_ARTIFACT_SHA"] == PR38_ARTIFACT_SHA
    assert manifest["PR38_RECOVERY_COHORT_SHA"] == PR38_RECOVERY_COHORT_SHA
    assert manifest["SESSION_DIAGNOSTIC_COHORT_SHA"] == sha256_payload(["gemc", "incb"])
    assert manifest["DIAGNOSTIC_EVENTS_TOTAL"] == 2
    assert manifest["DIAGNOSTIC_EVENT_IDS"] == ["gemc", "incb"]
    assert manifest["PR38_DATASET_PRESERVED"] == "YES"
    assert manifest["EXISTING_FEATURE_ROWS_PRESERVED"] == "PASS"
    assert manifest["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert manifest["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert manifest["ALIGNMENT_METHODOLOGY_CHANGED"] is False
    assert manifest["MARKET_DATA_METHOD_CHANGED"] is False
    assert manifest["DIAGNOSTIC_ARTIFACT_CONTAINS_NO_PRICE_VALUES"] == "PASS"


def test_session_unknown_common_candle_too_far_and_timestamp_audit(tmp_path: Path) -> None:
    pr38_root = _write_pr38_fixture(tmp_path)
    _write_pair_cache(pr38_root, "GEMC", "uid-GEMC", _gap_times())
    manifest = run_session_alignment_diagnostics(
        pr38_root=pr38_root,
        output_root=tmp_path / "unknown",
        base_main_sha="4775d0cbbacf5faeac5726b73e120f7299370faa",
        git_sha="5" * 40,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    row = cast("list[dict[str, Any]]", manifest["PER_EVENT_DIAGNOSTICS"])[0]
    assert row["SESSION_STATE"] == "OTHER/UNKNOWN"
    assert row["NEXT_COMMON_CANDLE_BEGIN"] == "2026-07-20T10:02:00+00:00"
    assert row["NEXT_COMMON_DELTA_SECONDS"] == 90
    assert row["ROOT_CAUSE"] == "SESSION_UNKNOWN_COMMON_CANDLE_TOO_FAR"
    assert row["RECOVERY_RECOMMENDATION_TYPE"] == "METHODOLOGY_CHANGE_REQUIRED"
    assert row["BASELINE_WINDOW_EQUAL"] is True
    assert row["EFFECTIVE_WINDOW_EQUAL"] is True
    assert row["SECURITY_BASELINE_END"] == "2026-07-20T10:00:00+00:00"
    assert row["BENCHMARK_BASELINE_END"] == "2026-07-20T10:00:00+00:00"
    assert row["SECURITY_EFFECTIVE_BEGIN"] == "2026-07-20T10:02:00+00:00"
    assert row["BENCHMARK_EFFECTIVE_BEGIN"] == "2026-07-20T10:02:00+00:00"
    assert row["CACHE_WINDOW_SUFFICIENT"] is True
    assert "more than one minute" in str(row["WHY_SESSION_STATE"])


def test_baseline_and_effective_missing_detection(tmp_path: Path) -> None:
    baseline_root = _write_pr38_fixture(
        tmp_path / "baseline", event_specs=[_spec("base", "BASE", "uid-BASE")]
    )
    _write_pair_cache(
        baseline_root,
        "BASE",
        "uid-BASE",
        ("2026-07-20T10:01:00+00:00", "2026-07-20T10:31:00+00:00"),
    )
    benchmark_root = _write_pr38_fixture(
        tmp_path / "benchmark", event_specs=[_spec("bench", "BENH", "uid-BENH")]
    )
    _write_security_cache(benchmark_root, "BENH", "uid-BENH", _gap_times())
    _write_benchmark_cache(
        benchmark_root, ("2026-07-20T10:01:00+00:00", "2026-07-20T10:31:00+00:00")
    )
    effective_root = _write_pr38_fixture(
        tmp_path / "effective", event_specs=[_spec("eff", "EFFS", "uid-EFFS")]
    )
    _write_security_cache(
        effective_root,
        "EFFS",
        "uid-EFFS",
        ("2026-07-20T09:30:00+00:00", "2026-07-20T09:59:00+00:00"),
    )
    _write_benchmark_cache(effective_root, _gap_times())

    assert (
        _single_root_cause(baseline_root, tmp_path / "baseline-out") == "BASELINE_SECURITY_MISSING"
    )
    assert (
        _single_root_cause(benchmark_root, tmp_path / "benchmark-out")
        == "BASELINE_BENCHMARK_MISSING"
    )
    assert (
        _single_root_cause(effective_root, tmp_path / "effective-out")
        == "EFFECTIVE_SECURITY_MISSING"
    )


def test_effective_and_baseline_window_mismatch_detection(tmp_path: Path) -> None:
    effective_root = _write_pr38_fixture(
        tmp_path / "effective", event_specs=[_spec("eff-mis", "EFCM", "uid-EFCM")]
    )
    _write_security_cache(
        effective_root,
        "EFCM",
        "uid-EFCM",
        (
            "2026-07-20T09:30:00+00:00",
            "2026-07-20T09:59:00+00:00",
            "2026-07-20T10:00:45+00:00",
            "2026-07-20T10:01:00+00:00",
            "2026-07-20T10:31:00+00:00",
        ),
    )
    _write_benchmark_cache(
        effective_root,
        (
            "2026-07-20T09:30:00+00:00",
            "2026-07-20T09:59:00+00:00",
            "2026-07-20T10:01:00+00:00",
            "2026-07-20T10:31:00+00:00",
        ),
    )
    baseline_root = _write_pr38_fixture(
        tmp_path / "baseline", event_specs=[_spec("base-mis", "BLCM", "uid-BLCM")]
    )
    _write_security_cache(
        baseline_root,
        "BLCM",
        "uid-BLCM",
        (
            "2026-07-20T09:30:00+00:00",
            "2026-07-20T09:58:00+00:00",
            "2026-07-20T09:59:30+00:00",
            "2026-07-20T10:01:00+00:00",
            "2026-07-20T10:31:00+00:00",
        ),
    )
    _write_benchmark_cache(
        baseline_root,
        (
            "2026-07-20T09:30:00+00:00",
            "2026-07-20T09:59:00+00:00",
            "2026-07-20T10:01:00+00:00",
            "2026-07-20T10:31:00+00:00",
        ),
    )

    assert (
        _single_root_cause(effective_root, tmp_path / "effective-mismatch-out")
        == "SECURITY_BENCHMARK_EFFECTIVE_WINDOW_MISMATCH"
    )
    assert (
        _single_root_cause(baseline_root, tmp_path / "baseline-mismatch-out")
        == "SECURITY_BENCHMARK_BASELINE_WINDOW_MISMATCH"
    )


def test_incomplete_candle_filtering_and_cache_window_sufficiency(tmp_path: Path) -> None:
    incomplete_root = _write_pr38_fixture(
        tmp_path / "incomplete", event_specs=[_spec("inc", "INCM", "uid-INCM")]
    )
    _write_security_cache(
        incomplete_root,
        "INCM",
        "uid-INCM",
        (
            "2026-07-20T09:30:00+00:00",
            "2026-07-20T09:59:00+00:00",
            "2026-07-20T10:01:00+00:00",
            "2026-07-20T10:31:00+00:00",
        ),
        incomplete={"2026-07-20T10:01:00+00:00"},
    )
    _write_benchmark_cache(
        incomplete_root,
        (
            "2026-07-20T09:30:00+00:00",
            "2026-07-20T09:59:00+00:00",
            "2026-07-20T10:01:00+00:00",
            "2026-07-20T10:31:00+00:00",
        ),
    )
    narrow_root = _write_pr38_fixture(
        tmp_path / "narrow", event_specs=[_spec("narrow", "NARR", "uid-NARR")]
    )
    _write_pair_cache(
        narrow_root, "NARR", "uid-NARR", ("2026-07-20T09:59:00+00:00", "2026-07-20T10:02:00+00:00")
    )

    incomplete = _single_row(incomplete_root, tmp_path / "incomplete-out")
    assert incomplete["ROOT_CAUSE"] == "SECURITY_CANDLE_INCOMPLETE"
    assert incomplete["SECURITY_INCOMPLETE_NEARBY_COUNT"] == 1
    narrow = _single_row(narrow_root, tmp_path / "narrow-out")
    assert narrow["ROOT_CAUSE"] == "CACHE_WINDOW_TOO_NARROW"
    assert narrow["CACHE_WINDOW_SUFFICIENT"] is False


def test_deterministic_replay_and_no_price_values_in_artifact(tmp_path: Path) -> None:
    pr38_root = _write_pr38_fixture(tmp_path)
    _write_pair_cache(pr38_root, "GEMC", "uid-GEMC", _gap_times())
    left = run_session_alignment_diagnostics(
        pr38_root=pr38_root,
        output_root=tmp_path / "left",
        base_main_sha="4775d0cbbacf5faeac5726b73e120f7299370faa",
        git_sha="5" * 40,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    right = run_session_alignment_diagnostics(
        pr38_root=pr38_root,
        output_root=tmp_path / "right",
        base_main_sha="4775d0cbbacf5faeac5726b73e120f7299370faa",
        git_sha="6" * 40,
        created_at=datetime(2026, 8, 22, tzinfo=UTC),
    )
    assert left["ARTIFACT_SHA"] == right["ARTIFACT_SHA"]
    assert left["DETERMINISTIC_REPLAY"] == "PASS"
    assert _artifact_has_no_forbidden_price_keys(tmp_path / "left")


def test_frozen_contracts_and_documentation() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert prompt_hash() == QWEN_PROMPT_SHA
    assert schema_hash() == QWEN_SCHEMA_SHA
    assert FUTURE_EVENT_HOLDOUT_START.isoformat() == "2026-08-11"
    text = (
        Path(__file__).parents[2] / "docs" / "exact-event-session-alignment-diagnostics-v1.md"
    ).read_text(encoding="utf-8")
    assert "diagnostics-only" in text
    assert "timestamp-only" in text
    assert "ALIGNMENT_METHODOLOGY_CHANGED=false" in text
    assert "MODEL_TRAINING_PERFORMED=false" in text
    assert "TEST_OUTCOME_USED=false" in text


def _single_root_cause(pr38_root: Path, output_root: Path) -> str:
    return str(_single_row(pr38_root, output_root)["ROOT_CAUSE"])


def _single_row(pr38_root: Path, output_root: Path) -> dict[str, Any]:
    manifest = run_session_alignment_diagnostics(
        pr38_root=pr38_root,
        output_root=output_root,
        base_main_sha="4775d0cbbacf5faeac5726b73e120f7299370faa",
        git_sha="5" * 40,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    return cast("list[dict[str, Any]]", manifest["PER_EVENT_DIAGNOSTICS"])[0]


def _write_pr38_fixture(
    tmp_path: Path,
    *,
    event_specs: Sequence[dict[str, object]] | None = None,
) -> Path:
    root = tmp_path / "pr38"
    root.mkdir(parents=True)
    specs = list(event_specs or [_spec("gemc", "GEMC", "uid-GEMC")])
    events = [_event(spec) for spec in specs]
    features = [
        {
            "event_id": "old-ready",
            "feature_cutoff": "2026-07-20T10:00:30+00:00",
            "event_features": {"primary_event_type": "DIVIDEND"},
            "market_features": {"feature_cutoff": "2026-07-20T10:00:30+00:00"},
        }
    ]
    _write_json(
        root / "manifest.json",
        {
            "ARTIFACT_SHA": PR38_ARTIFACT_SHA,
            "INPUT_DATASET_SHA": "19c822112f8b79e09b4067fa253c10118d448981592f66c5b40bfb01495ffc46",
            "OUTPUT_DATASET_SHA": INPUT_DATASET_SHA,
            "RECOVERY_COHORT_SHA": (
                "b9e9ef0b3e9c65b33c30b492161399dac9dcf812e075ebec68991a72c56630f4"
            ),
            "EXISTING_FEATURE_ROWS_PRESERVED": "PASS",
            "FUTURE_EVENT_HOLDOUT_USED": False,
            "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
            "TEST_OUTCOME_USED": False,
        },
    )
    _write_jsonl(root / "events.jsonl", [*events, _old_ready_event()])
    _write_jsonl(root / "features.jsonl", features)
    _write_jsonl(root / "per-event-recovery.jsonl", [_recovery_row(spec) for spec in specs])
    return root


def _spec(
    event_id: str,
    ticker: str,
    uid: str,
    *,
    published_at: str = "2026-07-20T10:00:30+00:00",
    status: str = "BLOCKED",
    blocker: str | None = "SESSION_ALIGNMENT_FAILED",
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "ticker": ticker,
        "uid": uid,
        "published_at": published_at,
        "status": status,
        "blocker": blocker,
    }


def _event(spec: dict[str, object]) -> dict[str, object]:
    published = datetime.fromisoformat(str(spec["published_at"])).astimezone(UTC)
    future = published.date() >= FUTURE_EVENT_HOLDOUT_START
    return {
        "metadata": {
            "event_id": spec["event_id"],
            "ticker": spec["ticker"],
            "issuer": f"{spec['ticker']} Issuer",
            "instrument_uid": spec["uid"],
            "source_code": "SYNTHETIC_EXACT",
            "publication_timestamp_utc": published.isoformat(),
            "publication_date": published.date().isoformat(),
            "timestamp_source_field": "synthetic exact timestamp",
            "future_holdout": future,
        },
        "event_features": {"primary_event_type": "DIVIDEND"},
        "pre_event_market_features": {},
        "target_availability": {
            "reaction_ready": False,
            "feature_ready": False,
            "research_outcomes_visible": False,
            "status": "BLOCKED",
            "missing_reason": "SESSION_ALIGNMENT_FAILED",
        },
        "quality": {},
    }


def _old_ready_event() -> dict[str, object]:
    return {
        "metadata": {
            "event_id": "old-ready",
            "ticker": "OLD",
            "issuer": "Old Issuer",
            "instrument_uid": "uid-OLD",
            "publication_timestamp_utc": "2026-07-20T10:00:30+00:00",
            "publication_date": "2026-07-20",
            "future_holdout": False,
        },
        "event_features": {"primary_event_type": "DIVIDEND"},
        "pre_event_market_features": {},
        "target_availability": {
            "reaction_ready": True,
            "feature_ready": True,
            "research_outcomes_visible": True,
            "status": "REACTION_READY",
            "missing_reason": None,
        },
        "quality": {},
    }


def _recovery_row(spec: dict[str, object]) -> dict[str, object]:
    published = datetime.fromisoformat(str(spec["published_at"])).astimezone(UTC)
    return {
        "EVENT_ID": spec["event_id"],
        "TICKER": spec["ticker"],
        "PUBLICATION_TIMESTAMP": published.isoformat(),
        "FIGI": f"figi-{spec['ticker']}",
        "UID": spec["uid"],
        "CLASS_CODE": "TQBR",
        "RECOVERY_STATUS": spec["status"],
        "FINAL_BLOCKER": spec["blocker"],
    }


def _gap_times() -> tuple[str, ...]:
    return (
        "2026-07-20T09:30:00+00:00",
        "2026-07-20T09:59:00+00:00",
        "2026-07-20T10:02:00+00:00",
        "2026-07-20T10:31:00+00:00",
    )


def _write_pair_cache(root: Path, ticker: str, uid: str, begins: Sequence[str]) -> None:
    _write_security_cache(root, ticker, uid, begins)
    _write_benchmark_cache(root, begins)


def _write_security_cache(
    root: Path,
    ticker: str,
    uid: str,
    begins: Sequence[str],
    *,
    incomplete: set[str] | None = None,
) -> None:
    _write_candle_cache(
        root / "raw-minute-cache" / ticker / "2026-07-20-day.jsonl",
        uid,
        begins,
        ticker=ticker,
        figi=f"figi-{ticker}",
        class_code="TQBR",
        incomplete=incomplete or set(),
    )


def _write_benchmark_cache(root: Path, begins: Sequence[str]) -> None:
    _write_candle_cache(
        root / "raw-minute-cache" / "IMOEX" / "2026-07-20-day.jsonl",
        "uid-IMOEX",
        begins,
        incomplete=set(),
    )


def _write_candle_cache(
    path: Path,
    uid: str,
    begins: Sequence[str],
    *,
    ticker: str | None = None,
    figi: str | None = None,
    class_code: str | None = None,
    incomplete: set[str],
) -> None:
    rows: list[dict[str, object]] = []
    for index, begin_text in enumerate(begins, start=1):
        begin = datetime.fromisoformat(begin_text).astimezone(UTC)
        row: dict[str, object] = {
            "instrument_uid": uid,
            "begin_at": begin.isoformat(),
            "end_at": (begin + timedelta(minutes=1)).isoformat(),
            "open": str(100 + index),
            "high": str(100 + index),
            "low": str(100 + index),
            "close": str(100 + index),
            "volume": index,
            "is_complete": begin.isoformat() not in incomplete,
            "source": "TINVEST_API",
        }
        if ticker is not None:
            row["ticker"] = ticker
        if figi is not None:
            row["figi"] = figi
        if class_code is not None:
            row["class_code"] = class_code
        rows.append(row)
    _write_jsonl(path, rows)


def _artifact_has_no_forbidden_price_keys(root: Path) -> bool:
    forbidden = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "security_return",
        "benchmark_return",
        "abnormal_return",
        "security_log_return",
        "benchmark_log_return",
        "abnormal_log_return",
        "target_class",
    }
    for path in root.glob("*.json*"):
        for row in _read_payloads(path):
            if not _payload_has_no_keys(row, forbidden):
                return False
    return True


def _payload_has_no_keys(payload: object, forbidden: set[str]) -> bool:
    if isinstance(payload, dict):
        items = cast("dict[object, object]", payload).items()
        return all(
            str(key) not in forbidden and _payload_has_no_keys(value, forbidden)
            for key, value in items
        )
    if isinstance(payload, list):
        return all(_payload_has_no_keys(item, forbidden) for item in cast("list[object]", payload))
    return True


def _read_payloads(path: Path) -> list[object]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [json.loads(path.read_text(encoding="utf-8"))]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
