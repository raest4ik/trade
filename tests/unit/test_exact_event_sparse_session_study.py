from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from apps.cli.study_exact_event_sparse_sessions import build_parser
from src.exact_event_sparse_session_study.application import run_sparse_session_study
from src.exact_event_sparse_session_study.domain import (
    ARTIFACT_VERSION,
    DECISION_RULES,
    INPUT_DATASET_SHA,
    OUTPUT_DATASET_SHA,
    PR39_ARTIFACT_SHA,
    PR39_SESSION_DIAGNOSTIC_COHORT_SHA,
    sha256_payload,
    sparse_study_safety_flags,
)


def test_cli_defaults_to_sparse_session_study_artifact() -> None:
    args = build_parser().parse_args(["--base-main-sha", "a" * 40])
    assert args.events == "artifacts/exact-event-security-history-recovery-v1/events.jsonl"
    assert args.split_manifest == (
        "artifacts/exact-event-predictive-baseline-v1/15m-split-manifest.json"
    )
    assert args.output_dir == "artifacts/exact-event-sparse-session-methodology-study-v1"


def test_safety_flags_close_model_test_future_and_trading_surfaces() -> None:
    flags = sparse_study_safety_flags()
    assert flags["MODEL_TRAINING_PERFORMED"] is False
    assert flags["TEST_OUTCOME_USED"] is False
    assert flags["TEST_EVALUATION_PERFORMED"] is False
    assert flags["OBSERVED_TEST_ROWS_USED"] == 0
    assert flags["FUTURE_EVENT_HOLDOUT_USED"] is False
    assert flags["FUTURE_EVENT_HOLDOUT_OBSERVED"] is False
    assert flags["STRICT_EXACT_METHODOLOGY_CHANGED"] is False
    assert flags["REAL_ORDER_SUBMISSION_ALLOWED"] is False
    assert flags["SANDBOX_ORDER_SUBMISSION_ALLOWED"] is False


def test_train_and_validation_rows_allowed_test_rows_excluded_before_analysis(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(
        tmp_path,
        [
            _event("train", "AAA", "2026-01-10T10:00:00+00:00", split="TRAIN"),
            _event("validation", "BBB", "2026-06-01T10:00:00+00:00", split="VALIDATION"),
            _event("test", "CCC", "2026-07-10T10:00:00+00:00", split="TEST"),
        ],
    )
    _write_complete_pair(fixture.cache_root, "AAA", "uid-AAA", "2026-01-10T09:00:00+00:00", 100)
    _write_complete_pair(fixture.cache_root, "BBB", "uid-BBB", "2026-06-01T09:00:00+00:00", 100)
    _write_complete_pair(fixture.cache_root, "CCC", "uid-CCC", "2026-07-10T09:00:00+00:00", 100)

    manifest = _run(tmp_path, fixture)

    assert manifest["DEVELOPMENT_EXACT_TOTAL"] == 2
    assert manifest["TIMESTAMP_STUDY_ELIGIBLE"] == 2
    assert manifest["OBSERVED_TEST_ROWS_USED"] == 0
    assert manifest["TEST_ROWS_EXCLUDED_BEFORE_ANALYSIS"] == "PASS"
    excluded = cast("list[dict[str, Any]]", manifest["SPLIT_EXCLUSIONS"])
    assert [row["EVENT_ID"] for row in excluded] == ["test"]


def test_future_rows_excluded_and_unknown_split_fails_closed(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path,
        [
            _event("train", "AAA", "2026-01-10T10:00:00+00:00", split="TRAIN"),
            _event("unknown", "BBB", "2026-07-04T10:00:00+00:00", split=None),
            _event("future", "CCC", "2026-08-11T10:00:00+00:00", split=None),
        ],
    )
    _write_complete_pair(fixture.cache_root, "AAA", "uid-AAA", "2026-01-10T09:00:00+00:00", 100)
    manifest = _run(tmp_path, fixture)

    assert manifest["DEVELOPMENT_EXACT_TOTAL"] == 1
    split_exclusions = cast("list[dict[str, Any]]", manifest["SPLIT_EXCLUSIONS"])
    future_exclusions = cast("list[dict[str, Any]]", manifest["FUTURE_HOLDOUT_EXCLUSIONS"])
    assert [row["EVENT_ID"] for row in split_exclusions] == ["unknown"]
    assert [row["EVENT_ID"] for row in future_exclusions] == ["future"]
    assert manifest["FUTURE_EVENT_HOLDOUT_USED"] is False


def test_first_common_complete_delay_no_rounding_and_delay_bins(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path,
        [
            _event("delay60", "AAA", "2026-01-10T10:00:00+00:00", split="TRAIN"),
            _event("delay61", "BBB", "2026-01-11T10:00:59+00:00", split="TRAIN"),
            _event("delay301", "CCC", "2026-01-12T10:00:59+00:00", split="TRAIN"),
        ],
    )
    _write_complete_pair(
        fixture.cache_root,
        "AAA",
        "uid-AAA",
        "2026-01-10T09:00:00+00:00",
        100,
        skip_security_begins={"2026-01-10T10:00:00+00:00"},
    )
    _write_complete_pair(
        fixture.cache_root,
        "BBB",
        "uid-BBB",
        "2026-01-11T09:00:00+00:00",
        100,
        skip_security_begins={"2026-01-11T10:01:00+00:00"},
    )
    _write_complete_pair(
        fixture.cache_root,
        "CCC",
        "uid-CCC",
        "2026-01-12T09:00:00+00:00",
        100,
        skip_security_begins={
            "2026-01-12T10:01:00+00:00",
            "2026-01-12T10:02:00+00:00",
            "2026-01-12T10:03:00+00:00",
            "2026-01-12T10:04:00+00:00",
            "2026-01-12T10:05:00+00:00",
        },
    )

    rows = _rows(_run(tmp_path, fixture))

    assert rows["delay60"]["COMMON_CANDLE_DELAY_SECONDS"] == 60
    assert rows["delay60"]["DELAY_BIN"] == "0-60 sec"
    assert rows["delay61"]["COMMON_CANDLE_DELAY_SECONDS"] == 61
    assert rows["delay61"]["DELAY_BIN"] == ">60-120 sec"
    assert rows["delay301"]["COMMON_CANDLE_DELAY_SECONDS"] == 301
    assert rows["delay301"]["DELAY_BIN"] == ">300-600 sec"


def test_candidate_threshold_coverage(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path,
        [
            _event("d0", "AAA", "2026-01-10T10:00:00+00:00", split="TRAIN"),
            _event("d180", "BBB", "2026-01-11T10:00:00+00:00", split="TRAIN"),
            _event("d600", "CCC", "2026-01-12T10:00:00+00:00", split="TRAIN"),
        ],
    )
    _write_complete_pair(fixture.cache_root, "AAA", "uid-AAA", "2026-01-10T09:00:00+00:00", 100)
    _write_complete_pair(
        fixture.cache_root,
        "BBB",
        "uid-BBB",
        "2026-01-11T09:00:00+00:00",
        100,
        skip_security_begins={"2026-01-11T10:00:00+00:00", "2026-01-11T10:01:00+00:00"},
    )
    _write_complete_pair(
        fixture.cache_root,
        "CCC",
        "uid-CCC",
        "2026-01-12T09:00:00+00:00",
        100,
        skip_security_begins={f"2026-01-12T10:{minute:02d}:00+00:00" for minute in range(10)},
    )
    manifest = _run(tmp_path, fixture)
    coverage = cast("dict[str, dict[str, Any]]", manifest["CANDIDATE_THRESHOLD_COVERAGE"])

    assert coverage["60"]["EVENT_COUNT"] == 1
    assert coverage["180"]["EVENT_COUNT"] == 2
    assert coverage["600"]["EVENT_COUNT"] == 3
    assert coverage["180"]["INCREMENTAL_COUNT_VS_60S"] == 1
    assert coverage["600"]["INCREMENTAL_COUNT_VS_60S"] == 2


def test_incomplete_candles_are_ignored(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path, [_event("row", "AAA", "2026-01-10T10:00:00+00:00", split="TRAIN")]
    )
    _write_complete_pair(
        fixture.cache_root,
        "AAA",
        "uid-AAA",
        "2026-01-10T09:00:00+00:00",
        100,
        incomplete_security_begins={"2026-01-10T10:00:00+00:00"},
    )
    row = _rows(_run(tmp_path, fixture))["row"]

    assert row["FIRST_COMMON_COMPLETE_CANDLE_BEGIN"] == "2026-01-10T10:01:00+00:00"
    assert row["COMMON_CANDLE_DELAY_SECONDS"] == 60


def test_cache_coverage_uncertain_is_audit_only_not_recommendation(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path, [_event("row", "AAA", "2026-01-10T10:00:00+00:00", split="TRAIN")]
    )
    _write_complete_pair(fixture.cache_root, "AAA", "uid-AAA", "2026-01-10T09:00:00+00:00", 75)
    manifest = _run(tmp_path, fixture)
    row = _rows(manifest)["row"]

    assert row["CACHE_COVERAGE_STATUS"] == "CACHE_COVERAGE_UNCERTAIN"
    assert manifest["CACHE_COVERAGE_UNCERTAIN_COUNT"] == 1
    assert manifest["DELAY_DISTRIBUTION"][0]["COUNT"] == 0


def test_pre_event_density_uses_only_timestamps_ending_before_publication(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path, [_event("row", "AAA", "2026-01-10T10:00:30+00:00", split="TRAIN")]
    )
    _write_complete_pair(
        fixture.cache_root,
        "AAA",
        "uid-AAA",
        "2026-01-10T09:00:00+00:00",
        100,
        skip_security_begins={"2026-01-10T09:30:00+00:00", "2026-01-10T09:31:00+00:00"},
    )
    row = _rows(_run(tmp_path, fixture))["row"]

    assert row["PRE_EVENT_CANDLE_DENSITY_60M"] == 0.966102
    assert row["MAX_DENSITY_INPUT_TIMESTAMP"] == "2026-01-10T10:00:00+00:00"
    assert row["MAX_DENSITY_INPUT_TIMESTAMP"] < row["PUBLICATION_TIMESTAMP"]


def test_no_price_volume_return_target_or_prediction_keys_in_artifact(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path,
        [_event("row", "AAA", "2026-01-10T10:00:00+00:00", split="TRAIN")],
        forbidden_tail=', "pre_event_market_features": {"pre_return_5m": not_json}',
    )
    _write_complete_pair(
        fixture.cache_root,
        "AAA",
        "uid-AAA",
        "2026-01-10T09:00:00+00:00",
        100,
        broken_price_values=True,
    )
    manifest = _run(tmp_path, fixture)

    assert manifest["STUDY_ARTIFACT_OUTCOME_FREE"] == "PASS"
    assert manifest["STUDY_ARTIFACT_PRICE_FREE"] == "PASS"
    assert _artifact_has_no_forbidden_keys(fixture.output_root)


def test_decision_rules_and_artifact_sha_are_deterministic(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path, [_event("row", "AAA", "2026-01-10T10:00:00+00:00", split="TRAIN")]
    )
    _write_complete_pair(fixture.cache_root, "AAA", "uid-AAA", "2026-01-10T09:00:00+00:00", 100)
    first = _run(tmp_path, fixture, git_sha="1" * 40, output_name="one")
    second = _run(tmp_path, fixture, git_sha="2" * 40, output_name="two")

    assert first["DECISION_RULES_SHA"] == sha256_payload(DECISION_RULES)
    assert first["ARTIFACT_SHA"] == second["ARTIFACT_SHA"]
    assert first["DETERMINISTIC_REPLAY"] == "PASS"


def test_strict_alignment_code_not_imported_or_changed() -> None:
    import src.exact_event_sparse_session_study.application as application

    source = Path(application.__file__).read_text(encoding="utf-8")
    assert "align_exact_event" not in source
    assert "classify_session" not in source
    assert "HORIZONS_MINUTES" not in source


def test_no_model_or_broker_write_calls() -> None:
    source = Path("src/exact_event_sparse_session_study/application.py").read_text(encoding="utf-8")
    forbidden = (
        "LogisticRegression",
        "RandomForest",
        "XGBoost",
        "CatBoost",
        "OrdersService",
        "post_order",
        "BUY",
        "SELL",
    )
    assert all(item not in source for item in forbidden)


def test_manifest_preserves_dataset_and_pr39_lineage(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path, [_event("row", "AAA", "2026-01-10T10:00:00+00:00", split="TRAIN")]
    )
    _write_complete_pair(fixture.cache_root, "AAA", "uid-AAA", "2026-01-10T09:00:00+00:00", 100)
    manifest = _run(tmp_path, fixture)

    assert manifest["INPUT_DATASET_SHA"] == INPUT_DATASET_SHA
    assert manifest["OUTPUT_DATASET_SHA"] == OUTPUT_DATASET_SHA
    assert manifest["PR39_ARTIFACT_SHA"] == PR39_ARTIFACT_SHA
    assert manifest["PRODUCTION_DATASET_CHANGED"] is False
    assert manifest["EXISTING_EVENTS_PRESERVED"] == "PASS"
    assert manifest["EXISTING_FEATURE_ROWS_PRESERVED"] == "PASS"


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.events_path = root / "events.jsonl"
        self.split_manifest_path = root / "15m-split-manifest.json"
        self.pr39_root = root / "pr39"
        self.cache_root = root / "cache"
        self.output_root = root / ARTIFACT_VERSION


def _write_fixture(
    tmp_path: Path, events: list[dict[str, Any]], *, forbidden_tail: str = ""
) -> Fixture:
    fixture = Fixture(tmp_path)
    fixture.pr39_root.mkdir(parents=True)
    _write_json(
        fixture.pr39_root / "manifest.json",
        {
            "ARTIFACT_SHA": PR39_ARTIFACT_SHA,
            "INPUT_DATASET_SHA": INPUT_DATASET_SHA,
            "OUTPUT_DATASET_SHA": OUTPUT_DATASET_SHA,
            "SESSION_DIAGNOSTIC_COHORT_SHA": PR39_SESSION_DIAGNOSTIC_COHORT_SHA,
            "FUTURE_EVENT_HOLDOUT_USED": False,
            "FUTURE_EVENT_HOLDOUT_OBSERVED": False,
            "TEST_OUTCOME_USED": False,
        },
    )
    assignments = [
        {"event_id": item["event_id"], "split": item["split"]}
        for item in events
        if item["split"] is not None
    ]
    _write_json(
        fixture.split_manifest_path,
        {
            "assignments": assignments,
            "cluster_integrity": "PASS",
            "counts": {"TRAIN": 1, "VALIDATION": 1, "TEST": 1},
            "date_ranges": {
                "TRAIN": {"from": "2026-01-01", "to": "2026-05-19"},
                "VALIDATION": {"from": "2026-05-20", "to": "2026-07-02"},
                "TEST": {"from": "2026-07-06", "to": "2026-08-10"},
            },
            "leakage_check": "PASS",
            "protocol": "DETERMINISTIC_CHRONOLOGICAL_60_20_20_GROUPED_V1",
            "split_sha": "split-sha",
            "target_outcomes_inspected_before_lock": False,
        },
    )
    fixture.events_path.write_text(
        "".join(_event_line(item, forbidden_tail=forbidden_tail) + "\n" for item in events),
        encoding="utf-8",
    )
    return fixture


def _event(event_id: str, ticker: str, published_at: str, *, split: str | None) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "ticker": ticker,
        "uid": f"uid-{ticker}",
        "issuer": f"{ticker} Issuer",
        "published_at": published_at,
        "split": split,
    }


def _event_line(item: dict[str, Any], *, forbidden_tail: str) -> str:
    published = datetime.fromisoformat(str(item["published_at"])).astimezone(UTC)
    metadata = {
        "event_id": item["event_id"],
        "ticker": item["ticker"],
        "issuer": item["issuer"],
        "instrument_uid": item["uid"],
        "source_code": f"{item['ticker']}_OFFICIAL_EXACT",
        "publication_timestamp_utc": published.isoformat(),
        "timestamp_quality": "EXACT",
        "future_holdout": published.date().isoformat() >= "2026-08-11",
    }
    if forbidden_tail:
        return '{"metadata": ' + json.dumps(metadata, sort_keys=True) + forbidden_tail + "}"
    return json.dumps(
        {
            "metadata": metadata,
            "pre_event_market_features": {"pre_return_5m": "FORBIDDEN_NOT_PARSED"},
            "target_availability": {"target_class": "FORBIDDEN_NOT_PARSED"},
        },
        sort_keys=True,
    )


def _write_complete_pair(
    cache_root: Path,
    ticker: str,
    uid: str,
    start: str,
    count: int,
    *,
    skip_security_begins: set[str] | None = None,
    incomplete_security_begins: set[str] | None = None,
    broken_price_values: bool = False,
) -> None:
    begin = datetime.fromisoformat(start).astimezone(UTC)
    security_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    skip_security_begins = skip_security_begins or set()
    incomplete_security_begins = incomplete_security_begins or set()
    for offset in range(count):
        current = begin + timedelta(minutes=offset)
        current_text = current.isoformat()
        benchmark_rows.append(
            _candle("uid-IMOEX", current, ticker="IMOEX", broken_price_values=broken_price_values)
        )
        if current_text in skip_security_begins:
            continue
        security_rows.append(
            _candle(
                uid,
                current,
                ticker=ticker,
                is_complete=current_text not in incomplete_security_begins,
                broken_price_values=broken_price_values,
            )
        )
    day = begin.date().isoformat()
    _write_jsonl(cache_root / ticker / f"{day}-day.jsonl", security_rows)
    _write_jsonl(cache_root / "IMOEX" / f"{day}-day.jsonl", benchmark_rows)


def _candle(
    uid: str,
    begin: datetime,
    *,
    ticker: str,
    is_complete: bool = True,
    broken_price_values: bool = False,
) -> dict[str, Any]:
    bad = "BROKEN_PRICE_VALUE" if broken_price_values else "100"
    return {
        "begin_at": begin.isoformat(),
        "end_at": (begin + timedelta(minutes=1)).isoformat(),
        "instrument_uid": uid,
        "is_complete": is_complete,
        "ticker": ticker,
        "open": bad,
        "high": bad,
        "low": bad,
        "close": bad,
        "volume": "BROKEN_VOLUME_VALUE" if broken_price_values else 1,
        "source": "TINVEST_API",
    }


def _run(
    tmp_path: Path,
    fixture: Fixture,
    *,
    git_sha: str = "5" * 40,
    output_name: str | None = None,
) -> dict[str, Any]:
    output = fixture.root / output_name if output_name is not None else fixture.output_root
    return run_sparse_session_study(
        events_path=fixture.events_path,
        split_manifest_path=fixture.split_manifest_path,
        pr39_root=fixture.pr39_root,
        output_root=output,
        base_main_sha="ac852f36e46cdf1f9e1bb119e41e3aaa6e622645",
        git_sha=git_sha,
        cache_roots=(fixture.cache_root,),
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )


def _rows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["EVENT_ID"]): row
        for row in cast("list[dict[str, Any]]", manifest["PER_EVENT_ROWS"])
    }


def _artifact_has_no_forbidden_keys(root: Path) -> bool:
    forbidden = {
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "security_return",
        "benchmark_return",
        "abnormal_return",
        "target_class",
        "prediction",
        "signal",
    }
    for path in root.glob("*.json*"):
        payloads = (
            [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if path.suffix == ".jsonl"
            else [json.loads(path.read_text(encoding="utf-8"))]
        )
        if not all(_payload_has_no_keys(payload, forbidden) for payload in payloads):
            return False
    return True


def _payload_has_no_keys(payload: object, forbidden: set[str]) -> bool:
    if isinstance(payload, dict):
        return all(
            str(key) not in forbidden and _payload_has_no_keys(value, forbidden)
            for key, value in cast("dict[object, object]", payload).items()
        )
    if isinstance(payload, list):
        return all(_payload_has_no_keys(item, forbidden) for item in cast("list[object]", payload))
    return True


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
