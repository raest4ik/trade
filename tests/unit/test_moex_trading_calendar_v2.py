from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest

from src.moex_trading_calendar_v2.application import (
    distinct_moex_trading_days,
    resolve_schedule_payload,
)
from src.moex_trading_calendar_v2.domain import (
    MoexSessionEvidenceV2,
    RuntimeSessionStatus,
    SessionKind,
    SessionStatus,
)
from src.moex_trading_calendar_v2.moex import MoexTradingCalendarResolver
from src.moex_trading_calendar_v2.reporting import build_calendar_artifact
from src.production_dry_run_burnin_v1.policy import MoexIssSessionVerifier

CHECKED_AT = datetime(2027, 1, 10, 12, tzinfo=UTC)


def _payload() -> dict[str, Any]:
    days = {
        "2026-01-01": {"tradedate": "2026-01-01", "is_traded": 0, "reason": "H"},
        "2026-05-01": {"tradedate": "2026-05-01", "is_traded": 1, "reason": "W"},
        "2026-05-04": {"tradedate": "2026-05-04", "is_traded": 1, "reason": "N"},
        "2026-09-12": {"tradedate": "2026-09-12", "is_traded": 0, "reason": "N"},
        "2026-09-19": {"tradedate": "2026-09-19", "is_traded": 1, "reason": "W"},
        "2026-09-20": {"tradedate": "2026-09-20", "is_traded": 1, "reason": "W"},
        "2026-09-21": {"tradedate": "2026-09-21", "is_traded": 1, "reason": "N"},
        "2026-11-28": {"tradedate": "2026-11-28", "is_traded": 0, "reason": "N"},
        "2026-12-05": {"tradedate": "2026-12-05", "is_traded": 1, "reason": "W"},
        "2026-12-07": {"tradedate": "2026-12-07", "is_traded": 1, "reason": "N"},
    }
    return {
        "market": "stock",
        "board": "TQBR",
        "days": days,
        "holidays": {"2026-01-01": "holiday", "2026-05-01": "holiday"},
        "windows": {
            "weekday": [{"from": "06:50", "till": "23:50"}],
            "weekend": [{"from": "09:50", "till": "19:00"}],
        },
    }


def _page(payload: dict[str, Any] | None = None) -> str:
    source = payload or _payload()
    years: dict[str, dict[str, Any]] = {}
    for day, row in source["days"].items():
        years.setdefault(day[:4], {})[day] = row
    holiday_years: dict[str, list[dict[str, str]]] = {}
    for day, kind in source["holidays"].items():
        holiday_years.setdefault(day[:4], []).append({"date": day, "type": kind})
    next_data = {
        "props": {
            "pageProps": {
                "initData": {
                    "offDays": {"stock": years},
                    "holidays": holiday_years,
                },
                "schedule": {
                    "marketsSchedules": [
                        {
                            "market": "stock",
                            "id": "stocks-market",
                            "daysSchedules": [
                                {"type": key, "intervals": value}
                                for key, value in source["windows"].items()
                            ],
                        }
                    ]
                },
            }
        }
    }
    return (
        '<html><script id="__NEXT_DATA__" type="application/json">'
        f"{json.dumps(next_data)}"
        "</script></html>"
    )


def _resolver(handler: Any) -> MoexTradingCalendarResolver:
    return MoexTradingCalendarResolver(
        runtime_base_url="https://iss.moex.com/iss",
        timeout_seconds=1,
        max_retries=0,
        user_agent="test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: CHECKED_AT,
    )


def _resolve(day: str, payload: dict[str, Any] | None = None) -> MoexSessionEvidenceV2:
    return resolve_schedule_payload(
        payload or _payload(),
        date.fromisoformat(day),
        checked_at=CHECKED_AT,
        evidence_sha="calendar-sha",
    )


def test_regular_weekday_open_from_authoritative_schedule() -> None:
    row = _resolve("2026-09-21")
    assert row.status == SessionStatus.OPEN
    assert row.session_kind == SessionKind.REGULAR
    assert row.moex_business_date == date(2026, 9, 21)


def test_weekend_trading_day_can_be_open() -> None:
    row = _resolve("2026-09-19")
    assert row.status == SessionStatus.OPEN
    assert row.session_kind == SessionKind.WEEKEND_ADDITIONAL


def test_weekend_closed_day_can_be_closed() -> None:
    assert _resolve("2026-09-12").status == SessionStatus.CLOSED


def test_holiday_trading_day_can_be_open() -> None:
    row = _resolve("2026-05-01")
    assert row.status == SessionStatus.OPEN
    assert row.session_kind == SessionKind.HOLIDAY_ADDITIONAL


def test_holiday_closed_day_can_be_closed() -> None:
    assert _resolve("2026-01-01").status == SessionStatus.CLOSED


def test_no_weekend_shortcut_exists() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=_page())

    row = _resolver(handler).resolve(date(2026, 9, 19))
    assert row.status == SessionStatus.OPEN
    assert [request.url.host for request in requests] == ["www.moex.com"]


def test_calendar_range_fetches_authoritative_schedule_once() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=_page())

    rows = _resolver(handler).resolve_many(
        [date(2026, 9, 19), date(2026, 9, 20), date(2026, 9, 21)]
    )
    assert [row.status for row in rows] == [
        SessionStatus.OPEN,
        SessionStatus.OPEN,
        SessionStatus.OPEN,
    ]
    assert len(requests) == 1


def test_unknown_authoritative_response_fails_closed() -> None:
    assert _resolve("2026-07-01").status == SessionStatus.UNKNOWN


def test_http_failure_returns_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    assert _resolver(handler).resolve(date(2026, 9, 19)).status == SessionStatus.UNKNOWN


def test_invalid_payload_returns_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>missing structured data</html>")

    assert _resolver(handler).resolve(date(2026, 9, 19)).status == SessionStatus.UNKNOWN


def test_wrong_market_scope_not_accepted() -> None:
    payload = _payload()
    payload["market"] = "currency"
    row = _resolve("2026-09-19", payload)
    assert row.status == SessionStatus.UNKNOWN
    assert row.reason == "WRONG_MARKET_SCOPE"


def test_weekend_session_has_authoritative_business_date() -> None:
    assert _resolve("2026-09-19").moex_business_date == date(2026, 9, 21)


def test_regular_session_business_date() -> None:
    assert _resolve("2026-09-21").moex_business_date == date(2026, 9, 21)


def test_weekend_and_following_regular_session_not_double_counted_when_same_business_date() -> None:
    assert distinct_moex_trading_days([_resolve("2026-09-19"), _resolve("2026-09-21")]) == 1


def test_distinct_moex_trading_days_uses_business_date_not_calendar_date() -> None:
    rows = [_resolve("2026-09-19"), _resolve("2026-09-20"), _resolve("2026-09-21")]
    assert {row.calendar_date for row in rows} == {
        date(2026, 9, 19),
        date(2026, 9, 20),
        date(2026, 9, 21),
    }
    assert distinct_moex_trading_days(rows) == 1


def test_latest_authoritative_amendment_is_represented_by_current_payload() -> None:
    assert _resolve("2026-11-28").status == SessionStatus.CLOSED
    assert _resolve("2026-12-05").status == SessionStatus.OPEN


def test_planned_closed_runtime_open_contradiction_fails_closed() -> None:
    row = resolve_schedule_payload(
        _payload(),
        date(2026, 9, 12),
        checked_at=CHECKED_AT,
        evidence_sha="calendar-sha",
        runtime_status=RuntimeSessionStatus.OPEN,
        runtime_evidence_sha="runtime-sha",
    )
    assert row.status == SessionStatus.UNKNOWN
    assert row.reason == "PLANNED_RUNTIME_CONTRADICTION"


def test_scheduler_readiness_requires_open_window_and_resolved_business_date() -> None:
    row = _resolve("2026-09-19")
    moscow = ZoneInfo("Europe/Moscow")
    assert row.is_session_scheduled is True
    assert row.is_within_session_window(datetime(2026, 9, 19, 10, tzinfo=moscow))
    assert not row.is_within_session_window(datetime(2026, 9, 19, 20, tzinfo=moscow))


def test_burnin_adapter_preserves_explicit_v2_semantics() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_page())

    verifier = MoexIssSessionVerifier(
        base_url="https://iss.moex.com/iss",
        timeout_seconds=1,
        user_agent="test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=lambda: CHECKED_AT,
    )
    row = verifier.verify(date(2026, 9, 19))
    assert row.status.value == "OPEN"
    assert row.moex_business_date == "2026-09-21"
    assert row.session_policy_version == "authoritative-moex-trading-calendar-v2"


def test_naive_scheduler_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="SESSION_CHECK_TIME_MUST_BE_TIMEZONE_AWARE"):
        _resolve("2026-09-19").is_within_session_window(datetime(2026, 9, 19, 10))


def test_open_session_without_following_business_date_fails_closed() -> None:
    payload = _payload()
    payload["days"] = {
        "2026-12-05": payload["days"]["2026-12-05"],
    }
    assert _resolve("2026-12-05", payload).status == SessionStatus.UNKNOWN


def test_calendar_range_cli_is_bounded() -> None:
    from apps.cli.production_dry_run_burnin import build_parser

    args = build_parser().parse_args(
        ["calendar-range", "--from", "2026-09-19", "--to", "2026-09-21"]
    )
    assert args.command == "calendar-range"


def test_calendar_artifact_rebuilds_byte_for_byte(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_calendar_artifact(first, base_main_sha="base", code_sha="code")
    second_manifest = build_calendar_artifact(second, base_main_sha="base", code_sha="code")
    assert first_manifest == second_manifest
    assert {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
