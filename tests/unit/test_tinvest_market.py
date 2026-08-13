from __future__ import annotations

import asyncio
import json
import math
import ssl
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import cast

import httpx
import pytest

from apps.cli.tinvest_market_build import build_parser
from src.events.domain.v3 import rules_v3_fingerprint
from src.holdout_evaluation.domain import EXPECTED_RULES_FINGERPRINT
from src.shared.config.settings import DEFAULT_OLLAMA_MODEL
from src.tinvest_market.application import MAX_CHUNK_DAYS, acquire_history, date_chunks
from src.tinvest_market.client import (
    TInvestAuthError,
    TInvestContour,
    TInvestInstrument,
    TInvestReadOnlyClient,
)
from src.tinvest_market.config import (
    READONLY_TOKEN_ENV,
    SANDBOX_TOKEN_ENV,
    TBANK_TLS_VERIFY_ENV,
    MissingTokenError,
    TInvestTokens,
    load_readonly_token,
    load_sandbox_token,
    tbank_tls_context,
)
from src.tinvest_market.domain import (
    BENCHMARK_TICKER,
    FEATURE_DATASET_VERSION,
    FEATURE_WINDOW_SESSIONS,
    RAW_DATASET_VERSION,
    SECURITY_TICKERS,
    DailyBar,
    SplitConfig,
    SplitName,
    build_dataset,
    resolve_instrument,
    temporal_split,
)
from src.tinvest_market.policy import (
    PRICE_ADJUSTMENT_STATUS,
    PRIVATE_MODEL_TRAINING_ALLOWED,
    PRIVATE_RESEARCH_BACKTEST_ALLOWED,
    PUBLIC_REDISTRIBUTION_ALLOWED,
    SOURCE,
    SOURCE_COST,
    execution_safety,
)
from src.tinvest_market.reporting import compare_moex_targets, write_feature_artifacts


def test_missing_tokens_fail_closed_with_env_name_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(READONLY_TOKEN_ENV, raising=False)
    monkeypatch.delenv(SANDBOX_TOKEN_ENV, raising=False)
    with pytest.raises(MissingTokenError) as readonly:
        load_readonly_token()
    with pytest.raises(MissingTokenError) as sandbox:
        load_sandbox_token()
    assert str(readonly.value) == READONLY_TOKEN_ENV
    assert str(sandbox.value) == SANDBOX_TOKEN_ENV


def test_token_repr_and_errors_never_disclose_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "test-secret-that-must-never-appear"
    tokens = TInvestTokens(readonly=secret, sandbox=secret)
    assert secret not in repr(tokens)
    monkeypatch.setenv(READONLY_TOKEN_ENV, secret)
    assert load_readonly_token() == secret
    assert secret not in str(TInvestAuthError("TINVEST_AUTH_FAILED"))


def test_tbank_tls_requires_explicit_verified_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TBANK_TLS_VERIFY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=TBANK_TLS_VERIFY_ENV):
        tbank_tls_context()
    monkeypatch.setenv(TBANK_TLS_VERIFY_ENV, "True")
    context = tbank_tls_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert version("t-tech-investments") == "1.49.3"


async def test_client_uses_exact_readonly_rest_path_and_sanitizes_auth() -> None:
    secret = "local-test-secret"
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(401, json={"message": secret})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TInvestReadOnlyClient(
            token=secret,
            contour=TInvestContour.READONLY_PRODUCTION,
            max_retries=0,
            client=http_client,
        )
        with pytest.raises(TInvestAuthError) as error:
            await client.find_instruments("SBER", instrument_kind="INSTRUMENT_TYPE_SHARE")
    assert requests[0].url.path.endswith(
        "/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/FindInstrument"
    )
    assert str(error.value) == "TINVEST_AUTH_FAILED"
    assert secret not in str(error.value)


async def test_client_retries_429_bounded_and_contours_never_fallback() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"instruments": [_instrument_payload("SBER", "uid-SBER")]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TInvestReadOnlyClient(
            token="secret",
            contour=TInvestContour.SANDBOX_READONLY_CONNECTIVITY,
            max_retries=1,
            client=http_client,
            sleep=False,
        )
        rows = await client.find_instruments("SBER", instrument_kind="INSTRUMENT_TYPE_SHARE")
    assert len(rows) == 1
    assert len(requests) == 2
    assert all(request.url.host == "sandbox-invest-public-api.tbank.ru" for request in requests)


async def test_client_retries_transient_protocol_error_bounded() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.RemoteProtocolError("incomplete response")
        return httpx.Response(200, json={"instruments": [_instrument_payload("SBER", "uid-SBER")]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TInvestReadOnlyClient(
            token="secret",
            contour=TInvestContour.READONLY_PRODUCTION,
            max_retries=1,
            client=http_client,
            sleep=False,
        )
        rows = await client.find_instruments("SBER", instrument_kind="INSTRUMENT_TYPE_SHARE")
    assert len(rows) == 1
    assert calls == 2


async def test_daily_candle_request_uses_exchange_source_without_incompatible_limit() -> None:
    request_body: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        request_body.update(json.loads(request.content))
        return httpx.Response(200, json={"candles": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TInvestReadOnlyClient(
            token="secret",
            contour=TInvestContour.READONLY_PRODUCTION,
            client=http_client,
        )
        await client.fetch_daily_candles(
            instrument_uid="uid-SBER",
            date_from=date(2020, 1, 1),
            date_to=date(2020, 1, 2),
        )
    assert request_body["interval"] == "CANDLE_INTERVAL_DAY"
    assert request_body["candleSourceType"] == "CANDLE_SOURCE_EXCHANGE"
    assert "limit" not in request_body


async def test_indicative_parser_allows_optional_class_code() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        payload = _instrument_payload("IMOEX", "uid-IMOEX", "INSTRUMENT_TYPE_INDEX")
        payload.pop("classCode")
        return httpx.Response(200, json={"instruments": [payload]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TInvestReadOnlyClient(
            token="secret",
            contour=TInvestContour.READONLY_PRODUCTION,
            client=http_client,
        )
        rows = await client.list_indicatives()
    assert rows[0].ticker == "IMOEX"
    assert rows[0].class_code == ""


def test_client_has_no_generic_or_execution_surface() -> None:
    public = {name for name in dir(TInvestReadOnlyClient) if not name.startswith("_")}
    assert public == {
        "aclose",
        "contour",
        "fetch_daily_candles",
        "fetch_schedules",
        "find_instruments",
        "get_instrument_by_uid",
        "list_indicatives",
    }
    source = (
        Path("src/tinvest_market").read_text(encoding="utf-8")
        if Path("src/tinvest_market").is_file()
        else "".join(
            path.read_text(encoding="utf-8") for path in Path("src/tinvest_market").glob("*.py")
        )
    )
    banned = ("PostOrder", "ReplaceOrder", "CancelOrder", "PostStopOrder", "CancelStopOrder")
    assert not any(item in source for item in banned)
    assert "OrdersService" not in source
    assert "SandboxService" not in source
    assert "t_tech.invest.Client" not in source
    assert "verify=False" not in source
    assert "CERT_NONE" not in source


def test_execution_safety_is_immutable_at_false() -> None:
    assert execution_safety()
    assert not any(execution_safety().values())


def test_policy_is_zero_cost_private_and_not_redistributable() -> None:
    assert SOURCE == "TINVEST_API"
    assert SOURCE_COST == "ZERO_RUB"
    assert PRIVATE_MODEL_TRAINING_ALLOWED is True
    assert PRIVATE_RESEARCH_BACKTEST_ALLOWED is True
    assert PUBLIC_REDISTRIBUTION_ALLOWED is False
    assert PRICE_ADJUSTMENT_STATUS == "UNVERIFIED_TINVEST_DAILY_CANDLE_PRICES"


def test_instrument_resolution_is_exact_and_fails_on_ambiguity() -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    exact = _instrument("SBER", "uid-1")
    fuzzy = _instrument("SBERP", "uid-2")
    assert resolve_instrument("SBER", (fuzzy, exact), resolved_at=now).instrument_uid == "uid-1"
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        resolve_instrument("SBER", (exact, replace(exact, instrument_uid="uid-3")), resolved_at=now)
    with pytest.raises(ValueError, match="MISSING"):
        resolve_instrument("SBER", (fuzzy,), resolved_at=now)
    alternate_board = replace(exact, instrument_uid="uid-3", class_code="SPEQ")
    assert (
        resolve_instrument("SBER", (alternate_board, exact), resolved_at=now).instrument_uid
        == "uid-1"
    )


def test_chunks_are_bounded_and_cover_each_day_once() -> None:
    start, end = date(2000, 1, 1), date(2026, 8, 1)
    chunks = date_chunks(start, end)
    assert chunks[0][0] == start and chunks[-1][1] == end
    assert all((right - left).days <= MAX_CHUNK_DAYS for left, right in chunks)
    assert all(
        previous[1] + timedelta(days=1) == current[0] for previous, current in pairwise(chunks)
    )


def test_default_history_floor_precedes_api_reported_instrument_starts() -> None:
    assert build_parser().parse_args([]).date_from == "1970-01-01"


async def test_acquisition_is_real_uid_based_checkpointed_and_idempotent(tmp_path: Path) -> None:
    candle_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal candle_calls
        body = json.loads(request.content)
        if request.url.path.endswith("InstrumentsService/FindInstrument"):
            ticker = body["query"]
            return httpx.Response(
                200, json={"instruments": [_instrument_payload(ticker, f"uid-{ticker}")]}
            )
        if request.url.path.endswith("InstrumentsService/Indicatives"):
            return httpx.Response(
                200,
                json={
                    "instruments": [
                        _instrument_payload("IMOEX", "uid-IMOEX", "INSTRUMENT_TYPE_INDEX")
                    ]
                },
            )
        if request.url.path.endswith("InstrumentsService/GetInstrumentBy"):
            uid = str(body["id"])
            ticker = uid.removeprefix("uid-")
            kind = "INSTRUMENT_TYPE_INDEX" if ticker == "IMOEX" else "INSTRUMENT_TYPE_SHARE"
            return httpx.Response(
                200,
                json={"instrument": _instrument_payload(ticker, uid, kind)},
            )
        if request.url.path.endswith("MarketDataService/GetCandles"):
            candle_calls += 1
            return httpx.Response(200, json={"candles": [_candle_payload(date(2020, 1, 2), 100.0)]})
        raise AssertionError(request.url.path)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TInvestReadOnlyClient(
            token="secret", contour=TInvestContour.READONLY_PRODUCTION, client=http_client
        )
        first = await acquire_history(
            client,
            raw_dir=tmp_path,
            date_from=date(2020, 1, 1),
            date_to=date(2020, 1, 3),
            git_sha="abc",
            resolved_at=datetime(2026, 8, 13, tzinfo=UTC),
        )
        first_calls = candle_calls
        second = await acquire_history(
            client,
            raw_dir=tmp_path,
            date_from=date(2020, 1, 1),
            date_to=date(2020, 1, 3),
            git_sha="abc",
            resolved_at=datetime(2026, 8, 13, tzinfo=UTC),
        )
    assert first_calls == len(SECURITY_TICKERS) + 1
    assert candle_calls == first_calls
    assert first.manifest["dataset_sha"] == second.manifest["dataset_sha"]
    assert set(second.manifest["cache_hits"]) == {*SECURITY_TICKERS, BENCHMARK_TICKER}
    assert first.manifest["source"] == SOURCE
    assert first.manifest["resolved_securities_count"] == len(SECURITY_TICKERS)
    assert first.manifest["unresolved_security_tickers"] == []
    assert all(item["rows_synthesized"] == 0 for item in first.manifest["gap_diagnostics"].values())
    artifact_text = await asyncio.to_thread(_read_tree, tmp_path)
    assert "secret" not in artifact_text


async def test_incomplete_candle_is_not_persisted_or_cached_as_complete(tmp_path: Path) -> None:
    candle_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal candle_calls
        body = json.loads(request.content)
        if request.url.path.endswith("InstrumentsService/FindInstrument"):
            ticker = body["query"]
            return httpx.Response(
                200, json={"instruments": [_instrument_payload(ticker, f"uid-{ticker}")]}
            )
        if request.url.path.endswith("InstrumentsService/Indicatives"):
            return httpx.Response(200, json={"instruments": []})
        if request.url.path.endswith("InstrumentsService/GetInstrumentBy"):
            uid = str(body["id"])
            ticker = uid.removeprefix("uid-")
            return httpx.Response(200, json={"instrument": _instrument_payload(ticker, uid)})
        if request.url.path.endswith("MarketDataService/GetCandles"):
            candle_calls += 1
            candle = _candle_payload(date(2020, 1, 2), 100.0)
            candle["isComplete"] = False
            return httpx.Response(200, json={"candles": [candle]})
        raise AssertionError(request.url.path)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TInvestReadOnlyClient(
            token="secret", contour=TInvestContour.READONLY_PRODUCTION, client=http_client
        )
        first = await acquire_history(
            client,
            raw_dir=tmp_path,
            date_from=date(2020, 1, 1),
            date_to=date(2020, 1, 3),
            tickers=("SBER",),
            resolved_at=datetime(2026, 8, 13, tzinfo=UTC),
        )
        await acquire_history(
            client,
            raw_dir=tmp_path,
            date_from=date(2020, 1, 1),
            date_to=date(2020, 1, 3),
            tickers=("SBER",),
            resolved_at=datetime(2026, 8, 13, tzinfo=UTC),
        )
    assert first.security_bars["SBER"] == ()
    assert candle_calls == 2
    checkpoint = json.loads((tmp_path / "checkpoints/SBER.json").read_text(encoding="utf-8"))
    assert checkpoint["complete"] is False


async def test_ambiguous_security_fails_closed_without_blocking_other_tickers(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("InstrumentsService/FindInstrument"):
            ticker = body["query"]
            instruments = [_instrument_payload(ticker, f"uid-{ticker}")]
            if ticker == "SBER":
                instruments.append(_instrument_payload(ticker, "uid-SBER-duplicate"))
            return httpx.Response(200, json={"instruments": instruments})
        if request.url.path.endswith("InstrumentsService/Indicatives"):
            return httpx.Response(200, json={"instruments": []})
        if request.url.path.endswith("InstrumentsService/GetInstrumentBy"):
            uid = str(body["id"])
            ticker = uid.removeprefix("uid-")
            return httpx.Response(200, json={"instrument": _instrument_payload(ticker, uid)})
        if request.url.path.endswith("MarketDataService/GetCandles"):
            return httpx.Response(200, json={"candles": [_candle_payload(date(2020, 1, 2), 100.0)]})
        raise AssertionError(request.url.path)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = TInvestReadOnlyClient(
            token="secret", contour=TInvestContour.READONLY_PRODUCTION, client=http_client
        )
        result = await acquire_history(
            client,
            raw_dir=tmp_path,
            date_from=date(2020, 1, 1),
            date_to=date(2020, 1, 3),
            tickers=("SBER", "GAZP"),
            resolved_at=datetime(2026, 8, 13, tzinfo=UTC),
        )
    assert set(result.security_bars) == {"GAZP"}
    assert result.manifest["unresolved_security_tickers"] == ["SBER"]
    diagnostics = cast("list[dict[str, object]]", result.manifest["resolution_diagnostics"])
    assert any(item["ticker"] == "SBER" for item in diagnostics)


def test_dataset_is_deterministic_and_targets_are_separate() -> None:
    securities, benchmark = _history()
    first = build_dataset(securities, benchmark)
    second = build_dataset(
        {key: tuple(reversed(value)) for key, value in securities.items()},
        tuple(reversed(benchmark)),
    )
    assert first.dataset_sha == second.dataset_sha
    assert first.feature_schema_sha == second.feature_schema_sha
    assert [item.row_id for item in first.features] == [item.row_id for item in first.targets]
    assert FEATURE_DATASET_VERSION == "tinvest-market-baseline-features-v1"
    assert RAW_DATASET_VERSION == "tinvest-market-raw-v1"
    assert all("next_session_return" not in item.values for item in first.features)


def test_duplicate_ticker_date_is_rejected() -> None:
    securities, benchmark = _history()
    duplicated = securities["SBER"] + (securities["SBER"][0],)
    with pytest.raises(ValueError, match="duplicate SBER"):
        build_dataset({"SBER": duplicated}, benchmark)


def test_target_day_cannot_change_features() -> None:
    securities, benchmark = _history()
    baseline = build_dataset(securities, benchmark)
    feature = baseline.features[3]
    rows = list(securities[feature.ticker])
    index = next(i for i, item in enumerate(rows) if item.trade_date == feature.trade_date)
    rows[index] = replace(
        rows[index],
        open=rows[index].open * 2,
        high=rows[index].high * 2,
        low=rows[index].low * 2,
        close=rows[index].close * 2,
        volume=rows[index].volume * 100,
    )
    changed = build_dataset({feature.ticker: tuple(rows)}, benchmark)
    changed_feature = next(item for item in changed.features if item.row_id == feature.row_id)
    changed_target = next(item for item in changed.targets if item.row_id == feature.row_id)
    original_target = next(item for item in baseline.targets if item.row_id == feature.row_id)
    assert changed_feature.values == feature.values
    assert changed_target.next_session_return != original_target.next_session_return


def test_no_forward_fill_no_synthetic_sessions_and_no_uid_join() -> None:
    securities, benchmark = _history(sessions=90)
    missing = securities["SBER"][40].trade_date
    shortened = tuple(item for item in securities["SBER"] if item.trade_date != missing)
    result = build_dataset({"SBER": shortened}, benchmark)
    assert result.quality["prices_forward_filled"] is False
    assert result.quality["synthetic_market_rows"] == 0
    assert not any(item.trade_date == missing for item in result.features)
    mixed = list(shortened)
    mixed[-1] = replace(mixed[-1], instrument_uid="different-uid")
    with pytest.raises(ValueError, match="multiple historical identities"):
        build_dataset({"SBER": tuple(mixed)}, benchmark)


def test_imoex_is_optional_and_moex_is_never_used_as_fallback() -> None:
    securities, _ = _history()
    result = build_dataset(securities, None)
    assert result.features
    assert all(not item.benchmark_available for item in result.features)
    assert all(item.next_session_abnormal_return is None for item in result.targets)
    assert result.quality["moex_rows_used"] == 0
    assert result.quality["benchmark_source"] is None


def test_abnormal_return_uses_tinvest_imoex_same_session() -> None:
    securities, benchmark = _history()
    result = build_dataset(securities, benchmark)
    target = result.targets[0]
    security = {item.trade_date: item for item in securities[target.ticker]}
    imoex = {item.trade_date: item for item in benchmark}
    security_return = (
        security[target.trade_date].close / security[target.baseline_trade_date].close - 1
    )
    benchmark_return = imoex[target.trade_date].close / imoex[target.baseline_trade_date].close - 1
    assert math.isclose(target.next_session_return, security_return)
    assert math.isclose(target.imoex_next_session_return or 0.0, benchmark_return)
    assert math.isclose(
        target.next_session_abnormal_return or 0.0, security_return - benchmark_return
    )


def test_benchmark_alignment_does_not_create_permanent_weekend_cutoff() -> None:
    start = date(2025, 2, 1)
    security = tuple(_bar("SBER", start + timedelta(days=i), 100 + i, i) for i in range(100))
    benchmark = tuple(
        _bar("IMOEX", start + timedelta(days=i), 1000 + i, i)
        for i in range(100)
        if (start + timedelta(days=i)).weekday() < 5
    )
    result = build_dataset({"SBER": security}, benchmark)
    aligned_dates = {item.trade_date for item in benchmark}
    assert result.features[-1].trade_date == benchmark[-1].trade_date
    assert all(item.trade_date in aligned_dates for item in result.features)
    attrition = cast("dict[str, dict[str, object]]", result.quality["row_attrition"])["SBER"]
    losses = cast("dict[str, int]", attrition["rows_lost"])
    assert losses == {
        "warmup": FEATURE_WINDOW_SESSIONS + 1,
        "missing_lag_history": 0,
        "benchmark_alignment": len(security) - len(benchmark),
        "target_tail": 0,
        "other": 0,
    }
    assert attrition["reconciled"] is True


def test_extremes_are_retained_without_clipping_or_winsorization() -> None:
    securities, benchmark = _history()
    rows = list(securities["SBER"])
    rows[35] = replace(
        rows[35],
        open=rows[35].open * 4,
        high=rows[35].high * 4,
        low=rows[35].low * 4,
        close=rows[35].close * 4,
    )
    result = build_dataset({"SBER": tuple(rows)}, benchmark)
    assert int(str(result.price_audit["count_abs_return_gt_50pct"])) >= 2
    assert result.price_audit["rows_removed_by_audit"] == 0
    assert result.price_audit["targets_clipped"] is False
    assert result.price_audit["targets_winsorized"] is False
    ticker_statistics = cast(
        "dict[str, dict[str, object]]", result.price_audit["ticker_statistics"]
    )
    statistics = ticker_statistics["SBER"]
    assert set(statistics) == {"count", "min", "max", "p0.1", "p1", "p99", "p99.9"}
    reviewed = cast("list[dict[str, object]]", result.price_audit["review_observations"])
    assert len(reviewed) >= 2
    assert all(item["classification"] in {"MARKET_MOVE", "UNRESOLVED"} for item in reviewed)


def test_temporal_split_is_grouped_purged_embargoed_and_deterministic() -> None:
    securities, benchmark = _history(tickers=("SBER", "GAZP"), sessions=130)
    dataset = build_dataset(securities, benchmark)
    config = SplitConfig(purge_sessions=2, embargo_sessions=2)
    split = temporal_split(dataset.features, config)
    repeated = temporal_split(tuple(reversed(dataset.features)), config)
    assert split.split_sha == repeated.split_sha
    by_date: dict[date, set[SplitName]] = {}
    rows = {item.row_id: item for item in dataset.features}
    for row_id, name in split.assignments.items():
        by_date.setdefault(rows[row_id].trade_date, set()).add(name)
    assert all(len(names) == 1 for names in by_date.values())
    assert split.purged_row_ids and split.embargoed_row_ids
    assert len(split.purged_dates) == 4
    assert len(split.embargoed_dates) == 4


def test_reports_keep_sources_separate_and_moex_comparison_diagnostic(tmp_path: Path) -> None:
    securities, benchmark = _history(tickers=("SBER", "GAZP"), sessions=100)
    result = build_dataset(securities, benchmark)
    split = temporal_split(result.features)
    moex_path = tmp_path / "moex-targets.jsonl"
    moex_path.write_text(
        json.dumps(
            {
                "row_id": result.targets[0].row_id,
                "next_session_return": result.targets[0].next_session_return,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = write_feature_artifacts(
        tmp_path / "features",
        result=result,
        split=split,
        acquisition_manifest={"dataset_sha": "raw-sha", "instrument_mapping_sha": "mapping-sha"},
        git_sha="git-sha",
        event_daily_feature_ready=34,
        moex_targets_path=moex_path,
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    diagnostic = json.loads(paths["moex_overlap"].read_text(encoding="utf-8"))
    assert manifest["source"] == SOURCE
    assert manifest["dataset_semantics"]["moex_data_used"] is False
    assert manifest["feature_cutoff_audit"] == {
        "cause": "ROLLING_WINDOWS_BUILT_BEFORE_BENCHMARK_SESSION_INTERSECTION",
        "was_bug": True,
        "alignment_policy": "COMMON_REAL_SESSIONS_NO_FORWARD_FILL",
    }
    assert manifest["model_trained"] is False
    assert diagnostic["overlap_rows"] == 1
    assert diagnostic["missing_in_moex_rows"] == len(result.targets) - 1
    assert diagnostic["moex_rows"] == 1
    assert diagnostic["data_sources_combined"] is False
    assert diagnostic["t_invest_filled_from_moex"] is False
    assert compare_moex_targets(result, None)["status"] == "MOEX_DIAGNOSTIC_UNAVAILABLE"


def test_event_rules_qwen_and_modeling_remain_frozen() -> None:
    assert rules_v3_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert DEFAULT_OLLAMA_MODEL == "qwen3.5:9b"
    source = "".join(
        path.read_text(encoding="utf-8") for path in Path("src/tinvest_market").glob("*.py")
    )
    assert "train_predictive" not in source
    assert "src.events" not in source
    assert "generate_signal" not in source
    assert "submit_order" not in source


def _instrument(ticker: str, uid: str) -> TInvestInstrument:
    return TInvestInstrument(
        ticker, "TQBR", uid, f"figi-{uid}", "INSTRUMENT_TYPE_SHARE", date(2010, 1, 1), ticker
    )


def _instrument_payload(
    ticker: str, uid: str, kind: str = "INSTRUMENT_TYPE_SHARE"
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "classCode": "TQBR" if ticker != "IMOEX" else "INDX",
        "uid": uid,
        "figi": f"figi-{uid}",
        "instrumentType": kind,
        "first1dayCandleDate": "2020-01-01T00:00:00Z",
        "name": ticker,
    }


def _candle_payload(trade_date: date, close: float) -> dict[str, object]:
    return {
        "open": _quotation(close - 0.2),
        "high": _quotation(close + 0.5),
        "low": _quotation(close - 0.5),
        "close": _quotation(close),
        "volume": "1000",
        "time": f"{trade_date.isoformat()}T00:00:00Z",
        "isComplete": True,
    }


def _quotation(value: float) -> dict[str, int]:
    units = int(value)
    return {"units": units, "nano": round((value - units) * 1_000_000_000)}


def _read_tree(root: Path) -> str:
    return "".join(path.read_text(encoding="utf-8") for path in root.rglob("*.*"))


def _history(
    *, tickers: tuple[str, ...] = ("SBER",), sessions: int = 90
) -> tuple[dict[str, tuple[DailyBar, ...]], tuple[DailyBar, ...]]:
    start = date(2020, 1, 1)
    benchmark = tuple(
        _bar("IMOEX", start + timedelta(days=i), 1000 + i * 2, i) for i in range(sessions)
    )
    securities = {
        ticker: tuple(
            _bar(ticker, start + timedelta(days=i), 100 + i * 0.7 + offset, i)
            for i in range(sessions)
        )
        for offset, ticker in enumerate(tickers)
    }
    return securities, benchmark


def _bar(ticker: str, trade_date: date, close: float, index: int) -> DailyBar:
    return DailyBar(
        ticker,
        f"uid-{ticker}",
        trade_date,
        close - 0.2,
        close + 0.5,
        close - 0.5,
        close,
        1000 + index,
        True,
    )
