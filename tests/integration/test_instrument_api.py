from __future__ import annotations

import asyncio
from uuid import uuid4

from httpx import AsyncClient


async def create_instrument(
    client: AsyncClient,
    ticker: str,
    *,
    issuer_name: str | None = None,
    instrument_type: str = "COMMON_STOCK",
) -> str:
    response = await client.post(
        "/api/v1/instruments",
        json={
            "ticker": ticker,
            "short_name": issuer_name or ticker,
            "full_name": issuer_name or ticker,
            "issuer_name": issuer_name or ticker,
            "exchange": "MOEX",
            "currency": "RUB",
            "instrument_type": instrument_type,
        },
    )
    assert response.status_code == 201
    assert response.json()["ticker"] == ticker.upper()
    return response.json()["id"]


async def add_alias(
    client: AsyncClient,
    instrument_id: str,
    alias: str,
    alias_type: str,
    *,
    priority: int = 100,
) -> None:
    response = await client.post(
        f"/api/v1/instruments/{instrument_id}/aliases",
        json={"alias": alias, "alias_type": alias_type, "priority": priority},
    )
    assert response.status_code == 201


async def create_news(client: AsyncClient, raw_content: str) -> str:
    unique = uuid4()
    response = await client.post(
        "/api/v1/news",
        json={
            "source_id": f"source-{unique}",
            "source_name": "Test News",
            "source_url": f"https://example.com/news/{unique}",
            "title": "Market news",
            "raw_content": raw_content,
            "language": "ru",
            "published_at": "2026-08-06T08:00:00Z",
            "received_at": "2026-08-06T08:00:01Z",
        },
    )
    assert response.status_code == 201
    assert response.json()["raw_content"] == raw_content
    return response.json()["id"]


async def seed_sber_pair(client: AsyncClient) -> tuple[str, str]:
    sber_id = await create_instrument(client, "SBER", issuer_name="ПАО Сбербанк")
    sberp_id = await create_instrument(
        client,
        "SBERP",
        issuer_name="ПАО Сбербанк",
        instrument_type="PREFERRED_STOCK",
    )
    await add_alias(client, sber_id, "SBER", "TICKER", priority=10)
    await add_alias(client, sberp_id, "SBERP", "TICKER", priority=10)
    await add_alias(client, sber_id, "Сбербанк", "OFFICIAL_NAME")
    await add_alias(client, sberp_id, "Сбербанк", "OFFICIAL_NAME")
    return sber_id, sberp_id


async def test_post_and_list_instruments(client: AsyncClient) -> None:
    instrument_id = await create_instrument(client, "gazp", issuer_name="ПАО Газпром")

    response = await client.get("/api/v1/instruments")

    assert response.status_code == 200
    assert response.json()[0]["id"] == instrument_id
    assert response.json()[0]["ticker"] == "GAZP"


async def test_sber_ticker_endpoint_match_is_not_ambiguous(client: AsyncClient) -> None:
    await seed_sber_pair(client)
    news_id = await create_news(client, "SBER вырос после отчета")

    response = await client.post(f"/api/v1/news/{news_id}/match-instruments")

    assert response.status_code == 200
    body = response.json()
    assert body["matcher_version"] == "deterministic-v1"
    assert [(match["matched_alias"], match["is_ambiguous"]) for match in body["matches"]] == [
        ("SBER", False)
    ]


async def test_sberp_does_not_create_sber_match(client: AsyncClient) -> None:
    await seed_sber_pair(client)
    news_id = await create_news(client, "SBERP выросла")

    response = await client.post(f"/api/v1/news/{news_id}/match-instruments")

    assert response.status_code == 200
    assert [match["matched_alias"] for match in response.json()["matches"]] == ["SBERP"]


async def test_sberbank_alias_is_ambiguous_between_common_and_preferred(
    client: AsyncClient,
) -> None:
    await seed_sber_pair(client)
    news_id = await create_news(client, "Сбербанк раскрыл финансовые результаты")

    response = await client.post(f"/api/v1/news/{news_id}/match-instruments")

    assert response.status_code == 200
    matches = response.json()["matches"]
    assert len(matches) == 2
    assert {match["is_ambiguous"] for match in matches} == {True}


async def test_gazprom_legal_name_and_positions(client: AsyncClient) -> None:
    gazp_id = await create_instrument(client, "GAZP", issuer_name="ПАО Газпром")
    await add_alias(client, gazp_id, "Газпром", "OFFICIAL_NAME")
    await add_alias(client, gazp_id, 'ПАО "Газпром"', "LEGAL_NAME")
    raw_content = 'Сегодня ПАО "Газпром" сообщило результаты'
    news_id = await create_news(client, raw_content)

    response = await client.post(f"/api/v1/news/{news_id}/match-instruments")

    assert response.status_code == 200
    match = response.json()["matches"][0]
    assert match["instrument_id"] == gazp_id
    assert raw_content[match["start_position"] : match["end_position"]] == "Газпром"


async def test_one_news_can_match_multiple_issuers(client: AsyncClient) -> None:
    gazp_id = await create_instrument(client, "GAZP", issuer_name="ПАО Газпром")
    lkoh_id = await create_instrument(client, "LKOH", issuer_name="ПАО ЛУКОЙЛ")
    await add_alias(client, gazp_id, "Газпром", "OFFICIAL_NAME")
    await add_alias(client, lkoh_id, "ЛУКОЙЛ", "OFFICIAL_NAME")
    news_id = await create_news(client, "Газпром и ЛУКОЙЛ опубликовали отчеты")

    response = await client.post(f"/api/v1/news/{news_id}/match-instruments")

    assert response.status_code == 200
    assert {match["instrument_id"] for match in response.json()["matches"]} == {gazp_id, lkoh_id}


async def test_repeated_endpoint_call_is_idempotent(client: AsyncClient) -> None:
    gazp_id = await create_instrument(client, "GAZP", issuer_name="ПАО Газпром")
    await add_alias(client, gazp_id, "Газпром", "OFFICIAL_NAME")
    news_id = await create_news(client, "Газпром сообщил. Газпром подтвердил.")

    first = await client.post(f"/api/v1/news/{news_id}/match-instruments")
    second = await client.post(f"/api/v1/news/{news_id}/match-instruments")
    stored = await client.get(f"/api/v1/news/{news_id}/instruments")

    assert first.status_code == 200
    assert second.status_code == 200
    assert stored.status_code == 200
    assert len(second.json()["matches"]) == 1
    assert len(stored.json()["matches"]) == 1


async def test_concurrent_repeated_endpoint_call_does_not_duplicate(client: AsyncClient) -> None:
    gazp_id = await create_instrument(client, "GAZP", issuer_name="ПАО Газпром")
    await add_alias(client, gazp_id, "Газпром", "OFFICIAL_NAME")
    news_id = await create_news(client, "Газпром сообщил результаты")

    responses = await asyncio.gather(
        *(client.post(f"/api/v1/news/{news_id}/match-instruments") for _ in range(3))
    )
    stored = await client.get(f"/api/v1/news/{news_id}/instruments")

    assert all(response.status_code == 200 for response in responses)
    assert len(stored.json()["matches"]) == 1


async def test_unknown_news_returns_404(client: AsyncClient) -> None:
    response = await client.post(f"/api/v1/news/{uuid4()}/match-instruments")

    assert response.status_code == 404


async def test_invalid_uuid_is_handled_by_fastapi(client: AsyncClient) -> None:
    response = await client.post("/api/v1/news/not-a-uuid/match-instruments")

    assert response.status_code == 422


async def test_short_non_ticker_alias_returns_422(client: AsyncClient) -> None:
    instrument_id = await create_instrument(client, "TEST", issuer_name="Test")

    response = await client.post(
        f"/api/v1/instruments/{instrument_id}/aliases",
        json={"alias": "Т", "alias_type": "BRAND"},
    )

    assert response.status_code == 422
