from __future__ import annotations

import asyncio

from httpx import AsyncClient


def payload(**overrides: str) -> dict[str, str]:
    data = {
        "source_id": "event-source-001",
        "source_name": "Event News",
        "source_url": "https://example.com/event/1",
        "title": "Company published financial results",
        "raw_content": "Выручка за 2025 год составила 1,2 млрд руб., EBITDA выросла на 15% г/г.",
        "language": "ru",
        "published_at": "2026-08-06T08:00:00+05:00",
        "received_at": "2026-08-06T08:00:01+05:00",
    }
    data.update(overrides)
    return data


async def test_analyze_event_persists_and_gets_result(client: AsyncClient) -> None:
    created = await client.post("/api/v1/news", json=payload())
    news_id = created.json()["id"]

    analyzed = await client.post(f"/api/v1/news/{news_id}/analyze-event?debug=true")
    fetched = await client.get(f"/api/v1/news/{news_id}/event-analysis")

    assert analyzed.status_code == 200
    body = analyzed.json()
    assert body["analysis_version"] == "event-rules-v1"
    assert body["primary_event_type"] == "FINANCIAL_RESULTS"
    assert body["debug"]["rules_version"] == "event-rules-v1"
    assert {fact["metric"] for fact in body["financial_facts"]} >= {"REVENUE", "EBITDA"}
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]


async def test_get_event_analysis_before_analyze_returns_404(client: AsyncClient) -> None:
    created = await client.post(
        "/api/v1/news",
        json=payload(source_id="event-source-002", source_url="https://example.com/event/2"),
    )

    response = await client.get(f"/api/v1/news/{created.json()['id']}/event-analysis")

    assert response.status_code == 404


async def test_analyze_event_replaces_same_version_idempotently(client: AsyncClient) -> None:
    created = await client.post(
        "/api/v1/news",
        json=payload(source_id="event-source-003", source_url="https://example.com/event/3"),
    )
    news_id = created.json()["id"]

    first = await client.post(f"/api/v1/news/{news_id}/analyze-event")
    second = await client.post(f"/api/v1/news/{news_id}/analyze-event")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["analysis_version"] == "event-rules-v1"
    assert len(second.json()["financial_facts"]) == len(first.json()["financial_facts"])


async def test_concurrent_event_analysis_does_not_create_conflicting_results(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/v1/news",
        json=payload(source_id="event-source-004", source_url="https://example.com/event/4"),
    )
    news_id = created.json()["id"]

    responses = await asyncio.gather(
        *(client.post(f"/api/v1/news/{news_id}/analyze-event") for _ in range(5))
    )
    fetched = await client.get(f"/api/v1/news/{news_id}/event-analysis")

    assert [response.status_code for response in responses] == [200, 200, 200, 200, 200]
    assert fetched.status_code == 200
    assert fetched.json()["analysis_version"] == "event-rules-v1"
