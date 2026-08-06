from __future__ import annotations

import asyncio
from uuid import uuid4

from httpx import AsyncClient


def payload(**overrides: str) -> dict[str, str]:
    data = {
        "source_id": "test-source-001",
        "source_name": "Test News",
        "source_url": "https://example.com/news/1",
        "title": "Company published financial results",
        "raw_content": "Original publication text.",
        "language": "en",
        "published_at": "2026-08-06T08:00:00+05:00",
        "received_at": "2026-08-06T08:00:01+05:00",
    }
    data.update(overrides)
    return data


async def test_create_news_returns_created_item(client: AsyncClient) -> None:
    response = await client.post("/api/v1/news", json=payload())

    assert response.status_code == 201
    body = response.json()
    assert body["id"]
    assert body["raw_content"] == "Original publication text."
    assert body["raw_content_hash"]


async def test_get_news_by_uuid(client: AsyncClient) -> None:
    created = await client.post("/api/v1/news", json=payload())
    news_id = created.json()["id"]

    response = await client.get(f"/api/v1/news/{news_id}")

    assert response.status_code == 200
    assert response.json()["id"] == news_id


async def test_get_unknown_news_returns_404(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/news/{uuid4()}")

    assert response.status_code == 404


async def test_duplicate_news_returns_existing_item(client: AsyncClient) -> None:
    first = await client.post("/api/v1/news", json=payload())
    second = await client.post("/api/v1/news", json=payload())

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]


async def test_concurrent_duplicate_news_returns_single_item(client: AsyncClient) -> None:
    responses = await asyncio.gather(
        *(client.post("/api/v1/news", json=payload()) for _ in range(5))
    )

    ids = {response.json()["id"] for response in responses}
    statuses = sorted(response.status_code for response in responses)
    assert len(ids) == 1
    assert statuses == [200, 200, 200, 200, 201]


async def test_same_title_different_news_are_distinct(client: AsyncClient) -> None:
    first = await client.post("/api/v1/news", json=payload())
    second = await client.post(
        "/api/v1/news",
        json=payload(
            source_url="https://example.com/news/2",
            raw_content="Different original publication text.",
        ),
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]


async def test_empty_raw_content_returns_422(client: AsyncClient) -> None:
    response = await client.post("/api/v1/news", json=payload(raw_content="   "))

    assert response.status_code == 422


async def test_invalid_url_returns_422(client: AsyncClient) -> None:
    response = await client.post("/api/v1/news", json=payload(source_url="not-a-url"))

    assert response.status_code == 422


async def test_missing_published_at_returns_422(client: AsyncClient) -> None:
    data = payload()
    del data["published_at"]

    response = await client.post("/api/v1/news", json=data)

    assert response.status_code == 422


async def test_timestamps_are_returned_in_utc(client: AsyncClient) -> None:
    response = await client.post("/api/v1/news", json=payload())

    assert response.status_code == 201
    assert response.json()["published_at"] == "2026-08-06T03:00:00Z"
    assert response.json()["received_at"] == "2026-08-06T03:00:01Z"


async def test_ready_returns_ok_when_database_is_available(client: AsyncClient) -> None:
    response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
