from __future__ import annotations

from types import TracebackType

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import SQLAlchemyError

from apps.api.main import create_app
from src.shared.config.settings import Settings


class BrokenSession:
    async def __aenter__(self) -> BrokenSession:
        raise SQLAlchemyError("database is down")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


def broken_session_factory() -> BrokenSession:
    return BrokenSession()


async def test_ready_returns_503_when_database_is_unavailable() -> None:
    app: FastAPI = create_app(Settings(database_url="sqlite+aiosqlite:///unused"))
    app.state.session_factory = broken_session_factory
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
