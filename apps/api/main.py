from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import cast

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src.instruments.presentation.routes import router as instruments_router
from src.news.presentation.routes import router as news_router
from src.shared.config.settings import Settings, get_settings
from src.shared.database.session import create_engine, create_session_factory
from src.shared.logging.middleware import RequestLoggingMiddleware
from src.shared.logging.setup import configure_logging


def create_app(
    settings: Settings | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        yield
        engine = request_engine(app)
        if engine is not None:
            await engine.dispose()

    app = FastAPI(title=resolved_settings.app_name, lifespan=lifespan)
    app.state.settings = resolved_settings
    if session_factory is None:
        engine = create_engine(resolved_settings.database_url)
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
    else:
        app.state.engine = None
        app.state.session_factory = session_factory

    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(news_router, prefix="/api/v1")
    app.include_router(instruments_router, prefix="/api/v1")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> Response:
        factory = cast(
            Callable[[], AbstractAsyncContextManager[AsyncSession]],
            request.app.state.session_factory,
        )
        try:
            async with factory() as session:
                await session.execute(text("SELECT 1"))
        except SQLAlchemyError:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "unavailable"},
            )
        return JSONResponse(content={"status": "ready"})

    return app


def request_engine(app: FastAPI) -> AsyncEngine | None:
    return cast("AsyncEngine | None", app.state.engine)


app = create_app()
