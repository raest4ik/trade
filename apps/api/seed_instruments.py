from __future__ import annotations

import asyncio

from src.instruments.domain.entities import Instrument, IssuerAlias
from src.instruments.infrastructure.repositories import SqlAlchemyInstrumentRepository
from src.instruments.infrastructure.seed import SEED_INSTRUMENTS
from src.shared.config.settings import get_settings
from src.shared.database.session import create_engine, create_session_factory


async def seed() -> None:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        repository = SqlAlchemyInstrumentRepository(session)
        for seed_instrument in SEED_INSTRUMENTS:
            result = await repository.save_instrument(
                Instrument.create(
                    ticker=seed_instrument.ticker,
                    figi=None,
                    isin=None,
                    short_name=seed_instrument.short_name,
                    full_name=seed_instrument.full_name,
                    issuer_name=seed_instrument.issuer_name,
                    exchange="MOEX",
                    currency="RUB",
                    instrument_type=seed_instrument.instrument_type,
                )
            )
            for seed_alias in seed_instrument.aliases:
                await repository.save_alias(
                    IssuerAlias.create(
                        instrument_id=result.instrument.id,
                        alias=seed_alias.alias,
                        alias_type=seed_alias.alias_type,
                        priority=seed_alias.priority,
                    )
                )
    await engine.dispose()


def main() -> None:
    asyncio.run(seed())


if __name__ == "__main__":
    main()
