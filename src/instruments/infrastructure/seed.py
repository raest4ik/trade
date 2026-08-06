from __future__ import annotations

from dataclasses import dataclass

from src.instruments.domain.enums import AliasType, InstrumentType


@dataclass(frozen=True, slots=True)
class SeedAlias:
    alias: str
    alias_type: AliasType
    priority: int = 100


@dataclass(frozen=True, slots=True)
class SeedInstrument:
    ticker: str
    short_name: str
    full_name: str
    issuer_name: str
    instrument_type: InstrumentType
    aliases: tuple[SeedAlias, ...]


SEED_INSTRUMENTS: tuple[SeedInstrument, ...] = (
    SeedInstrument(
        ticker="SBER",
        short_name="Сбербанк",
        full_name="ПАО Сбербанк, обыкновенная акция",
        issuer_name="ПАО Сбербанк",
        instrument_type=InstrumentType.COMMON_STOCK,
        aliases=(
            SeedAlias("SBER", AliasType.TICKER, 10),
            SeedAlias("Сбербанк", AliasType.OFFICIAL_NAME, 20),
            SeedAlias("ПАО Сбербанк", AliasType.LEGAL_NAME, 30),
            SeedAlias("Сбер", AliasType.BRAND, 40),
            SeedAlias("обыкновенная акция Сбербанк", AliasType.MANUAL, 50),
        ),
    ),
    SeedInstrument(
        ticker="SBERP",
        short_name="Сбербанк ап",
        full_name="ПАО Сбербанк, привилегированная акция",
        issuer_name="ПАО Сбербанк",
        instrument_type=InstrumentType.PREFERRED_STOCK,
        aliases=(
            SeedAlias("SBERP", AliasType.TICKER, 10),
            SeedAlias("Сбербанк", AliasType.OFFICIAL_NAME, 20),
            SeedAlias("ПАО Сбербанк", AliasType.LEGAL_NAME, 30),
            SeedAlias("Сбер", AliasType.BRAND, 40),
            SeedAlias("привилегированная акция Сбербанк", AliasType.MANUAL, 50),
        ),
    ),
    SeedInstrument(
        ticker="GAZP",
        short_name="Газпром",
        full_name="ПАО Газпром",
        issuer_name="ПАО Газпром",
        instrument_type=InstrumentType.COMMON_STOCK,
        aliases=(
            SeedAlias("GAZP", AliasType.TICKER, 10),
            SeedAlias("Газпром", AliasType.OFFICIAL_NAME, 20),
            SeedAlias('ПАО "Газпром"', AliasType.LEGAL_NAME, 30),
            SeedAlias("ПАО Газпром", AliasType.LEGAL_NAME, 30),
        ),
    ),
    SeedInstrument(
        ticker="LKOH",
        short_name="ЛУКОЙЛ",
        full_name="ПАО ЛУКОЙЛ",
        issuer_name="ПАО ЛУКОЙЛ",
        instrument_type=InstrumentType.COMMON_STOCK,
        aliases=(
            SeedAlias("LKOH", AliasType.TICKER, 10),
            SeedAlias("ЛУКОЙЛ", AliasType.OFFICIAL_NAME, 20),
            SeedAlias("Лукойл", AliasType.BRAND, 30),
            SeedAlias("ПАО ЛУКОЙЛ", AliasType.LEGAL_NAME, 40),
        ),
    ),
    SeedInstrument(
        ticker="ROSN",
        short_name="Роснефть",
        full_name="ПАО НК Роснефть",
        issuer_name="ПАО НК Роснефть",
        instrument_type=InstrumentType.COMMON_STOCK,
        aliases=(
            SeedAlias("ROSN", AliasType.TICKER, 10),
            SeedAlias("Роснефть", AliasType.OFFICIAL_NAME, 20),
            SeedAlias("ПАО НК Роснефть", AliasType.LEGAL_NAME, 30),
        ),
    ),
    SeedInstrument(
        ticker="NVTK",
        short_name="НОВАТЭК",
        full_name="ПАО НОВАТЭК",
        issuer_name="ПАО НОВАТЭК",
        instrument_type=InstrumentType.COMMON_STOCK,
        aliases=(
            SeedAlias("NVTK", AliasType.TICKER, 10),
            SeedAlias("НОВАТЭК", AliasType.OFFICIAL_NAME, 20),
            SeedAlias("Новатэк", AliasType.BRAND, 30),
            SeedAlias("ПАО НОВАТЭК", AliasType.LEGAL_NAME, 40),
        ),
    ),
    SeedInstrument(
        ticker="YDEX",
        short_name="Яндекс",
        full_name="МКПАО Яндекс",
        issuer_name="МКПАО Яндекс",
        instrument_type=InstrumentType.COMMON_STOCK,
        aliases=(
            SeedAlias("YDEX", AliasType.TICKER, 10),
            SeedAlias("Яндекс", AliasType.BRAND, 20),
            SeedAlias("МКПАО Яндекс", AliasType.LEGAL_NAME, 30),
        ),
    ),
    SeedInstrument(
        ticker="T",
        short_name="Т-Технологии",
        full_name="МКПАО Т-Технологии",
        issuer_name="МКПАО Т-Технологии",
        instrument_type=InstrumentType.COMMON_STOCK,
        aliases=(
            SeedAlias("T", AliasType.TICKER, 10),
            SeedAlias("Т-Технологии", AliasType.OFFICIAL_NAME, 20),
            SeedAlias("МКПАО Т-Технологии", AliasType.LEGAL_NAME, 30),
        ),
    ),
    SeedInstrument(
        ticker="VTBR",
        short_name="ВТБ",
        full_name="Банк ВТБ",
        issuer_name="Банк ВТБ",
        instrument_type=InstrumentType.COMMON_STOCK,
        aliases=(
            SeedAlias("VTBR", AliasType.TICKER, 10),
            SeedAlias("ВТБ", AliasType.OFFICIAL_NAME, 20),
            SeedAlias("Банк ВТБ", AliasType.LEGAL_NAME, 30),
        ),
    ),
    SeedInstrument(
        ticker="GMKN",
        short_name="Норильский никель",
        full_name="ПАО ГМК Норильский никель",
        issuer_name="ПАО ГМК Норильский никель",
        instrument_type=InstrumentType.COMMON_STOCK,
        aliases=(
            SeedAlias("GMKN", AliasType.TICKER, 10),
            SeedAlias("Норильский никель", AliasType.OFFICIAL_NAME, 20),
            SeedAlias("ПАО ГМК Норильский никель", AliasType.LEGAL_NAME, 30),
            SeedAlias("ГМК Норильский никель", AliasType.SHORT_NAME, 40),
        ),
    ),
)
