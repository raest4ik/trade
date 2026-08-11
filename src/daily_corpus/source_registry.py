from __future__ import annotations

from src.daily_corpus.domain import (
    SourceAcceptanceStatus,
    SourceBlocker,
    SourceVerification,
)


def daily_source_verifications() -> tuple[SourceVerification, ...]:
    records = (
        _blocked(
            "SBER_IR_ARCHIVE",
            ("SBER", "SBERP"),
            "https://www.sberbank.com/ru/investor-relations",
            estimated=0,
            sampled=0,
            accessible=0,
            dated=0,
            blockers=(SourceBlocker.ACCESS_POLICY, SourceBlocker.TECHNICAL_PARSE),
            note="Controlled access did not expose a stable machine-readable archive contract.",
        ),
        _blocked(
            "GAZPROM_PRESS_ARCHIVE",
            ("GAZP",),
            "https://www.gazprom.com/press/news/",
            estimated=5000,
            sampled=0,
            accessible=0,
            dated=0,
            blockers=(SourceBlocker.ACCESS_POLICY, SourceBlocker.ROBOTS),
            note=(
                "Official pages show timestamps, but automated archive access policy "
                "remains unverified."
            ),
        ),
        _blocked(
            "LUKOIL_PRESS_ARCHIVE",
            ("LKOH",),
            "https://www.lukoil.com/PressCenter/Pressreleases",
            estimated=2480,
            sampled=20,
            accessible=20,
            dated=20,
            blockers=(SourceBlocker.ACCESS_POLICY, SourceBlocker.STORAGE_POLICY),
            note=(
                "A deterministic first-page sample has dates and stable links; "
                "reuse/storage terms are not verified."
            ),
        ),
        _blocked(
            "NOVATEK_PRESS_ARCHIVE",
            ("NVTK",),
            "https://www.novatek.ru/ru/press/releases/",
            estimated=500,
            sampled=20,
            accessible=20,
            dated=20,
            blockers=(SourceBlocker.ACCESS_POLICY, SourceBlocker.STORAGE_POLICY),
            note=(
                "Official date-only archive is visible; automation and storage permissions "
                "remain unverified."
            ),
        ),
        _blocked(
            "T_BANK_NEWS_ARCHIVE",
            ("T",),
            "https://www.tbank.ru/about/news-archive/",
            estimated=0,
            sampled=0,
            accessible=0,
            dated=0,
            blockers=(SourceBlocker.TECHNICAL_PARSE, SourceBlocker.ACCESS_POLICY),
            note=(
                "JavaScript archive did not provide a verified stable item/date interface "
                "for bounded automation."
            ),
        ),
        _blocked(
            "VTB_PRESS_ARCHIVE",
            ("VTBR",),
            "https://www.vtb.ru/about/press/archiv/",
            estimated=0,
            sampled=20,
            accessible=20,
            dated=20,
            blockers=(SourceBlocker.ACCESS_POLICY, SourceBlocker.STORAGE_POLICY),
            note=(
                "Official archive exposes source dates, but automation and excerpt storage "
                "policy are unverified."
            ),
        ),
        _blocked(
            "NORNICKEL_IR_ARCHIVE",
            ("GMKN",),
            "https://nornickel.com/news-and-media/press-releases-and-news",
            estimated=576,
            sampled=20,
            accessible=20,
            dated=20,
            blockers=(SourceBlocker.ACCESS_POLICY, SourceBlocker.STORAGE_POLICY),
            note=(
                "Official archive has date metadata; machine access and storage permission "
                "remain unverified."
            ),
        ),
        _blocked(
            "YANDEX_IR_DATE_ARCHIVE",
            ("YDEX",),
            "https://ir.yandex/press-releases",
            estimated=600,
            sampled=20,
            accessible=20,
            dated=20,
            blockers=(SourceBlocker.ACCESS_POLICY, SourceBlocker.STORAGE_POLICY),
            note=(
                "Year pages expose publication dates, but archive automation/storage "
                "permission is not proven."
            ),
        ),
    )
    for record in records:
        record.validate()
    return records


def _blocked(
    source_code: str,
    tickers: tuple[str, ...],
    source_url: str,
    *,
    estimated: int,
    sampled: int,
    accessible: int,
    dated: int,
    blockers: tuple[SourceBlocker, ...],
    note: str,
) -> SourceVerification:
    return SourceVerification(
        source_code=source_code,
        tickers=tickers,
        source_url=source_url,
        status=SourceAcceptanceStatus.BLOCKED,
        blockers=blockers,
        official_or_provenance_verified=True,
        free=True,
        automation_allowed=False,
        stable_identity_verified=accessible > 0,
        storage_policy_verified=False,
        estimated_items=estimated,
        sample_limit=20,
        sampled_items=sampled,
        verified_accessible_items=accessible,
        verified_date_items=dated,
        verified_exact_items=0,
        verified_daily_eligible_items=0,
        sampling_order="archive order, first bounded page/items; no model or market outcomes",
        evidence_note=note,
    )
