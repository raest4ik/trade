from __future__ import annotations

from src.historical_news.domain.enums import ContentStoragePolicy, HistoricalNewsSourceKind
from src.news.domain.enums import PublicationTimestampQuality
from src.official_sources.domain import OfficialSourceConfig, OfficialSourceStatus

YANDEX_FEED_URL = "https://ir.yandex.ru/press-releases/news.rss"


def official_source_configs() -> tuple[OfficialSourceConfig, ...]:
    return (
        _blocked(
            "AUDIT_SBER",
            ("SBER", "SBERP"),
            "Sberbank",
            "https://www.sberbank.com/ru/investor-relations",
            "ACCESS_BLOCKED",
            "Controlled public request timed out; no stable issuer feed contract was verified.",
        ),
        _blocked(
            "AUDIT_GAZP",
            ("GAZP",),
            "Gazprom",
            "https://www.gazprom.com/press/news/",
            "ACCESS_BLOCKED",
            "Controlled public request timed out; no access control was bypassed.",
        ),
        OfficialSourceConfig(
            source_code="AUDIT_LKOH_RSS",
            tickers=("LKOH",),
            issuer="PJSC LUKOIL",
            owner="PJSC LUKOIL",
            landing_url="https://www.lukoil.com/PressCenter/Servicesforjournalists",
            feed_url=None,
            source_kind=HistoricalNewsSourceKind.ISSUER_RSS,
            status=OfficialSourceStatus.UNSTABLE_SOURCE,
            timestamp_quality=PublicationTimestampQuality.UNKNOWN,
            timestamp_semantics=(
                "Official Get RSS link flow returns RSS 2.0, but the observed channel was empty; "
                "item pubDate and timezone contract could not be verified"
            ),
            stable_identity="Generated channel endpoint observed; no item GUID was available",
            storage_policy=ContentStoragePolicy.UNKNOWN,
            legal_public_access="Public issuer subscription form; no CAPTCHA is used for RSS mode",
            bounded_method="One generated RSS channel request; no pagination",
            historical_depth="Observed channel had zero items and no historical archive depth",
            blocker="EXACT item timestamps and stable item identity remain unverified",
        ),
        OfficialSourceConfig(
            source_code="ROSNEFT_PRESS_RELEASES_RSS",
            tickers=("ROSN",),
            issuer="Rosneft Oil Company",
            owner="Rosneft Oil Company",
            landing_url="https://www.rosneft.com/press/releases/rss/",
            feed_url="https://www.rosneft.com/press/releases/rss/",
            source_kind=HistoricalNewsSourceKind.ISSUER_RSS,
            status=OfficialSourceStatus.REACTION_READY,
            timestamp_quality=PublicationTimestampQuality.EXACT,
            timestamp_semantics="RFC 822 pubDate includes numeric +0300 offset",
            stable_identity="Issuer release URL is used as stable item identity",
            storage_policy=ContentStoragePolicy.EXCERPT_ALLOWED,
            legal_public_access="Public issuer RSS",
            bounded_method="Single RSS page with explicit date range and limit",
            historical_depth="20 current official releases; no pagination",
        ),
        OfficialSourceConfig(
            source_code="AUDIT_NVTK",
            tickers=("NVTK",),
            issuer="PAO NOVATEK",
            owner="PAO NOVATEK",
            landing_url="https://www.novatek.ru/en/press/releases/",
            feed_url=None,
            source_kind=HistoricalNewsSourceKind.MANUAL_RESEARCH,
            status=OfficialSourceStatus.NLP_ONLY_DATE_ONLY,
            timestamp_quality=PublicationTimestampQuality.DATE_ONLY,
            timestamp_semantics=(
                "Official release text exposes a calendar date; no time/offset in HTML, "
                "OpenGraph, or JSON-LD"
            ),
            stable_identity="Stable issuer archive id_4 query parameter",
            storage_policy=ContentStoragePolicy.UNKNOWN,
            legal_public_access="Public issuer archive page",
            bounded_method="One archive page and one release page inspected",
            historical_depth="Paginated multi-year archive",
            blocker="Exact publication time cannot be proven and is not imputed",
        ),
        OfficialSourceConfig(
            source_code="YANDEX_IR_PRESS_RELEASES_RSS",
            tickers=("YDEX",),
            issuer="Yandex",
            owner="Yandex",
            landing_url="https://ir.yandex/press-releases",
            feed_url=YANDEX_FEED_URL,
            source_kind=HistoricalNewsSourceKind.ISSUER_RSS,
            status=OfficialSourceStatus.REACTION_READY,
            timestamp_quality=PublicationTimestampQuality.EXACT,
            timestamp_semantics="RFC 822 pubDate includes numeric +0300 offset",
            stable_identity="Unique issuer-owned GUID and release link",
            storage_policy=ContentStoragePolicy.EXCERPT_ALLOWED,
            legal_public_access="Public RSS link published on issuer IR page",
            bounded_method="Single 20-item RSS page with explicit date range and limit",
            historical_depth="20 current official releases; no pagination",
        ),
        _date_only(
            "AUDIT_T",
            ("T",),
            "T-Technologies",
            "https://www.tbank.ru/about/news/",
            "Official archive observations expose dates without a verified exact-time contract.",
        ),
        _blocked(
            "AUDIT_VTBR",
            ("VTBR",),
            "VTB Bank",
            "https://www.vtb.com/ir/statements/results/",
            "ACCESS_BLOCKED",
            "Controlled public request timed out; no stable exact-time feed was verified.",
        ),
        _date_only(
            "AUDIT_GMKN",
            ("GMKN",),
            "Nornickel",
            "https://www.nornickel.com/investors/news-and-releases/",
            "Prior official archive observation was date-only; current controlled request closed.",
        ),
    )


def reaction_ready_configs() -> tuple[OfficialSourceConfig, ...]:
    return tuple(
        config
        for config in official_source_configs()
        if config.status == OfficialSourceStatus.REACTION_READY
    )


def _blocked(
    source_code: str,
    tickers: tuple[str, ...],
    issuer: str,
    url: str,
    status: str,
    blocker: str,
) -> OfficialSourceConfig:
    return OfficialSourceConfig(
        source_code=source_code,
        tickers=tickers,
        issuer=issuer,
        owner=issuer,
        landing_url=url,
        feed_url=None,
        source_kind=HistoricalNewsSourceKind.MANUAL_RESEARCH,
        status=OfficialSourceStatus(status),
        timestamp_quality=PublicationTimestampQuality.UNKNOWN,
        timestamp_semantics="No verified machine-readable publication timestamp contract",
        stable_identity="No accepted live item identity",
        storage_policy=ContentStoragePolicy.UNKNOWN,
        legal_public_access="Public issuer page only; no access controls bypassed",
        bounded_method="One controlled page request",
        historical_depth="Not established by a compliant live endpoint",
        blocker=blocker,
    )


def _date_only(
    source_code: str,
    tickers: tuple[str, ...],
    issuer: str,
    url: str,
    blocker: str,
) -> OfficialSourceConfig:
    return OfficialSourceConfig(
        source_code=source_code,
        tickers=tickers,
        issuer=issuer,
        owner=issuer,
        landing_url=url,
        feed_url=None,
        source_kind=HistoricalNewsSourceKind.MANUAL_RESEARCH,
        status=OfficialSourceStatus.NLP_ONLY_DATE_ONLY,
        timestamp_quality=PublicationTimestampQuality.DATE_ONLY,
        timestamp_semantics="Calendar date only; no exact time or timezone contract",
        stable_identity="Issuer archive release URL",
        storage_policy=ContentStoragePolicy.UNKNOWN,
        legal_public_access="Public issuer archive",
        bounded_method="One controlled archive request",
        historical_depth="Official archive observed; not reaction-ready",
        blocker=blocker,
    )
