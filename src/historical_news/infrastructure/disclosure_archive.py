from __future__ import annotations

from pathlib import Path

from src.historical_news.infrastructure.local_archive import LocalArchiveNewsSource


class InterfaxDisclosureArchiveAdapter(LocalArchiveNewsSource):
    """Parses a locally supplied licensed/provider-neutral JSONL archive.

    This adapter has no credentials, endpoint, scraper, or access-control behavior.
    """

    def __init__(self, path: Path, *, max_items: int = 10_000) -> None:
        super().__init__(path, max_items=max_items)
