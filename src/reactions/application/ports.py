from __future__ import annotations

from typing import Protocol
from uuid import UUID

from src.reactions.domain.entities import NewsMarketReaction


class ReactionRepository(Protocol):
    async def replace_reactions(
        self,
        *,
        news_id: UUID,
        reaction_version: str,
        reactions: list[NewsMarketReaction],
    ) -> list[NewsMarketReaction]: ...

    async def get_news_reactions(
        self,
        *,
        news_id: UUID,
        reaction_version: str | None = None,
    ) -> list[NewsMarketReaction]: ...
