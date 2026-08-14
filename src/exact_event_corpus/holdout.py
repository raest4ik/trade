from __future__ import annotations

from datetime import date

from src.exact_event_corpus.domain import FUTURE_EVENT_HOLDOUT_START


class FutureEventHoldoutReadError(RuntimeError):
    pass


def guard_outcome_read(publication_date: date) -> None:
    if publication_date >= FUTURE_EVENT_HOLDOUT_START:
        raise FutureEventHoldoutReadError("FUTURE_EVENT_HOLDOUT_READ_ATTEMPT")


def is_future_holdout(publication_date: date) -> bool:
    return publication_date >= FUTURE_EVENT_HOLDOUT_START
