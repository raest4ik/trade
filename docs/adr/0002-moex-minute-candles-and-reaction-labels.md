# ADR 0002: MOEX Minute Candles And Reaction Labels

## Status

Accepted

## Context

The project needs historical labels that describe what actually happened to a
matched instrument after a public news item. This phase still excludes
forecasting, LLMs, sentiment analysis, trading integration, schedulers, and
real-time feeds.

## Decision

Use MOEX ISS historical candle endpoints for the first market data adapter. The
first supported interval is one minute because it is available through a simple
official HTTP endpoint and is enough to build coarse historical labels before
moving to trades.

The system performs explicit backfill by API or CLI. It does not poll in real
time and does not subscribe to WebSocket, ALGOPACK, order book, or trade feeds.

## Event Time

`NewsItem.published_at` is the event time because it represents when the news was
publicly available. `received_at` is not used as event time because it measures
when this system observed the news and may include crawler, network, or ingestion
delay. The delay is stored as `publication_to_receipt_ms`.

## Timezone Rule

MOEX ISS returns candle `begin` and `end` values without timezone offsets. This
adapter interprets them as `Europe/Moscow`, documents that assumption, and
converts them to UTC-aware datetimes before creating domain objects.

## Look-Ahead Bias

Baseline is the close of the last fully completed minute candle whose
`end_at <= published_at`. A candle that closes after publication is never used as
baseline. The reaction use case uses saved instrument matches and does not rerun
the matcher automatically, so the label calculation consumes persisted facts.

## Horizons

Effective event time is the `begin_at` of the first saved candle with
`begin_at >= published_at`. If the publication timestamp is exactly equal to a
minute candle start, that candle is treated as the first post-publication candle.
Horizon target time is calendar elapsed time
`effective_event_at + horizon_minutes`; it is not “N trading minutes”. For each
horizon, target price is the close of the first candle with `end_at >= target_at`.
The actual `observed_at` is stored because gaps in trading can move the observed
candle away from the target.

## Outside Session

The first version does not implement a trading calendar. If a large gap exists
between publication and the first available candle, the reaction can be marked
`OUTSIDE_SESSION`, while still storing the next available trading reaction when
data exists.

## Minute Candle Limits

Minute candles cannot reveal the exact price at the second of publication. A news
item may appear inside a candle, and part of the move between publication and the
next minute boundary cannot be separated. Low-latency analysis should later use
trades or more granular market data.

## Future Work

Market-adjusted return can be added by storing an index or benchmark candle
series and subtracting benchmark return for the same horizon. Trades can replace
minute candles by changing baseline and target lookup to trade timestamps while
keeping `reaction_version` explicit so old labels remain reproducible.
