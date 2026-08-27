# exact-event-live-official-collection-v1

This collector performs zero-cost, data-only acquisition from official issuer sources that have
already been audited as strict EXACT live sources.

The first enabled source is `CHEP_CHTPZ_TMK_RSS_EXACT_LIVE_V1`:

- ticker: `CHEP`
- issuer: `ЧТПЗ`
- source: `https://chtpz.tmk-group.ru/rss`
- mechanism: RSS
- timestamp field: item-level `pubDate`
- timestamp policy: publication date, clock time, and explicit numeric timezone such as `+0300`

Date-only values, fetch timestamps, HTTP timestamps, sitemap `lastmod`, file timestamps, inferred
midnight/noon values, and guessed timezones are rejected as EXACT.

## One-Shot Command

```powershell
uv run python -m apps.cli.acquire_exact_event_live_official `
  --base-main-sha <BASE_MAIN_SHA> `
  --output-dir artifacts/exact-event-live-official-collection-v1
```

For replay-safe operation, pass the previous run's `dedupe-state.json`:

```powershell
uv run python -m apps.cli.acquire_exact_event_live_official `
  --base-main-sha <BASE_MAIN_SHA> `
  --state-file artifacts/exact-event-live-official-collection-v1/dedupe-state.json `
  --output-dir artifacts/exact-event-live-official-collection-v1-next
```

An environment-specific scheduler may run the one-shot command every 60 minutes. The scheduler must
not call broker services, market maturation, model training, backtests, TEST evaluation, or trading
actions.

## Safety

Future-holdout events on or after `2026-08-11` are metadata-only. The collector writes source
provenance, raw snapshots, deterministic event metadata, and dedupe state. It does not read targets,
post-event reactions, abnormal returns, predictions, model metrics, or broker state.
