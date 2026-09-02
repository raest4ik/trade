# Free Live Issuer Accumulation v1

This PR defines a research-only live accumulation path for free, official,
issuer-originated strict-EXACT events. It does not train models, run backtests,
open the old future holdout, compute live targets, or change Rules v3.

## Dataset Lifecycle

The historical issuer corpus is frozen. Events with `published_at >= 2026-08-11`
enter `LIVE_SHADOW_CORPUS`, an immutable append-only epoch. They are not
`TRAIN_READY` in this PR.

Allowed live event statuses are:

- `DISCOVERED`
- `TIMESTAMP_VERIFIED`
- `TICKER_RESOLVED`
- `RAW_SNAPSHOT_FROZEN`
- `SEMANTIC_READY`
- `PRE_EVENT_FEATURE_READY`
- `SHADOW_READY`
- `REJECTED`

`TRAIN_READY` is intentionally absent.

## Holdout Seal

The old future holdout starts on `2026-08-11`. Live shadow records may preserve
source URL, source item ID, issuer, ticker, publication timestamp, timezone
evidence, title, raw publication content, source metadata, and point-in-time
features available at or before publication.

The following remain forbidden for sealed live events:

- post-event return
- abnormal return
- benchmark reaction after the event
- 1m/5m/15m/30m/60m labels
- direction class from future returns
- model prediction or score
- profitability
- future target distribution

The artifact counters must stay:

- `LIVE_POST_EVENT_PRICE_READS=0`
- `LIVE_TARGETS_COMPUTED=0`
- `LIVE_OUTCOMES_READ=0`
- `LIVE_MODEL_PREDICTIONS=0`
- `OLD_FUTURE_HOLDOUT_OPENED=false`

Any target/reaction attempt for `LIVE_SHADOW_CORPUS` fails closed with
`SEALED_LIVE_EPOCH_OUTCOME_READ_ATTEMPT`.

## Free Official Source Contract

Accepted sources must be free, public, issuer-controlled, deterministic for
issuer/ticker binding, and preserve raw publication material. Publication time
is accepted only when the item timestamp has an explicit UTC offset/Z, or a
first-party source contract unambiguously defines the timezone.

The registry is `config/live_issuer_sources_v1.json`. Each source stores:
source ID, ticker, issuer, canonical domain, discovery URL/type, parser,
timezone contract, timestamp path, identity path, content path, enabled flag,
polling policy, source version, and contract fingerprint.

Paid or commercial sources are recorded only as `OUT_OF_SCOPE_PAID_SOURCE` and
are not investigated further.

## Collector Shape

`src.free_live_issuer_accumulation` implements bounded one-shot collection:

- no daemon requirement
- bounded source count and item count
- rate-limit friendly HTTP client reuse
- deterministic dedupe by source item ID first, canonical URL second
- content hash only as revision/update signal
- immutable first raw snapshot
- revision log for changed content under the same identity
- separate `published_at`, `first_observed_at`, and `fetched_at`
- semantic replay through frozen `EventAnalyzerV3`
- shadow corpus rows with `TARGET_STATUS=SEALED`

Smoke mode may use deterministic fixture RSS for CI/local validation. Real
network smoke records unavailable endpoints as `ENVIRONMENT_UNAVAILABLE` and
does not synthesize success. Items older than `2026-08-11` are rejected from the
live shadow corpus because the historical corpus is frozen.

## Promotion Protocol

Promotion is out of scope. A future PR may promote part of the live shadow epoch
only after it first fixes a new cutoff, TRAIN/VALIDATION period, untouched future
TEST period, and split policy. Outcomes must not be inspected before the split
policy is locked.

## Current Decision

Current free official evidence is not enough to answer YES for at least three
new MOEX issuer tickers relative to the frozen historical seven. The current
strict answer is therefore `NO` with blocker:

`insufficient issuer-originated free strict-EXACT sources for at least 3 new MOEX tickers`

The implemented path is still useful: it can accumulate accepted free official
sources now while keeping the old future holdout sealed.
