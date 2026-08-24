# EXACT Event Live Official Source Snapshot v1

This PR acquires a bounded live snapshot of official zero-cost public source mechanisms and then
runs the existing v5 discovery pipeline against that snapshot. It does not create a new
`ExactEvent` pipeline and does not perform market maturation.

Frozen v5 bounds are reused:

- `MAX_TICKERS=50`
- `MAX_URLS_PER_TICKER=10`
- `MAX_REQUESTS_PER_DOMAIN=25`
- `MAX_PAGES_PER_SOURCE=5`
- `MAX_ITEMS_PER_SOURCE=200`

Network limits are fixed before live acquisition:

- `REQUEST_TIMEOUT_SECONDS=10`
- `MAX_RESPONSE_BYTES=1000000`
- `MAX_REDIRECTS=3`
- `MIN_DOMAIN_DELAY_SECONDS=0.5`
- supported content types are static HTML, XHTML, XML, RSS, Atom, JSON, and JSON-LD

The acquisition layer writes a v5-compatible cache under `<artifact>/live-source-snapshot-cache`.
The downstream v5 artifact is written under `<artifact>/v5-downstream` and is the only layer that
updates source registry rows or extracts canonical exact events.

Exact timestamp rules:

- date plus clock time plus explicit offset/timezone is required
- RSS `pubDate`, Atom `published`, HTML/JSON-LD `datePublished`, and official JSON `published_at`
  are accepted when exact
- date-only values are rejected as exact
- fetch time, HTTP Date, file mtime, cache time, and search-engine timestamps are never publication
  timestamps

Safety invariants:

- `MODEL_TRAINING_PERFORMED=false`
- `TEST_OUTCOME_USED=false`
- `TEST_EVALUATION_PERFORMED=false`
- `FUTURE_EVENT_HOLDOUT_USED=false`
- `FUTURE_EVENT_HOLDOUT_OBSERVED=false`
- `DATE_ONLY_COERCIONS=0`
- `FETCH_TIME_USED_AS_PUBLICATION_TIME=false`
- `STRICT_EXACT_METHODOLOGY_CHANGED=false`
- `SPARSE_FAMILY_CREATED=false`
- no sparse label family
- `RULES_V3_CHANGED=false`
- `QWEN_CHANGED=false`
- `NLP_TUNING_PERFORMED=false`
- `TINVEST_READONLY_USED=false`
- no market candles, market maturation, model, TEST, backtest, paper trading, orders, BUY/SELL,
  predictions, or signals
