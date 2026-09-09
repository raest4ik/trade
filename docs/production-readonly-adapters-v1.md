# Production Read-Only Adapters V1

This layer connects Paper Trading Operation V1 to a local Ollama model and the free
public MOEX ISS market-data endpoint. It remains a one-shot, read-only production
dry run by default. It does not add a broker client, scheduler, real execution, model
training, backtesting, paid data, or performance claims.

## Data Flow

```text
MOEX Public ISS + existing live research
  -> ProductionPaperOperationContextProvider
  -> Agent V1 read-only tools
  -> OllamaAgentModel
  -> Risk Engine V1
  -> Paper Trading Operation V1
```

The context provider selects every held paper position first, then fills the remaining
operation universe with sorted canonical candidates. A held position is never dropped
because of universe truncation. Missing, stale, invalid, or future held marks remain
visible to Risk V1 and prevent exposure-increasing decisions.

The live provider has a two-phase time contract:

```text
cycle_started_at
  -> restore portfolio and select universe
  -> fetch immutable raw MOEX quotes
  -> decision_as_of = market_fetch_completed_at
  -> validate market timestamps and quality
  -> load events and research at decision_as_of
  -> preflight -> Agent -> Risk -> DRY_RUN
```

`decision_as_of` is the single PIT cutoff used by Agent, Risk, portfolio marks,
events, research, operation audit, and the trading-date/session operation identity.
Exact cutoff seconds do not enter `operation_id`, preserving session idempotency.

## Market Contract

`MoexIssFreshMarketAdapter` accepts only `https://iss.moex.com` and canonical `TQBR`
shares. `market_data_as_of` comes from MOEX `marketdata.SYSTIME`, interpreted in the
Moscow exchange timezone when the source has no offset. Local receipt time is stored
separately as `fetched_at` and never substitutes for the exchange timestamp.

Every quote is classified as `FRESH`, `STALE`, `MISSING`, `INVALID`, or `FUTURE`.
The effective freshness threshold is the smaller of
`MARKET_CONTEXT_MAX_AGE_SECONDS` and `RiskPolicy.max_stale_market_age`. Missing or
invalid source timestamps and prices are not synthesized. Bid and ask remain null when
the source omits them.

Transport/timestamp readiness and book quality are audited separately. A crossed
`bid > ask` remains `BID_ABOVE_ASK`, is excluded from downstream market quotes, and is
never repaired or used for a fill. A source timestamp after fetch completion remains
`MARKET_SOURCE_CLOCK_SKEW`/`FUTURE` with its signed delta and blocks before Agent.

## Model Contract

`OllamaAgentModel` implements the existing `AgentModel` protocol and calls only a
localhost Ollama endpoint. The existing Agent V1 read-only registry controls tool
access and validates the final proposals. The adapter provides no broker, shell,
Python, arbitrary URL, filesystem, or portfolio-mutation tools.

Timeouts, connection errors, malformed JSON, empty output, and schema violations fail
closed. There is no fallback model and no synthetic recommendation. Ollama `thinking`
or reasoning fields are ignored and are not persisted in operation audit or artifacts.

## Configuration

```text
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b
OLLAMA_THINK=false
AI_REQUEST_TIMEOUT_SECONDS=30
AI_MAX_RETRIES=2
AI_MAX_OUTPUT_TOKENS=4096
AI_RANDOM_SEED=0
MARKET_CONTEXT_MAX_AGE_SECONDS=300
PAPER_EXECUTION_ENABLED=false
REAL_EXECUTION_ENABLED=false
PAPER_OPERATION_SCHEDULE_ENABLED=false
```

The Ollama URL is restricted to `localhost`, `127.0.0.1`, or the exact
`host.docker.internal` bridge used by the Compose one-shot container. The MOEX
adapter uses bounded timeout, retry count, response size, and an explicit User-Agent.

## Commands

```powershell
uv run python -m apps.cli.paper_operation model-smoke
uv run python -m apps.cli.paper_operation market-smoke SBER YDEX
uv run python -m apps.cli.paper_operation health
uv run python -m apps.cli.paper_operation run
```

Model and market smoke commands do not invoke a trading decision or mutate a ledger.
`run` defaults to `DRY_RUN`. A paper fill still requires both `--execute-paper` and
`PAPER_EXECUTION_ENABLED=true`; real execution remains disabled.

## Deterministic Evidence

Rebuild the committed mocked artifact without live network access:

```powershell
uv run python -m apps.cli.build_production_readonly_adapters_v1 `
  --base-main-sha <base-sha> `
  --head-sha <implementation-sha>
```

The builder refuses to overwrite a non-empty artifact directory. The explicit
implementation SHA avoids substituting the later evidence-commit SHA. Rebuilding into
two empty directories with the same base and implementation SHAs produces
byte-identical files.

The mocked artifact proves deterministic adapter, PIT, context, Agent, Risk, and
dry-run behavior with `DETERMINISTIC_PRODUCTION_ADAPTER_PROOF=PASS`. It intentionally
keeps `LIVE_PRODUCTION_DRY_RUN=NOT_RUN` and `PRODUCTION_DRY_RUN_READY=NO`; only a real
live MOEX + Ollama end-to-end run may promote the latter status.
