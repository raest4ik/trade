# Production dry-run burn-in V1

## Purpose

This module collects immutable operational evidence for the production read-only pipeline:
Live Research, fresh MOEX PIT market data, local Ollama, Agent V1, and Risk V1. Burn-in
proves operational reliability only. It does not prove profitability or alpha.

The runner always stops before paper execution. It does not evaluate PnL, returns, labels,
trade outcomes, strategy quality, or future prices.

## Safety boundary

The fixed policy requires all execution and scheduling capabilities to remain disabled:

```text
PAPER_EXECUTION_ENABLED=false
REAL_EXECUTION_ENABLED=false
PAPER_OPERATION_SCHEDULE_ENABLED=false
```

The burn-in CLI has no `--execute-paper` option. It creates no broker client and passes
`DRY_RUN` to the existing Paper Trading Operation V1 pipeline. Before and after every
attempt it hashes the paper ledger and restored portfolio. Any mutation, future-data
violation, holdout access, or nonzero real-execution counter prevents readiness.

## Fixed policy

`production-dry-run-burnin-v1` is versioned in source code. Environment variables
cannot weaken its thresholds:

```text
MIN_DISTINCT_MOEX_TRADING_DAYS=5
MIN_VALID_CYCLES=5
MAX_BLOCKED_PRIMARY_CYCLE_RATE=0.20
MIN_MARKET_FRESH_RATE=0.95
MIN_RESEARCH_READY_RATE=0.90
MIN_AGENT_VALID_RATE=0.90
MIN_RISK_COMPLETION_RATE=0.90
MAX_SOURCE_CLOCK_SKEW_SECONDS=5
```

All paper, real, future-data, and holdout violation counters must be zero.

## Run once

```powershell
uv run python -m apps.cli.production_burnin run
```

The runner verifies a real TQBR session from the free official MOEX ISS daily-candle
endpoint. Missing or malformed evidence produces `MARKET_SESSION_UNKNOWN` and no
model call. A confirmed session runs the existing production provider, which freezes
`decision_as_of` after market fetch and loads all PIT inputs at that exact cutoff.
Weekends are `CLOSED` without a network call; weekdays are never assumed open.

One primary observation is allowed for `<MOEX trading date>:BURNIN_EOD`. A repeated primary
command returns `ALREADY_PROCESSED`. A transient blocked cycle may be retried explicitly:

```powershell
uv run python -m apps.cli.production_burnin run `
  --retry 1 --retry-reason MARKET_CONTEXT_UNAVAILABLE
```

Retries preserve the primary operation identity but use `BURNIN_EOD_RETRY_<n>`. They do
not increase distinct trading days or improve primary success rates.

## Observations and recovery

Observations are appended to
`state/production-dry-run-burnin-v1/observations.jsonl`. Every row has a sequence number,
the previous row hash, and its own canonical hash. Corruption blocks reads and new writes.
Each completed append also writes a compact immutable snapshot under `snapshots/`.
A single-flight lock prevents concurrent runs from recording the same logical slot.

If Paper Operation completed but the process stopped before observation append, the next
run recovers the immutable completed audit as `RECOVERED_FROM_OPERATION_AUDIT`. It does not
call Ollama again. Observations contain final proposals, short proposal rationale, risk
reasons, tool-call metadata, hashes, counters, and latency metadata. They do not persist
chain-of-thought or Ollama thinking.

## Status and verification

```powershell
uv run python -m apps.cli.production_burnin status
uv run python -m apps.cli.production_burnin history
uv run python -m apps.cli.production_burnin inspect <observation_id>
uv run python -m apps.cli.production_burnin report
uv run python -m apps.cli.production_burnin calendar-status
```

`PASS` means Agent output was valid and Risk completed. `BLOCKED_EXPECTED` is a correct
fail-closed response to an external or data gate. `FAIL` is an unexpected internal failure.
`SAFETY_VIOLATION` covers mutation or PIT/holdout safety breaches. Availability and safety
behavior are reported separately; a correctly blocked crossed book is not a successful
primary cycle.

`PRODUCTION_DRY_RUN_BURNIN_READY=YES` means the infrastructure can safely collect evidence;
it does not mean the actual burn-in passed. Actual status starts at `NOT_STARTED`, remains
`IN_PROGRESS` below the fixed five-cycle/five-trading-day threshold, becomes `FAIL`
immediately on a safety violation, and becomes `PASS` only after all fixed thresholds pass.
No status enables paper execution or scheduling; promotion requires a separate review.
