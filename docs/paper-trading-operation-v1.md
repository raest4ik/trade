# Paper Trading Operation V1

Paper Trading Operation V1 joins the existing live research, read-only Agent V1,
Risk Engine V1, append-only paper ledger, and replay verifier into a one-shot
operational cycle. It does not contain or register a real broker client.

## Safety contract

```text
REAL_EXECUTION_READY=NO
REAL_BROKER_MUTATIONS=0
REAL_ORDERS_SENT=0
REAL_ORDERS_CANCELLED=0
REAL_POSITIONS_CHANGED=0
```

The default `run` mode is `DRY_RUN`. A paper ledger mutation requires both the
explicit `--execute-paper` flag and `PAPER_EXECUTION_ENABLED=true` in the operation
policy. `PAPER_EXECUTION_ENABLED` defaults to `false`. Real execution remains disabled
in both modes.

## Commands

```powershell
uv run python -m apps.cli.paper_operation health
uv run python -m apps.cli.paper_operation run
uv run python -m apps.cli.paper_operation run --execute-paper
uv run python -m apps.cli.paper_operation status
uv run python -m apps.cli.paper_operation history
uv run python -m apps.cli.paper_operation inspect <operation_id>
uv run python -m apps.cli.paper portfolio
uv run python -m apps.cli.paper replay
```

Exit codes are `0` for success, no action, or an already processed operation; `2`
for blocked; `3` for degraded; and `4` for failed.

## Preflight

The operation runs the Agent only after research is `READY`, operational burn-in is
`PASS`, source-failure isolation has an accepted proof level, the research seal is
verified, the paper ledger replays, PIT inputs contain no future timestamps, and real
execution is disabled. A failed critical gate produces `BLOCKED` without Agent or
portfolio mutation.

The universe is deterministic, canonical, supported, and limited to ten instruments.
Held instruments are selected first. Market context is requested for both candidates
and held positions. Missing or stale held marks remain visible to Risk V1 and prevent
exposure-increasing BUY decisions.

## Operation state

Mutable state is stored outside committed artifacts:

```text
state/paper-operation-v1/operation-ledger.jsonl
state/paper-operation-v1/agent-runs/
state/paper-operation-v1/operation.lock
state/paper-portfolio-v1/ledger.jsonl
```

The portfolio ledger remains the portfolio source of truth. The separate operation
ledger records orchestration. An execute operation writes a `PREPARED` audit record
before durable paper fills, then writes `COMPLETED` only after replay verification.

## Recovery

Each operation has a canonical slot such as `2026-09-09:EOD`. Wall-clock seconds are
preserved in `operation_as_of` as the PIT cutoff but are excluded from `operation_id`,
so another invocation of the same date/session is a duplicate even if the model,
portfolio-derived universe, policy, or code contract changes. Those reproducibility
inputs are preserved separately in `operation_contract_sha` and `universe_sha`. An
intentional same-day rerun requires an explicit distinct slot such as
`--operation-slot EOD_RETRY_1`.

If a process stops after a fill but before final audit, rerun the same operation slot.
The operation finds the `PREPARED` record, verifies existing
idempotency keys, reconstructs paper order and trade IDs from the durable portfolio
ledger, and appends `RECOVERED`. It does not delete or roll back a durable fill and
does not create a duplicate fill.

If `health` or `run` reports `BLOCKED`, inspect the explicit reason, repair the input
artifact or ledger, and rerun. Never replace a missing historical operation with a
decision built from current data.

The scheduler is external. The optional Compose `paper-operation` profile is a
one-shot dry run and does not spin continuously. Session names and scheduling remain
configuration, not hard-coded market hours.
