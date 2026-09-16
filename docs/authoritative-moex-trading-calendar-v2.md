# Authoritative MOEX trading calendar V2

## Source roles

The planned-session authority is the official MOEX trading calendar at
`https://www.moex.com/ru/tradingcalendar`. Its structured page payload contains
the stock-market `offDays.stock` schedule and market-specific session windows.
The resolver accepts only the stock-market schedule and records the `TQBR`
production scope; it does not combine stock, currency, derivatives, or bond
calendars.

The optional current-session source is the market-wide official MOEX ISS TQBR
board response at
`https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json`.
It can corroborate current runtime state, but cannot replace the planned
calendar. A completed daily candle for one security is not calendar authority.

MOEX's official weekend-session announcement documents that weekend sessions
run from 09:50 to 19:00 Moscow time and form part of the following trading day:
`https://www.moex.com/n95564`. The resolver therefore maps an authoritative
additional session to the next schedule row explicitly marked as a regular open
stock-market day. It never derives that date from a Monday-Friday assumption.

The annual schedule and later 2026 amendment demonstrate why the current
authoritative schedule must supersede initial publication:
`https://www.moex.com/n94172` and `https://www.moex.com/n103931`. A separate
official special-session notice provides a holiday example and explicitly names
its later trading date: `https://www.moex.com/n100780?nt=107`.

## Decision policy

- `OPEN` requires an authoritative stock/TQBR row, a resolved MOEX business
  date, and a timezone-aware Europe/Moscow session window.
- `CLOSED` requires an authoritative closed row.
- Missing, malformed, wrong-scope, or contradictory evidence becomes `UNKNOWN`.
- `UNKNOWN` and `CLOSED` retain the existing pre-model burn-in block.
- Distinct burn-in trading days use `moex_business_date` for V2 observations.
- Legacy observations without V2 fields retain their calendar-date fallback,
  original bytes, and original record hashes.
- Operation slot and idempotency identity remain calendar-slot based. Changing
  that identity is deliberately outside this PR.

The component exposes read-only scheduling predicates, but this PR does not
enable a scheduler, paper execution, or real execution.

## Operational freeze

Merging this PR changes production session semantics. It must remain unmerged
until an explicit requalification decision is made. Epoch 2 contains PRIMARY
records from more than one code SHA and is therefore non-qualifying. This PR
advances the qualification pointer to epoch 3, which starts with zero valid
cycles and zero distinct trading days after merge. Calendar checks are read-only
and must not write to the live burn-in state.

The Code SHA Homogeneity Guard anchors a qualification epoch to the code SHA of
its first PRIMARY record. A later PRIMARY with another SHA is rejected before
session verification, model invocation, paper operation, or ledger append.
