# chep-security-history-diagnostics-v1

Diagnostics-only audit for the 44 historical CHEP strict-EXACT events blocked by
`SECURITY_HISTORY_MISSING` in `chep-historical-exact-maturation-v1`.

The pass verifies the PR46 maturation artifact, re-resolves CHEP through the existing
read-only T-Invest client, runs bounded one-minute and daily candle probes on
earliest/median/latest historical CHEP events, records possible alternate identities without
canonical substitution, and optionally cross-checks official MOEX ISS public data as diagnostic
provenance only.

Safety invariants:

- DIAGNOSTICS_ONLY=true
- MODEL_TRAINING_PERFORMED=false
- TEST_OUTCOME_USED=false
- TEST_EVALUATION_PERFORMED=false
- FUTURE_EVENT_HOLDOUT_USED=false
- FUTURE_EVENT_HOLDOUT_OBSERVED=false
- FUTURE_CHEP_PRICE_LOOKUPS=0
- FUTURE_CHEP_REACTIONS_COMPUTED=0
- FUTURE_CHEP_TARGETS_COMPUTED=0
- MOEX_SUBSTITUTION_USED=false
- FORWARD_FILL_USED=false
- SYNTHETIC_MARKET_DATA_USED=false
- LOCAL_ACQUISITION_LOGIC_ROOT_CAUSE=false

Output goes to `artifacts/chep-security-history-diagnostics-v1/` and is not staged in git.
