# Authoritative MOEX trading calendar V2

Deterministic fixture evidence. It is not a live MOEX snapshot.

- session policy: authoritative-moex-trading-calendar-v2
- artifact code SHA: b6eb89ac7dfd5578565c8c54c98a7f123f07edeb
- artifact SHA: c83640d430a016484902de2e1b989fbc96fe0c1dc195aea452a2b6cde6ac2918
- official planned source: MOEX trading calendar
- optional runtime source: market-wide MOEX ISS TQBR state
- no weekday or weekend fallback
- additional sessions use explicit MOEX business-date semantics
- unavailable, malformed, or contradictory evidence fails closed
- legacy ledgers remain byte-preserved and hash-valid
- live ledger records verified: 4
- epoch 2 code SHA homogeneity: FAIL
- epoch 2 qualification: NON_QUALIFYING_MIXED_CODE
- current epoch: production-dry-run-burnin-v1-epoch-3
- current epoch qualification: NOT_STARTED
- current epoch starts with 0 valid cycles and 0 distinct trading days
- paper execution: disabled
- real execution: disabled
- scheduling: disabled
- merge during active epoch 2: prohibited
