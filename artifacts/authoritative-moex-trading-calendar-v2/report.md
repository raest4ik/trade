# Authoritative MOEX trading calendar V2

Deterministic fixture evidence. It is not a live MOEX snapshot.

- session policy: authoritative-moex-trading-calendar-v2
- artifact code SHA: ca94b208eb7fc9034fa7226d1990e207dd280843
- artifact SHA: b37721c7c96fc44d2fa3ff05d1ac412f599ff2b3fa3f4a77b52ee9484eed52b7
- official planned source: MOEX trading calendar
- optional runtime source: market-wide MOEX ISS TQBR state
- no weekday or weekend fallback
- additional sessions use explicit MOEX business-date semantics
- unavailable, malformed, or contradictory evidence fails closed
- legacy ledgers remain byte-preserved and hash-valid
- paper execution: disabled
- real execution: disabled
- scheduling: disabled
- merge during active epoch 2: prohibited
