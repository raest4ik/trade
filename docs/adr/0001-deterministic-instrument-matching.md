# ADR 0001: Deterministic Instrument Matching

## Status

Accepted

## Context

The project needs an MVP way to identify Russian exchange instruments mentioned
in stored news. The first matching version must be explainable, reproducible, and
safe for later evaluation. The scope explicitly excludes LLMs, embeddings,
external AI APIs, fuzzy matching, MOEX integration, and trading automation.

## Decision

Use a deterministic instrument registry made of `Instrument` and `IssuerAlias`
records. Normalize news text and aliases with fixed Unicode, case, whitespace,
quote, punctuation, and `ё`/`е` rules. Match only exact ticker and exact alias
tokens on word boundaries.

The matcher version is stored as `deterministic-v1` with each
`NewsInstrumentMatch`. A rerun for the same `news_id` and matcher version
replaces that version's saved rows, which makes the operation idempotent without
rewriting unrelated historical matcher versions.

## Why Not LLM

An LLM would make the first matching layer harder to reproduce and harder to
audit. The MVP needs deterministic behavior so tests can prove exact ticker
boundaries, alias ambiguity, positions, and confidence values.

## Why Not Fuzzy Matching

Fuzzy matching can silently turn a typo or unrelated word into a financial
instrument. That is risky for market news analysis because a false match can
later affect event classification and evaluation. The first version uses exact
matches only.

## Ambiguity

Ambiguity is returned explicitly. If one normalized alias points to several
instruments, every candidate is returned with `is_ambiguous=true`. The system
does not automatically choose `SBER` over `SBERP` because of liquidity or
popularity. Exact ticker matches remain unambiguous when the ticker token is
present in the text.

## Future Issuer Registry

The current MVP keeps issuer data on the instrument because it is enough for
manual seed data and API matching. A future registry should split `Issuer` from
`Instrument`: one issuer can have common stock, preferred stock, bonds, and
renamed or reorganized instruments. The current alias table is designed so it can
later move toward issuer-level aliases without changing the news raw text.

## Consequences

- Matching is explainable and testable.
- False positives from fuzzy matching are avoided.
- Some real mentions will be missed until aliases are added manually or imported
  from trusted reference data.
- Ambiguous issuer names require downstream logic or manual review before they
  can be treated as a single instrument.
