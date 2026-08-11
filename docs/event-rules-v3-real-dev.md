# Event Rules v3 on Fresh Real DEVELOPMENT

## Scope

This work uses only the ten human-reviewed Batch 004 `DEVELOPMENT` excerpts. The four
`FRESH_HOLDOUT` items remain behind a firewall: only their aggregate count and the frozen split
SHA are read from metadata. Their text, event distribution, and model outputs are unavailable to
this workflow.

Batch 003 remains an `OBSERVED_EVALUATION_SET`. Its records, errors, historical
`OTHER -> UNKNOWN` counts, and Qwen comparisons were not used to design v3.

## DEVELOPMENT gold

The human review is validated before evaluation and frozen as:

- dataset: `ru-corporate-events-real-batch-004-development-gold-v1`
- provenance: `REAL`
- purpose: `DEVELOPMENT`
- review basis: `EXCERPT_ONLY`
- records: 10
- split SHA: `a32956626d194158eb69869f6bdca510456ded47ac5810ca91fe90b86aa45dea`

The class distribution is:

| Event | Records |
| --- | ---: |
| FINANCIAL_RESULTS | 4 |
| OTHER | 3 |
| DIVIDEND | 1 |
| SANCTIONS | 1 |
| MANAGEMENT_CHANGE | 1 |

The only explicit financial fact is `DIVIDEND_PER_SHARE=14.68`, `RUB`, `ACTUAL`, for 2024.
No fact is inferred beyond the permitted excerpt.

## v2 diagnosis

`event-rules-v2` recognizes the dividend event but returns `UNKNOWN` for the other nine records.
`financial-facts-v2` does not recognize the English `14.68 roubles per share` expression. The
v2 code and version remain unchanged and reproducible.

## v3 changes

`event-rules-v3` layers four bounded English semantic patterns over the frozen v2 result:

| v3 rule | DEVELOPMENT evidence |
| --- | --- |
| IFRS results announcement | `publishes/announces its results ... IFRS` in four reviewed excerpts |
| restrictive measures | `impose restrictive measures` in the reviewed sanctions excerpt |
| elected leadership | `has been elected Chairman` in the reviewed management excerpt |
| cooperation agreement | signed/concluded `cooperation agreement` or `agreement of cooperation` in three reviewed OTHER excerpts |

The OTHER rule is not an unconditional `UNKNOWN -> OTHER` fallback. Unidentifiable text remains
`UNKNOWN`. Rules contain no ticker, issuer, date, URL, news ID, or source-item special cases.

`financial-facts-v3` adds one general English dividend-per-share expression for an explicit
numeric amount in RUB/roubles. It is additive; `financial-facts-v2` is unchanged.

## DEVELOPMENT performance

These are development-set measurements, not a blind benchmark and not a generalization score.

| System | Event micro F1 | Macro F1 | Primary accuracy | Fact semantic strict F1 |
| --- | ---: | ---: | ---: | ---: |
| event-rules-v2 | 0.181818 | 0.200000 | 0.100000 | 0.000000 |
| frozen Qwen | 0.625000 | 0.600000 | 0.500000 | 0.000000 |
| event-rules-v3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

Qwen used the frozen `qwen3.5:9b`, `think=false`, seed `0`, context `4096`, prompt, and schema.
It succeeded on 10/10 DEVELOPMENT items. It detected all three OTHER records, DIVIDEND, and
MANAGEMENT_CHANGE; it classified SANCTIONS as REGULATORY_ACTION and left all four
FINANCIAL_RESULTS records UNKNOWN. Its dividend fact matched metric and value but not all strict
semantic fields or the exact evidence span. Qwen output was not used as rule-design evidence or
as gold.

## Freeze policy

The generated `event-rules-v3-real-dev-candidate` manifest stores the rules fingerprint,
DEVELOPMENT gold SHA, frozen configuration, DEVELOPMENT metrics, and Git SHA. After the candidate
is frozen, v3 must not change before a separately authorized blind `FRESH_HOLDOUT` evaluation.

There is no Rules/Qwen hybrid, voting, fallback, or reconciliation. This work performs no
predictive ML training, market-reaction calculation, backtest, signals, or paper trading.
