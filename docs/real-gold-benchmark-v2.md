# Real Gold Benchmark v2

## Dataset

Batch 003 is separate from frozen Batch 001. It contains REAL issuer-owned news records with EXACT publication timestamps, but its human labels were reviewed only from the stored excerpts. It is EXCERPT_ONLY gold, not full-text human gold.

- dataset: `ru-corporate-events-real-batch-003-gold-v1`
- dataset SHA-256: `ca3fc96316d1974525fb0b65ef8d9af2e90dec458ae1e38b6fd921084b259139`
- human-review source SHA-256: `a994155301aa817aee0083fcc8945de321f7d13decc0e6f14b7678f2149ee170`
- frozen Batch 001 SHA-256: `4934b37b1c036eedb6191dae5ece2fa49e710d00455576cee3de081cc9e7c196` (unchanged)
- records: 26
- review basis: `EXCERPT_ONLY`
- tickers: `{"ROSN":10,"YDEX":16}`
- sources: `{"ROSNEFT_PRESS_RELEASES_RSS":10,"YANDEX_IR_PRESS_RELEASES_RSS":16}`
- months: `{"2026-06":10,"2026-07":12,"2026-08":4}`
- event distribution: `{"FINANCIAL_RESULTS":1,"GUIDANCE":1,"OTHER":24}`
- warnings: `SMALL_SAMPLE`, `CLASS_IMBALANCE`, `LOW_SOURCE_DIVERSITY`, `LOW_TICKER_DIVERSITY`

The 26 rows are too small and too concentrated by class, ticker, source, and time for broad conclusions. No future market returns, prices, volume, or reaction labels are analyzer inputs.

## Rules v2

- successful/failed: 26/0
- mean latency ms: 0.654
- input/output/total tokens: None/None/None
- event micro precision/recall/F1: 0.25/0.038462/0.066667
- event macro F1: 0.333333
- primary accuracy: 0.038462
- OTHER precision/recall/F1: 0.0/0.0/0.0
- FINANCIAL_RESULTS precision/recall/F1: 1.0/1.0/1.0
- GUIDANCE precision/recall/F1: 0.0/0.0/0.0
- fact value precision/recall/F1: 0.5/1.0/0.666667
- fact metric precision/recall/F1: 0.5/1.0/0.666667
- fact semantic strict precision/recall/F1: 0.0/0.0/0.0
- fact evidence span accuracy: 0.0
- fact field accuracies: `{"change_direction":1.0,"change_unit":1.0,"change_value":1.0,"comparison_type":0.0,"currency":1.0,"fact_role":0.0,"metric":1.0,"normalized_value":1.0,"period_month":1.0,"period_quarter":1.0,"period_type":0.0,"period_year":0.0,"scale":1.0,"unit":1.0}`
- primary confusion: `[{"count": 1, "gold": "GUIDANCE", "predicted": "UNKNOWN"}, {"count": 3, "gold": "OTHER", "predicted": "LITIGATION"}, {"count": 21, "gold": "OTHER", "predicted": "UNKNOWN"}]`

## Qwen 3.5 9B

- successful/failed: 26/0
- mean latency ms: 6767.577
- input/output/total tokens: 34304/2946/37250
- event micro precision/recall/F1: 0.928571/0.5/0.65
- event macro F1: 0.54955
- primary accuracy: 0.5
- OTHER precision/recall/F1: 0.923077/0.5/0.648649
- FINANCIAL_RESULTS precision/recall/F1: 1.0/1.0/1.0
- GUIDANCE precision/recall/F1: 0.0/0.0/0.0
- fact value precision/recall/F1: 0.0/0.0/0.0
- fact metric precision/recall/F1: 0.0/0.0/0.0
- fact semantic strict precision/recall/F1: 0.0/0.0/0.0
- fact evidence span accuracy: 1.0 (vacuous: no matched fact pairs)
- fact field accuracies: `{}`
- primary confusion: `[{"count": 1, "gold": "GUIDANCE", "predicted": "OTHER"}, {"count": 12, "gold": "OTHER", "predicted": "UNKNOWN"}]`

Any comparison describes performance only on this 26-example real excerpt-reviewed benchmark. It is not a claim of general superiority.

## Four-way primary outcome

- BOTH_CORRECT: 1 (0.038462)
- BOTH_WRONG: 13 (0.500000)
- QWEN_ONLY_CORRECT: 12 (0.461538)
- RULES_ONLY_CORRECT: 0 (0.000000)
- ORACLE_UPPER_BOUND primary accuracy: 0.500000 (diagnostic only)
- Rules/Qwen primary disagreement count: 16

ORACLE_UPPER_BOUND is not a hybrid, fallback, ensemble, reconciliation, or emitted prediction. It only counts records where at least one frozen system was correct.

## Error taxonomy

```json
{
  "counts": {
    "qwen3.5:9b": {
      "OTHER_ERROR": 3,
      "OTHER_INSTEAD_OF_SPECIFIC": 1,
      "UNKNOWN_INSTEAD_OF_OTHER": 12
    },
    "rules-v2": {
      "FACT_PERIOD_ERROR": 3,
      "FACT_ROLE_ERROR": 3,
      "FALSE_SPECIFIC_EVENT": 3,
      "MISSED_EVENT": 1,
      "OTHER_ERROR": 6,
      "UNKNOWN_INSTEAD_OF_OTHER": 21
    }
  },
  "models_unchanged": true,
  "most_common": {
    "qwen3.5:9b": "UNKNOWN_INSTEAD_OF_OTHER",
    "rules-v2": "UNKNOWN_INSTEAD_OF_OTHER"
  },
  "research_only": true
}
```

The taxonomy is research-only. No error was used to change Rules or Qwen in this PR.

## Limitations and test policy

- Human review is excerpt-only and may miss facts available on linked full articles.
- OTHER dominates the benchmark, so micro metrics can hide minority-class failures.
- ROSN and YDEX are the only represented tickers and issuer sources.
- Evidence accuracy is reported separately from semantic strict fact scoring.
- The three GUIDANCE percentage targets preserve PERCENT units and are not converted to percentage points.
- Batch 003 is OBSERVED after this evaluation and must not be used to tune v2 or Qwen.
- Future extractor changes require a fresh reviewed Batch 004 or another holdout.
- No prompt, schema, model, rules, reaction, or ML feature semantics were changed.
- No hybrid, predictive model training, backtest, or trading signal was created.
