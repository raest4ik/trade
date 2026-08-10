# ML Feature Dataset v1

## Purpose

`ml-feature-dataset-v1` is an auditable research dataset builder. It does not train a model,
produce predictions, backtest, or emit trading signals.

The central rule is:

> FEATURES contain only information available at or before publication. POST-EVENT RETURNS ARE
> LABELS AND MUST NEVER ENTER FEATURES.

Each row is reproducible from the news and instrument IDs plus `ml-features-v1`,
`event-rules-v2`, `financial-facts-v2`, `reaction-v2-benchmark-adjusted`, and
`pre-event-market-v1`. The manifest records these exact versions, the Git SHA, and a canonical
configuration hash.

## Eligibility

A supervised row requires an imported historical `NewsItem` with an `EXACT` publication
timestamp, exactly one non-ambiguous deterministic instrument match, an explicitly selected
`event-rules-v2` analysis, and at least one available benchmark-adjusted reaction label. An
optional required horizon can make that label mandatory.

`DATE_ONLY`, `UNKNOWN`, unmatched, ambiguous, analysis-free, and label-free candidates are
excluded with explicit reasons. Batch 001 is not a historical promoted candidate and remains
excluded from reaction training.

## Event And Fact Features

Event columns include the primary type, count, named flags, and a fixed flag for every current
`EventType`. Facts are filtered to `financial-facts-v2` and exported under semantic metric names.
Missing facts remain `NULL` with `has_<metric>=false`; they are never replaced with zero.

For duplicate metrics, deterministic selection is:

1. `ACTUAL`, `FORECAST`, `TARGET`, `PREVIOUS`, `CONSENSUS`, `CHANGE`, `UNKNOWN`;
2. latest explicit period;
3. confidence;
4. source position and UUID.

Values retain unit, currency, scale, and role columns. Change features are emitted only when
`change_unit=PERCENT`; percentage points are not mixed with percentages. Direction supplies the
sign, while the selected direction, unit, and comparison type remain explicit companion columns.
Guidance counts use `FORECAST` and `TARGET` roles. Dividend value and role remain explicit.

## Point-In-Time Market Context

All SQL market queries include `end_at <= published_at`. `PointInTimeFeatureBuilder` asserts the
same cutoff again and rejects the row with `POINT_IN_TIME_VIOLATION` if a future candle reaches
calculation. A candle ending exactly at publication is considered completed and may be used.

Security and IMOEX simple/log returns use the latest completed observation and the latest
completed observation at or before each 5/15/30/60-minute cutoff. Relative pre-event return is
security pre-return minus IMOEX pre-return. Missing baselines stay `NULL` and are listed in
quality metadata.

Realized volatility is the population standard deviation of consecutive one-minute log returns
inside each 15/30/60-minute pre-publication window, calculated with `Decimal`. It requires at
least two returns. Volume features use completed candles only: last minute, 5/15/60-minute sums,
and 5m/60m ratio when the denominator is positive.

Exchange-local hour, minute, weekday, and weekend fields use `Europe/Moscow`. No session bucket
is claimed in v1 because the project does not yet have a complete historical MOEX trading
calendar; clock fields are safer than hardcoded session hours.

## Labels

Labels are read unchanged from existing benchmark-adjusted reactions for 1/5/15/30/60 minutes:
security, IMOEX, and abnormal simple/log returns. They live only under `labels`.

UP/FLAT/DOWN classification is disabled by default. An operator may supply a non-negative
threshold for pipeline research; the manifest then marks it `RESEARCH_DEFAULT_NOT_CALIBRATED`.
It is not a trading threshold and must not be calibrated on TEST.

## Outputs

The builder writes under the ignored `artifacts/ml-feature-dataset-v1/` directory:

- `ml-feature-dataset-v1.jsonl`: canonical nested `metadata/features/labels/quality` rows;
- `ml-feature-dataset-v1.csv`: flat deterministic columns documented in `manifest.json`;
- `manifest.json`: versions, config hash, Git SHA, policies, columns, and readiness;
- `stats.json`: counts, missingness, and descriptive label distributions;
- `exclusions.jsonl`: every excluded candidate and reason.

Stats never feed back into feature generation. No global mean, standard deviation, fitted scaler,
future row, validation row, or test row is used for normalization. Samples of ten rows or fewer
are marked `INSUFFICIENT_SAMPLE_FOR_INFERENCE`.

## Commands

```bash
uv run python -m apps.cli.build_ml_feature_dataset --from 2026-01-01 --to 2026-12-31
uv run python -m apps.cli.export_ml_feature_dataset --from 2026-01-01 --to 2026-12-31
uv run python -m apps.cli.ml_feature_stats
uv run python -m apps.cli.ml_feature_dataset_smoke
```

Use `--dry-run` to compute eligibility, exclusions, and point-in-time validation without writing
an audit run or artifacts. `--limit` is applied after deterministic ordering by publication,
news ID, and instrument ticker/ID. Temporal split assignment is intentionally not automatic;
`published_at` is available for a later explicit older-to-newer split.

## Limitations

No AI inference runs during build and existing AI output does not override deterministic values.
AI availability fields are reserved metadata and remain false until there is a persisted,
versioned source. There are no embeddings, TF-IDF fitting, sentiment scores, hybrid Rules+Qwen
resolver, ML estimator, scaler, crawler, or new reaction calculation semantics in this change.

The current real reaction-ready corpus is far too small for model training. Until a larger legal
historical corpus is reviewed, the correct status is
`MODEL_TRAINING_NOT_READY_INSUFFICIENT_REAL_ROWS`.
