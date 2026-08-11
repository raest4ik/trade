# Predictive Baseline v1

## Scope

This infrastructure predicts the leakage-safe daily abnormal reaction of a REAL issuer event relative to IMOEX. It supports two separate tasks: three-class direction (`UP`, `FLAT`, `DOWN`) with a frozen 0.002 threshold, and regression of the raw daily abnormal return. It does not emit BUY/SELL recommendations and does not claim trading performance.

The current daily corpus has only 34 feature-ready rows. That is enough to exercise software paths, but not enough to estimate predictive performance. Real training is therefore blocked below 100 rows. Counts from 100-499 permit pilot training, 500-999 permit a baseline experiment, and at least 1000 satisfy the row-count gate for baseline training. Diversity and other research checks remain separate requirements.

## Leakage protection

Input features come only from `ml-daily-features-v1` and are available by the baseline session close, strictly before publication date. Targets are loaded from the physically separate label object and `date-safe-daily-reaction-v1` record. Names associated with future prices, volume, targets, or returns are rejected from `X`.

Rows are ordered chronologically by publication date. Whole publication-date groups go to TRAIN, VALIDATION, or TEST, so equal-date events cannot straddle a boundary. Labels that overlap the next split boundary are purged, and an explicit one-day embargo removes immediate boundary rows. Encoders, numeric medians, and scaling statistics are fit on TRAIN only. Unknown categories in later periods map to an explicit unknown bucket.

TEST is not used for model or threshold selection. Logistic Regression and Ridge use a fixed config and seed. Validation and frozen TEST reports include majority-class, TRAIN-mean, and zero-return naive baselines. Probability calibration is represented by an explicit readiness status but is not run until validation has at least 100 rows.

## Commands

```text
uv run python -m apps.cli.ml_readiness
uv run python -m apps.cli.train_daily_baseline
uv run python -m apps.cli.train_daily_baseline --development-smoke
```

The default training command exits successfully with `TRAINING_BLOCKED` while daily feature-ready count is below 100. `--development-smoke` explicitly permits an in-memory technical fit. Its output is marked `DEVELOPMENT_SMOKE_ONLY`, `NOT_VALID_FOR_TRADING`, and `NOT_A_PERFORMANCE_ESTIMATE`; no production model binary is saved.

## Artifacts and next step

Gitignored outputs live under `artifacts/predictive-baseline-v1/`. Future permitted real runs receive immutable directories and manifests containing dataset, feature schema, split, code, config, period, metrics, and model binary fingerprints. The clean prediction contract is ready for a future backtester, but this change does not implement one.

The next useful step is continued zero-cost growth of the REAL daily corpus. No NLP tuning, hosted inference, paid API, cloud training, or live collector change is part of this work.
