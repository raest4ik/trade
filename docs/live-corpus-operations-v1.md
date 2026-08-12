# Live Corpus Operations v1

## Purpose and policy

The hourly local job grows the existing REAL corpus from the two already accepted issuer-owned feeds: `ROSNEFT_PRESS_RELEASES_RSS` and `YANDEX_IR_PRESS_RELEASES_RSS`. It reuses source validation, stable-identity deduplication, deterministic instrument matching, bounded MOEX backfill, reaction builders, feature builders, and predictive readiness. It does not weaken source or storage policy and never substitutes `first_seen_at` or polling time for source `published_at`.

Telegram is excluded: `TELEGRAM_API = REJECTED_POLICY_FOR_ML`. There is no Telegram adapter. The job uses no paid API, dataset, cloud scheduler, hosted inference, or monitoring service.

New records mature over later runs through `INGESTED`, `MATCHED`, `WAITING_INTRADAY_TARGET`, `INTRADAY_READY`, `WAITING_DAILY_TARGET`, `DAILY_READY`, and `FEATURE_READY`. A missing future market window is a waiting condition, not invented reaction data. Source failures are isolated; a healthy source still runs when the other is unavailable. HTTP and MOEX clients retain bounded timeout, retry, backoff, payload, and domain restrictions.

## Manual operation

From any directory, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\path\to\trade Ai\scripts\windows\run-live-corpus.ps1"
```

Direct commands from the repository are:

```text
uv run python -m apps.cli.live_corpus_run
uv run python -m apps.cli.live_corpus_run --dry-run
uv run python -m apps.cli.live_corpus_status
```

`--dry-run` may fetch accepted feeds and read current state, but does not create DB rows, reactions, features, checkpoints, health, history, or logs. An exclusive local lock makes overlap return `ALREADY_RUNNING`. Stable source identity and database uniqueness remain the duplicate defense even if checkpoints are lost.

## Windows scheduling

Install the current-user task with the default one-hour cadence:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\install-live-corpus-task.ps1
```

Remove it with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\remove-live-corpus-task.ps1
```

The task uses the current interactive Windows account and does not store a password in the repository. It resolves the repository from the script location, sets an explicit working directory, starts only local PostgreSQL, and asks Task Scheduler to ignore overlapping instances. If Windows security policy blocks registration, run the install command manually in a PowerShell session permitted to create current-user scheduled tasks; do not bypass Windows security.

When the computer is off, no collection occurs. `StartWhenAvailable` requests one run after the next login/availability. Without internet, per-source/MOEX failures are recorded and later hourly runs retry within bounded rules.

## State and readiness

Gitignored operational files are under `artifacts/live-corpus-operations-v1/`: `checkpoints.json`, `health.json`, `growth-history.jsonl`, and retained per-run JSON logs. Logs contain run timing and counts but redact common secret fields. Old logs are rotated by count.

Every successful non-dry run rebuilds predictive readiness. `live_corpus_status` shows source health, daily/intraday feature counts, rows remaining to 100/500/1000, and the training gate. Reaching 100 changes readiness to `PILOT_TRAINING_ALLOWED`, but this job still does not train. A corpus above 100 with more than 70% of daily features in one ticker emits `LOW_TICKER_DIVERSITY`; collection continues so diversity can improve.

Training remains a separate explicit, human-reviewed command. No NLP component, Logistic/Ridge configuration, backtest, BUY/SELL output, or predictive-performance claim is part of this operation.
