# Official event-source expansion v2

## Scope

This phase expands issuer diversity while keeping the predictive unit equal to an official
issuer event. It does not train a model, inspect a future market holdout, run a backtest, or
create a trading capability. Rules v3, financial facts, Qwen, reaction methodology, and the
market feature schema remain frozen.

The complete 315-share TQBR/RUB instrument mapping is emitted as
`event-source-registry-v2`. A URL is never guessed. `SOURCE_READY` means that a public HTTPS
source and its deterministic parser were exercised successfully. Known but unusable sources
retain an explicit discovery, implementation, policy, structure, or access status.

## Immutable v1 carry-forward

`event-market-predictive-dataset-v1` remains immutable. Before a v2 build, the builder verifies
the frozen dataset, source-registry, provenance, and feature-schema SHA values. It also hashes
the parsed v1 `features.jsonl` and `targets.jsonl` payloads. V2 starts with those exact rows and
adds only records from newly implemented issuer families. A changed or incomplete v1 artifact
fails closed.

This avoids a historical archive changing underneath the dataset. In particular, a current
issuer archive may expose a different number of old pages than it did when v1 was frozen; that
must not silently remove or rewrite a v1 event.

## Source families

The shared `BOUNDED_OFFICIAL_ARCHIVE_V2` transport provides HTTPS-only requests, bounded retries,
payload limits, verified cache digests, deterministic limits, and checkpoint/resume behavior.
The audited sites do not share one real CMS schema, so parser profiles remain explicit rather
than being presented as a fictional multi-issuer format.

The ready registry includes the preserved ROSN, YDEX, and NVTK sources plus these issuer-owned
archives:

| Ticker | Official source | Accepted timestamp | Parser profile |
| --- | --- | --- | --- |
| LKOH | LUKOIL press releases | `DATE_ONLY` | `LUKOIL` |
| GMKN | Nornickel press releases and news | `DATE_ONLY` | `NORNICKEL_APP` |
| TATN | Tatneft legacy press-release archive | `DATE_ONLY` | `TATNEFT` |
| ALRS | ALROSA news year archives | `DATE_ONLY` | `ALROSA` |
| PHOR | PhosAgro company news | `DATE_ONLY` | `PHOSAGRO` |
| PLZL | Polyus press releases | `DATE_ONLY` | `POLYUS` |
| IRAO | Inter RAO company news | `DATE_ONLY` | `INTERRAO` |
| MGNT | Magnit press releases | `DATE_ONLY` | `MAGNIT_APP` |

Some pages contain an internal epoch or a displayed clock time. V2 still accepts only the date
when a stable publication-time and timezone contract is not proven. It never converts such a
field into an `EXACT` event by assumption.

GAZP, SBER/SBERP, and VTBR are access-blocked for the controlled client. T has a discovered
official archive but no accepted deterministic collector in this phase. No access control,
CAPTCHA, authentication, robots rule, or rate limit is bypassed.

## Rights and storage

Collection uses public, issuer-owned pages at zero external data cost. Generated records contain
canonical URL, publication identity/date, issuer identity, title hash, parser version, and source
rights status. Raw full text is not redistributed. Live responses, generated datasets, and caches
remain under gitignored `artifacts/` for private internal research.

## Point-in-time methodology

The event pipeline remains:

```text
official event -> deterministic event semantics -> ticker
-> pre-event market context -> separate post-event target -> event dataset
```

`DATE_ONLY` events use the existing `DATE_SAFE_DAILY` rule: baseline close strictly before the
publication date and target close on the first common security/IMOEX session strictly after it.
Same-day close is not used. Features and targets remain physically separate and the leakage audit
must pass. Source selection and event retention do not inspect returns, abnormal returns, volume,
or model performance.

## Rebuild

The frozen v1 artifacts, existing PostgreSQL data, 315-instrument mapping, and private T-Invest
daily artifacts must be present:

```powershell
uv run python -m apps.cli.build_event_market_dataset `
  --date-from 2022-01-01 `
  --date-to 2026-08-12 `
  --per-source-limit 2000
```

Output is written to `artifacts/event-market-predictive-dataset-v2/`. Repeating the command uses
verified page caches and must reproduce the same dataset, registry, provenance, and feature-schema
SHA values.

## Readiness

Volume and diversity are separate diagnostics. Diversity is `VERY_LOW` below five tickers, `LOW`
for five through nine, `PILOT` for ten through 24, `EXPERIMENT` for 25 through 49, and `BROAD` at
50 or more. `EVENT_MODEL_DATA_STATUS` becomes `READY_FOR_BASELINE_EXPERIMENT` only when at least
500 feature-ready rows and ten represented tickers exist. This is research metadata, never
permission to trade.

The manifest also reports top-ticker share, top-three share, issuer HHI, median events per ticker,
and p10/p90 counts. The corpus is not downsampled to hide concentration.
