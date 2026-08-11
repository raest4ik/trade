# Real Corpus Quality & Expansion v1

## Purpose

The first real reaction-ready corpus proves the production path from issuer news to
benchmark-adjusted labels and point-in-time features. It is not suitable for model training:
all ten rows are ROSN releases from one day and all ten deterministic primary events are
`UNKNOWN`.

This audit separates three readiness questions:

- `REACTION_DATA_READINESS`: whether enough exact-timestamp reaction rows exist.
- `EVENT_ANNOTATION_READINESS`: whether a human-review sample is large enough.
- `MODEL_TRAINING_READINESS`: whether row count, issuer diversity, event diversity, and event
  feature quality jointly support training.

An adequate reaction row count cannot override poor event features. More than 50% deterministic
`UNKNOWN` produces `EVENT_FEATURE_QUALITY_BLOCKER`.

## Frozen ROSN baseline

`artifacts/corpus-quality-v1/rosn-baseline.json` freezes the original ten rows by `news_id`.
It records ten `EXACT`, matched, reaction-ready, and feature-ready rows and ten deterministic
`UNKNOWN` results. The generator reconstructs this same ID set on later runs rather than silently
selecting a new first ten.

UNKNOWN diagnosis accepts only information available at publication time:

- title and stored RSS content;
- issuer URL and source identity;
- content length, excerpt flag, and storage policy;
- `event-rules-v2` / `financial-facts-v2` output and analysis status.

Reaction returns, future volume, labels, and AI output are absent from the diagnosis input type.
The output is research triage, not gold annotation.

## Diagnostic categories

- `TRUE_NO_SUPPORTED_EVENT`: the available release concerns activities such as personnel training
  or tourism cooperation and contains no supported material corporate-event signal.
- `CONTENT_TOO_THIN`: the issuer excerpt is too short to determine whether the linked release has
  material terms.
- `SOURCE_PARSE_OR_TRUNCATION`: the stored payload has an explicit truncation or malformed-content
  signal.
- `RULE_MISS_CANDIDATE`: publication-time text contains a supported event signal that deterministic
  rules did not detect.
- `UNCERTAIN`: available evidence cannot support a stronger diagnostic category.

The categories distinguish a true ontology boundary from inadequate source content and a possible
rule miss. They must be reviewed by a human before any rules are tuned.

## Rosneft RSS fullness

The official endpoint is `https://www.rosneft.com/press/releases/rss/`. A bounded observation of
one RSS snapshot and one issuer-owned linked release confirmed:

- the feed is RSS XML;
- items contain title, an HTML description excerpt, and an issuer-owned HTTPS release link;
- the description is present on a substantially larger linked HTML release page;
- the feed does not provide the full article body.

The resulting payload assessment is `CONTENT_TOO_THIN_FOR_EVENT_EXTRACTION` where the excerpt does
not establish material terms.

No `IssuerOwnedArticleEnricher` is implemented. The approved source policy is
`EXCERPT_ALLOWED`; it does not establish permission to persist the full issuer article. The audit
does not infer a broader storage right from ordinary page accessibility. No robots, authentication,
CAPTCHA, pagination, or access control is bypassed.

## Qwen shadow only

The optional shadow command uses the existing frozen prompt, schema, and model configuration:

```text
provider=ollama
model=qwen3.5:9b
think=false
AI_RANDOM_SEED=0
```

Its predictions are written only under `artifacts/corpus-quality-v1/qwen-shadow/`. The command
hashes deterministic database analyses and the existing ML feature corpus before and after the
run. It does not persist AI output to deterministic analyses, gold data, reaction labels, or ML
features.

The comparison report measures event coverage and agreement only. It is not accuracy because no
human gold labels exist for these rows. There is deliberately no voting, fallback, reconciliation,
ensemble, winner selection, or overwrite path. A hybrid can be considered only after independent
human review establishes errors and ontology expectations.

## Source expansion policy

The audited universe is SBER/SBERP, GAZP, LKOH, ROSN, NVTK, YDEX, T, VTBR, and GMKN. SBERP shares
the SBER issuer source while ticker ambiguity remains explicit.

A source is accepted only when all of the following are established:

- issuer ownership;
- exact publication timestamp and timezone semantics;
- stable item identity;
- explicit content storage policy;
- HTTPS;
- bounded acquisition with explicit `--from`, `--to`, and `--limit`.

The first source smoke is limited to ten items and later runs to at most 100. News selection uses
only source, date range, issuer universe, source order, and limit. Returns, abnormal returns, future
volume, and other post-publication labels cannot be selection criteria.

The current accepted source remains the Rosneft issuer RSS. The other official issuer audits do not
yet establish exact timestamp semantics, a stable machine-readable endpoint, or an allowed storage
policy. Therefore the honest cumulative corpus remains ROSN-only; no third-party aggregator or
unlicensed dump is substituted to meet a row-count target.

## Pipeline and market windows

Accepted records reuse the existing path:

```text
historical acquisition -> staging -> NewsItem -> deterministic matcher
-> event-rules-v2 -> financial-facts-v2 -> MOEX/IMOEX backfill
-> reaction-v2-benchmark-adjusted -> ml-feature-dataset-v1
```

Required market windows are calculated per `EXACT` + matched publication with 60 minutes of
pre-event context, 60 minutes of post-event labels, and a bounded safety margin. No whole-market
download is introduced. Weekend, holiday, and unavailable-candle cases remain exclusions under the
existing reaction semantics.

## Reaction-ready corpus v2

`artifacts/reaction-ready-corpus-v2/` is a new generated snapshot. It does not silently change the
meaning of v1. Its funnel adds `event-analyzed` and reports deterministic known/unknown coverage and
AI shadow coverage for the actually processed subset.

Research warnings are emitted when:

- `UNKNOWN > 50%`: `HIGH_UNKNOWN_EVENT_RATE`;
- one ticker exceeds 70%: `LOW_TICKER_DIVERSITY`;
- one primary event exceeds 70%: `LOW_EVENT_DIVERSITY`.

These warnings are diagnostics, not training truth.

## Annotation Batch 002

`artifacts/corpus-quality-v1/annotation-batch-002.jsonl` is selected deterministically across
ticker/source strata and then publication order. It includes only `REAL`, `EXACT`, matched records
whose stored text is permitted by policy. Every item is `DRAFT`, `UNASSIGNED`, and `is_gold=false`.

The desired review sample is 20-50 rows. If fewer eligible real rows exist, the generator emits the
honest smaller batch and leaves annotation readiness `NOT_READY`. It never fills the batch with
synthetic data, AI answers, return-selected news, or only rule failures.

Batch 002 is not imported into `ru-corporate-events-batch-001-gold-v1`. The frozen Batch 001 SHA-256
is checked during artifact generation and remains the deterministic benchmark.

## Commands

Build publication-time diagnostics and the v2 snapshot:

```bash
uv run python -m apps.cli.build_corpus_quality_report \
  --from 2024-08-11 --to 2026-08-11 --limit 100
```

Run the optional local shadow diagnostic, then rebuild the comparison report:

```bash
AI_PROVIDER=ollama OLLAMA_MODEL=qwen3.5:9b OLLAMA_THINK=false AI_RANDOM_SEED=0 \
uv run python -m apps.cli.run_corpus_quality_shadow
```

All generated real, shadow, and report artifacts remain gitignored. No ML training, backtest,
signal generation, rules tuning, prompt tuning, or hybrid inference is part of this work.
