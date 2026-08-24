# exact-event-official-domain-registry-enrichment-v1

Data-centric official-domain registry enrichment for underrepresented exact-event issuers.

This artifact exists because the previous bounded live source snapshot closed only part of the
problem: source discovery executed, but many priority tickers were blocked by
`NO_OFFICIAL_DOMAIN`. The new official domain registry is not a source registry. A confirmed
company domain is only a seed for later source discovery; it does not imply RSS, exact timestamps,
archive coverage, or canonical events.

## Inputs

- `INPUT_DATASET_SHA=62908b80f854c09c928bfd608009ea003ee887bcc93420b74ac556e0914853c4`
- Existing v5 official source discovery artifact.
- Optional PR #43 `source-report.jsonl`; if absent, the cohort falls back to existing v5 priority
  logic and excludes tickers that already have a proven source/domain.
- Optional zero-cost candidate-domain seed file produced outside CI from public discovery.

## Bounds

- `MAX_TICKERS=50`
- `MAX_SEARCH_QUERIES_PER_TICKER=2`
- `MAX_CANDIDATE_DOMAINS_PER_TICKER=5`
- `MAX_VALIDATION_URLS_PER_DOMAIN=5`
- `MAX_REQUESTS_PER_DOMAIN=10`
- `REQUEST_TIMEOUT_SECONDS=10.0`
- `MAX_RESPONSE_BYTES=1000000`
- `MAX_REDIRECTS=3`
- `MIN_DOMAIN_DELAY_SECONDS=0.5`

The bounds are fixed before live execution and must not be increased after seeing results.

## Acceptance

`official_domain_confirmed=true` requires issuer identity evidence from a public official page,
exchange/regulatory evidence, or an issuer-owned page that identifies the same legal/company
entity. Search results are discovery only and are never proof by themselves.

Fail-closed blockers include `NO_CANDIDATE_DOMAIN`, `OFFICIAL_DOMAIN_AMBIGUOUS`,
`LEGAL_ENTITY_MISMATCH`, `PARENT_SUBSIDIARY_AMBIGUITY`, `NO_IDENTITY_EVIDENCE`,
`ROBOTS_BLOCKED`, `RATE_LIMITED`, `AUTH_REQUIRED`, `CAPTCHA_BLOCKED`, `PAYMENT_REQUIRED`,
`TLS_FAILED`, `DNS_FAILED`, `TIMEOUT`, `HTTP_4XX`, `HTTP_5XX`, `RESPONSE_TOO_LARGE`,
`UNSUPPORTED_CONTENT_TYPE`, and `TECHNICAL_FETCH_FAILED`.

## Safety

- `DATA_COST_RUB=0`
- `TINVEST_READONLY_USED=false`
- `MODEL_TRAINING_PERFORMED=false`
- `TEST_OUTCOME_USED=false`
- `TEST_EVALUATION_PERFORMED=false`
- `FUTURE_EVENT_HOLDOUT_USED=false`
- `FUTURE_EVENT_HOLDOUT_OBSERVED=false`
- `DATE_ONLY_COERCIONS=0`
- `FETCH_TIME_USED_AS_PUBLICATION_TIME=false`
- `RULES_V3_CHANGED=false`
- `QWEN_CHANGED=false`
- `NLP_TUNING_PERFORMED=false`
- `STRICT_EXACT_METHODOLOGY_CHANGED=false`
- `SPARSE_FAMILY_CREATED=false`
- `BACKTEST_APPROVED=false`
- `PAPER_TRADING_APPROVED=false`
- `REAL_TRADING_APPROVED=false`

The enrichment does not acquire market candles, does not run models, does not inspect TEST
outcomes, does not perform market maturation, and does not submit or simulate trades.
