# Frozen Event Rules v3 Blind HOLDOUT Evaluation

## Protocol

The frozen `event-rules-v3-real-dev-candidate` was evaluated once on four independently reviewed
Batch 004 `FRESH_HOLDOUT` excerpts. Before the run, the workflow verified:

- candidate fingerprint: `3510511d1f7b3ce02a4efa245816b9422e6014088f1595b0339dcfd5be9e7f06`
- split SHA: `a32956626d194158eb69869f6bdca510456ded47ac5810ca91fe90b86aa45dea`
- gold records: 4 REAL, excerpt-only reviews
- gold distribution: OTHER=4
- gold financial facts: 0

The HOLDOUT dataset is frozen as
`ru-corporate-events-real-batch-004-holdout-gold-v1`. A single-run marker prevents an accidental
repeat evaluation.

## Results

| Metric | Value |
| --- | ---: |
| primary accuracy | 0.000000 |
| event micro precision | 0.000000 |
| event micro recall | 0.000000 |
| event micro F1 | 0.000000 |
| macro F1 | 0.000000 |
| OTHER precision | 0.000000 |
| OTHER recall | 0.000000 |
| OTHER F1 | 0.000000 |

Primary-event confusion counts:

| Gold | Prediction | Count |
| --- | --- | ---: |
| OTHER | GUIDANCE | 1 |
| OTHER | LITIGATION | 1 |
| OTHER | UNKNOWN | 2 |

Record-level results:

| news_id | Gold | Prediction | Correct |
| --- | --- | --- | --- |
| `d1b89ac6-5d97-4549-bf4a-f414f12045df` | OTHER | LITIGATION | false |
| `7cca99ed-510b-4d07-ac2d-051c1288e05d` | OTHER | GUIDANCE | false |
| `3d939700-abb4-4afc-aa87-81c1c5504a56` | OTHER | UNKNOWN | false |
| `e0c4abd0-668e-455a-b16b-e034443f2b02` | OTHER | UNKNOWN | false |

## Interpretation

This is a four-example holdout with very high statistical uncertainty. The result cannot support
a claim of general superiority or a stable population-level performance estimate. It does show
that the perfect DEVELOPMENT result did not transfer to this small blind sample.

The four records are now `OBSERVED_HOLDOUT`. They must not be used to tune v3. No rule changes,
new semantic version, reconciliation, or second evaluation are permitted from these errors. Any
future NLP change requires a new fresh dataset and a new blind holdout.

## NLP freeze

`NLP DEVELOPMENT CYCLE CLOSED`.

Frozen components:

- `event-rules-v3`
- current `financial-facts-v3`
- `qwen3.5:9b` with its frozen prompt, schema, seed, think, and context configuration

No Qwen run, hybrid, predictive ML training, backtest, or market-reaction calculation was part of
the HOLDOUT evaluation. The next project priority is `HISTORICAL_DATA_ACQUISITION`, not NLP
tuning.
