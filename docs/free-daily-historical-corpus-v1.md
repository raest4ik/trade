# Free Daily Historical Corpus v1

## Purpose

The zero-cost source audit found far more official historical releases with a reliable calendar date than releases with a defensible publication clock time. Relaxing the existing timestamp contract would contaminate intraday labels, so this project keeps two explicit and independent label families:

- `EXACT_INTRADAY` retains the existing `reaction-v2-benchmark-adjusted` 1/5/15/30/60-minute semantics.
- `DATE_SAFE_DAILY` uses `date-safe-daily-reaction-v1` and accepts `EXACT` or `DATE_ONLY` source timestamps without converting either one.

## Leakage-safe window

For source publication date `D`, the baseline is the closing minute of the last common MOEX security/IMOEX trading session strictly before `D`. The target is the closing minute of the first common session strictly after `D`. Security and IMOEX must use identical session dates. No candle from `D` is used, even when the release has an exact clock time.

The raw label is:

```text
security_return = target_security_close / baseline_security_close - 1
benchmark_return = target_imoex_close / baseline_imoex_close - 1
abnormal_return = security_return - benchmark_return
```

Daily feature rows use the separate `ml-daily-features-v1` namespace. Every feature is available by the baseline close; target prices and returns remain in the label object. Existing `ml-features-v1` is unchanged.

## Source acceptance

An archive is accepted only when issuer ownership or other provenance, free automated access, stable identity, source publication date, and storage policy are all verified. A visible web page is not enough. Verification samples are deterministic, bounded to 20 items per source, and independent of event predictions and market outcomes.

The v1 review covered SBER, Gazprom, LUKOIL, NOVATEK, T-Bank, VTB, Nornickel, and the Yandex yearly archive. LUKOIL, NOVATEK, VTB, Nornickel, and Yandex expose useful date metadata, but their automation and excerpt-storage permissions were not sufficiently established. The remaining sources also have access or technical blockers. Consequently, no archive is newly marked `COMPLIANT_DATE_SAFE_DAILY`, and no historical item is imported merely to meet a row target.

The existing bounded ROSN and YDEX official RSS collectors remain the accepted exact sources. Their records may receive a separate daily label when complete common market sessions exist.

The bounded v1 build imported no new historical news because no additional archive passed every policy gate. The cumulative REAL corpus remains 40 EXACT rows across ROSN and YDEX. Of 35 unambiguously matched records, event-adjacent MOEX/IMOEX windows produced 34 daily reaction rows and 34 daily feature rows. The newest matched release has no completed post-publication session in the observed data and remains explicitly excluded. The resulting readiness is `NOT_READY`, with 66 additional feature-ready rows needed to reach the first threshold and ticker diversity still below the three-ticker diagnostic gate.

## Dataset and split

Generated outputs live under `artifacts/free-daily-historical-v1/` and are gitignored. `daily-reactions.jsonl` stores the explicit label family and reaction version. `daily-feature-dataset.jsonl` stores baseline-only features and a separate label object. The split is deterministic chronological 70/15/15: older rows are TRAIN, newer rows VALIDATION, and newest rows TEST. Returns, event types, Rules, and Qwen outputs do not influence selection or splitting.

Readiness is based on daily feature-ready rows: fewer than 100 is `NOT_READY`, 100-499 is `DAILY_PILOT_READY`, 500-999 is `DAILY_BASELINE_EXPERIMENT_READY`, and at least 1000 is `DAILY_BASELINE_TRAINING_READY`. Insufficient ticker, source, or time diversity can downgrade readiness.

## Growth path

The path to 100, 500, and 1000 rows remains zero-cost: keep accumulating accepted exact RSS items and re-audit official archives only when their automation and storage policies become provable. Paid fallbacks, access-control bypasses, return-based sampling, NLP retuning, hybrid inference, and predictive model training are outside this work.
