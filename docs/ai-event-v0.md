# AI Event Analyzer v0

## Decision

The deterministic v2 extractor remains the deterministic, production-capable
baseline. AI Event Analyzer v0 is a separate experimental local semantic
analyzer; it does not replace, route to, fall back to, or combine with
deterministic v2.

The default AI provider is Ollama and the default local model is `qwen3.5:9b`
with thinking disabled. The OpenAI adapter remains optional.

## Benchmark basis

`qwen3.5:9b` was selected over `qwen3:8b` because it materially improved value
normalization and semantic fact extraction on the frozen TRAIN and VALIDATION
splits without degrading event, metric, or period quality. Latency and evidence
span accuracy remain separate tradeoffs.

Benchmark runs pin the prompt hash, JSON Schema hash, model digest, random seed,
context length, and dataset hash. Local LLM execution can still vary across
Ollama versions, model builds, hardware, and runtime scheduling.

TEST is a frozen holdout. Its aggregate result documents the selected AI v0
configuration and must not be used to tune the prompt, schema, model, ontology,
or deterministic rules in this pull request.
