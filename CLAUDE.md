# BabyLM Language Imbalance — Agent Context

## Project Summary

This repository studies **how language imbalance affects multilingual language model performance** using small GPT-2-style models trained from scratch on the BabyBabelLM corpora. The core metric is **Monolingual Token Equivalent (MLTE)**: how many monolingual tokens would be needed to match the perplexity achieved by bilingual training. MLTE > 1 means bilingual training is "free" transfer; MLTE < 1 means it hurts.

Research questions:
1. How does training data ratio affect per-language perplexity?
2. Does bilingual training provide efficient cross-lingual transfer?
3. Does transfer strength depend on language similarity (eng-dut vs eng-ind)?
4. Does trilingual training compound or dilute transfer effects?

## Languages & Datasets

| Short | Language   | HF Dataset                     | Family       |
|-------|------------|--------------------------------|--------------|
| eng   | English    | BabyLM-community/babylm-eng    | Germanic     |
| dut   | Dutch      | BabyLM-community/babylm-nld    | Germanic     |
| ind   | Indonesian | BabyLM-community/babylm-ind    | Austronesian |

English and Dutch are closely related (both West Germanic). Indonesian is typologically distant — different word order, morphology, no shared cognates. This makes **eng-ind** an interesting test of whether transfer can cross language families.

## Model Architecture

All models are GPT-2 causal LMs (decoder-only), trained from scratch with a shared BPE tokenizer (30k vocab) trained on all three languages.

| Config file            | n_embd | n_inner | n_head | n_layer | ~Params |
|------------------------|--------|---------|--------|---------|---------|
| `tiny_config.json`     | 128    | 512     | 4      | 10      | ~2M     |
| `small_config.json`    | 320    | 1280    | 4      | 8       | ~10M    |
| `twelveMio_config.json`| 256    | 1024    | 8      | 6       | ~12M    |

## Existing Results (completed before this session)

All runs used `tiny_config.json`, 10 epochs, 128-token context, batch size 32, lr=1e-4.

### Monolingual scaling laws
All three languages (eng, dut, ind) trained at 100k / 500k / 1M / 5M / 10M tokens.
Results: `scaling_results.csv`. Scaling law fit: `ppl ∝ tokens^b` with b ≈ -0.3 to -0.4.

### Bilingual eng-dut (10M total tokens)
Ratios: 50/50, 70/30, 90/10, 95/5, 99/1.
Results: `bilingual_results.csv`, `token_efficiency_results.csv`.

**Key finding**: Dutch (minority language) achieves MLTE ≈ 1.4–1.5 at 90/10 ratio — bilingual training is significantly more efficient. At 99/1 the benefit nearly disappears.

## What to Experiment Next

You are running on a Lambda Labs A100/H100 GPU. Use `small_config.json` (10M params) as the primary config — it fits easily on an A100 and gives better representations than tiny.

### Priority 1 — Hyperparameter optimization (fast, run first)
Use eng monolingual @ 1M tokens, 5 epochs. Find the best combo of:
- Learning rate: 5e-5, **1e-4** (current), 3e-4
- Context length: 128, 256
- Config: tiny vs small

This is cheap (~5 runs, ~5 min each on A100) and informs all subsequent experiments.

### Priority 2 — eng-ind bilingual (main new experiment)
Run at the same ratios as eng-dut (50/50, 70/30, 90/10, 95/5, 99/1), 10M tokens.
Compare MLTE to eng-dut. **Hypothesis**: transfer is weaker for eng-ind due to typological distance.

### Priority 3 — Trilingual experiments (eng + dut + ind)
Ratios: equal (33/33/33), English-dominant (60/20/20), very dominant (80/10/10), asymmetric (50/30/20).
**Hypothesis**: trilingual MLTE for minority languages is higher than the worst bilingual scenario predicts, because shared subword representations across all three languages provide additional leverage.

### Priority 4 — Monolingual scaling with small_config (if time allows)
Redo scaling law runs with small_config to check if the power-law exponent changes with model size.

## Setup on Lambda Labs

```bash
git clone https://github.com/B-Leucht/Language-Imbalance-BabyLM
cd Language-Imbalance-BabyLM
pip install -r requirements.txt
wandb login   # paste API key when prompted
export WANDB_PROJECT=language-imbalance-babylm
```

The shared tokenizer is already committed at `./shared_tokenizer` — do not retrain it.

## Running Experiments

```bash
# Upload existing results to W&B first
python upload_existing.py

# Run all new experiments (resumable — uses .checkpoints/ for state)
python run_agent_experiments.py 2>&1 | tee agent_run.log
```

`run_agent_experiments.py` covers all four priorities above in order, skipping already-completed runs.

## Logging

Write your findings, decisions, and hypotheses to **`agent_log.md`** throughout the session. Use timestamped entries. Include:
- Which experiment you're starting and why
- Any parameter choices that differ from defaults, with reasoning
- Intermediate findings after each experiment block
- A final summary of what was learned and what remains open

All training metrics go to W&B automatically (project: `language-imbalance-babylm`). Per-language perplexity is logged as `perplexity/eng`, `perplexity/dut`, `perplexity/ind`.

## MLTE Formula

Given monolingual scaling law fit `(log_a, b)` for a language:

```
mlte(lang, target_ppl) = exp((log(target_ppl) - log_a) / b)
token_efficiency = mlte / actual_token_budget
```

`token_efficiency > 1` = bilingual training used tokens more efficiently than monolingual.

## Key Files

| File                        | Purpose                                        |
|-----------------------------|------------------------------------------------|
| `train.py`                  | Core training script (all config via CLI args) |
| `run_experiments.py`        | Original scaling + eng-dut bilingual script    |
| `run_agent_experiments.py`  | New GPU script: hyperparam + cross-lingual     |
| `upload_existing.py`        | Upload prior CSV results to W&B                |
| `agent_log.md`              | Agent session log (append entries here)        |
| `small_config.json`         | Recommended config for this session            |
| `shared_tokenizer/`         | Pre-trained BPE tokenizer — do not retrain     |
| `.checkpoints/`             | Run state for resumable execution              |
| `scaling_results.csv`       | Existing monolingual scaling law data          |
| `bilingual_results.csv`     | Existing eng-dut bilingual results             |
| `token_efficiency_results.csv` | Existing MLTE data                          |
