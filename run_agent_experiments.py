"""
Cross-lingual generalization + hyperparameter optimization experiments.
Designed for GPU execution on Lambda Labs (A100/H100).

Usage:
    export WANDB_PROJECT=language-imbalance-babylm
    python run_agent_experiments.py 2>&1 | tee agent_run.log

Covers:
  1. Hyperparameter search (lr, context length, model size) on eng @ 1M tokens
  2. eng-ind bilingual runs at 5 ratios (typologically distant pair)
  3. Trilingual runs (eng+dut+ind) at 4 ratios
  4. Token efficiency (MLTE) computation for all new results
"""

import os
import subprocess
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────

DATASETS = {
    "eng": "BabyLM-community/babylm-eng",
    "dut": "BabyLM-community/babylm-nld",
    "ind": "BabyLM-community/babylm-ind",
}

EVAL_TO_LANG = {
    "babylm-eng": "eng",
    "babylm-nld": "dut",
    "babylm-ind": "ind",
}

TOKENIZER_DIR  = "./shared_tokenizer"
SEED           = 42
WANDB_PROJECT  = os.environ.get("WANDB_PROJECT", "language-imbalance-babylm")
BUDGET         = 10_000_000
# GPU-friendly defaults (A100 can handle much larger batches than the MPS runs)
BATCH_SIZE     = 128
MAX_LENGTH     = 128
EPOCHS         = 10

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open("agent_log.md", "a") as f:
        f.write(line + "\n")


def parse_perplexity(stdout: str) -> dict[str, float]:
    pattern = r"(\S+):\s*perplexity\s*=\s*([\d.]+)"
    return {
        EVAL_TO_LANG.get(m.group(1), m.group(1)): float(m.group(2))
        for m in re.finditer(pattern, stdout)
    }


def run(cmd: str, run_name: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["WANDB_PROJECT"] = WANDB_PROJECT
    if run_name:
        env["WANDB_RUN_NAME"] = run_name
    print(f"\n>>> {cmd}\n", flush=True)
    return subprocess.run(
        cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, env=env
    )


def checkpoint_path(name: str) -> Path:
    return Path(f".checkpoints/{name}.json")


def save_checkpoint(name: str, data) -> None:
    p = checkpoint_path(name)
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def load_checkpoint(name: str):
    p = checkpoint_path(name)
    return json.loads(p.read_text()) if p.exists() else []


def train_cmd(
    langs: list[str],
    output_dir: str,
    model_name: str,
    config: str,
    max_tokens: int | None = None,
    ratios: list[float] | None = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    report_to: str = "wandb",
) -> str:
    dataset_str = " ".join(DATASETS[lang] for lang in langs)
    cmd = (
        f"python train.py"
        f" --dataset {dataset_str}"
        f" --config {config}"
        f" --tokenizer_dir {TOKENIZER_DIR}"
        f" --output_dir {output_dir}"
        f" --model_name {model_name}"
        f" --max_length {max_length}"
        f" --batch_size {batch_size}"
        f" --epochs {epochs}"
        f" --seed {SEED}"
        f" --learning_rate {learning_rate}"
        f" --warmup_ratio {warmup_ratio}"
        f" --weight_decay {weight_decay}"
        f" --report_to {report_to}"
    )
    if max_tokens is not None:
        cmd += f" --max_tokens {max_tokens}"
    if ratios is not None:
        cmd += " --ratios " + " ".join(str(r) for r in ratios)
    return cmd


# ── Session header ────────────────────────────────────────────────────────────

log("=" * 70)
log(f"Agent session start | WANDB_PROJECT={WANDB_PROJECT}")
log("=" * 70)

# ── Step 1: Hyperparameter search (eng mono @ 1M tokens, 5 epochs) ───────────

log("\n=== Step 1: Hyperparameter search (eng @ 1M tokens, 5 epochs) ===")

PARAM_SEARCH = [
    # (label, lr, config, ctx)
    ("lr5e-5-ctx128-small",  5e-5,  "./small_config.json", 128),
    ("lr1e-4-ctx128-small",  1e-4,  "./small_config.json", 128),  # baseline
    ("lr3e-4-ctx128-small",  3e-4,  "./small_config.json", 128),
    ("lr1e-4-ctx256-small",  1e-4,  "./small_config.json", 256),
    ("lr1e-4-ctx128-tiny",   1e-4,  "./tiny_config.json",  128),
]

param_results = load_checkpoint("agent_param_search")
done_params = {r["label"] for r in param_results}

for label, lr, cfg, ctx in PARAM_SEARCH:
    if label in done_params:
        log(f"  Skipping {label} (already done)")
        continue

    log(f"  Running: {label}")
    cmd = train_cmd(
        langs=["eng"],
        output_dir=f"./param-{label}",
        model_name=f"param-{label}",
        config=cfg,
        max_tokens=1_000_000,
        max_length=ctx,
        learning_rate=lr,
        epochs=5,
        batch_size=BATCH_SIZE,
    )
    result = run(cmd, run_name=f"param-{label}")
    print(result.stdout[-2000:], flush=True)
    if result.returncode != 0:
        log(f"  ERROR on {label}: {result.stderr[-500:]}")
        continue

    ppls = parse_perplexity(result.stdout)
    param_results.append({"label": label, "lr": lr, "config": cfg, "ctx": ctx, "ppls": ppls})
    save_checkpoint("agent_param_search", param_results)
    log(f"  {label} -> eng ppl={ppls.get('eng', '?'):.2f}")

# Pick best config by eng perplexity
best_params = load_checkpoint("agent_best_params")
if not best_params and param_results:
    best = min(param_results, key=lambda r: r["ppls"].get("eng", float("inf")))
    best_params = {"lr": best["lr"], "config": best["config"], "ctx": best["ctx"]}
    save_checkpoint("agent_best_params", best_params)
    log(f"  Best params: lr={best_params['lr']}, config={best_params['config']}, ctx={best_params['ctx']}")
elif best_params:
    log(f"  Using saved best params: lr={best_params['lr']}, config={best_params['config']}, ctx={best_params['ctx']}")
else:
    best_params = {"lr": 1e-4, "config": "./small_config.json", "ctx": 128}
    log(f"  No param results; falling back to defaults: {best_params}")

BEST_LR     = best_params["lr"]
BEST_CONFIG = best_params["config"]
BEST_CTX    = best_params["ctx"]

# ── Step 2: eng-ind bilingual experiments ─────────────────────────────────────

log("\n=== Step 2: eng-ind bilingual runs (typologically distant pair) ===")
log("  Hypothesis: MLTE will be lower than eng-dut, testing language distance effect")

ENG_IND_RATIOS = [
    [0.50, 0.50],
    [0.70, 0.30],
    [0.90, 0.10],
    [0.95, 0.05],
    [0.99, 0.01],
]

cross_results = load_checkpoint("agent_cross_lingual")
done_cross = {r["label"] for r in cross_results}

for ratios in ENG_IND_RATIOS:
    label = f"eng{int(ratios[0]*100)}-ind{int(ratios[1]*100)}"
    if label in done_cross:
        log(f"  Skipping {label} (already done)")
        continue

    log(f"  Training {label} @ {BUDGET:,} tokens")
    cmd = train_cmd(
        langs=["eng", "ind"],
        output_dir=f"./bi-{label}",
        model_name=f"bi-{label}",
        config=BEST_CONFIG,
        max_tokens=BUDGET,
        ratios=ratios,
        learning_rate=BEST_LR,
        max_length=BEST_CTX,
    )
    result = run(cmd, run_name=f"bi-{label}")
    print(result.stdout[-3000:], flush=True)
    if result.returncode != 0:
        log(f"  ERROR on {label}: {result.stderr[-500:]}")
        continue

    ppls = parse_perplexity(result.stdout)
    cross_results.append({
        "label": label, "lang_a": "eng", "lang_b": "ind",
        "ratio_a": ratios[0], "ratio_b": ratios[1], "ppls": ppls,
    })
    save_checkpoint("agent_cross_lingual", cross_results)
    log(f"  {label} -> ppl: eng={ppls.get('eng', '?'):.2f}, ind={ppls.get('ind', '?'):.2f}")

# ── Step 3: Trilingual experiments ───────────────────────────────────────────

log("\n=== Step 3: Trilingual runs (eng+dut+ind) ===")
log("  Hypothesis: minority-language MLTE >= worst bilingual scenario")

TRILINGUAL_RATIOS = [
    [1/3,  1/3,  1/3 ],   # equal
    [0.60, 0.20, 0.20],   # eng-dominant
    [0.80, 0.10, 0.10],   # strongly eng-dominant
    [0.50, 0.30, 0.20],   # asymmetric minority weighting
]

tri_results = load_checkpoint("agent_trilingual")
done_tri = {r["label"] for r in tri_results}

for ratios in TRILINGUAL_RATIOS:
    label = f"tri-eng{int(ratios[0]*100)}-dut{int(ratios[1]*100)}-ind{int(ratios[2]*100)}"
    if label in done_tri:
        log(f"  Skipping {label} (already done)")
        continue

    log(f"  Training {label} @ {BUDGET:,} tokens")
    cmd = train_cmd(
        langs=["eng", "dut", "ind"],
        output_dir=f"./{label}",
        model_name=label,
        config=BEST_CONFIG,
        max_tokens=BUDGET,
        ratios=ratios,
        learning_rate=BEST_LR,
        max_length=BEST_CTX,
    )
    result = run(cmd, run_name=label)
    print(result.stdout[-3000:], flush=True)
    if result.returncode != 0:
        log(f"  ERROR on {label}: {result.stderr[-500:]}")
        continue

    ppls = parse_perplexity(result.stdout)
    tri_results.append({
        "label": label,
        "ratio_eng": ratios[0], "ratio_dut": ratios[1], "ratio_ind": ratios[2],
        "ppls": ppls,
    })
    save_checkpoint("agent_trilingual", tri_results)
    log(f"  {label} -> ppl: eng={ppls.get('eng', '?'):.2f}, "
        f"dut={ppls.get('dut', '?'):.2f}, ind={ppls.get('ind', '?'):.2f}")

# ── Step 4: Token efficiency for new results ─────────────────────────────────

log("\n=== Step 4: Token efficiency (MLTE) for new experiments ===")

scaling_csv = Path("scaling_results.csv")
if not scaling_csv.exists():
    log("  WARNING: scaling_results.csv not found — skipping MLTE computation")
else:
    scaling_df = pd.read_csv(scaling_csv)
    scaling_law_params: dict[str, tuple[float, float]] = {}
    for lang, grp in scaling_df.groupby("lang"):
        grp = grp.sort_values("max_tokens")
        b, log_a = np.polyfit(np.log(grp["max_tokens"]), np.log(grp["perplexity"]), 1)
        scaling_law_params[lang] = (log_a, b)
        log(f"  Scaling law {lang}: ppl = {np.exp(log_a):.2f} * tokens^{b:.3f}")

    def mlte(lang: str, target_ppl: float) -> float:
        log_a, b = scaling_law_params[lang]
        return np.exp((np.log(target_ppl) - log_a) / b)

    rows = []
    for r in cross_results:
        for eval_lang, ppl in r["ppls"].items():
            is_a = eval_lang == r["lang_a"]
            lang  = r["lang_a"] if is_a else r["lang_b"]
            ratio = r["ratio_a"] if is_a else r["ratio_b"]
            t_lang = BUDGET * ratio
            if lang in scaling_law_params:
                m = mlte(lang, ppl)
                rows.append({
                    "experiment": "cross_lingual",
                    "label": r["label"], "eval_lang": eval_lang, "lang": lang,
                    "ratio": ratio, "t_lang": t_lang,
                    "bilingual_ppl": ppl, "mlte": m, "token_efficiency": m / t_lang,
                })

    for r in tri_results:
        lang_ratio_map = {"eng": r["ratio_eng"], "dut": r["ratio_dut"], "ind": r["ratio_ind"]}
        for eval_lang, ppl in r["ppls"].items():
            ratio = lang_ratio_map.get(eval_lang, None)
            if ratio is None or eval_lang not in scaling_law_params:
                continue
            t_lang = BUDGET * ratio
            m = mlte(eval_lang, ppl)
            rows.append({
                "experiment": "trilingual",
                "label": r["label"], "eval_lang": eval_lang, "lang": eval_lang,
                "ratio": ratio, "t_lang": t_lang,
                "bilingual_ppl": ppl, "mlte": m, "token_efficiency": m / t_lang,
            })

    if rows:
        te_df = pd.DataFrame(rows)
        te_df.to_csv("agent_token_efficiency.csv", index=False)
        log(f"\n  Token efficiency results ({len(te_df)} rows) saved to agent_token_efficiency.csv")
        log("\n" + te_df[["label", "eval_lang", "ratio", "bilingual_ppl", "token_efficiency"]].to_string(index=False))

        # Upload to W&B
        try:
            import wandb
            wandb.init(project=WANDB_PROJECT, name="token-efficiency-summary", job_type="analysis")
            wandb.log({"token_efficiency_table": wandb.Table(dataframe=te_df)})
            wandb.finish()
        except Exception as e:
            log(f"  W&B upload skipped: {e}")

# ── Done ─────────────────────────────────────────────────────────────────────

log("\n=== Session complete ===")
log("Results: agent_token_efficiency.csv | W&B: language-imbalance-babylm | Log: agent_log.md")
