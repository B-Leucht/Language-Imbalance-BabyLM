"""
Run all BabyLM scaling law + bilingual experiments.

Usage:
    caffeinate -i python run_experiments.py 2>&1 | tee run.log
"""

import subprocess
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

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

CONFIG        = "./tiny_config.json"
TOKENIZER_DIR = "./shared_tokenizer"
MAX_LENGTH    = 128
SEED          = 42
BATCH_SIZE    = 32

EPOCHS_PER_BUDGET = {
    100_000:    20,
    500_000:    10,
    1_000_000:  6,
    5_000_000:  3,
    10_000_000: 2,
}

SCALING_BUDGETS = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]

BILINGUAL_BUDGET = 100_000_000
BILINGUAL_EPOCHS = EPOCHS_PER_BUDGET[BILINGUAL_BUDGET]
BILINGUAL_RATIOS = [
    ("eng", "dut", [0.5, 0.5]),
    ("eng", "dut", [0.7, 0.3]),
    ("eng", "dut", [0.9, 0.1]),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_perplexity(stdout: str) -> dict[str, float]:
    pattern = r"(\S+):\s*perplexity\s*=\s*([\d.]+)"
    return {
        EVAL_TO_LANG.get(m.group(1), m.group(1)): float(m.group(2))
        for m in re.finditer(pattern, stdout)
    }


def train_cmd(
    langs: list[str],
    output_dir: str,
    model_name: str,
    max_tokens: int | None = None,
    ratios: list[float] | None = None,
    epochs: int = 3,
) -> str:
    dataset_str = " ".join(DATASETS[lang] for lang in langs)
    cmd = (
        f"python train.py"
        f" --dataset {dataset_str}"
        f" --config {CONFIG}"
        f" --tokenizer_dir {TOKENIZER_DIR}"
        f" --output_dir {output_dir}"
        f" --model_name {model_name}"
        f" --max_length {MAX_LENGTH}"
        f" --batch_size {BATCH_SIZE}"
        f" --epochs {epochs}"
        f" --seed {SEED}"
    )
    if max_tokens is not None:
        cmd += f" --max_tokens {max_tokens}"
    if ratios is not None:
        cmd += " --ratios " + " ".join(str(r) for r in ratios)
    return cmd


def run(cmd: str) -> subprocess.CompletedProcess:
    print(f"\n>>> {cmd}\n", flush=True)
    return subprocess.run(cmd, shell=True, executable="/bin/bash",
                          capture_output=True, text=True)


def checkpoint_path(name: str) -> Path:
    return Path(f".checkpoints/{name}.json")


def save_checkpoint(name: str, data: list[dict]) -> None:
    p = checkpoint_path(name)
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def load_checkpoint(name: str) -> list[dict]:
    p = checkpoint_path(name)
    return json.loads(p.read_text()) if p.exists() else []

# ── Step 1: shared tokenizer ───────────────────────────────────────────────────

if not Path(TOKENIZER_DIR).exists():
    print("=" * 60)
    print("Step 1: Training shared tokenizer")
    print("=" * 60)
    dataset_str = " ".join(DATASETS.values())
    cmd = (
        f"python train.py"
        f" --dataset {dataset_str}"
        f" --config {CONFIG}"
        f" --output_dir {TOKENIZER_DIR}"
        f" --model_name unused"
        f" --max_length {MAX_LENGTH}"
        f" --seed {SEED}"
        f" --epochs 0"
    )
    result = run(cmd)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR: tokenizer training failed")
        print(result.stderr)
        raise SystemExit(1)
else:
    print(f"Tokenizer already exists at {TOKENIZER_DIR}, skipping.")

# ── Step 2: monolingual scaling law runs ──────────────────────────────────────

print("\n" + "=" * 60)
print("Step 2: Monolingual scaling law runs")
print("=" * 60)

scaling_results = load_checkpoint("scaling_results")
done_keys = {(r["lang"], r["max_tokens"]) for r in scaling_results}

for lang in DATASETS:
    for budget in SCALING_BUDGETS:
        if (lang, budget) in done_keys:
            print(f"  Skipping {lang} @ {budget:,} (already done)")
            continue

        epochs = EPOCHS_PER_BUDGET[budget]
        output_dir = f"./mono-{lang}-{budget // 1_000}k"
        print(f"\n  Training {lang} @ {budget:,} tokens, {epochs} epochs")
        cmd = train_cmd(
            langs=[lang],
            output_dir=output_dir,
            model_name=f"mono-{lang}-{budget // 1_000}k",
            max_tokens=budget,
            epochs=epochs,
        )
        result = run(cmd)
        print(result.stdout[-3000:], flush=True)
        if result.returncode != 0:
            print(f"ERROR on {lang} @ {budget}")
            print(result.stderr[-1000:])
            continue

        ppls = parse_perplexity(result.stdout)
        for lang_key, ppl in ppls.items():
            scaling_results.append({"lang": lang_key, "max_tokens": budget, "perplexity": ppl})
        save_checkpoint("scaling_results", scaling_results)

scaling_df = pd.DataFrame(scaling_results)
scaling_df.to_csv("scaling_results.csv", index=False)
print(f"\nScaling results saved ({len(scaling_df)} rows)")

# ── Step 3: bilingual runs ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 3: Bilingual runs")
print("=" * 60)

bilingual_results = load_checkpoint("bilingual_results")
done_labels = {r["label"] for r in bilingual_results}

for lang_a, lang_b, ratios in BILINGUAL_RATIOS:
    label = f"{lang_a}{int(ratios[0]*100)}-{lang_b}{int(ratios[1]*100)}"
    if label in done_labels:
        print(f"  Skipping {label} (already done)")
        continue

    output_dir = f"./bi-{label}"
    print(f"\n  Training bilingual {label} @ {BILINGUAL_BUDGET:,} tokens, {BILINGUAL_EPOCHS} epochs")
    cmd = train_cmd(
        langs=[lang_a, lang_b],
        output_dir=output_dir,
        model_name=f"bi-{label}",
        max_tokens=BILINGUAL_BUDGET,
        ratios=ratios,
        epochs=BILINGUAL_EPOCHS,
    )
    result = run(cmd)
    print(result.stdout[-3000:], flush=True)
    if result.returncode != 0:
        print(f"ERROR on bilingual {label}")
        print(result.stderr[-1000:])
        continue

    ppls = parse_perplexity(result.stdout)
    for lang_key, ppl in ppls.items():
        bilingual_results.append({
            "label": label,
            "lang_a": lang_a, "lang_b": lang_b,
            "ratio_a": ratios[0], "ratio_b": ratios[1],
            "eval_lang": lang_key,
            "perplexity": ppl,
        })
    save_checkpoint("bilingual_results", bilingual_results)

bilingual_df = pd.DataFrame(bilingual_results)
bilingual_df.to_csv("bilingual_results.csv", index=False)
print(f"\nBilingual results saved ({len(bilingual_df)} rows)")

# ── Step 4: token efficiency ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 4: Computing Token Efficiency")
print("=" * 60)

scaling_law_params = {}
for lang, grp in scaling_df.groupby("lang"):
    grp = grp.sort_values("max_tokens")
    b, log_a = np.polyfit(np.log(grp["max_tokens"]), np.log(grp["perplexity"]), 1)
    scaling_law_params[lang] = (log_a, b)
    print(f"  {lang}: ppl = {np.exp(log_a):.2f} × tokens^{b:.3f}")


def mlte(lang: str, target_ppl: float) -> float:
    log_a, b = scaling_law_params[lang]
    return np.exp((np.log(target_ppl) - log_a) / b)


rows = []
for _, r in bilingual_df.iterrows():
    is_lang_a = r["eval_lang"] == r["lang_a"]
    lang = r["lang_a"] if is_lang_a else r["lang_b"]
    ratio = r["ratio_a"] if is_lang_a else r["ratio_b"]
    t_lang = BILINGUAL_BUDGET * ratio
    ppl_bi = r["perplexity"]
    mlte_val = mlte(lang, ppl_bi)
    rows.append({
        "label": r["label"],
        "eval_lang": r["eval_lang"],
        "lang": lang,
        "ratio": ratio,
        "t_lang": t_lang,
        "bilingual_ppl": ppl_bi,
        "mlte": mlte_val,
        "token_efficiency": mlte_val / t_lang,
    })

te_df = pd.DataFrame(rows)
te_df.to_csv("token_efficiency_results.csv", index=False)

print("\nToken Efficiency results:")
print(te_df[["label", "eval_lang", "ratio", "bilingual_ppl", "mlte", "token_efficiency"]].to_string(index=False))

print("\nAll done.")
