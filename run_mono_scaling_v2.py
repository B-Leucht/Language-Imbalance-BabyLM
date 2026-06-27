"""
Monolingual scaling runs v2 — small config, shared_tokenizer2 (includes Javanese).

Languages : eng, nld, ind, jav
Token sizes: 1M, 3M, 10M, 30M, 50M
Hyperparams: lr=2e-4, epochs=8, batch_size=32, max_length=128
Config     : small_config.json
Tokenizer  : ./shared_tokenizer2
Hub upload : yes — models pushed to HuggingFace Hub after each run

Resumable via .checkpoints/mono_scaling_v2.json.
Results saved to mono_scaling_v2.csv.
"""

import os
import csv
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────────

DATASETS = {
    "eng": "BabyLM-community/babylm-eng",
    "nld": "BabyLM-community/babylm-nld",
    "ind": "BabyLM-community/babylm-ind",
    "jav": "BabyLM-community/babylm-jav",
}

EVAL_TO_LANG = {
    "babylm-eng": "eng",
    "babylm-nld": "nld",
    "babylm-ind": "ind",
    "babylm-jav": "jav",
}

TOKENIZER_DIR = "./shared_tokenizer2"
CONFIG        = "./small_config.json"
SEED          = 42
LR            = 2e-4
EPOCHS        = 8
BATCH_SIZE    = 32
MAX_LENGTH    = 128

TOKEN_SIZES = [1_000_000, 3_000_000, 10_000_000, 30_000_000, 50_000_000]
LANGUAGES   = ["eng", "nld", "ind", "jav"]

CHECKPOINT   = ".checkpoints/mono_scaling_v2.json"
RESULTS_CSV  = "mono_scaling_v2.csv"

# ── Helpers ──────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open("agent_log.md", "a") as f:
        f.write(line + "\n")


def load_checkpoint() -> list[dict]:
    p = Path(CHECKPOINT)
    return json.loads(p.read_text()) if p.exists() else []


def save_checkpoint(data: list[dict]) -> None:
    p = Path(CHECKPOINT)
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def parse_perplexity(stdout: str) -> dict[str, float]:
    pattern = r"(\S+):\s*perplexity\s*=\s*([\d.]+)"
    return {
        EVAL_TO_LANG.get(m.group(1), m.group(1)): float(m.group(2))
        for m in re.finditer(pattern, stdout)
    }


def run_cmd(cmd: str, run_name: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["WANDB_RUN_NAME"] = run_name
    print(f"\n>>> {cmd}\n", flush=True)
    return subprocess.run(cmd, shell=True, executable="/bin/bash",
                          capture_output=True, text=True, env=env)


def cleanup(output_dir: str) -> None:
    p = Path(output_dir)
    if p.exists():
        shutil.rmtree(p)
        print(f"  [cleanup] Removed {output_dir}", flush=True)


def append_csv(row: dict) -> None:
    p = Path(RESULTS_CSV)
    write_header = not p.exists()
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "lang", "tokens", "ppl",
                                               "lr", "epochs", "batch_size", "config"])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Main ─────────────────────────────────────────────────────────────────────────

log("=" * 70)
log("Mono scaling v2 | small_config | shared_tokenizer2 | lr=2e-4 | 8 epochs | bs=32")
log("Languages: eng, nld, ind, jav | Tokens: 1M 3M 10M 30M 50M")
log("=" * 70)

results = load_checkpoint()
done = {r["label"] for r in results}

for lang in LANGUAGES:
    for tokens in TOKEN_SIZES:
        t_label = f"{tokens // 1_000_000}m"
        label = f"small-{lang}-{t_label}-{EPOCHS}ep-2e-4-{BATCH_SIZE}b"

        if label in done:
            log(f"  Skipping {label} (already done)")
            continue

        log(f"  Training {label} ...")
        output_dir = f"./{label}"

        hub_name = f"B-Leucht/{label}"
        cmd = (
            f"python train.py"
            f" --dataset {DATASETS[lang]}"
            f" --config {CONFIG}"
            f" --tokenizer_dir {TOKENIZER_DIR}"
            f" --output_dir {output_dir}"
            f" --model_name {hub_name}"
            f" --max_length {MAX_LENGTH}"
            f" --batch_size {BATCH_SIZE}"
            f" --epochs {EPOCHS}"
            f" --seed {SEED}"
            f" --learning_rate {LR}"
            f" --max_tokens {tokens}"
            f" --push_to_hub"
            f" --report_to none"
        )

        result = run_cmd(cmd, run_name=label)
        print(result.stdout[-3000:], flush=True)

        if result.returncode != 0:
            log(f"  ERROR on {label}:\n{result.stderr[-1000:]}")
            cleanup(output_dir)
            continue

        ppls = parse_perplexity(result.stdout)
        ppl = ppls.get(lang, float("nan"))

        entry = {
            "label": label, "lang": lang, "tokens": tokens,
            "ppl": ppl, "ppls": ppls,
            "lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        }
        results.append(entry)
        save_checkpoint(results)

        append_csv({
            "label": label, "lang": lang, "tokens": tokens, "ppl": ppl,
            "lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "config": "small",
        })

        log(f"  {label} -> ppl={ppl:.2f}  (all langs: {ppls})")
        cleanup(output_dir)

log("=" * 70)
log(f"Done. {len(results)} runs completed. Results in {RESULTS_CSV}")
log("=" * 70)
