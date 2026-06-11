"""
Agent experiments v2 — filling bilingual gaps, trilingual runs, and validation.

Matches existing experimental setup for comparability:
  small_config.json | 50M tokens | 3 epochs | batch_size=128 | lr=1e-4

Skips Priority 1 (hyperparam): existing data clearly shows small > tiny, lr=1e-4 solid.
  tiny@50M:  eng=111.6, nld=128.6, ind=111.1
  small@50M: eng=59.7,  nld=55.6,  ind=62.4   (1.8–2.3x better)

Skips already-completed eng-ind / eng-nld runs at 50/50, 70/30, 90/10.

New runs:
  Step 1: eng-nld  95/5 and 99/1  (completing the ratio sweep)
  Step 2: eng-ind  95/5 and 99/1  (completing the ratio sweep)
  Step 3: Trilingual eng+dut+ind at 4 ratios (fully new)
  Step 4: Matched mono baselines (batch=128, warmup=0.05, wd=0.01) for valid MLTE
          NOTE: CSV mono baselines used batch=32, no warmup, no weight_decay.
          These matched runs enable unbiased MLTE for Steps 1-3.
  Step 5: Multi-seed validation — eng-nld 90/10 and eng-ind 90/10 with seeds 0,1
          Uses batch=32 (matching CSV) to directly test if ind>nld is noise.
  Step 6: MLTE computation
          - CSV bilingual vs CSV mono: original baselines (internally valid)
          - New bilingual/trilingual vs matched mono baselines (Step 4)
"""

import os
import shutil
import subprocess
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

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

# Map script internal lang keys -> babylm_runs_tidy.csv lang names (for scaling law lookup)
LANG_TO_CSV = {"eng": "eng", "dut": "nld", "ind": "ind"}

TOKENIZER_DIR = "./shared_tokenizer"
SEED          = 42
BUDGET        = 50_000_000   # match existing 50M token runs
BATCH_SIZE    = 128
MAX_LENGTH    = 128
EPOCHS        = 3            # match existing 3-epoch runs
CONFIG        = "./small_config.json"
LR            = 1e-4


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def run_cmd(cmd: str, run_name: str | None = None, retries: int = 3) -> subprocess.CompletedProcess:
    import time
    env = dict(os.environ)
    # Use cached datasets offline to avoid gated-dataset auth timeouts
    env["HF_DATASETS_OFFLINE"] = "1"
    if run_name:
        env["WANDB_RUN_NAME"] = run_name
    print(f"\n>>> {cmd}\n", flush=True)
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, env=env
        )
        if result.returncode == 0:
            return result
        is_network_err = any(s in result.stderr for s in (
            "ReadTimeout", "ConnectionError", "ConnectTimeout",
            "DatasetNotFoundError", "OfflineModeIsEnabled",
        ))
        if is_network_err and attempt < retries:
            wait = 30 * attempt
            log(f"  Network error on attempt {attempt}/{retries}, retrying in {wait}s ...")
            time.sleep(wait)
            continue
        return result
    return result


def checkpoint_path(name: str) -> Path:
    return Path(f".checkpoints/{name}.json")


def save_checkpoint(name: str, data) -> None:
    p = checkpoint_path(name)
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def load_checkpoint(name: str):
    p = checkpoint_path(name)
    return json.loads(p.read_text()) if p.exists() else []


def cleanup_model(output_dir: str) -> None:
    p = Path(output_dir)
    if p.exists():
        shutil.rmtree(p)
        print(f"  [cleanup] Removed {output_dir}", flush=True)


def train_cmd(
    langs: list[str],
    output_dir: str,
    model_name: str,
    max_tokens: int | None = None,
    ratios: list[float] | None = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    learning_rate: float = LR,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    seed: int = SEED,
) -> str:
    dataset_str = " ".join(DATASETS[lang] for lang in langs)
    cmd = (
        f"python train.py"
        f" --dataset {dataset_str}"
        f" --config {CONFIG}"
        f" --tokenizer_dir {TOKENIZER_DIR}"
        f" --output_dir {output_dir}"
        f" --model_name {model_name}"
        f" --max_length {max_length}"
        f" --batch_size {batch_size}"
        f" --epochs {epochs}"
        f" --seed {seed}"
        f" --learning_rate {learning_rate}"
        f" --warmup_ratio {warmup_ratio}"
        f" --weight_decay {weight_decay}"
        f" --report_to none"
    )
    if max_tokens is not None:
        cmd += f" --max_tokens {max_tokens}"
    if ratios is not None:
        cmd += " --ratios " + " ".join(str(r) for r in ratios)
    return cmd


# ── Session header ──────────────────────────────────────────────────────────────

log("=" * 70)
log("Agent session v2 | small_config | 50M tokens | 3 epochs | no W&B")
log("=" * 70)
log("Rationale for skipping hyperparam search:")
log("  Existing mono data (small vs tiny @ 50M): small is 1.8-2.3x better")
log("  lr=1e-4 is the validated default across all prior runs")
log("  ctx=128 is sufficient; expanding to 256 would quadratically increase compute")
log("Proceeding to fill missing bilingual ratios, then trilingual.")

# ── Step 1: eng-nld — missing ratios 95/5 and 99/1 ────────────────────────────

log("\n=== Step 1: eng-nld bilingual — completing ratio sweep (95/5, 99/1) ===")
log("  Already done: 50/50, 70/30, 90/10 @ 50M (small config)")
log("  Key question: does Dutch benefit at very low exposure (5%, 1%)?")

ENG_NLD_NEW = [[0.95, 0.05], [0.99, 0.01]]

nld_results = load_checkpoint("agent_eng_nld_new")
done_nld = {r["label"] for r in nld_results}

for ratios in ENG_NLD_NEW:
    label = f"eng{int(ratios[0]*100)}-nld{int(ratios[1]*100)}"
    if label in done_nld:
        log(f"  Skipping {label} (already done)")
        continue

    log(f"  Training {label} @ {BUDGET:,} tokens ...")
    output_dir = f"./bi-{label}"
    cmd = train_cmd(
        langs=["eng", "dut"],
        output_dir=output_dir,
        model_name=f"bi-{label}",
        max_tokens=BUDGET,
        ratios=ratios,
    )
    result = run_cmd(cmd, run_name=f"bi-{label}")
    print(result.stdout[-3000:], flush=True)
    if result.returncode != 0:
        log(f"  ERROR on {label}:\n{result.stderr[-1000:]}")
        continue

    ppls = parse_perplexity(result.stdout)
    nld_results.append({
        "label": label, "lang_a": "eng", "lang_b": "dut",
        "ratio_a": ratios[0], "ratio_b": ratios[1], "ppls": ppls,
    })
    save_checkpoint("agent_eng_nld_new", nld_results)
    log(f"  {label} -> eng={ppls.get('eng', '?'):.2f}, dut={ppls.get('dut', '?'):.2f}")
    cleanup_model(output_dir)

# ── Step 2: eng-ind — missing ratios 95/5 and 99/1 ────────────────────────────

log("\n=== Step 2: eng-ind bilingual — completing ratio sweep (95/5, 99/1) ===")
log("  Already done: 50/50, 70/30, 90/10 @ 50M (small config)")
log("  Key question: does Indonesian still transfer at very low exposure?")

ENG_IND_NEW = [[0.95, 0.05], [0.99, 0.01]]

ind_results = load_checkpoint("agent_eng_ind_new")
done_ind = {r["label"] for r in ind_results}

for ratios in ENG_IND_NEW:
    label = f"eng{int(ratios[0]*100)}-ind{int(ratios[1]*100)}"
    if label in done_ind:
        log(f"  Skipping {label} (already done)")
        continue

    log(f"  Training {label} @ {BUDGET:,} tokens ...")
    output_dir = f"./bi-{label}"
    cmd = train_cmd(
        langs=["eng", "ind"],
        output_dir=output_dir,
        model_name=f"bi-{label}",
        max_tokens=BUDGET,
        ratios=ratios,
    )
    result = run_cmd(cmd, run_name=f"bi-{label}")
    print(result.stdout[-3000:], flush=True)
    if result.returncode != 0:
        log(f"  ERROR on {label}:\n{result.stderr[-1000:]}")
        continue

    ppls = parse_perplexity(result.stdout)
    ind_results.append({
        "label": label, "lang_a": "eng", "lang_b": "ind",
        "ratio_a": ratios[0], "ratio_b": ratios[1], "ppls": ppls,
    })
    save_checkpoint("agent_eng_ind_new", ind_results)
    log(f"  {label} -> eng={ppls.get('eng', '?'):.2f}, ind={ppls.get('ind', '?'):.2f}")
    cleanup_model(output_dir)

# ── Step 3: Trilingual experiments ─────────────────────────────────────────────

log("\n=== Step 3: Trilingual (eng+dut+ind) — 4 ratios ===")
log("  Hypothesis: minority MLTE in trilingual >= worst bilingual scenario")
log("  Because: shared subword representations across 3 languages give extra leverage")

TRILINGUAL_RATIOS = [
    [1/3,  1/3,  1/3 ],   # equal
    [0.60, 0.20, 0.20],   # eng-dominant
    [0.80, 0.10, 0.10],   # strongly eng-dominant
    [0.50, 0.30, 0.20],   # asymmetric minority weighting
]

tri_results = load_checkpoint("agent_trilingual")
done_tri = {r["label"] for r in tri_results}

for ratios in TRILINGUAL_RATIOS:
    e_pct = int(round(ratios[0] * 100))
    d_pct = int(round(ratios[1] * 100))
    i_pct = int(round(ratios[2] * 100))
    label = f"tri-eng{e_pct}-dut{d_pct}-ind{i_pct}"
    if label in done_tri:
        log(f"  Skipping {label} (already done)")
        continue

    log(f"  Training {label} @ {BUDGET:,} tokens ...")
    output_dir = f"./{label}"
    cmd = train_cmd(
        langs=["eng", "dut", "ind"],
        output_dir=output_dir,
        model_name=label,
        max_tokens=BUDGET,
        ratios=ratios,
    )
    result = run_cmd(cmd, run_name=label)
    print(result.stdout[-3000:], flush=True)
    if result.returncode != 0:
        log(f"  ERROR on {label}:\n{result.stderr[-1000:]}")
        continue

    ppls = parse_perplexity(result.stdout)
    tri_results.append({
        "label": label,
        "ratio_eng": ratios[0], "ratio_dut": ratios[1], "ratio_ind": ratios[2],
        "ppls": ppls,
    })
    save_checkpoint("agent_trilingual", tri_results)
    log(
        f"  {label} -> eng={ppls.get('eng', '?'):.2f}, "
        f"dut={ppls.get('dut', '?'):.2f}, ind={ppls.get('ind', '?'):.2f}"
    )
    cleanup_model(output_dir)

# ── Step 4: Matched mono baselines ─────────────────────────────────────────────
# The CSV mono baselines used batch=32, no warmup, no weight_decay.
# Our new bilingual/trilingual runs use batch=128, warmup=0.05, wd=0.01.
# These runs provide matched scaling laws for unbiased MLTE of the new runs.

log("\n=== Step 4: Matched mono baselines (batch=128, warmup=0.05, wd=0.01) ===")
log("  CSV mono used batch=32 — these matched runs fix the hyperparameter confound")
log("  Token counts: 1M, 5M, 10M, 50M × 3 languages = 12 runs")

MONO_TOKENS = [1_000_000, 5_000_000, 10_000_000, 50_000_000]

mono_matched = load_checkpoint("agent_mono_matched")
done_mono_matched = {r["label"] for r in mono_matched}

for lang in ["eng", "dut", "ind"]:
    for tokens in MONO_TOKENS:
        t_label = f"{tokens // 1_000_000}M"
        label = f"mono-{lang}-{t_label}"
        if label in done_mono_matched:
            log(f"  Skipping {label} (already done)")
            continue
        log(f"  Training {label} ...")
        output_dir = f"./{label}"
        cmd = train_cmd(
            langs=[lang],
            output_dir=output_dir,
            model_name=label,
            max_tokens=tokens,
        )
        result = run_cmd(cmd, run_name=label)
        print(result.stdout[-2000:], flush=True)
        if result.returncode != 0:
            log(f"  ERROR on {label}:\n{result.stderr[-500:]}")
            continue
        ppls = parse_perplexity(result.stdout)
        csv_lang = LANG_TO_CSV[lang]
        mono_matched.append({"label": label, "lang": lang, "csv_lang": csv_lang,
                              "tokens": tokens, "ppls": ppls,
                              "ppl": ppls.get(lang, float("nan"))})
        save_checkpoint("agent_mono_matched", mono_matched)
        log(f"  {label} -> ppl={ppls.get(lang, '?'):.2f}")
        cleanup_model(output_dir)


# ── Step 5: Multi-seed validation ──────────────────────────────────────────────
# Tests whether the key finding (Indonesian > Dutch MLTE) holds across seeds.
# Uses batch=32, no warmup, no wd — EXACTLY matching the CSV hyperparams.
# seed=42 already exists in CSV; seeds 0 and 1 are new.
# 4 runs (2 seeds × 2 language pairs) @ ~35 min each on A100.

log("\n=== Step 5: Multi-seed validation — eng-nld vs eng-ind at 90/10 ===")
log("  Key question: is ind MLTE > nld MLTE robust to seed variation?")
log("  Hyperparams: batch=32, warmup=0, wd=0  (matching CSV seed-42 baseline)")

VALIDATION_SEEDS = [0, 1]
VALIDATION_PAIRS = [("dut", ["eng", "dut"]), ("ind", ["eng", "ind"])]

seed_results = load_checkpoint("agent_seed_validation")
done_seeds = {r["label"] for r in seed_results}

for seed in VALIDATION_SEEDS:
    for lang_b, langs in VALIDATION_PAIRS:
        label = f"val-eng90-{lang_b}10-s{seed}"
        if label in done_seeds:
            log(f"  Skipping {label} (already done)")
            continue
        log(f"  Training {label} (seed={seed}) ...")
        output_dir = f"./{label}"
        cmd = train_cmd(
            langs=langs,
            output_dir=output_dir,
            model_name=label,
            max_tokens=BUDGET,
            ratios=[0.90, 0.10],
            batch_size=32,      # match CSV hyperparams
            warmup_ratio=0.0,   # match CSV hyperparams
            weight_decay=0.0,   # match CSV hyperparams
            seed=seed,
        )
        result = run_cmd(cmd, run_name=label)
        print(result.stdout[-2000:], flush=True)
        if result.returncode != 0:
            log(f"  ERROR on {label}:\n{result.stderr[-500:]}")
            continue
        ppls = parse_perplexity(result.stdout)
        seed_results.append({
            "label": label, "lang_b": lang_b, "seed": seed,
            "ratio_eng": 0.90, "ratio_b": 0.10, "ppls": ppls,
        })
        save_checkpoint("agent_seed_validation", seed_results)
        log(f"  {label} -> eng={ppls.get('eng', '?'):.2f}, {lang_b}={ppls.get(lang_b, '?'):.2f}")
        cleanup_model(output_dir)


# ── Step 6b: Absorb Dutch seed=42 A100 result ──────────────────────────────────
# Closes the hardware gap: CSV Dutch 90/10 was on L4; this run is on A100 with same seed.
# Result file: val_nld_s42.log (produced by manual train.py call outside the script).

_s42_label = "val-eng90-dut10-s42"
_already = any(r["label"] == _s42_label for r in seed_results)
if not _already:
    _s42_log = Path("val_nld_s42.log")
    if _s42_log.exists():
        _ppls = parse_perplexity(_s42_log.read_text())
        if _ppls:
            seed_results.append({
                "label": _s42_label, "lang_b": "dut", "seed": 42,
                "ratio_eng": 0.90, "ratio_b": 0.10, "ppls": _ppls,
            })
            save_checkpoint("agent_seed_validation", seed_results)
            log(f"  Absorbed Dutch seed=42 A100 result: {_ppls}")
        else:
            log("  val_nld_s42.log exists but no perplexity found — run may not be complete")
    else:
        log("  val_nld_s42.log not found — Dutch seed=42 A100 run not yet done")


# ── Step 6: MLTE computation ────────────────────────────────────────────────────

log("\n=== Step 6: MLTE computation (token efficiency) ===")

tidy_csv = Path("babylm_runs_tidy.csv")
if not tidy_csv.exists():
    log("  WARNING: babylm_runs_tidy.csv not found — skipping MLTE")
else:
    tidy_df = pd.read_csv(tidy_csv)

    def parse_tokens(t: str) -> int:
        t = str(t).strip()
        if t.endswith("M"):
            return int(float(t[:-1]) * 1_000_000)
        return int(t)

    tidy_df["tokens_n"] = tidy_df["tokens"].apply(parse_tokens)

    def fit_scaling_laws(data: list[tuple[int, float]]) -> tuple[float, float] | None:
        """Fit ppl = exp(log_a) * tokens^b. Returns (log_a, b) or None if <2 points."""
        if len(data) < 2:
            return None
        xs = np.array([d[0] for d in data], dtype=float)
        ys = np.array([d[1] for d in data], dtype=float)
        b, log_a = np.polyfit(np.log(xs), np.log(ys), 1)
        return (log_a, b)

    # ── Scaling laws from CSV (batch=32, no warmup/wd) ─────────────────────────
    # Valid for: bilingual_existing (CSV bilingual data)
    mono_csv = tidy_df[
        (tidy_df["config"] == "small") &
        (tidy_df["lang"].isin(["eng", "nld", "ind"])) &
        (tidy_df["ratio"] == "mono") &
        (tidy_df["epochs"] == 3)
    ].copy()
    mono_csv = mono_csv.sort_values("perplexity").drop_duplicates(subset=["lang", "tokens_n"])

    scaling_csv: dict[str, tuple[float, float]] = {}
    for lang, grp in mono_csv.groupby("lang"):
        grp = grp.sort_values("tokens_n")
        params = fit_scaling_laws(list(zip(grp["tokens_n"], grp["perplexity"])))
        if params:
            scaling_csv[lang] = params
            log(f"  [CSV baseline] {lang}: ppl = {np.exp(params[0]):.4f} * tokens^{params[1]:.4f}")

    # ── Scaling laws from matched mono runs (batch=128, warmup=0.05, wd=0.01) ──
    # Valid for: bilingual_new (95/5, 99/1) and trilingual
    matched_by_lang: dict[str, list[tuple[int, float]]] = {}
    for r in mono_matched:
        lang = r["lang"]
        csv_lang = LANG_TO_CSV[lang]  # use csv_lang as key to match scaling_csv format
        ppl = r.get("ppl", r["ppls"].get(lang, float("nan")))
        if not np.isnan(ppl):
            matched_by_lang.setdefault(csv_lang, []).append((r["tokens"], ppl))

    scaling_matched: dict[str, tuple[float, float]] = {}
    for csv_lang, points in matched_by_lang.items():
        params = fit_scaling_laws(points)
        if params:
            scaling_matched[csv_lang] = params
            log(f"  [matched baseline] {csv_lang}: ppl = {np.exp(params[0]):.4f} * tokens^{params[1]:.4f}")

    def token_efficiency(target_ppl: float, t_actual: float,
                         scaling: dict[str, tuple[float, float]],
                         csv_lang: str) -> float:
        """MLTE / t_actual: >1 means bilingual more efficient than mono."""
        if csv_lang not in scaling:
            return float("nan")
        log_a, b = scaling[csv_lang]
        mono_equiv_tokens = np.exp((np.log(target_ppl) - log_a) / b)
        return mono_equiv_tokens / t_actual

    rows = []

    # ── Existing bilingual from tidy CSV (uses CSV baselines) ─────────────────
    bi_mask = (
        (tidy_df["config"] == "small") &
        (~tidy_df["lang"].isin(["eng", "nld", "ind"]))
    )
    for _, row in tidy_df[bi_mask].iterrows():
        lang_pair = str(row["lang"])       # e.g. "eng-nld"
        ratio_str = str(row["ratio"])      # e.g. "90-10"
        tokens_n  = row["tokens_n"]
        eval_lang = str(row["eval_lang"])  # e.g. "eng" or "nld"
        ppl       = float(row["perplexity"])

        parts       = lang_pair.split("-")
        ratio_parts = ratio_str.split("-")
        if len(parts) != 2 or len(ratio_parts) != 2:
            continue
        try:
            r = int(ratio_parts[parts.index(eval_lang)]) / 100
        except (ValueError, IndexError):
            continue

        t_lang = tokens_n * r
        eff = token_efficiency(ppl, t_lang, scaling_csv, eval_lang)
        rows.append({
            "experiment": "bilingual_existing",
            "baseline": "csv_batch32",
            "label": f"{lang_pair}_{ratio_str}_{tokens_n // 1_000_000}M",
            "eval_lang": eval_lang, "ratio": r, "t_lang": t_lang,
            "bilingual_ppl": ppl, "token_efficiency": eff,
        })

    # ── New bilingual 95/5, 99/1 (uses matched baselines) ─────────────────────
    _matched_ok = bool(scaling_matched)
    _fallback_note = "" if _matched_ok else " [NOTE: using CSV baselines — matched not ready]"
    for r in nld_results + ind_results:
        for eval_lang, ppl in r["ppls"].items():
            is_a = eval_lang == r["lang_a"]
            ratio = r["ratio_a"] if is_a else r["ratio_b"]
            t_lang = BUDGET * ratio
            csv_lang = LANG_TO_CSV.get(eval_lang, eval_lang)
            scaling = scaling_matched if _matched_ok else scaling_csv
            baseline_tag = "matched_batch128" if _matched_ok else "csv_batch32_fallback"
            eff = token_efficiency(ppl, t_lang, scaling, csv_lang)
            rows.append({
                "experiment": "bilingual_new",
                "baseline": baseline_tag,
                "label": r["label"] + _fallback_note, "eval_lang": eval_lang,
                "ratio": ratio, "t_lang": t_lang,
                "bilingual_ppl": ppl, "token_efficiency": eff,
            })

    # ── Trilingual (uses matched baselines) ────────────────────────────────────
    for r in tri_results:
        lang_ratio = {"eng": r["ratio_eng"], "dut": r["ratio_dut"], "ind": r["ratio_ind"]}
        for eval_lang, ppl in r["ppls"].items():
            ratio = lang_ratio.get(eval_lang)
            if ratio is None:
                continue
            t_lang = BUDGET * ratio
            csv_lang = LANG_TO_CSV.get(eval_lang, eval_lang)
            scaling = scaling_matched if _matched_ok else scaling_csv
            baseline_tag = "matched_batch128" if _matched_ok else "csv_batch32_fallback"
            eff = token_efficiency(ppl, t_lang, scaling, csv_lang)
            rows.append({
                "experiment": "trilingual",
                "baseline": baseline_tag,
                "label": r["label"], "eval_lang": eval_lang, "ratio": ratio,
                "t_lang": t_lang, "bilingual_ppl": ppl, "token_efficiency": eff,
            })

    # ── Multi-seed validation (uses CSV baselines — batch=32 matches) ──────────
    for r in seed_results:
        lang_b = r["lang_b"]
        seed = r["seed"]
        for eval_lang, ppl in r["ppls"].items():
            ratio = r["ratio_eng"] if eval_lang == "eng" else r["ratio_b"]
            t_lang = BUDGET * ratio
            csv_lang = LANG_TO_CSV.get(eval_lang, eval_lang)
            eff = token_efficiency(ppl, t_lang, scaling_csv, csv_lang)
            rows.append({
                "experiment": "seed_validation",
                "baseline": "csv_batch32",
                "label": r["label"], "eval_lang": eval_lang,
                "ratio": ratio, "t_lang": t_lang,
                "bilingual_ppl": ppl, "token_efficiency": eff,
            })

    if rows:
        te_df = pd.DataFrame(rows)
        te_df.to_csv("agent_token_efficiency.csv", index=False)
        log(f"\n  Saved {len(te_df)} rows to agent_token_efficiency.csv")

        # Summary per experiment type
        for exp_type, grp in te_df.groupby("experiment"):
            log(f"\n  [{exp_type}]")
            log(grp[["label", "eval_lang", "ratio", "bilingual_ppl", "token_efficiency"]]
                .sort_values(["eval_lang", "ratio"])
                .to_string(index=False))

        # Highlight best token efficiency per (eval_lang, experiment)
        log("\n  === Peak token efficiency per language ===")
        for (eval_lang, exp_type), grp in te_df.groupby(["eval_lang", "experiment"]):
            valid = grp["token_efficiency"].dropna()
            if valid.empty:
                log(f"  {eval_lang} | {exp_type}: all NaN (baseline not available)")
                continue
            best = grp.loc[valid.idxmax()]
            log(
                f"  {eval_lang} | {exp_type}: best eff={best['token_efficiency']:.3f} "
                f"at ratio={best['ratio']:.2f} (ppl={best['bilingual_ppl']:.1f})"
            )

        # ── Seed variance analysis ────────────────────────────────────────────
        seed_val = te_df[te_df["experiment"] == "seed_validation"]
        if not seed_val.empty:
            log("\n  === Seed variance analysis (dut vs ind @ 90/10) ===")
            log("  CSV seed=42: Dutch on L4, Indonesian on A100")
            log("  New seeds: all on A100 (batch=32, no warmup, no wd)")

            # CSV seed=42 — note: Dutch is keyed as 'nld' in bilingual_existing
            csv_90_10 = te_df[
                (te_df["experiment"] == "bilingual_existing") &
                (te_df["ratio"] == 0.10) &
                (te_df["label"].str.contains("90"))
            ].copy()
            # Normalize nld→dut so merge works with seed_val (uses 'dut')
            csv_90_10["eval_lang"] = csv_90_10["eval_lang"].replace({"nld": "dut"})

            # New seed_validation rows — extract seed number from label
            sv_ext = seed_val.copy()
            sv_ext["seed"] = sv_ext["label"].str.extract(r"-s(\d+)$")[0].astype(int)

            # A100 seed=42 Dutch (if run was completed and added to seed_validation)
            sv_s42_dut = sv_ext[
                (sv_ext["eval_lang"] == "dut") & (sv_ext["seed"] == 42)
            ]

            all_seeds = pd.concat([
                csv_90_10[["eval_lang", "bilingual_ppl", "token_efficiency"]]
                    .assign(seed=42, source="csv_L4_or_A100"),
                sv_ext[["eval_lang", "bilingual_ppl", "token_efficiency", "seed"]]
                    .assign(source="A100"),
            ], ignore_index=True)

            for lang in ["dut", "ind"]:
                sub = all_seeds[all_seeds["eval_lang"] == lang].sort_values("seed")
                if not sub.empty:
                    effs = sub["token_efficiency"].dropna()
                    log(f"  {lang} MLTE across seeds:")
                    log("  " + sub[["seed", "source", "bilingual_ppl", "token_efficiency"]]
                           .to_string(index=False))
                    log(f"    mean={effs.mean():.3f}, std={effs.std():.3f}, "
                        f"range=[{effs.min():.3f}, {effs.max():.3f}]")

# ── Done ───────────────────────────────────────────────────────────────────────

log("\n" + "=" * 70)
log("Session complete.")
log("Outputs: agent_token_efficiency.csv | agent_log.md | .checkpoints/")
log("New checkpoints: agent_mono_matched.json | agent_seed_validation.json")
log("=" * 70)
