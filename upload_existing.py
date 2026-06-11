"""
Upload existing CSV results and plots to Weights & Biases.
Run this once at the start of a Lambda Labs session to give W&B
a full picture of prior experiments before running new ones.

Usage:
    export WANDB_PROJECT=language-imbalance-babylm
    python upload_existing.py
"""

import os
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    print("wandb not installed. Run: pip install wandb")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("pandas not installed. Run: pip install pandas")
    sys.exit(1)

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "language-imbalance-babylm")

wandb.init(project=WANDB_PROJECT, name="upload-existing-results", job_type="data-upload")

# ── CSV tables ────────────────────────────────────────────────────────────────

CSV_FILES = [
    ("scaling_results.csv",         "scaling_results"),
    ("bilingual_results.csv",       "bilingual_results"),
    ("token_efficiency_results.csv","token_efficiency"),
    ("babylm_runs.csv",             "babylm_runs"),
    ("babylm_runs_tidy.csv",        "babylm_runs_tidy"),
]

for fname, table_name in CSV_FILES:
    p = Path(fname)
    if p.exists():
        df = pd.read_csv(p)
        wandb.log({table_name: wandb.Table(dataframe=df)})
        print(f"Uploaded {fname} ({len(df)} rows)")
    else:
        print(f"Skipping {fname} (not found)")

# ── Plots ─────────────────────────────────────────────────────────────────────

IMAGES = [
    "scaling_law_babylm.png",
    "scaling_law.png",
    "transfer_factor.png",
    "embeddings_eng50dut50.png",
    "embeddings_eng90dut10.png",
]

for img in IMAGES:
    p = Path(img)
    if p.exists():
        key = p.stem.replace("-", "_")
        wandb.log({key: wandb.Image(str(p))})
        print(f"Uploaded {img}")
    else:
        print(f"Skipping {img} (not found)")

# ── Checkpoint summaries ──────────────────────────────────────────────────────

import json

CHECKPOINTS = [
    (".checkpoints/scaling_results.json",   "ckpt_scaling"),
    (".checkpoints/bilingual_results.json", "ckpt_bilingual"),
]

for fname, artifact_name in CHECKPOINTS:
    p = Path(fname)
    if p.exists():
        art = wandb.Artifact(artifact_name, type="checkpoint")
        art.add_file(str(p))
        wandb.log_artifact(art)
        print(f"Uploaded artifact {fname}")
    else:
        print(f"Skipping artifact {fname} (not found)")

wandb.finish()
print("\nDone. All existing results uploaded to W&B project:", WANDB_PROJECT)
