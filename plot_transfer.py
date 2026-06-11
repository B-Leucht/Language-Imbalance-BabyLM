import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Load and clean babylm_runs.csv ────────────────────────────────────────────
raw = pd.read_csv("babylm_runs.csv")
raw.columns = raw.columns.str.strip()
raw = raw.rename(
    columns={
        "No.": "no",
        "Config": "config",
        "Lang(s)": "langs",
        "Ratio": "ratio",
        "Tokens": "tokens_str",
        "Perplexity": "perplexity_str",
    }
)


def parse_tokens(t):
    if not isinstance(t, str):
        return None
    m = re.match(r"(\d+)\s*M", t.strip(), re.IGNORECASE)
    return float(m.group(1)) * 1e6 if m else None


raw["tokens"] = raw["tokens_str"].apply(parse_tokens)
raw = raw[raw["config"].isin(["tiny", "small", "large"]) & raw["tokens"].notna()]


# ── Monolingual data + scaling law fit ───────────────────────────────────────
def parse_ppl_scalar(p):
    try:
        return float(str(p).strip().replace("\t", ""))
    except ValueError:
        return None


mono = raw[raw["ratio"] == "mono"].copy()
mono["ppl"] = mono["perplexity_str"].apply(parse_ppl_scalar)
mono = mono[mono["ppl"].notna() & mono["langs"].isin(["eng", "nld", "ind"])]

params = {}
for (config, lang), grp in mono.groupby(["config", "langs"]):
    b, log_a = np.polyfit(np.log(grp["tokens"]), np.log(grp["ppl"]), 1)
    params[(config, lang)] = (log_a, b)


def ppl_mono(config, lang, t):
    log_a, b = params[(config, lang)]
    return np.exp(log_a + b * np.log(t))


def mlte(config, lang, ppl_bi):
    log_a, b = params[(config, lang)]
    return np.exp((np.log(ppl_bi) - log_a) / b)


# ── Parse bilingual results ───────────────────────────────────────────────────
def parse_ppl_pair(p, lang2):
    if not isinstance(p, str):
        return None, None
    m_eng = re.search(r"eng\s*:\s*([0-9.]+)", p)
    m_l2 = re.search(rf"{lang2}\s*:\s*([0-9.]+)", p)
    if m_eng and m_l2:
        return float(m_eng.group(1)), float(m_l2.group(1))
    return None, None


def parse_ratio(r):
    m = re.match(r"(\d+)-(\d+)", str(r).strip())
    if m:
        return int(m.group(1)) / 100, int(m.group(2)) / 100
    return None, None


TOTAL = 50_000_000
bi = raw[raw["langs"].isin(["eng-nld", "eng-ind"]) & (raw["tokens"] == TOTAL)].copy()

records = []
for _, row in bi.iterrows():
    config = row["config"]
    lp = row["langs"]
    lang2 = lp.split("-")[1]
    r_eng, r_l2 = parse_ratio(row["ratio"])
    if r_eng is None:
        continue
    ppl_e, ppl_l2 = parse_ppl_pair(str(row["perplexity_str"]), lang2)
    if ppl_e is None:
        continue
    t_e, t_l2 = r_eng * TOTAL, r_l2 * TOTAL
    records.append(
        dict(
            config=config,
            lang_pair=lp,
            lang2=lang2,
            ratio_l2=r_l2,
            ppl_eng=ppl_e,
            ppl_l2=ppl_l2,
            mlpe_eng=ppl_mono(config, "eng", t_e),
            mlpe_l2=ppl_mono(config, lang2, t_l2),
            teff_eng=mlte(config, "eng", ppl_e) / t_e,
            teff_l2=mlte(config, lang2, ppl_l2) / t_l2,
        )
    )

results = pd.DataFrame(records)

# shared style
lang_pairs = ["eng-nld", "eng-ind"]
lang2_name = {"nld": "Dutch", "ind": "Indonesian"}
lang_names = {"eng": "English", "nld": "Dutch", "ind": "Indonesian"}
lang_colors = {"eng": "#4C72B0", "nld": "#DD8452", "ind": "#55A868"}
bi_colors = {"eng": "#4C72B0", "l2": "#DD8452"}
config_style = {"tiny": "-", "small": "--", "large": "-."}
t_fit = np.logspace(np.log10(0.8e6), np.log10(60e6), 100)

# ── Figure 1: monolingual scaling ────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4))

for ax, lang in zip(axes1, ["eng", "nld", "ind"]):
    for config in ["tiny", "small", "large"]:
        grp = mono[(mono["langs"] == lang) & (mono["config"] == config)].sort_values(
            "tokens"
        )
        if grp.empty:
            continue
        ls = config_style[config]
        marker = {"tiny": "o", "small": "s", "large": "^"}[config]
        ax.scatter(
            grp["tokens"],
            grp["ppl"],
            color=lang_colors[lang],
            marker=marker,
            s=30,
            zorder=3,
        )
        param_label = {"tiny": "tiny (2M)", "small": "small (10M)", "large": "large (85M)"}[config]
        ax.plot(
            t_fit,
            ppl_mono(config, lang, t_fit),
            ls,
            color=lang_colors[lang],
            label=param_label,
        )
    ax.set_xlim(0, 55e6)
    ax.set_xticks([1e6, 10e6, 20e6, 30e6, 40e6, 50e6])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1e6)}M"))
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Perplexity")
    ax.set_title(lang_names[lang])
    ax.legend(fontsize=8, title="Config")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig("scaling_law_babylm.pdf", bbox_inches="tight")
fig1.savefig("scaling_law_babylm.png", dpi=150, bbox_inches="tight")
print("Saved scaling_law_babylm.png")

# ── Figure 2: token efficiency ────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

from matplotlib.lines import Line2D

for ax, lp in zip(axes2, lang_pairs):
    lang2 = lp.split("-")[1]
    for config in ["tiny", "small", "large"]:
        sub = results[
            (results["lang_pair"] == lp) & (results["config"] == config)
        ].sort_values("ratio_l2")
        if sub.empty:
            continue
        marker = {"tiny": "o", "small": "s", "large": "^"}[config]
        xs = sub["ratio_l2"].values
        ax.plot(xs, sub["teff_eng"], marker=marker, linestyle="", color=bi_colors["eng"], markersize=7)
        ax.plot(xs, sub["teff_l2"], marker=marker, linestyle="", color=lang_colors[lang2], markersize=7)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)

    lang_handles = [
        Line2D(
            [0], [0], color=bi_colors["eng"], marker="o", linestyle="", markersize=7, label="English"
        ),
        Line2D(
            [0],
            [0],
            color=lang_colors[lang2],
            marker="o",
            linestyle="",
            markersize=7,
            label=lang2_name[lang2],
        ),
    ]
    config_handles = [
        Line2D(
            [0], [0], color="gray", marker="o", linestyle="", markersize=7, label="tiny (2M)"
        ),
        Line2D(
            [0], [0], color="gray", marker="s", linestyle="", markersize=7, label="small (10M)"
        ),
        Line2D(
            [0], [0], color="gray", marker="^", linestyle="", markersize=7, label="large (85M)"
        ),
        Line2D(
            [0], [0], color="gray", linestyle=":", linewidth=1.5, label="no transfer"
        ),
    ]
    leg1 = ax.legend(
        handles=lang_handles,
        fontsize=8,
        loc="upper right",
        title="Language",
        framealpha=0.9,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=config_handles,
        fontsize=8,
        loc="lower left",
        title="Model",
        framealpha=0.9,
    )

    ax.set_xlabel(f"{lang2_name[lang2]} token proportion")
    ax.set_title(f"{lp} – token efficiency (TEff)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

axes2[0].set_ylabel("TEff (MLTE / t_lang)")

plt.tight_layout()
fig2.savefig("transfer_factor.pdf", bbox_inches="tight")
fig2.savefig("transfer_factor.png", dpi=150, bbox_inches="tight")
print("Saved transfer_factor.pdf and transfer_factor.png")
