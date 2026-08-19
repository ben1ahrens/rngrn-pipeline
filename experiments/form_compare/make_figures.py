#!/usr/bin/env python
"""make_figures.py — form-robustness comparison figure (paper form-robustness unit).

Adapted (read-only source) from
worktrees/lift-ladder/experiments/figures_paper/robustness/make_robustness_figures.py's
`fig_robustness_benchmark`, extended to overlay TWO recovered-model populations (nc1 and
competitive) instead of one. See docs/DECISIONS.md::D-FORMCOMP-1.

Reads experiments/form_compare/numbers.json, written by analyze.py -- run that first.

Output -> experiments/form_compare/figures/1_form_robustness_comparison.png
"""
from __future__ import annotations
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WT = "/home/benja/projects/personal/rngrn/worktrees/paper-formcomp"
NUMBERS_JSON = os.path.join(WT, "experiments/form_compare/numbers.json")
OUT_DIR = os.path.join(WT, "experiments/form_compare/figures")
DPI = 220

PASS_TV10 = 0.90    # PREREGISTRATION.md Sec 3.2
PASS_TV48 = 0.95    # PREREGISTRATION.md Sec 3.2

LEVELS = ["0.01", "0.048", "0.1", "0.2"]
LABELS = ["1%", "4.8%", "10%", "20%"]
TV_KEYS = ["turing_volume_1pct", "turing_volume_4p8pct", "turing_volume_10pct", "turing_volume_20pct"]


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path} ({os.path.getsize(path)} bytes)")
    return path


def fig_comparison(numbers):
    baseline = numbers["baseline_summary"]
    nc1_pooled = numbers["nc1"]["pooled"]
    comp_pooled = numbers["competitive"]["pooled"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharey=True)
    for ax, level, label, tvkey in zip(axes, LEVELS, LABELS, TV_KEYS):
        pop_median = baseline[level]["median"]
        pop_mean = baseline[level]["mean"]
        pop_min = baseline[level]["min"]
        ax.bar([0], [pop_median], width=0.5, color="0.8", edgecolor="0.3", zorder=1,
               label="population median (n=127)" if ax is axes[0] else None)
        ax.scatter([0], [pop_mean], marker="_", s=220, color="0.2", zorder=3,
                   label="population mean" if ax is axes[0] else None)

        for xpos, pooled, color, form_label in [
                (1, nc1_pooled, "C0", "nc1 (c2_P_t8k8_consol)"),
                (2, comp_pooled, "C3", "competitive (form_compare)")]:
            vals = pooled.get(tvkey, {}).get("per_seed", [])
            med = pooled.get(tvkey, {}).get("median")
            if vals:
                rng = np.random.default_rng(0)
                jitter = rng.uniform(-0.15, 0.15, size=len(vals))
                ax.scatter(np.full(len(vals), xpos) + jitter, vals, color=color, s=24,
                          edgecolor="white", linewidth=0.4, zorder=4,
                          label=form_label if ax is axes[0] else None)
            if med is not None:
                ax.scatter([xpos], [med], marker="D", s=60, color=color,
                          edgecolor="black", linewidth=0.7, zorder=5)
            else:
                ax.annotate("n/a", xy=(xpos, 0.05), ha="center", fontsize=8, color=color)

        if label == "10%":
            ax.axhline(PASS_TV10, color="0.2", ls="--", lw=1.2,
                       label=f"Sec 3.2 bar ({PASS_TV10:.2f})" if ax is axes[0] else None)
        if label == "4.8%":
            ax.axhline(PASS_TV48, color="0.2", ls="--", lw=1.2,
                       label=f"Sec 3.2 bar ({PASS_TV48:.2f})" if ax is axes[0] else None)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["population\n(n=127)", "nc1\n(pooled)", "competitive\n(pooled)"],
                           fontsize=8)
        ax.set_title(f"{label} perturbation", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("local Turing volume (turing_volume_Xpct)")
    axes[0].legend(loc="lower left", fontsize=6.8, framealpha=0.92)

    fig.suptitle("Form-robustness comparison: nc1 vs competitive against the 127-system "
                  "generator population\n(nc1-tuned hyperparameters applied to both forms "
                  "-- see D-FORMCOMP-1 for the disclosed confound)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, "1_form_robustness_comparison.png")


def main():
    numbers = json.load(open(NUMBERS_JSON))
    fig_comparison(numbers)


if __name__ == "__main__":
    main()
