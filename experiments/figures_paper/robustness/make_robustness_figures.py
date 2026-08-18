#!/usr/bin/env python
"""make_robustness_figures.py — paper figures for ROBUSTNESS BENCHMARK (claim 4) and
LIFT-INVARIANCE (claim 2).

Reads persisted run records directly by absolute path — no rngrn import needed.

Data sources (read-only, absolute paths into the MAIN checkout and this worktree):
  MAIN/experiments/exp11_robustness_baseline.csv        — 127 generator systems x 400
                                                            perturbation draws each
                                                            (docs/ROBUSTNESS_MEASUREMENT.md)
  MAIN/experiments/c2_P_t8k8_consol/runs/*/results/train_results.json
                                                          — 16 recovered c2_P models
  WORKTREE/experiments/lift_ladder/v4/results/v4.json    — V4 lift-invariance survey,
                                                            27 models incl. d5/seed{1,3,5,6}

Pass bars are PREREGISTRATION.md §3.2 (median turing_volume_10pct >= 0.90, median
turing_volume_4p8pct >= 0.95). The lift-invariance robustness-vs-mu curve is REPORT-ONLY
under PREREGISTRATION.md §3.7 ("Reported, never gated") — it is not judged against §3.2's
bars; it is the withdrawn-but-retained §3.6 vehicle without §3.6's criterion.

Outputs -> WORKTREE/experiments/figures_paper/robustness/figures/
  1_robustness_benchmark.png   population volume-vs-noise + recovered overlay (claim 4)
  2_lift_invariance.png        volume-vs-mu and kstar-vs-mu for d5 learned models (claim 2)
"""
from __future__ import annotations
import csv
import glob
import json
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAIN = "/home/benja/projects/personal/rngrn/rngrn-pipeline"
WT = "/home/benja/projects/personal/rngrn/worktrees/lift-ladder"
BASELINE_CSV = os.path.join(MAIN, "experiments/exp11_robustness_baseline.csv")
C2P_ROOT = os.path.join(MAIN, "experiments/c2_P_t8k8_consol")
V4_JSON = os.path.join(WT, "experiments/lift_ladder/v4/results/v4.json")
OUT_DIR = os.path.join(WT, "experiments/figures_paper/robustness/figures")
DPI = 220

PASS_TV10 = 0.90    # PREREGISTRATION.md Sec 3.2
PASS_TV48 = 0.95    # PREREGISTRATION.md Sec 3.2

NOISE_LEVELS = ["0.01", "0.048", "0.1", "0.2"]
NOISE_LABELS = ["1%", "4.8%", "10%", "20%"]


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path} ({os.path.getsize(path)} bytes)")
    return path


def wilson_lower_bound(x: int, n: int, z: float = 1.959963984540054) -> float:
    """Wilson score interval lower bound for a binomial proportion x/n."""
    phat = x / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return center - margin


# ---------------------------------------------------------------------------
# Load population baseline (127 generator systems x 400 draws each)
# ---------------------------------------------------------------------------

def load_baseline():
    rows = list(csv.DictReader(open(BASELINE_CSV)))
    by_sigma = {s: [] for s in NOISE_LEVELS}
    for r in rows:
        s = r["sigma"]
        if s in by_sigma:
            by_sigma[s].append(float(r["frac_strict"]))
    return by_sigma


# ---------------------------------------------------------------------------
# Load 16 recovered c2_P models
# ---------------------------------------------------------------------------

def load_c2p():
    paths = sorted(glob.glob(os.path.join(C2P_ROOT, "runs/*/results/train_results.json")))
    tv10, tv48 = [], []
    for p in paths:
        m = json.load(open(p))["metric"]
        tv10.append(m["turing_volume_10pct"])
        tv48.append(m["turing_volume_4p8pct"])
    return np.array(tv10), np.array(tv48), len(paths)


# ---------------------------------------------------------------------------
# Figure 1 — robustness benchmark (claim 4)
# ---------------------------------------------------------------------------

def fig_robustness_benchmark(by_sigma, tv10, tv48, n_recovered):
    rng = np.random.default_rng(0)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # --- Panel A: population median/mean bars per noise level, "the Sec 3.2 bars",
    #     with the 16 recovered models overlaid at 4.8% and 10%.
    medians = [st.median(by_sigma[s]) for s in NOISE_LEVELS]
    means = [st.mean(by_sigma[s]) for s in NOISE_LEVELS]
    worst = [min(by_sigma[s]) for s in NOISE_LEVELS]
    x = np.arange(len(NOISE_LEVELS))

    axA.bar(x, medians, width=0.55, color="0.75", edgecolor="0.3", zorder=2,
            label="population median (n=127 generator systems)")
    axA.scatter(x, means, marker="_", s=340, color="0.2", linewidth=1.6, zorder=3,
                label="population mean")
    axA.scatter(x, worst, marker="x", s=45, color="0.45", zorder=3,
                label="population worst sample")

    # pass bars, Sec 3.2: 0.90 @ 10%, 0.95 @ 4.8%
    axA.hlines(PASS_TV48, x[1] - 0.3, x[1] + 0.3, color="C1", ls="--", lw=1.6, zorder=4,
               label=f"pass bar, 4.8% >= {PASS_TV48:.2f} (Sec 3.2)")
    axA.hlines(PASS_TV10, x[2] - 0.3, x[2] + 0.3, color="C0", ls="--", lw=1.6, zorder=4,
               label=f"pass bar, 10% >= {PASS_TV10:.2f} (Sec 3.2)")

    # recovered overlay at 4.8% and 10%
    jitter48 = rng.uniform(-0.12, 0.12, size=n_recovered)
    jitter10 = rng.uniform(-0.12, 0.12, size=n_recovered)
    axA.scatter(np.full(n_recovered, x[1]) + jitter48, tv48, color="C1", s=26,
                edgecolor="white", linewidth=0.4, zorder=5,
                label="recovered c2_P models (n=16)")
    axA.scatter(np.full(n_recovered, x[2]) + jitter10, tv10, color="C0", s=26,
                edgecolor="white", linewidth=0.4, zorder=5)
    axA.scatter([x[1]], [st.median(tv48)], marker="D", s=70, color="C1",
                edgecolor="black", linewidth=0.8, zorder=6,
                label="recovered median")
    axA.scatter([x[2]], [st.median(tv10)], marker="D", s=70, color="C0",
                edgecolor="black", linewidth=0.8, zorder=6)

    axA.set_xticks(x)
    axA.set_xticklabels(NOISE_LABELS)
    axA.set_xlabel("perturbation level (log-normal SD on nonzero J, D entries)")
    axA.set_ylabel("local Turing volume (fraction of 400 draws)")
    axA.set_ylim(0, 1.05)
    axA.set_title("(a) population baseline vs. recovered models")
    axA.legend(loc="lower left", fontsize=7.2, framealpha=0.9)
    axA.grid(axis="y", alpha=0.25)

    # --- Panel B: distribution at 10% noise specifically — the headline comparison
    pop10 = by_sigma["0.1"]
    axB.violinplot([pop10], positions=[0], widths=0.7, showmedians=False,
                   showextrema=False)
    parts = axB.violinplot([pop10], positions=[0], widths=0.7, showmedians=False,
                            showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("0.75")
        pc.set_edgecolor("0.4")
        pc.set_alpha(0.9)
    jitterB = rng.uniform(-0.08, 0.08, size=len(pop10))
    axB.scatter(jitterB, pop10, s=8, color="0.4", alpha=0.5, zorder=2)
    jitterC = rng.uniform(-0.08, 0.08, size=n_recovered)
    axB.scatter(np.full(n_recovered, 1) + jitterC, tv10, s=26, color="C0",
                edgecolor="white", linewidth=0.4, zorder=3)
    axB.hlines(st.median(pop10), -0.35, 0.35, color="0.2", lw=2.0, zorder=4)
    axB.hlines(st.median(tv10), 0.65, 1.35, color="C0", lw=2.0, zorder=4)
    axB.axhline(PASS_TV10, color="0.2", ls="--", lw=1.2,
                label=f"pass bar >= {PASS_TV10:.2f}")
    axB.set_xticks([0, 1])
    axB.set_xticklabels([f"population\n(n=127)\nmedian={st.median(pop10):.3f}",
                          f"recovered c2_P\n(n=16)\nmedian={st.median(tv10):.3f}"])
    axB.set_ylabel("turing_volume_10pct")
    axB.set_ylim(0, 1.05)
    axB.set_title("(b) headline: recovered >= population's own median at 10%")
    axB.legend(loc="lower left", fontsize=8)
    axB.grid(axis="y", alpha=0.25)

    fig.suptitle("Robustness benchmark: recovered c2_P models against the 127-system "
                  "generator population (claim 4)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "1_robustness_benchmark.png")


# ---------------------------------------------------------------------------
# Figure 2 — lift-invariance (claim 2)
# ---------------------------------------------------------------------------

def fig_lift_invariance(v4):
    mu_band = v4["mu_band"]
    labels = ["d5/seed1", "d5/seed3", "d5/seed5", "d5/seed6"]
    colors = ["C0", "C1", "C2", "C3"]
    recs = {r["label"]: r for r in v4["robustness"]}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    cost_pp = []
    for lab, c in zip(labels, colors):
        r = recs[lab]
        mus = [row["mu"] for row in r["rows"]]
        ft = [row["frac_turing"] for row in r["rows"]]
        qss = r["qss"]["frac_turing"]
        axA.semilogx(mus, ft, "o-", color=c, ms=4, lw=1.6, label=f"{lab} (QSS={qss:.3f})")
        axA.axhline(qss, color=c, ls=":", lw=0.8, alpha=0.5)
        cost_pp.append(100 * (qss - ft[-1]))

    axA.axvspan(mu_band[0], mu_band[1], color="0.85", alpha=0.5, zorder=0,
                label=f"biological $\\mu$ band [{mu_band[0]:.1e}, {mu_band[1]:.1e}]")
    axA.axhline(PASS_TV10, color="0.2", ls="--", lw=1.2,
                label=f"Sec 3.2 robustness bar ({PASS_TV10:.2f}, for reference only —\n"
                      f"not gated here, PREREGISTRATION Sec 3.7")
    axA.set_xlabel(r"$\mu$ (timescale-separation parameter)")
    axA.set_ylabel("perturbation-cloud Turing volume (frac_turing)")
    axA.set_ylim(0, 1.0)
    axA.set_title("(a) Turing volume vs. $\\mu$ — n=4 learned models")
    axA.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=6.3, framealpha=0.92)
    axA.grid(alpha=0.25)

    cost_lo, cost_hi = min(cost_pp), max(cost_pp)
    axA.annotate(
        f"lift-INVARIANT (volume cost {cost_lo:.0f}-{cost_hi:.0f} percentage points "
        f"across the band)\nabsolute volumes 0.29-0.43 sit BELOW the Sec 3.2 robustness "
        f"bar (0.90):\nclaim is invariance under the lift, not absolute robustness",
        xy=(0.5, 0.03), xycoords="axes fraction", fontsize=7.0, ha="center", va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="0.4", alpha=0.95))

    n_qss = v4["n_qss_turing"]
    n_lift = v4["n_lifted_turing_band_all"]
    lb = wilson_lower_bound(n_lift, n_qss)
    axA.annotate(
        f"aggregate, all {n_qss} QSS-Turing models in this V4 run\n"
        f"(incl. harvest population, not just d5):\n"
        f"P(lifted-Turing | QSS-Turing) = {n_lift}/{n_qss} = {n_lift/n_qss:.3f}\n"
        f"(Wilson 95% LB {lb:.4f})",
        xy=(0.99, 0.99), xycoords="axes fraction", fontsize=6.8, ha="right", va="top",
        bbox=dict(boxstyle="round", fc="#eef6ff", ec="0.4", alpha=0.95))

    for lab, c in zip(labels, colors):
        r = recs[lab]
        mus = [row["mu"] for row in r["rows"]]
        km = [row["kstar_mean"] for row in r["rows"]]
        ks = [row["kstar_std"] for row in r["rows"]]
        axB.errorbar(mus, km, yerr=ks, fmt="o-", color=c, ms=4, lw=1.6, capsize=2,
                     label=lab)
    axB.set_xscale("log")
    axB.axvspan(mu_band[0], mu_band[1], color="0.85", alpha=0.5, zorder=0)
    axB.set_xlabel(r"$\mu$ (timescale-separation parameter)")
    axB.set_ylabel(r"$k^*_{\rm mean}$ (surviving draws)")
    axB.set_title("(b) $k^*$ vs. $\\mu$ — essentially flat")
    axB.legend(loc="best", fontsize=8)
    axB.grid(alpha=0.25)

    fig.suptitle("Lift-invariance: robustness under the finite-$\\mu$ dynamical lift, "
                  "report-only under PREREGISTRATION Sec 3.7 (claim 2)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "2_lift_invariance.png")


def main():
    by_sigma = load_baseline()
    tv10, tv48, n_recovered = load_c2p()
    v4 = json.load(open(V4_JSON))

    print("Population baseline (n=127 systems, 400 draws each):")
    for s, lab in zip(NOISE_LEVELS, NOISE_LABELS):
        vals = by_sigma[s]
        print(f"  {lab}: median={st.median(vals):.3f} mean={st.mean(vals):.3f} "
              f"worst={min(vals):.3f}")
    print(f"Recovered c2_P (n={n_recovered}): "
          f"tv10 median={st.median(tv10):.4f} tv48 median={st.median(tv48):.4f}")

    fig_robustness_benchmark(by_sigma, tv10, tv48, n_recovered)
    fig_lift_invariance(v4)


if __name__ == "__main__":
    main()
