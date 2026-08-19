#!/usr/bin/env python
"""make_recovery_figures.py — paper figures for RECOVERY PANELS (claim 1).

Reads persisted arrays/run-index rows directly by absolute path — no rngrn import needed,
since plot_arrays.npz is a plain npz + JSON-encoded meta field (same shape as
rngrn.plotdata.load_plot_arrays / scripts/make_figures.py's `_decode`). Plotting logic for
the representative-run panels (pattern / dispersion / RAPS) is adapted from
scripts/make_figures.py in the MAIN checkout (fig_pattern, fig_dispersion, fig_raps),
trimmed to what this figure needs and bumped to >=200 dpi.

Data sources (read-only, absolute paths into the MAIN checkout and this worktree):
  MAIN/experiments/c2_P_t8k8_consol/            — the c2_P campaign, 16 runs
  WORKTREE/experiments/diag_fft/d5/             — the D5 contrast (10 seeds)

Outputs -> WORKTREE/experiments/figures_paper/recovery/figures/
  1_representative_panel.png   target | model, RAPS overlay, dispersion sigma(k)
  2_campaign_summary.png       kstar_fft_rel_err + turing_volume bars, 16 runs
  3_d5_contrast.png            D5: interior-k* successes vs grid-floor failures
"""
from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Collected into paper_results/ (branch feature/paper-results): both source
# checkouts' experiments/ trees now live under this pack root, so MAIN and WT
# resolve to the same place. Original roots: the main checkout and the
# feature/lift-ladder worktree.
PACK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MAIN = PACK
WT = PACK
C2P_ROOT = os.path.join(MAIN, "experiments/c2_P_t8k8_consol")
D5_ROOT = os.path.join(WT, "experiments/diag_fft/d5")
OUT_DIR = os.path.join(WT, "experiments/figures_paper/recovery/figures")
DPI = 220

REP_RUN_ID = "nc1_m3_registry_20260803_190648_seed4"

PASS_KSTAR_FFT_RELERR = 0.083   # PREREGISTRATION.md §3.3, half an FFT bin
PASS_TV10 = 0.90                # PREREGISTRATION.md §3.2
PASS_TV48 = 0.95                # PREREGISTRATION.md §3.2


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path} ({os.path.getsize(path)} bytes)")
    return path


def load_npz_arrays(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"].item()))
    return out


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ======================================================================================
# 1. representative panel — target | model, RAPS overlay, dispersion sigma(k)
# ======================================================================================
def fig_representative_panel():
    run_dir = os.path.join(C2P_ROOT, "runs", REP_RUN_ID)
    z = load_npz_arrays(os.path.join(run_dir, "arrays/plot_arrays.npz"))
    meta = z["meta"]

    fig = plt.figure(figsize=(15.5, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.25, 1.25], wspace=0.38)

    # --- (a,b) target | model field, shared colour scale ---
    tgt = np.asarray(z["target_frame"], float)[0]
    mdl = np.asarray(z["model_field"], float)[0]
    vmin, vmax = min(tgt.min(), mdl.min()), max(tgt.max(), mdl.max())
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(tgt, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    ax0.set_title(f"target (channel 0)\n$k^*_{{raps}}$={meta['kstar_raps_target']:.4f}",
                  fontsize=9)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(mdl, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    ax1.set_title(f"recovered rollout, shared scale\n"
                  f"$k^*_{{raps}}$={meta['kstar_raps_model']:.4f}", fontsize=9)
    ax1.set_xticks([]); ax1.set_yticks([])
    fig.colorbar(im, ax=[ax0, ax1], shrink=0.8, label="concentration (shared scale)")

    # --- (c) RAPS overlay ---
    axr = fig.add_subplot(gs[0, 2])
    k = np.asarray(z["raps_k"], float)
    pt = np.asarray(z["raps_target"], float)
    pm = np.asarray(z["raps_model"], float)
    peak = max(float(np.max(pt)), float(np.max(pm)))
    axr.semilogy(k, np.where(pt > 0, pt, np.nan), color="C0", lw=1.6, label="target")
    axr.axvline(meta["kstar_raps_target"], color="C0", ls="--", lw=1.0,
                label=f"target $k^*$={meta['kstar_raps_target']:.4f}")
    axr.semilogy(k, np.where(pm > 0, pm, np.nan), color="C1", lw=1.6, label="model rollout")
    axr.axvline(meta["kstar_raps_model"], color="C1", ls="--", lw=1.0,
                label=f"model $k^*$={meta['kstar_raps_model']:.4f}")
    kfft = meta.get("kstar_fft_true")
    if kfft is not None:
        axr.axvline(kfft, color="C3", ls=":", lw=1.4,
                    label=f"answer key $k^*_{{fft}}$={kfft:.4f}")
    axr.set_xlim(0, min(k.max(), 6.0 * meta["kstar_raps_target"]))
    axr.set_ylim(peak * 1e-6, peak * 5)
    axr.set_xlabel("$k$ (rad/length)"); axr.set_ylabel("radially averaged power")
    axr.set_title("RAPS: target vs model rollout", fontsize=9)
    axr.legend(fontsize=6.5, loc="upper right")

    # --- (d) dispersion sigma(k) ---
    axd = fig.add_subplot(gs[0, 3])
    kd = np.asarray(z["dispersion_k"], float)
    sig = np.asarray(z["dispersion_sigma"], float)
    sig_true = np.asarray(z["dispersion_sigma_true"], float)
    axd.axhline(0.0, color="0.6", lw=0.8)
    axd.plot(kd, sig, color="C0", lw=1.8, label="recovered $\\sigma(k)$")
    axd.plot(kd, sig_true, color="C3", lw=1.5, label="answer key $\\sigma(k)$")
    kr = meta["kstar_dispersion_recovered"]
    kt = meta["kstar_dispersion_true"]
    axd.axvline(kr, color="C0", ls="--", lw=1.0, label=f"recovered $k^*_{{obs}}$={kr:.4f}")
    axd.axvline(kt, color="C3", ls="--", lw=1.0, label=f"true $k^*_{{obs}}$={kt:.4f}")
    lo = min(float(sig.min()), float(sig_true.min()), 0.0)
    hi = max(float(sig.max()), float(sig_true.max()))
    axd.set_ylim(max(lo, -2.0 * max(hi, 1e-3)), hi * 1.35)
    axd.set_xlabel("$k$ (rad/length)")
    axd.set_ylabel("$\\sigma(k)=\\max_i\\mathrm{Re}\\,\\mathrm{eig}(J-k^2D)$")
    axd.set_title("dispersion relation", fontsize=9)
    axd.legend(fontsize=6.5, loc="lower left")

    kfft_rel_err = abs(meta["kstar_model"] - meta["kstar_fft_true"]) / meta["kstar_fft_true"]
    fig.suptitle(
        f"representative recovery — {meta['run_id']}  "
        f"({meta['dataset_id']}/{meta['sample_key']}, seed={meta['seed']})  "
        f"$k^*_{{fft}}$ rel. err = {kfft_rel_err:.4f}"
        f"  |  best-of-64 restarts, L={meta['L']:.1f}",
        fontsize=10)
    return _save(fig, "1_representative_panel.png")


# ======================================================================================
# 2. campaign summary — 16 runs, kstar_fft_rel_err + turing_volume, thresholds drawn
# ======================================================================================
def fig_campaign_summary():
    rows = read_jsonl(os.path.join(C2P_ROOT, "runs.jsonl"))
    rows.sort(key=lambda r: (r["sample_key"], r["seed"]))
    labels = [f"{r['sample_key'][-2:]}/s{r['seed']}" for r in rows]
    kfe = [r["kstar_fft_rel_err"] for r in rows]
    tv10 = [r["turing_volume_10pct"] for r in rows]
    tv48 = [r["turing_volume_4p8pct"] for r in rows]
    x = np.arange(len(rows))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 4.6))

    colors0 = ["C2" if v <= PASS_KSTAR_FFT_RELERR else "C3" for v in kfe]
    ax0.bar(x, kfe, color=colors0)
    ax0.axhline(PASS_KSTAR_FFT_RELERR, color="0.2", ls="--", lw=1.2,
                label=f"§3.3 pass line = {PASS_KSTAR_FFT_RELERR:.3f} (half FFT bin)")
    ax0.set_xticks(x); ax0.set_xticklabels(labels, rotation=60, fontsize=7)
    ax0.set_ylabel("$k^*_{fft}$ relative error")
    ax0.set_title(f"per-run $k^*_{{fft}}$ error — {sum(1 for v in kfe if v <= PASS_KSTAR_FFT_RELERR)}/"
                  f"{len(kfe)} pass §3.3, median={np.median(kfe):.4f}", fontsize=9)
    ax0.legend(fontsize=8)

    width = 0.38
    ax1.bar(x - width / 2, tv10, width=width, color="C0", label="turing_volume_10pct")
    ax1.bar(x + width / 2, tv48, width=width, color="C1", label="turing_volume_4p8pct")
    ax1.axhline(PASS_TV10, color="C0", ls="--", lw=1.2,
                label=f"§3.2 median floor 10% = {PASS_TV10:.2f}")
    ax1.axhline(PASS_TV48, color="C1", ls="--", lw=1.2,
                label=f"§3.2 median floor 4.8% = {PASS_TV48:.2f}")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=60, fontsize=7)
    ax1.set_ylabel("turing_volume")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"robustness — median tv10={np.median(tv10):.3f}, "
                  f"median tv4.8={np.median(tv48):.3f}", fontsize=9)
    ax1.legend(fontsize=7, loc="lower right")

    fig.suptitle("c2_P campaign summary — 16 runs (three_gene_qvar, 2 targets x 8 seeds, "
                 "best-of-64 restarts)", fontsize=10)
    return _save(fig, "2_campaign_summary.png")


# ======================================================================================
# 3. D5 contrast — interior-k* successes vs grid-floor failures
# ======================================================================================
def fig_d5_contrast():
    rows = read_jsonl(os.path.join(D5_ROOT, "runs.jsonl"))
    rows.sort(key=lambda r: r["seed"])
    seeds = [r["seed"] for r in rows]
    kmodel = [r["kstar_model"] for r in rows]
    krel = [r["kstar_fft_rel_err"] for r in rows]
    kfft_true = rows[0]["kstar_fft_true"]
    grid_floor = min(kmodel)
    is_success = [abs(v - grid_floor) > 1e-6 for v in kmodel]
    colors = ["C2" if s else "C3" for s in is_success]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    x = np.arange(len(rows))
    ax0.bar(x, kmodel, color=colors)
    ax0.axhline(kfft_true, color="0.2", ls="--", lw=1.2, label=f"$k^*_{{fft}}$ true={kfft_true:.4f}")
    ax0.axhline(grid_floor, color="C3", ls=":", lw=1.2, label=f"k-grid floor={grid_floor:.5f}")
    ax0.set_xticks(x); ax0.set_xticklabels([f"s{s}" for s in seeds])
    ax0.set_ylabel("$k^*_{model}$ (dispersion argmax)")
    ax0.set_title(f"D5: 10 seeds, same target — "
                  f"{sum(is_success)} interior / {len(rows) - sum(is_success)} grid-floor",
                  fontsize=9)
    ax0.legend(fontsize=7.5)

    ax1.bar(x, krel, color=colors)
    ax1.axhline(PASS_KSTAR_FFT_RELERR, color="0.2", ls="--", lw=1.0,
                label=f"§3.3 pass line = {PASS_KSTAR_FFT_RELERR:.3f}")
    ax1.set_xticks(x); ax1.set_xticklabels([f"s{s}" for s in seeds])
    ax1.set_yscale("log")
    ax1.set_ylabel("$k^*_{fft}$ relative error (log)")
    ax1.set_title("same target, restart-only variation: bimodal outcome", fontsize=9)
    ax1.legend(fontsize=7.5)

    fig.suptitle("D5 contrast — grid-floor failure is a restart-selection artefact, not a "
                 "target property (experiments/diag_fft/d5)", fontsize=10)
    return _save(fig, "3_d5_contrast.png")


if __name__ == "__main__":
    print("make_recovery_figures:")
    fig_representative_panel()
    fig_campaign_summary()
    fig_d5_contrast()
