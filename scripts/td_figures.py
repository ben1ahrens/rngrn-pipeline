#!/usr/bin/env python
"""td_figures.py — figure helpers for inspecting the TRAINING DATA itself.

WHY THIS FILE EXISTS
--------------------
Every registered dataset is the output of a reaction-diffusion simulation, and the whole
inverse problem rests on the premise that those outputs actually carry Turing patterns.
That premise had never been checked by eye across the corpus: the one existing pattern
gallery (``experiments/figures_report/stage0/f3_patterns_*.png``) covers Stage-0
biological-viability rollouts, not the registered datasets, and its panels carry colorbars
but NO axis labels and NO tick marks, so a reader cannot tell the physical scale of a
pattern from the image.

These helpers draw the registered data with a colorbar, labelled axes and ticks in
PHYSICAL units (the domain extent L), so a panel is self-describing. ``tests/test_td_figures.py``
holds those three properties as assertions rather than conventions.

WHICH SIDE OF THE FIREWALL IS THIS?
-----------------------------------
Data side, and deliberately so. This module reads ``payload.h5`` — the generating kinetics
via ``params_json``, ``x_star``, ``D``, and the generator's own ``k_star`` / ``sigma_max``
— which is exactly what CLAUDE.md section 5 forbids to *recovery*-side code. (It does NOT
read the stored ``jacobian`` or ``interaction_matrix`` datasets; the interaction matrix
arrives only inside ``params_json``.) Nothing here is imported by ``model.py``,
``observables.py``, ``recover.py``, ``losses/`` or ``eval/``; it lives in ``scripts/``
alongside the other figure code and is used only by the inspection notebook. It must never
be imported from the recovery path.

The one thing it borrows from the package is ``rngrn.observables.raps`` — the image-only
radially-averaged power spectrum. That is the measurement side, and using it means the k*
drawn on a spectrum panel is computed the same way recovery computes it.
"""
from __future__ import annotations

import json
import os
import sys

import h5py
import matplotlib
import numpy as np

if matplotlib.get_backend().lower() not in ("agg",):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.ticker import FormatStrFormatter                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from rngrn.observables import raps                                 # noqa: E402

# The generator rejects a simulation whose species-0 coefficient of variation falls below
# this (gen_tg3.simulate_and_classify: ``if cv0 < 0.05: return None``). The verdict here
# reuses the SAME number so that "patterned" in a figure means what it meant at generation
# time, rather than introducing a second, uncalibrated threshold.
CV_PATTERNED_MIN = 0.05

# A field made of a few isolated blobs also clears the cv rule, because cv only asks "is
# there contrast?", never "is the contrast PERIODIC?". The discriminator is where the RAPS
# peak sits: a periodic pattern peaks at an interior bin, whereas a handful of localized
# spikes gives a monotonically decaying spectrum whose argmax is the first resolvable bin.
#
# Calibrated DESCRIPTIVELY against all 413 registered samples on 2026-08-10: the peak-bin
# distribution is min=1, 1st pct=3, median=6, max=14, and exactly ONE sample sits at bin 1
# (three_gene_qvar/sample_0032) while every other sample is at bin 3 or above. Bin 2 is
# empty, so the threshold falls in a real gap rather than cutting through populated data.
#
# HONEST LIMITS, because "calibrated" is a strong word (CLAUDE.md section 8):
#   * On real data the boundary has been exercised n = 1 times. One sample is separated;
#     412 are on the other side. That is an outlier detector, not a transferable criterion.
#   * PEAK_BIN_MIN = 2 gives an IDENTICAL partition on this corpus; 4 would flag 15. So the
#     threshold is insensitive downward and brittle upward.
#   * peak_bin >= 1 is structural, not a property of the data: observables.raps forces
#     power[0] = 0, so bin 0 can never win.
#   * The known-answer controls live in the tests, not in the corpus:
#     tests/test_td_figures.py feeds a synthetic periodic field (must pass) and a synthetic
#     isolated-blob field (must fail) through this exact screen.
# See docs/DECISIONS.md D-TDPLOT-1.
PEAK_BIN_MIN = 3

SPECIES_LABELS = ("u", "v", "w")
DEFAULT_CMAP = "viridis"


# ======================================================================================
# loading
# ======================================================================================
def default_datasets_root():
    return os.path.join(HERE, "..", "data", "datasets")


def available_datasets(datasets_root=None):
    """Dataset ids that have a READABLE payload, sorted. Missing payloads are excluded."""
    root = datasets_root or default_datasets_root()
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "payload.h5")
        if os.path.isdir(os.path.join(root, name)) and os.path.exists(p):
            out.append(name)
    return out


def load_samples(dataset_id, datasets_root=None, limit=None, with_trajectory=False):
    """Read every sample of a dataset into plain dicts.

    ``with_trajectory`` defaults to False on purpose. A trajectory is 6 frames per sample,
    so loading them for the whole corpus costs ~170 MB against a 17.6 GiB machine that
    CLAUDE.md section 7a records as being OOM-bound. The galleries only need the final
    frame; ask for trajectories per sample, when you are about to draw one.

    Fails loud if the payload is absent — the loader must not guess (CLAUDE.md section 6).
    """
    root = datasets_root or default_datasets_root()
    path = os.path.join(root, dataset_id, "payload.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no payload for dataset '{dataset_id}' at {path}. Payloads are gitignored; "
            f"provision them with `bash scripts/link_payloads.sh` (CLAUDE.md section 6a).")
    samples = []
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())
        if limit is not None:
            keys = keys[:limit]
        for k in keys:
            g = f[k]
            attrs = {a: g.attrs[a] for a in g.attrs}
            frame = np.asarray(g["final_frame"])
            rec = {
                "key": k,
                "dataset_id": dataset_id,
                "final_frame": frame,
                "n_species": int(frame.shape[0]),
                "L": float(attrs["L"]),
                "k_star": float(attrs.get("k_star", np.nan)),
                "morphology": str(attrs.get("morphology", "n/a")),
                "grid": int(attrs.get("grid", frame.shape[-1])),
                "attrs": attrs,
                "trajectory": (np.asarray(g["trajectory"])
                               if (with_trajectory and "trajectory" in g) else None),
                "has_trajectory": bool("trajectory" in g),
                "times": np.asarray(g["times"]) if "times" in g else None,
                "params": json.loads(attrs["params_json"]) if "params_json" in attrs else None,
                "x_star": np.asarray(g["x_star"]) if "x_star" in g else None,
                "D": np.asarray(g["D"]) if "D" in g else None,
            }
            samples.append(rec)
    return samples


# ======================================================================================
# re-simulation — actually RUNNING the simulation that made a training sample
# ======================================================================================
def is_resimulatable(sample):
    """Can this sample's simulation be re-run from what the payload stores?

    Only the ``gen_tg3`` families qualify. The requirement is the full generating kinetics
    (``params_json``) AND the simulation seed, because the initial condition is a seeded
    random perturbation of the homogeneous state — without the seed a re-run gives a
    different, merely statistically similar, pattern.

    The older ``three_gene_train/test/val`` sets store neither: their generator lived in
    the gitignored ``data/staging/`` and their kinetics are unrecoverable from the payload
    (this is the auditability defect that ``scripts/gen_tg3.py`` exists to fix). The
    ``*_classical`` families store kinetics but no seed, and their sub-family generators
    (``schnak_cross``, ``gm_relay``, ...) are not in ``rngrn.data.rd_models`` at all.
    """
    a = sample.get("attrs", {})
    return bool(sample.get("params")
                and "interaction_matrix" in (sample.get("params") or {})
                and "sim_seed" in a and "periods_per_box" in a
                and sample.get("x_star") is not None)


def resimulate(sample):
    """Re-run the reaction-diffusion simulation behind one training sample.

    Returns the ``gen_tg3.simulate_and_classify`` result dict. Verified bit-exact
    (relative L2 error 0.000e+00) on 6 of the 126 samples in ``three_gene_qvar`` /
    ``three_gene_multiL`` — the first three of each — NOT on the full datasets.
    """
    if not is_resimulatable(sample):
        raise ValueError(
            f"{sample['dataset_id']}/{sample['key']} cannot be re-simulated: "
            f"the payload does not carry both the generating kinetics and the sim seed")
    sys.path.insert(0, HERE)
    import gen_tg3                                                 # noqa: E402

    a = sample["attrs"]
    p = dict(sample["params"])
    p["_M"] = p["interaction_matrix"]
    p["x_star"] = np.asarray(sample["x_star"])
    p["k_star"] = float(a["k_star"])
    p["periods_per_box"] = int(a["periods_per_box"])
    p["sim_seed"] = int(a["sim_seed"])
    out = gen_tg3.simulate_and_classify(p, grid=int(a["grid"]), seed=int(a["sim_seed"]))
    if out is None:
        raise RuntimeError(
            f"re-simulation of {sample['dataset_id']}/{sample['key']} returned None "
            f"(diverged, or collapsed to homogeneous) — the stored sample did not")
    return out


# ======================================================================================
# the patterning verdict — measured, not asserted
# ======================================================================================
def patterning_verdict(field, L):
    """Is this 2-D field patterned? Returns the MEASUREMENTS behind the verdict.

    Two independent questions, deliberately kept apart rather than merged into one
    boolean, because they can disagree and the disagreement is the interesting case:

    * ``has_contrast`` — is there spatial contrast at all? ``cv`` is the spatial
      coefficient of variation, which is the generator's own accept/reject statistic.

      **This is nearly a tautology on the registered corpus and must not be reported as
      evidence of patterning.** Every generator here applies ``cv0 < 0.05 -> reject``
      (``gen_tg3.simulate_and_classify``), so a corpus filtered at 0.05 having a minimum
      above 0.05 is what the filter does. What it *does* establish is payload integrity:
      the cv recomputed from the stored frame should match the stored ``cv0`` attribute.

    * ``periodic`` — is that contrast PERIODIC? This is the informative screen. A few
      isolated blobs pass the cv test but produce a monotonically decaying spectrum.
      ``peak_bin`` is the RAPS bin holding the maximum; a field with a characteristic
      wavelength peaks at an interior bin.

    NOTE — ``has_contrast`` is deliberately NOT called ``patterned``. ``eval/rollout.py``
    already owns that name for a different quantity (rollout amplitude against
    ``max(1e-3, 0.02*|x*_0|)``), and that one is pre-registered. The two are never
    comparable; see docs/DECISIONS.md D-TDPLOT-1.

    ``k_star_obs`` is the image-only RAPS peak — the same estimator recovery uses — in
    physical angular units (rad / length).
    """
    field = np.asarray(field, float)
    mean = float(field.mean())
    cv = float(field.std() / max(abs(mean), 1e-9))
    kc, power, k_obs = raps(field, float(L))
    peak_bin = int(np.argmax(power))
    return {
        "cv": cv,
        "k_star_obs": float(k_obs),
        "wavelength_obs": float(2 * np.pi / k_obs) if k_obs > 0 else float("inf"),
        "peak_bin": peak_bin,
        "periods_across_box": float(L * k_obs / (2 * np.pi)) if k_obs > 0 else 0.0,
        "has_contrast": bool(cv >= CV_PATTERNED_MIN),
        "periodic": bool(peak_bin >= PEAK_BIN_MIN),
        "mean": mean,
        "min": float(field.min()),
        "max": float(field.max()),
    }


# ======================================================================================
# panels
# ======================================================================================
def _physical_ticks(L, n=5):
    return np.linspace(0.0, float(L), n)


def plot_field_panel(ax, field, L, *, title=None, cmap=DEFAULT_CMAP,
                     cbar_label="concentration", cbar=True, vmin=None, vmax=None):
    """Draw one concentration field with physical axes, tick marks and a colorbar.

    ``extent=(0, L, 0, L)`` is the point of the whole helper: the axes then read in length
    units, so a wavelength can be measured off the figure directly instead of in pixels.
    """
    field = np.asarray(field, float)
    if field.ndim != 2:
        raise ValueError(f"plot_field_panel needs a 2-D field, got shape {field.shape}")
    L = float(L)
    if not np.isfinite(L) or L <= 0:
        raise ValueError(f"domain extent L must be finite and positive, got {L}")

    im = ax.imshow(field, origin="lower", extent=(0.0, L, 0.0, L), cmap=cmap,
                   aspect="equal", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_xlabel("x  (length units)")
    ax.set_ylabel("y  (length units)")
    ax.set_xticks(_physical_ticks(L))
    ax.set_yticks(_physical_ticks(L))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    ax.tick_params(direction="out", length=3.5, width=0.8, labelsize=7)
    if title:
        ax.set_title(title, fontsize=7.5)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
    return im


def plot_raps_panel(ax, field, L, *, k_ref=None, color="#2b6cb0"):
    """Radially averaged power spectrum with the measured peak marked.

    Returns the measured k* (rad / length). ``k_ref`` — the generator's stored k* — is
    drawn for comparison when supplied; it is ground truth, legal here because this is a
    data-inspection figure, not a recovery criterion.
    """
    kc, power, kstar = raps(np.asarray(field, float), float(L))
    ax.plot(kc, power, color=color, lw=1.2, label="RAPS (measured)")
    ax.axvline(kstar, color="#e53e3e", ls="--", lw=1.1,
               label=f"$k^*_{{obs}}$ = {kstar:.3f}")
    if k_ref is not None and np.isfinite(k_ref):
        ax.axvline(float(k_ref), color="#38a169", ls=":", lw=1.4,
                   label=f"$k^*$ generator = {float(k_ref):.3f}")
    ax.set_xlabel("wavenumber $k$  (rad / length)")
    ax.set_ylabel("radially averaged power")
    ax.set_yscale("log")
    ax.set_xlim(0, min(kc.max(), 6 * max(kstar, 1e-6)))
    # A raw log axis here spans ~16 decades and squashes the peak into a flat line. Show
    # six decades below the maximum, which is where the whole shape of the spectrum lives.
    pmax = float(power.max())
    if pmax > 0:
        ax.set_ylim(pmax * 1e-6, pmax * 3)
    ax.tick_params(direction="out", length=3.5, width=0.8, labelsize=7)
    ax.legend(fontsize=6, framealpha=0.9)
    return kstar


# ======================================================================================
# figures
# ======================================================================================
def _panel_title(s, species):
    v = patterning_verdict(s["final_frame"][species], s["L"])
    flags = ""
    if not v["has_contrast"]:
        flags += "  NO CONTRAST"
    if not v["periodic"]:
        flags += "  NOT PERIODIC"
    return (f"{s['key']}  [{s['morphology']}]{flags}\n"
            f"L={s['L']:.1f}  $k^*$={s['k_star']:.3f}  cv={v['cv']:.2f}")


def gallery_figure(samples, *, ncols=6, title="", species=0, panel_size=2.35):
    """A contact sheet: one panel per sample, each self-describing.

    Per-panel colorbars are deliberate — the fields have wildly different concentration
    ranges, so one shared colorbar would make most panels unreadable.
    """
    if not samples:
        raise ValueError("gallery_figure got no samples to draw")
    n = len(samples)
    ncols = int(max(1, min(ncols, n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size * 1.45 * ncols, panel_size * 1.30 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, s in zip(axes, samples):
        sp = min(species, s["n_species"] - 1)
        plot_field_panel(ax, s["final_frame"][sp], s["L"],
                         title=_panel_title(s, sp),
                         cbar_label=f"{SPECIES_LABELS[sp]}  (conc.)")
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11, y=0.999)
    fig.tight_layout()
    return fig


def detail_figure(sample, *, panel_size=3.1):
    """All species of one sample side by side, plus the spectrum of species 0.

    This is the QUANTITATIVE patterning check: the spectrum shows whether the field has a
    single dominant finite wavenumber, which is what a saturated Turing pattern looks
    like, rather than broadband noise or monotone decay.

    A finite-wavenumber peak is NECESSARY, NOT SUFFICIENT. "Turing-unstable" is a
    statement about sigma(k) of the generating Jacobian and is not evaluated anywhere in
    this module. Do not read one as a proxy for the other.
    """
    s = sample
    ns = int(s["n_species"])
    fig, axes = plt.subplots(1, ns + 1, figsize=(panel_size * 1.45 * (ns + 1), panel_size * 1.25))
    axes = np.atleast_1d(axes).ravel()
    for i in range(ns):
        v = patterning_verdict(s["final_frame"][i], s["L"])
        plot_field_panel(axes[i], s["final_frame"][i], s["L"],
                         title=(f"species {i} ({SPECIES_LABELS[i]})\n"
                                f"cv={v['cv']:.3f}  range [{v['min']:.3g}, {v['max']:.3g}]"),
                         cbar_label=f"{SPECIES_LABELS[i]}  (concentration)")
    plot_raps_panel(axes[ns], s["final_frame"][0], s["L"], k_ref=s.get("k_star"))
    axes[ns].set_title("spectrum of species 0", fontsize=7.5)
    topo = s["attrs"].get("topology", s["attrs"].get("family", "n/a"))
    fig.suptitle(
        f"{s['dataset_id']} / {s['key']}   topology={topo}   morphology={s['morphology']}   "
        f"L={s['L']:.2f}   grid={s['grid']}",
        fontsize=10)
    fig.tight_layout()
    return fig


def resim_figure(*, stored, resim, L, title, cmap=DEFAULT_CMAP):
    """Stored frame vs a freshly RE-SIMULATED frame, and their difference.

    Returns ``(fig, rel_err)`` where ``rel_err`` is ``||resim - stored|| / ||stored||`` in
    the Frobenius norm — 0 for an exact reproduction.
    """
    stored = np.asarray(stored, float)
    resim = np.asarray(resim, float)
    if stored.shape != resim.shape:
        raise ValueError(f"shape mismatch: stored {stored.shape} vs re-simulated {resim.shape}")
    diff = resim - stored
    denom = float(np.linalg.norm(stored))
    rel_err = float(np.linalg.norm(diff) / denom) if denom > 0 else float(np.linalg.norm(diff))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    lo = float(min(stored.min(), resim.min()))
    hi = float(max(stored.max(), resim.max()))
    plot_field_panel(axes[0], stored, L, title="stored in payload.h5",
                     cbar_label="concentration", vmin=lo, vmax=hi)
    plot_field_panel(axes[1], resim, L, title="re-simulated now",
                     cbar_label="concentration", vmin=lo, vmax=hi)
    amp = float(np.abs(diff).max()) or 1.0
    plot_field_panel(axes[2], diff, L,
                     title=f"re-simulated - stored\nrelative L2 error = {rel_err:.2e}",
                     cbar_label="difference", cmap="coolwarm", vmin=-amp, vmax=amp)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig, rel_err


def trajectory_figure(sample, *, species=0, panel_size=2.5):
    """The stored simulation TRAJECTORY — the pattern forming over time."""
    traj, times = sample.get("trajectory"), sample.get("times")
    if traj is None:
        raise ValueError(f"{sample['dataset_id']}/{sample['key']} stores no trajectory")
    n = int(traj.shape[0])
    fig, axes = plt.subplots(1, n, figsize=(panel_size * 1.5 * n, panel_size * 1.35))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        f = traj[i][species]
        v = patterning_verdict(f, sample["L"])
        t = float(times[i]) if times is not None else float(i)
        plot_field_panel(axes[i], f, sample["L"], title=f"t = {t:.1f}\ncv={v['cv']:.3f}",
                         cbar_label=f"{SPECIES_LABELS[species]}  (conc.)")
    fig.suptitle(f"{sample['dataset_id']} / {sample['key']} — species {species} over time "
                 f"(L={sample['L']:.2f}, morphology={sample['morphology']})", fontsize=10)
    fig.tight_layout()
    return fig


def corpus_summary_figure(records, *, figsize=(15.5, 4.6)):
    """The corpus-level patterning check, so the verdict is not eyeball-only.

    (a) the spatial coefficient of variation per dataset against the generator's own
        ``cv < 0.05`` reject rule — anything at or below the line never patterned;
    (b) the image-only k* against the generator's stored k* — points on y=x mean the
        pattern in the image has the wavenumber the generating system predicts;
    (c) the morphology classes present per dataset.

    ``records`` are the dicts produced by the notebook's measurement pass: each needs
    ``dataset_id``, ``cv``, ``k_star_obs``, ``k_star`` and ``morphology``.
    """
    if not records:
        raise ValueError("corpus_summary_figure got no records")
    datasets = sorted({r["dataset_id"] for r in records})
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # (a) cv per dataset
    ax = axes[0]
    rng = np.random.default_rng(0)          # jitter only; never affects a reported number
    for i, ds in enumerate(datasets):
        vals = np.array([r["cv"] for r in records if r["dataset_id"] == ds], float)
        ax.scatter(i + rng.uniform(-0.22, 0.22, vals.size), vals, s=11, alpha=0.65,
                   color="#2b6cb0", edgecolors="none")
        ax.hlines(np.median(vals), i - 0.3, i + 0.3, color="black", lw=1.6, zorder=3)
    ax.axhline(CV_PATTERNED_MIN, color="#e53e3e", ls="--", lw=1.2,
               label=f"generator reject rule (cv < {CV_PATTERNED_MIN})")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=38, ha="right", fontsize=6.5)
    ax.set_ylabel("spatial coefficient of variation  (species 0)")
    ax.set_yscale("log")
    # Titles are DERIVED from the records, never asserted as static text: a hard-coded
    # "every sample is above the line" would keep printing over a panel showing points
    # below it. Panel (a) is also labelled as the near-tautology it is.
    n_ok = sum(1 for r in records if r["cv"] >= CV_PATTERNED_MIN)
    ax.set_title(f"(a) {n_ok}/{len(records)} samples above the generator's own cv reject\n"
                 f"rule — which the generator already enforced (near-tautology)",
                 fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.tick_params(direction="out", length=3.5, labelsize=7)

    # (b) measured k* vs the generator's k*
    ax = axes[1]
    ko = np.array([r["k_star_obs"] for r in records], float)
    kt = np.array([r["k_star"] for r in records], float)
    ok = np.isfinite(ko) & np.isfinite(kt) & (kt > 0)
    ax.scatter(kt[ok], ko[ok], s=13, alpha=0.6, color="#2b6cb0", edgecolors="none")
    lim = [0, float(max(kt[ok].max(), ko[ok].max())) * 1.05]
    ax.plot(lim, lim, color="#e53e3e", ls="--", lw=1.2, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("$k^*$ stored by the generator  (rad / length)")
    ax.set_ylabel("$k^*$ measured from the image  (rad / length)")
    rel = np.abs(ko[ok] - kt[ok]) / kt[ok]
    signed = (ko[ok] - kt[ok]) / kt[ok]
    ax.set_title(f"(b) measured vs generator $k^*$ — median |dev| {np.median(rel)*100:.1f}%, "
                 f"p90 {np.percentile(rel,90)*100:.1f}%\n"
                 f"signed median {np.median(signed)*100:+.1f}%, "
                 f"{100*(signed>0).mean():.0f}% above the line",
                 fontsize=8.5)
    ax.legend(fontsize=6.5)
    ax.tick_params(direction="out", length=3.5, labelsize=7)

    # (c) morphology classes
    ax = axes[2]
    classes = sorted({r["morphology"] for r in records})
    palette = plt.get_cmap("tab10")
    bottom = np.zeros(len(datasets))
    for j, c in enumerate(classes):
        counts = np.array([sum(1 for r in records
                               if r["dataset_id"] == ds and r["morphology"] == c)
                           for ds in datasets], float)
        ax.bar(range(len(datasets)), counts, bottom=bottom, label=c,
               color=palette(j % 10), edgecolor="white", lw=0.4)
        bottom += counts
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=38, ha="right", fontsize=6.5)
    ax.set_ylabel("samples")
    ax.set_title("(c) morphology classes present", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.tick_params(direction="out", length=3.5, labelsize=7)

    fig.tight_layout()
    return fig


def save(fig, out_dir, name, dpi=130):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}  ({os.path.getsize(path) / 1e6:.2f} MB)")
    return path
