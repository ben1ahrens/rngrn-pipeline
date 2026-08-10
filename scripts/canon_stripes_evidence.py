#!/usr/bin/env python
"""canon_stripes_evidence.py — the figure behind D-CANON-2.

The claim is that ``stripes`` in this corpus is largely an artefact of a small box rather
than a distinct pattern type. That claim rests on three measurements, and a reader should
be able to see all three at once rather than take them on trust:

  (a) anisotropy against periods-per-box, with the A = 0.55 stripe cut drawn. Anisotropy is
      a nematic order parameter over the dominant Fourier ring: a small box admits few
      orientations, so power concentrates on one axis and A rises.
  (b) the fraction of samples labelled ``stripes`` per periods-per-box bin, which falls to
      zero above p = 10.
  (c) the outcome of the stability probe — re-simulating each system at a different box
      size and asking whether its class survives.

Data side: reads payload.h5. Not imported by anything under src/rngrn/.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import canon_select as CS                                          # noqa: E402
import td_figures as TD                                            # noqa: E402

OUT = os.path.join(HERE, "..", "experiments", "figures_report", "canonical")
CLASS_COLOUR = {"spots": "#2b6cb0", "labyrinth": "#38a169", "stripes": "#e53e3e"}


def collect():
    rows = []
    for ds in CS.ELIGIBLE_DATASETS:
        for s in TD.load_samples(ds):
            rows.append({
                "p": int(round(s["L"] * s["k_star"] / (2 * np.pi))),
                "A": float(s["attrs"]["anisotropy"]),
                "morphology": str(s["morphology"]),
            })
    return rows


def figure(rows, stability=None):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    P = np.array([r["p"] for r in rows])
    A = np.array([r["A"] for r in rows])

    # (a) anisotropy vs periods-per-box
    ax = axes[0]
    for cls in ("spots", "labyrinth", "stripes"):
        m = np.array([r["morphology"] == cls for r in rows])
        if m.any():
            ax.scatter(P[m], A[m], s=22, alpha=0.75, label=cls,
                       color=CLASS_COLOUR[cls], edgecolors="none")
    ax.axhline(CS.ANISO_STRIPES_MIN, color="black", ls="--", lw=1.2,
               label=f"stripe cut A = {CS.ANISO_STRIPES_MIN}")
    ax.axvspan(16, 32, color="#805ad5", alpha=0.12)
    ax.text(17.5, 0.95, "canonical\nrange", fontsize=7, color="#553c9a", va="top")
    r = float(np.corrcoef(P, A)[0, 1])
    ax.set_xlabel("periods per box  $p = L k^* / 2\\pi$")
    ax.set_ylabel("anisotropy $A$  (nematic order of the Fourier ring)")
    ax.set_title(f"(a) anisotropy falls as the box grows\nPearson r = {r:+.3f}", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.tick_params(direction="out", length=3.5, labelsize=7)

    # (b) stripes fraction per p bin
    ax = axes[1]
    bins = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 14)]
    xs, fr, ns = [], [], []
    for lo, hi in bins:
        m = (P >= lo) & (P <= hi)
        if not m.any():
            continue
        xs.append(f"{lo}-{hi}")
        fr.append(100 * np.mean([rows[i]["morphology"] == "stripes"
                                 for i in np.where(m)[0]]))
        ns.append(int(m.sum()))
    bars = ax.bar(xs, fr, color="#e53e3e", alpha=0.85, edgecolor="white")
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7, f"n={n}",
                ha="center", fontsize=6.5)
    ax.set_xlabel("periods per box")
    ax.set_ylabel("% of samples labelled stripes")
    ax.set_title("(b) the stripes label vanishes above p = 10,\n"
                 "far below the canonical range of 16-32", fontsize=9)
    ax.tick_params(direction="out", length=3.5, labelsize=7)

    # (c) stability probe outcome
    ax = axes[2]
    if stability:
        labels = ["spots", "labyrinth", "stripes"]
        stable = [stability.get(c, {}).get("stable", 0) for c in labels]
        flipped = [stability.get(c, {}).get("flipped", 0) for c in labels]
        x = np.arange(len(labels))
        ax.bar(x, stable, 0.6, label="label survives", color="#38a169")
        ax.bar(x, flipped, 0.6, bottom=stable, label="label FLIPS", color="#e53e3e")
        for i, (s, f) in enumerate(zip(stable, flipped)):
            if s + f:
                ax.text(i, s + f + 0.3, f"{f}/{s+f} flip", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("systems")
        ax.set_title("(c) re-simulated at a different box size:\n"
                     "every stripes system loses its label", fontsize=9)
        ax.legend(fontsize=6.5)
    ax.tick_params(direction="out", length=3.5, labelsize=7)

    fig.suptitle("D-CANON-2 — 'stripes' is largely a small-box artefact, not a pattern type",
                 fontsize=11)
    fig.tight_layout()
    return fig


def main(argv=None):
    rows = collect()
    cache_path = os.path.join(HERE, "..", "data", "canonical_stability_cache.json")
    stability = None
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))
        table = {r["uid"]: r for r in CS.candidate_table()}
        stability = {}
        for uid, ok in cache.items():
            cls = table[uid]["morphology"] if uid in table else None
            if cls is None:
                continue
            d = stability.setdefault(cls, {"stable": 0, "flipped": 0})
            d["stable" if ok else "flipped"] += 1
        missing = [c for c in ("spots", "labyrinth", "stripes") if c not in stability]
        if missing:
            raise RuntimeError(
                f"no stability record for {missing}; re-run scripts/canon_select.py so the "
                f"cache covers every gated class. This figure must not fall back to a "
                f"hardcoded count — the whole point is that the numbers are traceable.")
    fig = figure(rows, stability)
    TD.save(fig, OUT, "s1_stripes_is_a_small_box_artefact.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
