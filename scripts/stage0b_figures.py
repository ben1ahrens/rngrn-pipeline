#!/usr/bin/env python
"""stage0b_figures.py — figures for Stage 0b (finite-mu robustness of the lifted circuit).

Every panel is drawn from the arrays scripts/stage0b_mu.py wrote under
experiments/figures_report/stage0b/arrays/; nothing is recomputed here, so the figures and
the audit trail cannot drift apart. Both the PNGs and those arrays are tracked in git.

  g1  turing volume vs mu, per system, with the pre-registered §3.2 bars and the two named
      finite mu values marked — THE DELIVERABLE.
  g2  the pattern ladder: QSS field beside the lifted field at each mu, 64x64.
  g3  the same ladder for the two RECOVERED networks (sample_0003/0004, prior-ON).
  g4  sigma_full(k) at several mu with the QSS curve overlaid (context).
  g5  dt-convergence and the mu -> 0 agreement with the QSS rollout — integrator evidence.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from rngrn.eval import lifted                                      # noqa: E402

OUT = "experiments/figures_report/stage0b"
ARR = os.path.join(OUT, "arrays")
MU_FINITE = (1e-3, 1e-2)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path} ({os.path.getsize(path)} bytes)")
    return path


def _json(name):
    with open(os.path.join(ARR, name)) as fh:
        return json.load(fh)


# ======================================================================================
# g1 — THE DELIVERABLE
# ======================================================================================
def fig_robust():
    z = np.load(os.path.join(ARR, "robust.npz"))
    rows = _json("robust.json")
    mu = z["mu_context"]
    levels = ["4p8pct", "10pct"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)
    for ax, lvl in zip(axes, levels):
        bar = lifted.PREREG_BARS[lvl]
        for r in [q for q in rows if q["level"] == lvl]:
            key = f"{r['label']}__{lvl}".replace("/", "_")
            v = z[f"{key}__vol"]
            rec = r["recovered"]
            ax.plot(mu, v, lw=(2.6 if rec else 1.0), alpha=(1.0 if rec else 0.45),
                    color=("C3" if rec else ("C0" if r["form"] == "competitive" else "C1")),
                    marker=("o" if rec else None), ms=4,
                    label=(r["label"] if rec else None), zorder=(5 if rec else 2))
        ax.axhline(bar, color="k", ls="--", lw=1.4)
        ax.text(mu[0], bar + 0.012, f"pre-registered §3.2 bar = {bar:.2f}", fontsize=8)
        for m in MU_FINITE:
            ax.axvline(m, color="0.35", ls=":", lw=1.2)
            ax.text(m, 1.02, f"mu={m:g}", fontsize=8, ha="center", color="0.25")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\mu$  (TF-promoter binding time / protein turnover time)")
        ax.set_title(f"parameter noise {100 * lifted.CLOUD_SIGMA_LEVELS[lvl]:g}%"
                     f"  ($\\sigma_{{\\log}}$={lifted.CLOUD_SIGMA_LEVELS[lvl]})")
        ax.set_ylim(-0.03, 1.08)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("strict Turing volume of the LIFTED system")
    axes[0].legend(fontsize=8, loc="lower left", title="RECOVERED networks")
    n = int(z["n_samples"])
    fig.suptitle(
        "Stage 0b — is the circuit still robust once the QSS reduction is UNDONE?\n"
        f"Strict Turing volume (max Re eig $J_{{full}}$ < 0 at k=0 AND some k>0 unstable) of "
        f"the FULL $N+2N^2=21$-dimensional lifted system, {n} lognormal draws per point, "
        f"seed {int(z['seed'])}, N=3, n_hill=2.\n"
        "One cloud is drawn per system and EVERY $\\mu$ is evaluated on the SAME draws, so "
        "the only thing changing along a curve is the gate timescale. Thick red = the two "
        "recovered networks carrying the project's claim;\nthin blue/orange = box-"
        "constrained synthetic survivors (competitive / nc1). The $\\mu\\to0$ end of each "
        "curve IS that system's own QSS volume under the same perturbation model.",
        fontsize=8.4)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, "g1_robust_vs_mu.png")


# ======================================================================================
# g2 / g3 — the pattern ladders
# ======================================================================================
def _ladder_fig(tag, title, fname):
    p = os.path.join(ARR, f"{tag}.npz")
    if not os.path.exists(p):
        print(f"  skip {fname}: {p} missing")
        return None
    z = np.load(p, allow_pickle=False)
    labels = z["label"].astype(str)
    mus = z["mu"].astype(float)
    uniq = list(dict.fromkeys(labels))
    cols = sorted(set(mus))
    fig, axes = plt.subplots(len(uniq), len(cols),
                             figsize=(2.9 * len(cols), 4.3 * len(uniq)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, lab in enumerate(uniq):
        for c, m in enumerate(cols):
            k = np.where((labels == lab) & (mus == m))[0]
            if not k.size:
                continue
            i = int(k[0])
            ax = axes[r][c]
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            ch = int(z["chan"][i])
            ax.imshow(z["fields"][i][ch], cmap="viridis", origin="lower")
            patt = bool(z["patterned_any"][i])
            stop = str(z["stopped_reason"][i])
            trunc = " TRUNCATED" if stop == "step_budget" else ""
            head = "QSS (reduced)" if m == 0 else f"lifted, $\\mu$={m:g}"
            ax.set_title(
                f"{lab}\n{head}{trunc}\n"
                f"{'PATTERNED' if patt else 'NOT patterned'}  ch{ch}  "
                f"std$_{{max}}$={float(z['std_max'][i]):.3g}\n"
                f"$k^*_{{field}}$={float(z['kstar_field'][i]):.4g}  "
                f"morph={str(z['morphology'][i])}\n"
                f"rel dev vs QSS={float(z['rel_dev_vs_qss'][i]):.2e}  stop={stop}",
                fontsize=6.8, color=("k" if patt else "C3"))
    fig.suptitle(title, fontsize=8.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=6.0)
    return _save(fig, fname)


def fig_ladder():
    return _ladder_fig(
        "ladder",
        "Stage 0b — VISUAL VERIFICATION: the QSS field beside the LIFTED field at finite "
        "$\\mu$, 64x64 periodic, identical L, seed, noise and dt in every row.\n"
        "The lifted runs integrate the full (x, GA, GR) system with the gate substep solved "
        "EXACTLY at frozen x (Strang splitting), so no finite-$\\mu$ pattern loss here can "
        "be an integrator artefact; see g5.\n"
        "Each panel shows the channel with the largest spatial std (channel 0 is arbitrary "
        "for these shuffled synthetic draws — the Stage-0 channel-0 trap).",
        "g2_pattern_ladder.png")


def fig_recovered():
    return _ladder_fig(
        "recovered",
        "Stage 0b — THE HEADLINE CASE: the two RECOVERED networks (three_gene_qvar "
        "sample_0003 / sample_0004, prior-ON, plausibility_score = 1.0 AND patterned)\n"
        "through the same $\\mu$ ladder. These are the circuits carrying the project's "
        "claim today; the QSS column is what the project reports now.",
        "g3_recovered_ladder.png")


# ======================================================================================
# g4 — the extended dispersion relation (context)
# ======================================================================================
MU_SHOW = (1e-4, 1e-2, 1e-1, 1.0, 10.0)


def _dispersion_curves():
    """sigma_full(k) at MU_SHOW plus the QSS curve, for the two RECOVERED networks and two
    synthetic survivors. Computed here (it is ~1 s of 21x21 eigenproblems) and SAVED, so the
    panel still has a committed audit trail behind it."""
    import sys as _sys
    _sys.path.insert(0, HERE)
    from stage0_bio_viability import (ARR as S0, balanced_picks, build_model, params_at)
    from rngrn.io import load_checkpoint
    from rngrn.losses.terms import steady_state

    runs = "experiments/stage0_bioviab/runs"
    systems = []
    for s, rid in (("rec/sample_0003", "stage0_prior_on_0003_20260803_130206"),
                   ("rec/sample_0004", "stage0_prior_on_0004_20260803_130302")):
        m, _ = load_checkpoint(os.path.join(runs, rid))
        systems.append((s, m))
    zh = np.load(os.path.join(S0, "part1_harvest_generator.npz"))
    for form in ("competitive", "nc1"):
        scheme, i = balanced_picks(zh, form, 1)[0]
        p = params_at(zh, form, scheme, i)
        systems.append((f"{form[:4]}/{scheme}/{i}",
                        build_model(p["alpha"], p["delta"], p["beta"], p["D"], p["s"],
                                    p["gate_logit"], form)))
    out = {}
    for tag, m in systems:
        xs, _ = steady_state(m)
        xstar = xs.detach().cpu().numpy()
        out[f"{tag}__qss"] = lifted.qss_dispersion(m, xstar)[0]
        J1 = lifted.lifted_jacobian(m, xstar, 1.0)
        for mu in MU_SHOW:
            out[f"{tag}__mu{mu:g}"] = lifted.lifted_dispersion(
                m, xstar, mu, J=lifted.rescale_mu(J1, m.N, mu))[0]
    np.savez_compressed(os.path.join(ARR, "dispersion_curves.npz"),
                        kgrid=lifted.KGRID, mus=np.array(MU_SHOW), **out)
    print(f"  wrote {ARR}/dispersion_curves.npz")
    return [t for t, _ in systems], np.load(os.path.join(ARR, "dispersion_curves.npz"))


def fig_dispersion():
    show, z = _dispersion_curves()
    k = z["kgrid"]
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 4.4), squeeze=False)
    for ax, t in zip(axes[0], show):
        sq = z[f"{t}__qss"]
        ax.plot(k[1:], sq[1:], "k-", lw=2.4, label="QSS (reduced, 3x3)")
        for j, m in enumerate(MU_SHOW):
            ax.plot(k[1:], z[f"{t}__mu{m:g}"][1:], lw=1.2, alpha=0.85, color=f"C{j}",
                    ls="--", label=f"lifted 21x21, $\\mu$={m:g}")
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_xscale("log")
        ax.set_ylim(-2.0, max(0.3, float(np.nanmax(sq[1:])) * 1.8))
        ax.set_xlabel("k")
        ax.set_title(t, fontsize=8)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(r"$\sigma(k) = \max \mathrm{Re}\,\mathrm{eig}(J - k^2 D)$")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(
        "Stage 0b (context) — the extended dispersion relation. Lifted: 21x21 $J_{full}$ "
        "with diffusion diag($D_x$, 0, 0) — only x diffuses, promoters are DNA-bound.\n"
        "The lifted curve converges to the QSS one as $\\mu\\to0$ at the O($\\mu$) rate a "
        "singular perturbation must give (measured, tests/test_lifted.py).",
        fontsize=8.4)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, "g4_dispersion_mu.png")


# ======================================================================================
# g5 — integrator evidence
# ======================================================================================
def fig_dtconv():
    p = os.path.join(ARR, "dtconv.npz")
    if not os.path.exists(p):
        print("  skip g5: dtconv.npz missing")
        return None
    rows = _json("dtconv.json")
    halved = [r for r in rows if int(r["dt_div"]) > 0]
    cross = [r for r in rows if int(r["dt_div"]) == 0]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    ax = axes[0]
    keys = sorted({(r["form"], float(r["mu"])) for r in halved})
    for j, (form, mu) in enumerate(keys):
        sub = sorted([r for r in halved if r["form"] == form and float(r["mu"]) == mu],
                     key=lambda r: int(r["dt_div"]))
        ax.plot([1.0 / int(r["dt_div"]) for r in sub], [float(r["kstar"]) for r in sub],
                "o-", color=f"C{j}", label=f"{form}, $\\mu$={mu:g}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("dt (relative to the default)")
    ax.set_ylabel("$k^*$ of the final field")
    ax.set_title("halve dt twice: $k^*$ does not move")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    lbl, val = [], []
    for r in halved:
        if int(r["dt_div"]) > 1:
            lbl.append(f"{r['form'][:4]}\n$\\mu$={float(r['mu']):g}\ndt/{int(r['dt_div'])}")
            val.append(max(float(r["rel_field_dev"]), 1e-17))
    for r in cross:
        lbl.append(f"{r['form'][:4]}\nQSS vs\nlift@1e-6")
        val.append(max(float(r["rel_field_dev"]), 1e-17))
    ax.bar(range(len(val)), val, color=["C0"] * (len(val) - len(cross)) + ["C3"] * len(cross))
    ax.set_yscale("log")
    ax.set_xticks(range(len(val)))
    ax.set_xticklabels(lbl, fontsize=7)
    ax.set_ylabel("max|$\\Delta$field| / max|field|")
    ax.set_title("relative field change (blue) and the $\\mu\\to0$ QSS cross-check (red)")
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle(
        "Stage 0b — INTEGRATOR EVIDENCE. A stiff scheme that silently damped the "
        "instability would fake 'the pattern died at finite $\\mu$', the most dangerous "
        "failure available here.\n"
        "The gate substep is EXACT at frozen x (rank-one matrix exponential for "
        "competitive, diagonal for nc1), so it cannot damp; these panels are the empirical "
        "check that the Strang splitting error is small at the dt used.",
        fontsize=8.4)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return _save(fig, "g5_dt_convergence.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=None)
    args = ap.parse_args(argv)
    os.chdir(os.path.join(HERE, ".."))
    all_figs = dict(g1=fig_robust, g2=fig_ladder, g3=fig_recovered,
                    g4=fig_dispersion, g5=fig_dtconv)
    for k in (args.figs or list(all_figs)):
        print(f"{k}:")
        all_figs[k]()


if __name__ == "__main__":
    main()
