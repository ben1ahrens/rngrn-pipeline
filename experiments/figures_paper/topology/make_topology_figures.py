#!/usr/bin/env python
"""make_topology_figures.py — GRN topology + non-identifiability figures for the paper.

SCORING-SIDE: READS GROUND TRUTH; NEVER IMPORT FROM TRAINING CODE.
================================================================
This script opens ``payload.h5`` and reads the answer key (``interaction_matrix``,
``jacobian``, ``D``, ``x_star``, ``params_json``). That is legal here because it runs
strictly POST-HOC, to draw reporting figures from runs that have already finished. It
imports only:

  * ``rngrn.eval.topology.plot_topology``  — a pure renderer that reads a model's own
    learned parameters and nothing else (see its module docstring),
  * ``rngrn.scoring.permutation``          — the scoring-side alignment module.

It imports NOTHING from ``rngrn.model`` / ``recover`` / ``losses`` / ``train``, and it is
deliberately located OUTSIDE ``scripts/`` so it is not importable by bare top-level name
and therefore does not need an entry in ``tests/test_firewall.py::FORBIDDEN``
(CLAUDE.md section 5).

WHY THE LEARNED PARAMETERS COME FROM JSON, NOT THE CHECKPOINT
-------------------------------------------------------------
``checkpoints/model.pt`` stores UNCONSTRAINED ``theta_*`` tensors, which only
``rngrn.model.RNGRN`` can decode into ``KA/KR/alpha/beta/delta/D`` — importing it would
put training code in this script's import graph for no benefit. The decoded values are
already recorded in ``results/train_results.json`` under ``recovered.*``. Those two were
verified byte-identical: decoding all 10 D5 checkpoints through ``RNGRN`` and comparing
against the JSON gave a maximum relative discrepancy of **0.0** across
``KA, KR, alpha, beta, delta, D`` (verification run 2026-08-18; see README.md).

Usage
-----
    .venv/bin/python experiments/figures_paper/topology/make_topology_figures.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
MAIN = Path("/home/benja/projects/personal/rngrn/rngrn-pipeline")
sys.path.insert(0, str(WORKTREE / "src"))

from rngrn.eval.topology import plot_topology                      # noqa: E402
from rngrn.scoring.permutation import permuted_sign_match, sign_structure  # noqa: E402

FIGS = HERE / "figures"
DATA = HERE / "data"
PANELS = DATA / "panels"
for d in (FIGS, DATA, PANELS):
    d.mkdir(parents=True, exist_ok=True)

DPI = 220
RNG_SEED = 20260818          # fixed so the shuffled null is reproducible
N_NULL_DRAWS = 200_000

# Two zero-tolerances are reported side by side ON PURPOSE, because they answer different
# questions and they DISAGREE on this data.
#   1e-9  — the pipeline convention (validate._sign_structure:77 and
#           scoring.permutation.SIGN_ZERO_RTOL). It is what the recorded
#           `sign_match_frac` uses, so it is the number that is comparable to every
#           previously reported score. It treats ANY nonzero float as a real edge.
#   0.05  — the STRUCTURAL convention, the one `metric.repro_sign_vector` already uses
#           (metric.repro_sign_zero_rtol == 0.05 in every run record here). An entry
#           below 5% of max|J| is not a regulatory edge in any meaningful sense.
# Measured consequence on this data: c2_P target A seeds 1/4/7 share ONE sign vector at
# 1e-9 but are THREE distinct structures at 0.05, because entries as small as 2e-4 of
# max|J| are being counted as edges. So 1e-9 UNDERSTATES the diversity and INFLATES the
# top scores. Both are reported; neither is silently preferred.
RTOL_PIPELINE = 1e-9
RTOL_STRUCTURAL = 0.05


def sign_structure_rtol(J, rtol):
    """Elementwise sign with |J_ij| <= rtol * max|J| collapsed to 0.

    Same rule as ``scoring.permutation.sign_structure`` / ``validate._sign_structure``,
    with the tolerance exposed so the pipeline and structural conventions can be
    compared directly.
    """
    a = np.asarray(J, float)
    scale = np.abs(a).max()
    return np.sign(np.where(np.abs(a) > rtol * (scale + 1e-12), a, 0.0))

# ---------------------------------------------------------------------------
# Run groups. Each entry: (group key, run glob, payload path, sample key, pretty label)
# ---------------------------------------------------------------------------
GROUPS = [
    ("D5",
     WORKTREE / "experiments/diag_fft/d5/runs",
     "m3_registry_20260812_092043_seed*",
     WORKTREE / "data/datasets/turing_labyrinth/payload.h5",
     "sample_0000",
     "D5 / turing_labyrinth sample_0000"),
    ("c2P_A",
     MAIN / "experiments/c2_P_t8k8_consol/runs",
     "nc1_m3_registry_20260803_190648_seed*",
     MAIN / "data/datasets/three_gene_qvar/payload.h5",
     "sample_0001",
     "c2_P target A / three_gene_qvar sample_0001"),
    ("c2P_B",
     MAIN / "experiments/c2_P_t8k8_consol/runs",
     "nc1_m3_registry_20260803_192351_seed*",
     MAIN / "data/datasets/three_gene_qvar/payload.h5",
     "sample_0004",
     "c2_P target B / three_gene_qvar sample_0004"),
]

OBSERVED_IDX = (0, 1, 2)     # every run here is fully observed (N = m = 3)


class ParamView:
    """Duck-typed stand-in that exposes exactly what ``plot_topology`` reads.

    ``plot_topology`` touches only ``.KA .KR .alpha .beta .delta .D .N`` and accepts
    plain numpy arrays, so no model object is needed.
    """

    def __init__(self, KA, KR, alpha, beta, delta, D):
        self.KA = np.asarray(KA, float)
        self.KR = np.asarray(KR, float)
        self.alpha = np.asarray(alpha, float)
        self.beta = np.asarray(beta, float)
        self.delta = np.asarray(delta, float)
        self.D = np.asarray(D, float)
        self.N = int(self.KA.shape[0])

    def permuted(self, perm):
        """Relabel species by ``perm`` (``perm[i]`` = model index at aligned slot i)."""
        p = list(perm)
        ix = np.ix_(p, p)
        alpha = self.alpha[ix] if self.alpha.ndim == 2 else self.alpha[p]
        return ParamView(self.KA[ix], self.KR[ix], alpha,
                         self.beta[p], self.delta[p], self.D[p])


def read_truth(payload: Path, sample: str) -> dict:
    """Answer-key read. Ground truth — post-hoc reporting only."""
    with h5py.File(payload, "r") as f:
        g = f[sample]
        params = json.loads(g.attrs["params_json"])
        return {
            "interaction_matrix": np.array(g["interaction_matrix"]),
            "J_true": np.array(g["jacobian"], dtype=float),
            "D_true": np.array(g["D"], dtype=float),
            "x_star_true": np.array(g["x_star"], dtype=float),
            "params": params,
            "topology_name": str(g.attrs["topology"]),
            "system_id": int(g.attrs["system_id"]),
            "L": float(g.attrs["L"]),
            "k_star": float(g.attrs["k_star"]),
            "payload": str(payload),
            "sample": sample,
        }


def truth_view(truth: dict) -> ParamView:
    """Render-ready view of the TRUE network.

    HONEST LIMITATION, stated at the site that produces it: the generator's reaction is
    ``dx_i/dt = b_i + V_i * prod_j g(x_j; K_i, n) - mu_i * x_i`` (scripts/gen_tg3.py:93),
    in which ``K`` is indexed by the TARGET gene and shared across all of that gene's
    regulators. The truth therefore has NO per-edge binding strength comparable to the
    model's ``KA/KR``. Edges in the truth panel are consequently drawn at UNIFORM unit
    magnitude, and edge WIDTH carries meaning only in the learned panels. The per-gene
    quantities do map one-to-one by role: V -> alpha, b -> beta, mu -> delta, D -> D.
    """
    IM = truth["interaction_matrix"]
    p = truth["params"]
    return ParamView(
        KA=(IM > 0).astype(float),
        KR=(IM < 0).astype(float),
        alpha=np.asarray(p["V"], float),
        beta=np.asarray(p["b"], float),
        delta=np.asarray(p["mu"], float),
        D=np.asarray(p["D"], float),
    )


def harvest():
    """Collect every run's metrics and re-run the permutation search from scratch."""
    out = {}
    for key, root, pat, payload, sample, label in GROUPS:
        truth = read_truth(payload, sample)
        J_true = truth["J_true"]
        runs = []
        for d in sorted(root.glob(pat)):
            res = json.loads((d / "results/train_results.json").read_text())
            rec, met = res["recovered"], res["metric"]
            J_rec = np.array(rec["J"], dtype=float)

            # Re-run the search rather than trusting the stored best_perm: every stored
            # value has n_permutations_searched == 1 (identity only).
            pinned = permuted_sign_match(J_rec, J_true, OBSERVED_IDX, free_observed=False)
            free = permuted_sign_match(J_rec, J_true, OBSERVED_IDX, free_observed=True)

            s_true_struct = sign_structure_rtol(J_true, RTOL_STRUCTURAL)
            s_rec_struct = sign_structure_rtol(J_rec, RTOL_STRUCTURAL)

            runs.append({
                "run_dir": str(d),
                "seed": int(d.name.split("seed")[-1]),
                "sign_vector_rtol1e-9": sign_structure(J_rec).astype(int).ravel().tolist(),
                "sign_vector_rtol0.05": s_rec_struct.astype(int).ravel().tolist(),
                "sign_match_frac_structural": float(np.mean(s_rec_struct == s_true_struct)),
                "repro_sign_vector_rtol0.05": met["repro_sign_vector"],
                "J_rec": J_rec.tolist(),
                "params": {k: rec[k] for k in ("KA", "KR", "alpha", "beta", "delta", "D")},
                "stored_sign_match_frac": met.get("sign_match_frac"),
                "stored_best_perm": met.get("best_perm"),
                "stored_n_permutations_searched": met.get("n_permutations_searched"),
                "rerun_pinned": pinned,
                "rerun_free6": free,
                "recovered_turing": met.get("recovered_turing"),
                "rollout_patterned": met.get("rollout_patterned"),
                "morphology_match": met.get("morphology_match"),
                "morphology_pred": met.get("morphology_pred"),
                "kstar_rel_err": met.get("kstar_rel_err"),
                "kstar_model": met.get("kstar_model"),
                "D_ratio": met.get("repro_D_ratio"),
            })
        out[key] = {"label": label, "truth": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                              for k, v in truth.items()},
                    "runs": runs}
    return out


# ---------------------------------------------------------------------------
# Panel rendering / grid composition
# ---------------------------------------------------------------------------
def render_panel(view: ParamView, title: str, path: Path) -> Path:
    fig = plot_topology(view, title=title)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def compose_grid(panel_paths, titles, ncols, out_path, suptitle, subcaption=None,
                 cell_w=6.2):
    import matplotlib.image as mpimg
    imgs = [mpimg.imread(p) for p in panel_paths]
    n = len(imgs)
    nrows = int(np.ceil(n / ncols))
    ar = imgs[0].shape[0] / imgs[0].shape[1]
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell_w * ncols, cell_w * ar * nrows + 1.0))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=11, pad=4)
    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=0.997)
    if subcaption:
        fig.text(0.5, 0.004, subcaption, ha="center", va="bottom", fontsize=10,
                 color="#333333", wrap=True)
    fig.tight_layout(rect=[0, 0.025 if subcaption else 0, 1, 0.972])
    fig.subplots_adjust(hspace=0.16)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_topology_grids(H):
    """Figure 1 and 2: learned topologies, one panel per seed."""
    # --- Fig 1: D5's four Turing-positive seeds -----------------------------
    d5 = H["D5"]["runs"]
    turing = [r for r in d5 if r["recovered_turing"]]
    paths, titles = [], []
    for r in turing:
        v = ParamView(**r["params"])
        p = render_panel(v, f"D5 seed {r['seed']}", PANELS / f"d5_seed{r['seed']}.png")
        paths.append(p)
        titles.append(f"seed {r['seed']}  |  patterned={r['rollout_patterned']}  "
                      f"morph={r['morphology_pred']}  "
                      f"sign match={r['rerun_pinned']['sign_match_frac_identity']:.3f}")
    compose_grid(
        paths, titles, 2, FIGS / "fig1_d5_turing_topologies.png",
        "Learned GRN topologies — the four Turing-positive D5 seeds",
        "All four reproduce the target labyrinth morphology; all four learned different "
        "networks. Source: experiments/diag_fft/d5/runs/m3_registry_20260812_092043_seed{1,3,5,6}.")

    # --- Fig 2: all 16 c2_P seeds ------------------------------------------
    paths, titles = [], []
    for key, tag in (("c2P_A", "A"), ("c2P_B", "B")):
        for r in H[key]["runs"]:
            v = ParamView(**r["params"])
            p = render_panel(v, f"c2_P target {tag} — seed {r['seed']}",
                             PANELS / f"{key}_seed{r['seed']}.png")
            paths.append(p)
            titles.append(f"target {tag}, seed {r['seed']}  |  "
                          f"patterned={r['rollout_patterned']}  "
                          f"sign match={r['rerun_pinned']['sign_match_frac_identity']:.3f}")
    compose_grid(
        paths, titles, 4, FIGS / "fig2_c2P_topologies.png",
        "Learned GRN topologies — all 16 c2_P seeds (8 per target)",
        "Two targets x 8 seeds. 16/16 sign structures are distinct. Source: "
        "experiments/c2_P_t8k8_consol/runs/nc1_m3_registry_2026080{3_190648,3_192351}_seed*.",
        cell_w=5.4)


def _sign_matrix_ax(ax, S, title, cmap, norm):
    ax.imshow(S, cmap=cmap, norm=norm)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            ax.text(j, i, {-1: "−", 0: "0", 1: "+"}[int(S[i, j])],
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="#111111")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([f"$x_{i+1}$" for i in range(3)], fontsize=9)
    ax.set_yticklabels([f"$x_{i+1}$" for i in range(3)], fontsize=9)
    ax.set_title(title, fontsize=10)


def fig_learned_vs_true(H):
    """Figure 3: learned network beside the true network, per target, 6-perm aligned."""
    cmap = ListedColormap(["#c0392b", "#ecf0f1", "#2471a3"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    for key, tag in (("c2P_A", "A"), ("c2P_B", "B")):
        grp = H[key]
        truth = grp["truth"]
        J_true = np.array(truth["J_true"], float)
        S_true = sign_structure(J_true).astype(int)

        # representative seeds: best / median / worst by the 6-permutation aligned score
        runs = sorted(grp["runs"], key=lambda r: r["rerun_free6"]["sign_match_frac_aligned"])
        picks = [("worst", runs[0]), ("median", runs[len(runs) // 2]), ("best", runs[-1])]

        tv = truth_view({k: (np.array(v) if k in ("interaction_matrix", "J_true", "D_true",
                                                  "x_star_true") else v)
                         for k, v in truth.items()})
        tpath = render_panel(tv, f"TRUE network — {truth['topology_name']}",
                             PANELS / f"{key}_TRUE.png")

        paths = [tpath]
        titles = [f"GROUND TRUTH ({truth['topology_name']}, system_id "
                  f"{truth['system_id']})\nedge widths uniform: truth has no per-edge strength"]
        for role, r in picks:
            perm = r["rerun_free6"]["best_perm"]
            v = ParamView(**r["params"]).permuted(perm)
            p = render_panel(v, f"LEARNED — seed {r['seed']}, aligned by perm {perm}",
                             PANELS / f"{key}_seed{r['seed']}_aligned.png")
            paths.append(p)
            titles.append(
                f"seed {r['seed']} ({role} by 6-perm)   perm={perm}\n"
                f"sign_match_frac: pinned(1 perm, admissible)="
                f"{r['rerun_pinned']['sign_match_frac_identity']:.3f}   "
                f"free(6 perm, upper bound)="
                f"{r['rerun_free6']['sign_match_frac_aligned']:.3f}")

        compose_grid(
            paths, titles, 4, FIGS / f"fig3_learned_vs_true_target{tag}.png",
            f"Learned vs true network — c2_P target {tag} ({truth['sample']})",
            "Learned panels are relabelled by the best of all 6 permutations (free_observed=True). "
            "NOTE: these runs are FULLY OBSERVED (N=m=3), so the gate pins every channel and the "
            "admissible (pinned) search has exactly 1 candidate; the 6-permutation score discards a "
            "KNOWN correspondence and is therefore an UPPER BOUND, not the headline number.",
            cell_w=5.6)

        # companion: Jacobian sign matrices, which is what the score actually compares
        fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.1))
        _sign_matrix_ax(axes[0], S_true, "TRUE sign(J)", cmap, norm)
        for ax, (role, r) in zip(axes[1:], picks):
            perm = list(r["rerun_free6"]["best_perm"])
            S = sign_structure(np.array(r["J_rec"], float))[np.ix_(perm, perm)].astype(int)
            frac = r["rerun_free6"]["sign_match_frac_aligned"]
            pin = r["rerun_pinned"]["sign_match_frac_identity"]
            ax.set_facecolor("white")
            # BOTH numbers, because the 6-perm score is an upper bound obtained by
            # discarding a correspondence the gate actually pins (these runs are fully
            # observed). Showing only the aligned score would flatter the result.
            # NB: worst/median/best rank by the 6-perm score, and the pinned score does NOT
            # follow that order (target B seed 7 is "best" on 6-perm but lowest pinned).
            # The label says which ranking it is so the two are not read as one.
            _sign_matrix_ax(ax, S, f"seed {r['seed']} ({role} by 6-perm), perm {tuple(perm)}\n"
                                   f"6-perm (upper bound) = {frac:.3f} "
                                   f"({int(round(frac*9))}/9)\n"
                                   f"pinned (admissible)  = {pin:.3f} "
                                   f"({int(round(pin*9))}/9)", cmap, norm)
        fig.suptitle(f"Jacobian sign structure — c2_P target {tag}: what sign_match_frac "
                     f"actually compares", fontsize=13, fontweight="bold")
        fig.legend(handles=[Patch(facecolor="#2471a3", label="+ (activating)"),
                            Patch(facecolor="#ecf0f1", edgecolor="#999", label="0"),
                            Patch(facecolor="#c0392b", label="− (repressing)")],
                   loc="lower center", ncol=3, frameon=False, fontsize=10,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.10, 1, 0.90])
        fig.savefig(FIGS / f"fig3b_sign_matrices_target{tag}.png", dpi=DPI,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {FIGS / f'fig3b_sign_matrices_target{tag}.png'}")


def fig_nonidentifiability(H, stats):
    """Figure 4: the headline. (a) sign-vector heatmap, (b) score distribution vs chance."""
    cmap = ListedColormap(["#c0392b", "#ecf0f1", "#2471a3"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig = plt.figure(figsize=(16.5, 10.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.30, 1.0],
                          height_ratios=[10, 8, 8], hspace=0.55, wspace=0.30)
    SHORT = {"D5": "D5 / turing_labyrinth s0000",
             "c2P_A": "c2_P target A / qvar s0001",
             "c2P_B": "c2_P target B / qvar s0004"}

    # ---- (a) heatmaps, one block per target -------------------------------
    for row, key in enumerate(("D5", "c2P_A", "c2P_B")):
        grp = H[key]
        ax = fig.add_subplot(gs[row, 0])
        S_true = sign_structure_rtol(np.array(grp["truth"]["J_true"], float),
                                     RTOL_STRUCTURAL).astype(int).ravel()
        rows = [S_true] + [r["sign_vector_rtol0.05"] for r in grp["runs"]]
        ylab = ["TRUE"] + [f"seed {r['seed']}" for r in grp["runs"]]
        M = np.array(rows)
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, {-1: "−", 0: "0", 1: "+"}[int(M[i, j])],
                        ha="center", va="center", fontsize=8.5, color="#111111")
        ax.axhline(0.5, color="#111111", lw=2.2)
        ax.set_yticks(range(len(ylab)))
        ax.set_yticklabels(ylab, fontsize=8.5)
        ax.set_xticks(range(9))
        ax.set_xticklabels([f"$J_{{{i+1}{j+1}}}$" for i in range(3) for j in range(3)],
                           fontsize=8.5)
        n = len(grp["runs"])
        d = stats[key]["n_distinct_sign_structures_rtol0.05"]
        d9 = stats[key]["n_distinct_sign_structures_rtol1e-9"]
        extra = "" if d == d9 else f"\n({d9}/{n} at the looser 1e-9 tolerance)"
        ax.set_title(f"{SHORT[key]} — {d}/{n} distinct sign structures{extra}",
                     fontsize=10.5, fontweight="bold")
        if row == 0:
            ax.legend(handles=[Patch(facecolor="#2471a3", label="+ activating"),
                               Patch(facecolor="#ecf0f1", edgecolor="#999", label="0"),
                               Patch(facecolor="#c0392b", label="− repressing")],
                      loc="lower left", bbox_to_anchor=(0.0, 1.30), ncol=3,
                      frameon=False, fontsize=9)

    # ---- (b) score distribution vs chance ---------------------------------
    axb = fig.add_subplot(gs[:2, 1])
    colors = {"D5": "#8e44ad", "c2P_A": "#16a085", "c2P_B": "#d35400"}
    offs = {"D5": -0.22, "c2P_A": 0.0, "c2P_B": 0.22}
    rng = np.random.default_rng(RNG_SEED)
    for i, key in enumerate(("D5", "c2P_A", "c2P_B")):
        vals = np.array([r["rerun_pinned"]["sign_match_frac_identity"]
                         for r in H[key]["runs"]])
        x = np.full(vals.size, i, float) + rng.uniform(-0.13, 0.13, vals.size)
        axb.scatter(x, vals, s=68, color=colors[key], alpha=0.85, zorder=3,
                    edgecolor="white", linewidth=1.1)
        axb.hlines(np.median(vals), i - 0.3, i + 0.3, color=colors[key], lw=3.2, zorder=4)
        axb.text(i + 0.33, np.median(vals), f"median\n{np.median(vals):.3f}",
                 fontsize=9, va="center", color=colors[key], fontweight="bold")
    axb.axhline(1 / 3, color="#111111", ls="--", lw=2.0, zorder=2,
                label="uniform chance over $\\{-,0,+\\}$ = 0.333")
    axb.axhline(stats["baselines"]["best_constant_guess"], color="#c0392b", ls=":", lw=2.0,
                zorder=2,
                label=f"best constant guess (all −) = "
                      f"{stats['baselines']['best_constant_guess']:.3f}")
    axb.axhline(stats["baselines"]["shuffled_null_mean"], color="#7f8c8d", ls="-.", lw=1.8,
                zorder=2,
                label=f"label-shuffled null = {stats['baselines']['shuffled_null_mean']:.3f}")
    axb.set_xticks(range(3))
    axb.set_xticklabels([f"D5\n(n={len(H['D5']['runs'])})",
                         f"c2_P A\n(n={len(H['c2P_A']['runs'])})",
                         f"c2_P B\n(n={len(H['c2P_B']['runs'])})"], fontsize=10)
    axb.set_ylabel("sign_match_frac  (identity = the only admissible labelling)", fontsize=10)
    axb.set_ylim(0, 1.0)
    axb.set_xlim(-0.55, 2.75)
    axb.grid(axis="y", alpha=0.3)
    axb.legend(fontsize=8.8, loc="upper left", framealpha=0.95)
    axb.set_title("Sign recovery sits at chance", fontsize=12, fontweight="bold")

    # ---- (c) the contrast: pattern reproduction ---------------------------
    axc = fig.add_subplot(gs[2, 1])
    keys = ["D5", "c2P_A", "c2P_B"]
    # Patterning is conditioned on the run having converged to a Turing-unstable solution.
    # The unconditional D5 rate (4/10) mixes in a CONVERGENCE failure — 6 D5 seeds never
    # reached a Turing solution at all — which is a different claim from "a Turing solution
    # fails to pattern". Both denominators are in topology_stats.json.
    pat = [stats[k]["n_patterned_given_turing"] / stats[k]["n_turing_denominator"]
           for k in keys]
    sgn = [stats[k]["median_sign_match_frac"] for k in keys]
    xx = np.arange(3)
    axc.bar(xx - 0.19, pat, 0.36, color="#27ae60",
            label="patterned | Turing-unstable")
    axc.bar(xx + 0.19, sgn, 0.36, color="#95a5a6", label="median sign_match_frac")
    axc.axhline(1 / 3, color="#111111", ls="--", lw=1.6)
    for x, v, k in zip(xx - 0.19, pat, keys):
        axc.text(x, v + 0.02, f"{stats[k]['n_patterned_given_turing']}/"
                              f"{stats[k]['n_turing_denominator']}",
                 ha="center", fontsize=9, fontweight="bold")
    for x, v in zip(xx + 0.19, sgn):
        axc.text(x, v + 0.07, f"{v:.3f}", ha="center", fontsize=9)
    axc.set_xticks(xx)
    axc.set_xticklabels(["D5", "c2_P A", "c2_P B"], fontsize=10)
    axc.set_ylim(0, 1.45)
    axc.legend(fontsize=8.8, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    axc.set_title("Phenotype is reproduced; topology is not identified",
                  fontsize=11.5, fontweight="bold", pad=8)

    fig.suptitle("Non-identifiability: every seed learns a different network, "
                 "and they all reproduce the pattern",
                 fontsize=16, fontweight="bold", y=0.995)
    fig.savefig(FIGS / "fig4_nonidentifiability.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  wrote {FIGS / 'fig4_nonidentifiability.png'}")


# ---------------------------------------------------------------------------
def compute_stats(H):
    rng = np.random.default_rng(RNG_SEED)
    stats = {}
    all_scores, null_scores = [], []
    for key, grp in H.items():
        S_true = sign_structure(np.array(grp["truth"]["J_true"], float)).astype(int).ravel()
        S_true_struct = sign_structure_rtol(
            np.array(grp["truth"]["J_true"], float), RTOL_STRUCTURAL).astype(int).ravel()
        vecs = [tuple(r["sign_vector_rtol1e-9"]) for r in grp["runs"]]
        vecs005 = [tuple(r["sign_vector_rtol0.05"]) for r in grp["runs"]]
        scores = [r["rerun_pinned"]["sign_match_frac_identity"] for r in grp["runs"]]
        struct = [r["sign_match_frac_structural"] for r in grp["runs"]]
        free = [r["rerun_free6"]["sign_match_frac_aligned"] for r in grp["runs"]]
        all_scores += scores
        for v in vecs:
            draws = np.array([rng.permutation(v) for _ in range(N_NULL_DRAWS // len(vecs))])
            null_scores.append((draws == S_true).mean())
        pat = [bool(r["rollout_patterned"]) for r in grp["runs"]]
        tur = [bool(r["recovered_turing"]) for r in grp["runs"]]
        pat_given_tur = [bool(r["rollout_patterned"]) for r in grp["runs"]
                         if r["recovered_turing"]]
        stats[key] = {
            "label": grp["label"],
            "n_runs": len(vecs),
            "n_distinct_sign_structures": len(set(vecs005)),          # structural: primary
            "n_distinct_sign_structures_rtol0.05": len(set(vecs005)),
            "n_distinct_sign_structures_rtol1e-9": len(set(vecs)),
            "true_sign_vector": S_true.tolist(),
            "true_sign_vector_rtol0.05": S_true_struct.tolist(),
            "sign_match_frac_identity": scores,
            "median_sign_match_frac": float(np.median(scores)),
            "min_sign_match_frac": float(np.min(scores)),
            "max_sign_match_frac": float(np.max(scores)),
            "sign_match_frac_structural": struct,
            "median_sign_match_frac_structural": float(np.median(struct)),
            "sign_match_frac_free6": free,
            "median_sign_match_frac_free6": float(np.median(free)),
            "n_turing": int(sum(tur)),
            "n_patterned": int(sum(pat)),
            "n_patterned_given_turing": int(sum(pat_given_tur)),
            "n_turing_denominator": len(pat_given_tur),
            "kstar_rel_err_median": float(np.median(
                [r["kstar_rel_err"] for r in grp["runs"] if r["kstar_rel_err"] is not None])),
        }

    # baselines, computed against c2_P target A's truth (identical to D5's; see README)
    S_true = np.array(stats["c2P_A"]["true_sign_vector"])
    counts = {s: int((S_true == s).sum()) for s in (-1, 0, 1)}
    stats["baselines"] = {
        "uniform_chance": 1 / 3,
        "best_constant_guess": max(counts.values()) / 9,
        "best_constant_guess_symbol": max(counts, key=counts.get),
        "true_sign_counts": counts,
        "shuffled_null_mean": float(np.mean(null_scores)),
        "shuffled_null_n_draws": N_NULL_DRAWS,
        "rng_seed": RNG_SEED,
    }
    stats["pooled"] = {
        "n_runs": len(all_scores),
        "median_sign_match_frac": float(np.median(all_scores)),
        "mean_sign_match_frac": float(np.mean(all_scores)),
    }
    return stats


def main():
    print("harvesting runs + re-running the permutation search ...")
    H = harvest()
    stats = compute_stats(H)

    (DATA / "topology_runs.json").write_text(json.dumps(H, indent=2))
    (DATA / "topology_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  wrote {DATA/'topology_runs.json'} and {DATA/'topology_stats.json'}")

    print("rendering topology grids ...")
    fig_topology_grids(H)
    print("rendering learned-vs-true panels ...")
    fig_learned_vs_true(H)
    print("rendering non-identifiability figure ...")
    fig_nonidentifiability(H, stats)

    print("\n--- headline numbers ---")
    for k in ("D5", "c2P_A", "c2P_B"):
        s = stats[k]
        print(f"{k:7s} distinct {s['n_distinct_sign_structures_rtol0.05']}/{s['n_runs']} "
              f"(struct) {s['n_distinct_sign_structures_rtol1e-9']}/{s['n_runs']} (1e-9) | "
              f"median smf={s['median_sign_match_frac']:.4f} "
              f"(struct {s['median_sign_match_frac_structural']:.4f}) | "
              f"turing {s['n_turing']}/{s['n_runs']} | "
              f"patterned|turing {s['n_patterned_given_turing']}/{s['n_turing_denominator']}")
    print(f"baselines: {stats['baselines']}")
    print(f"pooled: {stats['pooled']}")


if __name__ == "__main__":
    main()
