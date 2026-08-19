#!/usr/bin/env python
"""Post-run analysis for the real-stripes experiment (dataset stripes_colony_2ch).

For every finished per-sample run under a runs-root this script: reloads the trained
model in PHYSICAL units (eval.lgen_eval.physical_model_from_checkpoint — the nondim
trap, D-EVID-14), renders the learned topology, compares signed adjacencies and
parameters across runs, computes the QSS dispersion relation sigma(k), measures the
dynamical lift (mu_critical + a finite-mu simulate_lifted rollout next to the REAL
observed frame), and writes analysis_summary.json plus a stdout table.

The dataset's k_star attr is an FFT measurement of the observed frame, not ground truth
(real data has no answer key); every figure labels it "k* (FFT anchor)".

Usage: .venv/bin/python experiments/real_stripes/analyze.py [--runs-root ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(1)  # never compete with a live training batch for CPU

from rngrn.eval import lifted
from rngrn.eval.lgen_eval import physical_model_from_checkpoint
from rngrn.eval.topology import plot_topology
from rngrn.io import load_checkpoint
from rngrn.losses.terms import steady_state

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PAYLOAD = os.path.join(REPO, "data", "datasets", "stripes_colony_2ch", "payload.h5")
# Node labels: observed channels C1 (index 0), C2 (index 1); latent node H (index 2).
LABELS = ["$C_1$", "$C_2$", "$H$"]
SIGN_CHAR = {1: "+", -1: "−", 0: "·"}
# Same edge threshold convention as eval/topology.py::plot_topology (threshold_frac=0.02
# of the run's max binding strength over KA and KR together).
THRESHOLD_FRAC = 0.02
LIFTED_N = 64          # modest grid: analysis runs beside a live training batch
LIFTED_MAX_STEPS = 20000  # bounded cost; a truncated run reports stopped_reason


# --------------------------------------------------------------------------------------
# run discovery
# --------------------------------------------------------------------------------------
def discover_runs(runs_root: str) -> dict[str, str]:
    """sample_key -> run dir, from target_reports.jsonl (last row per sample wins),
    falling back to globbing runs/ and reading sample_key off frozen_config.yaml."""
    found: dict[str, str] = {}
    idx = os.path.join(runs_root, "target_reports.jsonl")
    if os.path.exists(idx):
        with open(idx) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                run_ids = json.loads(row["run_ids"])
                if not run_ids:
                    continue
                rid = next(iter(run_ids.values()))  # 1 seed per sample in this batch
                found[row["sample_key"]] = os.path.join(runs_root, "runs", rid)
    else:
        runs_dir = os.path.join(runs_root, "runs")
        for rid in sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []:
            rdir = os.path.join(runs_dir, rid)
            cfg = os.path.join(rdir, "config", "frozen_config.yaml")
            if not os.path.exists(cfg):
                continue
            with open(cfg) as fh:
                keys = [ln.split(":", 1)[1].strip() for ln in fh
                        if ln.strip().startswith("sample_key:")]
            if keys:
                found[keys[0]] = rdir
    for sample, rdir in found.items():
        ckpt = os.path.join(rdir, "checkpoints", "model.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"{sample}: no checkpoint at {ckpt}")
    # Run dirs are named <run_name>_<UTC-second>_seedN, so same-second concurrent starts
    # of the same run_name COLLIDE: every process writes the one dir and the last
    # finisher's checkpoint silently survives (bit frames 3/4/5 on 2026-08-19, 19:36:54).
    # When several samples map to one dir, keep only the sample whose dataset FFT anchor
    # matches the checkpoint's kstar_obs; drop the clobbered ones loudly.
    by_dir: dict[str, list[str]] = {}
    for sample, rdir in found.items():
        by_dir.setdefault(rdir, []).append(sample)
    for rdir, samples in by_dir.items():
        if len(samples) == 1:
            continue
        import torch as _torch
        k_ckpt = float(_torch.load(os.path.join(rdir, "checkpoints", "model.pt"),
                                   weights_only=False)["kstar_obs"])
        for sample in samples:
            _, k_anchor = observed_frame(sample)
            if abs(k_anchor - k_ckpt) / k_ckpt < 1e-9:
                print(f"[LOUD] {os.path.basename(rdir)} claimed by {len(samples)} samples "
                      f"(run-dir collision) — checkpoint belongs to {sample}")
            else:
                print(f"[LOUD] DROPPING {sample}: its checkpoint was clobbered in the "
                      f"run-dir collision on {os.path.basename(rdir)} (anchor "
                      f"{k_anchor:.4f} != checkpoint kstar_obs {k_ckpt:.4f})")
                del found[sample]
    if not found:
        raise FileNotFoundError(f"no analyzable runs under {runs_root}")
    return dict(sorted(found.items()))


def load_physical(run_dir: str):
    """(physical-units model, L_train, kstar_obs) for one run directory."""
    model, payload = load_checkpoint(run_dir)
    model_phys, L_train = physical_model_from_checkpoint(model, payload)
    if "kstar_obs" not in payload:
        raise KeyError(f"{run_dir}: checkpoint carries no kstar_obs extra")
    return model_phys, L_train, float(payload["kstar_obs"])


def observed_frame(sample_key: str) -> tuple[np.ndarray, float]:
    """(channel-0 observed frame, dataset k_star_fft attr) for one sample."""
    with h5py.File(PAYLOAD, "r") as fh:
        grp = fh[sample_key]
        return np.asarray(grp["final_frame"][0], float), float(grp.attrs["k_star_fft"])


# --------------------------------------------------------------------------------------
# topology
# --------------------------------------------------------------------------------------
def signed_adjacency(model) -> np.ndarray:
    """(N, N) signed adjacency: entry [i, j] is the regulation of target i by regulator j.

    Same convention as plot_topology's show_both=False branch (topology.py:175-177, and
    :205 for the diagonal): the DOMINANT of KA[i,j] vs KR[i,j] wins, drawn only above
    thr = THRESHOLD_FRAC * max binding strength of this run. +1 activation, -1
    repression, 0 below threshold."""
    KA = model.KA.detach().cpu().numpy()
    KR = model.KR.detach().cpu().numpy()
    allmag = np.concatenate([KA.ravel(), KR.ravel()])
    allmag = allmag[allmag > 0]
    thr = THRESHOLD_FRAC * (allmag.max() if allmag.size else 1.0)
    out = np.zeros(KA.shape, int)
    out[(KA >= KR) & (KA > thr)] = 1
    out[(KR > KA) & (KR > thr)] = -1
    return out


def fig_topology_per_run(runs: dict[str, dict], fig_dir: str) -> None:
    for sample, r in runs.items():
        fig = plot_topology(r["model"], title=f"Learned GRN — {sample}", labels=LABELS)
        fig.savefig(os.path.join(fig_dir, f"{sample}_topology.png"), dpi=200,
                    bbox_inches="tight")
        plt.close(fig)


def _annotate_adjacency(ax, A: np.ndarray, extra: np.ndarray | None = None) -> None:
    ax.imshow(A, cmap="coolwarm", vmin=-1, vmax=1)
    names = [s[1:-1].replace("_", "") for s in LABELS]
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    for i in range(3):
        for j in range(3):
            txt = SIGN_CHAR[int(A[i, j])]
            if extra is not None:
                txt += f"\n{extra[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11)


def fig_topology_grid(runs: dict[str, dict], fig_dir: str) -> None:
    n = len(runs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, (sample, r) in zip(axes.ravel(), runs.items()):
        ax.set_visible(True)
        _annotate_adjacency(ax, r["adjacency"])
        ax.set_title(sample, fontsize=10)
        ax.set_xlabel("regulator")
        ax.set_ylabel("target")
    fig.suptitle("Signed adjacency per run (dominant KA/KR above "
                 f"{THRESHOLD_FRAC:g}·max)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "topology_grid.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def fig_topology_consensus(runs: dict[str, dict], fig_dir: str) -> np.ndarray:
    """Modal-sign consensus over runs. Returns the modal sign matrix."""
    stack = np.stack([r["adjacency"] for r in runs.values()])       # (R, N, N)
    modal = np.zeros((3, 3), int)
    frac = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            vals, counts = np.unique(stack[:, i, j], return_counts=True)
            k = int(np.argmax(counts))
            modal[i, j] = int(vals[k])
            frac[i, j] = counts[k] / stack.shape[0]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    im = ax.imshow(frac, cmap="viridis", vmin=0, vmax=1)
    names = [s[1:-1].replace("_", "") for s in LABELS]
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("regulator")
    ax.set_ylabel("target")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{SIGN_CHAR[int(modal[i, j])]}\n{frac[i, j]:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if frac[i, j] < 0.6 else "black")
    ax.set_title(f"Edge-sign consensus over {stack.shape[0]} run(s)\n"
                 "(modal sign; fraction of runs agreeing)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="agreement fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "topology_consensus.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    return modal


# --------------------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------------------
def _param_arrays(model) -> dict[str, np.ndarray]:
    g = lambda t: t.detach().cpu().numpy()
    return dict(alpha=g(model.alpha), beta=g(model.beta), delta=g(model.delta),
                D=g(model.D), KA=g(model.KA), KR=g(model.KR))


def fig_params(runs: dict[str, dict], fig_dir: str) -> None:
    names = [s[1:-1].replace("_", "") for s in LABELS]
    edge = [f"{names[j]}→{names[i]}" for i in range(3) for j in range(3)]
    per_species = list(names)
    layout = [("alpha", edge), ("KA", edge), ("KR", edge),
              ("beta", per_species), ("delta", per_species), ("D", per_species)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    rng = np.random.default_rng(0)  # jitter only, cosmetic
    for ax, (fam, ticks) in zip(axes.ravel(), layout):
        vals = np.stack([r["params"][fam].ravel() for r in runs.values()])  # (R, P)
        pos = np.arange(vals.shape[1])
        for prow in range(vals.shape[0]):
            ax.scatter(pos + rng.uniform(-0.15, 0.15, len(pos)), vals[prow],
                       s=14, alpha=0.6)
        ax.scatter(pos, np.median(vals, axis=0), marker="_", s=300, color="black",
                   label="median")
        ax.set_yscale("log")
        ax.set_xticks(pos, ticks, rotation=45 if len(ticks) > 5 else 0, fontsize=8)
        ax.set_title(fam)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Recovered parameters across {vals.shape[0]} run(s) "
                 "(points = runs, bar = median)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "params_families.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# dispersion
# --------------------------------------------------------------------------------------
def dispersion_of(r: dict) -> None:
    """Attach sigma(k) on lifted.KGRID, k*_model and the Turing verdict to a run record.

    k* convention matches lifted.turing_verdict_lifted: argmax over k>0, k=0 excluded.
    """
    sig, _ = lifted.qss_dispersion(r["model"], r["xstar"], lifted.KGRID)
    i = 1 + int(np.argmax(sig[1:]))
    r.update(sigma=sig, kstar_model=float(lifted.KGRID[i]),
             sigma_max=float(sig[i]), turing=bool(sig[i] > 0.0),
             stable_uniform=bool(sig[0] < 0.0))


def fig_dispersion_per_run(runs: dict[str, dict], fig_dir: str) -> None:
    for sample, r in runs.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lifted.KGRID[1:], r["sigma"][1:], color="tab:blue")
        ax.axhline(0.0, color="black", lw=0.8)
        ax.axvline(r["kstar_model"], color="tab:blue", ls="--", lw=1,
                   label=f"k*_model = {r['kstar_model']:.3g}")
        ax.axvline(r["kstar_fft"], color="tab:red", ls=":", lw=1.2,
                   label=f"k* (FFT anchor) = {r['kstar_fft']:.3g}")
        ax.set_xscale("log")
        top = max(abs(r["sigma_max"]), 0.1)
        ax.set_ylim(-4 * top, 1.5 * top)
        ax.set_xlabel("k [rad/length]")
        ax.set_ylabel("$\\sigma(k)$")
        ax.set_title(f"QSS dispersion — {sample}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{sample}_dispersion.png"), dpi=200,
                    bbox_inches="tight")
        plt.close(fig)


def fig_dispersion_overlay(runs: dict[str, dict], fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    top = max(max(abs(r["sigma_max"]) for r in runs.values()), 0.1)
    for sample, r in runs.items():
        ax.plot(lifted.KGRID[1:], r["sigma"][1:], lw=1.4, label=sample)
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xscale("log")
    ax.set_ylim(-4 * top, 1.5 * top)
    ax.set_xlabel("k [rad/length]")
    ax.set_ylabel("$\\sigma(k)$")
    ax.set_title("QSS dispersion relations, all runs (k* anchors are FFT-measured, "
                 "not truth)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dispersion_overlay.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# dynamical lift
# --------------------------------------------------------------------------------------
def lift_of(r: dict, sample: str) -> None:
    """mu_critical + one simulate_lifted rollout, attached to the run record.

    mu choice: lifted.MU_BIO_CENTRAL (7.2e-4, the module's cited central biological
    gate timescale) is the default — small enough to sit in the QSS-faithful regime
    (mu -> 0 is exactly QSS, lifted.py:27-30). If mu_critical brackets an upper edge
    BELOW 10x that value, mu is lowered to mu_crit/10 so the sim stays a decade inside
    the Turing interval rather than straddling its edge.
    """
    try:
        mc = lifted.mu_critical(r["model"], r["xstar"])
    except Exception as exc:  # fail LOUDLY, keep the other runs
        print(f"[LOUD] {sample}: mu_critical FAILED: {type(exc).__name__}: {exc}")
        mc = dict(status=f"error: {exc}", mu_crit=float("nan"))
    mu_crit = float(mc.get("mu_crit", float("nan")))
    r["mu_crit"] = mu_crit
    r["mu_crit_status"] = str(mc.get("status"))
    if r["mu_crit_status"] != "turing_at_hi" and not np.isfinite(mu_crit):
        print(f"[LOUD] {sample}: mu_critical status={r['mu_crit_status']} "
              f"(no finite mu_crit — model is not strictly Turing at mu->0)")
    mu = lifted.MU_BIO_CENTRAL
    if np.isfinite(mu_crit) and mu_crit / 10.0 < mu:
        mu = mu_crit / 10.0
        print(f"[LOUD] {sample}: mu_crit={mu_crit:.3g} is small; "
              f"lowering sim mu to mu_crit/10 = {mu:.3g}")
    sim = lifted.simulate_lifted(r["model"], L=r["L_train"], mu=mu, n=LIFTED_N,
                                 seed=0, max_steps=LIFTED_MAX_STEPS,
                                 xstar=r["xstar"])
    if sim["blew_up"]:
        print(f"[LOUD] {sample}: simulate_lifted BLEW UP at mu={mu:.3g} "
              f"(step {sim['nsteps_run']}/{sim['nsteps']})")
    elif sim["stopped_reason"] == "step_budget":
        print(f"[LOUD] {sample}: simulate_lifted TRUNCATED at {LIFTED_MAX_STEPS} steps "
              "— the field is not a statement about the attractor")
    r.update(mu_sim=float(mu), lifted_patterned=bool(sim["patterned"]),
             lifted_kstar=float(sim["kstar"]), lifted_field=sim["fields"][0],
             lifted_stopped=str(sim["stopped_reason"]))


def fig_lifted_vs_observed(runs: dict[str, dict], fig_dir: str) -> None:
    for sample, r in runs.items():
        obs, _ = observed_frame(sample)
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
        for ax, field, name in ((axes[0], obs, f"observed C1 ({sample})"),
                                (axes[1], r["lifted_field"],
                                 f"lifted sim C1 (mu={r['mu_sim']:.2g}, "
                                 f"{r['lifted_stopped']})")):
            im = ax.imshow(field, cmap="viridis")
            ax.set_title(name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{sample}_lifted_vs_observed.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------------------
def summary_rows(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for sample, r in runs.items():
        med = {k: float(np.median(v)) for k, v in r["params"].items()}
        rows.append(dict(
            sample_key=sample, run_dir=os.path.relpath(r["run_dir"], REPO),
            turing=r["turing"], stable_uniform=r["stable_uniform"],
            kstar_model=r["kstar_model"], sigma_max=r["sigma_max"],
            kstar_fft_anchor=r["kstar_fft"],
            mu_crit=r["mu_crit"], mu_crit_status=r["mu_crit_status"],
            mu_sim=r["mu_sim"], lifted_patterned=r["lifted_patterned"],
            lifted_kstar=r["lifted_kstar"], lifted_stopped=r["lifted_stopped"],
            xstar_converged=bool(r["xstar_converged"]),
            param_medians=med,
            signed_adjacency=[[int(v) for v in row] for row in r["adjacency"]],
        ))
    return rows


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'sample':<12} {'turing':<7} {'k*_model':>9} {'k*_fft':>8} "
           f"{'mu_crit':>9} {'patterned':>9} {'k*_lifted':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['sample_key']:<12} {str(r['turing']):<7} "
              f"{r['kstar_model']:>9.3g} {r['kstar_fft_anchor']:>8.3g} "
              f"{r['mu_crit']:>9.3g} {str(r['lifted_patterned']):>9} "
              f"{r['lifted_kstar']:>9.3g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default=HERE,
                    help="runs root holding target_reports.jsonl and runs/ "
                         "(default: experiments/real_stripes; smoke/ is a separate "
                         "root and is never scanned from the default)")
    ap.add_argument("--fig-dir",
                    default=os.path.join(REPO, "experiments", "figures_report",
                                         "real_stripes"),
                    help="where PNGs land (tracked: experiments/figures_report/**)")
    ap.add_argument("--out-json", default=None,
                    help="summary path (default: <runs-root>/analysis_summary.json)")
    args = ap.parse_args()
    out_json = args.out_json or os.path.join(args.runs_root, "analysis_summary.json")
    os.makedirs(args.fig_dir, exist_ok=True)

    runs: dict[str, dict] = {}
    for sample, rdir in discover_runs(args.runs_root).items():
        model, L_train, kstar_obs = load_physical(rdir)
        _, kstar_fft = observed_frame(sample)
        if not np.isclose(kstar_obs, kstar_fft, rtol=1e-9):
            print(f"[LOUD] {sample}: checkpoint kstar_obs={kstar_obs!r} != dataset "
                  f"k_star_fft={kstar_fft!r} — using the checkpoint's (what training saw)")
        xs, converged = steady_state(model)
        if not converged:
            raise RuntimeError(
                f"{sample}: steady_state did NOT converge for the physical model — every "
                f"downstream quantity (dispersion, lift) would be built on a non-fixed "
                f"point. Refusing to continue.")
        runs[sample] = dict(run_dir=rdir, model=model, L_train=L_train,
                            kstar_fft=kstar_obs, xstar=xs.detach().cpu().numpy(),
                            xstar_converged=converged,
                            adjacency=signed_adjacency(model),
                            params=_param_arrays(model))
        dispersion_of(runs[sample])
        lift_of(runs[sample], sample)
        print(f"loaded {sample}: turing={runs[sample]['turing']} "
              f"k*_model={runs[sample]['kstar_model']:.3g} "
              f"lifted_patterned={runs[sample]['lifted_patterned']}")

    fig_topology_per_run(runs, args.fig_dir)
    fig_topology_grid(runs, args.fig_dir)
    fig_topology_consensus(runs, args.fig_dir)
    fig_params(runs, args.fig_dir)
    fig_dispersion_per_run(runs, args.fig_dir)
    fig_dispersion_overlay(runs, args.fig_dir)
    fig_lifted_vs_observed(runs, args.fig_dir)

    rows = summary_rows(runs)
    summary = dict(
        runs_root=os.path.relpath(args.runs_root, REPO), n_runs=len(rows),
        dataset_id="stripes_colony_2ch",
        kstar_anchor_note=("k*_fft_anchor is an FFT measurement of the observed frame, "
                           "NOT ground truth — real data has no answer key"),
        adjacency_convention=("[i][j] = regulation of target i by regulator j; dominant "
                              f"of KA/KR above {THRESHOLD_FRAC:g}*max binding strength "
                              "(eval/topology.py plot_topology convention)"),
        node_labels={"0": "C1 (observed)", "1": "C2 (observed)", "2": "H (latent)"},
        per_run=rows)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nwrote {out_json} and {len(os.listdir(args.fig_dir))} figure files "
          f"to {args.fig_dir}\n")
    print_table(rows)


if __name__ == "__main__":
    main()
