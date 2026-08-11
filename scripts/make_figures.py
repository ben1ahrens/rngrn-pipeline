#!/usr/bin/env python
"""make_figures.py — draw the paper-shaped figures from a run's persisted arrays.

    python scripts/make_figures.py --runs-root experiments/lgen_transfer \
        --run-id m3_registry_20260730_005701 --out-dir figures/lgen_transfer

FIGURES
  a_pattern.png       target pattern | recovered pattern, side by side, SHARED colour scale
  b_dispersion.png    sigma(k) recovered vs TRUE, with k* marked on both
  c_raps.png          radially averaged power spectrum, target vs model, log y, k* marked
  d_cross_L.png       the pattern at each domain size in a row, q and k* annotated
  e_training.png      total loss and every term vs step, with the staging ramp shaded
  f_parameters.png    parameter trajectories vs step (D per species and the D-ratio)
  g_hyperparams.png   outcome vs hyperparameter across the whole run index (sweep panel)

Each figure is drawn only if its arrays exist, and the script says which it skipped and why.
Figures are REGENERABLE and stay gitignored; the ARRAYS are the artefact.

FALLBACK FOR PRE-ARRAY RUNS. A run recorded before `solver.save_plot_arrays` existed has no
npz. `--rebuild-arrays` reconstructs one from the tracked checkpoint plus the dataset — the
same computation `train.fit` does after recovery (rollout on the target's own grid with the
config's own knobs, sigma(k) from the checkpoint's J/D and the answer key's), with
`meta.rebuilt_from_checkpoint = true` so a reconstructed npz can never be mistaken for the
one its run wrote. Without the flag the script rebuilds in memory and writes nothing.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))

from rngrn import plotdata as PD                                    # noqa: E402
from rngrn.config import load_config                                # noqa: E402
from rngrn.index import open_index                                  # noqa: E402

TWO_PI = 2.0 * np.pi


# ======================================================================================
# loading, with the pre-array fallback
# ======================================================================================
def _rebuild_arrays(run_dir: str, write: bool) -> dict:
    """Reconstruct a run's plot arrays from its checkpoint + its dataset.

    Runs the SAME post-recovery computation `train.fit` does, driven by the run's own frozen
    config, so the reconstruction is not a different measurement dressed as the original:
    the rollout uses the target frame's grid and L and the config's rollout knobs, and the
    morphology vectors are recomputed from the fields actually simulated here.

    There is NO training history in a rebuild — the trajectory only exists while recovery
    runs, and it cannot be recovered from a checkpoint. Figures (e) and (f) are therefore
    skipped for a rebuilt run, and say so.
    """
    from rngrn import io as IO
    from rngrn.eval.lgen_eval import physical_model_from_checkpoint
    from rngrn.losses.terms import steady_state
    from rngrn.scoring.morphology import morphology_vector
    from rngrn.train import _morphology_rollout, _resolve_recovery_input
    import torch

    cfg = load_config(os.path.join(run_dir, "config", "frozen_config.yaml"))
    ri, answer_key = _resolve_recovery_input(cfg)
    model, payload = IO.load_checkpoint(run_dir)
    model, L_train = physical_model_from_checkpoint(model, payload)
    xs, _ = steady_state(model)
    xstar = xs.detach().cpu().numpy()
    J = model.jacobian(torch.as_tensor(xstar), create_graph=False).detach().cpu().numpy()
    D = model.D.detach().cpu().numpy()
    kstar_obs = float(payload["kstar_obs"])

    result = SimpleNamespace(model=model, xstar=xstar)
    model_field, rollout_row = _morphology_rollout(cfg, result, ri)
    morph_tgt = morphology_vector(np.asarray(ri.frame, float)[0])
    morph_mdl = None if model_field is None else morphology_vector(
        np.asarray(model_field, float)[0])

    arrays = PD.build_plot_arrays(
        target_frame=ri.frame, L=ri.L, J_recovered=J, D_recovered=D, kstar_obs=kstar_obs,
        morph_vector_target=morph_tgt, model_field=model_field,
        rollout_status=rollout_row.get("rollout_status"), morph_vector_model=morph_mdl,
        J_true=getattr(answer_key, "J", None), D_true=getattr(answer_key, "D", None),
        meta=dict(run_id=os.path.basename(run_dir), rebuilt_from_checkpoint=True,
                  dataset_id=cfg.data.dataset_id, sample_key=cfg.data.sample_key,
                  form=cfg.model.form, N=cfg.model.N, m=cfg.model.m,
                  seed=int(cfg.train.seed), L_train_checkpoint=L_train,
                  kstar_obs=kstar_obs,
                  kstar_true=getattr(answer_key, "kstar", None),
                  kstar_fft_true=getattr(answer_key, "kstar_fft", None)))
    if write:
        n = PD.save_plot_arrays(PD.plot_arrays_path(run_dir), arrays)
        print(f"  wrote rebuilt {PD.plot_arrays_path(run_dir)} ({n} bytes)")
    return PD.load_plot_arrays(PD.plot_arrays_path(run_dir)) if write else _decode(arrays)


def _decode(arrays: dict) -> dict:
    """In-memory arrays in the same shape `load_plot_arrays` returns (meta as a dict)."""
    out = dict(arrays)
    out["meta"] = json.loads(str(out["meta"].item()))
    return out


def load_run(runs_root: str, run_id: str, rebuild: str) -> dict:
    run_dir = os.path.join(runs_root, "runs", run_id)
    path = PD.plot_arrays_path(run_dir)
    if os.path.exists(path) and rebuild != "force":
        print(f"  arrays: {path}")
        return PD.load_plot_arrays(path)
    if rebuild == "no":
        raise FileNotFoundError(
            f"{path} does not exist. This run predates solver.save_plot_arrays. Pass "
            f"--rebuild-arrays to reconstruct it from the checkpoint + dataset, or re-run "
            f"the recovery.")
    print(f"  arrays: MISSING -> rebuilding from checkpoint + dataset ({run_dir})")
    return _rebuild_arrays(run_dir, write=(rebuild == "write"))


# ======================================================================================
# figures
# ======================================================================================
def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path} ({os.path.getsize(path)} bytes)")
    return path


def fig_pattern(z, out_dir):
    """(a) target | recovered on a SHARED colour scale, plus the recovered field on its own.

    The shared scale is the honest comparison and is the primary one: independently
    normalised panels make any two fields look alike, which is exactly the comparison this
    figure exists to make. Measured on m3_registry_20260730_005701 the recovered model
    patterns at a far SMALLER amplitude than the target, and on the shared scale it reads as
    nearly flat — which is the result. A third panel therefore shows the same recovered field
    on its own scale, labelled as such, so its MORPHOLOGY is still readable; dropping the
    shared scale to make the picture prettier would hide the amplitude gap.
    """
    tgt = np.asarray(z["target_frame"], float)[0]
    mdl = None if "model_field" not in z else np.asarray(z["model_field"], float)[0]
    meta = z["meta"]
    n = 1 if mdl is None else 3
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.4), squeeze=False)
    if mdl is None:
        vmin, vmax = float(tgt.min()), float(tgt.max())
    else:
        vmin = min(float(tgt.min()), float(mdl.min()))
        vmax = max(float(tgt.max()), float(mdl.max()))
    im = axes[0][0].imshow(tgt, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[0][0].set_title(f"target (channel 0)\n$k^*$={meta['kstar_raps_target']:.4f}",
                         fontsize=9)
    if mdl is not None:
        axes[0][1].imshow(mdl, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
        axes[0][1].set_title(f"recovered rollout, SHARED scale\n"
                             f"$k^*$={meta['kstar_raps_model']:.4f}", fontsize=9)
        im2 = axes[0][2].imshow(mdl, cmap="viridis", origin="lower")
        axes[0][2].set_title(f"recovered rollout, OWN scale\n"
                             f"amplitude {0.5 * (mdl.max() - mdl.min()):.3g} vs target "
                             f"{0.5 * (tgt.max() - tgt.min()):.3g}", fontsize=9)
        fig.colorbar(im2, ax=axes[0][2], shrink=0.75)
    else:
        axes[0][0].set_xlabel(f"no model field: rollout_status={meta['rollout_status']!r}")
    for ax in axes[0]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[0][:2].tolist() if mdl is not None else axes[0].tolist(),
                 shrink=0.75, label="concentration (shared scale)")
    fig.suptitle(f"{meta.get('run_id')}  {meta.get('dataset_id')}/{meta.get('sample_key')}"
                 f"  L={meta['L']:.2f}", fontsize=9)
    return _save(fig, out_dir, "a_pattern.png")


def fig_dispersion(z, out_dir):
    """(b) THE canonical Turing figure: sigma(k) recovered against true, k* marked on both."""
    k = np.asarray(z["dispersion_k"], float)
    sig = np.asarray(z["dispersion_sigma"], float)
    meta = z["meta"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
    ax.plot(k, sig, color="C0", lw=1.8, label="recovered  $\\sigma(k)$")
    kr = meta["kstar_dispersion_recovered"]
    ax.axvline(kr, color="C0", ls="--", lw=1.0,
               label=f"recovered $k^*$={kr:.4f}")
    lo, hi = float(sig.min()), float(sig.max())
    if "dispersion_sigma_true" in z:
        st = np.asarray(z["dispersion_sigma_true"], float)
        ax.plot(k, st, color="C3", lw=1.5, label="answer key  $\\sigma(k)$")
        kt = meta["kstar_dispersion_true"]
        ax.axvline(kt, color="C3", ls="--", lw=1.0, label=f"true $k^*$={kt:.4f}")
        lo, hi = min(lo, float(st.min())), max(hi, float(st.max()))
    # Zoom on the band that decides the verdict: the region around and above sigma = 0. The
    # far tail dives to large negative values on a strongly-diffusive true model and would
    # otherwise squash both peaks — and the peaks are the whole comparison.
    span = hi - min(lo, 0.0)
    ax.set_ylim(min(lo, 0.0) if span < 4 * (hi + 1e-9) else -2.0 * max(hi, 1e-3), hi * 1.35)
    ax.set_xlabel("$k$  (rad / length)")
    ax.set_ylabel("$\\sigma(k) = \\max_i \\mathrm{Re}\\,\\mathrm{eig}(J - k^2 D)$")
    ax.set_title(f"dispersion relation — {meta.get('run_id')}", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    return _save(fig, out_dir, "b_dispersion.png")


def fig_raps(z, out_dir):
    """(c) the RAPS, log y, k* marked — WHERE the headline k* number comes from."""
    k = np.asarray(z["raps_k"], float)
    pt = np.asarray(z["raps_target"], float)
    meta = z["meta"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    curves = [("target", pt, "C0", meta["kstar_raps_target"])]
    if "raps_model" in z:
        curves.append(("model rollout", np.asarray(z["raps_model"], float), "C1",
                       meta["kstar_raps_model"]))
    peak = max(float(np.max(p)) for _, p, _, _ in curves)
    for label, p, colour, ks in curves:
        # MASK the non-positive bins rather than flooring them: a floored zero bin plots as a
        # spike to 1e-30 and stretches the log axis over 30 decades, hiding the peak this
        # figure exists to show.
        ax.semilogy(k, np.where(p > 0, p, np.nan), color=colour, lw=1.6, label=label)
        ax.axvline(ks, color=colour, ls="--", lw=1.0, label=f"{label} $k^*$={ks:.4f}")
    kfft = meta.get("kstar_fft_true")
    if kfft is not None and np.isfinite(kfft):
        ax.axvline(kfft, color="C3", ls=":", lw=1.4,
                   label=f"answer key $k^*_{{fft}}$={kfft:.4f}")
    ax.set_xlabel("$k$  (rad / length)")
    ax.set_ylabel("radially averaged power")
    ax.set_ylim(peak * 1e-8, peak * 5)
    ax.set_xlim(0, min(k.max(), 6.0 * max(meta["kstar_raps_target"], 1e-6)))
    ax.set_title(f"radially averaged power spectrum — {meta.get('run_id')}", fontsize=10)
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "c_raps.png")


def fig_cross_L(run_dir, out_dir):
    """(d) the pattern at each domain size, q and k* annotated. The transfer figure."""
    path = PD.lgen_fields_path(run_dir)
    if not os.path.exists(path):
        print(f"  SKIP d_cross_L: no {path} — run `rngrn evaluate --eval-L ...` first")
        return None
    z = PD.load_plot_arrays(path)
    Ls = np.asarray(z["L_values"], float)
    ks = np.asarray(z["kstar_phys"], float)
    qs = np.asarray(z["periods_per_box"], float)
    meta = z["meta"]
    order = np.argsort(Ls)
    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.6), squeeze=False)
    for slot, i in enumerate(order):
        f = np.asarray(z[f"field_{int(i):02d}"], float)[0]
        ax = axes[0][slot]
        # Each panel gets its OWN scale: the amplitudes at different L are not the
        # comparison this figure makes (the WAVELENGTH is), and a shared scale would wash
        # out the small-amplitude boxes.
        ax.imshow(f, cmap="viridis", origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        cls = (meta.get("morph_class") or [None] * n)[int(i)]
        ref = (meta.get("is_L_train") or [False] * n)[int(i)]
        ax.set_title(f"L={Ls[i]:.2f}  ({Ls[i]/meta['L_train']:.2f}$\\times$)"
                     f"{'  [L_train]' if ref else ''}\n"
                     f"$k^*$={ks[i]:.4f}  q={qs[i]:.2f}\n{cls}", fontsize=8)
    fig.suptitle(f"cross-L transfer — {meta.get('run_id')}  "
                 f"(L_train={meta['L_train']:.2f}, grid_rule={meta['grid_rule']})",
                 fontsize=9)
    return _save(fig, out_dir, "d_cross_L.png")


def _hist(z):
    """(steps, names, block (S,B,Q+P)) or None when the run has no trajectory."""
    if "hist_step" not in z:
        return None
    names = [str(s) for s in z["hist_scalar_names"]] + [str(s) for s in z["hist_param_names"]]
    block = np.concatenate([np.asarray(z["hist_scalars"], float),
                            np.asarray(z["hist_params"], float)], axis=2)
    return np.asarray(z["hist_step"], float), names, block


def fig_training(z, out_dir):
    """(e) total loss and every term vs step, with the STAGING RAMP shaded.

    The shading is not decoration: DataFirstStaging holds `turing` at weight 0 for the first
    off_frac of the budget and ramps it over the next ramp_frac, so a jump in the total at
    25% of the budget is the schedule, not the optimiser.
    """
    h = _hist(z)
    if h is None:
        print("  SKIP e_training: this run has no training history "
              "(train.history_every=0, or a run recorded before unit P1)")
        return None
    steps, names, block = h
    meta = z["meta"]
    best = meta.get("history_best_member")
    b = 0 if best is None else int(best)
    fig, (ax, axw) = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.4, 1.0]))
    term_cols = [n for n in names if n.startswith("L_")]
    ax.semilogy(steps, np.maximum(block[:, b, names.index("total")], 1e-12),
                color="k", lw=2.0, label="total (weighted)")
    for i, n in enumerate(term_cols):
        ax.semilogy(steps, np.maximum(block[:, b, names.index(n)], 1e-12),
                    lw=1.2, color=f"C{i}", label=n[2:] + " (unweighted)")
    ax.set_ylabel("loss")
    ax.set_title(f"training trajectory — {meta.get('run_id')}  "
                 f"member {b} of {meta.get('history_n_members')} (the reported winner)",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    for n in [n for n in names if n.startswith("w_")]:
        v = block[:, b, names.index(n)]
        if np.allclose(np.nan_to_num(v), np.nan_to_num(v)[0]):
            axw.plot(steps, v, lw=1.0, ls=":", label=f"{n} (const)")
        else:
            axw.plot(steps, v, lw=1.6, label=n)
    axw.set_ylabel("weight in force")
    axw.set_xlabel("Adam step")
    axw.legend(fontsize=7, ncol=3)
    total = meta.get("history_total_steps")
    if total:
        # shade the staging window read off the trajectory itself: the span over which any
        # weight is still moving
        moving = [n for n in names if n.startswith("w_")
                  and not np.allclose(np.nan_to_num(block[:, b, names.index(n)]),
                                      np.nan_to_num(block[:, b, names.index(n)])[0])]
        if moving:
            v = block[:, b, names.index(moving[0])]
            fin = np.isfinite(v)
            ramp_end = steps[fin][np.argmax(v[fin] >= v[fin].max() - 1e-12)]
            for a in (ax, axw):
                a.axvspan(0, ramp_end, color="0.85", zorder=0)
            ax.text(ramp_end * 0.5, ax.get_ylim()[1], "staging window\n(held off, then ramped)",
                    ha="center", va="top", fontsize=7, color="0.35")
    return _save(fig, out_dir, "e_training.png")


def fig_parameters(z, out_dir):
    """(f) parameter learning: D per species, and the D-ratio — the open question.

    Whether recovery reaches biologically plausible diffusivities is the project's live
    question (docs/DECISIONS.md D1/D3), and it is a statement about the D trajectory, so D
    and the D-ratio get their own panels; every other physical parameter goes in a third.
    """
    h = _hist(z)
    if h is None:
        print("  SKIP f_parameters: this run has no training history")
        return None
    steps, names, block = h
    meta = z["meta"]
    best = meta.get("history_best_member")
    b = 0 if best is None else int(best)
    d_cols = [n for n in names if n.startswith("D[")]
    other = [n for n in names if n[0] in "KAabd" and "[" in n and not n.startswith("D[")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for i, n in enumerate(d_cols):
        axes[0].semilogy(steps, block[:, b, names.index(n)], lw=1.6, color=f"C{i}", label=n)
    axes[0].set_ylabel("$D_i$"); axes[0].set_title("diffusivities", fontsize=10)
    axes[0].legend(fontsize=8)
    if "d_ratio" in names:
        axes[1].semilogy(steps, block[:, b, names.index("d_ratio")], lw=2.0, color="C3")
    axes[1].set_title("D-ratio (two most mobile species)", fontsize=10)
    axes[1].axhline(7.5, color="0.5", ls="--", lw=1.0)
    axes[1].annotate("literature centre 7.5 (Nodal/Lefty)", (steps[0], 7.5), fontsize=7,
                     color="0.35", textcoords="offset points", xytext=(2, 4))
    for i, n in enumerate(other):
        axes[2].plot(steps, block[:, b, names.index(n)], lw=0.8, alpha=0.7)
    axes[2].set_title(f"the other {len(other)} physical parameters\n"
                      f"(KA, KR, alpha, delta, beta)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Adam step")
    fig.suptitle(f"parameter learning — {meta.get('run_id')} member {b}", fontsize=10)
    return _save(fig, out_dir, "f_parameters.png")


def fig_hyperparams(runs_root, out_dir, backend, x_key, y_key):
    """(g) outcome vs hyperparameter over the WHOLE run index — the sweep panel.

    One point per run row, so a sweep (which writes one row per cell) plots directly. Groups
    by `form` because pooling model forms would average two different experiments.
    """
    rows = open_index(runs_root, "runs", backend).read()
    pts = [(r.get(x_key), r.get(y_key), r.get("form")) for r in rows]
    pts = [(float(x), float(y), f) for x, y, f in pts
           if isinstance(x, (int, float)) and isinstance(y, (int, float))
           and np.isfinite(float(x)) and np.isfinite(float(y))]
    if not pts:
        print(f"  SKIP g_hyperparams: no run row carries both {x_key!r} and {y_key!r} "
              f"as finite numbers ({len(rows)} rows read)")
        return None
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    forms = sorted({f for _, _, f in pts}, key=str)
    for i, f in enumerate(forms):
        xs = [x for x, _, ff in pts if ff == f]
        ys = [y for _, y, ff in pts if ff == f]
        ax.plot(xs, ys, "o", ms=7, color=f"C{i}", alpha=0.8, label=f"form={f}")
    ax.set_xlabel(x_key); ax.set_ylabel(y_key)
    ax.set_title(f"{y_key} vs {x_key}\n{len(pts)} of {len(rows)} run rows carry both, in "
                 f"{os.path.basename(os.path.normpath(runs_root))}", fontsize=9)
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "g_hyperparams.png")


# ======================================================================================
def main(argv=None):
    p = argparse.ArgumentParser("make_figures")
    p.add_argument("--runs-root", default="experiments")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", default=None,
                   help="default: <runs-root>/runs/<run-id>/figures")
    p.add_argument("--index-backend", choices=["jsonl", "sqlite"], default="jsonl")
    p.add_argument("--rebuild-arrays", choices=["no", "memory", "write", "force"],
                   default="memory",
                   help="'memory' (default) reconstructs a missing npz in memory; 'write' "
                        "also saves it into the run dir (marked rebuilt_from_checkpoint); "
                        "'force' rebuilds even when an npz exists; 'no' raises instead.")
    p.add_argument("--hp-x", default="adam_steps", help="figure (g) x axis: a run-row column")
    p.add_argument("--hp-y", default="kstar_fft_rel_err", help="figure (g) y axis")
    args = p.parse_args(argv)

    run_dir = os.path.join(args.runs_root, "runs", args.run_id)
    out_dir = args.out_dir or os.path.join(run_dir, "figures")
    print(f"make_figures: {args.run_id}")
    z = load_run(args.runs_root, args.run_id, args.rebuild_arrays)
    made = [fig_pattern(z, out_dir), fig_dispersion(z, out_dir), fig_raps(z, out_dir),
            fig_cross_L(run_dir, out_dir), fig_training(z, out_dir),
            fig_parameters(z, out_dir),
            fig_hyperparams(args.runs_root, out_dir, args.index_backend,
                            args.hp_x, args.hp_y)]
    made = [m for m in made if m]
    print(f"make_figures: {len(made)}/7 figures written to {out_dir}")


if __name__ == "__main__":
    main()
