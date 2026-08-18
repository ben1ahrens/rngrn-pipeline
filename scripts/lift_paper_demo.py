#!/usr/bin/env python
"""lift_paper_demo.py — the paper's centrepiece figure: a TRAINED model's recovered
parameters, simulated under the DYNAMICAL LIFT, forming a Turing pattern.

WHAT THIS IS. An assembly script, not new machinery. Every number and every field it writes
comes out of code that already exists and has been reviewed:

  population    `lift_ladder.d5_models`   — the 4 Turing-unstable D5 RECOVERED models
                                            (seeds 1/3/5/6, `analysis.json::turing_flags`),
                                            rebuilt from their tracked checkpoints.
  box           `lift_ladder.box_size`    — L = 8 * 2*pi / k*_qss, p = 8 periods per box.
  lifted arm    `eval/lifted_torch.simulate_lifted_torch`  (CUDA, Strang-split full lift)
  QSS control   `eval/rollout.simulate`                    (numpy, the same rollout every
                                            other rollout number in this project came from)
  flags         `observables.classify`, `observables.kstar_of` (inside the simulators),
                `eval/ladder.one_radial_bin` (SPEC §9.1's 2*pi/L), `eval/ladder._rel_l2`.

WHY IT IS NOT JUST `ladder.v3_spatial`. That function computes exactly these flags and is
the reviewed path — but it returns only scalars, and the deliverable here is the FIELD. So
`_demo_row` mirrors its call pattern line for line (same `step_policy` dt and horizon via
`dt=None, T=None`, same `seed` into both arms, same `noise=1e-2`, same `xstar`) and keeps the
final frames. The derived quantities are computed with the same primitives v3_spatial calls,
so a row here and a row there are the same measurement.

SCOPE, STATED ONCE AND HONESTLY. This runs at n = 128. §5.3 V3 licenses the MORPHOLOGY-CLASS
equality claim at 512^2 ONLY, so `morphology_claimable` is False for every row here by
construction and the morphology labels below are descriptive, not a claim. What 128^2 does
support is the pattern-formation statement: whether the lifted field's amplitude clears the
pattern floor, and where its k* sits relative to the QSS control's within one radial bin.

"TURING-UNSTABLE" (a linear verdict on the dispersion relation) and "PATTERNS" (a simulated
field that cleared the floor) are different claims and are never used for one another here.

NOT a trainer, sweep or target-report, and it opens no `payload.h5` — it reads model
checkpoints only — so neither `.claude/hooks/guard_trainer.py` nor `tests/test_firewall.py`'s
`FORBIDDEN` list applies to it (CLAUDE.md §5, §7a).

    .venv/bin/python scripts/lift_paper_demo.py --out experiments/lift_paper_demo
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from lift_ladder import D5_PREFIX, D5_RUNS, D5_TURING_SEEDS, box_size, d5_models  # noqa: E402

from rngrn import observables as obs  # noqa: E402
from rngrn.eval import ladder, rollout  # noqa: E402
from rngrn.eval.lifted import MU_BIO_CENTRAL, MU_BIO_HI, MU_BIO_LO  # noqa: E402
from rngrn.eval.lifted_torch import simulate_lifted_torch  # noqa: E402

torch.set_default_dtype(torch.float64)

REPO = pathlib.Path(__file__).resolve().parents[1]

# mu_gate = 1e-3 is the owner decision D-REDESIGN-5; mu_central = 7.2e-4 is
# `lifted.MU_BIO_CENTRAL`. Both sit inside the biological band [MU_BIO_LO, MU_BIO_HI].
MU_GATE = 1e-3
DEMO_MUS = (MU_GATE, MU_BIO_CENTRAL)
DEMO_N = 128
DEMO_SEED = 0


def _prov_for_seed(s: int) -> dict:
    rd = D5_RUNS / f"{D5_PREFIX}{s}"
    return dict(seed=int(s), run_dir=str(rd.relative_to(REPO)),
                checkpoint=str((rd / "checkpoints" / "model.pt").relative_to(REPO)))


def _demo_row(model, label, prov, mus, n, seed, device):
    """One model: the QSS control once, then the lifted arm at each mu. Mirrors
    `ladder.v3_spatial`'s call pattern; keeps the fields."""
    L, k_qss_linear, xstar = box_size(model)

    q = rollout.simulate(model, L=L, n=n, T=None, dt=None, seed=seed, noise=1e-2,
                         xstar=xstar, max_steps=200000)
    if q["blew_up"]:
        raise RuntimeError(f"{label}: the QSS control blew up at L={L}, n={n}, dt={q['dt']} "
                           "-- there is no control to read the lifted arms against.")
    qf = q["fields"]
    bin_width = ladder.one_radial_bin(L)

    row = dict(
        label=label, provenance=prov, form=model.form, N=int(model.N),
        L=float(L), n=int(n), seed=int(seed), device=device,
        kstar_qss_linear=float(k_qss_linear), one_radial_bin=float(bin_width),
        periods_requested=8, xstar=[float(v) for v in xstar],
        D=[float(v) for v in model.D.detach().cpu().numpy()],
        pattern_floor=float(max(1e-3, 0.02 * abs(xstar[0]))),
        dt=float(q["dt"]), nsteps=int(q["nsteps"]), sig_max=float(q["sig_max"]),
        qss=dict(patterned=bool(q["patterned"]), morphology=obs.classify(qf[0]),
                 kstar=float(q["kstar"]), amplitude=float(q["amplitude"]),
                 blew_up=bool(q["blew_up"]), stopped_reason=q["stopped_reason"],
                 nsteps=int(q["nsteps"]), nsteps_run=int(q["nsteps_run"]),
                 seconds=float(q["seconds"])),
        lifted={})
    fields = {"qss": qf}

    for mu in mus:
        r = simulate_lifted_torch(model, L=L, mu=mu, n=n, T=None, dt=None, seed=seed,
                                  noise=1e-2, xstar=xstar, max_steps=200000, device=device)
        f = r["fields"]
        d_sim = float(abs(r["kstar"] - q["kstar"]))
        d_lin = float(abs(r["kstar"] - k_qss_linear))
        row["lifted"][f"{mu:g}"] = dict(
            mu=float(mu),
            patterned=bool(r["patterned"]),
            morphology=(obs.classify(f[0]) if not r["blew_up"] else None),
            # MORPHOLOGY CLASS IS A 512^2-ONLY CLAIM (ladder.V3_MORPHOLOGY_MIN_N, §5.3 V3).
            morphology_claimable=bool(n >= ladder.V3_MORPHOLOGY_MIN_N),
            morphology_agrees_qss=bool((not r["blew_up"])
                                       and obs.classify(f[0]) == row["qss"]["morphology"]),
            kstar=float(r["kstar"]), amplitude=float(r["amplitude"]),
            kstar_abs_diff_vs_qss_sim=d_sim,
            kstar_abs_diff_vs_qss_linear=d_lin,
            kstar_within_one_bin_vs_qss_sim=bool(d_sim <= bin_width),
            kstar_within_one_bin_vs_qss_linear=bool(d_lin <= bin_width),
            patterned_agrees_qss=bool(bool(r["patterned"]) == bool(q["patterned"])),
            rel_l2_vs_qss=float(ladder._rel_l2(f, qf)),
            rel_l2_dev_vs_qss=float(ladder._rel_l2_dev(f, qf)),
            blew_up=bool(r["blew_up"]), stopped_reason=r["stopped_reason"],
            nsteps=int(r["nsteps"]), nsteps_run=int(r["nsteps_run"]),
            dt=float(r["dt"]), dt_over_mu=float(r["dt"] / mu),
            seconds=float(r["seconds"]))
        fields[f"lifted_mu{mu:g}"] = f
        s = row["lifted"][f"{mu:g}"]
        print(f"  {label} mu={mu:g}: patterned={s['patterned']} (qss "
              f"{row['qss']['patterned']}) morph={s['morphology']}/"
              f"{row['qss']['morphology']} k*={s['kstar']:.4f} vs qss_sim "
              f"{row['qss']['kstar']:.4f} (|d|={d_sim:.4f}, bin={bin_width:.4f}) "
              f"l2={s['rel_l2_vs_qss']:.3e} {s['stopped_reason']} "
              f"{s['nsteps_run']}/{s['nsteps']} steps ({s['seconds']:.0f}s)", flush=True)
    return row, fields


# ======================================================================================
# figures
# ======================================================================================
def _panel(ax, field, title, sub):
    im = ax.imshow(field, cmap="viridis", origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlabel(sub, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return im


def _sub_lifted(s):
    return (f"{s['morphology']}   k*={s['kstar']:.3f}   "
            f"amp={s['amplitude']:.3f}   {'PATTERNS' if s['patterned'] else 'NO PATTERN'}")


def _sub_qss(row):
    q = row["qss"]
    return (f"{q['morphology']}   k*={q['kstar']:.3f}   "
            f"amp={q['amplitude']:.3f}   {'PATTERNS' if q['patterned'] else 'NO PATTERN'}")


def make_figures(rows, fields_by_label, mus, fig_dir, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for row in rows:
        label = row["label"]
        ff = fields_by_label[label]
        ncol = 1 + len(mus)
        fig, axes = plt.subplots(1, ncol, figsize=(4.0 * ncol, 4.6))
        ims = [_panel(axes[0], ff["qss"][0], "QSS control (reduced model)", _sub_qss(row))]
        for j, mu in enumerate(mus):
            s = row["lifted"][f"{mu:g}"]
            ims.append(_panel(axes[j + 1], ff[f"lifted_mu{mu:g}"][0],
                              rf"lifted, $\mu$ = {mu:g}", _sub_lifted(s)))
        for ax, im in zip(axes, ims):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.suptitle(f"{label} — recovered model, channel 0 final frame  "
                     f"(n={row['n']}, L={row['L']:.1f}, one bin={row['one_radial_bin']:.4f})",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = fig_dir / f"demo_{label.replace('/', '_')}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(p))
        print(f"wrote {p}", flush=True)

    if len(rows) >= 2:
        mu0 = mus[0]
        fig, axes = plt.subplots(2, len(rows), figsize=(3.4 * len(rows), 7.4),
                                 squeeze=False)
        for j, row in enumerate(rows):
            ff = fields_by_label[row["label"]]
            s = row["lifted"][f"{mu0:g}"]
            im0 = _panel(axes[0][j], ff[f"lifted_mu{mu0:g}"][0], row["label"],
                         _sub_lifted(s))
            im1 = _panel(axes[1][j], ff["qss"][0], "QSS control", _sub_qss(row))
            fig.colorbar(im0, ax=axes[0][j], fraction=0.046, pad=0.03)
            fig.colorbar(im1, ax=axes[1][j], fraction=0.046, pad=0.03)
        axes[0][0].set_ylabel(rf"lifted, $\mu$ = {mu0:g}", fontsize=11)
        axes[1][0].set_ylabel("QSS control", fontsize=11)
        fig.suptitle("Turing patterns from RECOVERED parameters under the dynamical lift "
                     f"(channel 0, n={rows[0]['n']})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = fig_dir / "demo_panel_all.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(p))
        print(f"wrote {p}", flush=True)
    return paths


def _provenance():
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return dict(git_commit=sha, host=platform.node(), torch_version=torch.__version__,
                cuda_available=bool(torch.cuda.is_available()),
                device_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available()
                             else None),
                date=time.strftime("%Y-%m-%d %H:%M:%S"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="experiments/lift_paper_demo")
    ap.add_argument("--n", type=int, default=DEMO_N)
    ap.add_argument("--seed", type=int, default=DEMO_SEED)
    ap.add_argument("--mus", type=float, nargs="+", default=list(DEMO_MUS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(D5_TURING_SEEDS),
                    help="which D5 model seeds to run (default: all four Turing ones)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    for mu in args.mus:
        if not (MU_BIO_LO <= mu <= MU_BIO_HI):
            raise ValueError(f"mu={mu:g} is outside the biological band "
                             f"[{MU_BIO_LO:g}, {MU_BIO_HI:g}]; refusing to run it silently.")

    out_dir = pathlib.Path(args.out)
    for sub in ("results", "arrays", "figures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    models, labels = d5_models()
    keep = [(m, l, s) for m, l, s in zip(models, labels, D5_TURING_SEEDS)
            if s in args.seeds]
    print(f"lift paper demo: {len(keep)} recovered models, mus={args.mus}, n={args.n}, "
          f"device={args.device}", flush=True)

    t0 = time.perf_counter()
    rows, fields_by_label, arrays = [], {}, {}
    for m, label, s in keep:
        row, fields = _demo_row(m, label, _prov_for_seed(s), args.mus, args.n, args.seed,
                                args.device)
        rows.append(row)
        fields_by_label[label] = fields
        for key, f in fields.items():
            arrays[f"{label.replace('/', '_')}__{key}"] = f.astype(np.float32)

    paths = make_figures(rows, fields_by_label, args.mus, out_dir / "figures", args.dpi)

    payload = dict(
        what="lifted-simulation pattern demonstration from the 4 Turing-unstable D5 "
             "RECOVERED models",
        population="d5", mus=[float(mu) for mu in args.mus], n=int(args.n),
        seed=int(args.seed), device=args.device,
        mu_band=[float(MU_BIO_LO), float(MU_BIO_HI)],
        morphology_claim_min_n=int(ladder.V3_MORPHOLOGY_MIN_N),
        scope_note=(f"n={args.n} supports the PATTERN-FORMATION statement (amplitude vs the "
                    "pattern floor, and k* against the QSS control within one radial bin). "
                    "The morphology-CLASS equality claim is licensed at 512^2 only "
                    "(REDESIGN §5.3 V3); see experiments/lift_ladder/v3 for that."),
        rows=rows,
        summary=dict(
            n_models=len(rows),
            n_lifted_patterned=int(sum(r["lifted"][f"{mu:g}"]["patterned"]
                                       for r in rows for mu in args.mus)),
            n_lifted_runs=len(rows) * len(args.mus),
            n_qss_patterned=int(sum(r["qss"]["patterned"] for r in rows)),
            n_kstar_within_one_bin_vs_qss_sim=int(
                sum(r["lifted"][f"{mu:g}"]["kstar_within_one_bin_vs_qss_sim"]
                    for r in rows for mu in args.mus)),
            n_kstar_within_one_bin_vs_qss_linear=int(
                sum(r["lifted"][f"{mu:g}"]["kstar_within_one_bin_vs_qss_linear"]
                    for r in rows for mu in args.mus)),
            n_morphology_agrees_qss=int(sum(r["lifted"][f"{mu:g}"]["morphology_agrees_qss"]
                                            for r in rows for mu in args.mus))),
        figures=[str(pathlib.Path(p).relative_to(out_dir)) for p in paths],
        provenance=_provenance(),
        wall_seconds=time.perf_counter() - t0)

    rp = out_dir / "results" / "demo.json"
    with open(rp, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {rp}")
    npz = out_dir / "arrays" / "final_frames.npz"
    np.savez_compressed(npz, **arrays)
    print(f"wrote {npz}")
    print(f"total {payload['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
