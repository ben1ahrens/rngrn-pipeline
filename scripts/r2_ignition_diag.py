#!/usr/bin/env python
"""D-R2-1 diagnosis: why the pinned+boxed R2 ignition objective does not ignite.

READ-ONLY ANALYSIS. This script trains nothing. It re-derives every number in
`docs/DIAGNOSTICS_r2_ignition.md` from (a) the tracked `arrays/plot_arrays.npz` of runs that
already exist under `experiments/redesign_r2/`, and (b) gradient probes evaluated at the r2
arm's own initialisation, rebuilt from the tracked `config/frozen_config.yaml`.

It does NOT open `payload.h5` -- `kstar_obs` and the pin are read from the frozen config, which
is exactly the point: the diagnosis needs the *objective's* geometry, not the target frame. It
is therefore not a payload-reading script and needs no entry in
`tests/test_firewall.py::FORBIDDEN`. (It does import `r2_ignition_run` for `_load_box` and the
run constants; that module is on FORBIDDEN, but importing it opens nothing -- the payload read
lives inside its `main()`.)

THREE MEASUREMENTS, matching the task brief's Steps 1, 3 and 4:

  step1  Reproduce the headline outcomes from each run's own arrays: ignition counts, distinct
         sign structures, the final coupling median, and the box's re-centring of the default
         init (D-R2-1's own subject).

  step3  Is the decoupling where members START or where they GO? Reconstructs the per-member
         Jacobian at every recorded step from `hist_params` using the model's OWN autodiff
         kernel (`model._reaction_raw` + `torch.func.jacrev`), never a hand-derived formula,
         and validates the reconstruction against the separately-stored final `J` before
         trusting any trajectory point.

  step4  Per-term gradient magnitudes and SIGNS at the r2 init, decomposed onto the parameters
         that actually carry cross-regulation (`theta_s`, `theta_g`) and further onto the
         off-diagonal of the binding budget. This is what identifies the term responsible.

Usage:  .venv/bin/python scripts/r2_ignition_diag.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

torch.set_default_dtype(torch.float64)

from rngrn.model import BatchedRNGRN, _reaction_raw          # noqa: E402
import rngrn.recover as R                                     # noqa: E402
import rngrn.losses.terms as T                                # noqa: E402
from r2_ignition_run import _load_box, N_SPECIES, FORM, HINGE_MARGIN   # noqa: E402

N = N_SPECIES
ROOT = "experiments/redesign_r2"

# The r2 arm's own frozen config (experiments/redesign_r2/phase1/phase1_r2_B512/config/).
KSTAR_OBS = 0.28387817575592444
XPIN = [0.4193849837438677, 0.754671536968317, 0.3504568664953922]
MODEL_SEED, LR, STEPS = 0, 0.05, 1500

# hist_param_names layout: 9 KA, 9 KR, 9 alpha, 3 delta, 3 beta, 3 D  (36 columns).
SLICES = dict(KA=(0, 9), KR=(9, 18), alpha=(18, 27), delta=(27, 30), beta=(30, 33), D=(33, 36))

ARMS = {
    "r2_B64":         f"{ROOT}/phase1/phase1_r2_B64",
    "r2_B128":        f"{ROOT}/phase1/phase1_r2_B128",
    "r2_B256":        f"{ROOT}/phase1/phase1_r2_B256",
    "r2_B512":        f"{ROOT}/phase1/phase1_r2_B512",
    "a0_B64":         f"{ROOT}/phase1/phase1_a0_B64",
    "a0_B128":        f"{ROOT}/phase1/phase1_a0_B128",
    "r2_nobox_B64":   f"{ROOT}/phase1_ablation/phase1_r2_nobox_B64",
    "r2_noprior_B64": f"{ROOT}/phase1_ablation/phase1_r2_noprior_B64",
}


# ---------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------
def _load(path):
    return np.load(os.path.join(path, "arrays", "plot_arrays.npz"), allow_pickle=True)


def _unpack(P):
    """(M, 36) recorded EFFECTIVE parameters -> the kwargs `_reaction_raw` takes."""
    def grab(key, shape):
        a, b = SLICES[key]
        return torch.tensor(P[:, a:b], dtype=torch.float64).reshape(-1, *shape)
    return dict(KA=grab("KA", (N, N)), KR=grab("KR", (N, N)), alpha=grab("alpha", (N, N)),
                delta=grab("delta", (N,)), beta=grab("beta", (N,)), D=grab("D", (N,)))


def _jacobian(p, x):
    """Per-member J via the MODEL'S OWN kernel + autodiff. Never hand-derived."""
    def f(kA, kR, al, be, de, xx):
        return _reaction_raw(kA, kR, al, be, de, xx, FORM, 2)
    with torch.no_grad():
        J = torch.func.vmap(torch.func.jacrev(f, argnums=5))(
            p["KA"], p["KR"], p["alpha"], p["beta"], p["delta"], x)
    return J.reshape(-1, N, N)


def _coupling(J):
    """T16's metric, per member: max|off-diagonal J| / max|diagonal J|.

    A population that has converged to uncoupled systems scores ~0 here, and cannot pattern by
    construction -- a Turing instability needs cross-regulation.
    """
    M = J.shape[0]
    eye = torch.eye(N, dtype=torch.bool).expand(M, N, N)
    off = J.masked_fill(eye, 0.0).abs().amax(dim=(1, 2))
    dia = torch.diagonal(J, dim1=1, dim2=2).abs().amax(dim=1)
    return (off / dia).numpy()


def _build(box, seeds, device):
    return BatchedRNGRN.from_seeds(
        N=N, seeds=seeds, form=FORM, dispersion_backend="cubic", init="default",
        pin_xstar=XPIN, param_boxes=box).to(device)


# ---------------------------------------------------------------------------------------
# step 1 -- reproduce the headline outcomes from each run's own arrays
# ---------------------------------------------------------------------------------------
def step1(box, seeds):
    out = {"arms": {}, "box_recentring": {}}
    for tag, path in ARMS.items():
        if not os.path.isdir(path):
            continue
        z = _load(path)
        ss = z["sign_structures"]
        offmask = ~np.eye(N, dtype=bool)
        rec = dict(
            run_path=path,
            B=int(len(z["turing"])),
            turing=int(z["turing"].sum()),
            alive=int(z["alive"].sum()),
            distinct_sign_structures=len({tuple(r) for r in ss.tolist()}),
            nonzero_offdiag_signs=int((ss.reshape(-1, N, N)[:, offmask] != 0).sum()),
            coupling_median_final=float(np.median(_coupling(torch.tensor(z["J"])))),
        )
        rec["ignition_rate"] = rec["turing"] / rec["B"]
        out["arms"][tag] = rec

    # D-R2-1 itself: the box moves where the DEFAULT init lands.
    sp, bx = _build(None, seeds, torch.device("cpu")), _build(box, seeds, torch.device("cpu"))
    for nm in ("alpha", "delta"):
        a = float(np.median(getattr(sp, nm).detach().numpy()))
        b = float(np.median(getattr(bx, nm).detach().numpy()))
        out["box_recentring"][nm] = dict(softplus_median=a, boxed_median=b, ratio=b / a)
    out["box"] = {k: list(v) for k, v in box.items()}
    return out


# ---------------------------------------------------------------------------------------
# step 3 -- START or GO? the coupling trajectory
# ---------------------------------------------------------------------------------------
def step3():
    x = torch.tensor(XPIN, dtype=torch.float64)
    out = {}
    for tag, path in ARMS.items():
        if not os.path.isdir(path):
            continue
        z = _load(path)
        if "hist_params" not in z.files or z["hist_params"].size == 0:
            out[tag] = dict(run_path=path, trajectory=None,
                            note="no parameter trace recorded for this cell (telemetry off)")
            continue
        HP, HS, Jfin = z["hist_params"], z["hist_step"], z["J"]
        B = HP.shape[1]
        xb = x.expand(B, N)

        # VALIDATE the reconstruction before trusting any point of it: rebuild J at the last
        # recorded step and compare to the independently-stored final J. `hist_params` is
        # stored float32 while the run was float64, so agreement is expected at ~1e-7, not 0.
        Jr = _jacobian(_unpack(HP[-1].astype(np.float64)), xb).numpy()
        rel = float(np.abs(Jr - Jfin).max() / max(1e-30, np.abs(Jfin).max()))

        traj = []
        for i, s in enumerate(HS):
            c = _coupling(_jacobian(_unpack(HP[i].astype(np.float64)), xb))
            traj.append(dict(step=int(s), coupling_median=float(np.median(c)),
                             coupling_p5=float(np.percentile(c, 5)),
                             coupling_p95=float(np.percentile(c, 95))))
        out[tag] = dict(run_path=path, B=B, recon_vs_stored_J_max_rel_err=rel, trajectory=traj)
    return out


# ---------------------------------------------------------------------------------------
# step 4 -- which term moves the coupling, and in which direction
# ---------------------------------------------------------------------------------------
def step4(box, seeds):
    dev = torch.device("cpu")
    kgrid = R._kgrid_for(KSTAR_OBS, device=dev)
    B = len(seeds)
    x = torch.tensor(XPIN, dtype=torch.float64, device=dev).expand(B, N)
    eye = torch.eye(N, dtype=torch.bool).expand(B, N, N)

    terms = {
        "kstar_si":    lambda m: T.kstar_anchor_si_batched(m, x, kgrid, KSTAR_OBS)[0],
        "turing":      lambda m: T.turing_hinges_split_batched(
            m, x, kgrid, margin=HINGE_MARGIN, k_min_frac=0.1)[0],
        "param_prior": lambda m: T.param_prior_batched(
            m, dratio_centre=7.5, dratio_spread=1.0, box=box)[0],
        "beta_hinge":  lambda m: m.beta_hinge(),
    }

    out = {}
    for tag, fn in terms.items():
        m = _build(box, seeds, dev)
        for p in m.parameters():
            p.grad = None
        L = fn(m)
        L.sum().backward()
        g = {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
             for n, p in m.named_parameters()}
        gs = g["theta_s"]
        s_off = gs.masked_fill(eye, 0.0)
        # descent moves theta_s by -grad; a NEGATIVE mean means the term shrinks coupling.
        out[tag] = dict(
            term_value_mean=float(L.mean().detach()),
            grad_norm_total=float(torch.sqrt(sum((v ** 2).sum() for v in g.values()))),
            grad_norm_theta_s=float(gs.norm()),
            grad_norm_theta_g=float(g["theta_g"].norm()),
            grad_norm_s_offdiag=float(s_off.norm()),
            grad_norm_s_diag=float((gs * eye).norm()),
            mean_descent_on_s_offdiag=float(-s_off.mean()),
            frac_members_pushed_down=float(((-s_off).mean(dim=(1, 2)) < 0).double().mean()),
        )

    # the objective that is ACTUALLY active while `turing` is staged off
    m = _build(box, seeds, dev)
    for p in m.parameters():
        p.grad = None
    (terms["kstar_si"](m) + terms["param_prior"](m) + terms["beta_hinge"](m)).sum().backward()
    st = -m.theta_s.grad.masked_fill(eye, 0.0)
    active = sum(out[t]["grad_norm_s_offdiag"] for t in ("kstar_si", "param_prior", "beta_hinge"))
    out["_staging_window_active_sum"] = dict(
        terms=["kstar_si", "param_prior", "beta_hinge"],
        mean_descent_on_s_offdiag=float(st.mean()),
        frac_members_pushed_down=float((st.mean(dim=(1, 2)) < 0).double().mean()),
        share_of_offdiag_gradient={
            t: out[t]["grad_norm_s_offdiag"] / active
            for t in ("kstar_si", "param_prior", "beta_hinge")},
    )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=f"{ROOT}/ignition_diag")
    ap.add_argument("--B", type=int, default=512, help="members for the gradient probes")
    args = ap.parse_args(argv)

    box = _load_box("configs/bio_box.yaml")
    seeds = [R._restart_seed(MODEL_SEED, r) for r in range(args.B)]

    result = dict(
        provenance=dict(
            what="D-R2-1 diagnosis; read-only re-analysis of tracked runs plus gradient probes",
            frozen_config_source=f"{ROOT}/phase1/phase1_r2_B512/config/frozen_config.yaml",
            kstar_obs=KSTAR_OBS, pin_xstar=XPIN, model_seed=MODEL_SEED,
            probe_B=args.B, lr=LR, steps=STEPS, device="cpu",
            staging="off_frac=0.25, ramp_frac=0.25 -> `turing` weight is 0 for steps 0-374",
        ),
        step1_reproduction=step1(box, seeds),
        step3_coupling_trajectory=step3(),
        step4_term_gradients=step4(box, seeds),
    )

    dest = os.path.join(args.out, "results")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "diagnosis.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(f"wrote {path}")
    return result


if __name__ == "__main__":
    main()
