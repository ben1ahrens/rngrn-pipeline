"""diag_fft_d1_stall.py — diagnose the seed-1 D1 re-witness stall (2026-08-12).

The re-witness run (`diag_fft_d1.py --seed 1 --out experiments/diag_fft/d1_rerun`)
died BEFORE the FD sweep: the loss-target solve at theta + 0.01*N(0,1) (main():496)
stalled Newton at ||F||/||u|| = 1.29e-7 against the 1e-9 requirement. Seed 0's
perturbation direction converged; seed 1's does not. This script reproduces that exact
solve (same rng consumption: default_rng(seed+1), one discarded (3,n,n) draw, then the
six leaf perturbations in THETA_NAMES order) and separates three hypotheses:

  A. TRUE STALL — iterate the verified newton_polish one iteration at a time for 60
     iterations and print the residual trajectory. A plateau = true stall (a residual
     component Newton's step cannot remove); steady decrease = budget issue.
  B. PREMATURE RELAX SATURATION — the ETDRK4 relax declares saturation on amp/k*
     flatness (flat_tol=1e-4 over 5 chunks), which a slowly-annealing labyrinth defect
     can satisfy while the state still creeps. Re-relax with 4x chunk budget and 10x
     tighter flat_tol, then the same Newton trajectory.
  C. (covered by A's 60 > default 30 iterations.)

DIAGNOSTIC SCRIPT — no training, no payload.h5 (fixture is the tracked recovered
checkpoint), so NOT a tests/test_firewall.py::FORBIDDEN entry.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

torch.set_default_dtype(torch.float64)

from diag_fft_d1 import (                     # noqa: E402
    load_known_turing_model, make_spatial_F, translation_modes, newton_polish,
    relax_to_pattern, _spectral_k2)
from rngrn.losses.terms import steady_state   # noqa: E402
from rngrn.model import THETA_NAMES           # noqa: E402
from rngrn import observables as obs          # noqa: E402
from rngrn.utils import git_revision          # noqa: E402


def build_failing_model(ckpt: str, seed: int, n: int):
    """Replicate diag_fft_d1.main() up to the target-theta solve, bit-for-bit."""
    model = load_known_turing_model(ckpt)
    xs, ok = steady_state(model)
    assert ok
    J = model.jacobian(xs, create_graph=False).detach()
    kg = torch.linspace(1e-3, 10.0, 2000)
    sig = model.dispersion(xs, kg, J=J).detach()
    kstar_lin = float(kg[int(sig.argmax())])
    L = 8.0 * 2.0 * np.pi / kstar_lin
    jac_rate = float(torch.linalg.eigvals(J).abs().max())
    dt = 0.2 / jac_rate

    # rng consumption in main(): default_rng(seed+1); ONE (3,n,n) draw for v_rand
    # (the zero-mode check), THEN the six 0.01-perturbations.
    rng = np.random.default_rng(seed + 1)
    _ = rng.standard_normal((3, n, n))
    model_t = load_known_turing_model(ckpt)
    with torch.no_grad():
        for nm in THETA_NAMES:
            p = getattr(model_t, nm)
            p.add_(0.01 * torch.from_numpy(rng.standard_normal(p.shape)))
    return model_t, L, dt, jac_rate


def newton_trajectory(model, X: np.ndarray, n: int, L: float, gamma: float,
                      iters: int) -> tuple[list[float], torch.Tensor]:
    """Iterate the VERIFIED newton_polish one iteration at a time; return residuals."""
    F_fn = make_spatial_F(model, n, L)
    D = model.D.detach().numpy()
    k2_full = _spectral_k2(n, L)
    modes_of = lambda uu: list(translation_modes(uu, n, L))  # noqa: E731
    u = torch.from_numpy(np.ascontiguousarray(X))
    res_traj = []
    for it in range(iters):
        u, res = newton_polish(F_fn, u, modes_of, k2_full, D, gamma, n_iter=1)
        res_traj.append(res)
        print(f"  newton[{it + 1:02d}] ||F||/||u|| = {res:.3e}", flush=True)
        if res < 1e-11:
            break
        # true-stall early exit: 5 consecutive iterations without 1% improvement
        if len(res_traj) >= 6 and res > 0.99 * res_traj[-6]:
            print("  -> stalled (5 iterations without 1% improvement)", flush=True)
            break
    return res_traj, u


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/tune_comp/runs/"
                    "m3_registry_20260803_190250_seed3/checkpoints/model.pt")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", default="experiments/diag_fft/d1_stall")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model_t, L, dt, gamma = build_failing_model(a.ckpt, a.seed, a.n)
    xs, ok = steady_state(model_t)
    if not ok:
        raise RuntimeError("perturbed model has no homogeneous steady state")
    xstar = xs.detach().numpy()
    print(f"perturbed model (seed {a.seed}): L={L:.2f} dt={dt:.4f} gamma={gamma:.4f}",
          flush=True)

    report = dict(ckpt=a.ckpt, seed=a.seed, n=a.n, L=L, dt=dt, gamma=gamma,
                  git_rev=git_revision(), torch_version=torch.__version__, arms={})

    # Arm A — exact repro: default relax budget, then the Newton trajectory.
    print("ARM A: default relax (chunk=500, max_chunks=400, flat_tol=1e-4)", flush=True)
    XA = relax_to_pattern(model_t, xstar, a.n, L, dt, a.seed)
    ampA, kA = float(XA[0].std()), float(obs.kstar_of(XA[0], L=L))
    print(f"  relaxed: amp={ampA:.4f} k*={kA:.4f}", flush=True)
    trajA, _ = newton_trajectory(model_t, XA, a.n, L, gamma, a.iters)
    report["arms"]["A_default_relax"] = dict(amp=ampA, kstar=kA, residuals=trajA)

    # Arm B — 4x relax budget, 10x tighter saturation tolerance.
    print("ARM B: extended relax (max_chunks=1600, flat_tol=1e-5)", flush=True)
    XB = relax_to_pattern(model_t, xstar, a.n, L, dt, a.seed,
                          max_chunks=1600, flat_tol=1e-5)
    ampB, kB = float(XB[0].std()), float(obs.kstar_of(XB[0], L=L))
    extra_steps = None  # how far B's state moved past A's, translation-invariantly
    damp = abs(ampB - ampA) / max(ampA, 1e-300)
    print(f"  relaxed: amp={ampB:.4f} k*={kB:.4f} (amp moved {damp:.2e} vs arm A)",
          flush=True)
    trajB, _ = newton_trajectory(model_t, XB, a.n, L, gamma, a.iters)
    report["arms"]["B_extended_relax"] = dict(amp=ampB, kstar=kB, residuals=trajB,
                                              amp_rel_shift_vs_A=damp,
                                              extra=extra_steps)

    convA = trajA and trajA[-1] < 1e-9
    convB = trajB and trajB[-1] < 1e-9
    verdict = ("budget" if convA else
               "premature_saturation" if convB else
               "true_stall")
    report["verdict"] = verdict
    print(f"VERDICT: arm A converged={convA} (last {trajA[-1]:.2e}), "
          f"arm B converged={convB} (last {trajB[-1]:.2e}) -> {verdict}", flush=True)

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {a.out}/results.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
