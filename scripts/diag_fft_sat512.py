"""diag_fft_sat512.py — calibrate the relax→Newton handoff criterion at 512^2 (CUDA).

Motivation (GPU-port unit, 2026-08-12): the D1 saturation detector (amp AND k* trailing-5
relative range < flat_tol=1e-4) never fires at 512^2 within 200k ETDRK4 steps — the
one-shot CUDA fresh solve exhausted its full relax budget and failed
(`/home/benja/.claude/jobs/f7707403/tmp/cuda512.log`; 96^2 fires the same detector fine).
The detector's tolerance is therefore the blocking UNCALIBRATED knob for 512^2-native
training. This script measures, from ONE captured relax trajectory, the handoff point for
a LADDER of candidate tolerances and whether the D-FFT-10 Newton polish converges
(||F||/||u|| <= 1e-9) from each — i.e. how early the relax may hand off and still reach a
true F=0 point. No src/ default is changed here; the numbers feed a DECISIONS closure.

Protocol per orchestrator brief: n=512, CUDA fp64, tracked fixture checkpoint, BASE theta.
One relax trajectory, states snapshotted at each candidate's first-crossing chunk; capped
Newton polish per candidate (newton_polish iterated n_iter=1 with a wall clock between
iterations — identical per-iteration semantics to the standard call, damping included;
the cap is recorded, never hidden). tol=1e-2 row doubles as the closest analogue of
eval/rollout's own detector (saturation_tol=0.01, window 5 — rollout checks every 200
steps where this script chunks by 500; noted in the JSON, not silently equated).

DIAGNOSTIC SCRIPT — no training, no payload.h5 (fixture = tracked recovered checkpoint),
so NOT a tests/test_firewall.py::FORBIDDEN entry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

torch.set_default_dtype(torch.float64)

from rngrn.etdrk4_torch import (_torch_reaction_builder, integrate_etdrk4_rfft_torch,
                                torch_half_coeffs)                      # noqa: E402
from rngrn.eval.numerics import _spectral_k2                            # noqa: E402
from rngrn.forward import make_spatial_F, newton_polish, translation_modes  # noqa: E402
from rngrn.losses.terms import steady_state                             # noqa: E402
from rngrn.model import RNGRN                                           # noqa: E402
from rngrn import observables as obs                                    # noqa: E402
from rngrn.utils import git_revision                                    # noqa: E402

TOLS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)      # candidate handoff flat_tols, loosest first


def load_fixture(ckpt: str) -> RNGRN:
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    if ck.get("nondim"):
        raise ValueError("fixture must be a dimensional run")
    m = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    m.load_state_dict(ck["state_dict"])
    return m


def capped_newton(model, u: torch.Tensor, n: int, L: float, k2_full, D_np,
                  gamma: float, cap_s: float) -> tuple[float, float, int, bool]:
    """newton_polish iterated one iteration at a time with a wall cap between
    iterations. Per-iteration semantics identical to the standard n_iter=30 call
    (each call runs one damped bulk-LSMR step + 2x2 translation correction);
    convergence bar is the standard <=1e-9 on ||F||/||u||."""
    F_fn = make_spatial_F(model, n, L)
    modes_of = lambda uu: list(translation_modes(uu, n, L))  # noqa: E731
    t0 = time.perf_counter()
    res = float("inf")
    iters = 0
    capped = False
    for it in range(30):
        u, res = newton_polish(F_fn, u, modes_of, k2_full, D_np, gamma, n_iter=1)
        iters = it + 1
        el = time.perf_counter() - t0
        print(f"    newton[{iters:02d}] res={res:.3e} ({el:.0f}s)", flush=True)
        if res <= 1e-9:
            break
        if el > cap_s:
            capped = True
            print(f"    -> wall cap {cap_s:.0f}s reached; recording residual honestly",
                  flush=True)
            break
    return time.perf_counter() - t0, res, iters, capped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/tune_comp/runs/"
                    "m3_registry_20260803_190250_seed3/checkpoints/model.pt")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--periods", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--max-chunks", type=int, default=200)
    ap.add_argument("--newton-cap-s", type=float, default=900.0)
    ap.add_argument("--out", default="experiments/diag_fft/sat512")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("this calibration is a CUDA measurement; no CUDA device found")
    dev = torch.device("cuda")

    model = load_fixture(a.ckpt).to(dev)
    xs, ok = steady_state(model)
    if not ok:
        raise RuntimeError("steady state failed on the fixture")
    xstar = xs.detach().cpu().numpy()
    J = model.jacobian(xs, create_graph=False).detach()
    kg = torch.linspace(1e-3, 10.0, 2000, device=dev)
    sig = model.dispersion(xs, kg, J=J).detach()
    kstar_lin = float(kg[int(sig.argmax())])
    L = a.periods * 2.0 * np.pi / kstar_lin
    jac_rate = float(torch.linalg.eigvals(J).abs().max())
    dt = 0.2 / jac_rate
    gamma = jac_rate
    D_np = model.D.detach().cpu().numpy()
    k2_full = _spectral_k2(a.n, L)
    print(f"fixture: k*_lin={kstar_lin:.4f} L={L:.2f} dt={dt:.4f} n={a.n} on "
          f"{torch.cuda.get_device_name(0)}", flush=True)

    report = dict(ckpt=a.ckpt, args=vars(a), git_rev=git_revision(),
                  torch_version=torch.__version__,
                  device=torch.cuda.get_device_name(0),
                  kstar_lin=kstar_lin, L=L, dt=dt, gamma=gamma,
                  tols=list(TOLS),
                  rollout_reference_note=(
                      "eval/rollout's own detector is saturation_tol=0.01 over a "
                      "5-check window every 200 steps; the tol=1e-2 row here uses "
                      "500-step chunks, so it is the closest analogue, not a replica"),
                  trajectory=[], candidates={})

    # ---- 1. one relax trajectory, flatness recorded per chunk --------------------
    reaction_t = _torch_reaction_builder(model, dev)
    coeffs = torch_half_coeffs(D_np, a.n, L, dt, dev)
    rng = np.random.default_rng(a.seed)
    X = torch.from_numpy(
        xstar[:, None, None] + 1e-2 * rng.standard_normal((model.N, a.n, a.n))
    ).to(dev).unsqueeze(0)
    amps: list[float] = []
    ks: list[float] = []
    snapshots: dict[float, torch.Tensor] = {}
    fired_at: dict[float, int] = {}
    relax_t: dict[float, float] = {}
    t0 = time.perf_counter()
    for chunk_i in range(1, a.max_chunks + 1):
        X, blew = integrate_etdrk4_rfft_torch(X, reaction_t, a.n, dt, a.chunk, coeffs)
        if blew:
            raise RuntimeError(f"relax blew up at chunk {chunk_i}")
        ch0 = X[0, 0].detach().cpu().numpy()
        amps.append(float(ch0.std()))
        ks.append(float(obs.kstar_of(ch0, L=L)))
        row = dict(chunk=chunk_i, steps=chunk_i * a.chunk, amp=amps[-1], kstar=ks[-1],
                   seconds=time.perf_counter() - t0)
        if len(amps) >= 5:
            am = np.array(amps[-5:]); km = np.array(ks[-5:])
            row["amp_range"] = float((am.max() - am.min()) / am.mean())
            row["kstar_range"] = float((km.max() - km.min()) / km.mean())
            flat = max(row["amp_range"], row["kstar_range"])
            for tol in TOLS:
                if tol not in fired_at and flat < tol:
                    fired_at[tol] = chunk_i
                    relax_t[tol] = row["seconds"]
                    snapshots[tol] = X[0].detach().cpu().clone()
                    print(f"  tol={tol:g} fired at chunk {chunk_i} "
                          f"({row['seconds']:.0f}s)", flush=True)
        report["trajectory"].append(row)
        if chunk_i % 20 == 0 or chunk_i <= 5:
            print(f"  chunk {chunk_i:3d}: amp={amps[-1]:.4f} k*={ks[-1]:.4f} "
                  f"flat={row.get('amp_range', float('nan')):.2e}/"
                  f"{row.get('kstar_range', float('nan')):.2e} "
                  f"({row['seconds']:.0f}s)", flush=True)
        if len(fired_at) == len(TOLS):
            print("  every candidate tol has fired; stopping the trajectory early",
                  flush=True)
            break
    for tol in TOLS:
        if tol not in fired_at:
            print(f"  tol={tol:g} did NOT fire within {a.max_chunks} chunks", flush=True)

    # ---- 2. capped Newton polish from each candidate handoff state ----------------
    for tol in TOLS:
        if tol not in fired_at:
            report["candidates"][str(tol)] = dict(fired=False)
            continue
        print(f"candidate tol={tol:g} (chunk {fired_at[tol]}): Newton polish...",
              flush=True)
        u0 = snapshots[tol].to(dev)
        nt, res, iters, capped = capped_newton(model, u0, a.n, L, k2_full, D_np,
                                               gamma, a.newton_cap_s)
        report["candidates"][str(tol)] = dict(
            fired=True, chunk=fired_at[tol], steps=fired_at[tol] * a.chunk,
            relax_seconds=relax_t[tol], newton_seconds=nt, newton_iters=iters,
            newton_residual=res, converged=bool(res <= 1e-9), wall_capped=capped,
            total_seconds=relax_t[tol] + nt)
        print(f"  -> res={res:.2e} converged={res <= 1e-9} newton={nt:.0f}s "
              f"total={relax_t[tol] + nt:.0f}s capped={capped}", flush=True)

    fired_ok = [t for t in TOLS if report["candidates"].get(str(t), {}).get("converged")]
    report["loosest_converged_tol"] = max(fired_ok) if fired_ok else None
    if fired_ok:
        best = report["candidates"][str(max(fired_ok))]
        report["per_solve_seconds_at_512"] = best["total_seconds"]
        print(f"\nSUMMARY: loosest tol whose handoff Newton-converges = "
              f"{max(fired_ok):g}; 512^2 fresh-solve cost = "
              f"{best['total_seconds']:.0f}s (relax {best['relax_seconds']:.0f}s + "
              f"newton {best['newton_seconds']:.0f}s)", flush=True)
    else:
        print("\nSUMMARY: NO candidate handoff state Newton-converged — the finding "
              "is that the handoff needs something other than a looser flat_tol",
              flush=True)

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {a.out}/results.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
