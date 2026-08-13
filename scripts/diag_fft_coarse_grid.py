"""diag_fft_coarse_grid.py — fix the TRAINING-solve grid: the finest n with reliable
strict Newton convergence (||F||/||u|| <= 1e-9) at the training geometry, on CUDA.

Context (2026-08-13): D-FFT-11 flag 1 and the F-D1-4 stall finding left the training
grid OPEN, and `experiments/diag_fft/sat512/results.json` closed the 512^2-native
question NEGATIVELY — at 512^2 the Newton polish stalls at ~1.3e-4-2.1e-4 true residual
from EVERY candidate relax handoff (five tolerances, 15-20 min wall each), so the
data-native grid is not reachable with the current machinery. This script measures the
ladder below it.

GEOMETRY NOTE: L is FIXED at 185.01 — the canonical data box (turing_labyrinth), NOT the
periods-derived L=285.49 the other diag scripts use. The fixture's k*_lin=0.176 gives
~5.2 pattern periods in this box: still comfortably multi-period, and it is the box the
training solve will actually run in (recover.py passes L_model).

Protocol (CUDA fp64, fixture checkpoint, torch relax + torch-LSMR Newton — the verified
CUDA path of rngrn.forward):
1. n in (96, 128, 192, 256): instrumented fresh solve at the BASE theta (D1 saturation
   detector flat_tol=1e-4/trailing-5; relax and Newton wall-capped and reported
   honestly, never hidden).
2. At each convergent n: the FD tripwire (1 direction, spec_shape-form + amp_fluct-form
   local closures, eps {1e-4, 1e-5} best-of, 5e-3 tripwire; measured value reported).
3. Stall-rate arm at the two largest convergent n: 3 draws of theta + 0.01*N(0,1)
   (seeds 11/12/13 — the F-D1-4 stall-trigger scale), fresh solve each.
4. Band-content sanity: spec_shape-form distance between the recommended n's u* band
   shape and the next grid up's, on [0.60, 1.55]*k*_solve (same L => same band bins).

DIAGNOSTIC SCRIPT — no training, no payload.h5 (fixture = tracked recovered
checkpoint), so NOT a tests/test_firewall.py::FORBIDDEN entry.
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
from rngrn.forward import (PatternSolver, make_spatial_F, newton_polish,
                           translation_modes)                           # noqa: E402
from rngrn.losses.spectral import band_mask, raps_torch                 # noqa: E402
from rngrn.losses.terms import steady_state                             # noqa: E402
from rngrn.model import RNGRN, THETA_NAMES                              # noqa: E402
from rngrn import observables as obs                                    # noqa: E402
from rngrn.utils import git_revision                                    # noqa: E402

L_DATA = 185.01                 # the canonical data box (see module docstring)
GRIDS = (96, 128, 192, 256)
CONV_TOL = 1e-9                 # PatternSolver.CONVERGENCE_TOL, D1 verbatim
STALL_SEEDS = (11, 12, 13)


def load_fixture(ckpt: str, dev: torch.device) -> RNGRN:
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    if ck.get("nondim"):
        raise ValueError("fixture must be a dimensional run")
    m = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    m.load_state_dict(ck["state_dict"])
    return m.to(dev)


def model_scales(model: RNGRN):
    xs, ok = steady_state(model)
    if not ok:
        return None
    J = model.jacobian(xs, create_graph=False).detach()
    jac_rate = float(torch.linalg.eigvals(J).abs().max())
    return xs.detach().cpu().numpy(), 0.2 / jac_rate, jac_rate


def fresh_solve(model: RNGRN, n: int, L: float, seed: int, dev: torch.device,
                relax_cap_s: float, newton_cap_s: float) -> dict:
    """Instrumented mirror of PatternSolver.solve()'s fresh path: chunked torch relax
    with the D1 detector (flat_tol=1e-4 over trailing 5 chunks, max 400 chunks), then
    the torch-LSMR Newton polish iterated one iteration at a time under a wall cap."""
    out: dict = dict(n=n, seed=seed)
    scales = model_scales(model)
    if scales is None:
        out.update(status="no_steady_state")
        return out
    xstar, dt, gamma = scales
    D_np = model.D.detach().cpu().numpy()
    k2_full = _spectral_k2(n, L)
    reaction_t = _torch_reaction_builder(model, dev)
    coeffs = torch_half_coeffs(D_np, n, L, dt, dev)
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(
        xstar[:, None, None] + 1e-2 * rng.standard_normal((model.N, n, n))
    ).to(dev).unsqueeze(0)

    amps: list[float] = []
    ks: list[float] = []
    t0 = time.perf_counter()
    saturated = False
    for chunk_i in range(1, 401):
        X, blew = integrate_etdrk4_rfft_torch(X, reaction_t, n, dt, 500, coeffs)
        if blew:
            out.update(status="relax_blew_up", relax_chunks=chunk_i,
                       relax_seconds=time.perf_counter() - t0)
            return out
        ch0 = X[0, 0].detach().cpu().numpy()
        amps.append(float(ch0.std()))
        ks.append(float(obs.kstar_of(ch0, L=L)))
        if len(amps) >= 5:
            am = np.array(amps[-5:]); km = np.array(ks[-5:])
            if ((am.max() - am.min()) / am.mean() < 1e-4
                    and (km.max() - km.min()) / km.mean() < 1e-4):
                saturated = True
                break
        if time.perf_counter() - t0 > relax_cap_s:
            out.update(status="relax_wall_capped", relax_chunks=chunk_i,
                       relax_seconds=time.perf_counter() - t0)
            return out
    relax_s = time.perf_counter() - t0
    out.update(relax_chunks=len(amps), relax_seconds=relax_s,
               relax_saturated=saturated, amp=amps[-1], kstar=ks[-1])
    if not saturated:
        out.update(status="relax_no_saturation")
        return out

    u = X[0].detach().clone()
    F_fn = make_spatial_F(model, n, L)
    modes_of = lambda uu: list(translation_modes(uu, n, L))  # noqa: E731
    t1 = time.perf_counter()
    res = float("inf")
    iters = 0
    capped = False
    for it in range(30):
        u, res = newton_polish(F_fn, u, modes_of, k2_full, D_np, gamma, n_iter=1)
        iters = it + 1
        el = time.perf_counter() - t1
        if res <= 1e-11:
            break
        if el > newton_cap_s:
            capped = True
            break
    newton_s = time.perf_counter() - t1
    pattern_floor = max(1e-3, 0.02 * abs(float(xstar[0])))
    amp = float(u[0].std())
    out.update(newton_seconds=newton_s, newton_iters=iters, residual=res,
               newton_wall_capped=capped, converged=bool(res <= CONV_TOL),
               patterned=bool(amp > pattern_floor), final_amp=amp,
               total_seconds=relax_s + newton_s,
               status="converged" if res <= CONV_TOL else "newton_stalled")
    out["u_star"] = u          # tensor rider, stripped before JSON
    return out


# ---- FD tripwire helpers (local closures, as in tests/test_forward_solve.py) --------

def _local_terms(u_base: torch.Tensor, L: float, kstar: float):
    rng = np.random.default_rng(7)
    kc, p = raps_torch(u_base[0], L)
    mask = torch.from_numpy((kc >= 0.5 * kstar) & (kc <= 1.5 * kstar))
    t_raw = (p[mask] / p[mask].sum()).detach()
    t_band = t_raw * torch.from_numpy(np.exp(0.2 * rng.standard_normal(len(t_raw))))
    t_band = (t_band / t_band.sum()).detach()
    t_logstd = [float(torch.log(u_base[c].std())) + 0.1
                for c in range(u_base.shape[0])]

    def spec_shape(u):
        _, pw = raps_torch(u[0], L)
        s = pw[mask] / pw[mask].sum()
        return ((torch.log(s + 1e-300) - torch.log(t_band + 1e-300)) ** 2).sum()

    def amp_fluct(u):
        return sum((torch.log(u[c].std()) - t_logstd[c]) ** 2
                   for c in range(u.shape[0]))

    return spec_shape, amp_fluct


def fd_tripwire(ckpt: str, dev, n: int, L: float, u_base: torch.Tensor,
                seed: int) -> dict:
    model = load_fixture(ckpt, dev)
    solver = PatternSolver(model, n, L, seed)          # warm_mode="newton" default
    solver._warm = u_base.clone()
    u, reason = solver.solve()
    if reason != "ok":
        return dict(status=f"base_warm_solve_{reason}")
    kstar = float(obs.kstar_of(u.detach()[0].cpu().numpy(), L=L))
    spec_shape, amp_fluct = _local_terms(u.detach().cpu(), L, kstar)
    params = [getattr(model, nm) for nm in THETA_NAMES]
    loss = spec_shape(u.cpu()) + amp_fluct(u.cpu())
    g = torch.autograd.grad(loss, params)
    g_flat = torch.cat([gi.reshape(-1) for gi in g]).cpu().numpy()
    sizes = sum(getattr(model, nm).numel() for nm in THETA_NAMES)
    rng = np.random.default_rng(seed + 1)
    v = rng.standard_normal(sizes)
    v /= np.linalg.norm(v)
    d_ift = float(g_flat @ v)

    rel_errs = {}
    for eps in (1e-4, 1e-5):
        vals = {}
        for sgn in (+1.0, -1.0):
            m2 = load_fixture(ckpt, dev)
            off = 0
            with torch.no_grad():
                for nm in THETA_NAMES:
                    p = getattr(m2, nm)
                    dv = torch.from_numpy(v[off:off + p.numel()]
                                          ).reshape(p.shape).to(dev)
                    p.add_(sgn * eps * dv)
                    off += p.numel()
            s2 = PatternSolver(m2, n, L, seed)
            s2._warm = u.detach().clone()
            u2, r2 = s2.solve()
            if r2 != "ok":
                return dict(status=f"fd_solve_{r2}", eps=eps, sgn=sgn)
            drift = float((u2.detach() - u.detach()).norm() / u.detach().norm())
            if drift > 0.2:
                return dict(status="fd_left_branch", drift=drift, eps=eps)
            vals[sgn] = float(spec_shape(u2.detach().cpu())
                              + amp_fluct(u2.detach().cpu()))
        d_fd = (vals[+1.0] - vals[-1.0]) / (2.0 * eps)
        rel_errs[str(eps)] = abs(d_fd - d_ift) / max(abs(d_fd), abs(d_ift), 1e-300)
    best = min(rel_errs.values())
    return dict(status="ok", rel_errs=rel_errs, best=best,
                passes_tripwire=bool(best <= 5e-3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/tune_comp/runs/"
                    "m3_registry_20260803_190250_seed3/checkpoints/model.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relax-cap-s", type=float, default=700.0)
    ap.add_argument("--newton-cap-s", type=float, default=500.0)
    ap.add_argument("--out", default="experiments/diag_fft/coarse_grid")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("this is a CUDA measurement; no CUDA device found")
    dev = torch.device("cuda")

    report = dict(ckpt=a.ckpt, args=vars(a), git_rev=git_revision(),
                  torch_version=torch.__version__,
                  device=torch.cuda.get_device_name(0), L=L_DATA,
                  grids=list(GRIDS), stall_seeds=list(STALL_SEEDS),
                  base={}, fd={}, stall={}, band_sanity={})

    # ---- 1. fresh solves at base theta ------------------------------------------
    u_stars: dict[int, torch.Tensor] = {}
    for n in GRIDS:
        print(f"=== base solve n={n} (L={L_DATA}) ===", flush=True)
        model = load_fixture(a.ckpt, dev)
        row = fresh_solve(model, n, L_DATA, a.seed, dev,
                          a.relax_cap_s, a.newton_cap_s)
        u = row.pop("u_star", None)
        if u is not None:
            u_stars[n] = u
        report["base"][str(n)] = row
        print(f"  {row['status']}: relax {row.get('relax_seconds', float('nan')):.0f}s"
              f"/{row.get('relax_chunks', '?')} chunks, newton "
              f"{row.get('newton_seconds', float('nan')):.0f}s, res "
              f"{row.get('residual', float('nan')):.2e}, amp "
              f"{row.get('final_amp', float('nan')):.4f}", flush=True)

    convergent = [n for n in GRIDS
                  if report["base"][str(n)].get("converged")
                  and report["base"][str(n)].get("patterned")]
    print(f"convergent+patterned grids: {convergent}", flush=True)

    # ---- 2. FD tripwire at each convergent n -------------------------------------
    for n in convergent:
        print(f"=== FD tripwire n={n} ===", flush=True)
        fd = fd_tripwire(a.ckpt, dev, n, L_DATA, u_stars[n], a.seed)
        report["fd"][str(n)] = fd
        print(f"  {fd}", flush=True)

    # ---- 3. stall-rate arm at the two largest convergent n -----------------------
    for n in convergent[-2:]:
        rows = []
        for s in STALL_SEEDS:
            print(f"=== stall arm n={n} seed={s} (theta + 0.01*N(0,1)) ===", flush=True)
            m = load_fixture(a.ckpt, dev)
            rng = np.random.default_rng(s)
            with torch.no_grad():
                for nm in THETA_NAMES:
                    p = getattr(m, nm)
                    p.add_(0.01 * torch.from_numpy(
                        rng.standard_normal(tuple(p.shape))).to(dev))
            row = fresh_solve(m, n, L_DATA, a.seed, dev,
                              a.relax_cap_s, a.newton_cap_s)
            row.pop("u_star", None)
            row["perturb_seed"] = s
            rows.append(row)
            print(f"  {row['status']}: res {row.get('residual', float('nan')):.2e} "
                  f"amp {row.get('final_amp', float('nan')):.4f}", flush=True)
        report["stall"][str(n)] = rows

    # ---- 4. band-content sanity: recommended n vs the next grid up ---------------
    if len(convergent) >= 1:
        rec = convergent[-1]
        nxt = next((g for g in GRIDS if g > rec and g in u_stars), None)
        if nxt is not None:
            u_a = u_stars[rec][0].detach().cpu()
            u_b = u_stars[nxt][0].detach().cpu()
            kstar = float(obs.kstar_of(u_a.numpy(), L=L_DATA))
            ka, pa = raps_torch(u_a, L_DATA)
            kb, pb = raps_torch(u_b, L_DATA)
            ma = band_mask(ka, kstar, 0.60, 1.55)
            mb = band_mask(kb, kstar, 0.60, 1.55)
            if int(ma.sum()) != int(mb.sum()):
                raise RuntimeError("band bin mismatch across grids at the same L")
            sa = pa[torch.from_numpy(ma)]; sa = sa / sa.sum()
            sb = pb[torch.from_numpy(mb)]; sb = sb / sb.sum()
            dist = float(((torch.log(sa + 1e-300)
                           - torch.log(sb + 1e-300)) ** 2).sum())
            report["band_sanity"] = dict(
                n_recommended=rec, n_reference=nxt, kstar=kstar,
                spec_shape_distance=dist, n_band_bins=int(ma.sum()),
                patch_control_floor_note=(
                    "D3's patch-to-patch spec_shape floor on the observed frame was "
                    "0.122-0.621 (mean 0.389); a cross-grid distance well below that "
                    "is within estimation noise"))
            print(f"band sanity {rec} vs {nxt}: spec_shape distance {dist:.4f} "
                  f"({int(ma.sum())} bins)", flush=True)
        else:
            report["band_sanity"] = dict(
                note=f"no grid above the recommended n={rec} produced a u* to "
                     "compare against")

    report["recommended_n"] = convergent[-1] if convergent else None
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nrecommended training grid: {report['recommended_n']}; "
          f"wrote {a.out}/results.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
