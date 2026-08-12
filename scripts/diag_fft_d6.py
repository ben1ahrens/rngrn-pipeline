"""diag_fft_d6.py — D6: invariance checks (PLAN_fourier_training.md §3 D6; SPEC §4, §9.4).

CHECK A — torch RAPS vs numpy reference, bin-for-bin.
    `torch_raps` below is the prototype for the future `losses/spectral.py`. It replicates
    `observables.raps` exactly: P = |FFT2(v - mean)|^2, radial bins of width dk = 2*pi/L
    with edges arange(0, KR.max()+dk, dk), assignment by digitize on KR (bucketize with
    right=True is the torch equivalent — verified: np.digitize(x, bins) == torch.bucketize
    (x, bins, right=True)), per-bin MEAN power, DC bin zeroed, power-weighted-centroid k*
    over argmax +/- 2 bins. torch.fft.fftfreq is bit-identical to np.fft.fftfreq (verified
    on n in {64, 256, 512}), so the bin assignment is bit-for-bit the numpy one; the only
    daylight left is the FFT backend itself (~1e-15 relative). PASS demands per-bin
    |dP|/peak <= 1e-12 and |dk*| <= 1e-12.
    Test fields: observed frame channel 0 of turing_labyrinth/sample_0000 (RecoveryInput
    only) + 3 seeded synthetic random fields (256^2, L=100).

CHECK B — time-rescaling degeneracy (SPEC §9.4), verified numerically.
    Claim: jointly scaling (f, D) by a constant c > 0 leaves the stationary pattern
    invariant — only time is rescaled. On a known-Turing recovered checkpoint:
    (1) analytically: c*J - k^2*(c*D) = c*(J - k^2*D), so sigma_c(k) = c*sigma(k) and the
        argmax k* is invariant. Scaling f by c is NOT expressible exactly in RNGRN's
        parameterisation (alpha/beta/delta live behind softplus; the inverse-softplus
        round-trip is float-approximate, and c*f also rescales the production term through
        alpha only if K's are held fixed — which the theta parameterisation entangles), so
        the check builds the dispersion matrix directly in numpy from J = model.jacobian
        (x*) and D. The D-only half of the scaling IS expressible (theta_D += log c) and
        is exercised as a mechanism check against numpy.
    (2) dynamically: the base model is integrated to a patterned state (eval.rollout.
        simulate, n=64, seed=0, early_stop=True, its own model.D); the scaled system
        (reaction*c, D*c) is integrated with integrate_etdrk4_rfft directly from the SAME
        initial condition (same seed-0 noise around the same x*) with dt/c and the same
        number of steps. The trajectories are exactly time-rescaled, so the final fields
        must agree to integrator rounding (expected ~1e-10; gate 1e-6 rel to channel std)
        with k* identical.
    Invariant combinations (SPEC §9.4): D-ratios, alpha/delta, beta/delta, K's,
    D*k*^2/delta — each is confirmed unchanged under the joint scaling BY CONSTRUCTION
    (numerically, c=3): the c's cancel. NOT pinned: absolute rates (alpha, beta, delta, D
    each carry the free factor c).

FIREWALL: this script OPENS payload.h5 (via rngrn.data.gate.from_registry) to read the
OBSERVED frame of turing_labyrinth/sample_0000, so its module name belongs in
tests/test_firewall.py::FORBIDDEN. It uses ONLY RecoveryInput.frame and RecoveryInput.L;
the AnswerKey is discarded unread.

DIAGNOSTIC SCRIPT — no training. Everything float64 on CPU. Exit code 0 = both checks
pass, 1 = any check fails (results.json is written either way).

Usage:
    KMP_AFFINITY=disabled OMP_NUM_THREADS=1 .venv/bin/python scripts/diag_fft_d6.py \
        [--ckpt experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/checkpoints/model.pt] \
        [--out experiments/diag_fft/d6]
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rngrn.model import RNGRN                                    # noqa: E402
from rngrn.losses.terms import steady_state                      # noqa: E402
from rngrn.eval.numerics import integrate_etdrk4_rfft            # noqa: E402
from rngrn.eval.rollout import simulate, _reaction_np_builder    # noqa: E402
from rngrn import observables as obs                             # noqa: E402
from rngrn.data.gate import from_registry                        # noqa: E402

torch.set_default_dtype(torch.float64)

# ---- pass tolerances (stated before running; near-exactness is DEMANDED, per the plan) --
TOL_RAPS_REL = 1e-12     # per-bin |P_torch - P_numpy| / peak(P_numpy)
TOL_KSTAR_ABS = 1e-12    # |k*_torch - k*_numpy|, rad/length
TOL_SIGMA_SCALE_REL = 1e-10   # max|sigma_c - c*sigma| / max|c*sigma|
TOL_THETA_D_REL = 1e-8   # theta_D += log(c) mechanism vs numpy (exp(t+log c) != c*exp(t) exactly)
TOL_FIELD_REL = 1e-6     # max|dX_ch| / std(X_base_ch); expected ~1e-10 (rounding-seeded)
TOL_DKSTAR_DYN = 1e-9    # |k*_scaled - k*_base| of the final fields, rad/length
C_VALUES = (0.1, 3.0, 10.0)


# ------------------------------------------------------------------ CHECK A: torch RAPS

def torch_raps(field: torch.Tensor, L: float = 1.0):
    """Torch replica of `observables.raps` — prototype for losses/spectral.py.

    Same binning bit-for-bit: torch.fft.fftfreq matches np.fft.fftfreq exactly, torch
    elementary ops (mul/add/sqrt) are correctly rounded like numpy's, and bucketize
    (right=True) is digitize. Returns (k_centers, power, k_star).
    """
    f = field - field.mean()
    n = f.shape[0]
    F = torch.abs(torch.fft.fft2(f)) ** 2
    kfreq = torch.fft.fftfreq(n, d=L / n, dtype=torch.float64) * (2.0 * math.pi)
    KX, KY = torch.meshgrid(kfreq, kfreq, indexing="ij")   # KR is symmetric in (KX, KY)
    KR = torch.sqrt(KX ** 2 + KY ** 2)
    dk = 2.0 * math.pi / L
    # np.arange(0, stop, dk) yields EXACTLY i*dk with length ceil(stop/dk); torch.arange
    # with a float step accumulates differently (measured: 63/364 edges off by 1 ulp on
    # the 512^2 frame), so build the edges as integer-arange * dk to stay bit-identical.
    stop = float(KR.max()) + dk
    kbins = torch.arange(math.ceil(stop / dk), dtype=torch.float64) * dk
    nb = kbins.numel() - 1
    idx = torch.clamp(torch.bucketize(KR.reshape(-1), kbins, right=True) - 1, 0, nb - 1)
    power = torch.bincount(idx, weights=F.reshape(-1), minlength=nb)
    counts = torch.bincount(idx, minlength=nb)
    power = power / torch.clamp(counts, min=1)
    kcent = 0.5 * (kbins[:-1] + kbins[1:])
    power[0] = 0.0                                          # drop k=0
    pk = int(torch.argmax(power))
    lo, hi = max(1, pk - 2), min(nb, pk + 3)
    w = power[lo:hi]
    kstar = float(torch.sum(kcent[lo:hi] * w) / (torch.sum(w) + 1e-12))
    return kcent, power, kstar


def _compare_raps(name: str, field: np.ndarray, L: float) -> dict:
    field = np.ascontiguousarray(field, dtype=np.float64)
    kc_np, p_np, ks_np = obs.raps(field, L)
    kc_t, p_t, ks_t = torch_raps(torch.from_numpy(field), L)
    kc_t, p_t = kc_t.numpy(), p_t.numpy()
    if kc_np.shape != kc_t.shape:
        raise RuntimeError(f"[{name}] bin-count mismatch: numpy {kc_np.shape} vs torch "
                           f"{kc_t.shape} — the binning is NOT bin-for-bin; investigate")
    if not np.array_equal(kc_np, kc_t):
        raise RuntimeError(f"[{name}] bin CENTERS differ (max |d| = "
                           f"{np.max(np.abs(kc_np - kc_t)):.3e}) — edges are not bit-identical")
    peak = float(p_np.max())
    max_abs = float(np.max(np.abs(p_t - p_np)))
    max_rel = max_abs / peak
    dks = abs(ks_t - ks_np)
    out = dict(nbins=int(len(kc_np)), peak_power=peak, max_abs_dpower=max_abs,
               max_rel_dpower_to_peak=max_rel, kstar_numpy=float(ks_np),
               kstar_torch=float(ks_t), abs_dkstar=float(dks),
               rel_dkstar=float(dks / ks_np) if ks_np != 0 else float("nan"),
               passed=bool(max_rel <= TOL_RAPS_REL and dks <= TOL_KSTAR_ABS))
    print(f"  [{name}] bins={out['nbins']}  max|dP|/peak={max_rel:.3e}  "
          f"|dk*|={dks:.3e}  (k*={ks_np:.6f})  -> {'PASS' if out['passed'] else 'FAIL'}")
    return out


def check_a() -> dict:
    print("CHECK A — torch RAPS vs numpy observables.raps, bin-for-bin")
    results = {}
    # (1) observed frame, channel 0. RecoveryInput ONLY; the AnswerKey is discarded unread.
    ri, _unread_key = from_registry("data/datasets", "turing_labyrinth", "sample_0000",
                                    3, (0, 1, 2))
    del _unread_key
    results["turing_labyrinth/sample_0000 ch0"] = _compare_raps(
        "turing_labyrinth/sample_0000 ch0", ri.frame[0], float(ri.L))
    # (2) three seeded synthetic random fields
    rng = np.random.default_rng(0)
    for i in range(3):
        f = rng.standard_normal((256, 256))
        results[f"synthetic_{i}"] = _compare_raps(f"synthetic_{i} (256^2, L=100)", f, 100.0)
    passed = all(r["passed"] for r in results.values())
    print(f"CHECK A: {'PASS' if passed else 'FAIL'} "
          f"(gates: |dP|/peak <= {TOL_RAPS_REL:g}, |dk*| <= {TOL_KSTAR_ABS:g})")
    return dict(fields=results, passed=passed,
                tol=dict(rel_dpower_to_peak=TOL_RAPS_REL, abs_dkstar=TOL_KSTAR_ABS))


# ------------------------------------------------------------------ CHECK B: rescaling

def load_known_turing_model(ckpt_path: str) -> RNGRN:
    """Recovered checkpoint (a recovery OUTPUT, never an answer key), as in diag_fft_d1."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if ck.get("nondim"):
        raise ValueError(f"{ckpt_path} was trained on the nondim path — its D is D/L^2 "
                         "and this script assumes physical D; pick a dimensional run")
    model = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    model.load_state_dict(ck["state_dict"])
    return model


def _sigma_np(J: np.ndarray, D: np.ndarray, kgrid: np.ndarray) -> np.ndarray:
    """max Re eig(J - k^2 diag(D)) over a k grid, batched numpy."""
    M = J[None, :, :] - (kgrid ** 2)[:, None, None] * np.diag(D)[None, :, :]
    return np.linalg.eigvals(M).real.max(axis=-1)


def check_b_analytic(model: RNGRN, xstar_t: torch.Tensor) -> tuple[dict, float]:
    print("CHECK B.1 — dispersion scaling sigma_c(k) = c*sigma(k), k* invariant")
    J = model.jacobian(xstar_t, create_graph=False).detach().numpy()
    D = model.D.detach().numpy()
    # locate k* on a wide log grid, then refine linearly around it
    ks_wide = np.geomspace(1e-4, 1e4, 8000)
    s_wide = _sigma_np(J, D, ks_wide)
    k0 = ks_wide[int(np.argmax(s_wide))]
    kgrid = np.linspace(k0 / 5.0, 5.0 * k0, 2001)
    sigma = _sigma_np(J, D, kgrid)
    pk = int(np.argmax(sigma))
    if not (0 < pk < len(kgrid) - 1):
        raise RuntimeError("dispersion argmax landed on the grid edge — widen the grid")
    if sigma[pk] <= 0:
        raise RuntimeError(f"checkpoint is not Turing-unstable (sigma_max = {sigma[pk]:.3e}) "
                           "— pick a known-Turing checkpoint")
    kstar = float(kgrid[pk])
    print(f"  base: sigma_max = {sigma[pk]:.6e} at k* = {kstar:.6f}")

    per_c = {}
    for c in C_VALUES:
        sigma_c = _sigma_np(c * J, c * D, kgrid)             # c*J - k^2*(c*D), built directly
        rel = float(np.max(np.abs(sigma_c - c * sigma)) / np.max(np.abs(c * sigma)))
        pk_c = int(np.argmax(sigma_c))
        dkstar = float(abs(kgrid[pk_c] - kstar))
        ok = rel <= TOL_SIGMA_SCALE_REL and pk_c == pk
        per_c[str(c)] = dict(max_rel_dev_sigma=rel, argmax_index_base=pk,
                             argmax_index_scaled=pk_c, abs_dkstar=dkstar, passed=bool(ok))
        print(f"  c={c:5g}: max|sigma_c - c*sigma|/max|c*sigma| = {rel:.3e}  "
              f"|dk*| = {dkstar:.3e}  -> {'PASS' if ok else 'FAIL'}")

    # theta_D mechanism: D-only scaling via theta_D += log(c) vs numpy J - k^2*(c*D).
    # NOTE: this is NOT the degeneracy (only the joint (f, D) scaling preserves the
    # pattern); it demonstrates that the parameterisation expresses the D half exactly-ish
    # (exp(theta + log c) differs from c*exp(theta) at the ulp level).
    c = 3.0
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        m2.theta_D += math.log(c)
    sig_t = m2.dispersion(xstar_t, torch.from_numpy(kgrid)).detach().numpy()
    sig_ref = _sigma_np(J, c * D, kgrid)
    rel_td = float(np.max(np.abs(sig_t - sig_ref)) / np.max(np.abs(sig_ref)))
    ok_td = rel_td <= TOL_THETA_D_REL
    print(f"  theta_D += log(3) mechanism vs numpy: max rel dev = {rel_td:.3e}  "
          f"-> {'PASS' if ok_td else 'FAIL'}")
    # bridge: torch dispersion == numpy arithmetic for the base model (reported)
    sig_base_t = model.dispersion(xstar_t, torch.from_numpy(kgrid)).detach().numpy()
    rel_bridge = float(np.max(np.abs(sig_base_t - sigma)) / np.max(np.abs(sigma)))
    print(f"  torch model.dispersion vs numpy eig (base): max rel dev = {rel_bridge:.3e}")

    passed = all(v["passed"] for v in per_c.values()) and ok_td
    return (dict(kstar_base=kstar, sigma_max_base=float(sigma[pk]), per_c=per_c,
                 theta_D_route_rel_dev=rel_td, theta_D_route_passed=bool(ok_td),
                 torch_vs_numpy_dispersion_rel_dev=rel_bridge, passed=bool(passed),
                 note=("scaling f by c is NOT expressible exactly in RNGRN's "
                       "parameterisation (softplus'd rates); verified at the ODE level "
                       "by building c*J - k^2*(c*D) = c*(J - k^2*D) directly in numpy")),
            kstar)


def check_b_dynamic(model: RNGRN, xstar_t: torch.Tensor, kstar: float) -> dict:
    print("CHECK B.2 — dynamic: time-rescaled PDE reaches the SAME pattern")
    N = model.N
    xstar = xstar_t.detach().numpy()
    D = model.D.detach().numpy()
    n = 64
    L = 8.0 * 2.0 * math.pi / kstar        # ~8 periods per box, like the Stage-0 sample
    res = simulate(model, L, n=n, seed=0, early_stop=True, D=D, xstar=xstar,
                   integrator="etdrk4_rfft")
    if res["blew_up"]:
        raise RuntimeError("base rollout blew up — cannot test the rescaling on it")
    print(f"  base rollout: patterned={res['patterned']}  amp={res['amplitude']:.4e}  "
          f"k*={res['kstar']:.6f}  steps={res['nsteps_run']}  dt={res['dt']:.4e}  "
          f"stopped={res['stopped_reason']}")
    if not res["patterned"]:
        raise RuntimeError("base rollout did not pattern — the dynamic check needs a "
                           "patterned state (checkpoint or horizon problem; investigate)")
    dt, nsteps = float(res["dt"]), int(res["nsteps_run"])

    # the SAME IC simulate used: seed-0 noise around the SAME x* (replicated exactly)
    rng = np.random.default_rng(0)
    X0 = xstar[:, None, None] + 1e-2 * rng.standard_normal((N, n, n))
    reaction_np = _reaction_np_builder(model)
    X_base, blew = integrate_etdrk4_rfft(X0.copy(), D, reaction_np, n, L, dt, nsteps)
    if blew:
        raise RuntimeError("base direct integration blew up")
    ks_base = float(obs.kstar_of(X_base[0], L=L))
    # one-call vs simulate's chunked driving (reported context, not a gate)
    sim_vs_direct = float(np.max(np.abs(res["fields"] - X_base)) / X_base[0].std())
    print(f"  simulate (chunked) vs direct one-call base: max|dX|/std0 = {sim_vs_direct:.3e}")

    per_c = {}
    for c in C_VALUES:
        reaction_c = (lambda X, _c=c: _c * reaction_np(X))
        X_c, blew = integrate_etdrk4_rfft(X0.copy(), c * D, reaction_c, n, L, dt / c, nsteps)
        if blew:
            raise RuntimeError(f"scaled integration (c={c}) blew up — rescaling broken")
        rel_ch = [float(np.max(np.abs(X_c[ch] - X_base[ch])) / X_base[ch].std())
                  for ch in range(N)]
        ks_c = float(obs.kstar_of(X_c[0], L=L))
        dks = abs(ks_c - ks_base)
        ok = max(rel_ch) <= TOL_FIELD_REL and dks <= TOL_DKSTAR_DYN
        per_c[str(c)] = dict(max_rel_dfield=max(rel_ch), per_channel_rel_dfield=rel_ch,
                             kstar_scaled=ks_c, abs_dkstar=float(dks), passed=bool(ok))
        print(f"  c={c:5g}: max|dX|/std = {max(rel_ch):.3e}  |dk*| = {dks:.3e}  "
              f"-> {'PASS' if ok else 'FAIL'}")

    passed = all(v["passed"] for v in per_c.values())
    return dict(n=n, L=float(L), dt=dt, nsteps=nsteps, patterned=bool(res["patterned"]),
                kstar_base_field=ks_base, sim_vs_direct_rel=sim_vs_direct, per_c=per_c,
                passed=bool(passed),
                tol=dict(rel_dfield=TOL_FIELD_REL, abs_dkstar=TOL_DKSTAR_DYN))


def check_b_invariants(model: RNGRN, kstar: float) -> dict:
    """SPEC §9.4's invariant combinations under (f, D) -> (c*f, c*D), c = 3.

    Under the joint scaling: alpha -> c*alpha, beta -> c*beta, delta -> c*delta,
    D -> c*D, K's (KA, KR, s) unchanged, k* unchanged (verified in B.1/B.2). Each listed
    combination is invariant BY CONSTRUCTION — the c's cancel; the numbers below are the
    float residue of that cancellation. Absolute rates are NOT pinned: alpha, beta,
    delta, D each individually carry the free factor c.
    """
    print("CHECK B.3 — invariant combinations (c=3, by construction; float residue)")
    c = 3.0
    alpha = model.alpha.detach().numpy()
    beta = model.beta.detach().numpy()
    delta = model.delta.detach().numpy()
    D = model.D.detach().numpy()
    KA = model.KA.detach().numpy()
    KR = model.KR.detach().numpy()

    def rel_dev(a, b):
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))

    combos = {
        # k* is invariant (B.1: argmax index identical; B.2: |dk*| at rounding level)
        "D_ratios": rel_dev((c * D)[:, None] / (c * D)[None, :], D[:, None] / D[None, :]),
        "alpha_over_delta": rel_dev((c * alpha) / (c * delta)[:, None],
                                    alpha / delta[:, None]),
        "beta_over_delta": rel_dev((c * beta) / (c * delta), beta / delta),
        "K_A": rel_dev(KA, KA),        # untouched by the scaling: exactly 0
        "K_R": rel_dev(KR, KR),
        "D_kstar2_over_delta": rel_dev((c * D) * kstar ** 2 / (c * delta),
                                       D * kstar ** 2 / delta),
    }
    not_pinned = {
        "alpha_scale_factor": float(np.mean((c * alpha) / alpha)),
        "beta_scale_factor": float(np.mean((c * beta) / beta)),
        "delta_scale_factor": float(np.mean((c * delta) / delta)),
        "D_scale_factor": float(np.mean((c * D) / D)),
    }
    for k, v in combos.items():
        print(f"  {k:22s} rel dev under scaling: {v:.3e}")
    print(f"  NOT pinned (each scales by c={c:g}): alpha, beta, delta, D "
          f"(measured factors {[round(v, 12) for v in not_pinned.values()]})")
    passed = all(v <= 1e-14 for v in combos.values())
    return dict(c=c, invariant_rel_devs=combos, not_pinned_scale_factors=not_pinned,
                passed=bool(passed))


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default="experiments/tune_comp/runs/"
                    "m3_registry_20260803_190250_seed3/checkpoints/model.pt")
    ap.add_argument("--out", default="experiments/diag_fft/d6")
    args = ap.parse_args()

    res_a = check_a()

    print()
    model = load_known_turing_model(args.ckpt)
    xstar_t, converged = steady_state(model)
    if not converged:
        raise RuntimeError(f"steady state did not converge for {args.ckpt}")
    res_b1, kstar = check_b_analytic(model, xstar_t)
    res_b2 = check_b_dynamic(model, xstar_t, kstar)
    res_b3 = check_b_invariants(model, kstar)
    passed_b = res_b1["passed"] and res_b2["passed"] and res_b3["passed"]
    print(f"CHECK B: {'PASS' if passed_b else 'FAIL'}")

    passed = res_a["passed"] and passed_b
    out = dict(diagnostic="D6", ckpt=args.ckpt,
               check_a=res_a,
               check_b=dict(analytic=res_b1, dynamic=res_b2, invariants=res_b3,
                            passed=bool(passed_b)),
               passed=bool(passed))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nD6 overall: {'PASS' if passed else 'FAIL'}  -> {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
