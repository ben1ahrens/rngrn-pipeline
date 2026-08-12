"""diag_fft_d2.py — D2: forward-solve characterisation for Fourier-space training.

PLAN_fourier_training.md §3 D2 / §7 (solver-acceleration targets); SPEC §3 (ignition
floor), §8 (validation grid). Measures, on the KNOWN-TURING fixture checkpoint:

  1. full-rollout cost vs grid (96/128/256/512, 3 seeds) via eval/rollout.simulate
     with early_stop, recording seconds / steps / dt / stopped_reason / patterned /
     amplitude / k* and the ru_maxrss profile (a HIGH-WATER mark: grids run in
     ascending order so the profile is honest — a later smaller reading is impossible);
  2. time-to-saturation per grid (mean/min/max of seconds and steps until stopped);
  3. the ignition amplitude floor: amplitude trajectories (std of channel 0 every
     check) for 3 seeds of the Turing fixture at 96² vs the same protocol on a
     verified NON-Turing random model — the proposed floor is the geometric mean of
     the two final-amplitude ranges' nearest edges, reported next to the repo's
     existing pattern_floor = max(1e-3, 0.02·|x*_0|) (eval/rollout.py:234). The
     proposal is a MEASUREMENT for the parent to record, not a decision made here;
  4. grid fidelity at fixed L: does a cheaper grid reproduce the 512² k* (within one
     radial bin 2π/L) and phase_topology.measure class call (from the same rollouts);
  5. a torch ETDRK4 rfft prototype, ported from eval/numerics.integrate_etdrk4_rfft
     (same contour-integral coefficients, complex128), batched over ICs for ONE
     theta only (batched-over-theta is M1 work): verified against numpy (max |Δfield|
     after 100 steps at 96², B=1, ≤ 1e-12) and timed per step at 96²/128² for
     B ∈ {1, 8, 32} on CPU, and on CUDA when available.

DIAGNOSTIC SCRIPT — no training, and it never opens payload.h5 (the fixture theta is
a tracked RECOVERED checkpoint, a recovery output; the non-Turing model is a random
construction), so no FORBIDDEN entry is needed. Float64 throughout, deterministic
seeds. --smoke (default) runs one 96² rollout, the torch-vs-numpy equivalence check,
and the non-Turing sig_max verification; --full runs everything.

Usage (CLAUDE.md §7a: --full goes through the memory guard):
    KMP_AFFINITY=disabled OMP_NUM_THREADS=1 .venv/bin/python -u scripts/diag_fft_d2.py --smoke
    bash scripts/guarded_run.sh .venv/bin/python -u scripts/diag_fft_d2.py --full
"""
from __future__ import annotations
import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rngrn.model import RNGRN                                        # noqa: E402
from rngrn.losses.terms import steady_state                          # noqa: E402
from rngrn.eval.numerics import integrate_etdrk4_rfft, _spectral_k2_half  # noqa: E402
from rngrn.eval.rollout import simulate, _reaction_np_builder        # noqa: E402
from rngrn import observables as obs                                 # noqa: E402
import phase_topology                                                # noqa: E402

torch.set_default_dtype(torch.float64)

CKPT_DEFAULT = ("experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/"
                "checkpoints/model.pt")
OUT_DEFAULT = "experiments/diag_fft/d2"
GRIDS = (96, 128, 256, 512)          # ascending — required for an honest ru_maxrss profile
SEEDS = (0, 1, 2)
CHECK_EVERY = 200
PERIODS_PER_BOX = 8                  # L = 8·2π/k*_lin, matching the Stage-0 sample's p=8
EQUIV_TOL = 1e-12                    # torch-vs-numpy max |Δfield| after 100 steps, B=1
TRAJ_MAX_STEPS = 20000               # cap for the manual trajectory runs; hitting it is reported


def _rss_kib() -> int:
    """ru_maxrss (KiB on Linux) — a process-lifetime HIGH-WATER mark, never decreasing."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _j(x):
    """JSON-safe scalar: NaN/inf -> None (strict JSON), numpy scalars -> python."""
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


# ------------------------------------------------------------------ model construction

def load_fixture(ckpt_path: str) -> RNGRN:
    """The known-Turing fixture: a tracked recovered checkpoint (recovery OUTPUT)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if ck["nondim"]:
        raise ValueError(f"{ckpt_path} was trained on the nondim path — its D is D/L^2 "
                         "and this script assumes physical D; pick a dimensional run")
    model = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    model.load_state_dict(ck["state_dict"])
    return model


def linear_kstar(model: RNGRN) -> tuple[float, float, np.ndarray]:
    """(k*_lin, sig_max_lin, xstar) from steady_state + model.dispersion.

    The k-grid upper edge doubles until the dispersion argmax is interior (fail loud
    after 6 doublings). The fixture MUST be Turing-unstable — asserted, not assumed.
    """
    xs, converged = steady_state(model)
    if not converged:
        raise RuntimeError("steady_state did not converge on the fixture model")
    J = model.jacobian(xs, create_graph=False)
    kmax = 1.0
    for _ in range(6):
        kg = torch.linspace(1e-3, kmax, 4000, dtype=model.dtype)
        sig = model.dispersion(xs, kg, J=J).detach().numpy()
        i = int(np.argmax(sig))
        if i < int(0.98 * len(kg)):
            break
        kmax *= 2.0
    else:
        raise RuntimeError(f"dispersion argmax still at the grid edge at kmax={kmax}")
    k_lin = float(kg[i])
    sig_max = float(sig[i])
    if sig_max <= 0:
        raise RuntimeError(f"fixture is not Turing-unstable: sig_max = {sig_max:.4g}")
    return k_lin, sig_max, xs.detach().numpy()


def nonturing_sig_max(model: RNGRN, n: int, L: float) -> tuple[float, np.ndarray]:
    """(sig_max, xstar) for the contrast model over the SAME k-grid simulate uses.

    Fails loud if the steady state does not converge or if the model is NOT stable
    (this diagnostic needs a decaying contrast case, and a wrong one is useless).
    """
    xs, converged = steady_state(model)
    if not converged:
        raise RuntimeError("steady_state did not converge on the non-Turing model")
    J = model.jacobian(xs, create_graph=False)
    kg = torch.linspace(1e-3, 2 * np.pi * (n // 2) / L, 2000, dtype=model.dtype)
    sig = model.dispersion(xs, kg, J=J).detach().numpy()
    sig_max = float(sig.max())
    if sig_max >= 0:
        raise RuntimeError(f"contrast model is not linearly stable: sig_max = "
                           f"{sig_max:.4g} — pick a different seed")
    return sig_max, xs.detach().numpy()


# ------------------------------------------------------------------ rollout measurements

def rollout_run(model: RNGRN, L: float, n: int, seed: int) -> dict:
    """One simulate() rollout + morphology of the final field. Drops the field arrays."""
    rss_before = _rss_kib()
    r = simulate(model, L, n=n, seed=seed, integrator="etdrk4_rfft",
                 early_stop=True, check_every=CHECK_EVERY)
    rss_after = _rss_kib()
    morph = (phase_topology.measure(r["fields"][0])["morphology_measured"]
             if not r["blew_up"] else None)
    return dict(n=n, seed=seed, seconds=_j(r["seconds"]), nsteps_run=r["nsteps_run"],
                nsteps_planned=r["nsteps"], dt=_j(r["dt"]),
                stopped_reason=r["stopped_reason"], patterned=r["patterned"],
                amplitude=_j(r["amplitude"]), kstar=_j(r["kstar"]),
                sig_max=_j(r["sig_max"]), morphology=morph,
                rss_before_kib=rss_before, rss_after_kib=rss_after,
                rss_delta_kib=rss_after - rss_before)


def saturation_stats(rollouts: list[dict]) -> dict:
    """Per-grid mean/min/max of wall seconds and steps until stopped."""
    out = {}
    for n in sorted({r["n"] for r in rollouts}):
        rs = [r for r in rollouts if r["n"] == n]
        secs = [r["seconds"] for r in rs]
        steps = [r["nsteps_run"] for r in rs]
        out[str(n)] = dict(
            seconds_mean=float(np.mean(secs)), seconds_min=float(np.min(secs)),
            seconds_max=float(np.max(secs)), steps_mean=float(np.mean(steps)),
            steps_min=int(np.min(steps)), steps_max=int(np.max(steps)),
            stopped_reasons=sorted({r["stopped_reason"] for r in rs}))
    return out


def grid_fidelity(rollouts: list[dict], L: float) -> dict:
    """Does each cheaper grid reproduce the 512² k* (within one radial bin 2π/L) and
    morphology class, seed-by-seed? Built from the SAME rollouts as (1) — same L."""
    bin_width = 2 * np.pi / L
    ref = {r["seed"]: r for r in rollouts if r["n"] == 512}
    if not ref:
        raise RuntimeError("grid_fidelity needs the 512² reference runs")
    rows = []
    for r in sorted(rollouts, key=lambda r: (r["n"], r["seed"])):
        rr = ref[r["seed"]]
        dk = (abs(r["kstar"] - rr["kstar"])
              if (r["kstar"] is not None and rr["kstar"] is not None) else None)
        rows.append(dict(
            n=r["n"], seed=r["seed"], kstar=r["kstar"], kstar_512=rr["kstar"],
            dk=_j(dk), within_one_bin=(None if dk is None else bool(dk <= bin_width)),
            morphology=r["morphology"], morphology_512=rr["morphology"],
            class_match=(None if r["morphology"] is None or rr["morphology"] is None
                         else r["morphology"] == rr["morphology"])))
    return dict(bin_width=_j(bin_width), rows=rows)


# ------------------------------------------------------------------ amplitude trajectories

def amp_trajectory(model: RNGRN, n: int, L: float, seed: int) -> dict:
    """std-of-channel-0 every CHECK_EVERY steps, driving integrate_etdrk4_rfft in
    chunks, seeded exactly like simulate(): X = x* + 1e-2·rng.standard_normal.

    No early stop — the full horizon (40 rate times, capped at TRAJ_MAX_STEPS) is
    integrated so the saturated / decayed final amplitude is the attractor's, not a
    stopping rule's. dt and horizon are set exactly as simulate() sets them.
    """
    xs, converged = steady_state(model)
    if not converged:
        raise RuntimeError("steady_state did not converge for the trajectory run")
    xstar = xs.detach().numpy().reshape(model.N)
    D = model.D.detach().numpy()
    J = model.jacobian(xs, create_graph=False).detach().numpy()
    kg = np.linspace(1e-3, 2 * np.pi * (n // 2) / L, 2000)
    sigd = np.array([np.max(np.real(np.linalg.eigvals(J - kk**2 * np.diag(D))))
                     for kk in kg])
    sig_max = float(sigd.max())
    jac_rate = float(np.max(np.abs(np.linalg.eigvals(J))))
    dt = 0.2 / (jac_rate + 1e-9)
    T = 40.0 / max(abs(sig_max), 1e-12)
    nsteps = int(np.clip(T / dt, 200, TRAJ_MAX_STEPS))
    hit_cap = (T / dt) > TRAJ_MAX_STEPS

    rng = np.random.default_rng(seed)
    X = xstar[:, None, None] + 1e-2 * rng.standard_normal((model.N, n, n))
    reaction_np = _reaction_np_builder(model)
    amps = [float(X[0].std())]
    done = 0
    while done < nsteps:
        chunk = min(CHECK_EVERY, nsteps - done)
        X, blew_up = integrate_etdrk4_rfft(X, D, reaction_np, n, L, dt, chunk)
        done += chunk
        if blew_up:
            raise RuntimeError(f"trajectory run blew up at step {done} (seed {seed})")
        amps.append(float(X[0].std()))
    return dict(seed=seed, n=n, dt=_j(dt), nsteps=nsteps, hit_step_cap=hit_cap,
                sig_max=_j(sig_max), xstar0=_j(float(xstar[0])),
                final_amplitude=_j(amps[-1]), amps=[_j(a) for a in amps])


def ignition_floor(turing_trajs: list[dict], decay_trajs: list[dict],
                   fixture_xstar0: float) -> dict:
    """The floor proposal: geometric mean of the two final-amplitude ranges' nearest
    edges. Compared against the repo's pattern_floor (eval/rollout.py:234). Both are
    STATED; the decision is the parent's to record (SPEC §9.8: UNCALIBRATED at birth)."""
    t_fin = [t["final_amplitude"] for t in turing_trajs]
    d_fin = [t["final_amplitude"] for t in decay_trajs]
    t_lo, t_hi = float(np.min(t_fin)), float(np.max(t_fin))
    d_lo, d_hi = float(np.min(d_fin)), float(np.max(d_fin))
    gap_ok = t_lo > d_hi
    proposed = float(np.sqrt(t_lo * d_hi)) if gap_ok else None
    repo_floor = max(1e-3, 0.02 * abs(fixture_xstar0))
    return dict(turing_final_range=[_j(t_lo), _j(t_hi)],
                decayed_final_range=[_j(d_lo), _j(d_hi)],
                gap_ratio=_j(t_lo / d_hi if d_hi > 0 else np.inf),
                ranges_separated=gap_ok,
                proposed_floor_geomean=_j(proposed),
                repo_pattern_floor=_j(repo_floor),
                note=("proposed = sqrt(min Turing final · max non-Turing final); "
                      "None means the ranges OVERLAP and no floor can be proposed "
                      "from this data — that would be a finding, not a fallback"))


# ------------------------------------------------------------------ torch ETDRK4 prototype

def _phi_contour_torch(Lop: torch.Tensor, dt: float, M: int = 32):
    """Torch port of numerics._phi_contour: same contour integral, complex128.

    Lop is the real (N, n, n//2+1) half-spectrum linear operator; returns the six
    real-valued ETDRK4 coefficient tensors on Lop's device.
    """
    E = torch.exp(dt * Lop)
    E2 = torch.exp(dt * Lop / 2.0)
    theta = torch.pi * (torch.arange(1, M + 1, dtype=torch.float64,
                                     device=Lop.device) - 0.5) / M
    r = torch.exp(1j * theta)                                  # complex128 roots circle
    LR = (dt * Lop)[..., None].to(torch.complex128) + r
    Q = dt * torch.real(torch.mean((torch.exp(LR / 2.0) - 1.0) / LR, dim=-1))
    f1 = dt * torch.real(torch.mean(
        (-4.0 - LR + torch.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, dim=-1))
    f2 = dt * torch.real(torch.mean(
        (2.0 + LR + torch.exp(LR) * (-2.0 + LR)) / LR**3, dim=-1))
    f3 = dt * torch.real(torch.mean(
        (-4.0 - 3.0 * LR - LR**2 + torch.exp(LR) * (4.0 - LR)) / LR**3, dim=-1))
    return E, E2, Q, f1, f2, f3


def _torch_reaction_builder(model: RNGRN, device: torch.device):
    """Batched reaction for (B, N, n, n) fields — competitive form only (the fixture's).

    Same arithmetic as rollout._reaction_np_builder's competitive branch, in torch.
    nc1 is refused loudly rather than half-ported: this prototype only needs the
    fixture's form, and a silent wrong port would poison the equivalence check.
    """
    if model.form != "competitive":
        raise NotImplementedError(
            f"torch prototype supports form='competitive' only, got {model.form!r}")
    KA = model.KA.detach().to(device)
    KR = model.KR.detach().to(device)
    alpha = model.alpha.detach().to(device)
    beta = model.beta.detach().to(device)
    delta = model.delta.detach().to(device)
    n_h = model.n_hill

    def reaction_t(X: torch.Tensor) -> torch.Tensor:           # (B, N, n, n)
        xn = torch.clamp(X, min=0.0) ** n_h
        denom = 1.0 + torch.einsum("ij,bjxy->bixy", KA + KR, xn)
        prod = torch.einsum("ij,bjxy->bixy", alpha * KA, xn) / denom
        return beta[None, :, None, None] + prod - delta[None, :, None, None] * X

    return reaction_t


def torch_half_coeffs(D: np.ndarray, n: int, L: float, dt: float,
                      device: torch.device):
    """ETDRK4 half-spectrum coefficients as torch tensors (mirrors _cached_half_coeffs)."""
    k2 = torch.from_numpy(_spectral_k2_half(n, L)).to(device)
    Lop = torch.stack([-float(D[i]) * k2 for i in range(len(D))])
    return _phi_contour_torch(Lop, dt)


def integrate_etdrk4_rfft_torch(X0: torch.Tensor, reaction_t, n: int, dt: float,
                                nsteps: int, coeffs) -> tuple[torch.Tensor, bool]:
    """Line-for-line torch port of numerics.integrate_etdrk4_rfft, batched over a
    leading IC dimension: X0 is (B, N, n, n), coefficients broadcast over B.

    One theta only — batching over theta is M1 work, not this prototype's. The
    per-step isfinite check is kept from the numpy original; note it forces a device
    sync per step on CUDA (reported as-is, not hidden from the timing).
    """
    E, E2, Q, f1, f2, f3 = coeffs

    def Nfun(vhat):
        X = torch.fft.irfft2(vhat, s=(n, n), dim=(-2, -1))
        return torch.fft.rfft2(reaction_t(X), dim=(-2, -1))

    v = torch.fft.rfft2(X0, dim=(-2, -1))
    for _ in range(nsteps):
        Nv = Nfun(v)
        a = E2 * v + Q * Nv
        Na = Nfun(a)
        b = E2 * v + Q * Na
        Nb = Nfun(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = Nfun(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        if not torch.isfinite(v).all():
            return torch.fft.irfft2(v, s=(n, n), dim=(-2, -1)), True
    return torch.fft.irfft2(v, s=(n, n), dim=(-2, -1)), False


def torch_equivalence(model: RNGRN, n: int, L: float, dt: float,
                      xstar: np.ndarray) -> dict:
    """max |Δfield| numpy-vs-torch after 100 steps at n², B=1, same IC. Tol 1e-12."""
    rng = np.random.default_rng(42)
    X0 = xstar[:, None, None] + 1e-2 * rng.standard_normal((model.N, n, n))
    D = model.D.detach().numpy()
    Xnp, blew_np = integrate_etdrk4_rfft(X0.copy(), D, _reaction_np_builder(model),
                                         n, L, dt, 100)
    if blew_np:
        raise RuntimeError("numpy reference blew up in the equivalence check")
    with torch.no_grad():
        dev = torch.device("cpu")
        coeffs = torch_half_coeffs(D, n, L, dt, dev)
        Xt, blew_t = integrate_etdrk4_rfft_torch(
            torch.from_numpy(X0.copy())[None], _torch_reaction_builder(model, dev),
            n, dt, 100, coeffs)
    if blew_t:
        raise RuntimeError("torch port blew up in the equivalence check")
    delta = float(np.max(np.abs(Xt[0].numpy() - Xnp)))
    return dict(n=n, nsteps=100, max_abs_delta=_j(delta), tol=EQUIV_TOL,
                passed=bool(delta <= EQUIV_TOL))


def torch_step_timings(model: RNGRN, L: float, dt: float, xstar: np.ndarray,
                       grids=(96, 128), batches=(1, 8, 32), nsteps=50,
                       warmup=5) -> list[dict]:
    """Per-step cost of the batched torch integrator, CPU and (if available) CUDA.

    Coefficient construction is excluded (the numpy path caches it too). CUDA rows
    appear only when torch.cuda.is_available(); their timing includes the per-step
    isfinite sync inherited from the ported scheme."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    D = model.D.detach().numpy()
    rows = []
    with torch.no_grad():
        for dev in devices:
            reaction_t = _torch_reaction_builder(model, dev)
            for n in grids:
                coeffs = torch_half_coeffs(D, n, L, dt, dev)
                for B in batches:
                    rng = np.random.default_rng(7)
                    X0 = (xstar[:, None, None]
                          + 1e-2 * rng.standard_normal((B, model.N, n, n)))
                    Xt = torch.from_numpy(X0).to(dev)
                    _, blew = integrate_etdrk4_rfft_torch(Xt, reaction_t, n, dt,
                                                          warmup, coeffs)
                    if blew:
                        raise RuntimeError(f"torch warmup blew up (n={n}, B={B})")
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _, blew = integrate_etdrk4_rfft_torch(Xt, reaction_t, n, dt,
                                                          nsteps, coeffs)
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    dt_wall = time.perf_counter() - t0
                    if blew:
                        raise RuntimeError(f"torch timing run blew up (n={n}, B={B})")
                    rows.append(dict(device=dev.type, n=n, B=B, nsteps=nsteps,
                                     ms_per_step=_j(1e3 * dt_wall / nsteps),
                                     ms_per_step_per_member=_j(
                                         1e3 * dt_wall / nsteps / B)))
    return rows


# ------------------------------------------------------------------ drivers

def fixture_dt(model: RNGRN, xstar: np.ndarray) -> float:
    """dt exactly as simulate() sets it: 0.2 / spectral radius of the reaction Jacobian."""
    xs_t = torch.from_numpy(xstar)
    J = model.jacobian(xs_t, create_graph=False).detach().numpy()
    return 0.2 / (float(np.max(np.abs(np.linalg.eigvals(J)))) + 1e-9)


def run_smoke(model: RNGRN, L: float, k_lin: float, sig_max_lin: float,
              xstar: np.ndarray, report: dict) -> bool:
    print("== smoke: one 96² rollout, seed 0 ==")
    r = rollout_run(model, L, 96, 0)
    print(f"  {r['seconds']:.2f} s, {r['nsteps_run']} steps, "
          f"stopped={r['stopped_reason']}, patterned={r['patterned']}, "
          f"kstar={r['kstar']}, morphology={r['morphology']}")
    report["rollout_96_seed0"] = r

    print("== smoke: torch-vs-numpy equivalence, 100 steps at 96², B=1 ==")
    dt = fixture_dt(model, xstar)
    eq = torch_equivalence(model, 96, L, dt, xstar)
    print(f"  max |Δfield| = {eq['max_abs_delta']:.3e} "
          f"(tol {EQUIV_TOL:g}) -> {'PASS' if eq['passed'] else 'FAIL'}")
    report["torch_equivalence"] = eq

    print("== smoke: non-Turing contrast model sig_max ==")
    contrast = RNGRN(N=3, form="competitive", n_hill=2, seed=1)
    sig_max_nt, _ = nonturing_sig_max(contrast, 96, L)
    print(f"  sig_max = {sig_max_nt:.4f} (< 0 verified)")
    report["nonturing_sig_max"] = _j(sig_max_nt)

    ok = bool(eq["passed"] and r["patterned"] and not r["stopped_reason"] == "blew_up")
    report["passed"] = ok
    return ok


def run_full(model: RNGRN, L: float, k_lin: float, sig_max_lin: float,
             xstar: np.ndarray, report: dict) -> bool:
    # (1) rollout cost vs grid — ascending, so ru_maxrss (high-water) reads honestly
    print("== full: rollouts, grids ascending, 3 seeds each ==")
    rollouts = []
    for n in GRIDS:
        for seed in SEEDS:
            r = rollout_run(model, L, n, seed)
            rollouts.append(r)
            print(f"  n={n:4d} seed={seed}: {r['seconds']:7.2f} s, "
                  f"{r['nsteps_run']:6d} steps, stopped={r['stopped_reason']}, "
                  f"patterned={r['patterned']}, kstar={r['kstar']}, "
                  f"morph={r['morphology']}, rss_after={r['rss_after_kib']} KiB")
    report["rollouts"] = rollouts
    report["rss_profile_kib"] = [dict(n=r["n"], seed=r["seed"],
                                      rss_after_kib=r["rss_after_kib"])
                                 for r in rollouts]

    # (2) time-to-saturation distribution
    report["saturation_stats"] = saturation_stats(rollouts)
    print("== full: time-to-stop per grid ==")
    for n, s in report["saturation_stats"].items():
        print(f"  n={n}: seconds mean/min/max {s['seconds_mean']:.2f}/"
              f"{s['seconds_min']:.2f}/{s['seconds_max']:.2f}, steps "
              f"{s['steps_mean']:.0f}/{s['steps_min']}/{s['steps_max']} "
              f"({','.join(s['stopped_reasons'])})")

    # (3) ignition floor: Turing vs non-Turing amplitude trajectories at 96²
    print("== full: amplitude trajectories, Turing fixture at 96² ==")
    turing_trajs = [amp_trajectory(model, 96, L, s) for s in SEEDS]
    for t in turing_trajs:
        print(f"  seed={t['seed']}: final amp {t['final_amplitude']:.4g} "
              f"({t['nsteps']} steps, cap hit={t['hit_step_cap']})")
    contrast = RNGRN(N=3, form="competitive", n_hill=2, seed=1)
    sig_max_nt, _ = nonturing_sig_max(contrast, 96, L)
    print(f"== full: non-Turing contrast (sig_max={sig_max_nt:.4f} < 0 verified) ==")
    decay_trajs = [amp_trajectory(contrast, 96, L, s) for s in SEEDS]
    for t in decay_trajs:
        print(f"  seed={t['seed']}: final amp {t['final_amplitude']:.4g} "
              f"({t['nsteps']} steps, cap hit={t['hit_step_cap']})")
    floor = ignition_floor(turing_trajs, decay_trajs, float(xstar[0]))
    report.update(turing_trajectories=turing_trajs, decay_trajectories=decay_trajs,
                  nonturing_sig_max=_j(sig_max_nt), ignition_floor=floor)
    print(f"  Turing final range {floor['turing_final_range']}, decayed final range "
          f"{floor['decayed_final_range']}, proposed floor (geomean) "
          f"{floor['proposed_floor_geomean']}, repo pattern_floor "
          f"{floor['repo_pattern_floor']}")

    # (4) grid fidelity vs 512² at the same L
    fid = grid_fidelity(rollouts, L)
    report["grid_fidelity"] = fid
    print(f"== full: grid fidelity vs 512² (one bin = {fid['bin_width']:.4g}) ==")
    for row in fid["rows"]:
        print(f"  n={row['n']:4d} seed={row['seed']}: kstar={row['kstar']} vs "
              f"{row['kstar_512']} (dk={row['dk']}, one-bin={row['within_one_bin']}), "
              f"class {row['morphology']} vs {row['morphology_512']} "
              f"match={row['class_match']}")

    # (5) torch prototype: equivalence + per-step timings
    print("== full: torch ETDRK4 prototype ==")
    dt = fixture_dt(model, xstar)
    eq = torch_equivalence(model, 96, L, dt, xstar)
    print(f"  equivalence: max |Δfield| = {eq['max_abs_delta']:.3e} "
          f"-> {'PASS' if eq['passed'] else 'FAIL'}")
    timings = torch_step_timings(model, L, dt, xstar)
    for row in timings:
        print(f"  {row['device']} n={row['n']:3d} B={row['B']:2d}: "
              f"{row['ms_per_step']:.2f} ms/step "
              f"({row['ms_per_step_per_member']:.3f} ms/step/member)")
    report.update(torch_equivalence=eq, torch_timings=timings,
                  cuda_available=torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("  CUDA not available in this shell — CPU timings only (reported, "
              "not a failure)")

    ok = bool(eq["passed"] and all(r["stopped_reason"] != "blew_up"
                                   for r in rollouts))
    report["passed"] = ok
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", default=True,
                      help="one 96² rollout + equivalence + non-Turing check (default)")
    mode.add_argument("--full", action="store_true",
                      help="the complete D2 benchmark (run via guarded_run.sh)")
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    a = ap.parse_args()
    full = bool(a.full)

    os.makedirs(a.out, exist_ok=True)
    model = load_fixture(a.ckpt)
    k_lin, sig_max_lin, xstar = linear_kstar(model)
    L = PERIODS_PER_BOX * 2 * np.pi / k_lin
    print(f"fixture: k*_lin={k_lin:.5f}, sig_max={sig_max_lin:.5f}, "
          f"L={L:.3f} ({PERIODS_PER_BOX} periods/box), x*_0={xstar[0]:.5f}")

    report = dict(
        mode="full" if full else "smoke", ckpt=a.ckpt,
        kstar_lin=_j(k_lin), sig_max_lin=_j(sig_max_lin), L=_j(L),
        periods_per_box=PERIODS_PER_BOX, xstar=[_j(float(x)) for x in xstar],
        check_every=CHECK_EVERY, torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        rss_baseline_kib=_rss_kib())

    runner = run_full if full else run_smoke
    ok = runner(model, L, k_lin, sig_max_lin, xstar, report)

    out_path = os.path.join(a.out, f"results_{'full' if full else 'smoke'}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nD2 {'full' if full else 'smoke'} "
          f"{'PASS' if ok else 'FAIL'} -> {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
