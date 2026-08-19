"""r3_fd_ab.py — the FD A/B on BOTH gradient paths at the operating point (R3's hard gate).

`docs/REDESIGN_rngrn.md` §4.2 ("A/B discipline") requires that, before either gradient path
feeds a training run, a D1-style finite-difference check — **10 directions x the active loss
terms, tol 1e-4** — runs on *both* paths at the *same theta*, at the operating point §4.3
defines: the adaptive commensurate solve box, the n=96 training grid, training-geometry theta.
§7 makes it a HARD GATE: "a path that misses it does not feed a training run". §8 register
item 8's promotion rule is PRE-SPECIFIED and is not re-decided here — this script only
measures the two quantities that rule reads (FD-faithfulness on converged AND stalled members,
and measured cost per path).

FIREWALL (CLAUDE.md §5). This script opens NO `payload.h5` and reads NO `AnswerKey`. Its only
input is the tracked known-Turing checkpoint the D1/D2 diagnostics and T12 already use;
everything else is derived from that checkpoint's own model. It therefore owes no entry in
`tests/test_firewall.py::FORBIDDEN`. See `surrogate_frame` for how the loss TARGETS are made
without an observed frame, and what that does and does not license.

WHAT IS COMPARED. Both paths are scored with the SAME functional — the five ACTIVE spectral
loss terms of `losses/spectral.py` (`spec_shape`, `spec_aniso`, `spec_amp_mean`,
`spec_amp_fluct`, `real_moments`), evaluated through the BATCHED term functions on a
(1, N, n, n) stack, so the two paths differ in exactly one thing: how du/dtheta is obtained.
The remaining training terms (Turing hinges, priors, guard rails) are functions of theta alone
— they never touch u* — so both paths return identical gradients for them by construction and
they are not part of this gate.

  * `adjoint`   — `forward.BatchedPatternSolve`, the implicit-function-theorem path: one
                  minimal-norm LSMR adjoint solve per member per backward, then
                  -lam^T dF/dtheta.
  * `unrolled`  — `unrolled.unrolled_relax`, truncated backprop through `segment_steps`
                  differentiable ETDRK4 steps from a DETACHED saturated warm state
                  (D-R3-2's caller contract; `SEGMENT_STEPS_DEFAULT = 128`).

FD PROTOCOL (D-R3-1's, inherited unchanged — this repo does not need a third harness):
central differences over `EPS_SWEEP`, BEST eps per direction, rel = |d_fd - d_an| /
max(|d_fd|, |d_an|, 1e-300), worst over `--ndir` random unit directions in the full theta
space. `--ndir` defaults to **10**, the spec's number: D-R3-1 says T12's 6-direction curve
must NOT be cited as discharging this gate.

WHAT IS HELD FIXED across every finite difference, and why it must be. dt, the solve box, the
loss targets and the warm state are all DETACHED in both analytic paths (§4.3: geometry is not
differentiated; §4.2: the unrolled warm state is detached unconditionally). An FD that let any
of them move would be differencing a different function and would disagree for a reason that
is not an error. So: ONE dt, ONE box, ONE targets dict, ONE saturated field per arm, and for
the adjoint path ONE Newton warm start shared by the base point and every FD point — which is
also `tests/test_forward_solve.py`'s FD contract ("FD points: Newton-ONLY warm start") and
F-D1-1's reason for it (an ETDRK4 re-relax drifts the pinned phase).

THE TWO ARMS. §4.2 requires the check on CONVERGED and STALLED members separately, because the
adjoint path's premise is F(u*) = 0 and a stall is exactly where that premise fails.

  * `converged` — Newton reaches `PatternSolver.CONVERGENCE_TOL` (1e-9) on the commensurate
    box. The adjoint path's designed regime.
  * `stalled`   — Newton does NOT. `--stall-trials` theta perturbations per scale are searched
    for a member that genuinely stalls under the TRAINING condition (its own operating point,
    fresh relax, full Newton). If none is found in that budget the arm falls back to a FORCED
    stall — `forced_stall_warm_state`: the saturated field displaced by a fixed seeded
    perturbation and the Newton budget capped, both sized so the achieved residual lands
    inside `--stall-band` (default [1e-4, 1e-2], D-FFT-10's MEASURED stall band, true residual
    5.5e-4-5.7e-3), and both HELD FIXED across every FD point. The mechanism is recorded as
    `arms.stalled.stall_mechanism` and MUST be quoted with every stalled-arm number. A forced
    stall still exercises both paths' gradients at the right operating point against a field
    that misses the 1e-9 bar by a representative amount; what it does not establish is the
    FREQUENCY of stalls, which is the separate stall-rate measurement §4.3 asks for and this
    script does not attempt.

    Note also that the unrolled path is STRUCTURALLY insensitive to the distinction: it runs
    no Newton polish, so "converged" and "stalled" differ for it only through the warm state
    it is handed. That is a result, not a redundancy, and the report says so.

Note on the adjoint path and stalled members: `BatchedPatternSolver.solve_subset` REFUSES a
member whose Newton misses 1e-9 ("solve_failed") — it never hands one to the optimiser. So the
stalled arm cannot go through that state machine, and this script builds
`BatchedPatternSolve`'s payload directly, exactly as `solve_subset` does (forward.py:1465).
The CONVERGED arm asserts that the directly built payload reproduces `solve_subset`'s u*, so
the bypass is verified rather than assumed.

Run — CPU, threads pinned to 1 (the batched CPU relax is ~215x slower at 14 threads, Task 11):

    bash scripts/guarded_run.sh .venv/bin/python scripts/r3_fd_ab.py \\
        --out experiments/redesign_r3/fd_ab
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import resource
import sys
import time

# BEFORE numpy/torch. `torch.set_num_threads(1)` pins torch's intra-op pool but NOT the
# BLAS/FFT threads scipy and numpy use, and the CPU Newton/adjoint legs run through scipy's
# LSMR and numpy's FFT (`forward._minnorm_solve`). Without these the "threads pinned"
# timings below are pinned on one library and not the other — measured here at ~7.5 cores of
# CPU on a nominally single-threaded run. `setdefault` so an explicit environment still wins.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

import rngrn.solve_box as sb
from rngrn.eval.numerics import _spectral_k2
from rngrn.forward import (BatchedPatternSolve, BatchedPatternSolver, PatternSolver,
                           _half_k_grids, _member_F, newton_polish,
                           relax_to_pattern_torch, translation_modes)
from rngrn.losses.spectral import (SPECTRAL_TERM_KEYS, SpectralConfig, build_frame_targets,
                                   spectral_terms_batched)
from rngrn.losses.terms import steady_state
from rngrn.model import BatchedRNGRN, RNGRN, THETA_NAMES
from rngrn.unrolled import SEGMENT_STEPS_DEFAULT, unrolled_relax

REPO = pathlib.Path(__file__).resolve().parents[1]
CKPT = (REPO / "experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/"
        "checkpoints/model.pt")
EPS_SWEEP = (1e-3, 1e-4, 1e-5, 1e-6)
FD_TOL = 1e-4                     # spec §4.2 / §7, D1's acceptance tolerance, unchanged
NEWTON_ITER_FULL = 30             # `BatchedPatternSolver._newton_member`'s default, verbatim


# ------------------------------------------------------------------------ fixture & theta

def load_fixture(pert: np.ndarray | None = None) -> RNGRN:
    """The tracked known-Turing checkpoint, optionally displaced by a flat theta OFFSET.

    Re-read from disk per FD point (D1's own pattern) so a perturbation cannot accumulate
    across points.
    """
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    m = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    m.load_state_dict(ck["state_dict"])
    if pert is not None:
        off = 0
        with torch.no_grad():
            for nm in THETA_NAMES:
                p = getattr(m, nm)
                p.add_(torch.from_numpy(pert[off:off + p.numel()]).reshape(p.shape))
                off += p.numel()
    return m


def theta_dim(model) -> int:
    return sum(getattr(model, nm).numel() for nm in THETA_NAMES)


def theta_params(model) -> list:
    return [getattr(model, nm) for nm in THETA_NAMES]


def flat_grad(grads) -> np.ndarray:
    return torch.cat([g.reshape(-1) for g in grads]).detach().numpy()


def directions(dim: int, ndir: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(ndir):
        v = rng.standard_normal(dim)
        out.append(v / np.linalg.norm(v))
    return out


# ------------------------------------------------------------------------------- geometry

def half_k2(n: int, L: float) -> np.ndarray:
    KX, KY = _half_k_grids(n, L)
    return KX ** 2 + KY ** 2


def operating_point(model: RNGRN, kgrid_max: float = 10.0, kgrid_n: int = 2000) -> dict:
    """The §4.3 operating point for one model: commensurate box, dt, x*, |eig J|_max.

    k-hat is the dispersion argmax and is DETACHED — §4.3 requires solve-box geometry not to
    be differentiated, which is also why dt and the box are frozen across the FD sweep.
    `gamma` is computed with numpy exactly as `BatchedPatternSolver.solve_subset` computes
    its `rates`, so the two agree bit-for-bit and the solver-equivalence check below is a
    check on the payload, not on two different eigensolvers.
    """
    xs, ok = steady_state(model)
    if not ok:
        raise RuntimeError("steady state failed — no operating point exists")
    J = model.jacobian(xs, create_graph=False).detach()
    kg = torch.linspace(1e-3, kgrid_max, kgrid_n)
    sig = model.dispersion(xs, kg, J=J).detach()
    if not float(sig.max()) > 0.0:
        raise RuntimeError("model is not Turing-unstable — there is no pattern to solve")
    khat = float(kg[int(sig.argmax())])
    box = sb.geometry(khat)
    rate = float(np.abs(np.linalg.eigvals(J.cpu().numpy())).max())
    if not (np.isfinite(rate) and rate > 0.0):
        raise RuntimeError(f"|eig(J)|_max = {rate!r} — not a usable timescale")
    return dict(khat=khat, box=box, n=int(box["n"]), L=float(box["L_solve"]),
                dt=0.2 / rate, gamma=rate, xstar=xs.detach().numpy())


# ------------------------------------------------------------------ the loss functional

def surrogate_frame(op: dict, seed: int) -> np.ndarray:
    """A NEARBY-theta pattern on the SAME solve box, standing in for the observed frame.

    D1's protocol verbatim (`scripts/diag_fft_d1.py`: "fixed targets for the loss terms
    (from a NEARBY theta's pattern)"). This is what keeps the script clear of `payload.h5`
    — and it costs the gate nothing, because every entry of the targets dict is a CONSTANT
    with respect to theta on BOTH paths, so FD faithfulness cannot depend on where the
    constant came from. What it does NOT license is any statement about how well a model
    fits the real target: this script measures gradients, not fit. The §4.3
    target-interpolation measurement (`solve_box.interpolate_targets`, observed log-RAPS
    re-binned into k/k*_obs units) is a different measurement and is not exercised here.
    """
    rng = np.random.default_rng(seed + 991)
    m2 = load_fixture(0.01 * rng.standard_normal(theta_dim(load_fixture())))
    op2 = operating_point(m2)
    X = relax_to_pattern_torch(m2, op2["xstar"], op["n"], op["L"], op2["dt"], seed,
                               torch.device("cpu"))
    return X.detach().numpy()


def eval_terms(u_stack: torch.Tensor, targets: dict, cfg: SpectralConfig) -> dict:
    """The five ACTIVE spectral terms on a (1, N, n, n) stack -> {name: scalar tensor}."""
    vals, _parts = spectral_terms_batched(u_stack, targets, cfg, members=[0])
    return {k: vals[k][0] for k in SPECTRAL_TERM_KEYS}


# ------------------------------------------------------------------------ adjoint path

class AdjointPath:
    """`forward.BatchedPatternSolve` on a B=1 batched model, Newton-only from a FIXED warm
    field. `newton_iter` is the Newton budget: `NEWTON_ITER_FULL` for a converged or a
    genuinely stalled member, a smaller cap for a forced stall (module docstring)."""

    name = "adjoint"

    def __init__(self, op: dict, u_warm: torch.Tensor, newton_iter: int):
        self.n, self.L, self.gamma = op["n"], op["L"], op["gamma"]
        self.u_warm = u_warm.detach().clone()
        self.newton_iter = int(newton_iter)
        # `forward._build_solver_grids`'s geometry, built here rather than by standing up a
        # whole BatchedPatternSolver whose state machine this path deliberately bypasses.
        self.k2_full = _spectral_k2(self.n, self.L)
        KX, KY = _half_k_grids(self.n, self.L)
        self.k2h = torch.from_numpy(KX ** 2 + KY ** 2)
        self.kxy = (torch.from_numpy(KX), torch.from_numpy(KY))
        self.k2_dev = torch.from_numpy(self.k2_full)

    def _newton(self, bmodel: BatchedRNGRN):
        F_j = _member_F(bmodel, 0, self.n, self.L, self.k2h)
        modes_of = lambda uu: list(                                          # noqa: E731
            translation_modes(uu, self.n, self.L, kxy=self.kxy))
        return newton_polish(F_j, self.u_warm, modes_of, self.k2_full,
                             bmodel.D.detach().cpu().numpy()[0], self.gamma,
                             n_iter=self.newton_iter, k2_dev=self.k2_dev,
                             D_dev=bmodel.D.detach()[0])

    def solve(self, bmodel: BatchedRNGRN):
        """-> (u_stack (1, N, n, n) differentiable through the IFT, residual).

        The payload is built exactly as `BatchedPatternSolver.solve_subset` builds it
        (forward.py:1465-1472); it is constructed here rather than obtained from that method
        because `solve_subset` REFUSES a stalled member and this gate must measure one.
        """
        u, res = self._newton(bmodel)
        u_stack = u.detach()[None]
        idx = torch.as_tensor([0], dtype=torch.long)
        payload = dict(model=bmodel, idx=idx, members=[0], u_star=u_stack,
                       n=self.n, L=self.L, k2_full=self.k2_full,
                       D_np=bmodel.D.detach().cpu().numpy(),
                       gamma=np.asarray([self.gamma]),
                       k2h=self.k2h, k2_dev=self.k2_dev,
                       D_dev=bmodel.D.detach().index_select(0, idx))
        out = BatchedPatternSolve.apply(payload, *theta_params(bmodel))
        return out, float(res)

    def analytic(self, serial: RNGRN, targets, cfg):
        bmodel = BatchedRNGRN([serial])
        u, res = self.solve(bmodel)
        terms = eval_terms(u, targets, cfg)
        grads = {k: flat_grad(torch.autograd.grad(v, theta_params(bmodel),
                                                  retain_graph=True))
                 for k, v in terms.items()}
        return grads, {k: float(v.detach()) for k, v in terms.items()}, res

    def value(self, serial: RNGRN, targets, cfg):
        """One FD point: (values, residual, drift-from-warm-start) or None."""
        with torch.no_grad():
            u, res = self._newton(BatchedRNGRN([serial]))
            if not bool(torch.isfinite(u).all()):
                return None
            drift = float((u - self.u_warm).norm() / self.u_warm.norm())
            vals = {k: float(v) for k, v in eval_terms(u[None], targets, cfg).items()}
        return vals, float(res), drift


# ----------------------------------------------------------------------- unrolled path

class UnrolledPath:
    """`unrolled.unrolled_relax` from a FIXED detached saturated field. SERIAL MODEL ONLY —
    no batched twin of this path exists (`unrolled.py`'s docstring), so every number here is
    per-member and is reported as such."""

    name = "unrolled"

    def __init__(self, op: dict, X_sat: torch.Tensor, segment_steps: int):
        self.n, self.L, self.dt = op["n"], op["L"], op["dt"]
        self.X_sat = X_sat.detach().clone()
        self.segment_steps = int(segment_steps)

    def _run(self, serial: RNGRN, checkpoint_every):
        return unrolled_relax(serial, self.X_sat, self.n, self.L, self.dt,
                              segment_steps=self.segment_steps,
                              checkpoint_every=checkpoint_every)

    def analytic(self, serial: RNGRN, targets, cfg):
        u = self._run(serial, 1)
        terms = eval_terms(u[None], targets, cfg)
        grads = {k: flat_grad(torch.autograd.grad(v, theta_params(serial),
                                                  retain_graph=True))
                 for k, v in terms.items()}
        # NaN by construction: this path runs no Newton polish, so it has no residual to
        # report. That is the whole point of it — §4.2's "requires no convergence at all".
        return grads, {k: float(v.detach()) for k, v in terms.items()}, float("nan")

    def value(self, serial: RNGRN, targets, cfg):
        with torch.no_grad():
            try:
                u = self._run(serial, None)
            except RuntimeError:
                return None                     # the segment blew up at this FD point
            vals = {k: float(v) for k, v in eval_terms(u[None], targets, cfg).items()}
        return vals, float("nan"), float("nan")


# --------------------------------------------------------------------------- the FD sweep

def fd_sweep(path, offset: np.ndarray, dirs, targets, cfg) -> dict:
    """D-R3-1's protocol on one path: the analytic directional derivative from one backward
    pass, against central differences over `EPS_SWEEP`, best eps per direction."""
    t0 = time.perf_counter()
    grads, base_vals, base_res = path.analytic(load_fixture(offset), targets, cfg)
    analytic_seconds = time.perf_counter() - t0

    per_term = {k: {"per_dir": []} for k in SPECTRAL_TERM_KEYS}
    fd_t0 = time.perf_counter()
    for d, v in enumerate(dirs):
        pts = {}
        for eps in EPS_SWEEP:
            for sgn in (+1.0, -1.0):
                pts[(eps, sgn)] = path.value(load_fixture(offset + sgn * eps * v),
                                             targets, cfg)
        for name in SPECTRAL_TERM_KEYS:
            d_an = float(grads[name] @ v)
            best, per_eps = None, {}
            for eps in EPS_SWEEP:
                p, m = pts[(eps, +1.0)], pts[(eps, -1.0)]
                if p is None or m is None:
                    per_eps[str(eps)] = {"fd_failed": True}
                    continue
                d_fd = (p[0][name] - m[0][name]) / (2.0 * eps)
                rel = abs(d_fd - d_an) / max(abs(d_fd), abs(d_an), 1e-300)
                per_eps[str(eps)] = {"d_fd": d_fd, "rel_err": rel,
                                     "residual_plus": p[1], "drift_plus": p[2]}
                if best is None or rel < best[1]:
                    best = (eps, rel, d_fd)
            per_term[name]["per_dir"].append(
                {"dir": d, "d_analytic": d_an,
                 "best_eps": None if best is None else best[0],
                 "rel_err": None if best is None else best[1],
                 "d_fd": None if best is None else best[2],
                 "per_eps": per_eps})
        print(f"    [{path.name}] dir {d}: " + ", ".join(
            f"{k}={per_term[k]['per_dir'][-1]['rel_err']!r}"
            for k in SPECTRAL_TERM_KEYS), flush=True)
    fd_seconds = time.perf_counter() - fd_t0

    for name in SPECTRAL_TERM_KEYS:
        errs = [r["rel_err"] for r in per_term[name]["per_dir"] if r["rel_err"] is not None]
        per_term[name]["worst_rel_err"] = max(errs) if errs else None
        per_term[name]["median_rel_err"] = float(np.median(errs)) if errs else None
        per_term[name]["n_dirs_with_fd"] = len(errs)
        per_term[name]["passes_tol"] = (bool(errs) and max(errs) <= FD_TOL
                                        and len(errs) == len(dirs))
    worst = [per_term[k]["worst_rel_err"] for k in SPECTRAL_TERM_KEYS
             if per_term[k]["worst_rel_err"] is not None]
    return {"path": path.name, "base_values": base_vals, "base_residual": base_res,
            "analytic_seconds": analytic_seconds, "fd_seconds": fd_seconds,
            "per_term": per_term,
            "worst_rel_err_over_terms": max(worst) if worst else None,
            "passes_tol_all_terms": all(per_term[k]["passes_tol"]
                                        for k in SPECTRAL_TERM_KEYS)}


# ------------------------------------------------------------------------- the stall hunt

def hunt_stall(scales, trials_per_scale: int, seed: int, dim: int) -> dict:
    """Look for a theta that genuinely stalls under the TRAINING condition on ITS OWN
    commensurate box: fresh relax + full Newton, residual > `PatternSolver.CONVERGENCE_TOL`.

    Each candidate gets its own operating point (its own k-hat, box and dt) — a stall is a
    property of the member, and pinning a perturbed member to the base member's geometry
    would measure something else. Returns the first genuine stall found, or a record of the
    worst residual seen if there was none.
    """
    rng = np.random.default_rng(seed + 4242)
    best, tried = None, []
    for scale in scales:
        for t in range(trials_per_scale):
            v = rng.standard_normal(dim)
            pert = float(scale) * v / np.linalg.norm(v)
            m = load_fixture(pert)
            try:
                op = operating_point(m)
                X = relax_to_pattern_torch(m, op["xstar"], op["n"], op["L"], op["dt"],
                                           seed, torch.device("cpu"))
            except RuntimeError as exc:
                tried.append({"scale": float(scale), "trial": t,
                              "skipped": str(exc)[:120]})
                print(f"  stall hunt scale={scale} trial={t}: skipped ({exc})"[:160],
                      flush=True)
                continue
            F_fn = _member_F(BatchedRNGRN([m]), 0, op["n"], op["L"],
                             torch.from_numpy(half_k2(op["n"], op["L"])))
            u, res = newton_polish(F_fn, X, _modes_of(op["n"], op["L"]),
                                   _spectral_k2(op["n"], op["L"]),
                                   m.D.detach().numpy(), op["gamma"],
                                   n_iter=NEWTON_ITER_FULL)
            stalled = bool(res > PatternSolver.CONVERGENCE_TOL)
            tried.append({"scale": float(scale), "trial": t, "residual": float(res),
                          "stalled": stalled})
            print(f"  stall hunt scale={scale} trial={t}: residual {res:.3e} "
                  f"{'STALLED' if stalled else 'converged'}", flush=True)
            if best is None or res > best["residual"]:
                best = dict(scale=float(scale), trial=int(t), residual=float(res),
                            pert=pert.tolist(), stalled=stalled)
            if stalled:
                return dict(found=True, tried=tried, **best)
    return dict(found=False, tried=tried, **(best or {}))


def _modes_of(n: int, L: float):
    return lambda uu: list(translation_modes(uu, n, L))


def newton_residual_ladder(F_fn, X: torch.Tensor, k2_full, D_np, gamma, n: int, L: float,
                           max_iter: int) -> list:
    """Residual after each Newton iteration, from ONE run: `newton_polish` is deterministic
    and stateless given u, so continuing it one iteration at a time is the same trajectory
    a single `n_iter=k` call would take."""
    modes_of = _modes_of(n, L)
    u = X.detach().clone()
    ladder = [{"n_iter": 0, "residual": float(F_fn(u).detach().norm() / u.norm())}]
    for k in range(1, max_iter + 1):
        u, r = newton_polish(F_fn, u, modes_of, k2_full, D_np, gamma, n_iter=1)
        ladder.append({"n_iter": k, "residual": float(r)})
        if r <= PatternSolver.CONVERGENCE_TOL:
            break
    return ladder


def forced_stall_warm_state(F_fn, X_sat: torch.Tensor, k2_full, D_np, gamma, n: int,
                            L: float, band, seed: int, newton_cap: int = 1) -> dict:
    """Construct a member that MISSES the 1e-9 bar, when no theta in the hunt's budget
    stalls of its own accord.

    MECHANISM, stated plainly because every stalled-arm number has to be read with it: the
    saturated field is displaced by a FIXED seeded perturbation of relative size `eta`, and
    the Newton budget is capped at `newton_cap` iterations. `eta` is found by bisection in
    log10 (fixed budget of 8 probes) as the value whose achieved residual lands inside
    `band` — default 1e-4..1e-2, the order D-FFT-10 measured a stalled solve at — so the
    constructed member misses the 1e-9 bar by a representative amount rather than an
    invented one. Both the displacement and the cap are then HELD FIXED across every FD
    point of the arm, so the map theta -> u* is deterministic and differencing it is well
    defined. The displaced field is required to stay SATURATED (channel-0 amplitude within
    5% of the relaxed field's), which is D-R3-2's caller contract for the unrolled path —
    both arms are handed the SAME warm state, exactly as
    `recover._spectral_solve_with_stall_switch` hands the unrolled path the same field the
    Newton polish was given.

    WHY THIS IS NEEDED AT ALL is itself a measurement (see the report): on the commensurate
    box the ETDRK4 relax alone lands the base fixture at ~1.6e-11, three orders BELOW the
    bar, so there is nothing to truncate — an iteration cap on an already-converged warm
    state produces a converged member, not a stalled one.

    WHAT IT DOES NOT ESTABLISH: the FREQUENCY of stalls at the operating point. That is the
    separate stall-rate distribution §4.3 asks for and this script does not measure.
    """
    lo, hi = band
    modes_of = _modes_of(n, L)
    g = torch.Generator().manual_seed(seed + 8081)
    xi = torch.randn(X_sat.shape, generator=g, dtype=X_sat.dtype)
    xi = xi / xi.norm() * X_sat.norm()          # unit relative size, so eta IS the fraction
    amp_sat = float(X_sat[0].std())
    rows = []

    def probe(eta: float) -> dict:
        Xw = (X_sat + float(eta) * xi).detach()
        r_warm = float(F_fn(Xw).detach().norm() / Xw.norm())
        _u, r = newton_polish(F_fn, Xw, modes_of, k2_full, D_np, gamma, n_iter=newton_cap)
        amp = float(Xw[0].std())
        row = {"eta": float(eta), "warm_residual": r_warm, "residual": float(r),
               "warm_amplitude": amp, "amplitude_ratio": amp / amp_sat}
        rows.append(row)
        print(f"  forced-stall probe: eta={eta:.4g} warm_res={r_warm:.3e} -> residual "
              f"{r:.3e} (amp {amp / amp_sat:.4f} x saturated)", flush=True)
        return row

    # Bisection in log10(eta), fixed budget. Robust to whatever the local residual-vs-eta
    # scaling is on this fixture, which a fixed geometric ladder is not: the measured
    # relation here is steeply superlinear, so a ladder fine enough to bracket the band
    # spends most of its rungs far below it.
    lo_e, hi_e = -6.0, -1.0
    for _ in range(8):
        row = probe(10.0 ** (0.5 * (lo_e + hi_e)))
        if lo <= row["residual"] <= hi:
            break
        if row["residual"] < lo:
            lo_e = np.log10(row["eta"])
        else:
            hi_e = np.log10(row["eta"])
    usable = [r for r in rows
              if r["residual"] > PatternSolver.CONVERGENCE_TOL
              and abs(r["amplitude_ratio"] - 1.0) <= 0.05]
    if not usable:
        raise RuntimeError(
            "no displacement on the ladder produces a member that both misses the 1e-9 bar "
            "and stays saturated (amplitude within 5% of the relaxed field) — refusing to "
            "report a 'stalled' arm that is either converged or no longer a saturated "
            "pattern (D-R3-2's caller contract)")
    centre = 0.5 * (np.log10(lo) + np.log10(hi))
    pick = min(usable, key=lambda r: abs(np.log10(r["residual"]) - centre))
    Xw = (X_sat + pick["eta"] * xi).detach()
    return {"eta": pick["eta"], "newton_cap": int(newton_cap), "ladder": rows,
            "picked": pick, "band": [lo, hi],
            "in_band": bool(lo <= pick["residual"] <= hi),
            "X_warm": Xw}


# ------------------------------------------------------------------------ cost, isolated

def _cost_child(q, which: str, batch: int, segment_steps: int, threads: int) -> None:
    """One path's forward+backward in a FRESH process, so `ru_maxrss` is that path's own
    peak rather than a high-water mark inherited from whatever ran before it."""
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(int(threads))
    base = load_fixture()
    op = operating_point(base)
    n, L, dt, gamma = op["n"], op["L"], op["dt"], op["gamma"]
    cfg = SpectralConfig()
    targets = build_frame_targets(surrogate_frame(op, 0), L, op["khat"], cfg)
    X = relax_to_pattern_torch(base, op["xstar"], n, L, dt, 0,
                               torch.device("cpu")).detach()
    rng = np.random.default_rng(7)
    dim = theta_dim(base)
    members = [load_fixture(None if b == 0 else 1e-3 * rng.standard_normal(dim))
               for b in range(batch)]
    ops = [operating_point(m) for m in members]
    rss_setup = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    if which == "adjoint":
        u_warm = newton_polish(
            _member_F(BatchedRNGRN([base]), 0, n, L, torch.from_numpy(half_k2(n, L))),
            X, _modes_of(n, L), _spectral_k2(n, L), base.D.detach().numpy(), gamma,
            n_iter=NEWTON_ITER_FULL)[0].detach()
        solver = BatchedPatternSolver(BatchedRNGRN(members), n=n, L=L,
                                      seeds=list(range(batch)), warm_mode="newton")
        bmodel = solver.model
        xstar_batch = torch.stack([torch.from_numpy(o["xstar"]) for o in ops])

        def step():
            for b in range(batch):
                solver._warm[b] = u_warm.clone()
            u, ok_members, reasons = solver.solve_subset(list(range(batch)), xstar_batch)
            if u is None:
                raise RuntimeError(f"cost step solved nothing: {reasons}")
            vals = spectral_terms_batched(u, targets, cfg, members=ok_members)[0]
            torch.autograd.grad(sum(v.sum() for v in vals.values()), theta_params(bmodel))
            return len(ok_members)
    else:
        def step():
            for m in members:
                u = unrolled_relax(m, X, n, L, dt, segment_steps=segment_steps,
                                   checkpoint_every=1)
                vals = spectral_terms_batched(u[None], targets, cfg, members=[0])[0]
                torch.autograd.grad(sum(v.sum() for v in vals.values()), theta_params(m))
            return batch

    step()                                   # warm-up: caches, RAPS bins, first allocation
    reps, t0, solved = 3, time.perf_counter(), 0
    for _ in range(reps):
        solved = step()
    seconds = (time.perf_counter() - t0) / reps
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    q.put(dict(path=which, batch=int(batch), members_solved=int(solved),
               seconds_per_step=seconds, seconds_per_member=seconds / max(solved, 1),
               rss_after_setup_mb=rss_setup, rss_peak_mb=rss_peak,
               rss_step_increment_mb=rss_peak - rss_setup, reps=reps, threads=int(threads)))


def measure_cost(which: str, batch: int, segment_steps: int, threads: int) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_cost_child, args=(q, which, batch, segment_steps, threads))
    p.start()
    try:
        row = q.get(timeout=7200)
    finally:
        p.join()
    print(f"  cost [{which} B={batch}] {row['seconds_per_member']:.3f} s/member "
          f"({row['seconds_per_step']:.3f} s/step), peak RSS {row['rss_peak_mb']:.0f} MB "
          f"(+{row['rss_step_increment_mb']:.0f} MB over setup)", flush=True)
    return row


# -------------------------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="run directory; results land in <out>/results/fd_ab.json")
    ap.add_argument("--ndir", type=int, default=10, help="spec §4.2's number; do not lower")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segment-steps", type=int, default=SEGMENT_STEPS_DEFAULT)
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads; pinned at 1 for every timing (Task 11's 215x)")
    ap.add_argument("--stall-trials", type=int, default=1,
                    help="theta perturbations per scale in the genuine-stall hunt. The hunt "
                         "is BUDGET-LIMITED by design (each candidate costs a fresh relax "
                         "plus a full 30-iteration Newton); finding no genuine stall in it "
                         "is NOT a stall-rate measurement and must not be reported as one")
    ap.add_argument("--stall-scales", type=float, nargs="+", default=[0.1, 0.5, 2.0])
    ap.add_argument("--stall-band", type=float, nargs=2, default=[1e-4, 1e-2],
                    help="target residual band for a FORCED stall — D-FFT-10's MEASURED "
                         "stall band (5.5e-4-5.7e-3); used only if the hunt finds none")
    ap.add_argument("--stall-cap", type=int, default=1,
                    help="Newton budget for a FORCED stall. Must be >= 1: at 0 the map "
                         "theta -> u* is constant and has no gradient to check")
    ap.add_argument("--cost-batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--no-cost", action="store_true")
    a = ap.parse_args()

    if a.stall_cap < 1:
        raise ValueError("--stall-cap must be at least 1: with a budget of 0 the map "
                         "theta -> u* is constant and there is no gradient to check")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    t_start = time.perf_counter()

    base = load_fixture()
    op = operating_point(base)
    n, L, dt = op["n"], op["L"], op["dt"]
    dim = theta_dim(base)
    print(f"operating point: k-hat={op['khat']:.6g} box={op['box']} dt={dt:.6g} "
          f"theta dim={dim}", flush=True)

    cfg = SpectralConfig()
    frame = surrogate_frame(op, a.seed)
    targets = build_frame_targets(frame, L, op["khat"], cfg)
    print(f"targets built from the surrogate frame {frame.shape}", flush=True)

    t0 = time.perf_counter()
    X_sat = relax_to_pattern_torch(base, op["xstar"], n, L, dt, a.seed,
                                   torch.device("cpu")).detach()
    relax_seconds = time.perf_counter() - t0
    print(f"saturated in {relax_seconds:.1f}s, amplitude {float(X_sat[0].std()):.4g}",
          flush=True)

    F_base = _member_F(BatchedRNGRN([base]), 0, n, L, torch.from_numpy(half_k2(n, L)))
    k2_full = _spectral_k2(n, L)
    D_np = base.D.detach().numpy()
    relax_residual = float(F_base(X_sat).detach().norm() / X_sat.norm())
    t0 = time.perf_counter()
    ladder = newton_residual_ladder(F_base, X_sat, k2_full, D_np, op["gamma"], n, L,
                                    NEWTON_ITER_FULL)
    newton_seconds = time.perf_counter() - t0
    res_base = ladder[-1]["residual"]
    print(f"relax residual {relax_residual:.3e}; Newton ladder in {newton_seconds:.1f}s -> "
          f"{res_base:.3e} in {ladder[-1]['n_iter']} iters "
          f"(bar {PatternSolver.CONVERGENCE_TOL:g})", flush=True)
    if res_base > PatternSolver.CONVERGENCE_TOL:
        raise RuntimeError(
            f"the base fixture does not converge on the commensurate box (residual "
            f"{res_base:.3e}) — there is no CONVERGED arm to measure")

    dirs = directions(dim, a.ndir, a.seed)
    zero = np.zeros(dim)

    # -- solver equivalence: the directly built payload vs the shipped state machine -------
    solver = BatchedPatternSolver(BatchedRNGRN([load_fixture()]), n=n, L=L, seeds=[a.seed],
                                  warm_mode="newton")
    solver._warm[0] = X_sat.clone()
    u_ss, _ok, reasons = solver.solve_subset([0], torch.from_numpy(op["xstar"])[None])
    if u_ss is None:
        raise RuntimeError(f"BatchedPatternSolver refused the converged fixture: {reasons}")
    u_direct, _res_direct = AdjointPath(op, X_sat, NEWTON_ITER_FULL).solve(
        BatchedRNGRN([load_fixture()]))
    payload_gap = float((u_direct.detach() - u_ss.detach()).abs().max()
                        / u_ss.detach().abs().max())
    print(f"solver equivalence: max|u_direct - u_solve_subset| / max|u| = "
          f"{payload_gap:.3e}", flush=True)
    if payload_gap > 1e-12:
        raise RuntimeError(
            f"the directly built BatchedPatternSolve payload does not reproduce "
            f"BatchedPatternSolver.solve_subset's u* (relative gap {payload_gap:.3e}) — the "
            f"stalled arm's bypass is unverified, so its numbers are not comparable")

    arms: dict = {}

    # ------------------------------------------------------------------ CONVERGED arm ---
    print("\n=== CONVERGED arm (base theta, Newton to the 1e-9 bar) ===", flush=True)
    arms["converged"] = {
        "theta": "base fixture",
        "stall_mechanism": None,
        "newton_iter": NEWTON_ITER_FULL,
        "newton_residual": res_base,
        "converged": True,
        "operating_point": {"khat": op["khat"], "box": op["box"], "dt": dt},
        "paths": {p.name: fd_sweep(p, zero, dirs, targets, cfg)
                  for p in (AdjointPath(op, X_sat, NEWTON_ITER_FULL),
                            UnrolledPath(op, X_sat, a.segment_steps))},
    }

    # -------------------------------------------------------------------- STALLED arm ---
    print("\n=== STALLED arm ===", flush=True)
    hunt = hunt_stall(a.stall_scales, a.stall_trials, a.seed, dim)
    forced_probe = None
    if hunt.get("found"):
        offset = np.asarray(hunt["pert"])
        m_st = load_fixture(offset)
        op_use = operating_point(m_st)
        X_use = relax_to_pattern_torch(m_st, op_use["xstar"], op_use["n"], op_use["L"],
                                       op_use["dt"], a.seed,
                                       torch.device("cpu")).detach()
        st_newton_iter, mechanism = NEWTON_ITER_FULL, "genuine"
    else:
        # FORCED — the mechanism, the reason it is needed, and what it does not establish
        # are all in `forced_stall_warm_state`'s docstring. Same theta, same box; the arm
        # differs from the converged one only in the warm state's displacement and the
        # Newton budget, which is what makes the two a controlled contrast on exactly the
        # property the adjoint path's premise turns on: F(u*) = 0.
        forced = forced_stall_warm_state(F_base, X_sat, k2_full, D_np, op["gamma"], n, L,
                                         a.stall_band, a.seed, newton_cap=a.stall_cap)
        X_use = forced.pop("X_warm")
        st_newton_iter, mechanism = forced["newton_cap"], "forced_displaced_warm_state"
        offset, op_use = zero, op
        forced_probe = forced
        print(f"  forced stall: eta={forced['eta']:.3g} cap={st_newton_iter} -> residual "
              f"{forced['picked']['residual']:.3e} (band {a.stall_band[0]:g}-"
              f"{a.stall_band[1]:g}, {'inside' if forced['in_band'] else 'OUTSIDE'})",
              flush=True)

    st_adj = AdjointPath(op_use, X_use, st_newton_iter)
    _u_st, res_st = st_adj.solve(BatchedRNGRN([load_fixture(offset)]))
    print(f"stalled arm: mechanism={mechanism} newton_iter={st_newton_iter} "
          f"residual={res_st:.3e} (bar {PatternSolver.CONVERGENCE_TOL:g})", flush=True)
    if res_st <= PatternSolver.CONVERGENCE_TOL:
        raise RuntimeError(
            f"the 'stalled' arm converged (residual {res_st:.3e}) — it is not a stalled "
            f"member and must not be reported as one")
    arms["stalled"] = {
        "theta": "base fixture" if mechanism.startswith("forced") else "perturbed fixture",
        "theta_offset_norm": float(np.linalg.norm(offset)),
        "stall_mechanism": mechanism,
        "hunt": hunt,
        "forced_probe": forced_probe,
        "newton_iter": st_newton_iter,
        "newton_residual": res_st,
        "converged": False,
        "operating_point": {"khat": op_use["khat"], "box": op_use["box"],
                            "dt": op_use["dt"]},
        "paths": {p.name: fd_sweep(p, offset, dirs, targets, cfg)
                  for p in (st_adj, UnrolledPath(op_use, X_use, a.segment_steps))},
    }

    # -------------------------------------------------------------------------- cost ---
    costs = []
    if not a.no_cost:
        print("\n=== cost (threads pinned, isolated processes) ===", flush=True)
        for b in a.cost_batches:
            costs.append(measure_cost("adjoint", b, a.segment_steps, a.threads))
            costs.append(measure_cost("unrolled", b, a.segment_steps, a.threads))

    # ------------------------------------------------------------------------- output ---
    summary = {arm: {p: {"worst_rel_err_over_terms": r["worst_rel_err_over_terms"],
                         "passes_tol_all_terms": r["passes_tol_all_terms"],
                         "worst_per_term": {k: r["per_term"][k]["worst_rel_err"]
                                            for k in SPECTRAL_TERM_KEYS}}
                     for p, r in row["paths"].items()}
               for arm, row in arms.items()}

    out_dir = pathlib.Path(a.out)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    payload = {
        "what": "FD A/B on both gradient paths at the operating point "
                "(docs/REDESIGN_rngrn.md §4.2 A/B discipline, §7 hard gate; "
                "PLAN_redesign_R3 Task 14)",
        "gate": {"tolerance": FD_TOL,
                 "criterion": "worst relative error over the active spectral loss terms and "
                              "all directions, per path, per arm",
                 "note": "register item 8's promotion rule is PRE-SPECIFIED and is NOT "
                         "decided here; this file reports the quantities it reads"},
        "method": {
            "checkpoint": str(CKPT.relative_to(REPO)),
            "operating_point": {"khat": op["khat"], "box": op["box"], "dt": dt,
                                "n": n, "L_solve": L},
            "terms": list(SPECTRAL_TERM_KEYS),
            "terms_note": "the ACTIVE spectral terms — the only terms whose gradient flows "
                          "through the forward solve. Turing hinges and priors are functions "
                          "of theta alone and are identical on both paths by construction.",
            "targets": "surrogate frame from a NEARBY theta's pattern on the same box "
                       "(scripts/diag_fft_d1.py's protocol) — no payload.h5, no AnswerKey",
            "eps_sweep": list(EPS_SWEEP), "ndir": a.ndir, "seed": a.seed,
            "segment_steps": a.segment_steps, "checkpoint_every_analytic": 1,
            "newton_iter_full": NEWTON_ITER_FULL,
            "convergence_bar": PatternSolver.CONVERGENCE_TOL,
            "relax_residual_of_saturated_field": relax_residual,
            "newton_residual_ladder_base": ladder,
            "dt_box_targets_warm_state_held_fixed_across_fd": True,
            "solver_equivalence_relative_gap": payload_gap,
            "device": "cpu", "threads": a.threads, "torch": torch.__version__,
            "argv": list(sys.argv),
            "relax_seconds": relax_seconds, "newton_ladder_seconds": newton_seconds,
        },
        "summary": summary,
        "arms": arms,
        "cost": costs,
        "total_seconds": time.perf_counter() - t_start,
    }
    out = out_dir / "results" / "fd_ab.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
