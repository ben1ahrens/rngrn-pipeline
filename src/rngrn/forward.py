"""forward.py — the gradient-free pattern forward solve and its IFT/adjoint backward.

RECOVERY-SIDE. Ported from ``scripts/diag_fft_d1.py`` (D1-verified 2026-08-12: worst
FD-vs-IFT relative error 6.4e-8 against tolerance 1e-4, 10 directions x 5 terms at 96^2 —
`experiments/diag_fft/d1/results.json`). The port keeps the verified constants verbatim.
Two D1 findings are load-bearing and bind this module:

- **D-FFT-10 / F-D1-3**: the adjoint system A^T lam = dL/du at the patterned state is
  solved for the MINIMAL-NORM least-squares solution — right-preconditioned LSMR
  (M = (gamma + D k^2)^-1, symmetric) with iterative refinement on the TRUE residual —
  never by a Krylov method whose space is projected off the translation modes. The
  projected variant reported residual 1e-13 while its true residual stalled at
  5.5e-4-5.7e-3, biasing every gradient by exactly that order.
- **F-D1-2**: grid pinning makes the translations NEAR-null, not null, so Newton polish
  carries an explicit 2x2 translation-subspace correction per step; without it the solve
  stalls at ~2.4e-8 instead of reaching <=1e-11.

Cost caveat, stated honestly: `PatternSolver`'s per-step cost at training grid sizes is
UNMEASURED — D1/D2 measured single solves at diagnostic sizes (64^2-96^2 relax + Newton;
0.9-1.7 s/field rollouts at eval sizes), not amortised per-Adam-step cost under warm
starting. A relax step-budget cap is an owner decision before any calibration run.

Nothing here reads the observed frame or any answer-key quantity: the solver consumes
only the model's own parameters and grid geometry supplied by the caller.
"""
from __future__ import annotations

import numpy as np
import torch

from .losses.terms import steady_state
from .eval.numerics import integrate_etdrk4_rfft, _spectral_k2
from .eval.rollout import _reaction_np_builder
from . import observables as obs
from .model import RNGRN, THETA_NAMES

# Sanity tripwire on the backward adjoint solve, NOT a solver knob: the refinement loop
# targets 1e-10 and D1 measured 2.3e-12-6.1e-12 at the committed 96^2 record
# (experiments/diag_fft/d1/results.json adjoint_residuals; 1.4-3.6e-12 at the 64^2
# probe); the measured BIASED regime started at 5.5e-4. A residual above this means the
# gradient is untrustworthy — fail loud rather than hand Adam a silently biased gradient
# (the exact D1 failure mode).
_ADJOINT_RESIDUAL_TRIPWIRE = 1e-8


# ------------------------------------------------------------------ field helpers

def reaction_fields(model: RNGRN, u: torch.Tensor) -> torch.Tensor:
    """Pointwise reaction applied to a (N, n, n) field via the model's own kernel."""
    x = u.permute(1, 2, 0)                     # (n, n, N)
    return model.reaction(x).permute(2, 0, 1)  # (N, n, n)


def make_spatial_F(model: RNGRN, n: int, L: float):
    """F(u) = D lap u + f(u), torch end-to-end (differentiable w.r.t. u AND theta).

    The Laplacian is spectral (rfft2), matching the training-side design: no stencil.
    The same closure serves the backward dF/dtheta vjp — it must stay differentiable
    through ``model``'s parameter properties.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    ky = 2.0 * np.pi * np.fft.rfftfreq(n, d=L / n)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2h = torch.from_numpy(KX**2 + KY**2)      # (n, n//2+1), rfft2 layout

    def spatial_F(u: torch.Tensor) -> torch.Tensor:
        lap = torch.fft.irfft2(-k2h * torch.fft.rfft2(u), s=(n, n))
        return model.D[:, None, None] * lap + reaction_fields(model, u)

    return spatial_F


def translation_modes(u: torch.Tensor, n: int, L: float) -> tuple[torch.Tensor, torch.Tensor]:
    """du/dx, du/dy of a (N, n, n) periodic field, spectrally (the exact zero modes of A)."""
    kx = torch.from_numpy(2.0 * np.pi * np.fft.fftfreq(n, d=L / n))
    ky = torch.from_numpy(2.0 * np.pi * np.fft.rfftfreq(n, d=L / n))
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    uh = torch.fft.rfft2(u)
    tx = torch.fft.irfft2(1j * KX * uh, s=(n, n))
    ty = torch.fft.irfft2(1j * KY * uh, s=(n, n))
    return tx, ty


# ------------------------------------------------------------------ linear algebra

def _minnorm_solve(apply_fwd, apply_adj, Mmv, b: np.ndarray, size: int,
                   tol: float = 1e-10, maxiter: int = 20000) -> tuple[np.ndarray, float]:
    """Minimal-norm least-squares solution of (apply_fwd) x = b by right-preconditioned
    LSMR (x = M y; matvec = apply_fwd o M, rmatvec = M o apply_adj with M symmetric) plus
    iterative refinement on the exact residual. Singular-safe: near-null components are
    resolved, not projected away (finding F-D1-3)."""
    from scipy.sparse.linalg import LinearOperator, lsmr

    Op = LinearOperator((size, size), matvec=lambda y: apply_fwd(Mmv(y)),
                        rmatvec=lambda z: Mmv(apply_adj(z)))
    bn = max(np.linalg.norm(b), 1e-300)
    x = np.zeros(size)
    res_prev = np.inf
    for _ in range(6):
        r = b - apply_fwd(x)
        res = float(np.linalg.norm(r) / bn)
        if res < tol or res > 0.9 * res_prev:
            break
        res_prev = res
        y = lsmr(Op, r, atol=1e-13, btol=1e-13, maxiter=maxiter)[0]
        x = x + Mmv(y)
    else:
        res = float(np.linalg.norm(b - apply_fwd(x)) / bn)
    return x, res


def solve_adjoint(F_fn, u_star: torch.Tensor, rhs: torch.Tensor, k2_full: np.ndarray,
                  D: np.ndarray, gamma: float, tol: float = 1e-10,
                  maxiter: int = 20000) -> tuple[torch.Tensor, float]:
    """MINIMAL-NORM solution of A^T lam = rhs via right-preconditioned LSMR with
    true-residual refinement (D-FFT-10; the WHY-not-projected-GMRES record lives there
    and in the module docstring). Right preconditioning by the SYMMETRIC Fourier-diagonal
    M = (gamma + D k^2)^-1: solve (A^T M) y = rhs, lam = M y; the transpose operator is
    (A^T M)^T = M A, available matrix-free."""
    shape_field = tuple(u_star.shape)
    Nsp, n, _ = shape_field
    u0 = u_star.detach()
    _, vjp_fn = torch.func.vjp(F_fn, u0)

    def AT(x: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
        (out,) = vjp_fn(v.reshape(shape_field))
        return out.detach().reshape(-1).numpy()

    def Amv(x: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
        _, out = torch.func.jvp(F_fn, (u0,), (v.reshape(shape_field),))
        return out.detach().reshape(-1).numpy()

    def Mmv(x: np.ndarray) -> np.ndarray:
        v = x.reshape(Nsp, n, n)
        vh = np.fft.fft2(v, axes=(1, 2))
        vh = vh / (gamma + D[:, None, None] * k2_full[None])
        return np.real(np.fft.ifft2(vh, axes=(1, 2))).reshape(-1)

    size = int(np.prod(shape_field))
    g = rhs.detach().reshape(-1).numpy()
    lam, res_full = _minnorm_solve(AT, Amv, Mmv, g, size, tol=tol, maxiter=maxiter)
    return torch.from_numpy(lam), res_full


def newton_polish(F_fn, u: torch.Tensor, modes_of, k2_full, D, gamma,
                  n_iter: int = 30, tol: float = 1e-11) -> tuple[torch.Tensor, float]:
    """Drive ||F(u)|| toward zero with DAMPED Newton steps: minimal-norm LSMR bulk step
    (unprojected — finding F-D1-3) plus an explicit 2-D translation-subspace correction
    for the pinned phase (finding F-D1-2). Damping (step halving when ||F|| does not
    decrease) was measured to matter for D1's eps=1e-3 FD points (large start residual);
    the same overshoot mechanism plausibly applies to far-moved warm starts, unmeasured."""
    u = u.detach().clone()
    scale = float(u.norm())
    r = float(F_fn(u).detach().norm()) / max(scale, 1e-300)
    Nsp, n, _ = u.shape

    def precon(x: np.ndarray) -> np.ndarray:
        v = x.reshape(Nsp, n, n)
        vh = np.fft.fft2(v, axes=(1, 2))
        vh = vh / (gamma + D[:, None, None] * k2_full[None])
        return np.real(np.fft.ifft2(vh, axes=(1, 2))).reshape(-1)

    for _ in range(n_iter):
        if r < tol:
            break
        u0 = u.detach()
        _, vjp_fn = torch.func.vjp(F_fn, u0)

        def Amv(x: np.ndarray) -> np.ndarray:
            v = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
            _, out = torch.func.jvp(F_fn, (u0,), (v.reshape(u.shape),))
            return out.detach().reshape(-1).numpy()

        def ATmv(x: np.ndarray) -> np.ndarray:
            v = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
            (out,) = vjp_fn(v.reshape(u.shape))
            return out.detach().reshape(-1).numpy()

        Fv = F_fn(u).detach()
        # Bulk step: minimal-norm LSMR on the UNPROJECTED system (finding F-D1-3 — a
        # projected inner solve hides near-null error and floors Newton at ~1e-9).
        dx, _ = _minnorm_solve(Amv, ATmv, precon, -Fv.reshape(-1).numpy(),
                               u.numel(), tol=1e-12, maxiter=8000)
        step = torch.from_numpy(dx).reshape(u.shape)
        # TRANSLATION-SUBSPACE CORRECTION (finding F-D1-2). Grid pinning makes the
        # translations near-null, not null; a theta-perturbed fixed point sits at a
        # slightly different phase and the phase part of the step is best solved
        # explicitly in the 2-D mode subspace: G c = <t_i, F - A step>, step += -t.c.
        modes = modes_of(u)
        Q = []
        for m in modes:
            q = m.detach().reshape(-1)
            for p in Q:
                q = q - (p @ q) * p
            if float(q.norm()) > 1e-12:
                Q.append(q / q.norm())
        if len(Q) == 2:
            resid_after = Fv.reshape(-1) + torch.from_numpy(Amv(dx))
            At = []
            for q in Q:
                _, Aq = torch.func.jvp(F_fn, (u0,), (q.reshape(u.shape),))
                At.append(Aq.detach().reshape(-1))
            G = torch.stack([torch.stack([q @ a for a in At]) for q in Q])
            rhs = torch.stack([q @ resid_after for q in Q])
            if float(torch.linalg.cond(G)) < 1e8:
                c = torch.linalg.solve(G, rhs)
                step = step - (c[0] * Q[0] + c[1] * Q[1]).reshape(u.shape)
        lam_damp = 1.0
        for _half in range(6):
            u_try = u + lam_damp * step
            r_try = float(F_fn(u_try).detach().norm()) / max(scale, 1e-300)
            if r_try < r:
                break
            lam_damp *= 0.5
        else:
            break                     # no descent at any damping — stalled, report r
        u, r = u_try, r_try
    return u, r


# ------------------------------------------------------------------ relax-to-pattern

def relax_to_pattern(model: RNGRN, xstar: np.ndarray, n: int, L: float, dt: float,
                     seed: int, noise: float = 1e-2, chunk: int = 500,
                     max_chunks: int = 400, flat_tol: float = 1e-4) -> np.ndarray:
    """ETDRK4 from x* + noise until channel-0 amplitude AND k* are flat to flat_tol over
    5 consecutive chunks. Deterministic for fixed seed. Raises on blow-up or
    non-saturation — the CALLER decides whether that is fatal (fresh solve inside
    training legitimately fails when theta wanders; a diagnostic fixture must not)."""
    D = model.D.detach().numpy()
    reaction_np = _reaction_np_builder(model)
    rng = np.random.default_rng(seed)
    X = xstar[:, None, None] + noise * rng.standard_normal((model.N, n, n))
    amps: list[float] = []
    ks: list[float] = []
    for _ in range(max_chunks):
        X, blew = integrate_etdrk4_rfft(X, D, reaction_np, n, L, dt, chunk)
        if blew:
            raise RuntimeError("forward relax blew up")
        amps.append(float(X[0].std()))
        ks.append(float(obs.kstar_of(X[0], L=L)))
        if len(amps) >= 5:
            a = np.array(amps[-5:]); k = np.array(ks[-5:])
            if ((a.max() - a.min()) / a.mean() < flat_tol
                    and (k.max() - k.min()) / k.mean() < flat_tol):
                return X
    raise RuntimeError(f"no saturation in {max_chunks * chunk} steps")


# ------------------------------------------------------------------ the autograd bridge

class PatternSolve(torch.autograd.Function):
    """u*(theta) as an autograd node: forward returns the DETACHED solved pattern; backward
    is the IFT chain — ONE minimal-norm adjoint solve on the incoming cotangent (autograd
    pre-accumulates every consuming term's dL/du at this node, so one adjoint per
    backward() by construction), then dL/dtheta = -lam^T dF/dtheta via the differentiable
    F closure. Exactly the chain D1 verified against finite differences."""

    @staticmethod
    def forward(ctx, payload, theta_s, theta_g, theta_alpha, theta_delta, theta_beta,
                theta_D):
        ctx.payload = payload
        return payload["u_star"].clone()

    @staticmethod
    def backward(ctx, grad_out):
        p = ctx.payload
        model, u_star = p["model"], p["u_star"]
        n, L = p["n"], p["L"]
        # backward() runs under no-grad; both the vjp inside solve_adjoint and the
        # dF/dtheta graph below need grad mode on.
        with torch.enable_grad():
            F_fn = make_spatial_F(model, n, L)
            lam, adj_res = solve_adjoint(F_fn, u_star, grad_out.contiguous(),
                                         p["k2_full"], p["D_np"], p["gamma"])
            if adj_res > _ADJOINT_RESIDUAL_TRIPWIRE:
                raise RuntimeError(
                    f"adjoint solve stalled at relative residual {adj_res:.2e} "
                    f"(tripwire {_ADJOINT_RESIDUAL_TRIPWIRE:g}) — the gradient would be "
                    f"biased by that order (D-FFT-10). Refusing to hand it to the "
                    f"optimiser.")
            p["last_adjoint_residual"] = adj_res
            params = [getattr(model, nm) for nm in THETA_NAMES]
            Fv = F_fn(u_star)      # u_star fixed; graph runs through theta only
            g_theta = torch.autograd.grad(Fv, params,
                                          grad_outputs=-lam.reshape(u_star.shape))
        return (None,) + tuple(g_theta)


class PatternSolver:
    """Per-restart owner of the forward-solve state: grid geometry, warm start, and the
    D1 solver knobs. The caller (recovery loop) decides WHETHER to solve (ignition,
    sig_max_pos, is not this class's concern); this class owns only solve success and
    patterned-ness, with the approved skip/fail semantics:

      solve() -> (u, "ok")               patterned, differentiable-through-IFT tensor
      solve() -> (None, "solve_failed")  relax blew up / no saturation / Newton stalled
                                         (warm state cleared)
      solve() -> (None, "not_patterned") converged but channel-0 amplitude at or below
                                         pattern_floor = max(1e-3, 0.02*|x*_0|) — the
                                         eval/rollout.py floor, kept by D-FFT-9 closure 2
                                         (warm state cleared: a homogeneous field would
                                         re-converge homogeneous forever as a warm start)

    Warm-start policy (D1): Newton-ONLY from the previous u* — an ETDRK4 re-relax would
    drift the pinned phase. A failed warm Newton falls back to a fresh relax before
    reporting failure. dt = 0.2/|eig(J)|_max and gamma = |eig(J)|_max are recomputed per
    solve from the current theta, as in the diagnostic.
    """

    #: Newton convergence requirement on ||F||/||u|| (D1 verbatim).
    CONVERGENCE_TOL = 1e-9

    def __init__(self, model: RNGRN, n: int, L: float, seed: int,
                 noise: float = 1e-2, chunk: int = 500, max_chunks: int = 400,
                 flat_tol: float = 1e-4):
        self.model = model
        self.n = int(n)
        self.L = float(L)
        self.seed = int(seed)
        self.noise = noise
        self.chunk = chunk
        self.max_chunks = max_chunks
        self.flat_tol = flat_tol
        self.k2_full = _spectral_k2(self.n, self.L)
        self._warm: torch.Tensor | None = None
        self.last_residual: float = float("nan")
        self.last_reason: str = "never_solved"

    # -- internals -------------------------------------------------------------------

    def _newton(self, u0: torch.Tensor, F_fn, gamma: float, D_np: np.ndarray):
        modes_of = lambda uu: list(translation_modes(uu, self.n, self.L))  # noqa: E731
        return newton_polish(F_fn, u0, modes_of, self.k2_full, D_np, gamma)

    def _finish(self, u: torch.Tensor | None, reason: str):
        self.last_reason = reason
        if reason != "ok":
            self._warm = None
            return None, reason
        return u, reason

    # -- the state machine -----------------------------------------------------------

    def solve(self) -> tuple[torch.Tensor | None, str]:
        model = self.model
        xs, ok = steady_state(model)
        if not ok:
            return self._finish(None, "solve_failed")
        xstar = xs.detach().numpy()
        J = model.jacobian(xs, create_graph=False).detach()
        jac_rate = float(torch.linalg.eigvals(J).abs().max())
        if not np.isfinite(jac_rate) or jac_rate <= 0.0:
            raise RuntimeError(f"|eig(J)|_max = {jac_rate!r} — not a usable timescale")
        dt = 0.2 / jac_rate
        gamma = jac_rate
        D_np = model.D.detach().numpy()
        F_fn = make_spatial_F(model, self.n, self.L)

        u, res = None, np.inf
        if self._warm is not None:
            u, res = self._newton(self._warm, F_fn, gamma, D_np)
        if u is None or res > self.CONVERGENCE_TOL:
            try:
                X = relax_to_pattern(model, xstar, self.n, self.L, dt, self.seed,
                                     noise=self.noise, chunk=self.chunk,
                                     max_chunks=self.max_chunks, flat_tol=self.flat_tol)
            except RuntimeError:
                return self._finish(None, "solve_failed")
            u, res = self._newton(torch.from_numpy(np.ascontiguousarray(X)),
                                  F_fn, gamma, D_np)
            if res > self.CONVERGENCE_TOL:
                return self._finish(None, "solve_failed")
        self.last_residual = float(res)

        # Patterned-ness: the eval/rollout.py floor (rollout.py:234), D-FFT-9 closure 2.
        pattern_floor = max(1e-3, 0.02 * abs(float(xstar[0])))
        amp = float(u[0].std())
        if not np.isfinite(amp):
            return self._finish(None, "solve_failed")
        if amp <= pattern_floor:
            return self._finish(None, "not_patterned")

        self._warm = u.detach().clone()
        payload = dict(model=model, u_star=u.detach(), n=self.n, L=self.L,
                       k2_full=self.k2_full, D_np=D_np, gamma=gamma)
        self.last_payload = payload
        out = PatternSolve.apply(payload, *(getattr(model, nm) for nm in THETA_NAMES))
        return self._finish(out, "ok")
