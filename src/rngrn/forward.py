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

Cost, measured 2026-08-12 (`experiments/diag_fft/cost/`, `gpu_probe/results.json`):
on CPU a FRESH solve is 938-1374 s at 96^2 and a Newton-only warm solve at Adam-scale
theta displacement is WORSE (5030 s) — the CPU path is not viable at training grids.
The CUDA integrator measures 3.25 ms/ETDRK4-step at 512^2 fp64 (~21x the numpy path's
64 ms/step), which is why this module carries a device path and a warm_mode="relax"
policy. Amortised per-Adam-step cost under warm relaxing is still UNMEASURED; the
warm_max_chunks budget cap is UNCALIBRATED.

Nothing here reads the observed frame or any answer-key quantity: the solver consumes
only the model's own parameters and grid geometry supplied by the caller.
"""
from __future__ import annotations

import numpy as np
import torch

from .losses.terms import steady_state
from .etdrk4_torch import (_torch_reaction_builder, integrate_etdrk4_rfft_torch,
                           torch_half_coeffs)
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


def _half_k_grids(n: int, L: float) -> tuple[np.ndarray, np.ndarray]:
    """(KX, KY) of the rfft2 half spectrum, (n, n//2+1) each — pure grid geometry."""
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    ky = 2.0 * np.pi * np.fft.rfftfreq(n, d=L / n)
    return np.meshgrid(kx, ky, indexing="ij")


def make_spatial_F(model: RNGRN, n: int, L: float, k2h: torch.Tensor | None = None):
    """F(u) = D lap u + f(u), torch end-to-end (differentiable w.r.t. u AND theta).

    The Laplacian is spectral (rfft2), matching the training-side design: no stencil.
    The same closure serves the backward dF/dtheta vjp — it must stay differentiable
    through ``model``'s parameter properties.

    ``k2h`` is the (n, n//2+1) rfft2-layout |k|^2 on the model's device. It is pure grid
    geometry, so ``PatternSolver`` builds it ONCE and passes it here (this closure is
    rebuilt per solve AND per backward); None rebuilds it from numpy, which is what a
    standalone caller — the scripts/ diagnostics — gets.
    """
    if k2h is None:
        KX, KY = _half_k_grids(n, L)
        # on the model's device (no-op on the CPU default path)
        k2h = torch.from_numpy(KX**2 + KY**2).to(model.device)

    def spatial_F(u: torch.Tensor) -> torch.Tensor:
        lap = torch.fft.irfft2(-k2h * torch.fft.rfft2(u), s=(n, n))
        return model.D[:, None, None] * lap + reaction_fields(model, u)

    return spatial_F


def translation_modes(u: torch.Tensor, n: int, L: float,
                      kxy: tuple[torch.Tensor, torch.Tensor] | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """du/dx, du/dy of a (N, n, n) periodic field, spectrally (the exact zero modes of A).

    ``kxy`` is the precomputed device-resident (KX, KY) half-spectrum pair; None rebuilds
    it from numpy. ``PatternSolver`` passes the cached pair because ``modes_of`` is called
    once per Newton ITERATION."""
    if kxy is None:
        KXn, KYn = _half_k_grids(n, L)
        KX = torch.from_numpy(KXn).to(u.device)
        KY = torch.from_numpy(KYn).to(u.device)
    else:
        KX, KY = kxy
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


def _sym_ortho_t(a: torch.Tensor, b: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable Givens rotation on 0-d tensors (scipy.sparse.linalg._isolve's `_sym_ortho`;
    BSD), with its three cases — b == 0, a == 0, |b| > |a| — selected BRANCHLESSLY so the
    rotation scalars never have to leave the device. Both non-degenerate branches are
    evaluated with a guarded denominator (`where(x == 0, 1, x)`), so the unselected one
    can never leak a NaN through the select. The float form this replaces is gone: its
    only caller was `_lsmr_torch`, which no longer keeps the rotation scalars on the host.
    """
    one = torch.ones_like(a)
    zero = torch.zeros_like(a)
    sign_a = torch.where(a < 0, -one, one)      # copysign(1, a), with +1 at a == 0
    sign_b = torch.where(b < 0, -one, one)
    b_is0, a_is0 = b == 0, a == 0
    tau1 = a / torch.where(b_is0, one, b)       # |b| > |a| branch
    s1 = sign_b / torch.sqrt(1.0 + tau1 * tau1)
    c1 = s1 * tau1
    r1 = b / s1
    tau2 = b / torch.where(a_is0, one, a)       # |a| >= |b| branch
    c2 = sign_a / torch.sqrt(1.0 + tau2 * tau2)
    s2 = c2 * tau2
    r2 = a / c2
    big_b = b.abs() > a.abs()
    c = torch.where(b_is0, sign_a, torch.where(a_is0, zero, torch.where(big_b, c1, c2)))
    s = torch.where(b_is0, zero, torch.where(a_is0, sign_b, torch.where(big_b, s1, s2)))
    r = torch.where(b_is0, a.abs(), torch.where(a_is0, b.abs(),
                                                torch.where(big_b, r1, r2)))
    return c, s, r


#: Iterations between `_lsmr_torch` stopping-criteria evaluations. Each evaluation is ONE
#: stacked device-to-host sync; the recurrences themselves never sync. Trades up to
#: (this - 1) extra iterations for 1/this of the syncs — see `_lsmr_torch`'s docstring.
_LSMR_STOP_CHECK_EVERY = 25


def _lsmr_torch(matvec, rmatvec, b: torch.Tensor, atol: float = 1e-13,
                btol: float = 1e-13, maxiter: int = 20000) -> torch.Tensor:
    """LSMR (Fong & Saunders 2011) on torch tensors, any device — a port of
    scipy.sparse.linalg.lsmr's recurrences (damp=0, x0=0, and scipy's DEFAULT
    conlim=1e8 stop included: the D1-verified `_minnorm_solve` calls scipy with that
    default, and on these near-singular operators the conlim test can fire, so dropping
    it would silently change the verified stopping behaviour). Exists so the CUDA
    solver path preserves D-FFT-10's minimal-norm semantics instead of bouncing every
    matvec through the CPU.

    DEVIATION FROM SCIPY — the stopping test runs on a CADENCE, not every iteration.
    Everything but the two degenerate early exits (``normb == 0``, ``alpha == 0``, which
    run once) stays on the device: the Golub-Kahan vectors AND the rotation/normalisation
    scalars, the latter as 0-d tensors through `_sym_ortho_t`. Scipy's per-iteration stop
    test needs those scalars on the host, which on CUDA costs >=3 blocking syncs per
    iteration at up to maxiter=20000 iterations; here they are pulled in ONE stacked sync
    every `_LSMR_STOP_CHECK_EVERY` iterations, and at maxiter. Consequences, both
    deliberate:

    - the solve may run up to `_LSMR_STOP_CHECK_EVERY - 1` iterations PAST the iteration
      where scipy would have stopped, so it is not iteration-for-iteration identical;
    - the returned x is whatever the recurrences hold at the stop check, not at the
      crossing. Results therefore match the scipy reference TO SOLVER TOLERANCE (LSMR's
      iterates keep decreasing the least-squares residual, so the extra iterations refine
      rather than degrade), never bit-for-bit.

    `tests/test_forward_solve.py` pins that agreement against scipy on the same operator.
    Exact Krylov breakdown (beta or alpha hitting 0, measure-zero in float64) leaves the
    vector unscaled instead of taking scipy's skip branch; a solve that goes non-finite is
    caught at the next stop check and raised, never returned."""
    u = b.clone()
    normb = float(torch.linalg.vector_norm(u))
    if normb == 0.0:
        return torch.zeros_like(rmatvec(b))
    u = u / normb
    v = rmatvec(u)
    alpha0 = float(torch.linalg.vector_norm(v))
    x = torch.zeros_like(v)
    if alpha0 > 0.0:
        v = v / alpha0
    else:
        return x
    if alpha0 * normb == 0.0:                 # zetabar == 0
        return x

    def _s(val: float) -> torch.Tensor:
        return torch.tensor(val, dtype=v.dtype, device=v.device)

    alpha, beta = _s(alpha0), _s(normb)
    zero, one, inf = _s(0.0), _s(1.0), _s(float("inf"))
    zetabar = _s(alpha0 * normb)
    alphabar = alpha.clone()
    rho = rhobar = cbar = one
    sbar = zero
    h = v.clone()
    hbar = torch.zeros_like(v)
    betadd = beta.clone()
    betad = zero
    rhodold = one
    tautildeold = thetatilde = zeta = d = zero
    normA2 = _s(alpha0 * alpha0)
    maxrbar = zero
    minrbar = _s(1e+100)
    ctol = 1e-8                      # 1/conlim, scipy's default conlim=1e8

    for itn in range(1, maxiter + 1):
        check = itn % _LSMR_STOP_CHECK_EVERY == 0 or itn == maxiter
        u = matvec(v) - alpha * u
        beta = torch.linalg.vector_norm(u)
        # scipy skips the normalisation (and the whole v/alpha update) when beta == 0;
        # here the guarded divisor leaves a zero u unscaled instead — see the docstring.
        u = u / torch.where(beta == 0, one, beta)
        v = rmatvec(u) - beta * v
        alpha = torch.linalg.vector_norm(v)
        v = v / torch.where(alpha == 0, one, alpha)
        chat, shat, alphahat = _sym_ortho_t(alphabar, zero)
        rhoold = rho
        c, s, rho = _sym_ortho_t(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho_t(rhotemp, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        hbar = h + (-(thetabar * rho / (rhoold * rhobarold))) * hbar
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v + (-(thetanew / rho)) * h
        # ||r|| estimate (scipy's rotation cascade, verbatim in structure). Only the
        # RECURRENCES run every iteration; the quantities read by the stop test alone
        # (taud, normr, normA, condA, normar, normx) are formed at the check below.
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho_t(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        d = d + betacheck * betacheck
        normA2 = normA2 + beta * beta
        if check:
            normA = torch.sqrt(normA2)
        normA2 = normA2 + alpha * alpha
        maxrbar = torch.maximum(maxrbar, rhobarold)
        if itn > 1:
            minrbar = torch.minimum(minrbar, rhobarold)
        if not check:
            continue
        taud = (zeta - thetatilde * tautildeold) / rhodold
        normr = torch.sqrt(d + (betad - taud) ** 2 + betadd * betadd)
        condA = torch.maximum(maxrbar, rhotemp) / torch.minimum(minrbar, rhotemp)
        normar = torch.abs(zetabar)
        normx = torch.linalg.vector_norm(x)
        test1 = normr / normb
        den = normA * normr
        test2 = torch.where(den != 0, normar / torch.where(den != 0, den, one), inf)
        test3 = 1.0 / condA
        t1 = test1 / (1.0 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb
        stop = ((1.0 + test3 <= 1.0) | (1.0 + test2 <= 1.0) | (1.0 + t1 <= 1.0)
                | (test3 <= ctol) | (test2 <= atol) | (test1 <= rtol))
        # ONE device-to-host transfer per check: the stop verdict and a finiteness
        # tripwire (a diverged LSMR must fail loud, not return a NaN solution).
        flags = torch.stack([stop.to(v.dtype), torch.isfinite(normx).to(v.dtype)]).cpu()
        if float(flags[1]) == 0.0:
            raise RuntimeError(
                f"_lsmr_torch diverged: ||x|| is not finite at iteration {itn}. The "
                "operator or its preconditioner is degenerate — refusing to return a "
                "non-finite least-squares solution.")
        if float(flags[0]) != 0.0:
            break
    return x


def _minnorm_solve_t(apply_fwd, apply_adj, Mmv, b: torch.Tensor,
                     tol: float = 1e-10, maxiter: int = 20000
                     ) -> tuple[torch.Tensor, float]:
    """Torch mirror of `_minnorm_solve` — same structure and constants (right
    preconditioning, LSMR atol=btol=1e-13, 6 refinement rounds on the TRUE residual,
    stall break at res > 0.9*res_prev), running entirely on b's device. Exists so the
    CUDA path honours D-FFT-10 without a per-matvec device round-trip.

    The INNER solve differs from scipy's in one respect: `_lsmr_torch` evaluates its
    stopping test on a cadence, so it can overshoot scipy's stop by up to
    `_LSMR_STOP_CHECK_EVERY - 1` iterations and its iterate is not bit-identical (see
    that function). The refinement loop around it is unchanged, and it is the TRUE
    residual — not the inner iterate — that decides convergence here, so the returned
    `res` still means exactly what the scipy path's does."""
    bn = max(float(torch.linalg.vector_norm(b)), 1e-300)
    x = torch.zeros_like(b)
    res_prev = np.inf
    for _ in range(6):
        r = b - apply_fwd(x)
        res = float(torch.linalg.vector_norm(r)) / bn
        if res < tol or res > 0.9 * res_prev:
            break
        res_prev = res
        y = _lsmr_torch(lambda z: apply_fwd(Mmv(z)), lambda z: Mmv(apply_adj(z)),
                        r, atol=1e-13, btol=1e-13, maxiter=maxiter)
        x = x + Mmv(y)
    else:
        res = float(torch.linalg.vector_norm(b - apply_fwd(x))) / bn
    return x, res


def solve_adjoint(F_fn, u_star: torch.Tensor, rhs: torch.Tensor, k2_full: np.ndarray,
                  D: np.ndarray, gamma: float, tol: float = 1e-10,
                  maxiter: int = 20000, k2_dev: torch.Tensor | None = None,
                  D_dev: torch.Tensor | None = None) -> tuple[torch.Tensor, float]:
    """MINIMAL-NORM solution of A^T lam = rhs via right-preconditioned LSMR with
    true-residual refinement (D-FFT-10; the WHY-not-projected-GMRES record lives there
    and in the module docstring). Right preconditioning by the SYMMETRIC Fourier-diagonal
    M = (gamma + D k^2)^-1: solve (A^T M) y = rhs, lam = M y; the transpose operator is
    (A^T M)^T = M A, available matrix-free.

    ``k2_dev``/``D_dev`` are the device-resident forms of ``k2_full``/``D`` used by the
    preconditioner on the non-CPU path; None transfers them here, as a standalone caller
    needs. ``PatternSolver`` passes them (k2 cached once per grid, D taken straight off
    the model) to keep the backward off the host round trip."""
    shape_field = tuple(u_star.shape)
    Nsp, n, _ = shape_field
    u0 = u_star.detach()
    _, vjp_fn = torch.func.vjp(F_fn, u0)

    if u0.device.type != "cpu":
        # Device path: identical algebra via the torch LSMR (`_minnorm_solve_t` mirrors
        # `_minnorm_solve` constant-for-constant); no per-matvec CPU round-trips.
        if D_dev is None:
            D_dev = torch.from_numpy(np.asarray(D)).to(u0.device)
        if k2_dev is None:
            k2_dev = torch.from_numpy(np.asarray(k2_full)).to(u0.device)

        def AT_t(x: torch.Tensor) -> torch.Tensor:
            (out,) = vjp_fn(x.reshape(shape_field))
            return out.detach().reshape(-1)

        def Amv_t(x: torch.Tensor) -> torch.Tensor:
            _, out = torch.func.jvp(F_fn, (u0,), (x.reshape(shape_field),))
            return out.detach().reshape(-1)

        def Mmv_t(x: torch.Tensor) -> torch.Tensor:
            vh = torch.fft.fft2(x.reshape(Nsp, n, n), dim=(-2, -1))
            vh = vh / (gamma + D_dev[:, None, None] * k2_dev[None])
            return torch.real(torch.fft.ifft2(vh, dim=(-2, -1))).reshape(-1)

        g_t = rhs.detach().reshape(-1)
        lam_t, res_full = _minnorm_solve_t(AT_t, Amv_t, Mmv_t, g_t,
                                           tol=tol, maxiter=maxiter)
        return lam_t, res_full

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


def _cond2x2(G: torch.Tensor) -> torch.Tensor:
    """2-norm condition number of a 2x2 matrix, closed form, as a 0-d tensor.

    sigma^2 are the eigenvalues of G^T G, whose trace is ||G||_F^2 and whose determinant
    is det(G)^2, so cond = sigma_max/sigma_min = (t + sqrt(t^2 - 4 det^2)) / (2 |det|)
    — the cancellation-free rearrangement of sqrt(lmax/lmin) (lmin = det^2/lmax). Replaces
    `torch.linalg.cond`, an SVD (a LAPACK/cuSOLVER dispatch plus its own sync) on a 2x2.
    A singular G gives +inf, which the caller's `< 1e8` gate rejects exactly as the SVD's
    inf did."""
    t = (G * G).sum()
    det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
    s = torch.sqrt(torch.clamp(t * t - 4.0 * det * det, min=0.0))
    return (t + s) / (2.0 * det.abs())


def _orthonormalise_modes(modes) -> list[torch.Tensor]:
    """Modified Gram-Schmidt over the TWO translation modes, dropping either whose
    residual norm falls to 1e-12 — the degeneracy guard `newton_polish` has always
    applied.

    The norms are computed on the device and pulled in ONE stacked transfer: the
    projected AND unprojected second-mode norms are both formed, so the host branch that
    decides whether mode 0 survived (and therefore whether mode 1 was projected against
    it at all) can be taken after the single sync instead of before it."""
    if len(modes) != 2:
        raise ValueError(f"newton_polish expects exactly 2 translation modes, got "
                         f"{len(modes)} — the 2x2 subspace correction (F-D1-2) is "
                         "defined for the (d/dx, d/dy) pair only")
    q0 = modes[0].detach().reshape(-1)
    n0 = torch.linalg.vector_norm(q0)
    q0h = q0 / torch.where(n0 > 1e-12, n0, torch.ones_like(n0))
    q1 = modes[1].detach().reshape(-1)
    q1p = q1 - (q0h @ q1) * q0h                      # used iff mode 0 survived
    norms = torch.stack([n0, torch.linalg.vector_norm(q1p),
                         torch.linalg.vector_norm(q1)]).cpu()
    n0_f, n1p_f, n1_f = float(norms[0]), float(norms[1]), float(norms[2])
    Q: list[torch.Tensor] = []
    if n0_f > 1e-12:
        Q.append(q0h)
        if n1p_f > 1e-12:
            Q.append(q1p / n1p_f)
    elif n1_f > 1e-12:
        Q.append(q1 / n1_f)
    return Q


def newton_polish(F_fn, u: torch.Tensor, modes_of, k2_full, D, gamma,
                  n_iter: int = 30, tol: float = 1e-11,
                  k2_dev: torch.Tensor | None = None,
                  D_dev: torch.Tensor | None = None) -> tuple[torch.Tensor, float]:
    """Drive ||F(u)|| toward zero with DAMPED Newton steps: minimal-norm LSMR bulk step
    (unprojected — finding F-D1-3) plus an explicit 2-D translation-subspace correction
    for the pinned phase (finding F-D1-2). Damping (step halving when ||F|| does not
    decrease) was measured to matter for D1's eps=1e-3 FD points (large start residual);
    the same overshoot mechanism plausibly applies to far-moved warm starts, unmeasured.

    ``k2_dev``/``D_dev`` are the device-resident preconditioner constants (see
    `solve_adjoint`); None transfers them here. The host reads TWO values per Newton
    iteration on the accepting path — the orthonormalisation norms (one stacked pull) and
    the residual of the first damping trial — plus one more per rejected halving. The
    2x2 conditioning gate is decided on the device, and F(u) is evaluated once per
    accepted step rather than twice."""
    u = u.detach().clone()
    scale = max(float(u.norm()), 1e-300)
    # F(u) is carried ACROSS iterations: the F evaluation that accepts a damped step is
    # bit-identical to the next iteration's F(u), so it is reused rather than recomputed.
    Fv = F_fn(u).detach()
    r = float(Fv.norm()) / scale
    Nsp, n, _ = u.shape
    on_device = u.device.type != "cpu"

    if on_device:
        if D_dev is None:
            D_dev = torch.from_numpy(np.asarray(D)).to(u.device)
        if k2_dev is None:
            k2_dev = torch.from_numpy(np.asarray(k2_full)).to(u.device)

        def precon_t(x: torch.Tensor) -> torch.Tensor:
            vh = torch.fft.fft2(x.reshape(Nsp, n, n), dim=(-2, -1))
            vh = vh / (gamma + D_dev[:, None, None] * k2_dev[None])
            return torch.real(torch.fft.ifft2(vh, dim=(-2, -1))).reshape(-1)
    else:
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

        def Amv_t(x: torch.Tensor) -> torch.Tensor:
            _, out = torch.func.jvp(F_fn, (u0,), (x.reshape(u.shape),))
            return out.detach().reshape(-1)

        def ATmv_t(x: torch.Tensor) -> torch.Tensor:
            (out,) = vjp_fn(x.reshape(u.shape))
            return out.detach().reshape(-1)

        if not on_device:
            def Amv(x: np.ndarray) -> np.ndarray:
                v = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
                _, out = torch.func.jvp(F_fn, (u0,), (v.reshape(u.shape),))
                return out.detach().reshape(-1).numpy()

            def ATmv(x: np.ndarray) -> np.ndarray:
                v = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
                (out,) = vjp_fn(v.reshape(u.shape))
                return out.detach().reshape(-1).numpy()

        # Bulk step: minimal-norm LSMR on the UNPROJECTED system (finding F-D1-3 — a
        # projected inner solve hides near-null error and floors Newton at ~1e-9).
        # Device path = the same algebra through `_minnorm_solve_t` (torch LSMR).
        if on_device:
            dx_t, _ = _minnorm_solve_t(Amv_t, ATmv_t, precon_t, -Fv.reshape(-1),
                                       tol=1e-12, maxiter=8000)
            step = dx_t.reshape(u.shape)
        else:
            dx, _ = _minnorm_solve(Amv, ATmv, precon, -Fv.reshape(-1).numpy(),
                                   u.numel(), tol=1e-12, maxiter=8000)
            step = torch.from_numpy(dx).reshape(u.shape)
        # TRANSLATION-SUBSPACE CORRECTION (finding F-D1-2). Grid pinning makes the
        # translations near-null, not null; a theta-perturbed fixed point sits at a
        # slightly different phase and the phase part of the step is best solved
        # explicitly in the 2-D mode subspace: G c = <t_i, F - A step>, step += -t.c.
        Q = _orthonormalise_modes(modes_of(u))
        if len(Q) == 2:
            resid_after = Fv.reshape(-1) + (Amv_t(dx_t) if on_device
                                            else torch.from_numpy(Amv(dx)))
            At = []
            for q in Q:
                _, Aq = torch.func.jvp(F_fn, (u0,), (q.reshape(u.shape),))
                At.append(Aq.detach().reshape(-1))
            G = torch.stack([torch.stack([q @ a for a in At]) for q in Q])
            rhs = torch.stack([q @ resid_after for q in Q])
            # The ill-conditioning gate stays on the device: solve a guarded G (identity
            # when rejected, so `linalg.solve` cannot raise on a singular 2x2) and scale
            # the correction by the 0/1 verdict. Same arithmetic as the python branch it
            # replaces, without the sync.
            keep = (_cond2x2(G) < 1e8).to(G.dtype)
            G_ok = torch.where(keep.bool(), G, torch.eye(2, dtype=G.dtype,
                                                         device=G.device))
            c = torch.linalg.solve(G_ok, rhs)
            step = step - keep * (c[0] * Q[0] + c[1] * Q[1]).reshape(u.shape)
        lam_damp = 1.0
        for _half in range(6):
            u_try = u + lam_damp * step
            F_try = F_fn(u_try).detach()
            r_try = float(F_try.norm()) / scale
            if r_try < r:
                break
            lam_damp *= 0.5
        else:
            break                     # no descent at any damping — stalled, report r
        u, r, Fv = u_try, r_try, F_try
    return u, r


# ------------------------------------------------------------------ relax-to-pattern

def relax_to_pattern(model: RNGRN, xstar: np.ndarray, n: int, L: float, dt: float,
                     seed: int, noise: float = 1e-2, chunk: int = 500,
                     max_chunks: int = 400, flat_tol: float = 1e-4,
                     X0: np.ndarray | None = None) -> np.ndarray:
    """ETDRK4 from x* + noise (or a supplied warm state X0 — the diag script's original
    warm-relax parameter, restored for warm_mode="relax") until channel-0 amplitude AND
    k* are flat to flat_tol over 5 consecutive chunks. Deterministic for fixed seed.
    Raises on blow-up or non-saturation — the CALLER decides whether that is fatal
    (fresh solve inside training legitimately fails when theta wanders; a diagnostic
    fixture must not)."""
    D = model.D.detach().cpu().numpy()
    reaction_np = _reaction_np_builder(model)
    if X0 is None:
        rng = np.random.default_rng(seed)
        X = xstar[:, None, None] + noise * rng.standard_normal((model.N, n, n))
    else:
        X = X0.copy()
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


def relax_to_pattern_torch(model: RNGRN, xstar: np.ndarray, n: int, L: float,
                           dt: float, seed: int, device: torch.device,
                           noise: float = 1e-2, chunk: int = 500,
                           max_chunks: int = 400, flat_tol: float = 1e-4,
                           X0: torch.Tensor | None = None) -> torch.Tensor:
    """`relax_to_pattern` on the D2-verified torch integrator (etdrk4_torch), fields on
    `device`. SAME initial condition as the numpy path (numpy rng, same seed) so the two
    backends relax the same trajectory up to FFT-backend round-off. Returns the (N, n, n)
    field ON THE DEVICE.

    The saturation detector runs ON THE DEVICE: `observables.kstar_of_torch` (the parity
    port of `kstar_of`) instead of pulling the whole channel-0 frame back per chunk — up
    to 400 x 2 MB of D2H plus a host FFT per solve at 512^2. Only the two summary scalars
    cross per chunk. `kstar_of_torch` bins by `floor(|k|/dk)` where `kstar_of` bins by
    `digitize`, equal for these uniform bins except where an ulp of the division lands a
    point on a bin edge, so a chunk's k* can differ from the numpy path's in the last
    bits and the flatness detector can stop one chunk earlier or later. The std is taken
    with `correction=0` to keep numpy's population convention (the ratio test is
    invariant to that factor either way)."""
    D = model.D.detach().cpu().numpy()
    reaction_t = _torch_reaction_builder(model, device)
    coeffs = torch_half_coeffs(D, n, L, dt, device)
    if X0 is None:
        rng = np.random.default_rng(seed)
        X_np = xstar[:, None, None] + noise * rng.standard_normal((model.N, n, n))
        X = torch.from_numpy(X_np).to(device).unsqueeze(0)      # (1, N, n, n)
    else:
        X = X0.detach().to(device).clone().unsqueeze(0)
    amps: list[float] = []
    ks: list[float] = []
    for _ in range(max_chunks):
        X, blew = integrate_etdrk4_rfft_torch(X, reaction_t, n, dt, chunk, coeffs)
        if blew:
            raise RuntimeError("forward relax blew up")
        ch0 = X[0, 0].detach()
        amps.append(float(ch0.std(correction=0)))
        ks.append(float(obs.kstar_of_torch(ch0, L=L)))
        if len(amps) >= 5:
            a = np.array(amps[-5:]); k = np.array(ks[-5:])
            if ((a.max() - a.min()) / a.mean() < flat_tol
                    and (k.max() - k.min()) / k.mean() < flat_tol):
                return X[0]
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
            F_fn = make_spatial_F(model, n, L, k2h=p["k2h"])
            lam, adj_res = solve_adjoint(F_fn, u_star, grad_out.contiguous(),
                                         p["k2_full"], p["D_np"], p["gamma"],
                                         k2_dev=p["k2_dev"], D_dev=p["D_dev"])
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

    Warm-start policy — TWO modes, dispatched by ``warm_mode``:

    - ``"newton"`` (default): Newton-ONLY from the previous u*, exactly the D1
      instrumentation contract — an ETDRK4 re-relax would drift the pinned phase and
      contaminate any finite-difference check with grid-phase sensitivity (F-D1-1).
      A failed warm Newton falls back to a fresh relax before reporting failure.
    - ``"relax"``: a SHORT warm ETDRK4 re-relax from the previous u* (budget
      ``warm_max_chunks``, then the standard saturation detector) followed by Newton
      polish; failure falls back to the fresh path. This is the TRAINING mode: the
      training losses are translation-invariant, so the phase drift that disqualifies
      a re-relax for FD instrumentation is harmless there, and the 2026-08-12 cost
      measurement showed Newton-only warm starts are pathological at Adam-scale theta
      displacement (warm Newton 5030 s vs fresh 938 s at 96^2 —
      `experiments/diag_fft/cost/run_attempt1.log`). ``warm_max_chunks=40`` is
      UNCALIBRATED (a budget cap, not a measured knob).

    Device: ``device=None`` derives the solve device from the model (recover.py moves
    the model to the training device). On CUDA the relax runs on the D2-verified torch
    integrator and Newton/adjoint run through the torch LSMR (`_minnorm_solve_t`,
    same D-FFT-10 semantics); on CPU the algebra is the D1 port's, with two round-off
    level departures from it taken for device efficiency and applied on BOTH devices:
    |eig(J)|_max is evaluated by numpy rather than torch (`solve`), and the torch LSMR
    stops on a cadence (`_lsmr_torch`). Neither changes what any quantity means.
    dt = 0.2/|eig(J)|_max and gamma = |eig(J)|_max are recomputed per solve from the
    current theta, as in the diagnostic. Grid geometry (the rfft2 |k|^2, the kx/ky pair
    and the full-spectrum |k|^2) is theta-independent and is built ONCE in __init__,
    device-resident, then threaded into the free functions.
    """

    #: Newton convergence requirement on ||F||/||u|| (D1 verbatim).
    CONVERGENCE_TOL = 1e-9

    def __init__(self, model: RNGRN, n: int, L: float, seed: int,
                 noise: float = 1e-2, chunk: int = 500, max_chunks: int = 400,
                 flat_tol: float = 1e-4, device: str | torch.device | None = None,
                 warm_mode: str = "newton", warm_max_chunks: int = 40):
        self.model = model
        self.n = int(n)
        self.L = float(L)
        self.seed = int(seed)
        self.noise = noise
        self.chunk = chunk
        self.max_chunks = max_chunks
        self.flat_tol = flat_tol
        self.device = torch.device(device) if device is not None else model.device
        if self.device.type != model.device.type:
            raise ValueError(
                f"PatternSolver device={self.device} but the model lives on "
                f"{model.device} — the spatial F closure mixes model parameters and "
                "field tensors, so they must share a device. Move the model, or drop "
                "the device argument (None derives it from the model).")
        if warm_mode not in ("newton", "relax"):
            raise ValueError(f"warm_mode must be 'newton' or 'relax', got {warm_mode!r}")
        self.warm_mode = warm_mode
        self.warm_max_chunks = int(warm_max_chunks)   # UNCALIBRATED budget cap
        self.k2_full = _spectral_k2(self.n, self.L)
        # Grid geometry: theta-INDEPENDENT, so built once here and threaded through the
        # free functions, which otherwise rebuild it on the host and re-transfer it per
        # solve (k2h, k2_full) or per Newton ITERATION (the kx/ky pair).
        KX, KY = _half_k_grids(self.n, self.L)
        self._k2h = torch.from_numpy(KX**2 + KY**2).to(self.device)
        self._kxy = (torch.from_numpy(KX).to(self.device),
                     torch.from_numpy(KY).to(self.device))
        self._k2_full_dev = torch.from_numpy(self.k2_full).to(self.device)
        self._warm: torch.Tensor | None = None
        self.last_residual: float = float("nan")
        self.last_reason: str = "never_solved"

    # -- internals -------------------------------------------------------------------

    def _newton(self, u0: torch.Tensor, F_fn, gamma: float, D_np: np.ndarray):
        modes_of = lambda uu: list(                                        # noqa: E731
            translation_modes(uu, self.n, self.L, kxy=self._kxy))
        # model.D is already the device tensor D_np was copied FROM, so passing it spares
        # the preconditioner a host round trip on the way back.
        return newton_polish(F_fn, u0, modes_of, self.k2_full, D_np, gamma,
                             k2_dev=self._k2_full_dev, D_dev=self.model.D.detach())

    def _relax(self, xstar: np.ndarray, dt: float, X0: torch.Tensor | None = None,
               max_chunks: int | None = None) -> torch.Tensor:
        """Dispatch the ETDRK4 relax by device. X0 = warm re-relax (warm_mode="relax");
        None = fresh from x* + noise. Returns a (N, n, n) tensor on the solve device."""
        mc = self.max_chunks if max_chunks is None else max_chunks
        if self.device.type == "cpu":
            X0_np = X0.detach().cpu().numpy() if X0 is not None else None
            X = relax_to_pattern(self.model, xstar, self.n, self.L, dt, self.seed,
                                 noise=self.noise, chunk=self.chunk,
                                 max_chunks=mc, flat_tol=self.flat_tol, X0=X0_np)
            return torch.from_numpy(np.ascontiguousarray(X))
        return relax_to_pattern_torch(self.model, xstar, self.n, self.L, dt,
                                      self.seed, self.device, noise=self.noise,
                                      chunk=self.chunk, max_chunks=mc,
                                      flat_tol=self.flat_tol, X0=X0)

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
        xstar = xs.detach().cpu().numpy()
        J = model.jacobian(xs, create_graph=False).detach()
        # |eig(J)|_max on the HOST: J is (N, N) with N=3, and `torch.linalg.eigvals` on a
        # small non-symmetric matrix has no batched cuSOLVER kernel (~700 us per call,
        # CLAUDE.md §7). A 9-double D2H plus numpy's LAPACK is microseconds, same value.
        jac_rate = float(np.abs(np.linalg.eigvals(J.cpu().numpy())).max())
        if not np.isfinite(jac_rate) or jac_rate <= 0.0:
            raise RuntimeError(f"|eig(J)|_max = {jac_rate!r} — not a usable timescale")
        dt = 0.2 / jac_rate
        gamma = jac_rate
        D_np = model.D.detach().cpu().numpy()
        F_fn = make_spatial_F(model, self.n, self.L, k2h=self._k2h)

        u, res = None, np.inf
        if self._warm is not None:
            if self.warm_mode == "newton":
                u, res = self._newton(self._warm, F_fn, gamma, D_np)
            else:
                # warm_mode="relax": short re-relax from the previous u*, then Newton.
                # A budget-exhausted or blown-up warm relax is NOT failure — it falls
                # through to the fresh path below, same as a stalled warm Newton.
                try:
                    Xw = self._relax(xstar, dt, X0=self._warm,
                                     max_chunks=self.warm_max_chunks)
                    u, res = self._newton(Xw, F_fn, gamma, D_np)
                except RuntimeError:
                    u, res = None, np.inf
        if u is None or res > self.CONVERGENCE_TOL:
            try:
                u0 = self._relax(xstar, dt)
            except RuntimeError:
                return self._finish(None, "solve_failed")
            u, res = self._newton(u0, F_fn, gamma, D_np)
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
                       k2_full=self.k2_full, D_np=D_np, gamma=gamma,
                       k2h=self._k2h, k2_dev=self._k2_full_dev,
                       D_dev=model.D.detach())
        self.last_payload = payload
        out = PatternSolve.apply(payload, *(getattr(model, nm) for nm in THETA_NAMES))
        return self._finish(out, "ok")
