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

The module carries TWO solvers. ``PatternSolver`` is the serial one described above and
remains the reference implementation; ``BatchedPatternSolver`` (bottom of the file) adds a
restart/member axis over a ``model.BatchedRNGRN``, which is what lets a spectral run train
B restarts at once on the GPU. The batched relax is one B-wide ETDRK4 stack; the Newton
polish and the backward adjoint LOOP over the ignited members, calling the D1-verified
``newton_polish`` / ``solve_adjoint`` unchanged. UNMEASURED: the relax/Newton/adjoint cost
split of a batched solve. A prior version of this paragraph quoted a 32^2/B=3 and 64^2/B=3
split ("relax 30%, Newton 40%, adjoint 30%"; "10/75/15") citing "the unit's report", which
does not exist — no run under `experiments/` backs those figures (reporting-numbers.md step
1), so the design conclusion drawn from them ("the per-member loop, not the relax, is what
a further optimisation has to attack") is UNSUPPORTED and is withdrawn along with the
numbers. `docs/PLAN_redesign_R3.md` Task 17 (the B/K curve with the forward solve in the
loop) is where this split would actually get measured and committed under `experiments/`.

Nothing here reads the observed frame or any answer-key quantity: the solver consumes
only the model's own parameters and grid geometry supplied by the caller.
"""
from __future__ import annotations

import numpy as np
import torch

from .losses.terms import steady_state
from .etdrk4_torch import (_phi_contour_torch, _torch_reaction_builder,
                           integrate_etdrk4_rfft_torch, torch_half_coeffs)
from .eval.numerics import integrate_etdrk4_rfft, _spectral_k2, _spectral_k2_half
from .eval.rollout import _reaction_np_builder
from . import observables as obs
from .model import RNGRN, THETA_NAMES, _reaction_raw

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
    cross per chunk. `kstar_of_torch` uses `kstar_of`'s OWN `np.digitize` binning (since
    2026-08-19, D-OBS-1) and is pinned against it by `tests/test_raps_torch_parity.py`, so
    a chunk's k* now differs from the numpy path's only by FFT backend and summation order
    (<= 4.5e-16 measured), not by a bin reassignment. The std is taken with `correction=0`
    to keep numpy's population convention (the ratio test is invariant to that factor
    either way)."""
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
    integrator and Newton/adjoint run through the torch LSMR (`_minnorm_solve_t`, same
    D-FFT-10 semantics, cadenced per D-PERF-5); on CPU the algebra is the D1 port's and
    NEVER enters `_lsmr_torch` — `newton_polish`/`solve_adjoint` select
    `_minnorm_solve_t`/`precon_t` only under `on_device = u.device.type != "cpu"`, so the
    CPU path routes through scipy's `_minnorm_solve` unchanged. Of the two round-off
    level departures from the D1 port taken for device efficiency, only ONE is applied on
    BOTH devices — |eig(J)|_max is evaluated by numpy rather than torch (`solve`) — and
    the torch LSMR cadence is CUDA-only. A reader asking whether CPU results are still
    the D1 reference should read "yes": neither departure changes what any quantity
    means, and the CPU path is unaffected by the LSMR cadence entirely.
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


# ======================================================================================
# BATCHED forward solve — a restart (member) axis over the SAME algebra
# ======================================================================================
# Everything below is additive. The serial path above is the REFERENCE implementation and
# is not reachable from here: nothing in this section is called by `PatternSolver`,
# `PatternSolve`, or any free function they use.
#
# WHAT IS BATCHED AND WHAT IS NOT (structural, from the code below; the relative COST of
# each part is UNMEASURED — see the module docstring's note on the withdrawn cost-split
# figures and `docs/PLAN_redesign_R3.md` Task 17):
#
#   * the ETDRK4 relax runs as ONE (b, N, n, n) stack through
#     `integrate_etdrk4_rfft_torch`'s existing batch axis, with per-member dt and
#     per-member D carried in the ETDRK4 coefficients;
#   * the Newton polish and the backward adjoint LOOP over the ignited members, calling
#     the D1-verified `newton_polish` / `solve_adjoint` UNCHANGED on a single-member view.
#     Each member's polish is an independent minimal-norm least-squares solve, so a loop
#     is exact by construction; a (b,)-batched LSMR would have to re-derive the D-FFT-10
#     stopping and refinement semantics per member, and no measurement here justifies that
#     risk against an unmeasured cost. The ONE thing that is batched in the backward is
#     the final dF/dtheta vjp: the per-member adjoints are stacked and hit the batched
#     parameters in a single `torch.autograd.grad`.
#
# Member identity is stable throughout: `idx` is a tensor of GLOBAL member indices into the
# `BatchedRNGRN`, and nothing here reorders the batch.


def batched_reaction_fields(model, u: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Pointwise reaction on a (b, N, n, n) stack, member ``idx[j]``'s parameters on slice j.

    Differentiable w.r.t. BOTH ``u`` and the batched model's raw parameters: the parameters
    are pulled with ``index_select``, so a gradient lands on exactly the selected members'
    rows of the (B, ...) parameter tensors and is zero elsewhere.

    `model.BatchedRNGRN.reaction` cannot serve here — it takes ONE state vector per member,
    (B, N), and refuses per-pixel states. `_reaction_raw`'s broadcasting contract explicitly
    permits the caller to broadcast the parameters itself, which is what the `[:, None, None]`
    inserts do: (b, 1, 1, N, N) against a (b, n, n, N) field. Elementwise-identical to the
    serial `reaction_fields` for each member (same ops, same reduction axis and order); only
    the broadcast shape differs.
    """
    x = u.permute(0, 2, 3, 1)                                    # (b, n, n, N)
    KA = model.KA.index_select(0, idx)[:, None, None]            # (b, 1, 1, N, N)
    KR = model.KR.index_select(0, idx)[:, None, None]
    alpha = model.alpha.index_select(0, idx)[:, None, None]
    beta = model.beta.index_select(0, idx)[:, None, None]        # (b, 1, 1, N)
    delta = model.delta.index_select(0, idx)[:, None, None]
    f = _reaction_raw(KA, KR, alpha, beta, delta, x, model.form, model.n_hill)
    return f.permute(0, 3, 1, 2)                                 # (b, N, n, n)


def make_spatial_F_batched(model, idx: torch.Tensor, n: int, L: float,
                           k2h: torch.Tensor | None = None):
    """`make_spatial_F` with a leading member axis: F(u) = D lap u + f(u) on (b, N, n, n).

    Same spectral Laplacian (rfft2, no stencil) and the same differentiability contract —
    the closure must stay differentiable through the batched model's parameter properties,
    because the backward's dF/dtheta vjp runs through it.
    """
    if k2h is None:
        KX, KY = _half_k_grids(n, L)
        k2h = torch.from_numpy(KX**2 + KY**2).to(model.device)

    def spatial_F(u: torch.Tensor) -> torch.Tensor:
        lap = torch.fft.irfft2(-k2h * torch.fft.rfft2(u), s=(n, n))
        D = model.D.index_select(0, idx)[:, :, None, None]        # (b, N, 1, 1)
        return D * lap + batched_reaction_fields(model, u, idx)

    return spatial_F


def _member_F(model, member: int, n: int, L: float, k2h: torch.Tensor):
    """A SERIAL-shaped ((N, n, n) -> (N, n, n)) F closure for one member of a batched model.

    This is what lets the Newton polish and the adjoint reuse `newton_polish` /
    `solve_adjoint` verbatim: they are written against a single field and a single set of
    parameters, and the wrapper hands them exactly that while the gradient still flows into
    the batched parameter tensors' row `member`.
    """
    idx = torch.as_tensor([int(member)], dtype=torch.long, device=model.device)
    Fb = make_spatial_F_batched(model, idx, n, L, k2h=k2h)
    return lambda u: Fb(u.unsqueeze(0)).squeeze(0)


def _batched_reaction_builder(model, idx: torch.Tensor):
    """`etdrk4_torch._torch_reaction_builder` with the batch axis carrying MEMBERS.

    The serial builder's batch axis is a stack of initial conditions sharing ONE member's
    parameters; here every slice has its own. Same competitive arithmetic (the einsum
    contractions are term-for-term the serial ones with the parameter tensors gaining a
    leading b), same DETACHED parameters — the relax is gradient-free by design; gradients
    come from the IFT adjoint, never from backprop through the integrator.

    nc1 is refused loudly for the same reason the serial builder refuses it: a silent
    half-port would poison the numbers rather than fail.
    """
    if model.form != "competitive":
        raise NotImplementedError(
            f"batched torch ETDRK4 reaction supports form='competitive' only, got "
            f"{model.form!r} — same restriction as etdrk4_torch._torch_reaction_builder "
            "(ported from diagnostic D2); port nc1 with its own equivalence check first")
    KA = model.KA.detach().index_select(0, idx)              # (b, N, N)
    KR = model.KR.detach().index_select(0, idx)
    alpha = model.alpha.detach().index_select(0, idx)
    beta = model.beta.detach().index_select(0, idx)          # (b, N)
    delta = model.delta.detach().index_select(0, idx)
    n_h = model.n_hill
    KAR = KA + KR
    aKA = alpha * KA

    def reaction_t(X: torch.Tensor) -> torch.Tensor:         # (b, N, n, n)
        xn = torch.clamp(X, min=0.0) ** n_h
        denom = 1.0 + torch.einsum("bij,bjxy->bixy", KAR, xn)
        prod = torch.einsum("bij,bjxy->bixy", aKA, xn) / denom
        return beta[:, :, None, None] + prod - delta[:, :, None, None] * X

    return reaction_t


def _half_coeffs_batched(D: np.ndarray, n: int, L: float, dt: np.ndarray,
                         device: torch.device):
    """Per-member ETDRK4 half-spectrum coefficients: (b, N, n, n//2+1) each.

    `torch_half_coeffs` builds one member's set from a scalar dt and a (N,) D. Batched, dt
    AND D are per member (dt = 0.2/|eig(J)|_max is recomputed per solve from that member's
    own theta), so `Lop` gains a leading b and dt enters as a (b, 1, 1, 1) tensor.
    `_phi_contour_torch` is shape-agnostic — every expression in it is elementwise over
    `Lop` with the contour on a trailing axis — so it is reused unchanged rather than
    re-derived.

    COST NOTE, deliberate: the contour integral materialises a (b, N, n, n//2+1, 32)
    complex128 intermediate, i.e. ~58 MB at b=8/n=96 and ~230 MB at b=32/n=96, once per
    batched solve (dt and D move every Adam step, so it cannot be cached). That is the
    price of one B-wide relax; chunking the contour would trade it back for kernel
    launches and is not done here.
    """
    k2 = torch.from_numpy(_spectral_k2_half(n, L)).to(device)          # (n, n//2+1)
    D_t = torch.as_tensor(np.asarray(D), device=device, dtype=k2.dtype)  # (b, N)
    Lop = -D_t[:, :, None, None] * k2                                   # (b, N, n, nh)
    dt_t = torch.as_tensor(np.asarray(dt), device=device,
                           dtype=k2.dtype).reshape(-1, 1, 1, 1)
    return _phi_contour_torch(Lop, dt_t)


def _kstar_of_torch_batched(fields: torch.Tensor, L: float) -> torch.Tensor:
    """`observables.kstar_of_torch` over a (b, n, n) stack, returning a (b,) DEVICE tensor.

    Shares `observables._raps_torch_bins`' cached binning, so the two cannot drift on bin
    edges — this is the batched twin of `observables.raps_torch`, not a second binning.

    ONE arithmetic difference, and it is why this is documented rather than assumed
    equivalent: the serial centroid sums the 5-bin slice `power[lo:hi]`, where
    `lo = max(1, pk-2)` and `hi = min(nbins, pk+3)`; the batched form multiplies by a
    boolean mask of exactly those bins and sums the whole vector. The selected terms and
    their order are identical and the rest are exact zeros, so the values agree to
    floating-point associativity (~1e-16 relative), not bit-for-bit. The only consumer is
    the saturation detector's flatness ratio at flat_tol=1e-4, which is five orders of
    magnitude above that.
    """
    f = fields - fields.mean(dim=(-2, -1), keepdim=True)
    b, n = f.shape[0], f.shape[-1]
    idx, counts, kcent = obs._raps_torch_bins(n, L, f.device, f.dtype)
    P = torch.abs(torch.fft.fft2(f)) ** 2
    power = torch.zeros(b, kcent.shape[0], device=f.device, dtype=f.dtype)
    power = power.scatter_add_(1, idx.unsqueeze(0).expand(b, -1), P.reshape(b, -1))
    power = power / counts
    power[:, 0] = 0.0                                              # drop k=0
    nb = kcent.shape[0]
    pk = torch.argmax(power, dim=1, keepdim=True)                  # (b, 1)
    bins = torch.arange(nb, device=f.device).unsqueeze(0)          # (1, nb)
    sel = (bins >= torch.clamp(pk - 2, min=1)) & (bins < torch.clamp(pk + 3, max=nb))
    w = power * sel
    return (kcent.unsqueeze(0) * w).sum(dim=1) / (w.sum(dim=1) + 1e-12)


def relax_to_pattern_torch_batched(model, idx: torch.Tensor, xstar: np.ndarray, n: int,
                                   L: float, dt: np.ndarray, seeds, device: torch.device,
                                   noise: float = 1e-2, chunk: int = 500,
                                   max_chunks: int = 400, flat_tol: float = 1e-4,
                                   X0: torch.Tensor | None = None):
    """`relax_to_pattern_torch` over a member axis. Returns (list of (N, n, n) or None,
    list of reason-or-None), both length b, in `idx` order.

    Same initial condition as the serial paths (numpy rng, the member's own seed, same
    x* + noise expression), the same D2-verified integrator, and the SAME saturation
    detector — channel-0 amplitude and k* flat to `flat_tol` over 5 consecutive chunks —
    evaluated PER MEMBER.

    SNAPSHOT-ON-FINISH, not stop-on-finish. The stack integrates until every member has
    finished or `max_chunks` is exhausted, but a member's returned field is the one it held
    at the chunk where ITS detector first fired. That is exactly what the serial call would
    have returned for that member; carrying it along afterwards costs integrator work for a
    field nobody reads, which is the price of one B-wide relax and is why the stack's cost
    is set by its SLOWEST member. Freezing a finished member instead would need a per-member
    mask inside the integrator, which lives in a module this unit does not own.

    FAILURE IS PER MEMBER, not per stack. `integrate_etdrk4_rfft_torch`'s own `blew` flag is
    an `all()` over the whole stack and is deliberately ignored here in favour of a
    per-member finiteness reduction: one member going non-finite cannot reach another (every
    op is per-member over the trailing field axes), so it is recorded as that member's
    failure and the rest continue. A member that never saturates gets the same
    "no saturation" verdict the serial path raises.

    Returns reasons "blew_up" / "no_saturation" for failed members, None for finished ones;
    the CALLER decides whether either is fatal (as with the serial function).
    """
    b = int(idx.shape[0])
    D = model.D.detach().index_select(0, idx).cpu().numpy()          # (b, N)
    reaction_t = _batched_reaction_builder(model, idx)
    coeffs = _half_coeffs_batched(D, n, L, dt, device)
    if X0 is None:
        N = int(model.N)
        X_np = np.empty((b, N, n, n))
        for j, s in enumerate(seeds):
            rng = np.random.default_rng(int(s))
            X_np[j] = xstar[j][:, None, None] + noise * rng.standard_normal((N, n, n))
        X = torch.from_numpy(X_np).to(device)
    else:
        X = X0.detach().to(device).clone()

    out: list = [None] * b
    reasons: list = [None] * b
    done = np.zeros(b, dtype=bool)
    amps: list = []
    ks: list = []
    for _ in range(max_chunks):
        # `dt` is carried by `coeffs`; integrate_etdrk4_rfft_torch takes the argument for
        # signature parity only and never reads it (etdrk4_torch.py). Passing the
        # per-member dt keeps the call site honest about which dt the coefficients hold.
        X, _blew_stack = integrate_etdrk4_rfft_torch(X, reaction_t, n, dt, chunk, coeffs)
        ch0 = X[:, 0].detach()
        # ONE device->host transfer per chunk for the whole stack: amplitude, k*, finiteness.
        stats = torch.stack([
            ch0.std(dim=(-2, -1), correction=0),
            _kstar_of_torch_batched(ch0, L),
            torch.isfinite(X.detach()).reshape(b, -1).all(dim=1).to(ch0.dtype),
        ]).cpu().numpy()
        amps.append(stats[0]); ks.append(stats[1])
        finite = stats[2] > 0.5
        for j in range(b):
            if done[j]:
                continue
            if not finite[j]:
                done[j] = True
                reasons[j] = "blew_up"
                continue
            if len(amps) < 5:
                continue
            a = np.array([amps[-i][j] for i in range(1, 6)])
            k = np.array([ks[-i][j] for i in range(1, 6)])
            if ((a.max() - a.min()) / a.mean() < flat_tol
                    and (k.max() - k.min()) / k.mean() < flat_tol):
                out[j] = X[j].detach().clone()
                done[j] = True
        if done.all():
            break
    for j in range(b):
        if out[j] is None and reasons[j] is None:
            reasons[j] = "no_saturation"
    return out, reasons


class BatchedPatternSolve(torch.autograd.Function):
    """`PatternSolve` over a member axis: u*(theta) for b ignited members as one node.

    Forward returns the DETACHED (b, N, n, n) solved stack. Backward is the same IFT chain,
    with the split this section's header states: ONE minimal-norm adjoint solve PER MEMBER
    (a loop over `solve_adjoint`, unchanged — each member's adjoint system is independent,
    so a loop is exact and inherits D-FFT-10's verified semantics rather than re-deriving
    them), then a SINGLE batched dL/dtheta = -lam^T dF/dtheta vjp through the batched F
    closure, which scatters the gradient onto exactly the solved members' rows of the
    (B, ...) parameter tensors.

    The adjoint residual tripwire is applied PER MEMBER and raises on the first breach, with
    the offending member named. This is the same severity as the serial path: a serial
    adjoint breach also propagates out of recover()'s Adam loop (nothing catches it), so
    batching does not turn a per-restart failure into a run failure — it was already one.
    """

    @staticmethod
    def forward(ctx, payload, theta_s, theta_g, theta_alpha, theta_delta, theta_beta,
                theta_D):
        ctx.payload = payload
        return payload["u_star"].clone()

    @staticmethod
    def backward(ctx, grad_out):
        p = ctx.payload
        model, u_star, idx = p["model"], p["u_star"], p["idx"]
        n, L = p["n"], p["L"]
        members = p["members"]
        with torch.enable_grad():
            lams = []
            worst = 0.0
            for j, mem in enumerate(members):
                F_j = _member_F(model, mem, n, L, p["k2h"])
                lam, adj_res = solve_adjoint(
                    F_j, u_star[j], grad_out[j].contiguous(), p["k2_full"],
                    p["D_np"][j], float(p["gamma"][j]),
                    k2_dev=p["k2_dev"], D_dev=p["D_dev"][j])
                if adj_res > _ADJOINT_RESIDUAL_TRIPWIRE:
                    raise RuntimeError(
                        f"adjoint solve stalled at relative residual {adj_res:.2e} for "
                        f"batch member {mem} (tripwire {_ADJOINT_RESIDUAL_TRIPWIRE:g}) — the "
                        f"gradient would be biased by that order (D-FFT-10). Refusing to "
                        f"hand it to the optimiser.")
                worst = max(worst, adj_res)
                lams.append(lam.reshape(u_star[j].shape))
            p["last_adjoint_residual"] = worst
            lam_stack = torch.stack(lams)
            F_fn = make_spatial_F_batched(model, idx, n, L, k2h=p["k2h"])
            params = [getattr(model, nm) for nm in THETA_NAMES]
            Fv = F_fn(u_star)          # u_star fixed; graph runs through theta only
            g_theta = torch.autograd.grad(Fv, params, grad_outputs=-lam_stack)
        return (None,) + tuple(g_theta)


class BatchedPatternSolver:
    """`PatternSolver` with a restart (member) axis over a `model.BatchedRNGRN`.

    Same state machine, same knobs, same skip/fail vocabulary — "ok" / "solve_failed" /
    "not_patterned" — evaluated PER MEMBER, and the same warm-start policy. Three things
    differ, all forced by the member axis and none of them a change of meaning:

    1. **Ignition is per member**, so the caller passes the member indices to solve for.
       `solve_subset` solves ONLY those, returning the stack of the ones that succeeded plus
       a reason per member that did not. Members are never reordered: `members` are GLOBAL
       indices into the batched model and the warm-state store is keyed by them, so a lane's
       identity is stable for the whole run and a member that ignites at step 300 starts
       from a fresh relax exactly as a serial restart would.
    2. **The steady state is supplied, not recomputed.** `losses.total.compute_terms_batched`
       has already solved it for the whole batch this step; the serial `PatternSolver` calls
       `losses.terms.steady_state` itself and gets the same value the serial `compute_terms`
       computed a moment earlier. Passing it in removes that duplicate solve without changing
       which x* the relax starts from.
    3. **Two relax passes at most per call.** Members that hold warm state take the short
       warm re-relax first (warm_mode="relax"); those that fail it join the members with no
       warm state in a single fresh full relax. That is the serial fall-through, batched: at
       steady state after the first step, the fresh pass is empty.

    `warm_mode` accepts the same two values as `PatternSolver` and carries the same meaning;
    "relax" is the training policy for the reason stated there (translation-invariant losses,
    and Newton-only warm starts measured pathological at Adam-scale theta displacement).
    """

    #: Newton convergence requirement on ||F||/||u|| — the serial value, verbatim.
    CONVERGENCE_TOL = PatternSolver.CONVERGENCE_TOL

    def __init__(self, model, n: int, L: float, seeds, noise: float = 1e-2,
                 chunk: int = 500, max_chunks: int = 400, flat_tol: float = 1e-4,
                 device: str | torch.device | None = None, warm_mode: str = "newton",
                 warm_max_chunks: int = 40):
        self.model = model
        self.B = int(model.B)
        self.seeds = [int(s) for s in seeds]
        if len(self.seeds) != self.B:
            raise ValueError(
                f"BatchedPatternSolver needs one seed per member: got {len(self.seeds)} "
                f"seeds for B={self.B} members")
        self.n = int(n)
        self.L = float(L)
        self.noise = noise
        self.chunk = chunk
        self.max_chunks = max_chunks
        self.flat_tol = flat_tol
        self.device = torch.device(device) if device is not None else model.device
        if self.device.type != model.device.type:
            raise ValueError(
                f"BatchedPatternSolver device={self.device} but the model lives on "
                f"{model.device} — the spatial F closure mixes model parameters and field "
                "tensors, so they must share a device.")
        if warm_mode not in ("newton", "relax"):
            raise ValueError(f"warm_mode must be 'newton' or 'relax', got {warm_mode!r}")
        self.warm_mode = warm_mode
        self.warm_max_chunks = int(warm_max_chunks)   # UNCALIBRATED budget cap
        self.k2_full = _spectral_k2(self.n, self.L)
        KX, KY = _half_k_grids(self.n, self.L)
        self._k2h = torch.from_numpy(KX**2 + KY**2).to(self.device)
        self._kxy = (torch.from_numpy(KX).to(self.device),
                     torch.from_numpy(KY).to(self.device))
        self._k2_full_dev = torch.from_numpy(self.k2_full).to(self.device)
        # PER-MEMBER warm state, keyed by GLOBAL member index and persisting across Adam
        # steps. None = no warm start (never solved, or cleared by a failure).
        self._warm: list = [None] * self.B
        self.last_residual = np.full(self.B, np.nan)
        self.last_reason: list = ["never_solved"] * self.B

    # -- internals -------------------------------------------------------------------

    def _newton_member(self, member: int, u0: torch.Tensor, gamma: float,
                       D_np: np.ndarray):
        """One member's Newton polish, through the UNCHANGED D1-verified `newton_polish`."""
        F_fn = _member_F(self.model, member, self.n, self.L, self._k2h)
        modes_of = lambda uu: list(                                        # noqa: E731
            translation_modes(uu, self.n, self.L, kxy=self._kxy))
        D_dev = self.model.D.detach()[member]
        return newton_polish(F_fn, u0, modes_of, self.k2_full, D_np, gamma,
                             k2_dev=self._k2_full_dev, D_dev=D_dev)

    def _relax(self, members, xstar: np.ndarray, dt: np.ndarray,
               X0: torch.Tensor | None = None, max_chunks: int | None = None):
        """Batched ETDRK4 relax for `members`. X0 = warm re-relax; None = fresh from
        x* + noise. Returns (list of (N, n, n) or None, list of reason-or-None)."""
        mc = self.max_chunks if max_chunks is None else max_chunks
        idx = torch.as_tensor(list(members), dtype=torch.long, device=self.device)
        return relax_to_pattern_torch_batched(
            self.model, idx, xstar, self.n, self.L, dt,
            [self.seeds[m] for m in members], self.device, noise=self.noise,
            chunk=self.chunk, max_chunks=mc, flat_tol=self.flat_tol, X0=X0)

    # -- the state machine -----------------------------------------------------------

    def solve_subset(self, members, xstar_batch: torch.Tensor):
        """Solve the patterned steady state for `members` (global indices, ascending).

        `xstar_batch` is the (B, N) DETACHED homogeneous steady state this step's batched
        loss already computed — the same quantity the serial solver takes from
        `losses.terms.steady_state`.

        Returns ``(u_stack, ok_members, reasons)``:
          * ``u_stack``  (b_ok, N, n, n) differentiable-through-IFT, or None if none solved;
          * ``ok_members`` the global member indices in `u_stack`'s row order;
          * ``reasons``  {member: "solve_failed" | "not_patterned"} for the rest.
        """
        members = [int(m) for m in members]
        if not members:
            return None, [], {}
        model = self.model
        xs_np = xstar_batch.detach().cpu().numpy()
        # |eig(J)|_max on the HOST, per member: J is (B, N, N) with N=3 and
        # `torch.linalg.eigvals` has no batched cuSOLVER kernel for small non-symmetric
        # matrices (CLAUDE.md §7). numpy's LAPACK over the stack is microseconds and gives
        # the same values the serial path's per-matrix call does.
        J = model.jacobian(xstar_batch, create_graph=False).detach().cpu().numpy()
        rates = np.abs(np.linalg.eigvals(J)).max(axis=-1)                  # (B,)
        bad = [m for m in members if not (np.isfinite(rates[m]) and rates[m] > 0.0)]
        if bad:
            raise RuntimeError(
                f"|eig(J)|_max is not a usable timescale for batch member(s) {bad}: "
                f"{rates[bad]!r}")
        D_all = model.D.detach().cpu().numpy()                             # (B, N)
        reasons: dict = {}

        # ---- pass 1: warm re-relax for members that hold warm state --------------------
        polished: dict = {}                                                # member -> u
        pending: list = []
        warm_members = [m for m in members if self._warm[m] is not None]
        fresh_members = [m for m in members if self._warm[m] is None]
        if warm_members:
            if self.warm_mode == "newton":
                for m in warm_members:
                    u, res = self._newton_member(m, self._warm[m], float(rates[m]),
                                                 D_all[m])
                    if res <= self.CONVERGENCE_TOL:
                        polished[m] = (u, res)
                    else:
                        pending.append(m)
            else:
                X0 = torch.stack([self._warm[m] for m in warm_members])
                fields, _why = self._relax(
                    warm_members, xs_np[warm_members],
                    0.2 / rates[warm_members], X0=X0,
                    max_chunks=self.warm_max_chunks)
                for j, m in enumerate(warm_members):
                    # A budget-exhausted or blown-up warm relax is NOT failure: it falls
                    # through to the fresh pass, exactly as the serial path does.
                    if fields[j] is None:
                        pending.append(m)
                        continue
                    u, res = self._newton_member(m, fields[j], float(rates[m]), D_all[m])
                    if res <= self.CONVERGENCE_TOL:
                        polished[m] = (u, res)
                    else:
                        pending.append(m)

        # ---- pass 2: fresh full relax for everyone the warm pass did not settle --------
        rest = sorted(fresh_members + pending)
        if rest:
            fields, _why = self._relax(rest, xs_np[rest], 0.2 / rates[rest])
            for j, m in enumerate(rest):
                if fields[j] is None:
                    reasons[m] = "solve_failed"
                    continue
                u, res = self._newton_member(m, fields[j], float(rates[m]), D_all[m])
                if res > self.CONVERGENCE_TOL:
                    reasons[m] = "solve_failed"
                else:
                    polished[m] = (u, res)

        # ---- patterned-ness, per member ------------------------------------------------
        ok_members: list = []
        us: list = []
        for m in sorted(polished):
            u, res = polished[m]
            self.last_residual[m] = float(res)
            # the eval/rollout.py floor (rollout.py:234), D-FFT-9 closure 2
            pattern_floor = max(1e-3, 0.02 * abs(float(xs_np[m][0])))
            amp = float(u[0].std())
            if not np.isfinite(amp):
                reasons[m] = "solve_failed"
            elif amp <= pattern_floor:
                reasons[m] = "not_patterned"
            else:
                ok_members.append(m)
                us.append(u.detach())
        for m in members:
            self.last_reason[m] = "ok" if m in ok_members else reasons.get(m, "solve_failed")
        # warm state: kept only for members that produced a usable pattern. A failed or
        # homogeneous member is cleared, for the serial path's reason — a homogeneous field
        # would re-converge homogeneous forever as a warm start.
        for m in members:
            self._warm[m] = None
        if not ok_members:
            return None, [], reasons
        u_stack = torch.stack(us)
        for j, m in enumerate(ok_members):
            self._warm[m] = u_stack[j].clone()
        idx = torch.as_tensor(ok_members, dtype=torch.long, device=self.device)
        payload = dict(model=model, idx=idx, members=ok_members, u_star=u_stack,
                       n=self.n, L=self.L, k2_full=self.k2_full,
                       D_np=D_all[ok_members], gamma=rates[ok_members],
                       k2h=self._k2h, k2_dev=self._k2_full_dev,
                       D_dev=model.D.detach().index_select(0, idx))
        self.last_payload = payload
        out = BatchedPatternSolve.apply(
            payload, *(getattr(model, nm) for nm in THETA_NAMES))
        return out, ok_members, reasons
