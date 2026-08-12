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


def make_spatial_F(model: RNGRN, n: int, L: float):
    """F(u) = D lap u + f(u), torch end-to-end (differentiable w.r.t. u AND theta).

    The Laplacian is spectral (rfft2), matching the training-side design: no stencil.
    The same closure serves the backward dF/dtheta vjp — it must stay differentiable
    through ``model``'s parameter properties.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    ky = 2.0 * np.pi * np.fft.rfftfreq(n, d=L / n)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    # (n, n//2+1), rfft2 layout, on the model's device (no-op on the CPU default path)
    k2h = torch.from_numpy(KX**2 + KY**2).to(model.device)

    def spatial_F(u: torch.Tensor) -> torch.Tensor:
        lap = torch.fft.irfft2(-k2h * torch.fft.rfft2(u), s=(n, n))
        return model.D[:, None, None] * lap + reaction_fields(model, u)

    return spatial_F


def translation_modes(u: torch.Tensor, n: int, L: float) -> tuple[torch.Tensor, torch.Tensor]:
    """du/dx, du/dy of a (N, n, n) periodic field, spectrally (the exact zero modes of A)."""
    kx = torch.from_numpy(2.0 * np.pi * np.fft.fftfreq(n, d=L / n)).to(u.device)
    ky = torch.from_numpy(2.0 * np.pi * np.fft.rfftfreq(n, d=L / n)).to(u.device)
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


def _sym_ortho(a: float, b: float) -> tuple[float, float, float]:
    """Stable Givens rotation (ported from scipy.sparse.linalg._isolve; BSD)."""
    import math
    if b == 0.0:
        return math.copysign(1.0, a) if a != 0.0 else 1.0, 0.0, abs(a)
    if a == 0.0:
        return 0.0, math.copysign(1.0, b), abs(b)
    if abs(b) > abs(a):
        tau = a / b
        s = math.copysign(1.0, b) / math.sqrt(1.0 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = math.copysign(1.0, a) / math.sqrt(1.0 + tau * tau)
        s = c * tau
        r = a / c
    return c, s, r


def _lsmr_torch(matvec, rmatvec, b: torch.Tensor, atol: float = 1e-13,
                btol: float = 1e-13, maxiter: int = 20000) -> torch.Tensor:
    """LSMR (Fong & Saunders 2011) on torch tensors, any device — a faithful port of
    scipy.sparse.linalg.lsmr's recurrences (damp=0, x0=0, and scipy's DEFAULT
    conlim=1e8 stop included: the D1-verified `_minnorm_solve` calls scipy with that
    default, and on these near-singular operators the conlim test can fire, so dropping
    it would silently change the verified stopping behaviour). Exists so the CUDA
    solver path preserves D-FFT-10's minimal-norm semantics instead of bouncing every
    matvec through the CPU. Rotation scalars stay Python floats (float64), exactly like
    the scipy reference; only the Golub-Kahan vectors live on the device.
    ``tests/test_forward_solve.py`` pins parity against scipy on the same operator."""
    u = b.clone()
    normb = float(torch.linalg.vector_norm(u))
    if normb == 0.0:
        return torch.zeros_like(rmatvec(b))
    beta = normb
    u = u / beta
    v = rmatvec(u)
    alpha = float(torch.linalg.vector_norm(v))
    x = torch.zeros_like(v)
    if alpha > 0.0:
        v = v / alpha
    else:
        return x

    zetabar = alpha * beta
    alphabar = alpha
    rho = rhobar = cbar = 1.0
    sbar = 0.0
    h = v.clone()
    hbar = torch.zeros_like(v)
    betadd = beta
    betad = 0.0
    rhodold = 1.0
    tautildeold = thetatilde = zeta = d = 0.0
    normA2 = alpha * alpha
    normr = beta
    maxrbar = 0.0
    minrbar = 1e+100
    ctol = 1e-8                      # 1/conlim, scipy's default conlim=1e8
    if zetabar == 0.0:
        return x

    for itn in range(1, maxiter + 1):
        u = matvec(v) - alpha * u
        beta = float(torch.linalg.vector_norm(u))
        if beta > 0.0:
            u = u / beta
            v = rmatvec(u) - beta * v
            alpha = float(torch.linalg.vector_norm(v))
            if alpha > 0.0:
                v = v / alpha
        chat, shat, alphahat = _sym_ortho(alphabar, 0.0)
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        hbar = h + (-(thetabar * rho / (rhoold * rhobarold))) * hbar
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v + (-(thetanew / rho)) * h
        # ||r|| estimate (scipy's rotation cascade, verbatim in structure)
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = float(np.sqrt(d + (betad - taud) ** 2 + betadd * betadd))
        normA2 = normA2 + beta * beta
        normA = float(np.sqrt(normA2))
        normA2 = normA2 + alpha * alpha
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
        normar = abs(zetabar)
        normx = float(torch.linalg.vector_norm(x))
        test1 = normr / normb
        test2 = normar / (normA * normr) if (normA * normr) != 0.0 else np.inf
        test3 = 1.0 / condA
        t1 = test1 / (1.0 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb
        if (1.0 + test3 <= 1.0 or 1.0 + test2 <= 1.0 or 1.0 + t1 <= 1.0
                or test3 <= ctol or test2 <= atol or test1 <= rtol):
            break
    return x


def _minnorm_solve_t(apply_fwd, apply_adj, Mmv, b: torch.Tensor,
                     tol: float = 1e-10, maxiter: int = 20000
                     ) -> tuple[torch.Tensor, float]:
    """Torch mirror of `_minnorm_solve` — SAME semantics and constants (right
    preconditioning, LSMR atol=btol=1e-13, 6 refinement rounds on the TRUE residual,
    stall break at res > 0.9*res_prev), running entirely on b's device. Exists so the
    CUDA path honours D-FFT-10 without a per-matvec device round-trip."""
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

    if u0.device.type != "cpu":
        # Device path: identical algebra via the torch LSMR (`_minnorm_solve_t` mirrors
        # `_minnorm_solve` constant-for-constant); no per-matvec CPU round-trips.
        D_dev = torch.from_numpy(np.asarray(D)).to(u0.device)
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
    on_device = u.device.type != "cpu"

    def precon(x: np.ndarray) -> np.ndarray:
        v = x.reshape(Nsp, n, n)
        vh = np.fft.fft2(v, axes=(1, 2))
        vh = vh / (gamma + D[:, None, None] * k2_full[None])
        return np.real(np.fft.ifft2(vh, axes=(1, 2))).reshape(-1)

    if on_device:
        D_dev = torch.from_numpy(np.asarray(D)).to(u.device)
        k2_dev = torch.from_numpy(np.asarray(k2_full)).to(u.device)

        def precon_t(x: torch.Tensor) -> torch.Tensor:
            vh = torch.fft.fft2(x.reshape(Nsp, n, n), dim=(-2, -1))
            vh = vh / (gamma + D_dev[:, None, None] * k2_dev[None])
            return torch.real(torch.fft.ifft2(vh, dim=(-2, -1))).reshape(-1)

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

        def Amv_t(x: torch.Tensor) -> torch.Tensor:
            _, out = torch.func.jvp(F_fn, (u0,), (x.reshape(u.shape),))
            return out.detach().reshape(-1)

        def ATmv_t(x: torch.Tensor) -> torch.Tensor:
            (out,) = vjp_fn(x.reshape(u.shape))
            return out.detach().reshape(-1)

        Fv = F_fn(u).detach()
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
        modes = modes_of(u)
        Q = []
        for m in modes:
            q = m.detach().reshape(-1)
            for p in Q:
                q = q - (p @ q) * p
            if float(q.norm()) > 1e-12:
                Q.append(q / q.norm())
        if len(Q) == 2:
            resid_after = Fv.reshape(-1) + (Amv_t(dx_t) if on_device
                                            else torch.from_numpy(Amv(dx)))
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
    backends relax the same trajectory up to FFT-backend round-off; the saturation
    detector reads channel 0 on the host per chunk (one small transfer per 500 steps).
    Returns the (N, n, n) field ON THE DEVICE."""
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
        ch0 = X[0, 0].detach().cpu().numpy()
        amps.append(float(ch0.std()))
        ks.append(float(obs.kstar_of(ch0, L=L)))
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
    same D-FFT-10 semantics); on CPU every path is byte-identical to the D1 port.
    dt = 0.2/|eig(J)|_max and gamma = |eig(J)|_max are recomputed per solve from the
    current theta, as in the diagnostic.
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
        self._warm: torch.Tensor | None = None
        self.last_residual: float = float("nan")
        self.last_reason: str = "never_solved"

    # -- internals -------------------------------------------------------------------

    def _newton(self, u0: torch.Tensor, F_fn, gamma: float, D_np: np.ndarray):
        modes_of = lambda uu: list(translation_modes(uu, self.n, self.L))  # noqa: E731
        return newton_polish(F_fn, u0, modes_of, self.k2_full, D_np, gamma)

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
        jac_rate = float(torch.linalg.eigvals(J).abs().max())
        if not np.isfinite(jac_rate) or jac_rate <= 0.0:
            raise RuntimeError(f"|eig(J)|_max = {jac_rate!r} — not a usable timescale")
        dt = 0.2 / jac_rate
        gamma = jac_rate
        D_np = model.D.detach().cpu().numpy()
        F_fn = make_spatial_F(model, self.n, self.L)

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
                       k2_full=self.k2_full, D_np=D_np, gamma=gamma)
        self.last_payload = payload
        out = PatternSolve.apply(payload, *(getattr(model, nm) for nm in THETA_NAMES))
        return self._finish(out, "ok")
