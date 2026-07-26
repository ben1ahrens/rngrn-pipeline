"""numerics.py — stiff time integrators for the lift-and-simulate rollout.

Three integrators behind one interface, selected by SolverConfig.integrator:

  imex_split          : split-step spectral IMEX (the reference). Fast, but COLLAPSES
                        under stiff recovered diffusion (exp(-D k^2 dt) -> 0 kills every
                        mode in one step) — design-doc pitfall 8.1.
  etdrk4              : exponential time differencing 4th-order Runge-Kutta on the
                        coupled linear operator (J - k^2 D) per Fourier mode, nonlinear
                        remainder explicit. The stiff-safe method (Cox & Matthews 2002,
                        J Comput Phys 176:430; Kassam & Trefethen 2005, SIAM J Sci Comput
                        26:1214 — contour-integral phi-functions for numerical stability).
  bdf1_newton_krylov  : implicit Euler with a Newton-Krylov nonlinear solve (the
                        validation-plan alternative). Structured stub.

Each integrator advances the reaction-diffusion PDE d x/dt = D lap(x) + f(X), where
`reaction_np(X)` is supplied by the caller (rollout.py builds it from the model).

TEMPLATE STATUS: imex_split and etdrk4 both run and are exercised by the dry run;
etdrk4 uses a per-mode SCALAR linear operator (-D_i k^2) with the Kassam-Trefethen
contour integral for the phi-functions — correct and stiff-stable for the diagonal-
diffusion case here. The FULLY COUPLED (J - k^2 D) matrix-exponential variant and
bdf1_newton_krylov are stubs for Claude Code (see TUNING.md §Rollout).
"""
from __future__ import annotations
import numpy as np


def _spectral_k2(n, L):
    k = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
    KX, KY = np.meshgrid(k, k)
    return KX**2 + KY**2


def _phi_contour(Lop, dt, M=32):
    """Kassam-Trefethen contour-integral evaluation of the ETDRK4 coefficients for a
    (possibly stiff, possibly near-zero) linear operator `Lop` (array over Fourier modes).

    Returns E, E2, Q, f1, f2, f3 — the standard ETDRK4 stencil coefficients, evaluated by
    averaging over M points on a circle around each eigenvalue to avoid cancellation when
    Lop*dt is small (the numerical-stability fix that motivated Kassam-Trefethen 2005).
    """
    E = np.exp(dt * Lop)
    E2 = np.exp(dt * Lop / 2.0)
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)   # roots of unity on a circle
    LR = dt * Lop[..., None] + r[tuple([None] * Lop.ndim + [slice(None)])]
    Q  = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=-1))
    f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, axis=-1))
    f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR**3, axis=-1))
    f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR**3, axis=-1))
    return E, E2, Q, f1, f2, f3


def integrate_imex_split(X0, D, reaction_np, n, L, dt, nsteps):
    """Split-step spectral IMEX (reference; stiff-unsafe)."""
    K2 = _spectral_k2(n, L)
    N = X0.shape[0]
    expD = np.exp(-np.stack([D[i] * K2 for i in range(N)]) * dt)
    X = X0.copy()
    for _ in range(nsteps):
        Xr = X + dt * reaction_np(X)
        Xhat = np.fft.fft2(Xr, axes=(1, 2)) * expD
        X = np.real(np.fft.ifft2(Xhat, axes=(1, 2)))
        if not np.isfinite(X).all():
            return X, True
    return X, False


def integrate_etdrk4(X0, D, reaction_np, n, L, dt, nsteps):
    """ETDRK4 on the diagonal linear diffusion operator per species/mode; reaction explicit.

    Linear operator per species i and mode: L_i(k) = -D_i k^2 (Fourier-diagonal). The
    nonlinear remainder is the full reaction f(X) (the -D_i k^2 part is the linear piece).
    Coefficients via the contour integral so small |L dt| stays accurate.
    """
    K2 = _spectral_k2(n, L)
    N = X0.shape[0]
    Lop = np.stack([-D[i] * K2 for i in range(N)])            # (N,n,n) linear operator in Fourier
    E, E2, Q, f1, f2, f3 = _phi_contour(Lop, dt)

    def Nfun(vhat):
        X = np.real(np.fft.ifft2(vhat, axes=(1, 2)))
        return np.fft.fft2(reaction_np(X), axes=(1, 2))

    v = np.fft.fft2(X0, axes=(1, 2))
    for _ in range(nsteps):
        Nv = Nfun(v)
        a = E2 * v + Q * Nv
        Na = Nfun(a)
        b = E2 * v + Q * Na
        Nb = Nfun(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = Nfun(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        if not np.isfinite(v).all():
            return np.real(np.fft.ifft2(v, axes=(1, 2))), True
    X = np.real(np.fft.ifft2(v, axes=(1, 2)))
    return X, False


def integrate_bdf1_newton_krylov(X0, D, reaction_np, n, L, dt, nsteps):
    """Implicit-Euler (BDF1) with a Newton-Krylov nonlinear solve. STUB.

    TODO(claude-code): each step solve X^{n+1} - dt(D lap + f)(X^{n+1}) = X^n with
    scipy.optimize.newton_krylov (diffusion applied spectrally inside the residual).
    Falls back to ETDRK4 so the pipeline stays runnable.
    """
    return integrate_etdrk4(X0, D, reaction_np, n, L, dt, nsteps)


INTEGRATORS = dict(
    imex_split=integrate_imex_split,
    etdrk4=integrate_etdrk4,
    bdf1_newton_krylov=integrate_bdf1_newton_krylov,
)
