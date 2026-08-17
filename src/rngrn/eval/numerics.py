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
  etdrk4_rfft         : bit-for-bit the same scheme as `etdrk4`, but exploiting the fact
                        that the state is REAL: the half-spectrum rfft2/irfft2 pair does
                        half the transform work. Measured on an N=3 model, CPU,
                        OMP_NUM_THREADS=1, mean over 100 steps: 1.44 -> 0.81 ms/step at
                        64x64, 3.06 -> 1.70 at 96x96, 5.70 -> 2.97 at 128x128 (1.8-1.9x),
                        with max |X_rfft - X_full| = 4.4e-16 (64), 6.7e-16 (96),
                        4.4e-16 (128) after 100 steps from the same X0. Use this one for
                        the morphology rollout; `etdrk4` stays the untouched reference.

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
import functools

import numpy as np


def _spectral_k2(n, L):
    k = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
    KX, KY = np.meshgrid(k, k)
    return KX**2 + KY**2


def _spectral_k2_half(n, L):
    """|k|^2 on the HALF spectrum produced by rfft2 over axes (1, 2) of an (N, n, n) field.

    rfft2 keeps the full frequency axis on dim 1 and the non-negative half on dim 2, so the
    two axes carry different frequency vectors and the meshgrid MUST be 'ij'-indexed to line
    up with the array axes. (`_spectral_k2` gets away with the default 'xy' indexing only
    because its two axes carry the same vector, making KX^2+KY^2 transpose-symmetric.)
    """
    kx = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
    ky = np.fft.rfftfreq(n, d=L / n) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
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


@functools.lru_cache(maxsize=4)
def _half_coeffs_cached(D_bytes, N, n, L, dt):
    """ETDRK4 coefficients for the half spectrum, memoised on (D, N, n, L, dt).

    WHY A CACHE. `simulate(early_stop=True)` re-enters the integrator every `check_every`
    steps to look at the field, and the coefficients depend only on (D, n, L, dt) — none of
    which change during a run. `_phi_contour` was measured at 27.6 ms on a (3, 64, 33)
    half-grid and 68.2 ms on (3, 96, 49), i.e. 34 and 40 steps' worth of work respectively;
    rebuilding it once per 200-step chunk would cost ~+32% on a 609-step 96x96 rollout.
    With the cache, chunked and one-call driving of that same rollout differ by -1.8%
    (n=64) and +1.5% (n=96), best of three — i.e. by nothing.

    The cache keys on the raw BYTES of D, so two models with different diffusivities cannot
    collide. Size 4 is deliberately small: each entry is six (N, n, n//2+1) float arrays.
    The returned arrays are shared, so they are marked read-only.
    """
    D = np.frombuffer(D_bytes, dtype=np.float64)
    if D.shape != (N,):
        raise ValueError(f"the state has {N} species but D has {D.shape[0]} diffusivities; "
                         f"refusing to broadcast one onto the other")
    K2 = _spectral_k2_half(n, L)
    Lop = np.stack([-D[i] * K2 for i in range(N)])            # (N, n, n//2+1)
    coeffs = _phi_contour(Lop, dt)
    for c in coeffs:                 # shared across calls — never mutate them in place
        c.flags.writeable = False
    return coeffs


def _cached_half_coeffs(D, N, n, L, dt):
    D = np.ascontiguousarray(D, dtype=np.float64).ravel()
    return _half_coeffs_cached(D.tobytes(), N, n, L, dt)


def integrate_etdrk4_rfft(X0, D, reaction_np, n, L, dt, nsteps):
    """ETDRK4, half-spectrum. Identical scheme to `integrate_etdrk4`, ~1.9x cheaper.

    The state is real, so its Fourier transform is conjugate-symmetric and the negative-
    frequency half of `fft2` is redundant. `rfft2`/`irfft2` store and transform only the
    non-negative half, halving both the transform cost and the coefficient arrays. The
    arithmetic is otherwise line-for-line `integrate_etdrk4`, and `irfft2` enforces exactly
    the conjugate symmetry that `np.real(ifft2(...))` was projecting onto — so this is not
    an approximation. Measured agreement after 100 steps: <= 6.7e-16 (see module docstring).
    """
    N = X0.shape[0]
    E, E2, Q, f1, f2, f3 = _cached_half_coeffs(D, N, int(n), float(L), float(dt))

    def Nfun(vhat):
        X = np.fft.irfft2(vhat, s=(n, n), axes=(1, 2))
        return np.fft.rfft2(reaction_np(X), axes=(1, 2))

    v = np.fft.rfft2(X0, axes=(1, 2))
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
            return np.fft.irfft2(v, s=(n, n), axes=(1, 2)), True
    return np.fft.irfft2(v, s=(n, n), axes=(1, 2)), False


def integrate_bdf1_newton_krylov(X0, D, reaction_np, n, L, dt, nsteps):
    """STUB — deliberately loud. docs/REDESIGN_rngrn.md §5.2 removed the silent
    ETDRK4 fallback (a result labelled bdf1 that was not bdf1). Implemented at the
    gate milestone (R4 plan) as the 128² cross-check integrator."""
    raise NotImplementedError(
        "bdf1_newton_krylov is not implemented; it previously fell back to ETDRK4 "
        "silently. Use 'etdrk4'/'etdrk4_rfft', or implement the BDF1 scheme."
    )


INTEGRATORS = dict(
    imex_split=integrate_imex_split,
    etdrk4=integrate_etdrk4,
    etdrk4_rfft=integrate_etdrk4_rfft,
    bdf1_newton_krylov=integrate_bdf1_newton_krylov,
)
