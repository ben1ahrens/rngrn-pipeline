"""solver.py — independent forward RD solver (ANSWER-KEY SIDE).

Integrates a reference system to a converged Turing pattern to MANUFACTURE
training frames. This is the generator, not the model; it is on the answer-key
side of the firewall. Spectral (FFT) diffusion + explicit reaction, IMEX split-
step — adequate for generating attractor frames of the reference systems (the
STIFF-integration concern is a recovery-side rollout problem, handled by
eval/numerics.py; here the generator params are benign and known-stable).

TUNING (Claude Code, on CUDA): T_max / dt for convergence at higher resolution;
this is a data-generation knob, deliberately OFF the model tuning axis.
"""
from __future__ import annotations
import numpy as np


def _jac_at(system, x, eps=1e-6):
    """Finite-difference Jacobian of the reaction at a single state x (answer-key side)."""
    x = np.asarray(x, float); N = x.size
    J = np.zeros((N, N)); f0 = np.asarray(system.reaction(x), float)
    for j in range(N):
        xp = x.copy(); xp[j] += eps
        J[:, j] = (np.asarray(system.reaction(xp), float) - f0) / eps
    return J


def _laplacian_fft(field, L):
    n = field.shape[-1]
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    k2 = KX**2 + KY**2
    out = np.empty_like(field)
    for i in range(field.shape[0]):
        out[i] = np.real(np.fft.ifft2(-k2 * np.fft.fft2(field[i])))
    return out


def simulate_to_attractor(system, resolution=128, T_max=4000.0, dt=0.1,
                          seed=0, noise_init=1e-2):
    """Integrate the reference system from a perturbed homogeneous state to a
    converged pattern. Returns (frame (n_species,H,W), L)."""
    rng = np.random.default_rng(seed)
    xstar = np.asarray(system.steady_state(), float)
    N = xstar.size
    n = resolution
    L = system.L
    D = np.asarray(system.D, float)
    field = (xstar[:, None, None]
             + noise_init * rng.standard_normal((N, n, n)))
    # Stability-safe timestep: explicit reaction is limited by the fastest linear rate
    # (stiff systems like Schnakenberg gamma=100 need dt ~ 0.2/max|eig J|). Cap the
    # requested dt and preserve the horizon by taking more sub-steps. This keeps the
    # reference generator stable without a config edit; on GPU Claude Code can swap in
    # an implicit/ETD generator for higher resolution.
    J0 = _jac_at(system, xstar)
    rate = float(np.max(np.abs(np.linalg.eigvals(J0)))) if np.all(np.isfinite(J0)) else 0.0
    dt_stable = 0.2 / (rate + 1e-9)
    dt_eff = min(dt, dt_stable)
    n_steps = int(np.ceil(T_max / dt_eff))
    dt = dt_eff
    for step in range(n_steps):
        lap = _laplacian_fft(field, L)
        rhs_reaction = _reaction_grid(system, field)
        field = field + dt * (D[:, None, None] * lap + rhs_reaction)
        if not np.all(np.isfinite(field)):
            raise FloatingPointError(f"solver diverged at step {step}; reduce dt")
    return field.astype(np.float32), float(L)


def _reaction_grid(system, field):
    """Pointwise reaction over a grid. field: (N,H,W) -> (N,H,W).

    The reference reactions are elementwise closed forms, so passing the component
    planes straight through system.reaction vectorises over the whole grid in one
    call (u, v = field unpacks to (H,W) arrays)."""
    out = system.reaction(field)            # np.array([f_plane, g_plane, ...])
    return np.asarray(out, float).reshape(field.shape)
