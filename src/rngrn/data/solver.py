"""solver.py — independent forward RD solver (ANSWER-KEY SIDE).

Integrates a reference system to a converged Turing pattern to MANUFACTURE
training frames. This is the generator, not the model; it is on the answer-key
side of the firewall.

SCHEME: Lie-split IMEX — an explicit-Euler reaction step, then a diffusion step solved
EXACTLY in Fourier space (multiply by exp(-D k^2 dt)). Diffusion is unconditionally
stable, so only the reaction constrains dt. First order in dt.

*** THE OUTPUT IS STABLE BUT NOT dt-CONVERGED. UNCALIBRATED — see D-EVID-17. ***
Refining dt -> dt/4 moves Gierer-Meinhardt's pattern contrast ~34 % (0.877 -> 0.577)
and shifts k* by a whole FFT bin. That is NOT attractor selection: the control is 6
seeds at fixed dt, which spread only 2.6 % with k* pinned to one bin — so the dt
dependence is ~13x the seed variation. Worse, the REFINED answers agree with each
other far better than with the shipped one: across two schemes and two refinements the
contrast clusters at 0.45-0.58, while the shipped dt alone gives ~0.88. The shipped
timestep is the outlier. Frames from this path are therefore fine for plumbing and for
morphology, but NO k* or contrast value measured from them may be quoted as a
reference number until this is closed.

Two candidate fixes were tried and rejected on measurement: symmetric (Strang)
splitting changed nothing (34 % -> 35 %) while costing an extra FFT pair per step, and
an RK2 midpoint reaction substep was worse still — it destroyed pattern formation
outright at the shipped dt (contrast -> 0.000). The dominant error is therefore NOT
splitting order, and is not yet diagnosed.

FIXED 2026-08-11 (D-EVID-17). This docstring previously claimed "IMEX split-step"
while the loop was `field + dt*(D*lap + reaction)` — fully EXPLICIT in both halves,
so the promised unconditional stability did not exist. The timestep was capped by
the reaction Jacobian alone (`0.2/max|eig J|`) and never by diffusion, which made
BOTH shipped reference systems diverge at their own defaults: `FloatingPointError`
at step 133 for gierer_meinhardt (Dv=100) and step 29 for schnakenberg (Dv=40).
At resolution=128, L=100 the explicit diffusion bound is 6.2e-4 and 1.5e-3
respectively, while the reaction cap left dt=0.1 — 162x and 65x too large.

Making the old scheme merely CFL-safe would have cost 2.6-6.5 million steps per
frame (~1.4-3.6 h at the measured 2 ms/step), which is presumably why the bound was
omitted. Solving diffusion exactly removes it instead: `stable_dt` still enforces
the explicit-diffusion bound so the returned step is safe under EITHER scheme, and
`simulate_to_attractor` stops when the field stops moving rather than always running
the full horizon.

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


def _k2_grid(n, L):
    """|k|^2 on the FFT grid. Shared by the diffusion solve and the stability bound."""
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    return KX**2 + KY**2


def stable_dt(system, resolution=128, dt_requested=0.1, scheme="explicit"):
    """Largest stable timestep for the requested `scheme`.

    Two bounds, whichever binds:

    * REACTION — explicit Euler on the reaction needs `dt * |lambda| <= 2`; we keep the
      original conservative `0.2 / max|eig J|` (a 10x margin), which is what stiff
      systems like Schnakenberg (gamma=100) actually need. Applies to both schemes.
    * DIFFUSION — with a SPECTRAL Laplacian the grid's largest diffusion eigenvalue is
      `D_max * k2_max`, and on an n-point periodic grid of size L the Nyquist mode gives
      `k2_max = 2*(pi*n/L)**2` in 2D. Explicit Euler then needs
      `dt * D_max * k2_max <= 2`. THIS BOUND WAS ABSENT and is the D-EVID-17 defect.

    `scheme="explicit"` (default) applies both — use it if you integrate diffusion
    explicitly. `scheme="imex"` applies only the reaction bound, which is correct when
    diffusion is solved exactly, as `simulate_to_attractor` does. The difference is not
    cosmetic: at the shipped 128/L=100, Gierer-Meinhardt is 6.3e-2 under imex against
    5.6e-4 under explicit, i.e. ~110x fewer steps for the same horizon.
    """
    if scheme not in ("explicit", "imex"):
        raise ValueError(f"scheme must be 'explicit' or 'imex', got {scheme!r}")
    xstar = np.asarray(system.steady_state(), float)
    J0 = _jac_at(system, xstar)
    rate = float(np.max(np.abs(np.linalg.eigvals(J0)))) if np.all(np.isfinite(J0)) else 0.0
    dt_reaction = 0.2 / (rate + 1e-9)

    if scheme == "imex":
        return float(min(dt_requested, dt_reaction))

    D_max = float(np.asarray(system.D, float).max())
    k2_max = 2.0 * (np.pi * resolution / float(system.L)) ** 2
    # 0.9 margin: at dt*D*k2 == 2 exactly the explicit amplification factor is -1, i.e.
    # NEUTRALLY stable — it oscillates instead of decaying. Sit strictly inside the disc.
    dt_diffusion = 0.9 * 2.0 / (D_max * k2_max) if D_max > 0 else np.inf

    return float(min(dt_requested, dt_reaction, dt_diffusion))


def simulate_to_attractor(system, resolution=128, T_max=4000.0, dt=0.1,
                          seed=0, noise_init=1e-2, atol=1e-6):
    """Integrate the reference system from a perturbed homogeneous state to a
    converged pattern. Returns (frame (n_species,H,W), L).

    Lie-split IMEX per step: explicit reaction, then EXACT diffusion in Fourier space.
    Returns early once the field stops moving — `atol` is on the per-unit-time rate of
    change, `max|dfield|/dt`, so the criterion is independent of the timestep. T_max is
    the ceiling, not the target.

    `atol=1e-6` is CALIBRATED, not inherited: the change-rate floor was measured on both
    shipped systems at n=64. Schnakenberg falls to 4.4e-13 within ~5k steps, but
    Gierer-Meinhardt PLATEAUS at ~4.2e-7 and never goes below it, while its pattern
    contrast is already stable to 4 significant figures (0.8771) from t~1250 onward. A
    tolerance below ~1e-6 therefore never fires for Gierer-Meinhardt and the run burns the
    whole horizon for no change in the answer. Note this is a stopping heuristic only —
    T_max still bounds the run, so too tight an atol costs time, never correctness.
    """
    rng = np.random.default_rng(seed)
    xstar = np.asarray(system.steady_state(), float)
    N = xstar.size
    n = resolution
    L = float(system.L)
    D = np.asarray(system.D, float)
    field = (xstar[:, None, None]
             + noise_init * rng.standard_normal((N, n, n)))

    # imex: diffusion is solved exactly below, so only the reaction constrains dt.
    dt = stable_dt(system, resolution=n, dt_requested=dt, scheme="imex")
    n_steps = int(np.ceil(T_max / dt))
    # Exact diffusion propagator for one step: u_hat <- exp(-D k^2 dt) u_hat.
    decay = np.exp(-D[:, None, None] * _k2_grid(n, L)[None, :, :] * dt)

    for step in range(n_steps):
        previous = field
        # Lie splitting: explicit-Euler reaction, then EXACT diffusion.
        reacted = field + dt * _reaction_grid(system, field)
        field = np.real(np.fft.ifft2(decay * np.fft.fft2(reacted, axes=(-2, -1)),
                                     axes=(-2, -1)))
        if not np.all(np.isfinite(field)):
            raise FloatingPointError(f"solver diverged at step {step}; reduce dt")
        if np.abs(field - previous).max() / dt < atol:
            break
    return field.astype(np.float32), L


def _reaction_grid(system, field):
    """Pointwise reaction over a grid. field: (N,H,W) -> (N,H,W).

    The reference reactions are elementwise closed forms, so passing the component
    planes straight through system.reaction vectorises over the whole grid in one
    call (u, v = field unpacks to (H,W) arrays)."""
    out = system.reaction(field)            # np.array([f_plane, g_plane, ...])
    return np.asarray(out, float).reshape(field.shape)
