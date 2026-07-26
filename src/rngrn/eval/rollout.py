"""rollout.py — lift-and-simulate: does the recovered circuit actually pattern?

Training constrains the reaction pointwise; it does NOT simulate the network.
This integrates the full lifted PDE from x* + small noise and measures whether a
pattern forms and at what k*. The integrator is pluggable (numerics.INTEGRATORS);
ETDRK4 is the stiff-safe default. The timestep/horizon are growth-rate-aware:
diffusion is handled inside the integrator (not a CFL limit), dt is set by the
fastest reaction rate, and the horizon is ~40x the growth timescale 1/sigma_max.
"""
from __future__ import annotations
import numpy as np
import torch

from .. import observables as obs
from .numerics import INTEGRATORS


def _reaction_np_builder(model):
    KA = model.KA.detach().cpu().numpy(); KR = model.KR.detach().cpu().numpy()
    alpha = model.alpha.detach().cpu().numpy(); beta = model.beta.detach().cpu().numpy()
    delta = model.delta.detach().cpu().numpy(); n_h = model.n_hill
    form = model.form

    def reaction_np(X):  # X: (N,n,n)
        xn = np.clip(X, 0, None) ** n_h
        if form == 'competitive':
            denom = 1.0 + np.einsum('ij,jxy->ixy', KA + KR, xn)
            prod = np.einsum('ij,ij,jxy->ixy', alpha, KA, xn) / denom
        else:
            thA = KA[:, :, None, None] * xn[None] / (1 + KA[:, :, None, None] * xn[None])
            thR = KR[:, :, None, None] * xn[None] / (1 + KR[:, :, None, None] * xn[None])
            act = np.einsum('ij,ijxy->ixy', alpha, thA)
            veto = np.prod(1 - thR, axis=1)
            prod = act * veto
        return beta[:, None, None] + prod - delta[:, None, None] * X
    return reaction_np


def simulate(model, L, n=128, T=None, dt=None, seed=0, noise=1e-2, xstar=None,
             integrator="etdrk4", horizon_growth_times=40.0, record_kstar=True):
    """Integrate d x/dt = D lap(x) + f(x) from x* + noise. Returns a result dict."""
    rng = np.random.default_rng(seed)
    N = model.N
    D = model.D.detach().cpu().numpy()
    if xstar is None:
        from ..losses.terms import steady_state
        xs, _ = steady_state(model); xstar = xs.detach().cpu().numpy()

    # growth-rate-aware dt and horizon
    xs_t = torch.tensor(xstar, device=model.device, dtype=model.dtype)
    Jn = model.jacobian(xs_t, create_graph=False).detach().cpu().numpy()
    kg = np.linspace(1e-3, 2 * np.pi * (n // 2) / L, 2000)
    sigd = np.array([np.max(np.real(np.linalg.eigvals(Jn - kk**2 * np.diag(D)))) for kk in kg])
    sig_max = max(sigd.max(), 1e-3)
    jac_rate = float(np.max(np.abs(np.linalg.eigvals(Jn))))
    if dt is None: dt = 0.2 / (jac_rate + 1e-9)
    if T is None:  T = horizon_growth_times / sig_max
    nsteps = int(np.clip(T / dt, 200, 200000))

    X0 = xstar[:, None, None] + noise * rng.standard_normal((N, n, n))
    reaction_np = _reaction_np_builder(model)
    if integrator not in INTEGRATORS:
        raise KeyError(f"unknown integrator '{integrator}'; have {sorted(INTEGRATORS)}")
    X, blew_up = INTEGRATORS[integrator](X0, D, reaction_np, n, L, dt, nsteps)
    if blew_up:
        return dict(fields=X, kstar=np.nan, patterned=False, amplitude=np.nan,
                    blew_up=True, integrator=integrator, nsteps=nsteps, dt=float(dt))
    amp = float(X[0].std())
    ks = obs.kstar_of(X[0], L=L) if (record_kstar and amp > 1e-6) else np.nan
    patterned = amp > max(1e-3, 0.02 * abs(xstar[0]))
    return dict(fields=X, kstar=ks, patterned=bool(patterned), amplitude=amp,
                blew_up=False, integrator=integrator, nsteps=nsteps, dt=float(dt),
                sig_max=float(sig_max))
