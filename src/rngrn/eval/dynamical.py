"""dynamical.py — the QSS -> dynamical lift and its mu->0 consistency check.

Training uses the quasi-steady-state (QSS) reaction. The lift makes the promoter
gates explicit fast variables with timescale mu; as mu->0 the lift must reproduce
the QSS reaction exactly. lift_check returns the max abs discrepancy (asserted
small in tests/test_science.py). Repackaged from rngrn_eval.py (tracked as SCAFFOLD).
"""
from __future__ import annotations
import numpy as np
import torch


def qss_reaction_np(model, x):
    with torch.no_grad():
        return model.reaction(torch.tensor(x, device=model.device, dtype=model.dtype)).detach().cpu().numpy()


def lift_check(model, x, mu=1e-4, n=4000, dt=None):
    """Integrate lifted promoter gates to equilibrium at fixed x, form the production
    term, compare to the QSS reaction. Returns max abs difference (-> 0 as mu -> 0)."""
    N = model.N; n_h = model.n_hill
    KA = model.KA.detach().cpu().numpy(); KR = model.KR.detach().cpu().numpy()
    alpha = model.alpha.detach().cpu().numpy(); beta = model.beta.detach().cpu().numpy()
    delta = model.delta.detach().cpu().numpy()
    x = np.asarray(x, float); xn = np.clip(x, 0, None) ** n_h
    GA = np.zeros((N, N)); GR = np.zeros((N, N))
    if dt is None: dt = mu * 0.1
    for _ in range(n):
        if model.form == 'competitive':
            free = 1.0 - GA.sum(1, keepdims=True) - GR.sum(1, keepdims=True)
            dGA = (KA * xn[None, :] * free - GA) / mu
            dGR = (KR * xn[None, :] * free - GR) / mu
        else:
            dGA = (KA * xn[None, :] * (1 - GA) - GA) / mu
            dGR = (KR * xn[None, :] * (1 - GR) - GR) / mu
        GA = GA + dt * dGA; GR = GR + dt * dGR
    if model.form == 'competitive':
        prod = (alpha * GA).sum(1)
    else:
        prod = (alpha * GA).sum(1) * np.prod(1 - GR, axis=1)
    f_lift = beta + prod - delta * x
    f_qss = qss_reaction_np(model, x)
    return float(np.max(np.abs(f_lift - f_qss)))
