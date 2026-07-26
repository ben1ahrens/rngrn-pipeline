"""analysis.py — linear stability, Turing conditions, and the robustness cloud.

Reads only the recovered model (firewall-safe). turing_ok evaluates the Turing
conditions on a numpy J, D; robustness_cloud is the payoff metric (fraction of a
log-normal parameter cloud that still patterns + k* spread).

PROVENANCE: repackaged from rngrn_eval.py, which the project tracks as SCAFFOLD
(not yet validated), unlike the VALIDATED model/observables core. The Turing-
condition and dispersion logic here still needs validation by Claude Code (a
science test against the reference answer keys is the natural anchor).
"""
from __future__ import annotations
import numpy as np
import torch

from ..model import RNGRN


def turing_ok(J, D, kgrid=None):
    """Evaluate the Turing conditions on numpy J, D. Returns (ok: bool, info dict)."""
    J = np.asarray(J); D = np.asarray(D)
    if kgrid is None:
        kgrid = np.linspace(1e-3, 50, 4000)
    tr0 = np.trace(J)
    sig = np.array([np.max(np.real(np.linalg.eigvals(J - k**2 * np.diag(D)))) for k in kgrid])
    stable_uniform = tr0 < 0
    unstable_struct = sig.max() > 1e-9
    ok = bool(stable_uniform and unstable_struct)
    return ok, dict(tr0=float(tr0), sig_max=float(sig.max()),
                    kstar=float(kgrid[np.argmax(sig)]),
                    stable_uniform=bool(stable_uniform),
                    unstable_struct=bool(unstable_struct))


def linear_stability(model, xstar):
    """Full linear-stability readout at a given steady state (numpy)."""
    xs_t = torch.as_tensor(np.asarray(xstar, float))
    J = model.jacobian(xs_t, create_graph=False).detach().cpu().numpy()
    D = model.D.detach().cpu().numpy()
    ok, info = turing_ok(J, D)
    info["J"] = J.tolist(); info["D"] = D.tolist()
    info["turing"] = ok
    return info


def robustness_cloud(model, n_samples=200, sigma_log=0.1, seed=0):
    """Log-normal multiplicative perturbation of recovered params; fraction still Turing
    and k* spread. Rebuilds the reaction each draw, re-solves x*, checks Turing conds."""
    from ..losses.terms import steady_state
    rng = np.random.default_rng(seed)
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    oks = []; kstars = []
    for _ in range(n_samples):
        pert = RNGRN(N=model.N, form=model.form, n_hill=model.n_hill)
        with torch.no_grad():
            for nm, p in pert.named_parameters():
                factor = torch.tensor(rng.lognormal(0.0, sigma_log, size=tuple(p.shape)))
                p.copy_(base[nm] * factor)
        try:
            xs, conv = steady_state(pert)
            J = pert.jacobian(xs, create_graph=False).detach().cpu().numpy()
            ok, info = turing_ok(J, pert.D.detach().cpu().numpy())
            oks.append(ok)
            if ok and np.isfinite(info['kstar']):
                kstars.append(info['kstar'])
        except Exception:
            oks.append(False)
    kstars = np.array(kstars)
    return dict(frac_turing=float(np.mean(oks)), n=n_samples, n_turing=int(np.sum(oks)),
                kstar_mean=float(kstars.mean()) if len(kstars) else float("nan"),
                kstar_std=float(kstars.std()) if len(kstars) else float("nan"))
