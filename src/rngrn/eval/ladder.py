"""ladder.py — PROVENANCE: ladder scaffolding, validated by tests/test_lift_ladder.py.

Library home for every rung of the lift-validation ladder (docs/REDESIGN_rngrn.md §5.3,
V0-V4): each rung is `vN_invariants(...)`, run against draws from `draw_models`, and each
licenses the next. This module currently implements V0 only; later tasks add v1-v4 here.

V0 — algebraic invariants (§5.3): the lifted fixed point equals the QSS fixed point at
EVERY mu (`lifted.fixed_point_residual`, bar 1e-7 — the bar tests/test_lifted.py already
holds on one hand-picked model, extended here to >=20 generator draws per form);
`rescale_mu` reproduces a fresh autodiff Jacobian (`lifted.lifted_jacobian`); and the QSS
gate occupancies fed through `production_from_gates` reproduce `model.reaction` exactly
(`lifted.gates_qss`). All three are proved once, algebraically, in eval/lifted.py's own
module docstring; V0 only extends the *sample size* the claim is checked over. Reuses
eval/lifted.py's functions verbatim -- no algebra is re-derived here.
"""
from __future__ import annotations

import numpy as np
import torch

from ..losses.terms import steady_state
from ..model import RNGRN
from . import lifted


def draw_models(n: int, form: str, seed: int, N: int = 3) -> list:
    """n RNGRN(N, form, init="low_basal", seed=...) models with a converged steady state.

    Deterministic given (n, form, seed): seeds are tried in the fixed sequence
    seed, seed+1, seed+2, ... regardless of which draws converge, so the same call always
    returns the same models. A draw whose steady_state does not converge is skipped, not
    counted toward n. Fails LOUD (raises) after 10*n unsuccessful (non-converged) draws --
    a shortfall is never returned silently.
    """
    models = []
    n_failures = 0
    max_failures = 10 * n
    s = seed
    while len(models) < n:
        m = RNGRN(N=N, form=form, init="low_basal", seed=s)
        s += 1
        _, converged = steady_state(m)
        if converged:
            models.append(m)
        else:
            n_failures += 1
            if n_failures >= max_failures:
                raise RuntimeError(
                    f"draw_models: {n_failures} unconverged draws (limit "
                    f"{max_failures} = 10*n) while collecting n={n} form={form!r} models "
                    f"starting from seed={seed}; only {len(models)} converged. This is a "
                    "shortfall, not a partial result -- raising rather than returning it.")
    return models


def v0_invariants(models: list, mus: list[float]) -> dict:
    """V0 rung: algebraic invariants of the lifted system, over every model and every mu.

    For each model: the gate-occupancy invariant (mu-independent) is checked once; the
    fixed-point-residual and rescale_mu invariants are checked at EVERY mu in `mus`. Every
    value returned is the WORST (max) observed across the whole (model, mu) grid -- this is
    a floor measurement, not a spot check, per docs/REDESIGN_rngrn.md §5.3 V0.

    Returns {"max_fixed_point_residual", "max_rescale_mu_err", "max_gate_qss_err",
    "n_models"}. `max_rescale_mu_err` is RELATIVE (max abs diff / max abs of the fresh
    Jacobian), matching tests/test_lifted.py::test_rescale_mu_matches_autodiff -- the gate
    rows of J_full scale as 1/mu, so an ABSOLUTE tolerance would not be comparable across
    the mu range in `mus` (a 1e-6 relative slip at mu=1e-6 is a 1.0-scale absolute one).
    `max_gate_qss_err` is absolute, matching
    tests/test_lifted.py::test_gates_qss_reproduce_the_qss_reaction (both f_lift and f_qss
    are O(x*), no scale sweep involved).
    """
    max_fp_residual = 0.0
    max_rescale_err = 0.0
    max_gate_err = 0.0
    for m in models:
        xs, converged = steady_state(m)
        if not converged:
            raise RuntimeError(
                "v0_invariants: received a model with no converged steady state -- "
                "draw_models should already have filtered this out")
        xstar = xs.detach().cpu().numpy()

        GA, GR = lifted.gates_qss(m, xstar)
        f_lift = (m.beta.detach().cpu().numpy() + lifted.production_from_gates(m, GA, GR)
                  - m.delta.detach().cpu().numpy() * xstar)
        f_qss = m.reaction(
            torch.as_tensor(xstar, device=m.device, dtype=m.dtype)).detach().cpu().numpy()
        max_gate_err = max(max_gate_err, float(np.max(np.abs(f_lift - f_qss))))

        J1 = lifted.lifted_jacobian(m, xstar, 1.0)
        for mu in mus:
            max_fp_residual = max(
                max_fp_residual, lifted.fixed_point_residual(m, xstar, mu))
            Jd = lifted.lifted_jacobian(m, xstar, mu)
            Jr = lifted.rescale_mu(J1, m.N, mu)
            rel = float(np.max(np.abs(Jr - Jd))) / max(float(np.max(np.abs(Jd))), 1e-300)
            max_rescale_err = max(max_rescale_err, rel)

    return dict(max_fixed_point_residual=max_fp_residual,
               max_rescale_mu_err=max_rescale_err,
               max_gate_qss_err=max_gate_err,
               n_models=len(models))
