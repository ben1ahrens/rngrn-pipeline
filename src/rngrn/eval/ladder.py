"""ladder.py — PROVENANCE: ladder scaffolding, validated by tests/test_lift_ladder.py.

Library home for every rung of the lift-validation ladder (docs/REDESIGN_rngrn.md §5.3,
V0-V4): each rung is `vN_invariants(...)`, run against draws from `draw_models`, and each
licenses the next. This module currently implements V0 only; later tasks add v1-v4 here.

V0 — algebraic invariants (§5.3): the lifted fixed point equals the QSS fixed point at
EVERY mu (`lifted.fixed_point_residual`); `rescale_mu` reproduces a fresh autodiff Jacobian
(`lifted.lifted_jacobian`); and the QSS gate occupancies fed through `production_from_gates`
reproduce `model.reaction` exactly (`lifted.gates_qss`). All three are proved once,
algebraically, in eval/lifted.py's own module docstring; V0 only extends the *sample size*
the claim is checked over. Reuses eval/lifted.py's functions verbatim -- no algebra is
re-derived here.

CONTROLLER RULING (ledger 2026-08-17, on Task 2's decision point). The single-model 1e-7
fixed-point-residual bar (tests/test_lifted.py, one hand-picked seed per form) does NOT hold
over a 20-draw low_basal population at N=3 (measured: competitive 2.56e-7, nc1 1.27e-6 --
see task-2-report.md). Two distinct causes were found, and the ruling addresses each
differently:
  (a) some low_basal draws are accepted by losses.terms.steady_state's own `converged` flag
      via its LOOSE relaxation-fallback path (||f(x*)|| < 1e-4), not the tight Newton path
      (||f(x*)|| < 1e-10, losses/terms.py:36-59,129). The lifted fixed point's mu-independence
      is proved for an EXACT root of the reaction; a loose x* measures steady_state's own
      solver quality, not the lift algebra. RULING: `draw_models` now filters to
      Newton-tight draws only (||f(x*)|| <= 1e-10), reporting the exclusion rate as a
      first-class deliverable rather than a hidden filter (see `ModelDraws.exclusion_rate`).
  (b) even at a Newton-tight x*, `fixed_point_residual` still grows like C/mu at the smallest
      tested mu: eval/lifted.py independently reconstructs the gate normalisation `free` two
      ways (gates_qss's closed form vs lifted_rhs_torch's `1 - GA.sum - GR.sum`), algebraically
      identical but not bit-identical, so their float64 difference (~1e-13-1e-16) gets divided
      by mu inside the residual. This is a real property of the EXISTING eval/lifted.py code,
      not a defect, and not something V0 re-derives or patches.

RULING ROUND 2 (ledger 2026-08-17, review of round 1's fix). Round 1's single amplification
constant A was measured as the max of `residual(mu) * mu / eps` over the WHOLE (model, mu)
grid, including the flat-in-mu region -- so it was FLOOR-dominated (the x-block round-off
floor, mechanism (b)'s "flat" part below, sets the max of that ratio at the LARGEST mu, not
the smallest), producing a bound ~5 orders of magnitude too loose exactly where V1 (mu in
{1e-7, 1e-6, 1e-5}) needs it tight. Two DISTINCT residual components, honestly measured
separately, replace the single A:
  - `residual_floor_F` = max fixed-point residual over mu in the FLAT region ({1e-2, 1, 1e2}
    in the standard mu list) -- this is the x-BLOCK round-off floor: `model.reaction(x*)`
    itself, mu-INDEPENDENT (it does not touch the gate rows at all).
  - `amplification_C` = max of `residual(mu) * mu / eps` over mu in the GATE-BLOCK
    amplification region ({1e-6, 1e-4} in the standard mu list) -- the C/mu mechanism proper,
    isolated from the floor by only sampling the region where it actually dominates.
tests/test_lift_ladder.py freezes both, per form, into
`max_residual(mu) <= max(1e-9, 10 * F_FORM, 10 * C_FORM * eps / mu)`, tight in both regimes.
"""
from __future__ import annotations

import numpy as np
import torch

from ..losses.terms import steady_state
from ..model import RNGRN
from . import lifted

# Newton-tight acceptance bar for draw_models (ruling, ledger 2026-08-17): matches
# losses/terms.py::_damped_newton's own convergence tolerance (tol=1e-10) on the SAME norm
# (torch.linalg.norm, Euclidean 2-norm) it uses internally, so "Newton-tight" here means
# exactly what steady_state's fast path itself would have accepted -- not an independently
# invented threshold.
NEWTON_TIGHT_TOL = 1e-10

# float64 machine epsilon, used to define `amplification_C` (ruling round 2, ledger
# 2026-08-17).
EPS64 = 2.220446049250313e-16

# The two mu regions ruling round 2 (ledger 2026-08-17) splits `fixed_point_residual` into,
# matching the standard mu list [1e-6, 1e-4, 1e-2, 1.0, 1e2] this ladder rung is measured on:
# mu >= FLOOR_REGION_MU_MIN is the flat, x-block-round-off-dominated region (residual_floor_F);
# mu <= AMPLIFICATION_REGION_MU_MAX is the gate-block 1/mu-amplified region (amplification_C).
# A mu strictly between the two (not present in the standard list) contributes to neither.
FLOOR_REGION_MU_MIN = 1e-2
AMPLIFICATION_REGION_MU_MAX = 1e-4


class ModelDraws(list):
    """list[RNGRN] returned by draw_models(), carrying its own sampling diagnostics.

    `exclusion_rate` (ruling, ledger 2026-08-17): of the draws losses.terms.steady_state
    ITSELF marked converged=True, the fraction that were nonetheless excluded here for being
    loose (||f(x*)|| > NEWTON_TIGHT_TOL, i.e. accepted only via steady_state's looser
    relaxation-fallback threshold of 1e-4, not its Newton path's 1e-10). NaN if draw_models
    never saw a steady_state-converged draw (e.g. n=0).
    """
    exclusion_rate: float = float("nan")


def draw_models(n: int, form: str, seed: int, N: int = 3) -> ModelDraws:
    """n RNGRN(N, form, init="low_basal", seed=...) models with a NEWTON-TIGHT steady state:
    ||model.reaction(x*)|| <= NEWTON_TIGHT_TOL (1e-10, the 2-norm losses.terms.steady_state's
    own Newton path uses internally). STRICTER than steady_state's own `converged` flag,
    which also accepts its looser relaxation-fallback path (up to 1e-4) -- excluded here per
    the controller ruling (ledger 2026-08-17, module docstring): the lift's algebraic
    invariants presume an EXACT fixed point, so a loose x* would measure solver quality, not
    the lift.

    Deterministic given (n, form, seed): seeds are tried in the fixed sequence
    seed, seed+1, seed+2, ... regardless of which draws are kept, loose-excluded, or hard
    failures, so the same call always returns the same models. A hard failure (steady_state
    itself does not converge) and a loose exclusion (converged, but not Newton-tight) both
    count toward the same fail-loud budget: raises after 10*n such draws rather than ever
    returning a shortfall silently.

    Returns a `ModelDraws` (list[RNGRN] + `.exclusion_rate`, see its docstring) -- the
    exclusion rate this call measured is a deliverable, read back out by `v0_invariants`.
    """
    models = ModelDraws()
    n_failures = 0
    max_failures = 10 * n
    n_steady_state_converged = 0   # denominator: draws steady_state itself accepted
    n_loose = 0                    # numerator: of those, how many were not Newton-tight
    s = seed
    while len(models) < n:
        m = RNGRN(N=N, form=form, init="low_basal", seed=s)
        s += 1
        xs, converged = steady_state(m)
        if not converged:
            n_failures += 1
        else:
            n_steady_state_converged += 1
            resid = float(torch.linalg.norm(m.reaction(xs)).detach())
            if resid <= NEWTON_TIGHT_TOL:
                models.append(m)
            else:
                n_loose += 1
                n_failures += 1
        if n_failures >= max_failures and len(models) < n:
            raise RuntimeError(
                f"draw_models: {n_failures} unsuccessful draws (limit "
                f"{max_failures} = 10*n; {n_loose} of those were steady_state-converged but "
                f"not Newton-tight) while collecting n={n} form={form!r} models starting "
                f"from seed={seed}; only {len(models)} collected. This is a shortfall, not "
                "a partial result -- raising rather than returning it.")
    models.exclusion_rate = (
        n_loose / n_steady_state_converged if n_steady_state_converged else float("nan"))
    return models


def v0_invariants(models: list, mus: list[float]) -> dict:
    """V0 rung: algebraic invariants of the lifted system, over every model and every mu.

    For each model: the gate-occupancy invariant (mu-independent) is checked once; the
    fixed-point-residual and rescale_mu invariants are checked at EVERY mu in `mus`. Every
    value returned is the WORST (max) observed across the whole (model, mu) grid -- this is
    a floor measurement, not a spot check, per docs/REDESIGN_rngrn.md §5.3 V0.

    Returns {"max_fixed_point_residual", "max_rescale_mu_err", "max_gate_qss_err",
    "n_models", "exclusion_rate", "residual_floor_F", "amplification_C"}.
    `max_rescale_mu_err` is RELATIVE (max abs diff / max abs of the fresh Jacobian), matching
    tests/test_lifted.py::test_rescale_mu_matches_autodiff -- the gate rows of J_full scale
    as 1/mu, so an ABSOLUTE tolerance would not be comparable across the mu range in `mus`
    (a 1e-6 relative slip at mu=1e-6 is a 1.0-scale absolute one). `max_gate_qss_err` is
    absolute, matching tests/test_lifted.py::test_gates_qss_reproduce_the_qss_reaction (both
    f_lift and f_qss are O(x*), no scale sweep involved).

    CONTROLLER RULING ROUND 2 (ledger 2026-08-17, module docstring has the full rationale):
    `exclusion_rate` is read back from `models.exclusion_rate` if `models` came from
    `draw_models` (a `ModelDraws`); NaN for a plain list, since the rate is meaningless
    without draw_models's own bookkeeping of steady_state-converged-but-loose draws.
    `fixed_point_residual(mu)` has TWO honestly-distinct components, and round 2 replaced
    round 1's single floor-dominated `amplification_A` with one measurement per component:
      - `residual_floor_F` = max residual over mu in the FLAT region (mu >=
        FLOOR_REGION_MU_MIN) -- the x-BLOCK round-off floor, i.e. how far
        `model.reaction(x*)` itself sits from zero; mu-INDEPENDENT, because the gate rows of
        the lifted RHS do not appear in this component at all.
      - `amplification_C` = max of `residual(mu) * mu / EPS64` over mu in the
        GATE-AMPLIFICATION region (mu <= AMPLIFICATION_REGION_MU_MAX) -- the C/mu mechanism
        proper (task-2-report.md §4 mechanism (a): eval/lifted.py's two independent,
        not-bit-identical reconstructions of the gate normalisation `free`), isolated from
        the floor by only sampling the region where it actually dominates.
    Both are MEASUREMENTS this function reports, not bars it enforces; the caller
    (tests/test_lift_ladder.py) freezes both, per form, into a bound tight in both regimes.
    """
    max_fp_residual = 0.0
    max_rescale_err = 0.0
    max_gate_err = 0.0
    residual_floor_F = 0.0
    amplification_C = 0.0
    for m in models:
        xs, converged = steady_state(m)
        if not converged:
            raise RuntimeError(
                "v0_invariants: received a model with no converged steady state -- "
                "draw_models should already have filtered this out")
        xstar = xs.detach().cpu().numpy()

        GA, GR = lifted.gates_qss(m, xstar)
        # Mirrors eval/lifted.py::lifted_rhs_torch's `dx = model.beta + prod - model.delta *
        # x` line (numpy here vs torch there, since this feeds a numpy f_qss comparison
        # below) -- a shared helper is deferred to Task 7's single-sourcing decision rather
        # than introduced here as a one-off (Minor, code review round 2).
        f_lift = (m.beta.detach().cpu().numpy() + lifted.production_from_gates(m, GA, GR)
                  - m.delta.detach().cpu().numpy() * xstar)
        f_qss = m.reaction(
            torch.as_tensor(xstar, device=m.device, dtype=m.dtype)).detach().cpu().numpy()
        max_gate_err = max(max_gate_err, float(np.max(np.abs(f_lift - f_qss))))

        J1 = lifted.lifted_jacobian(m, xstar, 1.0)
        for mu in mus:
            resid = lifted.fixed_point_residual(m, xstar, mu)
            max_fp_residual = max(max_fp_residual, resid)
            if mu >= FLOOR_REGION_MU_MIN:
                residual_floor_F = max(residual_floor_F, resid)
            if mu <= AMPLIFICATION_REGION_MU_MAX:
                amplification_C = max(amplification_C, resid * mu / EPS64)
            Jd = lifted.lifted_jacobian(m, xstar, mu)
            Jr = lifted.rescale_mu(J1, m.N, mu)
            rel = float(np.max(np.abs(Jr - Jd))) / max(float(np.max(np.abs(Jd))), 1e-300)
            max_rescale_err = max(max_rescale_err, rel)

    exclusion_rate = getattr(models, "exclusion_rate", float("nan"))
    return dict(max_fixed_point_residual=max_fp_residual,
               max_rescale_mu_err=max_rescale_err,
               max_gate_qss_err=max_gate_err,
               n_models=len(models),
               exclusion_rate=exclusion_rate,
               residual_floor_F=residual_floor_F,
               amplification_C=amplification_C)
