"""test_lift_ladder.py — V0 of the lift-validation ladder: algebraic invariants of the
lifted system (eval/ladder.py), at N=3, both regulation forms, over >=20 generator draws.

docs/REDESIGN_rngrn.md §5.3 V0 says V0 MEASURES the achievable floor and records it rather
than asserting an inherited one. The original draft of this test asserted the single-model
1e-7 bar from tests/test_lifted.py directly and found it did NOT hold over a 20-draw
population (task-2-report.md §4: competitive 2.56e-7, nc1 1.27e-6). CONTROLLER RULING
ROUND 1 (ledger 2026-08-17) on that finding, implemented here and in eval/ladder.py's module
docstring:
  1. draw_models now filters to NEWTON-TIGHT steady states only (excludes
     steady_state's looser relaxation-fallback acceptances) -- closes the mechanism where a
     loose x* (not the lift algebra) was driving the residual up.
  2. The remaining mu-dependence (a real property of eval/lifted.py's own arithmetic, not a
     ladder.py defect -- see ladder.py's module docstring, mechanism (b)) is captured by a
     MEASURED, mu-aware bound instead of a fixed constant.

RULING ROUND 2 (ledger 2026-08-17, code review of round 1). Round 1's single amplification
constant A was measured over the WHOLE mu grid, so it was FLOOR-dominated -- the x-block
round-off floor (flat in mu) sets the max of `residual*mu/eps` at the LARGEST mu, not the
smallest, making the resulting bound ~5 orders of magnitude too loose exactly where V1
(mu in {1e-7, 1e-6, 1e-5}) needs it tight. Round 2 measures the two mechanisms SEPARATELY,
each over only the mu region where it actually dominates -- see eval/ladder.py's module
docstring and `v0_invariants`'s docstring for the full mechanism writeup:
  - `residual_floor_F` = max residual over the FLAT region (mu in {1e-2, 1, 1e2}) -- the
    x-block round-off floor, mu-independent.
  - `amplification_C` = max of `residual(mu) * mu / eps` over the GATE-AMPLIFICATION region
    (mu in {1e-6, 1e-4}) -- the C/mu mechanism proper.
"""
import pytest

from rngrn.eval import ladder

MUS = [1e-6, 1e-4, 1e-2, 1.0, 1e2]

# float64 machine epsilon -- must match ladder.EPS64 exactly (repeated here rather than
# imported so the bound in this test is legible on its own).
EPS64 = 2.220446049250313e-16

# F_FORM (residual_floor_F) and C_FORM (amplification_C), MEASURED (ruling round 2, ledger
# 2026-08-17) by draw_models(n=20, form=form, seed=101) + v0_invariants(models, mus=MUS) on
# this branch, 2026-08-17:
#   competitive: F=6.266787e-11, C=1154.500000  (exclusion_rate 0/20 steady_state-converged)
#   nc1:         F=2.795897e-11, C=12.591600    (exclusion_rate 1/21 steady_state-converged)
# Frozen here with a small (<1%) rounding-up margin above the measured value; the per-mu
# bound below already carries its own 10x safety factor on top of that. UNCALIBRATED ->
# measured, per docs/DECISIONS.md discipline (CLAUDE.md §10) -- see task-2-report.md §§4,8,9
# for the full measurement history and the two-round decision-point exchange that produced
# this ruling.
F_FORM = {"competitive": 6.3e-11, "nc1": 2.8e-11}
C_FORM = {"competitive": 1160.0, "nc1": 12.6}


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_v0_invariants_hold_at_the_measured_bar(form):
    models = ladder.draw_models(n=20, form=form, seed=101)
    out = ladder.v0_invariants(models, mus=MUS)
    assert out["n_models"] >= 20

    # rescale_mu / gate_qss: the brief's original bars, UNCHANGED by either ruling round --
    # both pass by 6+ orders of magnitude (measured ~1e-15..1e-16 here).
    assert out["max_rescale_mu_err"] < 1e-9
    assert out["max_gate_qss_err"] < 1e-9

    # fixed_point_residual: MEASURED, mu-aware, two-mechanism bound (ruling round 2, ledger
    # 2026-08-17), tight in both the flat-floor and the gate-amplification regime. Checked
    # per mu, not just on the aggregate max, so a bar violation concentrated at one mu
    # cannot hide behind a looser one elsewhere in the grid.
    f, c = F_FORM[form], C_FORM[form]
    for mu in MUS:
        resid = ladder.v0_invariants(models, mus=[mu])["max_fixed_point_residual"]
        bound = max(1e-9, 10.0 * f, 10.0 * c * EPS64 / mu)
        assert resid <= bound, (form, mu, resid, bound)


def test_draw_models_reports_exclusion_rate_and_v0_invariants_surfaces_it():
    """exclusion_rate is a deliverable (ruling, ledger 2026-08-17), not a hidden filter."""
    models = ladder.draw_models(n=20, form="nc1", seed=101)
    assert 0.0 <= models.exclusion_rate <= 1.0
    out = ladder.v0_invariants(models, mus=[1.0])
    assert out["exclusion_rate"] == models.exclusion_rate


def test_v0_invariants_reports_nan_exclusion_rate_for_a_plain_list():
    """A model list not produced by draw_models carries no exclusion-rate bookkeeping --
    v0_invariants must say so (NaN) rather than fabricate a rate."""
    import math

    models = list(ladder.draw_models(n=3, form="nc1", seed=101))  # unwrap to a plain list
    out = ladder.v0_invariants(models, mus=[1.0])
    assert math.isnan(out["exclusion_rate"])


def test_draw_models_is_deterministic():
    """Same (n, form, seed) -> the same model parameters, every time."""
    a = ladder.draw_models(n=5, form="nc1", seed=7)
    b = ladder.draw_models(n=5, form="nc1", seed=7)
    assert len(a) == len(b) == 5
    for ma, mb in zip(a, b):
        assert (ma.beta.detach().numpy() == mb.beta.detach().numpy()).all()
        assert (ma.KA.detach().numpy() == mb.KA.detach().numpy()).all()


def test_draw_models_raises_after_10n_unsuccessful_attempts(monkeypatch):
    """Fail loud rather than silently returning a short list. Steady-state convergence is
    forced to always fail, so the 10n-attempt budget is exhausted in n draws, not n.
    """
    monkeypatch.setattr(ladder, "steady_state", lambda m: (None, False))
    with pytest.raises(RuntimeError):
        ladder.draw_models(n=3, form="nc1", seed=0)
