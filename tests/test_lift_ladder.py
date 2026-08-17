"""test_lift_ladder.py — V0 of the lift-validation ladder: algebraic invariants of the
lifted system (eval/ladder.py), at N=3, both regulation forms, over >=20 generator draws.

docs/REDESIGN_rngrn.md §5.3 V0 says V0 MEASURES the achievable floor and records it rather
than asserting an inherited one. The original draft of this test asserted the single-model
1e-7 bar from tests/test_lifted.py directly and found it did NOT hold over a 20-draw
population (task-2-report.md §4: competitive 2.56e-7, nc1 1.27e-6). CONTROLLER RULING
(ledger 2026-08-17) on that finding, implemented here and in eval/ladder.py's module
docstring:
  1. draw_models now filters to NEWTON-TIGHT steady states only (excludes
     steady_state's looser relaxation-fallback acceptances) -- closes the mechanism where a
     loose x* (not the lift algebra) was driving the residual up.
  2. The remaining mu-dependence (a real property of eval/lifted.py's own arithmetic, not a
     ladder.py defect -- see ladder.py's module docstring, mechanism (b)) is captured by a
     MEASURED, mu-aware bound instead of a fixed constant.
"""
import pytest

from rngrn.eval import ladder

MUS = [1e-6, 1e-4, 1e-2, 1.0, 1e2]

# float64 machine epsilon -- must match ladder.EPS64 exactly (repeated here rather than
# imported so the bound in this test is legible on its own).
EPS64 = 2.220446049250313e-16

# Amplification constant A, MEASURED (ruling, ledger 2026-08-17) by
# draw_models(n=20, form=form, seed=101) + v0_invariants(models, mus=MUS) on this branch,
# 2026-08-17: competitive A=2.822310e7 (exclusion_rate 0/20 steady_state-converged draws),
# nc1 A=1.259160e7 (exclusion_rate 1/21 steady_state-converged draws -- one loose draw
# excluded, one extra seed drawn to still reach n=20 Newton-tight). Frozen here with ~3%
# headroom above the measured value; the mu-aware bound below already carries its own 10x
# safety factor on top of that. UNCALIBRATED -> measured, per docs/DECISIONS.md discipline
# (CLAUDE.md §10) -- see task-2-report.md for the full measurement and the decision-point
# exchange that produced this ruling.
A_FORM = {"competitive": 2.9e7, "nc1": 1.3e7}


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_v0_invariants_hold_at_the_measured_bar(form):
    models = ladder.draw_models(n=20, form=form, seed=101)
    out = ladder.v0_invariants(models, mus=MUS)
    assert out["n_models"] >= 20

    # rescale_mu / gate_qss: the brief's original bars, UNCHANGED by the ruling -- both
    # pass by 6+ orders of magnitude (measured ~1e-15..1e-16 here).
    assert out["max_rescale_mu_err"] < 1e-9
    assert out["max_gate_qss_err"] < 1e-9

    # fixed_point_residual: MEASURED, mu-aware bound (ruling, ledger 2026-08-17). Checked
    # per mu, not just on the aggregate max, so a bar violation concentrated at one mu
    # cannot hide behind a looser one elsewhere in the grid.
    a = A_FORM[form]
    for mu in MUS:
        resid = ladder.v0_invariants(models, mus=[mu])["max_fixed_point_residual"]
        bound = max(1e-9, 10.0 * a * EPS64 / mu)
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
