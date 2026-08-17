"""test_lift_ladder.py — V0 of the lift-validation ladder: algebraic invariants of the
lifted system (eval/ladder.py), at N=3, both regulation forms, over >=20 generator draws.

docs/REDESIGN_rngrn.md §5.3 V0: the fixed-point residual bar (1e-7) is the bar
tests/test_lifted.py already holds on a single hand-picked model; this file extends the
same claim to a population of generator draws. The achievable floor is UNCALIBRATED and is
a MEASURED deliverable of the V0 script (scripts/, later task), not asserted here — this
test only pins the pre-registered bar.
"""
import pytest

from rngrn.eval import ladder


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_v0_invariants_hold_at_the_existing_bar(form):
    models = ladder.draw_models(n=20, form=form, seed=101)
    out = ladder.v0_invariants(models, mus=[1e-6, 1e-4, 1e-2, 1.0, 1e2])
    assert out["n_models"] >= 20
    # Bar = the bar the existing tests already hold (tests/test_lifted.py: 1e-7).
    # The achievable floor is UNCALIBRATED and is MEASURED by the v0 script, not
    # asserted here — docs/REDESIGN_rngrn.md §5.3 V0.
    assert out["max_fixed_point_residual"] < 1e-7
    assert out["max_rescale_mu_err"] < 1e-9
    assert out["max_gate_qss_err"] < 1e-9


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
