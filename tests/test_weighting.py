"""test_weighting.py — unit 13: ratio-based loss balancing + stub strategies.

Covers RatioWeighting (the Matas-Gil & Endres ratio scheme) and the gradnorm/ntk
stubs, which must now either work or raise NotImplementedError rather than
silently running with fixed weights.
"""
import pytest
import torch

from rngrn.losses.weighting import (
    WEIGHTING,
    RatioWeighting,
    GradNormWeighting,
    NTKWeighting,
    FixedWeighting,
    build_strategy,
)


def _terms(data=2.0, a=8.0, b=0.5):
    return dict(resid=torch.tensor(data), kstar=torch.tensor(a), turing=torch.tensor(b))


def test_ratio_equalises_physics_terms_to_data_loss_after_update():
    base = dict(resid=0.1, kstar=1.0, turing=1.0)
    strat = RatioWeighting(base, update_every=1)
    terms = _terms(data=2.0, a=8.0, b=0.5)

    total, w = strat.combine(terms, step=0)

    # weight_k = loss_data / loss_k for every term except the data term
    assert w["kstar"] == pytest.approx(2.0 / 8.0)
    assert w["turing"] == pytest.approx(2.0 / 0.5)
    # the data term keeps its fixed base weight, unrescaled
    assert w["resid"] == pytest.approx(0.1)
    # every weighted physics term now equals the current unweighted data loss
    assert w["kstar"] * float(terms["kstar"]) == pytest.approx(2.0)
    assert w["turing"] * float(terms["turing"]) == pytest.approx(2.0)


def test_ratio_weights_held_constant_between_updates():
    base = dict(resid=0.1, kstar=1.0, turing=1.0)
    strat = RatioWeighting(base, update_every=5)

    _, w0 = strat.combine(_terms(data=2.0, a=8.0, b=0.5), step=0)
    # different term magnitudes at step 1..4 must NOT change the weights
    _, w1 = strat.combine(_terms(data=100.0, a=3.0, b=7.0), step=1)
    _, w4 = strat.combine(_terms(data=100.0, a=3.0, b=7.0), step=4)

    assert w0 == w1 == w4

    # at the next update boundary (step 5) the weights DO change
    _, w5 = strat.combine(_terms(data=100.0, a=3.0, b=7.0), step=5)
    assert w5["kstar"] == pytest.approx(100.0 / 3.0)
    assert w5 != w0


def test_ratio_guards_against_zero_term():
    base = dict(resid=0.1, kstar=1.0, turing=1.0)
    strat = RatioWeighting(base, update_every=1, eps=1e-6)
    total, w = strat.combine(_terms(data=2.0, a=0.0, b=0.5), step=0)
    assert torch.isfinite(torch.as_tensor(w["kstar"]))


def test_build_strategy_ratio_reads_update_every_from_loss_cfg():
    class FakeLossCfg:
        strategy = "ratio"
        weights = dict(resid=0.1, kstar=1.0, turing=1.0)
        ratio_update_every = 7

    strat = build_strategy(FakeLossCfg())
    assert isinstance(strat, RatioWeighting)
    assert strat.update_every == 7


def test_gradnorm_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        GradNormWeighting(dict(resid=0.1, kstar=1.0))


def test_ntk_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        NTKWeighting(dict(resid=0.1, kstar=1.0))


def test_fixed_weighting_still_works_as_a_control():
    strat = FixedWeighting(dict(resid=0.1, kstar=1.0, turing=1.0))
    total, w = strat.combine(_terms(data=2.0, a=8.0, b=0.5), step=0)
    assert float(total) == pytest.approx(0.1 * 2.0 + 1.0 * 8.0 + 1.0 * 0.5)
    assert w == dict(resid=0.1, kstar=1.0, turing=1.0)


def test_ratio_registered_in_weighting_registry():
    assert WEIGHTING.get("ratio") is RatioWeighting
