"""tests/test_box_reparam.py — Task 13: box-sigmoid reparameterization for alpha/delta
(docs/REDESIGN_rngrn.md 3.3).

param_boxes makes alpha/delta map through low + (high-low)*sigmoid(raw) instead of
softplus, so the constrained value is HARD-BOXED in [low, high] rather than merely
positive. A0: param_boxes=None (the default) must stay bit-identical to legacy RNGRN.
"""
import torch
from rngrn.model import RNGRN, BatchedRNGRN

BOX = {"alpha": (0.5, 10.0), "delta": (0.4, 5.0)}


def test_boxed_params_lie_strictly_inside_their_box_across_seeds():
    for seed in range(50):
        m = RNGRN(N=3, form="competitive", seed=seed, param_boxes=BOX)
        alo, ahi = BOX["alpha"]
        dlo, dhi = BOX["delta"]
        assert (m.alpha > alo).all() and (m.alpha < ahi).all()
        assert (m.delta > dlo).all() and (m.delta < dhi).all()


def test_boxed_params_can_saturate_to_the_box_edge_at_extreme_raw():
    # sigmoid saturates in float64 well before the raw value overflows: sigmoid(40.0)
    # rounds to exactly 1.0 in float64, so the boxed value rounds to exactly `high`. The
    # "strictly inside" claim above is therefore honest only over the RANDOM-INIT raw
    # range (small, N(0, 0.5)-scaled) that test_boxed_params_lie_strictly_inside_their_box
    # actually exercises -- not a universal property of the map. Documented here rather
    # than silently assumed.
    m = RNGRN(N=1, form="competitive", seed=0, param_boxes=BOX)
    with torch.no_grad():
        m.theta_alpha.fill_(40.0)
        m.theta_delta.fill_(-40.0)
    alo, ahi = BOX["alpha"]
    dlo, dhi = BOX["delta"]
    assert m.alpha.item() == ahi
    assert m.delta.item() == dlo


def test_param_boxes_none_is_bit_identical_to_legacy():
    a = RNGRN(N=3, form="competitive", seed=3)
    b = RNGRN(N=3, form="competitive", seed=3, param_boxes=None)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb)
    x = torch.rand(3, dtype=torch.float64)
    assert torch.equal(a.reaction(x), b.reaction(x))


def test_boxed_alpha_and_delta_are_differentiable():
    m = RNGRN(N=3, form="competitive", seed=5, param_boxes=BOX)
    m.alpha.sum().backward()
    assert m.theta_alpha.grad is not None and torch.any(m.theta_alpha.grad != 0)

    m2 = RNGRN(N=3, form="competitive", seed=5, param_boxes=BOX)
    m2.delta.sum().backward()
    assert m2.theta_delta.grad is not None and torch.any(m2.theta_delta.grad != 0)


def test_unknown_box_key_is_rejected_loudly():
    try:
        RNGRN(N=3, form="competitive", seed=0, param_boxes={"D": (1.0, 60.0)})
    except ValueError:
        pass
    else:
        raise AssertionError("param_boxes with a key other than alpha/delta should raise ValueError")


def test_param_boxes_not_implemented_for_low_basal_init():
    try:
        RNGRN(N=3, form="competitive", seed=0, init="low_basal", param_boxes=BOX)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("param_boxes + init='low_basal' should raise NotImplementedError")


def test_pin_xstar_and_param_boxes_combine_fixed_point_and_gradients():
    xs = [0.7, 1.3, 1.0]
    m = RNGRN(N=3, form="competitive", seed=3, pin_xstar=xs, param_boxes=BOX)
    x = torch.tensor(xs, dtype=torch.float64)
    assert torch.allclose(m.reaction(x), torch.zeros(3), atol=1e-12)

    alo, ahi = BOX["alpha"]
    dlo, dhi = BOX["delta"]
    assert (m.alpha > alo).all() and (m.alpha < ahi).all()
    assert (m.delta > dlo).all() and (m.delta < dhi).all()

    m.beta_hinge().backward()
    assert m.theta_delta.grad is not None and torch.any(m.theta_delta.grad != 0)
    assert m.theta_alpha.grad is not None and torch.any(m.theta_alpha.grad != 0)


def test_batched_rngrn_refuses_boxed_members():
    m = RNGRN(N=3, form="competitive", seed=3, param_boxes=BOX)
    try:
        BatchedRNGRN([m])
    except NotImplementedError:
        pass
    else:
        raise AssertionError("BatchedRNGRN should refuse boxed members (T16 will add support)")
