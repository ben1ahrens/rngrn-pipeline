"""tests/test_pinned_model.py — Task 12: fixed-point pinning (docs/REDESIGN_rngrn.md §3.2).

pin_xstar makes beta a DERIVED quantity (beta_i = delta_i*xstar_i - prod_i(xstar)) so the
reaction has an EXACT fixed point at the given x*, with no free theta_beta parameter. A0:
pin_xstar=None (the default) must stay bit-identical to legacy RNGRN.
"""
import torch
from rngrn.model import RNGRN, BatchedRNGRN


def test_pinned_model_has_exact_fixed_point_and_no_free_beta():
    xs = [0.7, 1.3, 1.0]
    m = RNGRN(N=3, form="competitive", seed=3, pin_xstar=xs)
    x = torch.tensor(xs, dtype=torch.float64)
    assert torch.allclose(m.reaction(x), torch.zeros(3), atol=1e-12)
    assert not any(n == "theta_beta" for n, _ in m.named_parameters())


def test_unpinned_model_is_bit_identical_to_legacy():
    a = RNGRN(N=3, form="competitive", seed=3)
    b = RNGRN(N=3, form="competitive", seed=3, pin_xstar=None)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb)


def test_pinned_model_exact_fixed_point_nc1_form():
    xs = [0.4, 0.9]
    m = RNGRN(N=2, form="nc1", seed=7, pin_xstar=xs)
    x = torch.tensor(xs, dtype=torch.float64)
    assert torch.allclose(m.reaction(x), torch.zeros(2), atol=1e-12)


def test_beta_hinge_is_positive_and_differentiable_when_beta_forced_negative():
    # beta_i = delta_i*xstar_i - prod_i(xstar). Species 0's own x* is tiny (so
    # delta_0*xstar_0 is tiny) while species 1 and 2 are large, saturating the occupancy
    # terms that feed prod_0 (competitive form sums over ALL j, not just i) -- verified
    # numerically (seed=3) to push beta_0 negative.
    xs = [0.01, 20.0, 20.0]
    m = RNGRN(N=3, form="competitive", seed=3, pin_xstar=xs)
    beta = m.beta
    assert (beta < 0).any(), "test setup should force at least one beta component negative"

    hinge = m.beta_hinge()
    assert hinge.item() > 0.0

    hinge.backward()
    assert m.theta_delta.grad is not None and torch.any(m.theta_delta.grad != 0)
    assert m.theta_alpha.grad is not None and torch.any(m.theta_alpha.grad != 0)
    assert m.theta_s.grad is not None and torch.any(m.theta_s.grad != 0)


def test_batched_rngrn_supports_pinned_members():
    """SUPERSEDED BY TASK 16. This test used to assert BatchedRNGRN raised
    NotImplementedError for a pinned member ("T16 will add support"); T16 added it, because
    Phase I (docs/REDESIGN_rngrn.md §4.5) is a batched population on a pinned model. The
    full contract is tested in tests/test_batched_pinned_boxed.py; this keeps the
    serial-vs-batched agreement check at the site the refusal used to live."""
    xs = [0.7, 1.3, 1.0]
    m = RNGRN(N=3, form="competitive", seed=3, pin_xstar=xs)
    bm = BatchedRNGRN([m])
    assert bm.pin_xstar == tuple(xs)
    assert torch.allclose(bm.beta[0], m.beta, rtol=0, atol=1e-15)
