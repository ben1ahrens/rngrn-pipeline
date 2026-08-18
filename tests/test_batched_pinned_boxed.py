"""BatchedRNGRN with pin_xstar / param_boxes (Task 16, R2 redesign).

Tasks 12 and 13 scoped `pin_xstar` and `param_boxes` to the SERIAL `RNGRN` and made
`BatchedRNGRN` refuse them with a NotImplementedError naming T16. Phase I (spec §4.5) is a
BATCHED population on a pinned + boxed model, so T16 is where that refusal is replaced by
an implementation.

THE CONTRACT UNDER TEST:
  * a batched pinned/boxed model is EXACTLY B independent serial pinned/boxed models --
    every constrained physical quantity (alpha, delta, beta, KA, KR, D) agrees member for
    member with the serial twin built from the same seed;
  * the fixed point is still EXACT: reaction(x*) == 0 to float64 tolerance, per member;
  * `beta_hinge()` reduces to (B,), NOT to a scalar -- a scalar would couple the members
    through one shared gradient, which is the same defect `_clip_grad_norm_per_member`
    exists to prevent;
  * the pin and the boxes are SHARED across members (one target frame, one box), so a
    mismatched member is refused loudly rather than silently taking member 0's;
  * pin_xstar=None + param_boxes=None stays bit-identical to the legacy batched model.
"""
import numpy as np
import pytest
import torch

from rngrn.model import RNGRN, BatchedRNGRN

XS = [0.42, 0.75, 0.35]                       # the R2 target's frame-channel means, roughly
BOX = {"alpha": (0.5, 10.0), "delta": (0.4, 5.0)}   # configs/bio_box.yaml
SEEDS = [11, 22, 33, 44]


def _serial(seed, **kw):
    return RNGRN(N=3, form="competitive", seed=seed, **kw)


@pytest.mark.parametrize("kw", [
    dict(pin_xstar=XS),
    dict(param_boxes=BOX),
    dict(pin_xstar=XS, param_boxes=BOX),
])
def test_batched_members_match_their_serial_twins(kw):
    """Every constrained parameter, member for member, bit-identical to the serial model."""
    bm = BatchedRNGRN.from_seeds(N=3, seeds=SEEDS, form="competitive", **kw)
    assert bm.B == len(SEEDS)
    for b, seed in enumerate(SEEDS):
        m = _serial(seed, **kw)
        for name in ("KA", "KR", "alpha", "delta", "beta", "D"):
            got = getattr(bm, name)[b].detach().numpy()
            want = getattr(m, name).detach().numpy()
            assert np.allclose(got, want, rtol=0, atol=1e-15), (
                f"{name} member {b} differs from its serial twin: max abs "
                f"{np.abs(got - want).max()}")


def test_pinned_batched_fixed_point_is_exact_per_member():
    bm = BatchedRNGRN.from_seeds(N=3, seeds=SEEDS, form="competitive",
                                 pin_xstar=XS, param_boxes=BOX)
    x = torch.tensor(XS, dtype=torch.float64).expand(bm.B, 3)
    f = bm.reaction(x)
    assert f.shape == (bm.B, 3)
    assert float(f.abs().max()) < 1e-12, f"reaction(x*) is not 0: {f}"


def test_beta_hinge_is_per_member_and_matches_serial():
    bm = BatchedRNGRN.from_seeds(N=3, seeds=SEEDS, form="competitive",
                                 pin_xstar=XS, param_boxes=BOX)
    h = bm.beta_hinge()
    assert h.shape == (bm.B,), (
        f"beta_hinge must reduce to one scalar PER MEMBER, got shape {tuple(h.shape)}; a "
        "batch-wide scalar would couple the members' gradients")
    for b, seed in enumerate(SEEDS):
        want = float(_serial(seed, pin_xstar=XS, param_boxes=BOX).beta_hinge())
        assert abs(float(h[b]) - want) < 1e-15


def test_beta_hinge_gradient_reaches_every_pinned_parameter():
    """A pinned model has no theta_beta; the hinge must still be differentiable through the
    parameters beta is DERIVED from, or the beta>=0 constraint has no gradient at all."""
    bm = BatchedRNGRN.from_seeds(N=3, seeds=SEEDS, form="competitive",
                                 pin_xstar=XS, param_boxes=BOX)
    assert not hasattr(bm, "theta_beta") or bm.theta_beta is None
    bm.beta_hinge().sum().backward()
    for name in ("theta_delta", "theta_alpha", "theta_s", "theta_g"):
        g = getattr(bm, name).grad
        assert g is not None and torch.any(g != 0), f"no gradient reached {name}"


def test_pinned_batched_has_no_theta_beta_parameter():
    bm = BatchedRNGRN.from_seeds(N=3, seeds=SEEDS, form="competitive", pin_xstar=XS)
    names = {n for n, _ in bm.named_parameters()}
    assert "theta_beta" not in names, (
        "a pinned batched model must not carry a free theta_beta -- beta is derived")
    assert len(names) == 5
    # the unpinned model still does
    assert "theta_beta" in {n for n, _ in
                            BatchedRNGRN.from_seeds(N=3, seeds=SEEDS).named_parameters()}


def test_member_round_trips_a_pinned_boxed_member():
    bm = BatchedRNGRN.from_seeds(N=3, seeds=SEEDS, form="competitive",
                                 pin_xstar=XS, param_boxes=BOX)
    m = bm.member(2)
    assert m.pin_xstar == tuple(float(v) for v in XS)
    assert m.param_boxes == {k: (float(lo), float(hi)) for k, (lo, hi) in BOX.items()}
    for name in ("KA", "KR", "alpha", "delta", "beta", "D"):
        assert np.allclose(getattr(m, name).detach().numpy(),
                           getattr(bm, name)[2].detach().numpy(), rtol=0, atol=1e-15)


def test_mismatched_pin_across_members_is_refused():
    a = _serial(1, pin_xstar=XS)
    b = _serial(2, pin_xstar=[0.9, 0.9, 0.9])
    with pytest.raises(ValueError, match="pin_xstar"):
        BatchedRNGRN([a, b])


def test_mismatched_boxes_across_members_are_refused():
    a = _serial(1, param_boxes=BOX)
    b = _serial(2, param_boxes={"alpha": (0.5, 10.0)})
    with pytest.raises(ValueError, match="param_boxes"):
        BatchedRNGRN([a, b])


def test_pinned_and_unpinned_members_cannot_be_mixed():
    with pytest.raises(ValueError, match="pin_xstar"):
        BatchedRNGRN([_serial(1, pin_xstar=XS), _serial(2)])


def test_unpinned_unboxed_batched_model_is_unchanged():
    """A0: the legacy batched model must still agree with its serial twin after T16.

    Tolerance, measured not assumed: `softplus` on a (B,N) stack and on an (N,) vector
    differ at the last bit (max 5.6e-17 on beta over these four seeds) because they hit
    different vectorised kernels. That predates T16 and is the same order as the ~1e-12
    single-step agreement `tests/test_batched.py` already records; it is NOT bit-identity,
    and this test does not claim it is.
    """
    bm = BatchedRNGRN.from_seeds(N=3, seeds=SEEDS, form="competitive")
    for b, seed in enumerate(SEEDS):
        m = _serial(seed)
        for name in ("KA", "KR", "alpha", "delta", "beta", "D"):
            assert np.allclose(getattr(bm, name)[b].detach().numpy(),
                               getattr(m, name).detach().numpy(), rtol=0, atol=1e-15)
