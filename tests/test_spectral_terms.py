"""test_spectral_terms.py — unit tests for losses/spectral.py (Milestone M1, unit U1).

Covers: RAPS parity against `observables.raps` (numpy reference), the 2-D log-power block
parity against `scoring.morphology._spectral_block` (this test file MAY import `scoring`;
`losses/spectral.py` itself must not — that is the firewall boundary being tested
elsewhere, in test_firewall.py), the five term functions' zero-location and gradient flow,
the spec_shape leakage-rule pin (B_train-of-channel-0-only normalisation), the `is_ignited`
truth table, and the fail-loud paths of `build_frame_targets`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rngrn import observables as obs
from rngrn.losses import spectral as S
from rngrn.scoring import morphology as MORPH


# --------------------------------------------------------------------------------------
# 1. RAPS parity vs observables.raps
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,L,seed", [(64, 20.0, 0), (96, 33.7, 1),
                                      # the p=8/512^2 canonical geometry, so raps_torch's
                                      # "identical bin edges" claim is suite-pinned at the
                                      # size it will actually run at (evidence audit OC-1)
                                      (512, 185.01, 2)])
def test_raps_torch_matches_observables_raps(n, L, seed):
    rng = np.random.default_rng(seed)
    field_np = rng.standard_normal((n, n)).astype(np.float64)
    field_t = torch.from_numpy(field_np.copy()).requires_grad_(True)

    k_ref, p_ref, _ = obs.raps(field_np, L=L)
    k_t, p_t = S.raps_torch(field_t, L)

    assert np.array_equal(k_t, k_ref), "bin centers must match the numpy reference exactly"
    np.testing.assert_allclose(p_t.detach().numpy(), p_ref, rtol=1e-12, atol=1e-12)

    (grad,) = torch.autograd.grad(p_t.sum(), field_t)
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------------------
# 2. Block parity vs scoring.morphology._spectral_block
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("nblk", [24, 16])
def test_spectral_block_torch_matches_morphology(nblk):
    rng = np.random.default_rng(7)
    field_np = rng.standard_normal((96, 96)).astype(np.float64)

    ref = MORPH._spectral_block(field_np, nblk)
    got = S.spectral_block_torch(torch.from_numpy(field_np.copy()), nblk)
    got_np = got.detach().numpy().reshape(-1)

    np.testing.assert_allclose(got_np, ref, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------------------
# helpers shared by the term-function tests
# --------------------------------------------------------------------------------------
def _random_frame(seed, n=64, C=1, offset=5.0):
    """A positive-mean random 'frame' — offset keeps log(mean) defined, as a real Turing
    frame's absolute concentrations would."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((C, n, n)) + offset).astype(np.float64), rng


N, L, KSTAR = 64, 20.0, 3.0   # dk = 2*pi/20 = 0.314; B_train = [1.8, 4.65] rad/length


# --------------------------------------------------------------------------------------
# 3. Zero-location + gradient-flow per term
# --------------------------------------------------------------------------------------
def test_terms_are_zero_at_the_target_and_positive_with_gradient_after_perturbation():
    frame, rng = _random_frame(seed=11, n=N)
    cfg = S.SpectralConfig(channels=(0,))
    targets = S.build_frame_targets(frame, L=L, kstar_obs=KSTAR, cfg=cfg)

    u_zero = torch.tensor(frame, dtype=torch.float64, requires_grad=True)
    term_vals_zero, _ = S.spectral_terms(u_zero, targets, cfg)
    for name, val in term_vals_zero.items():
        v = float(val.detach())
        assert abs(v) < 1e-15, f"{name}: {v!r} not ~0 at its own target"

    perturbed = frame + 0.05 * rng.standard_normal(frame.shape)
    u_pert = torch.tensor(perturbed, dtype=torch.float64, requires_grad=True)
    term_vals_pert, _ = S.spectral_terms(u_pert, targets, cfg)
    for name, val in term_vals_pert.items():
        assert float(val.detach()) > 0.0, f"{name}: expected > 0 after perturbation"
        (grad,) = torch.autograd.grad(val, u_pert, retain_graph=True)
        assert torch.isfinite(grad).all(), f"{name}: non-finite gradient"
        assert float(grad.abs().sum()) > 0.0, f"{name}: zero gradient"


# --------------------------------------------------------------------------------------
# 4. Leakage-rule pin: spec_shape's normaliser reads only B_train of channel 0
# --------------------------------------------------------------------------------------
def test_raps_band_target_ignores_content_outside_band_and_other_channels():
    frame, rng = _random_frame(seed=21, n=N, C=2)
    cfg = S.SpectralConfig(channels=(0,))
    targets_a = S.build_frame_targets(frame, L=L, kstar_obs=KSTAR, cfg=cfg)

    frame2 = frame.copy()
    # channel 1 is not in cfg.channels and not read by spec_shape at all.
    frame2[1] += 50.0 * rng.standard_normal((N, N))
    # a pure tone on channel 0 at index 31 (k = 31*dk = 9.74 rad/length), far outside
    # B_train = [1.8, 4.65]; an integer-period sinusoid adds power at exactly one k-bin
    # (no spectral leakage) and does not touch the mean (DC bin).
    xx = np.arange(N)[None, :].repeat(N, axis=0)
    frame2[0] += 20.0 * np.sin(2 * np.pi * 31 * xx / N)
    targets_b = S.build_frame_targets(frame2, L=L, kstar_obs=KSTAR, cfg=cfg)

    np.testing.assert_allclose(targets_b["raps_band_target"].numpy(),
                               targets_a["raps_band_target"].numpy(), rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------------------
# 5. is_ignited truth table
# --------------------------------------------------------------------------------------
def test_is_ignited_truth_table():
    assert S.is_ignited(dict(ss_converged=True, sig_max_pos=0.5), margin=0.1) is True
    assert S.is_ignited(dict(ss_converged=True, sig_max_pos=0.05), margin=0.1) is False
    assert S.is_ignited(dict(ss_converged=False, sig_max_pos=0.5), margin=0.1) is False


def test_is_ignited_raises_keyerror_without_sig_max_pos():
    with pytest.raises(KeyError):
        S.is_ignited(dict(ss_converged=True), margin=0.1)


# --------------------------------------------------------------------------------------
# 6. Fail-loud paths
# --------------------------------------------------------------------------------------
def test_build_frame_targets_raises_on_constant_channel():
    frame = np.full((1, 32, 32), 3.0, dtype=np.float64)
    with pytest.raises(ValueError, match="constant"):
        S.build_frame_targets(frame, L=10.0, kstar_obs=1.0, cfg=S.SpectralConfig())


def test_build_frame_targets_raises_on_empty_band():
    frame, _ = _random_frame(seed=3, n=32)
    # kstar_obs far beyond the grid's achievable wavenumber range -> B_train is empty.
    with pytest.raises(ValueError, match="contains no RAPS bins"):
        S.build_frame_targets(frame, L=10.0, kstar_obs=1000.0, cfg=S.SpectralConfig())
