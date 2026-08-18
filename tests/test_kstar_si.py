"""test_kstar_si.py — the scale-invariant k* anchor (Task 14, R2 redesign).

`kstar_anchor_si` replaces `kstar_anchor`'s raw sigma with sigma normalised by its own
range (docs/REDESIGN_rngrn.md §4.4): sigbar = sig / (max sig - min sig + eps), loss =
lse(temp*sigbar)/temp - sigbar(k*_obs). Two properties are the contract (brief's own
words): invariance under sigma -> c*sigma, and closing the "flatten sigma to shrink the
raw gap" degenerate direction that the legacy anchor is exposed to.

Both terms take `model` (anything exposing `.dispersion(xstar, kgrid, J=None)`), not a
raw sigma tensor -- kstar_anchor's ACTUAL signature (read from source) wins over the
brief's `kstar_anchor_si(sig, kgrid, kstar_obs, ...)` transcription. `_StubDispersion`
below pins sigma directly, exactly as tests/test_losses.py's stub does for the hinges.
"""
import torch

from rngrn.losses.terms import kstar_anchor, kstar_anchor_si, kstar_anchor_si_batched

K = torch.linspace(0.05, 4.0, 400, dtype=torch.float64)
SIG = -((K - 1.5) ** 2) + 0.3


class _StubDispersion:
    """Minimal model surface: dispersion() returns a pinned sigma(k), ignoring xstar."""

    def __init__(self, sig):
        self._sig = sig

    def dispersion(self, xstar, kgrid, J=None):
        return self._sig


def test_kstar_si_is_invariant_under_sigma_rescaling():
    a, _ = kstar_anchor_si(_StubDispersion(SIG), None, K, kstar_obs=1.0)
    b, _ = kstar_anchor_si(_StubDispersion(SIG * 17.0), None, K, kstar_obs=1.0)
    assert torch.allclose(a, b, rtol=1e-9)


def test_flattening_does_not_reduce_kstar_si_but_reduces_the_legacy_anchor():
    legacy_full, _ = kstar_anchor(_StubDispersion(SIG), None, K, kstar_obs=1.0)
    legacy_flat, _ = kstar_anchor(_StubDispersion(SIG * 0.1), None, K, kstar_obs=1.0)
    si_full, _ = kstar_anchor_si(_StubDispersion(SIG), None, K, kstar_obs=1.0)
    si_flat, _ = kstar_anchor_si(_StubDispersion(SIG * 0.1), None, K, kstar_obs=1.0)
    legacy_gap = legacy_full - legacy_flat
    si_gap = si_full - si_flat
    assert legacy_gap > 0          # the measured degenerate direction exists
    assert abs(si_gap) < 1e-9      # and kstar_si closes it


def test_kstar_si_batched_matches_serial_to_1e12_on_a_b3_stack():
    scales = [1.0, 17.0, 0.1]
    sig_b = torch.stack([SIG * s for s in scales])          # (3, K)
    L_b, parts_b = kstar_anchor_si_batched(_StubDispersion(sig_b), None, K, kstar_obs=1.0)
    assert L_b.shape == (3,)
    for i, s in enumerate(scales):
        L_s, _ = kstar_anchor_si(_StubDispersion(SIG * s), None, K, kstar_obs=1.0)
        assert abs(float(L_b[i]) - float(L_s)) < 1e-12
    assert parts_b["kstar_model"].shape == (3,)
