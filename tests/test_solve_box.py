"""test_solve_box.py — the adaptive commensurate solve box (REDESIGN_rngrn.md §4.3).

Four contracts, in the order `docs/PLAN_redesign_R3.md` Task 11 states them:

1. **Commensurability.** `L_solve` holds exactly `p` periods of the model's current
   dispersion argmax k̂ — the property the whole construction exists for.
2. **Hysteresis.** A re-tile fires when k̂ drifts more than HALF a solve-box RAPS bin
   (`|Δk̂|/k̂ > 1/(2p)`) and not below it. The bin is the solve box's own, so the
   threshold is a pure function of `p`.
3. **k̂ is DETACHED.** Solve-box geometry is not differentiated: the geometry carries no
   grad, and a `backward()` through a loss that genuinely depends on `L_solve` populates
   no gradient path back to k̂.
4. **Re-tile clears warm starts and rebuilds the k-grid**, on both the serial
   `PatternSolver` and the `BatchedPatternSolver` (per-member warm store).

Plus a DRIFT TEST binding this module's solve-box bin geometry to
`losses/spectral.py::_raps_bins`, the binning the model spectrum is actually computed
with. The two are computed independently (§4.3's geometry here, F-D6-1's integer-arange
edges there); if they ever disagree the interpolated targets land on the wrong bins, and
nothing else in the suite would notice.
"""
import math

import numpy as np
import pytest
import torch

import rngrn.solve_box as sb
from rngrn.forward import BatchedPatternSolver, PatternSolver
from rngrn.losses.spectral import _raps_bins
from rngrn.model import RNGRN, BatchedRNGRN

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------------------
# 1. commensurability
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("kstar_hat", [0.5, 1.0, 2.5, 7.75, 31.0])
def test_L_solve_holds_exactly_p_periods_of_khat(kstar_hat):
    """The defining property: the box is commensurate with the model's own pattern."""
    box = sb.geometry(kstar_hat)
    periods = box["L_solve"] * kstar_hat / (2.0 * math.pi)
    assert periods == pytest.approx(box["p"], rel=1e-14)


def test_geometry_defaults_are_the_spec_fixed_values():
    """p=8 and n=96 are fixed by REDESIGN_rngrn.md §4.3; a silent change of either would
    change what every solve-box number means."""
    box = sb.geometry(1.0)
    assert box["p"] == 8
    assert box["n"] == 96
    assert set(box) == {"L_solve", "n", "p"}


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_geometry_refuses_a_nonpositive_or_nonfinite_khat(bad):
    with pytest.raises(ValueError, match="kstar_hat"):
        sb.geometry(bad)


# --------------------------------------------------------------------------------------
# 2. hysteresis
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("p", [4, 8, 12])
def test_retile_fires_just_above_half_a_solve_box_bin(p):
    at_tile = 3.0
    now = at_tile * (1.0 + (1.0 / (2 * p)) * (1.0 + 1e-9))
    assert sb.needs_retile(now, at_tile, p) is True


@pytest.mark.parametrize("p", [4, 8, 12])
def test_retile_does_not_fire_just_below_half_a_solve_box_bin(p):
    at_tile = 3.0
    now = at_tile * (1.0 + (1.0 / (2 * p)) * (1.0 - 1e-9))
    assert sb.needs_retile(now, at_tile, p) is False


@pytest.mark.parametrize("p", [4, 8, 12])
def test_hysteresis_is_symmetric_in_the_drift_direction(p):
    """A drift DOWN of the same relative size must fire too — |Δk̂|, not Δk̂."""
    at_tile = 3.0
    down = at_tile * (1.0 - (1.0 / (2 * p)) * (1.0 + 1e-9))
    assert sb.needs_retile(down, at_tile, p) is True


def test_no_drift_never_retiles():
    assert sb.needs_retile(2.0, 2.0, 8) is False


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_needs_retile_refuses_a_nonpositive_or_nonfinite_reference(bad):
    with pytest.raises(ValueError):
        sb.needs_retile(2.0, bad, 8)


# --------------------------------------------------------------------------------------
# 3. k̂ is detached
# --------------------------------------------------------------------------------------
def test_geometry_carries_no_gradient():
    """`L_solve` is a plain float — a python float cannot carry a grad_fn at all, which is
    the strongest form of the §4.3 requirement."""
    khat = torch.tensor(2.5, requires_grad=True)
    box = sb.geometry(khat)
    assert isinstance(box["L_solve"], float)
    assert torch.as_tensor(box["L_solve"]).requires_grad is False


def test_backward_through_a_solve_box_loss_reaches_no_grad_path_via_L():
    """A loss computed ON the box, differentiated w.r.t. a real parameter, must leave k̂
    with no gradient.

    The probe is a Gaussian bump of FIXED physical width on the box, scored by a
    bin-weighted RAPS sum — a quantity that genuinely depends on `L_solve` (the box's
    extent sets where the bump sits in k). Verified as a tripwire, not assumed: building
    the identical loss with a NON-detached `L = p*2*pi/khat` gives `khat.grad = 0.6547`,
    against `theta.grad = 66.44` either way. So a leak here is loud, not marginal.
    """
    from rngrn.losses.spectral import raps_torch

    khat = torch.tensor(2.5, requires_grad=True)
    theta = torch.tensor(0.7, requires_grad=True)
    box = sb.geometry(khat, n=16)
    n, L = box["n"], box["L_solve"]
    x = torch.arange(n, dtype=torch.float64) * (L / n)
    g = torch.exp(-((x - 0.5 * L) / 1.0) ** 2)
    field = theta * g.unsqueeze(0) * g.unsqueeze(1)
    _, power = raps_torch(field, L)
    loss = (power * torch.arange(power.shape[0], dtype=torch.float64)).sum()
    loss.backward()

    assert theta.grad is not None and abs(float(theta.grad)) > 1.0, (
        "the probe loss does not actually depend on the box — the test cannot detect a leak")
    assert khat.grad is None, (
        "k̂ received a gradient through the solve-box geometry; §4.3 requires it DETACHED")


# --------------------------------------------------------------------------------------
# 4. re-tile clears warm starts and rebuilds the k-grid
# --------------------------------------------------------------------------------------
def _model():
    return RNGRN(N=3, seed=0)


def test_serial_retile_clears_the_warm_start_and_rebuilds_the_kgrid():
    solver = PatternSolver(_model(), n=16, L=1.0, seed=0)
    k2_before = solver.k2_full.copy()
    k2h_before = solver._k2h.clone()
    solver._warm = torch.zeros(3, 16, 16)

    solver.retile(L=2.0)

    assert solver._warm is None, "re-tile must clear the warm start: it lives on the OLD grid"
    assert solver.L == 2.0
    assert not np.allclose(solver.k2_full, k2_before), "k-grid was not rebuilt"
    assert not torch.allclose(solver._k2h, k2h_before), "half-spectrum k-grid was not rebuilt"


def test_serial_retile_can_change_n_as_well_as_L():
    solver = PatternSolver(_model(), n=16, L=1.0, seed=0)
    solver.retile(L=2.0, n=32)
    assert solver.n == 32
    assert solver.k2_full.shape == (32, 32)
    assert solver._k2h.shape == (32, 32 // 2 + 1)


def test_batched_retile_clears_every_members_warm_start_and_rebuilds_the_kgrid():
    model = BatchedRNGRN([_model(), _model()])
    solver = BatchedPatternSolver(model, n=16, L=1.0, seeds=[0, 1])
    k2h_before = solver._k2h.clone()
    solver._warm = [torch.zeros(3, 16, 16), torch.zeros(3, 16, 16)]

    solver.retile(L=2.0)

    assert solver._warm == [None, None], "re-tile must clear EVERY member's warm start"
    assert solver.L == 2.0
    assert not torch.allclose(solver._k2h, k2h_before), "half-spectrum k-grid was not rebuilt"


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_retile_refuses_a_nonpositive_or_nonfinite_L(bad):
    solver = PatternSolver(_model(), n=16, L=1.0, seed=0)
    with pytest.raises(ValueError):
        solver.retile(L=bad)


# --------------------------------------------------------------------------------------
# target interpolation onto the solve box's bins, in k/k*_obs units
# --------------------------------------------------------------------------------------
def test_bin_centers_are_j_plus_half_over_p_in_khat_units():
    """The commensurate box's payoff: bin j sits at (j+0.5)/p in k/k̂ units, independent
    of k̂ itself. That is what makes the target scale-free."""
    box = sb.geometry(3.7)
    x = sb.bin_centers_scalefree(box)
    j = np.arange(x.shape[0])
    assert np.allclose(x, (j + 0.5) / box["p"], rtol=0, atol=1e-15)


@pytest.mark.parametrize("kstar_hat", [1.0, 3.7, 12.0])
def test_bin_centers_match_the_spectral_raps_binning(kstar_hat):
    """DRIFT TEST. `losses/spectral._raps_bins` is what the MODEL spectrum is binned by;
    if this module's bin geometry disagrees, targets land on the wrong bins."""
    box = sb.geometry(kstar_hat)
    _, _, k_centers = _raps_bins(box["n"], box["L_solve"],
                                 torch.device("cpu"), torch.float64)
    x = sb.bin_centers_scalefree(box)
    assert x.shape == k_centers.shape
    assert np.allclose(x * kstar_hat, k_centers, rtol=1e-13, atol=0)


def test_band_bins_are_the_bins_inside_B_train():
    box = sb.geometry(2.0)
    idx = sb.band_bins(box, 0.60, 1.55)
    x = sb.bin_centers_scalefree(box)
    assert idx.tolist() == np.nonzero((x >= 0.60) & (x <= 1.55))[0].tolist()
    assert len(idx) > 0


def test_interpolate_targets_reproduces_a_linear_log_raps_exactly():
    """Linear interpolation is exact on a linear function, so the only thing this can catch
    is the k/k*_obs unit mapping being wrong — which is the point."""
    box = sb.geometry(2.0)
    kstar_obs = 5.0
    k_obs = np.linspace(0.05, 60.0, 400)
    log_raps_obs = 3.0 - 1.25 * (k_obs / kstar_obs)          # linear in k/k*_obs

    got = sb.interpolate_targets(log_raps_obs, k_obs, kstar_obs, box)

    x = sb.bin_centers_scalefree(box)[sb.band_bins(box)]
    want = torch.from_numpy(3.0 - 1.25 * x)
    assert torch.allclose(got, want, rtol=1e-12, atol=1e-12)


def test_interpolate_targets_is_independent_of_khat():
    """A re-tile changes L_solve but not (p, n), so the scale-free target is UNCHANGED.
    Re-interpolating on re-tile is therefore idempotent, not a correction."""
    kstar_obs = 5.0
    k_obs = np.linspace(0.05, 60.0, 400)
    log_raps_obs = np.log(1.0 + np.exp(-((k_obs - kstar_obs) ** 2)))

    a = sb.interpolate_targets(log_raps_obs, k_obs, kstar_obs, sb.geometry(2.0))
    b = sb.interpolate_targets(log_raps_obs, k_obs, kstar_obs, sb.geometry(2.0 * 1.3))
    assert torch.equal(a, b)


def test_interpolate_targets_refuses_a_nonfinite_observed_target():
    """The DC RAPS bin is forced to zero, so log of it is -inf. Fail loud rather than
    interpolating a -inf into the band."""
    box = sb.geometry(2.0)
    k_obs = np.linspace(0.05, 60.0, 400)
    log_raps_obs = np.zeros_like(k_obs)
    log_raps_obs[0] = -np.inf
    with pytest.raises(ValueError, match="finite"):
        sb.interpolate_targets(log_raps_obs, k_obs, 5.0, box)


def test_interpolate_targets_refuses_to_extrapolate_below_the_observed_support():
    """A band bin the observed RAPS does not reach cannot be filled by interpolation.
    Silent extrapolation would invent target power the data never measured."""
    box = sb.geometry(2.0)                   # band x spans [0.6875, 1.5625] at p=8
    kstar_obs = 5.0
    k_obs = np.linspace(5.0, 60.0, 400)      # x_min = 1.0 > 0.6875
    log_raps_obs = np.zeros_like(k_obs)
    with pytest.raises(ValueError, match="outside the observed"):
        sb.interpolate_targets(log_raps_obs, k_obs, kstar_obs, box)


def test_interpolate_targets_refuses_a_non_increasing_observed_grid():
    box = sb.geometry(2.0)
    k_obs = np.linspace(60.0, 0.05, 400)     # descending
    with pytest.raises(ValueError, match="increasing"):
        sb.interpolate_targets(np.zeros_like(k_obs), k_obs, 5.0, box)
