"""test_rollout.py — the lift-and-simulate rollout: correctness, cost, and invariance.

WHY THESE TESTS EXIST
---------------------
`morphology_match` — whether the recovered model reproduces the target's pattern, the
owner's primary criterion — had never once been computed, because a rollout cost minutes
per field. Making it cheap is only useful if the cheap field is the SAME field, so this
module pins the three properties that make the fast path trustworthy:

  1. the half-spectrum integrator agrees with the full-FFT reference to floating point;
  2. the saturated pattern does not depend on the timestep;
  3. the saturated pattern does not depend on the grid resolution;

and one property that makes the metric trustworthy:

  4. the fast path and the fixed-horizon reference path land in the same morphology CLASS
     on a real dataset sample's geometry.

THE FIXTURE. Every test here needs a model whose uniform steady state is actually
Turing-unstable, and a randomly initialised RNGRN never is: over 800 random inits (400
seeds x 2 forms, N=3) measured on this branch, ZERO satisfied `turing_ok`. TURING_THETA
below is therefore a model found by 400 steps of Adam ascending sigma(k*) at
k* = 6*2pi/60 while penalising sigma(0) and keeping |log D| <= 1.2 (so D stays in
[0.30, 3.34], a diffusion ratio of ~11 rather than the 1e5 an unconstrained search
produces). Its measured properties: sig_max 0.144, linear k* 0.626, D = [0.467, 3.338,
0.299]. It is a SYNTHETIC test fixture, not a recovered model and not a claim about
biological viability — it exists so the invariance properties can be tested on a field
that actually patterns.
"""
from __future__ import annotations

import os
import pathlib
import time

import numpy as np
import pytest
import torch

from rngrn.model import RNGRN
from rngrn.eval.rollout import simulate, _saturated
from rngrn.eval.numerics import INTEGRATORS
from rngrn.scoring import morphology as MORPH


# Provenance in the module docstring. N=3, form='competitive', n_hill=2.
TURING_THETA = {
    "theta_s": [[-1.3052168946306033, -2.189848695301047, -2.7980611988022934],
                [0.6212180514047619, -1.0316834650390194, 1.1835943172044698],
                [-3.223599824541263, -0.13776639922210854, 0.20251799566659132]],
    "theta_g": [[-0.9425409313460333, 2.7829066248057654, 0.3108672315689075],
                [-1.1673567415439314, -0.6438150230230986, 4.420835935011109],
                [-0.8083169076058251, -4.439571035027455, 3.478151634287051]],
    "theta_alpha": [[-0.43101354311120216, 1.8694551679453901, -0.16351412948174512],
                    [-1.2983550902365115, 0.10769743634562622, 4.907630871811246],
                    [-0.6360533075010638, -4.073914857108889, 4.6943961382252475]],
    "theta_delta": [-0.012582809551264658, -0.6466432252029495, -0.09584603785043694],
    "theta_beta": [-2.019330245565732, -3.1170076178180692, -3.5921708996950885],
    "theta_D": [-0.7618163257573206, 1.2055044866315237, -1.2063645766617879],
}
FIXTURE_L = 60.0


def turing_model():
    m = RNGRN(N=3, form="competitive", n_hill=2, seed=0)
    m.load_state_dict({k: torch.tensor(v, dtype=torch.float64)
                       for k, v in TURING_THETA.items()})
    return m


def _fixture_is_turing():
    from rngrn.eval.analysis import turing_ok
    from rngrn.losses.terms import steady_state
    m = turing_model()
    xs, _ = steady_state(m)
    J = m.jacobian(xs, create_graph=False).detach().cpu().numpy()
    return turing_ok(J, m.D.detach().cpu().numpy())


# --------------------------------------------------------------------------------------
# 0. the fixture itself
# --------------------------------------------------------------------------------------
def test_fixture_is_actually_turing_unstable():
    """If this fails every other test below is measuring a decaying field, not a pattern."""
    ok, info = _fixture_is_turing()
    assert ok, f"TURING_THETA no longer satisfies the Turing conditions: {info}"
    assert info["sig_max"] > 0


def test_fixture_actually_patterns():
    r = simulate(turing_model(), L=FIXTURE_L, n=64, integrator="etdrk4_rfft", seed=0)
    assert r["patterned"] is True
    assert not r["blew_up"]
    assert r["amplitude"] > 1e-2


# --------------------------------------------------------------------------------------
# 1. the half-spectrum integrator is the same scheme, not an approximation
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [32, 64])
def test_etdrk4_rfft_agrees_with_full_fft(n):
    """rfft2 stores the non-redundant half of a REAL field's spectrum; irfft2 enforces
    exactly the conjugate symmetry np.real(ifft2(...)) was projecting onto. So the two
    must agree to floating point, not merely 'closely'."""
    from rngrn.eval.rollout import _reaction_np_builder
    from rngrn.losses.terms import steady_state
    m = turing_model()
    D = m.D.detach().cpu().numpy()
    xs, _ = steady_state(m)
    xstar = xs.detach().cpu().numpy()
    rng = np.random.default_rng(0)
    X0 = xstar[:, None, None] + 1e-2 * rng.standard_normal((3, n, n))
    rx = _reaction_np_builder(m)
    dt = 0.2
    a, blew_a = INTEGRATORS["etdrk4"](X0, D, rx, n, FIXTURE_L, dt, 200)
    b, blew_b = INTEGRATORS["etdrk4_rfft"](X0, D, rx, n, FIXTURE_L, dt, 200)
    assert not blew_a and not blew_b
    scale = max(np.abs(a).max(), 1e-12)
    assert np.max(np.abs(a - b)) / scale < 1e-12, (
        f"etdrk4_rfft drifted from etdrk4 by {np.max(np.abs(a - b)) / scale:.2e} relative "
        f"after 200 steps at n={n}; it is supposed to be the same scheme")


def test_coefficient_cache_does_not_leak_across_diffusivities():
    """The cache keys on D's bytes. Two different D must not return the same operator."""
    from rngrn.eval.numerics import _cached_half_coeffs
    c1 = _cached_half_coeffs(np.array([1.0, 2.0, 3.0]), 3, 16, 10.0, 0.1)
    c2 = _cached_half_coeffs(np.array([1.0, 2.0, 4.0]), 3, 16, 10.0, 0.1)
    assert not np.allclose(c1[0], c2[0])
    c1b = _cached_half_coeffs(np.array([1.0, 2.0, 3.0]), 3, 16, 10.0, 0.1)
    assert np.array_equal(c1[0], c1b[0])
    assert not c1[0].flags.writeable, "cached coefficients are shared; they must be read-only"
    with pytest.raises(ValueError, match="diffusivities"):
        _cached_half_coeffs(np.array([1.0, 2.0]), 3, 16, 10.0, 0.1)


# --------------------------------------------------------------------------------------
# 2/3. the saturated pattern is timestep- and resolution-independent
# --------------------------------------------------------------------------------------
# TOLERANCES. These are OBSERVED spreads on the fixture, rounded up — they are not derived
# bounds. Measured (fixture, L=60, seed 0, etdrk4_rfft, fixed 40-growth-time horizon;
# |dv| is the max absolute change over the four raw morphology features):
#   dt 0.4551 -> 0.2276 at n=64 : |dk*|/k* 1.9e-6   |dv| 2.9e-5
#   dt 0.4551 -> 0.1138 at n=64 : |dk*|/k* 1.1e-5   |dv| 5.5e-5
#   n 64 -> 96  at dt 0.4551    : |dk*|/k* 1.1e-2   |dv| 3.3e-2
#   n 64 -> 128 at dt 0.4551    : |dk*|/k* 6.3e-3   |dv| 7.3e-2
#   n 96 -> 128 at dt 0.4551    : |dk*|/k* 1.7e-2   |dv| 9.4e-2
# Timestep independence is essentially exact; RESOLUTION independence is three to four
# orders of magnitude looser, and that asymmetry is the honest result here. The four
# morphology features are computed on the Fourier grid, whose bin spacing changes with n,
# so the residual is largely re-binning — see the "not comparable across grids" guard in
# scoring/morphology.morphology_distance. This test compares the raw FEATURE VECTOR across
# grids ON PURPOSE, which is exactly what that guard forbids doing silently, and reports
# how far apart it actually is. A single tolerance covers both directions.
KSTAR_TOL = 0.02          # relative
FEATURE_TOL = 0.15        # absolute, per feature, in raw (un-z-scored) units


def _saturated_field(n, dt, integrator="etdrk4_rfft", **kw):
    r = simulate(turing_model(), L=FIXTURE_L, n=n, dt=dt, integrator=integrator,
                 seed=0, **kw)
    assert not r["blew_up"], "the fixture rollout blew up; the tolerances below are moot"
    assert r["patterned"], "the fixture did not pattern; there is nothing to compare"
    return r


def test_saturated_pattern_is_timestep_independent():
    """Halving dt must not move the recovered k* or the morphology feature vector."""
    coarse = _saturated_field(n=64, dt=0.4551)
    fine = _saturated_field(n=64, dt=0.4551 / 2)
    rel = abs(fine["kstar"] - coarse["kstar"]) / coarse["kstar"]
    assert rel < KSTAR_TOL, f"k* moved {rel:.3e} when dt was halved"
    v_c = MORPH.morphology_vector(coarse["fields"][0])
    v_f = MORPH.morphology_vector(fine["fields"][0])
    assert np.max(np.abs(v_c - v_f)) < FEATURE_TOL, (
        f"morphology features moved {np.abs(v_c - v_f)} when dt was halved "
        f"(order {MORPH.FEATURE_ORDER})")


def test_saturated_pattern_is_resolution_independent():
    """Same physical box, 64x64 vs 96x96: same k* and the same morphology features.

    This is what licenses scoring the model field on the TARGET's grid rather than on
    solver.n_grid — and, more broadly, it is the L-generalisation property in disguise:
    the pattern is set by the physics and the box, not by the discretisation.
    """
    lo = _saturated_field(n=64, dt=0.4551)
    hi = _saturated_field(n=96, dt=0.4551)
    rel = abs(hi["kstar"] - lo["kstar"]) / lo["kstar"]
    assert rel < KSTAR_TOL, f"k* moved {rel:.3e} between a 64x64 and a 96x96 grid"
    v_lo = MORPH.morphology_vector(lo["fields"][0])
    v_hi = MORPH.morphology_vector(hi["fields"][0])
    assert np.max(np.abs(v_lo - v_hi)) < FEATURE_TOL, (
        f"morphology features moved {np.abs(v_lo - v_hi)} between 64x64 and 96x96 "
        f"(order {MORPH.FEATURE_ORDER})")


# --------------------------------------------------------------------------------------
# 4. the horizon policy — the defect this unit was created to fix
# --------------------------------------------------------------------------------------
def test_a_linearly_stable_model_gets_a_decay_horizon_not_a_40000_unit_one():
    """A random-init RNGRN is linearly STABLE (sig_max < 0 for all 800 inits measured).

    Before unit 7 the horizon rate was max(sigd.max(), 1e-3), so a stable model with
    sig_max = -0.42 was handed T = 40/1e-3 = 40000 time units — 127904 to 200000 steps,
    i.e. 6-10 minutes for ONE 96x96 field. The rate must be |sig_max|.
    """
    m = RNGRN(N=3, seed=0)
    r = simulate(m, L=100.0, n=32, integrator="etdrk4_rfft", seed=0)
    assert r["sig_max"] < 0, "this fixture is supposed to be linearly stable"
    assert r["horizon_rate"] == pytest.approx(abs(r["sig_max"]))
    assert r["nsteps"] < 5000, (
        f"a linearly stable model still costs {r['nsteps']} steps — the 1e-3 floor is back")
    assert r["stopped_reason"] == "horizon"


def test_step_budget_is_reported_not_hidden():
    """A run cut short by max_steps must say so, so it cannot be read as converged."""
    m = turing_model()
    r = simulate(m, L=FIXTURE_L, n=32, integrator="etdrk4_rfft", seed=0, max_steps=250)
    assert r["nsteps_run"] == 250
    assert r["stopped_reason"] == "step_budget"


def test_unknown_integrator_raises_before_any_work():
    with pytest.raises(KeyError, match="unknown integrator"):
        simulate(turing_model(), L=FIXTURE_L, n=16, integrator="nope")


# --------------------------------------------------------------------------------------
# 5. the saturation stopping rule
# --------------------------------------------------------------------------------------
def test_saturation_rule_needs_both_amplitude_and_wavenumber_flat():
    """Either signal alone is a known false positive; the rule requires both."""
    flat = [1.0, 1.0, 1.0, 1.0, 1.0]
    drifting = [1.0, 1.1, 1.2, 1.3, 1.4]
    assert _saturated(flat, flat, 0.01, 5) is True
    assert _saturated(flat, drifting, 0.01, 5) is False      # k* still moving
    assert _saturated(drifting, flat, 0.01, 5) is False      # amplitude still growing
    assert _saturated(flat[:3], flat[:3], 0.01, 5) is False  # not enough checks yet


def test_collapse_rule_only_fires_for_a_decaying_stable_model():
    """The collapse rule anticipates patterned=False; it must never pre-empt a growing one."""
    from rngrn.eval.rollout import _collapsed
    decaying = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
    assert _collapsed(decaying, -0.1, 1e-2, 5) is True
    assert _collapsed(decaying, +0.1, 1e-2, 5) is False, "fired on an UNSTABLE model"
    assert _collapsed(decaying, -0.1, 1e-6, 5) is False, "fired above the level"
    rising = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    assert _collapsed(rising, -0.1, 1e-2, 5) is False, "fired on a rising transient"
    assert _collapsed(decaying[:3], -0.1, 1e-2, 5) is False, "fired before the window filled"


def test_collapse_stop_cuts_a_near_marginal_stable_model_short():
    """The e2e case: a stable model whose decay rate is small enough to eat the budget."""
    m = RNGRN(N=3, seed=0)     # linearly stable, sig_max ~ -0.42
    # horizon_growth_times is inflated to stand in for a near-marginal sig_max: what makes
    # the e2e case expensive is a large T/dt ratio, and this reproduces that ratio cheaply.
    kw = dict(L=100.0, n=32, integrator="etdrk4_rfft", seed=0,
              horizon_growth_times=800.0, max_steps=200000)
    slow = simulate(m, **kw)
    fast = simulate(m, early_stop=True, check_every=100, **kw)
    assert slow["patterned"] is False and fast["patterned"] is False, (
        "both must reach the same verdict; the collapse stop may only reach it sooner")
    assert fast["stopped_reason"] == "collapsed"
    assert fast["nsteps_run"] < slow["nsteps_run"] / 5


def test_an_unpatterned_rollout_is_not_handed_to_morphology_scoring():
    """REGRESSION. A stable model's field decays to float noise; classifying that noise and
    calling it a morphology MATCH is a false positive. train._morphology_rollout must
    withhold the field, and it must gate on `patterned`, not on a float-noise floor."""
    from rngrn.config import Config
    from rngrn.train import _morphology_rollout
    from rngrn.losses.terms import steady_state
    m = RNGRN(N=3, seed=0)                       # linearly stable — will not pattern
    xs, _ = steady_state(m)
    cfg = Config()
    target = np.stack([_pattern_like(32)])
    field, row = _morphology_rollout(cfg, _Res(m, xs.detach().cpu().numpy()),
                                     _RI(target, 100.0))
    assert field is None, "an unpatterned field was passed to morphology scoring"
    assert row["rollout_status"] == "unpatterned"
    assert row["rollout_patterned"] is False


def test_early_stop_rejects_a_degenerate_window():
    with pytest.raises(ValueError, match="saturation_window"):
        simulate(turing_model(), L=FIXTURE_L, n=16, early_stop=True, saturation_window=1)


def test_early_stop_reaches_the_same_field_when_it_does_not_fire():
    """early_stop only ever TRUNCATES; with a tolerance it never meets, chunked driving
    must reproduce the one-call field exactly (the coefficients are cached, not refitted)."""
    a = simulate(turing_model(), L=FIXTURE_L, n=32, integrator="etdrk4_rfft", seed=0)
    b = simulate(turing_model(), L=FIXTURE_L, n=32, integrator="etdrk4_rfft", seed=0,
                 early_stop=True, check_every=100, saturation_tol=0.0)
    assert b["stopped_reason"] == "horizon"
    assert b["nsteps_run"] == a["nsteps_run"]
    assert np.max(np.abs(a["fields"] - b["fields"])) < 1e-12


# --------------------------------------------------------------------------------------
# 6. cost, and the fast/reference agreement on a real sample's geometry
# --------------------------------------------------------------------------------------
_VAL = (pathlib.Path(__file__).resolve().parents[1]
        / "data" / "datasets" / "three_gene_val" / "payload.h5")


def _real_sample():
    h5py = pytest.importorskip("h5py")
    if not _VAL.exists():
        pytest.skip(f"{_VAL} is not present on this machine (see docs/LOCAL_DATA_SETUP.md)")
    with h5py.File(_VAL, "r") as fh:
        key = sorted(fh.keys())[0]
        g = fh[key]
        return key, float(g.attrs["L"]), int(g.attrs["grid"])


def test_fast_and_reference_paths_agree_on_a_real_sample_geometry():
    """Same morphology CLASS from the fast path and the full-FFT fixed-horizon reference.

    The 'real sample' supplies the GEOMETRY — its box length L and its 96x96 grid, which is
    what the model field must be simulated on for the morphology comparison to be defined.
    The model is the synthetic Turing fixture, not a recovery of that sample: this test is
    about the two integration PATHS agreeing, not about recovery quality.
    """
    key, L, grid = _real_sample()
    bank = MORPH.default_reference_bank()
    ref = simulate(turing_model(), L=L, n=grid, integrator="etdrk4", seed=0)
    fast = simulate(turing_model(), L=L, n=grid, integrator="etdrk4_rfft", seed=0,
                    early_stop=True, check_every=200)
    assert not ref["blew_up"] and not fast["blew_up"]
    c_ref = MORPH.classify_morphology(ref["fields"][0], bank)
    c_fast = MORPH.classify_morphology(fast["fields"][0], bank)
    assert c_fast.label == c_ref.label, (
        f"sample {key}: fast path called '{c_fast.label}' where the reference path called "
        f"'{c_ref.label}' (distances {c_fast.distances} vs {c_ref.distances})")


# --------------------------------------------------------------------------------------
# 7. the wire: simulate -> score_recovery -> morphology_scored == "compared"
# --------------------------------------------------------------------------------------
class _Res:
    """Minimal stand-in for a RecoveryResult carrying the Turing fixture."""
    def __init__(self, model, xstar):
        self.model = model
        self.xstar = xstar
        self.kstar_model = 0.6
        self.loss = 0.0


class _RI:
    def __init__(self, frame, L):
        self.frame = frame
        self.L = L
        self.observed_idx = (0,)
        self.N = 3


class _AK:
    J = None
    n_species_true = None
    kstar = 1.0
    kstar_fft = None


def test_morphology_rollout_produces_a_comparable_field_on_the_target_grid():
    """train._morphology_rollout must simulate on the TARGET's grid and L, and its field
    must be what turns morphology_scored into 'compared'."""
    from rngrn.config import Config
    from rngrn.train import _morphology_rollout
    from rngrn.validate import score_recovery
    from rngrn.losses.terms import steady_state

    m = turing_model()
    xs, _ = steady_state(m)
    cfg = Config()
    cfg.solver.morphology_max_steps = 20000
    target = np.stack([_pattern_like(64) for _ in range(1)])   # (m, H, W)
    field, row = _morphology_rollout(cfg, _Res(m, xs.detach().cpu().numpy()),
                                     _RI(target, FIXTURE_L))
    assert row["rollout_status"] == "ok", row
    assert row["rollout_grid"] == 64
    assert row["rollout_L"] == FIXTURE_L
    assert row["rollout_patterned"] is True
    assert field.shape == (3, 64, 64)

    out = score_recovery(_Res(m, xs.detach().cpu().numpy()), _AK(),
                         observed_idx=[0], target_frame=target, model_frame=field)
    assert out["morphology_scored"] == "compared"
    assert isinstance(out["morphology_match"], bool)
    assert np.isfinite(out["morphology_distance"])
    assert np.isfinite(out["spectral_distance_2d"])


def test_morphology_rollout_can_be_switched_off():
    from rngrn.config import Config
    from rngrn.train import _morphology_rollout
    cfg = Config()
    cfg.solver.morphology_rollout = False
    field, row = _morphology_rollout(cfg, None, None)
    assert field is None and row["rollout_status"] == "disabled"


def _pattern_like(n):
    """A deterministic striped target field — only its SHAPE and finiteness matter here."""
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.sin(2 * np.pi * 6 * x / n) + 0.3 * np.cos(2 * np.pi * 6 * y / n)


@pytest.mark.skipif(os.environ.get("RNGRN_SKIP_TIMING") == "1",
                    reason="timing assertion disabled by RNGRN_SKIP_TIMING")
def test_one_field_costs_well_under_thirty_seconds():
    """The unit's stated budget. Loose by design — this is a regression guard on the
    HORIZON POLICY, not a benchmark. Measured on the fixture at 96x96 on this machine:
    ~0.9 s (etdrk4_rfft), ~1.4 s (etdrk4). The bound is 30 s because a slower or loaded
    machine must not turn a policy check into a flake."""
    _, L, grid = _real_sample()
    t = time.perf_counter()
    simulate(turing_model(), L=L, n=grid, integrator="etdrk4_rfft", seed=0)
    assert time.perf_counter() - t < 30.0
