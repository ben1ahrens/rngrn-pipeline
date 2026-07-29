"""test_lgen_eval.py — the cross-domain-size instrument, with a POSITIVE and a NEGATIVE control.

WHY BOTH CONTROLS
-----------------
`eval/lgen_eval.py` claims to distinguish two outcomes: a recovered GRN whose physical k* is
invariant in the domain size (it generalises) from one whose pattern simply refits whatever
periodicity the box wants (it memorised a box). A metric that cannot fail proves nothing, so
this module pins both ends:

  * POSITIVE — the Turing fixture of test_rollout.py, simulated at L/L_train =
    (0.5, 1, 2, 4). Measured: kstar_phys_cv 0.021, periods_slope_rel_err 0.017, and q
    proportional to L to within 3%.
  * NEGATIVE — a synthetic field carrying a FIXED number of periods per box at every L
    (k* ~ 1/L). Constructed at the frame level, not from a simulation — no recovered model
    in this project is known to behave this way, and the point is to check the ARITHMETIC
    notices, so the synthetic frames are pushed through the same `_row_for` +
    `summarise_across_L` path a real rollout takes. Measured: kstar_phys_cv 0.71,
    periods_slope_rel_err 0.62, i.e. 30x/37x the positive control.

Plus the nondim round trip (the silent-wrong-diffusivity trap), and the two fail-loud
paths: below the pixels-per-wavelength floor, and a checkpoint missing the nondim/L extras.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rngrn import io as IO
from rngrn.eval.lgen_eval import (DEFAULT_N_GRID, PPW_FLOOR, evaluate_across_L, grid_for_L,
                                  physical_model_from_checkpoint, summarise_across_L,
                                  _row_for)
from rngrn.model import RNGRN
from rngrn.scoring import morphology as MORPH

# The Turing fixture, reused rather than copied: a randomly initialised RNGRN is never
# Turing-unstable (0 of 800 measured inits), so every positive control needs this model.
from test_rollout import FIXTURE_L, turing_model

# n_grid at L_train for the fixture. q(L_train) = 6.34, so this puts the fixture at
# 48/6.34 = 7.6 pixels per wavelength — above PPW_FLOOR = 6 at every L, since the grid rule
# holds pixels-per-wavelength fixed.
N_GRID = 48


# --------------------------------------------------------------------------------------
# 1. the grid rule
# --------------------------------------------------------------------------------------
def test_grid_rule_holds_dx_fixed_and_returns_even_grids():
    L_train, n_grid = 60.0, 96
    dx0 = L_train / n_grid
    for factor in (0.5, 1.0, 2.0, 3.0, 4.0):
        n = grid_for_L(factor * L_train, L_train, n_grid)
        assert n % 2 == 0, f"grid {n} is odd; the rfft path and the FFT bins want even"
        dx = (factor * L_train) / n
        assert abs(dx - dx0) / dx0 < 1.0 / n, (
            f"dx moved from {dx0} to {dx} at factor {factor} — the rule is supposed to hold "
            f"it fixed to within the even-rounding, i.e. one part in n")


def test_grid_rule_raises_rather_than_clamping():
    """A clamp would silently change dx, which is the one thing the rule holds fixed."""
    with pytest.raises(ValueError, match="above n_max"):
        grid_for_L(1000.0, 60.0, 96, n_max=512)
    with pytest.raises(ValueError, match="below n_min"):
        grid_for_L(1.0, 60.0, 96)
    with pytest.raises(ValueError, match="finite and > 0"):
        grid_for_L(-5.0, 60.0, 96)


def test_default_grid_is_the_registered_payload_grid():
    assert DEFAULT_N_GRID == 96, (
        "the default grid at L_train is documented as the grid every registered three_gene "
        "payload was generated on; changing it changes every recorded pixels_per_wavelength")


# --------------------------------------------------------------------------------------
# 2. POSITIVE CONTROL — a model that genuinely patterns keeps its PHYSICAL k*
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fixture_across_L():
    """The fixture evaluated at L/L_train = 0.5, 1, 2, 4. ~7 s; shared by three tests."""
    return evaluate_across_L(turing_model(), FIXTURE_L,
                             [0.5 * FIXTURE_L, FIXTURE_L, 2 * FIXTURE_L, 4 * FIXTURE_L],
                             n_grid=N_GRID, seed=0)


def test_every_L_actually_patterned(fixture_across_L):
    """If a row did not pattern, the two tests below are measuring noise, not invariance."""
    rows = fixture_across_L["per_L"]
    assert len(rows) == 4
    assert all(r["patterned"] for r in rows), [(r["L"], r["amplitude"]) for r in rows]
    assert all(r["pixels_per_wavelength"] >= PPW_FLOOR for r in rows), (
        [(r["L"], r["pixels_per_wavelength"]) for r in rows])


def test_physical_kstar_is_invariant_across_domain_size(fixture_across_L):
    """THE HEADLINE. k* is a property of (J, D); the box does not enter it.

    Tolerance: 0.10 on the CV. The MEASURED value is 0.021, and the floor it has to clear is
    the grid-to-grid spread of the k* estimator itself — +-3% on this fixture (see the scan
    in lgen_eval's module docstring) — plus the RAPS bin width, which at the smallest box
    here (q = 3.2) is 31% of k* on its own. 0.10 is comfortably above the measurement floor
    and 7x below the 0.71 the box-pinned negative control produces.
    """
    s = fixture_across_L["summary"]
    assert s["n_patterned"] == 4
    assert s["kstar_phys_cv"] < 0.10, (
        f"k* varied by CV {s['kstar_phys_cv']:.4f} across the four domain sizes; the "
        f"recovered network is supposed to select a wavelength, not a box. Rows: "
        f"{[(r['L'], r['kstar_phys']) for r in fixture_across_L['per_L']]}")


def test_periods_per_box_scale_linearly_with_L(fixture_across_L):
    """q(L) = L*k*/(2pi): double the box, double the number of stripes."""
    s = fixture_across_L["summary"]
    assert s["periods_slope_rel_err"] < 0.10, (
        f"the fitted q-vs-L slope {s['periods_slope_fit']:.5g} disagrees with the predicted "
        f"mean(k*)/2pi = {s['periods_slope_predicted']:.5g} by "
        f"{s['periods_slope_rel_err']:.3f}")
    rows = fixture_across_L["per_L"]
    # Anchored on the L_train row, NOT on the smallest box: the RAPS bin width is 1/q
    # relative, so the smallest box (q = 3.2, bin 31% wide) is the worst possible anchor —
    # anchoring there fails at 10.2% on measurement quantisation alone.
    ref = next(r for r in rows if r["is_L_train"])
    for r in rows:
        expected = ref["periods_per_box"] * (r["L"] / ref["L"])
        assert abs(r["periods_per_box"] - expected) / expected < 0.10, (
            f"L={r['L']:g} holds {r['periods_per_box']:.3f} periods; scaling from "
            f"L_train={ref['L']:g} predicts {expected:.3f}")


def test_summary_reports_no_pass_fail_boolean(fixture_across_L):
    """Thresholds are pre-registered and applied by the caller; none may live here."""
    vc = fixture_across_L["summary"]["verdict_components"]
    for key, value in vc.items():
        assert not isinstance(value, bool), f"verdict_components['{key}'] is a boolean verdict"
        assert isinstance(value, (int, float)), (key, type(value))
    joined = " ".join(fixture_across_L["summary"]).lower()
    for banned in ("pass", "fail", "verdict_ok", "generalises"):
        assert banned not in joined, f"summary key looks like a verdict: {banned}"


# --------------------------------------------------------------------------------------
# 3. NEGATIVE CONTROL — a pattern pinned to the BOX must be flagged
# --------------------------------------------------------------------------------------
def _box_pinned_frame(n: int, q: int) -> np.ndarray:
    """A field with EXACTLY q periods across the box whatever the box is.

    Two axis-aligned cosines at pixel-frequency q, so the RAPS ring sits at |k| = 2pi*q/L
    for any L: the periodicity is a property of the frame's index space, not of any physics.
    A little smooth noise is added so the morphology statistics are not degenerate.
    """
    x = np.arange(n) / n
    X, Y = np.meshgrid(x, x)
    rng = np.random.default_rng(0)
    return (np.cos(2 * np.pi * q * X) + np.cos(2 * np.pi * q * Y)
            + 0.05 * rng.standard_normal((n, n)))


def _synthetic_row(L, L_train, n, q, patterned=True):
    """Push a synthetic frame through the SAME row maths a real rollout row takes."""
    from rngrn import observables as obs
    field = _box_pinned_frame(n, q)
    res = dict(fields=field[None], kstar=obs.kstar_of(field, L=L), patterned=patterned,
               amplitude=float(field.std()), blew_up=False, sig_max=0.1,
               nsteps_run=0, stopped_reason="synthetic", seconds=0.0)
    return _row_for(res, L=L, L_train=L_train, n=n,
                    bank=MORPH.default_reference_bank())


def test_a_box_pinned_pattern_is_caught_by_both_headline_statistics():
    """THE NEGATIVE CONTROL. q fixed at 6 for every L means k* moves as 1/L — a network
    that memorised a box. Both headline numbers must be large, not merely nonzero.

    Constructed at the FRAME level (see the module docstring): the frames are synthetic and
    no simulation is involved, but they go through `_row_for` and `summarise_across_L`
    unchanged, so it is the shipped arithmetic being tested.
    """
    L_train, q = 60.0, 6
    rows = [_synthetic_row(f * L_train, L_train, grid_for_L(f * L_train, L_train, N_GRID), q)
            for f in (1.0, 0.5, 2.0, 4.0)]
    # The construction actually did pin the periodicity to the box: q is the SAME at every
    # L. (Its measured value is 6.5, not 6: the RAPS peak-CENTROID of a single pure mode
    # sitting on a bin edge reports that bin's centre, a fixed +0.5-bin offset that is
    # identical at every L and therefore harmless here.)
    for r in rows:
        assert abs(r["periods_per_box"] - rows[0]["periods_per_box"]) \
            / rows[0]["periods_per_box"] < 0.02, [rr["periods_per_box"] for rr in rows]

    s = summarise_across_L(rows)
    assert s["n_patterned"] == 4
    assert s["kstar_phys_cv"] > 0.5, (
        f"k* moved as 1/L and the CV only reached {s['kstar_phys_cv']:.4f}; the metric "
        f"cannot tell a memorised box from an invariant wavelength")
    assert s["periods_slope_rel_err"] > 0.3, (
        f"q was constant in L and the slope error only reached "
        f"{s['periods_slope_rel_err']:.4f}")
    # and it must be far from the positive control, not merely on the other side of a line
    assert s["kstar_phys_cv"] > 10 * 0.021476, "no separation from the measured positive control"
    # WORTH READING: the morphology class is preserved perfectly (1.0) by this box-pinned
    # table, because the synthetic texture really is the same at every L. Morphology
    # agreement alone would have called this a success — which is why the two k*-based
    # statistics are the headline and the class fraction is not.
    assert s["morphology_class_preserved"] == 1.0


def test_summary_needs_two_patterned_rows_and_says_so_with_nan():
    """A CV over one point is not a spread. The NaN is deliberate; n_patterned explains it."""
    L_train = 60.0
    rows = [_synthetic_row(L_train, L_train, N_GRID, 6),
            _synthetic_row(2 * L_train, L_train, 2 * N_GRID, 6, patterned=False)]
    s = summarise_across_L(rows)
    assert s["n_patterned"] == 1 and s["n_L"] == 2
    assert np.isnan(s["kstar_phys_cv"]) and np.isnan(s["periods_slope_rel_err"])


def test_summary_refuses_a_table_without_exactly_one_reference_row():
    rows = [_synthetic_row(2 * 60.0, 60.0, 2 * N_GRID, 6)]
    with pytest.raises(ValueError, match="is_L_train"):
        summarise_across_L(rows)


# --------------------------------------------------------------------------------------
# 4. THE NONDIM TRAP — a reloaded nondim checkpoint must simulate the same PHYSICS
# --------------------------------------------------------------------------------------
def _nondim_twin(model, L_train):
    """The same physical model as `model`, but with theta_D stored in nondim units."""
    import copy
    twin = copy.deepcopy(model)
    with torch.no_grad():
        twin.theta_D.copy_(torch.log(model.D.detach() / L_train ** 2))
    return twin


def test_nondim_checkpoint_converts_back_to_the_physical_diffusivity():
    m = turing_model()
    D_phys = m.D.detach().cpu().numpy().copy()
    twin = _nondim_twin(m, FIXTURE_L)
    assert not np.allclose(twin.D.detach().cpu().numpy(), D_phys), "twin is not nondim"

    conv, L_train = physical_model_from_checkpoint(twin, dict(nondim=True, L=FIXTURE_L))
    assert L_train == FIXTURE_L
    assert np.allclose(conv.D.detach().cpu().numpy(), D_phys, rtol=1e-10)
    # the caller's model is untouched — the conversion works on a copy
    assert not np.allclose(twin.D.detach().cpu().numpy(), D_phys)


def test_the_dimensional_path_is_left_alone():
    m = turing_model()
    conv, L_train = physical_model_from_checkpoint(m, dict(nondim=False, L=FIXTURE_L))
    assert np.allclose(conv.D.detach().cpu().numpy(), m.D.detach().cpu().numpy(), rtol=0)
    assert L_train == FIXTURE_L


def test_nondim_and_dimensional_checkpoints_give_the_same_physical_kstar():
    """THE ROUND TRIP. Same physics, two storage conventions, one answer per L."""
    L_values = [FIXTURE_L, 2 * FIXTURE_L]
    dim, _ = physical_model_from_checkpoint(turing_model(), dict(nondim=False, L=FIXTURE_L))
    nd, _ = physical_model_from_checkpoint(_nondim_twin(turing_model(), FIXTURE_L),
                                          dict(nondim=True, L=FIXTURE_L))
    a = evaluate_across_L(dim, FIXTURE_L, L_values, n_grid=N_GRID, seed=0)
    b = evaluate_across_L(nd, FIXTURE_L, L_values, n_grid=N_GRID, seed=0)
    for ra, rb in zip(a["per_L"], b["per_L"]):
        assert ra["L"] == rb["L"]
        assert rb["kstar_phys"] == pytest.approx(ra["kstar_phys"], rel=1e-6), (
            f"the nondim checkpoint simulated a different physical k* at L={ra['L']:g}: "
            f"{rb['kstar_phys']} vs {ra['kstar_phys']}")
    assert b["summary"]["kstar_phys_cv"] == pytest.approx(a["summary"]["kstar_phys_cv"],
                                                         rel=1e-6)


def test_an_unconverted_nondim_model_would_have_been_wrong():
    """The failure this guards against is silent, so pin that it IS a failure: simulating
    the raw nondim model integrates D/L**2 and does not reproduce the physical k*."""
    twin = _nondim_twin(turing_model(), FIXTURE_L)
    raw = evaluate_across_L(twin, FIXTURE_L, [FIXTURE_L], n_grid=N_GRID, seed=0)
    good = evaluate_across_L(turing_model(), FIXTURE_L, [FIXTURE_L], n_grid=N_GRID, seed=0)
    raw_row, good_row = raw["per_L"][0], good["per_L"][0]
    assert good_row["patterned"], "the reference row must pattern for this to mean anything"
    # D smaller by L**2 = 3600 destroys the instability outright — no pattern at all here.
    assert not (raw_row["patterned"] and
                abs(raw_row["kstar_phys"] - good_row["kstar_phys"])
                / good_row["kstar_phys"] < 0.1), (
        "the unconverted nondim model reproduced the physical k*; then the conversion in "
        "physical_model_from_checkpoint would be untested by this suite")


@pytest.mark.parametrize("payload", [dict(L=60.0), dict(nondim=True), dict()])
def test_a_checkpoint_without_the_nondim_extras_fails_loud(payload):
    """Older checkpoints carry neither flag, and the two paths are indistinguishable
    afterwards. Assuming dimensional would silently simulate the wrong diffusivity."""
    with pytest.raises(KeyError, match="cannot be placed in physical units"):
        physical_model_from_checkpoint(turing_model(), payload)


def test_a_real_checkpoint_round_trips_but_a_bare_one_raises(tmp_path):
    rdir = IO.run_dir(str(tmp_path), "run_x")
    m = RNGRN(N=3, seed=0)
    IO.save_checkpoint(rdir, m)                                     # no extras: legacy shape
    loaded, payload = IO.load_checkpoint(rdir)
    with pytest.raises(KeyError, match="nondim"):
        physical_model_from_checkpoint(loaded, payload)

    IO.save_checkpoint(rdir, m, extra=dict(nondim=True, L=40.0, kstar_obs=0.3))
    loaded, payload = IO.load_checkpoint(rdir)
    conv, L_train = physical_model_from_checkpoint(loaded, payload)
    assert L_train == 40.0
    assert np.allclose(conv.D.detach().cpu().numpy(),
                       m.D.detach().cpu().numpy() * 40.0 ** 2, rtol=1e-10)


def test_a_nonsense_checkpoint_L_raises():
    with pytest.raises(ValueError, match="finite and > 0"):
        physical_model_from_checkpoint(turing_model(), dict(nondim=True, L=0.0))


# --------------------------------------------------------------------------------------
# 5. the resolution floor, and the argument guards
# --------------------------------------------------------------------------------------
def test_below_the_pixels_per_wavelength_floor_it_refuses_to_report_a_number():
    """At n_grid=16 the fixture sits at 2.8 px/wavelength, where the measured k* error is
    6.9% and the morphology class flips. That is not a noisy measurement, it is a wrong one."""
    with pytest.raises(ValueError, match="pixels per wavelength"):
        evaluate_across_L(turing_model(), FIXTURE_L, [FIXTURE_L], n_grid=16, seed=0)


def test_the_floor_is_the_documented_one():
    assert PPW_FLOOR == 6.0, (
        "PPW_FLOOR is a recorded decision (docs/DECISIONS.md D15) with measured provenance; "
        "moving it silently re-interprets every k* this module has ever reported")


@pytest.mark.parametrize("kw", [dict(L=1.0), dict(n=64)])
def test_L_and_n_may_not_be_smuggled_through_sim_kw(kw):
    with pytest.raises(TypeError, match="sets"):
        evaluate_across_L(turing_model(), FIXTURE_L, [FIXTURE_L], n_grid=N_GRID, **kw)


def test_record_kstar_false_is_refused():
    with pytest.raises(ValueError, match="record_kstar"):
        evaluate_across_L(turing_model(), FIXTURE_L, [FIXTURE_L], n_grid=N_GRID,
                          record_kstar=False)


def test_L_train_is_always_evaluated_and_never_duplicated():
    out = evaluate_across_L(turing_model(), FIXTURE_L, [2 * FIXTURE_L], n_grid=N_GRID, seed=0)
    assert [r["L"] for r in out["per_L"]] == [FIXTURE_L, 2 * FIXTURE_L]
    assert sum(r["is_L_train"] for r in out["per_L"]) == 1
    out2 = evaluate_across_L(turing_model(), FIXTURE_L, [FIXTURE_L, FIXTURE_L],
                             n_grid=N_GRID, seed=0)
    assert len(out2["per_L"]) == 1, "the same domain size was simulated twice"


# --------------------------------------------------------------------------------------
# 6. a model that does not pattern is a RESULT, not an error
# --------------------------------------------------------------------------------------
def test_a_linearly_stable_model_reports_no_pattern_instead_of_classifying_noise():
    """A random-init RNGRN is linearly stable, so its field decays to float noise. Handing
    that noise to the morphology classifier would be a false positive (the same guard
    train._morphology_rollout applies), and there is no k* to be invariant."""
    m = RNGRN(N=3, seed=0)
    out = evaluate_across_L(m, 100.0, [100.0, 200.0], n_grid=32, seed=0)
    rows = out["per_L"]
    assert not any(r["patterned"] for r in rows), [r["amplitude"] for r in rows]
    assert all(r["morph_class"] is None and r["morph_vector"] is None for r in rows)
    assert all(np.isnan(r["pixels_per_wavelength"]) for r in rows), (
        "an unformed wavelength has no pixels per wavelength; that NaN is deliberate")
    s = out["summary"]
    assert s["n_patterned"] == 0 and s["n_L"] == 2
    assert np.isnan(s["kstar_phys_cv"]) and np.isnan(s["morphology_class_preserved"])
