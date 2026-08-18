"""test_lift_ladder.py — the lift-validation ladder (eval/ladder.py), at N=3 and both
regulation forms: V0 algebraic invariants over >=20 generator draws, V1 eigenvalue
continuation, V2 the 0-D lifted ODE against a Radau reference. One section per rung, in order.

V0 — algebraic invariants of the lifted system.

docs/REDESIGN_rngrn.md §5.3 V0 says V0 MEASURES the achievable floor and records it rather
than asserting an inherited one. The original draft of this test asserted the single-model
1e-7 bar from tests/test_lifted.py directly and found it did NOT hold over a 20-draw
population (task-2-report.md §4: competitive 2.56e-7, nc1 1.27e-6). CONTROLLER RULING
ROUND 1 (ledger 2026-08-17) on that finding, implemented here and in eval/ladder.py's module
docstring:
  1. draw_models now filters to NEWTON-TIGHT steady states only (excludes
     steady_state's looser relaxation-fallback acceptances) -- closes the mechanism where a
     loose x* (not the lift algebra) was driving the residual up.
  2. The remaining mu-dependence (a real property of eval/lifted.py's own arithmetic, not a
     ladder.py defect -- see ladder.py's module docstring, mechanism (b)) is captured by a
     MEASURED, mu-aware bound instead of a fixed constant.

RULING ROUND 2 (ledger 2026-08-17, code review of round 1). Round 1's single amplification
constant A was measured over the WHOLE mu grid, so it was FLOOR-dominated -- the x-block
round-off floor (flat in mu) sets the max of `residual*mu/eps` at the LARGEST mu, not the
smallest, making the resulting bound ~5 orders of magnitude too loose exactly where V1
(mu in {1e-7, 1e-6, 1e-5}) needs it tight. Round 2 measures the two mechanisms SEPARATELY,
each over only the mu region where it actually dominates -- see eval/ladder.py's module
docstring and `v0_invariants`'s docstring for the full mechanism writeup:
  - `residual_floor_F` = max residual over the FLAT region (mu in {1e-2, 1, 1e2}) -- the
    x-block round-off floor, mu-independent.
  - `amplification_C` = max of `residual(mu) * mu / eps` over the GATE-AMPLIFICATION region
    (mu in {1e-6, 1e-4}) -- the C/mu mechanism proper.
"""
import os
import sys

import numpy as np
import pytest

from rngrn.eval import ladder

MUS = [1e-6, 1e-4, 1e-2, 1.0, 1e2]

# float64 machine epsilon -- must match ladder.EPS64 exactly (repeated here rather than
# imported so the bound in this test is legible on its own).
EPS64 = 2.220446049250313e-16

# F_FORM (residual_floor_F) and C_FORM (amplification_C), MEASURED (ruling round 2, ledger
# 2026-08-17) by draw_models(n=20, form=form, seed=101) + v0_invariants(models, mus=MUS) on
# this branch, 2026-08-17:
#   competitive: F=6.266787e-11, C=1154.500000  (exclusion_rate 0/20 steady_state-converged)
#   nc1:         F=2.795897e-11, C=12.591600    (exclusion_rate 1/21 steady_state-converged)
# Frozen here with a small (<1%) rounding-up margin above the measured value; the per-mu
# bound below already carries its own 10x safety factor on top of that. UNCALIBRATED ->
# measured, per docs/DECISIONS.md discipline (CLAUDE.md §10) -- see task-2-report.md §§4,8,9
# for the full measurement history and the two-round decision-point exchange that produced
# this ruling.
F_FORM = {"competitive": 6.3e-11, "nc1": 2.8e-11}
C_FORM = {"competitive": 1160.0, "nc1": 12.6}


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_v0_invariants_hold_at_the_measured_bar(form):
    models = ladder.draw_models(n=20, form=form, seed=101)
    out = ladder.v0_invariants(models, mus=MUS)
    assert out["n_models"] >= 20

    # rescale_mu / gate_qss: the brief's original bars, UNCHANGED by either ruling round --
    # both pass by 6+ orders of magnitude (measured ~1e-15..1e-16 here).
    assert out["max_rescale_mu_err"] < 1e-9
    assert out["max_gate_qss_err"] < 1e-9

    # fixed_point_residual: MEASURED, mu-aware, two-mechanism bound (ruling round 2, ledger
    # 2026-08-17), tight in both the flat-floor and the gate-amplification regime. Checked
    # per mu, not just on the aggregate max, so a bar violation concentrated at one mu
    # cannot hide behind a looser one elsewhere in the grid.
    f, c = F_FORM[form], C_FORM[form]
    for mu in MUS:
        resid = ladder.v0_invariants(models, mus=[mu])["max_fixed_point_residual"]
        bound = max(1e-9, 10.0 * f, 10.0 * c * EPS64 / mu)
        assert resid <= bound, (form, mu, resid, bound)


def test_draw_models_reports_exclusion_rate_and_v0_invariants_surfaces_it():
    """exclusion_rate is a deliverable (ruling, ledger 2026-08-17), not a hidden filter."""
    models = ladder.draw_models(n=20, form="nc1", seed=101)
    assert 0.0 <= models.exclusion_rate <= 1.0
    out = ladder.v0_invariants(models, mus=[1.0])
    assert out["exclusion_rate"] == models.exclusion_rate


def test_v0_invariants_reports_nan_exclusion_rate_for_a_plain_list():
    """A model list not produced by draw_models carries no exclusion-rate bookkeeping --
    v0_invariants must say so (NaN) rather than fabricate a rate."""
    import math

    models = list(ladder.draw_models(n=3, form="nc1", seed=101))  # unwrap to a plain list
    out = ladder.v0_invariants(models, mus=[1.0])
    assert math.isnan(out["exclusion_rate"])


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


# ======================================================================================
# V1 — eigenvalue continuation of all N + 2N^2 = 21 lifted branches as mu -> 0
# (docs/REDESIGN_rngrn.md §5.3 V1)
# ======================================================================================
@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_v1_slow_branches_converge_to_qss_at_first_order(form):
    """The three V1 claims, on one draw per form.

    The 0.35 tolerance on the order is the brief's, and it is a tolerance on an EMPIRICAL
    log-log slope, not a physical constant: the claim being tested is "order ~ 1", i.e. the
    slow branches are O(mu) away from the QSS eigenvalues, not O(1) and not O(mu^2). The
    measured orders on this branch, 2026-08-17, are recorded in
    `test_v1_reports_the_measured_orders_and_errors` below.
    """
    m = ladder.draw_models(n=1, form=form, seed=7)[0]
    kgrid = ladder.default_kgrid(m)
    out = ladder.v1_continuation(m, kgrid)
    assert out["fast_branches_stable"] is True
    assert abs(out["slow_branch_order"] - 1.0) < 0.35
    assert out["kstar_grid_offset"] <= 1


def test_v1_all_21_branches_are_accounted_for():
    """N slow + 2N^2 fast = 21 at N=3: no branch is dropped and none is double-counted."""
    m = ladder.draw_models(n=1, form="competitive", seed=7)[0]
    out = ladder.v1_continuation(m, ladder.default_kgrid(m), mus=(1e-7, 1e-6))
    assert out["n_slow_branches"] == m.N
    assert out["n_fast_branches"] == 2 * m.N * m.N
    assert out["n_slow_branches"] + out["n_fast_branches"] == 21


def test_v1_error_falls_with_mu_and_fast_branches_scale_as_one_over_mu():
    """max_slow_err must DECREASE as mu -> 0 (that is the convergence claim), and every fast
    branch must sit at |Re lambda| * mu ~ 1 + u with u >= 0, i.e. the -(1+u)/mu scaling.

    The bar on the product is 0.9, not 1.0: lambda_fast = -(1+u)/mu + O(1), so the product is
    1 + u + O(mu) and can sit a hair BELOW 1 (measured 0.99999 on competitive seed 7, 0.98891
    on competitive seed 101). What is being tested is that the fast branches scale as 1/mu at
    all -- 0.9 rules out any other power without pretending the O(mu) correction is absent.
    """
    m = ladder.draw_models(n=1, form="nc1", seed=7)[0]
    out = ladder.v1_continuation(m, ladder.default_kgrid(m))
    errs = [out["max_slow_err"][mu] for mu in sorted(out["max_slow_err"])]
    assert errs == sorted(errs), errs          # monotone increasing in mu
    assert out["min_fast_mu_product"] >= 0.9, out["min_fast_mu_product"]


def test_v1_kstar_offset_is_measured_where_qss_kstar_is_INTERIOR():
    """The k* claim, exercised NON-vacuously.

    `draw_models` samples init="low_basal", and eval/analysis.py::turing_ok records 0 of 398
    such N=3 draws as strictly Turing -- so on most draws sigma_qss(k) is monotone decreasing,
    both k* pin to the grid FLOOR, and `kstar_grid_offset == 0` holds by construction rather
    than by branch tracking. nc1 seed 202 draw 0 is one of the 2-in-30 draws (both forms,
    seeds 7/101/202/303/404) whose sigma_qss(k) has an INTERIOR maximum -- grid index 3,
    k = 0.0385 -- so there the offset is a real measurement of the lifted band against the QSS
    band. Measured 0 at all three mu, 2026-08-17.

    The vacuous case is asserted to REPORT its vacuity (`kstar_at_grid_floor`), so
    `kstar_grid_offset == 0` is never read as evidence it is not.
    """
    interior = ladder.draw_models(n=1, form="nc1", seed=202)[0]
    out = ladder.v1_continuation(interior, ladder.default_kgrid(interior))
    assert out["kstar_at_grid_floor"] is False
    assert out["kstar_qss"] > ladder.default_kgrid(interior)[0]
    assert out["kstar_grid_offset"] <= 1

    floor = ladder.draw_models(n=1, form="nc1", seed=7)[0]
    at_floor = ladder.v1_continuation(floor, ladder.default_kgrid(floor))
    assert at_floor["kstar_at_grid_floor"] is True


def test_v1_kgrid_must_exclude_k_zero():
    """k = 0 is the uniform mode, not a wavenumber of the structured band -- and log(0)
    would enter no slope here, but a k=0 column would silently make `kstar_grid_offset`
    an offset against the uniform mode. Fail loud instead."""
    import numpy as np

    m = ladder.draw_models(n=1, form="nc1", seed=7)[0]
    with pytest.raises(ValueError):
        ladder.v1_continuation(m, np.array([0.0, 1.0, 2.0]), mus=(1e-6, 1e-5))


def test_v1_requires_at_least_two_mu_values():
    m = ladder.draw_models(n=1, form="nc1", seed=7)[0]
    with pytest.raises(ValueError):
        ladder.v1_continuation(m, ladder.default_kgrid(m), mus=(1e-6,))


def test_match_by_overlap_beats_real_part_sorting_on_a_crossing():
    """THE reason V1 tracks branches by eigenvector overlap and never by sorting Re.

    Two diagonal 2x2 matrices sharing the eigenvectors e1, e2. Between them the two
    eigenvalues CROSS in real part: (1.0, 2.0) -> (2.5, 0.5). Sorting by Re would pair
    1.0 <-> 0.5 and 2.0 <-> 2.5 -- swapping the branches. Overlap pairs e1 <-> e1.
    """
    import numpy as np

    Va = np.eye(2, dtype=complex)[None]                     # columns e1, e2
    Vb = np.eye(2, dtype=complex)[:, ::-1][None]            # columns e2, e1
    wa = np.array([[1.0, 2.0]])
    wb = np.array([[0.5, 2.5]])                             # column order matches Vb
    pair = ladder.match_by_overlap(Va, Vb)
    assert pair.tolist() == [[1, 0]]                        # e1 -> Vb col 1, e2 -> col 0
    matched = wb[0][pair[0]]
    assert matched.tolist() == [2.5, 0.5]                   # branch identity preserved
    resorted = np.sort(wb[0])[np.argsort(np.argsort(wa[0]))]
    assert resorted.tolist() == [0.5, 2.5]                  # what Re-sorting would give
    assert matched.tolist() != resorted.tolist()


def test_v1_overlap_matching_is_exercised_on_a_real_draw():
    """Not a synthetic crossing: on this draw the overlap assignment between adjacent mu
    disagrees with descending-Re relabelling at a non-zero number of (k, mu-pair) sites, so
    the Re-sorting shortcut would have tracked different branches."""
    m = ladder.draw_models(n=1, form="competitive", seed=7)[0]
    out = ladder.v1_continuation(m, ladder.default_kgrid(m))
    assert out["n_resort_mismatches"] > 0


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_v1_reports_the_measured_orders_and_errors(form):
    """V1's error constants are UNCALIBRATED per docs/REDESIGN_rngrn.md §5.3 ("measured and
    recorded, not invented"), so this test records the MEASURED values rather than asserting
    an inherited bar. Measured on this branch, 2026-08-17, on
    draw_models(n=1, form=form, seed=7)[0] over the full analysis scan band:

      competitive: slow_branch_order 1.0239 (all-k 1.0239), frac_k_separated 1.000,
                   max_slow_err {1e-7: 2.92e-4, 1e-6: 2.96e-3, 1e-5: 3.46e-2},
                   min_fast_mu_product 1.000, n_resort_mismatches 8000
      nc1:         slow_branch_order 1.0268 (all-k 1.0268), frac_k_separated 1.000,
                   max_slow_err {1e-7: 4.58e-4, 1e-6: 4.68e-3, 1e-5: 5.97e-2},
                   min_fast_mu_product 1.000, n_resort_mismatches 0

    The bands below are deliberately loose (a factor of 10 on the errors) -- they exist to
    catch a REGRESSION in the lift, not to pin a physical constant. `n_resort_mismatches` is
    0 for nc1 seed 7 and 8000 for competitive seed 7; the overlap machinery is exercised
    unconditionally by `test_match_by_overlap_beats_real_part_sorting_on_a_crossing` and on a
    real draw by the competitive case above, so nothing is asserted about it here.
    """
    expected_err = {"competitive": {1e-7: 2.92e-4, 1e-6: 2.96e-3, 1e-5: 3.46e-2},
                    "nc1": {1e-7: 4.58e-4, 1e-6: 4.68e-3, 1e-5: 5.97e-2}}[form]
    m = ladder.draw_models(n=1, form=form, seed=7)[0]
    out = ladder.v1_continuation(m, ladder.default_kgrid(m))
    assert 0.65 <= out["slow_branch_order"] <= 1.35
    assert out["frac_k_separated"] > 0.5
    for mu, e in expected_err.items():
        assert 0.1 * e <= out["max_slow_err"][mu] <= 10.0 * e, (mu, out["max_slow_err"][mu])


# ======================================================================================
# V2 — temporal, 0-D: the 21-dim well-mixed lifted ODE against a stiff Radau reference
# ======================================================================================
# THE HORIZON IS SHORTENED AT SMALL mu, AND HERE IS THE ARITHMETIC. The brief's test used
# T = 5.0 at every mu. At mu = 1e-5 the finest requested step is dt = mu/8 = 1.25e-6, so
# T = 5.0 is 4e6 Strang steps for ONE dt and 7e6 for the halving triplet -- at the measured
# ~120 us/step that is ~14 minutes for a single test case. The order measurement needs
# ENOUGH STEPS for the dt^2 term to dominate, not a long horizon, so the horizon is capped at
# 1000 gate relaxation times:
#
#     T(mu) = min(5.0, 1000 * mu)
#
# which leaves >= 8000 steps at the finest dt in every case (T/(mu/8) = 8000 when the cap
# binds, 4000 at mu = 1e-2 where the 5.0 rollout horizon binds instead). The gate transient
# has width ~mu, so 1000 gate times resolves it and then some. This is a COST adjustment to
# the brief, not a bar adjustment: the order band (2.0 +- 0.5) and the sup-norm bar (1e-4)
# are exactly as briefed. Recorded in task-4-report.md.
V2_ROLLOUT_HORIZON = 5.0
V2_GATE_TIMES = 1000.0


def _v2_horizon(mu):
    return min(V2_ROLLOUT_HORIZON, V2_GATE_TIMES * mu)


# mu = 1e-3 is mu_gate, the owner's evaluation point for the lifted gate (owner decision,
# 2026-08-17), added alongside the spec's mu_central = 7.2e-4; both are reported.
@pytest.mark.parametrize("mu", [1e-5, 7.2e-4, 1e-3, 1e-2])
def test_v2_strang_matches_radau_and_shows_order_two(mu):
    m = ladder.draw_models(n=1, form="competitive", seed=11)[0]
    out = ladder.v2_temporal(m, mu=mu, T=_v2_horizon(mu), dts=[mu / 2, mu / 4, mu / 8])
    assert out["radau_ok"], out
    assert abs(out["strang_order"] - 2.0) < 0.5, out          # dt <~ mu regime
    # <= 0.1 x pattern_floor with the x*-free 0-D floor, REDESIGN §5.3 V2: pattern_floor =
    # max(1e-3, 0.02*|x*_0|) and the ABSOLUTE arm (1e-3) is the x*-free one, so 0.1 x it is
    # 1e-4.
    assert out["sup_err_at_horizon"] < 1e-4, out


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_v2_numpy_rhs_mirrors_lifted_rhs_torch(form):
    """The Radau reference needs a numpy RHS (a torch call per function evaluation is ~45 us
    and Radau makes tens of thousands). `ladder._lifted_rhs_0d` is that mirror, and this
    pins it to eval/lifted.py::lifted_rhs_torch -- the authority -- rather than trusting that
    the two transcriptions agree."""
    m = ladder.draw_models(n=1, form=form, seed=11)[0]
    assert ladder.v2_rhs_mirror_error(m, mus=[1e-6, 1e-3, 1.0], n_states=25) < 1e-14


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_v2_lifted_trajectories_converge_to_qss_at_first_order(form):
    """Part (i) of REDESIGN §5.3 V2: the lifted trajectory converges to the QSS 3-dim ODE
    trajectory as mu -> 0. Both sides integrated by the SAME Radau reference, so this measures
    the lift and not a stepper. Order in mu is a MEASUREMENT (§5.3: "absolute error constants
    UNCALIBRATED"), so the band is loose and exists to catch a regression."""
    m = ladder.draw_models(n=1, form=form, seed=11)[0]
    out = ladder.v2_qss_limit(m, mus=[1e-6, 1e-5, 1e-4], T=5.0)
    assert out["radau_ok"], out
    assert abs(out["qss_order"] - 1.0) < 0.35, out
    assert out["qss_gap"][1e-6] < out["qss_gap"][1e-4], out


def test_v2_strang_order_degrades_for_dt_much_larger_than_mu():
    """THE KNOWN COUPLING TRAP, pinned so it is documented rather than discovered later.

    `lifted.gate_step_exact` buys unconditional STABILITY at any dt -- that is what it is for
    (eval/lifted.py's §"THE STIFFNESS" comment) -- but NOT second-order accuracy once dt >> mu:
    the Strang splitting error is controlled by commutators carrying the 1/mu gate rate, so the
    observed order falls away from 2 exactly where the exact substep is doing the most work.
    Measured on this branch, 2026-08-17, competitive seed 11 at mu = 1e-3 with
    dt/mu in {8, 4, 2}: order 1.49 (pairs 1.33, 1.65), against 1.99 at dt/mu <= 0.5.

    The run still COMPLETES (no blow-up) -- which is the stability half of the claim -- and
    `dt_over_mu_max` is what tells a reader which regime a V2 number came from.
    """
    m = ladder.draw_models(n=1, form="competitive", seed=11)[0]
    mu = 1e-3
    out = ladder.v2_temporal(m, mu=mu, T=1.0, dts=[8 * mu, 4 * mu, 2 * mu])
    assert out["radau_ok"], out
    assert out["dt_over_mu_max"] > 4.0, out
    assert out["strang_order"] < 1.75, out          # materially below the dt <~ mu order 2


def test_v2_reports_the_measured_orders_and_errors():
    """V2's absolute error constants are MEASUREMENTS (docs/REDESIGN_rngrn.md §5.3: "measured
    and recorded, not invented"), so this records them rather than asserting an inherited bar.
    Measured on this branch, 2026-08-17, draw_models(n=1, form="competitive", seed=11)[0],
    dts = [mu/2, mu/4, mu/8], T = _v2_horizon(mu):

      mu      T      order   sup_err(mu/2, mu/4, mu/8)          sup_err_at_horizon  radau margin
      1e-2    5.0    1.994   6.93e-4, 1.74e-4, 4.36e-5          4.36e-5             2.0e6
      1e-3    1.0    1.995   8.50e-6, 2.14e-6, 5.35e-7          5.35e-7             4.0e4
      7.2e-4  0.72   1.995   4.41e-6, 1.11e-6, 2.77e-7          2.77e-7             1.9e4
      1e-5    0.01   1.994   4.61e-9, 1.16e-9, 2.90e-10         2.81e-10            1.0e4

    The band below is a factor of 10 on the error -- it exists to catch a REGRESSION in the
    stepper, not to pin a physical constant. mu = 1e-2 is the binding case for the 1e-4 bar
    (4.36e-5, a factor 2.3 of margin) because it is the only mu where the 5.0 rollout horizon
    binds instead of the 1000-gate-time cap.
    """
    expected = {1e-2: 4.36e-5, 1e-3: 5.35e-7, 7.2e-4: 2.77e-7, 1e-5: 2.81e-10}
    m = ladder.draw_models(n=1, form="competitive", seed=11)[0]
    for mu, e in expected.items():
        out = ladder.v2_temporal(m, mu=mu, T=_v2_horizon(mu), dts=[mu / 2, mu / 4, mu / 8])
        assert 0.1 * e <= out["sup_err_at_horizon"] <= 10.0 * e, (mu, out)
        assert out["dt_over_mu_max"] <= 1.0, (mu, out)   # the dt <~ mu scope of the order claim


def test_v2_temporal_requires_at_least_two_dt_values():
    m = ladder.draw_models(n=1, form="competitive", seed=11)[0]
    with pytest.raises(ValueError, match=">=2 dt values"):
        ladder.v2_temporal(m, mu=1e-3, T=1.0, dts=[1e-4])


# ======================================================================================
# V3 — spatial: the lifted PDE against the QSS rollout
# ======================================================================================
# THE FIXTURE IS tests/test_lifted_torch.py's, VERBATIM: draw_models(n=1, form, seed=23)[0]
# on a 32x32 box of L = 20 with dt = 5e-4, T = 0.2, seed = 5. It is reused rather than
# reinvented so a V3 number is directly comparable to the CPU/GPU equivalence the Task 5
# tests pin on the same 400 steps, and because a SHORT horizon keeps the field in the
# growing-from-noise regime where the lifted-vs-QSS difference is a measurement rather than
# two independently saturated attractors compared at the end.
V3_KW = dict(L=20.0, n=32, dt=5e-4, T=0.2, seed=5)

# THE FULL mu SET the campaign runs V3(a) at: the spec's {1e-4, 1e-5, 1e-6} plus mu_gate =
# 1e-3 (owner decision D-REDESIGN-5, ledger 2026-08-18).
V3_MUS = [1e-3, 1e-4, 1e-5, 1e-6]

# The mu range over which the lifted-vs-QSS field difference is dominated by the LIFT, and
# the step-by-step monotonicity claim is therefore a claim at all. See
# `test_v3_l2_difference_floors_at_an_O_dt_scheme_difference` for the measurement that
# establishes the other regime and why this scoping is not a weakened bar.
V3_LIFT_DOMINATED_MUS = [1e-2, 1e-3, 1e-4]


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_v3_returns_the_briefed_keys_and_agrees_with_the_qss_rollout(form):
    m = ladder.draw_models(n=1, form=form, seed=23)[0]
    out = ladder.v3_spatial(m, mus=V3_MUS, **V3_KW)
    for key in ("patterned_agree", "morphology_agree", "kstar_within_one_bin",
                "l2_diff_by_mu"):
        assert key in out, (key, sorted(out))
    assert isinstance(out["patterned_agree"], bool)
    assert isinstance(out["morphology_agree"], bool)
    assert isinstance(out["kstar_within_one_bin"], bool)
    assert sorted(out["l2_diff_by_mu"]) == sorted(float(mu) for mu in V3_MUS)
    # the fixture grows a real field from noise on both sides, so the flag is not vacuous
    assert out["patterned_qss"] is True, out
    assert out["patterned_agree"] is True, out
    assert out["kstar_within_one_bin"] is True, out
    assert out["one_bin"] == pytest.approx(2 * np.pi / V3_KW["L"])


def test_v3_l2_difference_falls_with_mu():
    """§5.3 V3(a): 'field relative L2 difference decreasing with mu'.

    TWO claims, because the measurement supports two different strengths (see the floor test
    below): end-to-end over the whole campaign mu set, and step-by-step over the mu range
    where the lift term dominates the O(dt) scheme term.
    """
    m = ladder.draw_models(n=1, form="competitive", seed=23)[0]
    full = ladder.v3_spatial(m, mus=V3_MUS, **V3_KW)
    d = full["l2_diff_by_mu"]
    assert d[min(V3_MUS)] < d[max(V3_MUS)], d          # end-to-end, the whole campaign set

    lift = ladder.v3_spatial(m, mus=V3_LIFT_DOMINATED_MUS, **V3_KW)
    vals = [lift["l2_diff_by_mu"][float(mu)] for mu in sorted(V3_LIFT_DOMINATED_MUS)[::-1]]
    assert all(b <= a for a, b in zip(vals, vals[1:])), vals   # non-increasing as mu falls
    assert lift["l2_monotone"] is True, lift["l2_diff_by_mu"]


def test_v3_l2_difference_floors_at_an_O_dt_scheme_difference():
    """WHY THE MONOTONICITY CLAIM IS SCOPED, measured rather than asserted.

    The lifted-vs-QSS field difference has TWO components, and only one of them is the lift:

      * an O(mu) term -- the gate-tracking lag the lift is *about*, which vanishes as mu -> 0;
      * an O(dt) term that does NOT -- at mu -> 0 the exact gate substep sets G = G_qss(x)
        and the two runs still differ, because `simulate_lifted` FREEZES the production over
        the ETDRK4 step (Strang) while `rollout.simulate` re-evaluates the QSS reaction at
        every ETDRK4 stage. That is a difference of SCHEME at fixed dt, not of physics.

    Measured on this branch 2026-08-18, competitive seed 23, V3_KW with dt varied:
        mu = 1e-3:  5.01e-7, 5.36e-7, 5.45e-7  at dt = 5e-4, 2.5e-4, 1.25e-4  (dt-INDEPENDENT)
        mu = 1e-6:  2.04e-7, 9.70e-8, 4.70e-8  at the same dt                 (HALVES with dt)

    So at fixed dt the curve falls with mu until it reaches the O(dt) floor and then flattens
    (measured 1.63e-7 at mu = 1e-4 and 2.04e-7 at mu = 1e-5 and 1e-6 -- a 25% RISE, which is
    why step-by-step monotonicity is claimed only above the floor). V3(b)'s dt policy
    dt = min(0.2/jac_rate, mu/2) ties dt to mu precisely so both terms vanish together.
    """
    m = ladder.draw_models(n=1, form="competitive", seed=23)[0]
    kw = dict(V3_KW)
    kw.pop("dt")
    a = ladder.v3_spatial(m, mus=[1e-6], dt=5e-4, **kw)["l2_diff_by_mu"][1e-6]
    b = ladder.v3_spatial(m, mus=[1e-6], dt=2.5e-4, **kw)["l2_diff_by_mu"][1e-6]
    assert 1.7 < a / b < 2.3, (a, b)      # first order in dt at the mu -> 0 limit


def test_v3_torch_backend_reproduces_the_numpy_one_on_cpu():
    """V3(b) runs on the GPU (Task 5's port); this pins the driver's backend switch to the
    numpy path it is supposed to be a faster way of computing, on the CPU where the contract
    is bit-equality (tests/test_lifted_torch.py)."""
    m = ladder.draw_models(n=1, form="competitive", seed=23)[0]
    a = ladder.v3_spatial(m, mus=[1e-3], **V3_KW)
    b = ladder.v3_spatial(m, mus=[1e-3], backend="torch", device="cpu", **V3_KW)
    assert b["backend"] == "torch"
    assert b["patterned_by_mu"][1e-3] == a["patterned_by_mu"][1e-3]
    assert b["l2_diff_by_mu"][1e-3] == pytest.approx(a["l2_diff_by_mu"][1e-3], rel=1e-12)


def test_v3_rejects_an_unknown_backend():
    m = ladder.draw_models(n=1, form="competitive", seed=23)[0]
    with pytest.raises(ValueError, match="backend"):
        ladder.v3_spatial(m, mus=[1e-3], backend="jax", **V3_KW)


# ======================================================================================
# V4 — the re-entrant-band survey
# ======================================================================================
def test_v4_survey_returns_the_briefed_keys_on_a_low_basal_population():
    """SHAPE and the empty-numerator contract. `draw_models` samples init='low_basal', of
    which 0 of 398 N=3 draws are STRICTLY Turing (eval/analysis.py, quoted in
    `v1_continuation`'s docstring), so this population has no QSS-Turing member and
    `p_lifted_given_qss` is a 0/0 -- reported NaN, with `n_qss_turing` beside it, never 0.0.
    """
    models = ladder.draw_models(n=3, form="competitive", seed=101)
    out = ladder.v4_survey(models, n_mu=3)
    for key in ("p_lifted_given_qss", "frac_reentrant", "edges"):
        assert key in out, (key, sorted(out))
    assert len(out["edges"]) == 3
    assert out["n_models"] == 3
    assert out["n_qss_turing"] == 0, out
    assert np.isnan(out["p_lifted_given_qss"]), out
    assert np.isnan(out["frac_reentrant"]), out
    for row in out["edges"]:
        for key in ("qss_turing", "mu_crit", "reentrant", "status",
                    "lifted_turing_band_all"):
            assert key in row, (key, sorted(row))


def test_v4_qss_verdict_is_strict_and_not_the_trace_test():
    """The QSS control V4 conditions on is the SAME strict pair
    `lifted.turing_verdict_lifted` applies to the lifted system -- max Re eig(J) < 0 AND
    max_{k>0} sigma(k) > tol -- evaluated on the REDUCED N x N Jacobian. Not
    eval/analysis.py::turing_ok's trace test, which Stage 0 measured overcounting by 64x.
    """
    m = ladder.draw_models(n=1, form="competitive", seed=101)[0]
    from rngrn.losses.terms import steady_state
    xs, _ = steady_state(m)
    v = ladder.qss_verdict(m, xs.detach().cpu().numpy())
    for key in ("turing_strict", "stable_uniform", "unstable_k", "kstar", "sig_max_pos",
                "max_re_eig_J", "oscillatory"):
        assert key in v, (key, sorted(v))
    assert v["turing_strict"] == (v["stable_uniform"] and v["unstable_k"])


# --------------------------------------------------------------------------------------
# the campaign driver's populations — the part of scripts/lift_ladder.py that is not a CLI
# --------------------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import lift_ladder as LL                                            # noqa: E402


def test_harvest_population_is_the_23_strictly_turing_stage0_draws():
    """The rebuild inverts `model.py`'s link functions on the STORED constrained parameters,
    so this checks the round trip against the npz's own recorded verdict rather than trusting
    the transcription: every rebuilt model must still be strictly Turing under
    `ladder.qss_verdict`, which is the criterion docs/BIO_VIABILITY.md §1 harvested them by.
    """
    from rngrn.losses.terms import steady_state

    models, labels = LL.harvest_models()
    assert len(models) == 23, labels
    assert len(labels) == len(set(labels))
    assert {m.form for m in models} == {"competitive", "nc1"}
    for m, label in zip(models[:4], labels[:4]):
        xs, converged = steady_state(m)
        assert converged, label
        assert ladder.qss_verdict(m, xs.detach().cpu().numpy())["turing_strict"], label


def test_d5_population_is_the_four_turing_unstable_recovered_models():
    from rngrn.losses.terms import steady_state

    models, labels = LL.d5_models()
    assert labels == ["d5/seed1", "d5/seed3", "d5/seed5", "d5/seed6"]
    for m, label in zip(models, labels):
        xs, converged = steady_state(m)
        assert converged, label
        assert ladder.qss_verdict(m, xs.detach().cpu().numpy())["turing_strict"], label


def test_box_size_puts_exactly_p_periods_across_the_box():
    """`L = p * 2*pi / k*` is what makes the BINDING one-radial-bin k* tolerance 2*pi/L equal
    1/p of k* — 12.5% at the target's p = 8 (SPEC §9.1). If this drifts, every V3 k* number
    is judged at a different precision than the target's."""
    m = LL.harvest_models()[0][0]
    L, k, _ = LL.box_size(m, periods=8)
    assert L * k / (2 * np.pi) == pytest.approx(8.0)
    assert ladder.one_radial_bin(L) / k == pytest.approx(1.0 / 8.0)


def test_v4_survey_on_a_turing_positive_pair_reports_a_real_conditional():
    """The shape test above has an EMPTY QSS-Turing denominator by construction; this one
    does not, so it exercises the branch the campaign's headline number comes from."""
    models, labels = LL.harvest_models()
    out = ladder.v4_survey(models[:2], labels=labels[:2], n_mu=3)
    assert out["n_qss_turing"] == 2, out["edges"]
    assert 0.0 <= out["p_lifted_given_qss"] <= 1.0, out
    assert 0.0 <= out["frac_reentrant"] <= 1.0, out
    assert sorted(out["frac_turing_by_mu"]) == sorted(out["mu_grid"])
    assert out["mu_grid"][0] == pytest.approx(1.1e-5)      # MU_BIO_LO
    assert out["mu_grid"][-1] == pytest.approx(9.2e-3)     # MU_BIO_HI
