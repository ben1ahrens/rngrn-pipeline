"""test_lift_ladder.py — V0 of the lift-validation ladder: algebraic invariants of the
lifted system (eval/ladder.py), at N=3, both regulation forms, over >=20 generator draws.

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
