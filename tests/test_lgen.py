"""test_lgen.py — L-generalisation: non-dimensional recovery + the cross-L metric (unit 12).

Four tiers, in decreasing strength of what they prove:

  1. THE RESCALING IS EXACT. obs.kstar_of and obs.laplacian_torch are exactly homogeneous
     in L, and the dispersion is exactly invariant under (k, D) -> (k*L, D/L**2). These are
     algebraic identities, so they are asserted to float tolerance, not to a threshold.
  2. THE DIMENSIONAL PATH IS UNCHANGED. recover(nondim=False) is compared BIT-FOR-BIT
     against a hand-rolled reference loop that reproduces the pre-unit-12 code. A
     regression here would silently invalidate every number already recorded in the
     project, so it is checked by equality, not by approx.
  3. THE NON-DIMENSIONAL PATH IS L-INVARIANT. The same frame labelled with three different
     L values yields the identical recovered network, and a physical D that scales as L**2.
     This is invariance BY CONSTRUCTION — the nondim objective literally does not read L —
     and the test says so rather than presenting it as an empirical finding.
  4. THE AGGREGATOR BEHAVES. Synthetic run-index rows; no recovery involved.

WHAT THESE TESTS DO NOT SHOW: that recovery on REAL data at several domain sizes agrees
with itself. That needs the three_gene_multiL dataset (unit 11), which did not exist when
this was written. `test_multiL_dataset_is_groupable` consumes it the moment it lands and
skips until then.
"""
import math

import numpy as np
import pytest
import torch

from rngrn import observables as obs
from rngrn.model import RNGRN
from rngrn.scoring import lgen as LGEN


TWO_PI = 2.0 * math.pi


def _frame(n=32, periods=6, seed=0):
    """A deterministic patterned 3-channel frame. Content is irrelevant to every assertion
    here — only that it is the SAME array under all three L labels."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n, endpoint=False)
    X, Y = np.meshgrid(x, x)
    base = 1.0 + 0.3 * np.sin(TWO_PI * periods * X) * np.cos(TWO_PI * periods * Y)
    return np.stack([base + 0.01 * rng.standard_normal((n, n)) for _ in range(3)])


class _RI:
    """A RecoveryInput stand-in (frame, L, observed_idx, N) — the whole firewall surface."""
    def __init__(self, frame, L, N=3):
        self.frame = frame
        self.L = float(L)
        self.observed_idx = tuple(range(frame.shape[0]))
        self.N = N


# --------------------------------------------------------------------------------------
# 1. the rescaling x_hat = x/L is exact
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("L", [18.0, 57.0, 207.7])
def test_kstar_measurement_is_homogeneous_in_L_to_float_noise(L):
    """k*(field, L) * L == k*(field, 1), but only to ~1e-5 relative — MEASURED, not exact.

    raps() is homogeneous in L in exact arithmetic (every k scales as 1/L and so do the
    bin edges), but it bins with np.arange/np.digitize on floating-point edges, so a
    sample sitting near an edge can fall either side depending on L. Swept over 500 values
    of L in [18, 220] on this frame: max relative deviation 1.00e-05, median 4.1e-07.

    This is a bound on how closely the dimensional and non-dimensional paths are the SAME
    problem. It does not touch the L-invariance of the non-dimensional path itself, which
    is exact: there k*_obs is always measured at L = 1.
    """
    f = _frame()[0]
    assert obs.kstar_of(f, L=L) * L == pytest.approx(obs.kstar_of(f, L=1.0), rel=2e-5)


@pytest.mark.parametrize("L", [18.0, 57.0, 207.7])
def test_laplacian_is_exactly_homogeneous_in_L(L):
    """lap_L(f) * L**2 == lap_1(f) — the other half of the change of variables."""
    f = torch.tensor(_frame())
    lhs = obs.laplacian_torch(f, L=L) * (L ** 2)
    rhs = obs.laplacian_torch(f, L=1.0)
    assert torch.allclose(lhs, rhs, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("L", [18.0, 57.0, 207.7])
def test_dispersion_is_invariant_under_the_rescaling(L):
    """sigma from (J, D_hat) at k_hat equals sigma from (J, D_hat*L**2) at k_hat/L.

    The direction matters and is easy to get backwards: x_hat = x/L divides every second
    derivative by L**2, so the DIMENSIONLESS diffusivity is the SMALL one,
    D_hat = D_phys / L**2, and converting back MULTIPLIES by L**2.

    This is why the non-dimensional model is the same model: the Turing spectrum, and hence
    the Turing verdict, is untouched by the change of units.
    """
    hat = RNGRN(N=3, seed=3)
    phys = RNGRN(N=3, seed=3)
    with torch.no_grad():
        phys.theta_D.copy_(hat.theta_D + 2.0 * math.log(L))   # D = exp(theta) => D_hat*L**2
    assert torch.allclose(phys.D, hat.D * (L ** 2), rtol=1e-12, atol=0.0)

    xstar = torch.ones(3)
    khat = torch.linspace(1.0, 80.0, 200)
    sig_hat = hat.dispersion(xstar, khat)
    sig_phys = phys.dispersion(xstar, khat / L)
    assert torch.allclose(sig_hat, sig_phys, rtol=1e-9, atol=1e-12)


def test_stationarity_residual_is_invariant_under_the_rescaling():
    """The residual term is the one place L enters the objective. It must be the SAME
    number for (D_hat, unit box) and (D_hat*L**2, physical box) — because
    D_phys * lap_L == (D_hat * L**2) * (lap_1 / L**2) == D_hat * lap_1."""
    from rngrn.losses.terms import stationarity_residual
    L = 57.0
    fields = torch.tensor(_frame())
    hat = RNGRN(N=3, seed=5)
    phys = RNGRN(N=3, seed=5)
    with torch.no_grad():
        phys.theta_D.copy_(hat.theta_D + 2.0 * math.log(L))
    r_hat, _ = stationarity_residual(hat, fields, 1.0, [0, 1, 2])
    r_phys, _ = stationarity_residual(phys, fields, L, [0, 1, 2])
    assert float(r_hat.detach()) == pytest.approx(float(r_phys.detach()), rel=1e-10)


# --------------------------------------------------------------------------------------
# 2. the DIMENSIONAL path is byte-for-byte what it was
# --------------------------------------------------------------------------------------
def _reference_dimensional_fit(ri, steps, seed=0, form="competitive"):
    """Hand-rolled copy of the pre-unit-12 recover() inner loop, dimensional path.

    Deliberately duplicated rather than imported: the point is to have a reference that
    CANNOT be changed by an edit to recover.py. Keep it in sync only when the dimensional
    algorithm is intentionally changed — and then re-read every recorded number.
    """
    from rngrn.losses import total as LT
    from rngrn.losses.weighting import FixedWeighting
    from rngrn.recover import _kgrid_for

    frame = torch.tensor(np.asarray(ri.frame, dtype=float))
    strategy = FixedWeighting(dict(kstar=1.0, turing=1.0, resid=0.3,
                                   anticollapse=0.5, morphology=0.0))
    kstar_obs = obs.kstar_of(frame[0].numpy(), L=ri.L)
    kgrid = _kgrid_for(kstar_obs)
    model = RNGRN(N=ri.N, form=form, seed=seed)
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=0.05)
    for step in range(steps):
        opt.zero_grad()
        loss, _ = LT.total_loss(model, frame, ri.L, list(ri.observed_idx), kgrid, kstar_obs,
                                strategy, step=step, latent_fields=None,
                                tau=0.12, jac_floor=1.0, strict=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
    with torch.no_grad():
        loss, parts = LT.total_loss(model, frame, ri.L, list(ri.observed_idx), kgrid,
                                    kstar_obs, strategy, step=steps, latent_fields=None,
                                    tau=0.12, jac_floor=1.0)
    return float(loss), model.D.detach().numpy(), parts["kstar_model"], kstar_obs


def test_dimensional_path_is_bit_identical_to_the_pre_change_reference():
    """THE REGRESSION GUARD. If this drifts, every kstar_rel_err in the run index changes
    meaning and the whole recorded history stops being comparable."""
    from rngrn.recover import recover
    ri = _RI(_frame(), L=57.0)
    steps = 5
    ref_loss, ref_D, ref_kstar, ref_kobs = _reference_dimensional_fit(ri, steps)
    got = recover(ri, n_restarts=1, adam_steps=steps, lbfgs_steps=0, seed=0, nondim=False)

    assert got.loss == ref_loss, "dimensional total loss changed"
    assert np.array_equal(got.params["D"], ref_D), "dimensional D changed"
    assert got.kstar_model == ref_kstar, "dimensional k*_model changed"
    assert got.kstar_obs == ref_kobs, "dimensional k*_obs changed"
    # and the new bookkeeping must be inert on this path
    assert got.nondim is False
    assert np.array_equal(got.params["D"], got.params["D_model"])
    assert np.array_equal(got.D_phys, got.params["D_model"])


def test_dimensional_path_is_the_default():
    """nondim must default to False so no existing caller silently switches methods."""
    import inspect
    from rngrn.recover import recover
    from rngrn.config import ModelConfig
    assert inspect.signature(recover).parameters["nondim"].default is False
    assert ModelConfig().nondim is False


# --------------------------------------------------------------------------------------
# 3. the NON-DIMENSIONAL path is L-invariant, and physical D scales as L**2
# --------------------------------------------------------------------------------------
def test_nondim_recovery_is_invariant_across_three_domain_sizes():
    """One known frame, three domain sizes: the recovered NETWORK is identical.

    This is invariance BY CONSTRUCTION, and the test exists to prove the construction was
    implemented, not to discover a fact: with nondim=True the objective is written on the
    unit box, so L never enters the optimisation. The empirically interesting question —
    whether recovery from three genuinely DIFFERENT images of the same system at different
    L agrees — needs the three_gene_multiL dataset and is not answered here.
    """
    from rngrn.recover import recover
    frame = _frame()
    Ls = [18.0, 57.0, 207.7]
    results = [recover(_RI(frame, L), n_restarts=1, adam_steps=5, lbfgs_steps=0,
                       seed=0, nondim=True) for L in Ls]

    base = results[0]
    for L, res in zip(Ls, results):
        # the learned, dimensionless object is literally the same numbers
        assert np.array_equal(res.params["D_model"], base.params["D_model"])
        assert np.array_equal(res.topology["sign"], base.topology["sign"])
        assert res.loss == base.loss
        assert res.q_model == pytest.approx(base.q_model, rel=1e-12)
        assert res.kstar_model_nondim == pytest.approx(base.kstar_model_nondim, rel=1e-12)
        # ...and the PHYSICAL readouts scale exactly as the change of variables demands
        assert res.D_phys == pytest.approx(base.params["D_model"] * L ** 2, rel=1e-12)
        assert res.kstar_model == pytest.approx(base.kstar_model_nondim / L, rel=1e-12)
        assert res.L == L and res.nondim is True


def test_nondim_and_dimensional_paths_agree_when_L_is_one():
    """L = 1 is the fixed point of the change of variables, so the two paths must coincide
    there. This pins the conversion arithmetic itself (a wrong power of L would show up)."""
    from rngrn.recover import recover
    ri = _RI(_frame(), L=1.0)
    a = recover(ri, n_restarts=1, adam_steps=4, lbfgs_steps=0, seed=0, nondim=False)
    b = recover(ri, n_restarts=1, adam_steps=4, lbfgs_steps=0, seed=0, nondim=True)
    assert a.loss == b.loss
    assert np.array_equal(a.params["D"], b.params["D"])
    assert a.kstar_model == pytest.approx(b.kstar_model, rel=1e-12)


def test_nondim_rejects_a_nonsense_domain_size():
    """Fail loud: L <= 0 would divide by zero in the conversion back to physical units."""
    from rngrn.recover import recover
    for bad in (0.0, -3.0, float("nan")):
        with pytest.raises(ValueError, match="positive finite domain size"):
            recover(_RI(_frame(), bad), n_restarts=1, adam_steps=1, lbfgs_steps=0,
                    nondim=True)


# --------------------------------------------------------------------------------------
# 4. the per-run metrics and the cross-L aggregator
# --------------------------------------------------------------------------------------
def test_sign_string_uses_the_same_relative_zero_rule_as_validate():
    J = np.array([[1.0, -2.0, 0.0], [3.0, 0.0, -4.0], [0.0, 5.0, 6.0]])
    assert LGEN.sign_string(J) == "+-0" "+0-" "0++"
    # an entry far below the relative tolerance reads as no edge
    J2 = np.array([[1.0, 1e-12], [0.0, -1.0]])
    assert LGEN.sign_string(J2) == "+00-"


def test_sign_string_raises_on_a_non_square_jacobian():
    with pytest.raises(ValueError, match="square"):
        LGEN.sign_string(np.ones((2, 3)))


def test_d_ratio_is_invariant_under_a_uniform_rescale():
    D = np.array([1.0, 40.0, 20.0])
    assert LGEN.d_ratio(D) == pytest.approx(40.0)
    assert LGEN.d_ratio(D * 57.0 ** 2) == pytest.approx(LGEN.d_ratio(D))


def test_d_ratio_raises_on_a_non_positive_diffusivity():
    with pytest.raises(ValueError, match="positive"):
        LGEN.d_ratio(np.array([1.0, 0.0]))


def test_modal_sign_agreement_endpoints():
    assert LGEN.modal_sign_agreement(["++--", "++--", "++--"]) == pytest.approx(1.0)
    # one of three runs flips one of four entries -> 3 entries at 1.0, one at 2/3
    got = LGEN.modal_sign_agreement(["++--", "++--", "+---"])
    assert got == pytest.approx((1.0 + 2.0 / 3.0 + 1.0 + 1.0) / 4.0)


def test_modal_sign_agreement_refuses_to_pool_different_model_sizes():
    with pytest.raises(ValueError, match="differing lengths"):
        LGEN.modal_sign_agreement(["++--", "+++++++++"])


def _row(system_id, L, sign, d_ratio, kstar, nondim=True, form="competitive", n_model=3):
    return dict(lgen_system_id=system_id, lgen_L=L, lgen_J_sign=sign,
                lgen_D_ratio=d_ratio, kstar_model=kstar, lgen_nondim=nondim,
                form=form, n_model=n_model)


def test_cross_L_consistency_is_perfect_when_every_L_recovers_the_same_network():
    rows = [_row("sysA", L, "+-+-", 40.0, 0.6) for L in (18.0, 57.0, 120.0)]
    tbl = LGEN.lgen_consistency(rows)
    assert len(tbl) == 1
    r = tbl[0]
    assert r["n_L"] == 3 and r["n_runs"] == 3
    assert r["sign_agree_cross_L"] == pytest.approx(1.0)
    assert r["D_ratio_log10_std_cross_L"] == pytest.approx(0.0)
    assert r["kstar_log10_std_cross_L"] == pytest.approx(0.0)
    # only one run per L, so the within-L control is undefined — and says so
    assert np.isnan(r["sign_agree_within_L"])
    assert np.isnan(r["sign_agree_gap"])


def test_cross_L_disagreement_is_reported_against_its_within_L_control():
    """Two seeds at each of two L. Seeds already disagree on one entry at a fixed L, so the
    cross-L number must be read against that, not against 1.0."""
    rows = [
        _row("sysA", 18.0, "+-+-", 40.0, 0.6), _row("sysA", 18.0, "+-++", 41.0, 0.61),
        _row("sysA", 120.0, "--+-", 10.0, 0.2), _row("sysA", 120.0, "--++", 11.0, 0.21),
    ]
    r = LGEN.lgen_consistency(rows)[0]
    assert r["n_L"] == 2 and r["n_runs"] == 4
    # within one L: 3 of 4 entries unanimous, the 4th split 1/2 -> (1+1+1+0.5)/4
    assert r["sign_agree_within_L"] == pytest.approx(0.875)
    # across all four: entry0 unanimous, entry1 split 2/2, entry2 unanimous, entry3 2/2
    assert r["sign_agree_cross_L"] == pytest.approx((1.0 + 0.5 + 1.0 + 0.5) / 4.0)
    assert r["sign_agree_gap"] == pytest.approx(0.875 - 0.75)
    assert r["D_ratio_log10_std_within_L"] > 0.0
    assert r["kstar_log10_std_cross_L"] > r["kstar_log10_std_within_L"]


def test_single_L_groups_are_excluded_not_scored():
    """A system seen at ONE L cannot demonstrate cross-L consistency and must not be able
    to contribute a perfect-looking row."""
    rows = [_row("sysA", 57.0, "+-+-", 40.0, 0.6) for _ in range(4)]
    assert LGEN.lgen_consistency(rows) == []


def test_rows_without_a_system_id_are_excluded():
    rows = [_row(None, L, "+-+-", 40.0, 0.6) for L in (18.0, 120.0)]
    assert LGEN.lgen_consistency(rows) == []


def test_the_two_arms_are_never_pooled():
    """A dimensional run and a non-dimensional run of the same system are different
    methods; averaging them would hide exactly the effect the arm exists to measure."""
    rows = ([_row("sysA", L, "+-+-", 40.0, 0.6, nondim=True) for L in (18.0, 120.0)]
            + [_row("sysA", L, "----", 2.0, 0.9, nondim=False) for L in (18.0, 120.0)])
    tbl = LGEN.lgen_consistency(rows)
    assert len(tbl) == 2
    assert {r["nondim"] for r in tbl} == {True, False}


def test_markdown_says_so_when_there_is_nothing_to_report():
    assert "no cross-L groups" in LGEN.lgen_markdown([])


# --------------------------------------------------------------------------------------
# wiring: gate -> answer key -> score_recovery -> run index
# --------------------------------------------------------------------------------------
def test_gate_carries_system_id_to_the_answer_key_only(tmp_path):
    """The cross-L grouping label is truth-side metadata: scoring gets it, recovery does not."""
    import h5py
    from rngrn.data import gate, registry as reg

    droot = tmp_path / "datasets"
    (droot / "ml").mkdir(parents=True)
    rng = np.random.default_rng(0)
    with h5py.File(droot / "ml" / "payload.h5", "w") as f:
        g = f.create_group("sample_0000")
        g.create_dataset("final_frame", data=rng.standard_normal((3, 16, 16)))
        g.create_dataset("jacobian", data=rng.standard_normal((3, 3)))
        g.attrs["L"] = 57.0
        g.attrs["k_star"] = 6.0 * TWO_PI / 57.0
        g.attrs["system_id"] = "sys_007"
    reg.scan(str(droot), backend="jsonl")

    ri, ak = gate.from_registry(str(droot), "ml", "sample_0000", N=3, observed_idx=[0, 1, 2])
    assert ak.system_id == "sys_007"
    assert not hasattr(ri, "system_id"), "FIREWALL: the grouping label reached recovery"


def test_absent_system_id_is_none_not_an_error(tmp_path):
    """Every dataset registered before three_gene_multiL lacks the attribute."""
    import h5py
    from rngrn.data import gate, registry as reg

    droot = tmp_path / "datasets"
    (droot / "old").mkdir(parents=True)
    rng = np.random.default_rng(0)
    with h5py.File(droot / "old" / "payload.h5", "w") as f:
        g = f.create_group("sample_0000")
        g.create_dataset("final_frame", data=rng.standard_normal((3, 16, 16)))
        g.attrs["L"] = 57.0
        g.attrs["k_star"] = 6.0 * TWO_PI / 57.0
    reg.scan(str(droot), backend="jsonl")
    _, ak = gate.from_registry(str(droot), "old", "sample_0000", N=3, observed_idx=[0, 1, 2])
    assert ak.system_id is None


def test_score_recovery_records_the_per_run_lgen_columns():
    from rngrn.validate import score_recovery
    from test_experiment_arms import _Key, _Result

    res = _Result(N=3)
    res.L = 57.0
    res.nondim = True
    res.q_model = 6.0
    res.D_phys = np.array([1.0, 40.0, 20.0])
    key = _Key(np.ones((3, 3)), n_true=3)
    key.system_id = "sys_007"

    out = score_recovery(res, key, observed_idx=[0, 1, 2])
    assert out["lgen_system_id"] == "sys_007"
    assert out["lgen_L"] == pytest.approx(57.0)
    assert out["lgen_nondim"] is True
    assert out["lgen_D_ratio"] == pytest.approx(40.0)
    assert len(out["lgen_J_sign"]) == 9
    assert out["lgen_q_model"] == pytest.approx(6.0)


def test_lgen_columns_survive_the_no_true_J_early_return():
    """score_recovery returns early when the dataset carries no true J. The L-generalisation
    pieces do not depend on it and must still be there."""
    from rngrn.validate import score_recovery
    from test_experiment_arms import _Key, _Result

    out = score_recovery(_Result(N=3), _Key(None, n_true=None), observed_idx=[0, 1, 2])
    assert out["scoring_mode"] == "no_true_J"
    assert "lgen_J_sign" in out and "lgen_D_ratio" in out


def test_stand_in_result_without_D_phys_still_scores_on_the_model_D():
    """Duck-typed results (no D_phys) must fall back to model.D, not crash or NaN."""
    from rngrn.validate import score_recovery
    from test_experiment_arms import _Key, _Result

    res = _Result(N=3)
    out = score_recovery(res, _Key(np.ones((3, 3)), n_true=3), observed_idx=[0, 1, 2])
    assert out["lgen_D_ratio"] == pytest.approx(
        LGEN.d_ratio(res.model.D.detach().numpy()))
    assert np.isnan(out["lgen_L"])
    assert out["lgen_nondim"] is False


# --------------------------------------------------------------------------------------
# real multi-L data (unit 11) — consumed automatically, skipped until it exists
# --------------------------------------------------------------------------------------
def test_multiL_dataset_is_groupable():
    """When three_gene_multiL lands, assert it carries what the aggregator needs.

    This is a DATA CONTRACT check, not a recovery result: it asserts only that samples
    carry a `system_id` and that at least one system appears at more than one L. It does
    not run recovery and proves nothing about L-generalisation.
    """
    import os
    import pathlib
    import h5py
    from collections import defaultdict

    root = pathlib.Path(__file__).resolve().parents[1] / "data" / "datasets"
    path = root / "three_gene_multiL" / "payload.h5"
    if not os.path.exists(path):
        pytest.skip("three_gene_multiL not present (unit 11 dataset; datasets are gitignored)")

    by_system = defaultdict(set)
    with h5py.File(path, "r") as f:
        for k in f:
            a = f[k].attrs
            assert "system_id" in a, (
                f"{k}: three_gene_multiL carries no `system_id` attribute, so scoring.lgen "
                f"cannot group samples of the same system across L. Present: "
                f"{sorted(map(str, a.keys()))}")
            by_system[str(a["system_id"])].add(round(float(a["L"]), 6))
    assert by_system, "three_gene_multiL is empty"
    multi = {s: Ls for s, Ls in by_system.items() if len(Ls) > 1}
    assert multi, f"no system appears at more than one L: {dict(list(by_system.items())[:3])}"
