"""test_reproducibility_scoring.py — fast CPU tests for rngrn.scoring.reproducibility.

These test the SCORER, not recovery. Every Jacobian here is a synthetic matrix with a
known sign pattern (or known permutation/sign-flip applied by hand), so a failure means
the metric is wrong, never that the science is wrong. No torch, no fitting; runs in
milliseconds. Same spirit as tests/test_permutation_scoring.py.
"""
import json

import numpy as np
import pytest

from rngrn.scoring import reproducibility as R


def _J3(scale=1.0):
    return scale * np.array([
        [-1.0, -2.0, 0.5],
        [3.0, -1.5, -0.5],
        [-2.5, 1.0, -3.0],
    ])


# --------------------------------------------------------------------------------------
# 1. identical inputs give consistency 1.0
# --------------------------------------------------------------------------------------
def test_identical_matrices_give_full_consistency():
    Js = [_J3(), _J3(), _J3()]
    kstars = [1.0, 1.0, 1.0]
    dratios = [10.0, 10.0, 10.0]
    rep = R.reproducibility_report(Js, kstars, dratios)
    assert rep["mean_agreement"] == pytest.approx(1.0)
    assert rep["modal_fraction"] == pytest.approx(1.0)
    assert rep["topology_consistency"] == pytest.approx(1.0)
    assert rep["kstar_spread"] == pytest.approx(0.0)
    assert rep["Dratio_spread"] == pytest.approx(0.0)
    assert rep["K"] == 3
    assert np.array(rep["modal_structure"]).shape == (3, 3)


def test_identical_matrices_up_to_positive_rescaling_still_agree():
    """The overall scale of J is not pinned by the objective (documented elsewhere in
    this codebase); two seeds that agree on SIGN but differ in magnitude must still
    score full agreement, since this module scores sign structure only."""
    Js = [_J3(), _J3(scale=5.0), _J3(scale=0.2)]
    rep = R.reproducibility_report(Js, [1.0, 1.1, 0.9], [10.0, 11.0, 9.0])
    assert rep["mean_agreement"] == pytest.approx(1.0)
    assert rep["topology_consistency"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# 2. sign-flipped inputs give the documented value
# --------------------------------------------------------------------------------------
def test_fully_sign_flipped_pair_scores_zero_agreement_on_nonzero_entries():
    """J and -J have every nonzero entry disagree entrywise (no entry of _J3 is zero),
    so mean sign agreement across the pair must be exactly 0.0 and neither structure is
    modal over the other -> modal_fraction is 1/K (each of the two distinct structures
    ties, broken deterministically)."""
    J = _J3()
    Js = [J, -J]
    rep = R.reproducibility_report(Js, [1.0, 1.0], [10.0, 10.0])
    assert rep["mean_agreement"] == pytest.approx(0.0)
    assert rep["agreement_matrix"][0][0] == pytest.approx(1.0)   # self-agreement
    assert rep["agreement_matrix"][0][1] == pytest.approx(0.0)
    assert rep["n_distinct_structures"] == 2
    assert rep["modal_fraction"] == pytest.approx(0.5)
    assert rep["topology_consistency"] == pytest.approx(0.5)


def test_one_flipped_entry_among_three_seeds_reduces_but_does_not_zero_consistency():
    J = _J3()
    J_odd = J.copy()
    J_odd[0, 1] *= -1.0   # flip exactly one of the 9 entries
    Js = [J, J.copy(), J_odd]
    rep = R.reproducibility_report(Js, [1.0, 1.0, 1.0], [10.0, 10.0, 10.0])
    # the two identical J's form the modal structure -> 2/3
    assert rep["modal_fraction"] == pytest.approx(2.0 / 3.0)
    assert rep["topology_consistency"] == pytest.approx(2.0 / 3.0)
    # pairwise: (0,1) agree fully, (0,2) and (1,2) disagree on 1/9 entries
    assert rep["mean_agreement"] == pytest.approx((1.0 + 8 / 9 + 8 / 9) / 3)


# --------------------------------------------------------------------------------------
# 3. near-zero entries behave as documented (relative-magnitude dead zone)
# --------------------------------------------------------------------------------------
def test_near_zero_entry_does_not_inflate_agreement():
    """An entry at ~1e-6 relative to the matrix scale must be treated as 'no edge' (0),
    not as a real sign, under the default UNCALIBRATED threshold (5% of max|J|)."""
    J1 = _J3()
    J1[0, 2] = 1e-6      # was 0.5 in _J3; now negligible relative to scale ~3
    J2 = _J3()
    J2[0, 2] = -1e-6     # opposite tiny sign in the other seed

    s1 = R.sign_structure(J1)
    s2 = R.sign_structure(J2)
    assert s1[0, 2] == 0.0
    assert s2[0, 2] == 0.0
    # with both collapsed to 0 they agree at this entry despite opposite raw signs
    rep = R.reproducibility_report([J1, J2], [1.0, 1.0], [10.0, 10.0])
    assert rep["mean_agreement"] == pytest.approx(1.0)


def test_sign_zero_rtol_is_a_real_parameter():
    """Shrinking the threshold to near-zero recovers the raw (near-machine-precision)
    sign comparison, so the 1e-6 entries above are no longer collapsed to 0."""
    J1 = _J3()
    J1[0, 2] = 1e-6
    J2 = _J3()
    J2[0, 2] = -1e-6
    s1 = R.sign_structure(J1, sign_zero_rtol=1e-12)
    s2 = R.sign_structure(J2, sign_zero_rtol=1e-12)
    assert s1[0, 2] == 1.0
    assert s2[0, 2] == -1.0


def test_sign_structure_idempotent_on_already_signed_input():
    """A matrix already valued in {-1, 0, 1} passes through unchanged — this is the
    property optim.benchmark relies on to re-hydrate a stored repro_sign_vector."""
    s = R.sign_structure(_J3())
    s2 = R.sign_structure(s)
    assert np.array_equal(s, s2)


# --------------------------------------------------------------------------------------
# 4. K=1 raises rather than silently returning 1.0
# --------------------------------------------------------------------------------------
def test_k_equals_one_raises():
    with pytest.raises(ValueError, match="K"):
        R.reproducibility_report([_J3()], [1.0], [10.0])
    with pytest.raises(ValueError):
        R.pairwise_sign_agreement([_J3()])
    with pytest.raises(ValueError):
        R.modal_sign_structure([_J3()])
    with pytest.raises(ValueError):
        R.edge_consensus([_J3()])


def test_coefficient_of_variation_needs_k_ge_2():
    with pytest.raises(ValueError):
        R.coefficient_of_variation([1.0], "kstar")
    assert R.coefficient_of_variation([1.0, 1.0], "kstar") == pytest.approx(0.0)


def test_coefficient_of_variation_nonpositive_mean_raises():
    with pytest.raises(ValueError):
        R.coefficient_of_variation([1.0, -1.0], "kstar")   # mean == 0


# --------------------------------------------------------------------------------------
# 5. shape / finiteness validation
# --------------------------------------------------------------------------------------
def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        R.reproducibility_report([_J3(), np.eye(2)], [1.0, 1.0], [1.0, 1.0])


def test_mismatched_list_lengths_raise():
    with pytest.raises(ValueError):
        R.reproducibility_report([_J3(), _J3()], [1.0], [1.0, 1.0])


def test_non_finite_raises():
    J = _J3()
    J[0, 0] = np.nan
    with pytest.raises(ValueError):
        R.reproducibility_report([J, _J3()], [1.0, 1.0], [1.0, 1.0])


# --------------------------------------------------------------------------------------
# 6. edge_consensus per-entry semantics
# --------------------------------------------------------------------------------------
def test_edge_consensus_matches_manual_majority():
    J = _J3()
    Js = [J, J.copy(), -J]     # 2 seeds agree, 1 disagrees, on every nonzero entry
    ec = R.edge_consensus(Js)
    consensus = np.array(ec["consensus_matrix"])
    majority = np.array(ec["majority_sign_matrix"])
    s = R.sign_structure(J)
    # every nonzero entry: majority sign is J's own sign, at 2/3 consensus
    nonzero = s != 0
    assert np.allclose(consensus[nonzero], 2.0 / 3.0)
    assert np.array_equal(majority[nonzero], s[nonzero])


# --------------------------------------------------------------------------------------
# 7. per_run_fields — the validate.py hook point
# --------------------------------------------------------------------------------------
def test_per_run_fields_round_trips_through_json():
    J = _J3()
    D = np.array([1.0, 10.0, 3.0])
    fields = R.per_run_fields(J, D, kstar_model=0.42)
    assert fields["repro_N"] == 3
    assert fields["repro_kstar"] == pytest.approx(0.42)
    assert fields["repro_D_ratio"] == pytest.approx(10.0)   # max/min
    flat = json.loads(fields["repro_sign_vector"])
    assert len(flat) == 9
    restored = np.array(flat, dtype=float).reshape(3, 3)
    assert np.array_equal(restored, R.sign_structure(J))


def test_per_run_fields_rejects_nonpositive_D():
    with pytest.raises(ValueError):
        R.per_run_fields(_J3(), np.array([1.0, -1.0, 2.0]), kstar_model=1.0)
    with pytest.raises(ValueError):
        R.per_run_fields(_J3(), np.array([1.0, 0.0, 2.0]), kstar_model=1.0)


# --------------------------------------------------------------------------------------
# 8. firewall: no ground-truth / recovery-side dependency at all
# --------------------------------------------------------------------------------------
def test_reproducibility_module_has_no_recovery_or_answer_key_dependency():
    """This module compares recovered J's to each other, never to a ground truth, so it
    should not import anything answer-key- or recovery-side. Same spirit as
    test_permutation_scoring.py's firewall check."""
    import ast
    import pathlib

    src = pathlib.Path(R.__file__)
    tree = ast.parse(src.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names += [f"{mod}.{a.name}" for a in node.names] + [mod]
    joined = " ".join(names)
    for forbidden in ("rngrn.model", "rngrn.recover", "rngrn.losses", "rngrn.observables",
                      ".model", ".recover", ".losses", ".observables", "torch",
                      "answer_key", "AnswerKey", "rd_models", "data.solver", "data.cache"):
        assert forbidden not in joined, (
            f"scoring/reproducibility.py imports '{forbidden}'; scoring-side code must "
            f"not be reachable from recovery, and this module needs no ground truth.")


def test_no_recovery_side_module_imports_reproducibility():
    import ast
    import pathlib

    src_root = pathlib.Path(R.__file__).resolve().parents[1]
    recovery_side = [
        "model.py", "observables.py", "recover.py",
        "losses/terms.py", "losses/total.py", "losses/weighting.py",
        "eval/rollout.py", "eval/numerics.py", "eval/dynamical.py",
        "eval/analysis.py", "eval/topology.py",
    ]
    for relpath in recovery_side:
        path = src_root / relpath
        assert path.exists(), f"missing recovery-side module {relpath}"
        tree = ast.parse(path.read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names += [f"{mod}.{a.name}" for a in node.names] + [mod]
        joined = " ".join(names)
        assert "scoring" not in joined, (
            f"FIREWALL BREACH: {relpath} imports the scoring package.")
