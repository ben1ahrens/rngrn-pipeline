"""test_permutation_scoring.py — fast CPU tests for rngrn.scoring.permutation.

These test the SCORER, not recovery. Every J here is a synthetic matrix with a known
permutation applied by hand, so a failure means the metric is wrong, never that the
science is wrong. No torch, no fitting; runs in milliseconds.
"""
import itertools

import numpy as np
import pytest

from rngrn.scoring import permutation as P


# --------------------------------------------------------------------------------------
# fixtures: hand-built matrices with unambiguous sign structure
# --------------------------------------------------------------------------------------
def _J4():
    """4x4 with a distinct sign pattern in every row, so relabelling is detectable."""
    return np.array([
        [-1.0, -2.0, 0.5, 1.5],
        [3.0, -1.5, -0.5, 2.0],
        [-2.5, 1.0, -3.0, -1.0],
        [0.5, -1.0, 2.5, -2.0],
    ])


def _J3():
    return np.array([
        [-1.0, -2.0, 0.5],
        [3.0, -1.5, -0.5],
        [-2.5, 1.0, -3.0],
    ])


def _permute(J, perm):
    return J[np.ix_(perm, perm)]


# --------------------------------------------------------------------------------------
# 1. alignment recovers a known relabelling of the UNOBSERVED indices
# --------------------------------------------------------------------------------------
def test_align_recovers_known_latent_swap_and_beats_identity():
    """N=4, m=2 observed -> 2 latent indices -> 2 admissible relabellings.

    Build J_rec by swapping truth's two latent indices. The scorer must find exactly
    that swap, score 1.0 under it, and score strictly worse under the identity.
    """
    J_true = _J4()
    observed_idx = (0, 1)
    swap = (0, 1, 3, 2)                      # an involution: its own inverse
    J_rec = _permute(J_true, swap)

    res = P.align_permutation(J_rec, J_true, observed_idx)
    assert res.n_candidates == 2, res.n_candidates
    assert res.perm == swap
    assert res.score == pytest.approx(1.0)
    assert res.fro_rel_err == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(res.permuted_J, J_true)

    m = P.permuted_sign_match(J_rec, J_true, observed_idx)
    assert m["best_perm"] == swap
    assert m["sign_match_frac_aligned"] == pytest.approx(1.0)
    assert m["sign_match_frac_identity"] < 1.0
    assert m["sign_match_delta"] > 0.0
    assert m["fro_rel_err_aligned"] == pytest.approx(0.0, abs=1e-12)
    assert m["fro_rel_err_identity"] > 0.0
    assert m["n_permutations_searched"] == 2


def test_observed_channels_are_held_fixed():
    """Even when swapping two OBSERVED indices would score better, the pinned search
    must not do it — the gate pins observed model index o to true channel o."""
    J_true = _J4()
    observed_idx = (0, 1)
    obs_swap = (1, 0, 2, 3)
    J_rec = _permute(J_true, obs_swap)

    res = P.align_permutation(J_rec, J_true, observed_idx)
    assert res.perm[0] == 0 and res.perm[1] == 1, res.perm
    assert res.score < 1.0

    # the free_observed mode asks the weaker question and DOES find it
    free = P.align_permutation(J_rec, J_true, observed_idx, free_observed=True)
    assert free.n_candidates == 24
    assert free.score == pytest.approx(1.0)
    assert np.allclose(_permute(J_rec, free.perm), J_true)


def test_identity_scores_perfectly_when_no_relabelling_applied():
    J_true = _J4()
    m = P.permuted_sign_match(J_true.copy(), J_true, (0, 1))
    assert m["sign_match_frac_identity"] == pytest.approx(1.0)
    assert m["sign_match_frac_aligned"] == pytest.approx(1.0)
    assert m["sign_match_delta"] == pytest.approx(0.0)
    assert m["best_perm"] == (0, 1, 2, 3)


def test_experiment_a_shape_search_is_degenerate():
    """EXPERIMENT A (N=3, m=2): exactly ONE admissible relabelling, so alignment is a
    mathematical no-op and the delta is necessarily zero. This is the honest limitation
    documented in the module docstring, asserted so nobody later reports the aligned
    number as evidence that alignment rescued the score."""
    J_true = _J3()
    J_rec = np.array([
        [-1.0, 2.0, 0.5],
        [3.0, -1.5, 0.5],
        [2.5, 1.0, -3.0],
    ])
    m = P.permuted_sign_match(J_rec, J_true, (0, 1))
    assert m["n_permutations_searched"] == 1
    assert m["best_perm"] == (0, 1, 2)
    assert m["sign_match_delta"] == pytest.approx(0.0)
    assert m["sign_match_frac_aligned"] == m["sign_match_frac_identity"]


def test_align_score_is_a_true_global_maximum():
    """Brute-force the objective independently and confirm the returned score is the max
    over the admissible set."""
    rng = np.random.default_rng(0)
    J_true = rng.normal(size=(5, 5))
    J_rec = rng.normal(size=(5, 5))
    observed_idx = (1, 3)
    latent = P.unobserved_indices(5, observed_idx)
    s_true, s_rec = P.sign_structure(J_true), P.sign_structure(J_rec)

    scores = []
    for assign in itertools.permutations(latent):
        perm = list(range(5))
        for pos, src in zip(latent, assign):
            perm[pos] = src
        scores.append(float(np.mean(s_rec[np.ix_(perm, perm)] == s_true)))

    res = P.align_permutation(J_rec, J_true, observed_idx)
    assert res.n_candidates == len(scores) == 6
    assert res.score == pytest.approx(max(scores))


def test_unobserved_indices_matches_recovery_layout():
    assert P.unobserved_indices(3, (0, 1)) == (2,)
    assert P.unobserved_indices(4, (0, 2)) == (1, 3)
    assert P.unobserved_indices(2, (0, 1)) == ()


# --------------------------------------------------------------------------------------
# 2. observed sub-block works across a species-count mismatch (EXPERIMENT B)
# --------------------------------------------------------------------------------------
def test_observed_subblock_3x3_recovered_vs_2x2_true():
    """EXPERIMENT B: model N=3, truth 2x2. The 2x2 observed block is well-defined and
    must score, where whole-matrix comparison previously produced a silent NaN."""
    J_true = np.array([[-1.0, -2.0], [3.0, -1.5]])
    J_rec = np.array([
        [-0.5, -4.0, 1.0],
        [6.0, -0.75, -1.0],
        [0.1, 0.2, -2.0],
    ])
    out = P.observed_subblock_score(J_rec, J_true, (0, 1))
    assert out["n_species_model"] == 3 and out["n_species_true"] == 2
    assert out["n_observed"] == 2
    assert out["sign_match_frac_observed"] == pytest.approx(1.0)
    assert np.isfinite(out["fro_rel_err_observed"])
    assert out["fro_rel_err_observed"] > 0.0


def test_observed_subblock_exact_match_is_zero_error():
    J_true = np.array([[-1.0, -2.0], [3.0, -1.5]])
    J_rec = np.zeros((3, 3))
    J_rec[:2, :2] = J_true
    J_rec[2, 2] = -1.0
    out = P.observed_subblock_score(J_rec, J_true, (0, 1))
    assert out["sign_match_frac_observed"] == pytest.approx(1.0)
    assert out["fro_rel_err_observed"] == pytest.approx(0.0, abs=1e-12)


def test_observed_subblock_detects_sign_error():
    J_true = np.array([[-1.0, -2.0], [3.0, -1.5]])
    J_rec = np.array([[-1.0, +2.0], [3.0, -1.5]])   # one flipped entry of four
    out = P.observed_subblock_score(J_rec, J_true, (0, 1))
    assert out["sign_match_frac_observed"] == pytest.approx(0.75)


def test_observed_subblock_experiment_a_excludes_hidden_row():
    """Truth 3x3, model 3x3, hidden index 2 wrong: the observed block must ignore it."""
    J_true = _J3()
    J_rec = J_true.copy()
    J_rec[2, :] *= -5.0
    J_rec[:, 2] *= -5.0
    out = P.observed_subblock_score(J_rec, J_true, (0, 1))
    assert out["sign_match_frac_observed"] == pytest.approx(1.0)
    assert out["fro_rel_err_observed"] == pytest.approx(0.0, abs=1e-12)


def test_observed_index_invalid_for_truth_raises():
    """Fail loud rather than guess when observed_idx is not a channel of the truth."""
    J_true = np.array([[-1.0, -2.0], [3.0, -1.5]])
    J_rec = np.eye(3) * -1.0
    with pytest.raises(IndexError):
        P.observed_subblock_score(J_rec, J_true, (0, 2))


def test_align_permutation_refuses_shape_mismatch():
    J_true = np.array([[-1.0, -2.0], [3.0, -1.5]])
    J_rec = np.eye(3) * -1.0
    with pytest.raises(ValueError, match="observed_subblock_score"):
        P.align_permutation(J_rec, J_true, (0, 1))


def test_non_square_and_non_finite_raise():
    with pytest.raises(ValueError):
        P.align_permutation(np.zeros((2, 3)), np.eye(2), (0,))
    with pytest.raises(ValueError):
        P.observed_subblock_score(np.array([[np.nan, 0.0], [0.0, -1.0]]), np.eye(2), (0,))


# --------------------------------------------------------------------------------------
# 3. latent-field diagnostic
# --------------------------------------------------------------------------------------
def _pattern(H=16, W=16, phase=0.0):
    y, x = np.mgrid[0:H, 0:W]
    return np.sin(2 * np.pi * x / W + phase) * np.cos(2 * np.pi * y / H)


def test_latent_field_quality_perfect_and_offset_invariance():
    true_hidden = _pattern() + 2.0
    latent = 3.0 * (true_hidden - 2.0) - 7.0          # same pattern, rescaled + offset
    out = P.latent_field_quality(latent, true_hidden)
    assert out["pearson_r"] == pytest.approx(1.0)
    assert out["abs_pearson_r"] == pytest.approx(1.0)
    # mean-removed correlation is blind to offset and positive rescaling; the uncentered
    # cosine feels the offset, which is why both are reported
    assert out["cosine_uncentered"] < 0.0
    assert out["n_latent"] == 1 and out["n_hidden"] == 1
    assert out["best_pair"] == (0, 0)


def test_latent_field_quality_phase_inversion_is_negative_r():
    true_hidden = _pattern()
    out = P.latent_field_quality(-true_hidden, true_hidden)
    assert out["pearson_r"] == pytest.approx(-1.0)
    assert out["abs_pearson_r"] == pytest.approx(1.0)


def test_latent_field_quality_uncorrelated_is_small():
    rng = np.random.default_rng(3)
    out = P.latent_field_quality(rng.normal(size=(1, 16, 16)), _pattern())
    assert abs(out["pearson_r"]) < 0.4


def test_latent_field_quality_accepts_stack_and_reports_matrix():
    hidden = _pattern()
    noise = np.random.default_rng(1).normal(size=(16, 16))
    latent = np.stack([noise, hidden * 2.0])
    out = P.latent_field_quality(latent, hidden)
    assert out["n_latent"] == 2 and out["n_hidden"] == 1
    assert np.array(out["correlation_matrix"]).shape == (2, 1)
    assert out["best_pair"] == (1, 0)
    assert out["best_pearson_r"] == pytest.approx(1.0)
    assert "pearson_r" not in out          # scalar shortcut only for the 1x1 case
    assert len(out["latent_std"]) == 2


def test_latent_field_quality_shape_mismatch_raises_not_interpolates():
    with pytest.raises(ValueError, match="grid mismatch"):
        P.latent_field_quality(_pattern(16, 16), _pattern(8, 8))


def test_latent_field_quality_none_raises():
    with pytest.raises(ValueError, match="latent_fields is None"):
        P.latent_field_quality(None, _pattern())


def test_latent_field_quality_constant_field_raises():
    """A collapsed latent field is a real recovery outcome; it must be reported as
    collapse, not silently scored as r=0."""
    with pytest.raises(ValueError, match="constant"):
        P.latent_field_quality(np.full((16, 16), 0.3), _pattern())


def test_latent_field_quality_rejects_bad_rank():
    with pytest.raises(ValueError):
        P.latent_field_quality(np.zeros((2, 2, 16, 16)), _pattern())


# --------------------------------------------------------------------------------------
# 4. firewall: the scoring module must not pull in recovery-side code
# --------------------------------------------------------------------------------------
def test_scoring_module_does_not_import_recovery_side():
    """Static import audit, same spirit as tests/test_firewall.py but in the other
    direction: scoring may read the answer key, so it must stay off the recovery path."""
    import ast
    import pathlib

    src = pathlib.Path(P.__file__)
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
                      ".model", ".recover", ".losses", ".observables", "torch"):
        assert forbidden not in joined, (
            f"scoring/permutation.py imports '{forbidden}'; scoring-side code must not "
            f"be reachable from recovery.")


def test_no_recovery_side_module_imports_scoring():
    """The complementary direction: no recovery-side module may import rngrn.scoring.

    Mirrors tests/test_firewall.py's RECOVERY_SIDE list rather than editing it, so the
    two audits stay independent. latent_field_quality consumes the true hidden channel;
    if a recovery-side module could reach this package the firewall would be void.
    """
    import ast
    import pathlib

    src_root = pathlib.Path(P.__file__).resolve().parents[1]
    recovery_side = [
        "model.py", "observables.py", "recover.py",
        "losses/terms.py", "losses/total.py", "losses/weighting.py",
        "eval/rollout.py", "eval/numerics.py", "eval/dynamical.py",
        "eval/analysis.py", "eval/topology.py",
    ]
    for relpath in recovery_side:
        path = src_root / relpath
        assert path.exists(), f"missing recovery-side module {relpath}"
        text = path.read_text()
        tree = ast.parse(text)
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names += [f"{mod}.{a.name}" for a in node.names] + [mod]
        joined = " ".join(names)
        assert "scoring" not in joined, (
            f"FIREWALL BREACH: {relpath} imports the scoring package, which reads "
            f"answer-key quantities (true J, true hidden channel).")
