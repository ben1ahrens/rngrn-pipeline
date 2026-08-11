"""test_experiment_arms.py — scoring ROUTES correctly per experiment arm.

The unit behaviour of each scorer lives in test_permutation_scoring.py /
test_overparam_scoring.py. This file covers the integration those two deliberately did not:
that validate.score_recovery dispatches to the right mode, and — the original bug — that a
model/truth species-count mismatch produces EXPLICIT metrics instead of a silent NaN.
"""
import numpy as np
import pytest
import torch

from rngrn.model import RNGRN
from rngrn.validate import score_recovery


class _Key:
    def __init__(self, J, n_true, kstar=1.0):
        self.J = np.asarray(J, float)
        self.n_species_true = n_true
        self.kstar = kstar
        self.x_star = None
        self.sigma_max = None
        self.D = None
        self.coefficients = None


class _Result:
    """Minimal RecoveryResult stand-in: score_recovery needs model, xstar, kstar_model."""
    def __init__(self, N, seed=0):
        self.model = RNGRN(N=N, seed=seed)
        self.xstar = np.full(N, 1.0)
        self.kstar_model = 1.0
        self.latent_fields = None


def test_same_size_routes_to_permutation_alignment():
    res = _Result(N=3)
    key = _Key(np.ones((3, 3)), n_true=3)
    out = score_recovery(res, key, observed_idx=[0, 1])
    assert out["scoring_mode"] == "permutation_aligned"
    assert not np.isnan(out["sign_match_frac"])
    assert "n_permutations_searched" in out
    # Experiment A with 3 species and 2 observed has exactly ONE hidden index, so the
    # permutation search is degenerate by construction. Assert it, so a future reader is
    # never misled into thinking alignment did work here.
    assert out["n_permutations_searched"] == 1
    assert out["sign_match_frac_aligned"] == out["sign_match_frac_identity"]


def test_overparameterised_routes_to_spare_species_metrics():
    """THE original bug: 3x3 recovered vs 2x2 true used to give a silent NaN."""
    res = _Result(N=3)
    key = _Key(np.array([[1.0, -1.0], [1.0, -1.0]]), n_true=2)
    out = score_recovery(res, key, observed_idx=[0, 1])
    assert out["scoring_mode"] == "overparameterised"
    assert out["n_true"] == 2 and out["n_model"] == 3
    # explicit spare-species measurement, not a NaN
    assert "spare_species_inert" in out
    assert isinstance(out["spare_species_inert"], (bool, np.bool_))
    assert not np.isnan(out["extra_norm_frac"])
    # sign_match_frac is deliberately undefined here (no correct 3x3 exists)
    assert np.isnan(out["sign_match_frac"])


def test_observed_subblock_is_reported_in_every_arm():
    """The one cross-arm-comparable column must always be present."""
    for N, n_true, J in [(3, 3, np.ones((3, 3))),
                         (3, 2, np.ones((2, 2))),
                         (2, 2, np.ones((2, 2)))]:
        out = score_recovery(_Result(N=N), _Key(J, n_true=n_true), observed_idx=[0, 1])
        assert "subblock_sign_match" in out, (N, n_true)
        assert not np.isnan(out["subblock_sign_match"]), (N, n_true)


def test_no_true_J_is_explicit_not_silent():
    out = score_recovery(_Result(N=3), _Key(None, n_true=None), observed_idx=[0, 1])
    assert out["scoring_mode"] == "no_true_J"


def test_arm_classification_matches_shapes():
    """fit() derives `arm` from n_true vs N and the hidden set; check the same logic here."""
    cases = [(3, 3, [0, 1, 2], "fully_observed"),
             (3, 3, [0, 1], "hidden_channel"),
             (3, 2, [0, 1], "overparameterised"),
             (2, 3, [0, 1], "underparameterised")]
    for N, n_true, obs, expect in cases:
        hidden = [i for i in range(N) if i not in obs]
        if n_true is not None and N > n_true:
            arm = "overparameterised"
        elif n_true is not None and N < n_true:
            arm = "underparameterised"
        elif hidden:
            arm = "hidden_channel"
        else:
            arm = "fully_observed"
        assert arm == expect, (N, n_true, obs, arm)
