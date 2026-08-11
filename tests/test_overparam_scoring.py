"""test_overparam_scoring.py — unit tests for rngrn.scoring.overparam (EXPERIMENT B).

Synthetic Jacobians only: fast, CPU, no recovery run and no torch (the two metric
functions under test are pure numpy; only overparam_report needs torch, and it is
exercised here through a lightweight stub model rather than a real RNGRN fit).

What these tests establish: the metrics compute what their docstrings say on matrices
whose answer we know by construction, and they raise instead of returning a silent
NaN/zero on the degenerate cases. They establish NOTHING about whether the recovery
pipeline passes them on real data — the thresholds are uncalibrated.
"""
from __future__ import annotations

import numpy as np
import pytest

from rngrn.scoring.overparam import (
    DEFAULT_COUPLING_THRESHOLD,
    observed_block_agreement,
    overparam_report,
    spurious_species_metrics,
)

# The Experiment B geometry: truth is 2 species, both observed; the model has 3.
OBSERVED = (0, 1)
N_TRUE = 2
# A plausible 2x2 activator-inhibitor truth block (signs +,-,+,-; trace < 0).
J_TRUE_2x2 = np.array([[0.9, -1.6],
                       [1.1, -1.3]])
D_3 = np.array([0.05, 1.0, 0.4])


def _J3(extra_row, extra_col, diag_extra, obs_block=J_TRUE_2x2):
    """Assemble a 3x3 J with a prescribed observed block and third row/column.

    extra_row = J[2, :2] (edges INTO species 2), extra_col = J[:2, 2] (edges OUT of it).
    """
    J = np.zeros((3, 3))
    J[:2, :2] = obs_block
    J[2, :2] = extra_row
    J[:2, 2] = extra_col
    J[2, 2] = diag_extra
    return J


# ======================================================================================
# the two required cases
# ======================================================================================
def test_inert_third_species_passes():
    """Near-zero third row/column => spare_species_inert is True."""
    J = _J3(extra_row=[1e-4, -2e-4], extra_col=[3e-4, 1e-4], diag_extra=-0.8)
    m = spurious_species_metrics(J, D_3, OBSERVED, N_TRUE)

    assert m["overparameterised"] is True
    assert m["extra_idx"] == "(2,)"
    assert m["n_extra"] == 1
    assert m["spare_species_inert"] is True
    assert m["decoupled"] is True
    # couplings are the prescribed maxima, and both sit far below the realised cut
    assert m["max_abs_coupling_to_extra"] == pytest.approx(2e-4)
    assert m["max_abs_coupling_from_extra"] == pytest.approx(3e-4)
    assert m["coupling_threshold_used"] == pytest.approx(
        DEFAULT_COUPLING_THRESHOLD * np.abs(J_TRUE_2x2).max())
    # the retained self-decay term must NOT by itself break inertness
    assert m["max_abs_diag_extra"] == pytest.approx(0.8)
    # only the diagonal contributes appreciably to the extra species' norm share
    assert m["extra_norm_frac"] < 0.35


def test_single_strong_edge_fails_even_when_the_mean_is_small():
    """One strong spurious edge => not inert, and the MAX catches what a mean hides."""
    # 3x3 with N=3 model but a wider spare row so the mean is diluted: put one strong
    # edge out of species 2 into species 0, everything else negligible.
    J = _J3(extra_row=[1e-4, 1e-4], extra_col=[1.5, 1e-4], diag_extra=-0.5)
    m = spurious_species_metrics(J, D_3, OBSERVED, N_TRUE)

    assert m["spare_species_inert"] is False
    assert m["decoupled"] is False
    assert m["max_abs_coupling_from_extra"] == pytest.approx(1.5)
    # the max picks up the single strong edge; the mean over that column does not —
    # this is exactly why the metric is a max
    from_col_mean = float(np.abs(J[:2, 2]).mean())
    assert m["max_abs_coupling_from_extra"] > m["coupling_threshold_used"]
    assert from_col_mean == pytest.approx((1.5 + 1e-4) / 2)
    assert m["max_abs_coupling_from_extra"] > from_col_mean
    # and the direction that is genuinely quiet is still reported as quiet
    assert m["max_abs_coupling_to_extra"] == pytest.approx(1e-4)
    assert m["rel_coupling_from_extra"] == pytest.approx(1.5 / np.abs(J_TRUE_2x2).max())


def test_mean_dilution_is_defeated_in_a_wider_model():
    """With several spare-adjacent entries, a mean falls below the cut but the max does not."""
    # 4-species model, 3-species truth is NOT the supported geometry; instead widen the
    # observed set: 3 observed truth channels + 1 spare, one strong edge among many zeros.
    obs = (0, 1, 2)
    obs_block = np.array([[0.9, -1.6, 0.2],
                          [1.1, -1.3, 0.1],
                          [0.3, -0.2, -0.7]])
    J = np.zeros((4, 4))
    J[:3, :3] = obs_block
    J[3, :3] = [1e-5, 1e-5, 1e-5]
    J[:3, 3] = [1e-5, 1e-5, 1.2]      # one strong edge out of the spare species
    J[3, 3] = -0.6
    D4 = np.array([0.05, 1.0, 0.4, 0.2])

    m = spurious_species_metrics(J, D4, obs, n_true=3)
    mean_from = float(np.abs(J[:3, 3]).mean())          # ~0.4
    assert m["max_abs_coupling_from_extra"] == pytest.approx(1.2)
    assert mean_from < m["max_abs_coupling_from_extra"]
    assert m["spare_species_inert"] is False


# ======================================================================================
# threshold behaviour and provenance
# ======================================================================================
def test_threshold_is_tunable_and_flips_the_verdict():
    """A borderline coupling is a pass or a fail depending on the (explicit) cut."""
    # strongest observed entry is 1.6, so a coupling of 0.16 is exactly 10% of scale
    J = _J3(extra_row=[0.16, 0.0], extra_col=[0.0, 0.0], diag_extra=-0.5)
    loose = spurious_species_metrics(J, D_3, OBSERVED, N_TRUE, coupling_threshold=0.20)
    tight = spurious_species_metrics(J, D_3, OBSERVED, N_TRUE, coupling_threshold=0.05)
    assert loose["spare_species_inert"] is True
    assert tight["spare_species_inert"] is False
    assert tight["rel_coupling_to_extra"] == pytest.approx(0.1)
    # provenance: the argument and the realised cut are both recorded
    assert tight["coupling_threshold_arg"] == 0.05
    assert tight["coupling_threshold_mode"] == "relative"
    assert tight["coupling_threshold_used"] == pytest.approx(0.05 * 1.6)


def test_absolute_threshold_mode():
    J = _J3(extra_row=[0.3, 0.0], extra_col=[0.0, 0.0], diag_extra=-0.5)
    m = spurious_species_metrics(J, D_3, OBSERVED, N_TRUE,
                                 coupling_threshold=0.5, threshold_mode="absolute")
    assert m["coupling_threshold_used"] == pytest.approx(0.5)
    assert m["spare_species_inert"] is True
    assert np.isnan(m["rel_coupling_to_extra"])   # no scale is formed in absolute mode
    with pytest.raises(ValueError, match="threshold_mode"):
        spurious_species_metrics(J, D_3, OBSERVED, N_TRUE, threshold_mode="rel")


def test_norm_fraction_does_not_double_count_the_diagonal():
    """extra_norm_frac uses a union mask, so J[e,e] enters ||J_E||_F exactly once."""
    # everything zero except the extra diagonal => the extra block IS the whole matrix
    J = np.zeros((3, 3))
    J[:2, :2] = np.array([[1.0, 0.0], [0.0, 1.0]])
    J[2, 2] = 1.0
    m = spurious_species_metrics(J, D_3, OBSERVED, N_TRUE)
    # ||J||_F^2 = 3, extra mask captures only J[2,2] => frac = sqrt(1/3)
    assert m["extra_norm_frac"] == pytest.approx(np.sqrt(1.0 / 3.0))
    assert m["extra_energy_frac"] == pytest.approx(1.0 / 3.0)
    # a naive row-sum + column-sum would have counted J[2,2] twice => energy 2/3
    assert m["extra_energy_frac"] != pytest.approx(2.0 / 3.0)


# ======================================================================================
# undefined / refused cases — must not fabricate a pass
# ======================================================================================
def test_no_spare_species_returns_none_not_a_pass():
    """n_model == n_true: the criterion is undefined, reported as None."""
    m = spurious_species_metrics(J_TRUE_2x2, D_3[:2], OBSERVED, n_true=2)
    assert m["n_extra"] == 0
    assert m["overparameterised"] is False
    assert m["spare_species_inert"] is None
    assert m["decoupled"] is None
    assert np.isnan(m["extra_norm_frac"])


def test_ambiguous_spare_identity_raises():
    """n_true=3, model N=4, only 2 observed: which unobserved slot is spare is unknown."""
    J = np.eye(4) * -1.0
    with pytest.raises(NotImplementedError, match="ambiguous"):
        spurious_species_metrics(J, np.ones(4), (0, 1), n_true=3)


def test_under_parameterised_and_bad_inputs_raise():
    J = np.eye(3) * -1.0
    with pytest.raises(ValueError, match="cannot represent"):
        spurious_species_metrics(J, np.ones(3), (0, 1), n_true=4)
    with pytest.raises(ValueError, match="non-finite"):
        spurious_species_metrics(np.full((3, 3), np.nan), np.ones(3), OBSERVED, N_TRUE)
    with pytest.raises(ValueError, match="square"):
        spurious_species_metrics(np.zeros((3, 2)), np.ones(3), OBSERVED, N_TRUE)
    with pytest.raises(ValueError, match="D_rec"):
        spurious_species_metrics(J, np.ones(2), OBSERVED, N_TRUE)
    with pytest.raises(ValueError, match="outside range"):
        spurious_species_metrics(J, np.ones(3), (0, 7), N_TRUE)
    with pytest.raises(ValueError, match="duplicates"):
        spurious_species_metrics(J, np.ones(3), (0, 0), N_TRUE)
    with pytest.raises(ValueError, match="identically zero"):
        # zero observed block => no scale for a relative cut
        spurious_species_metrics(_J3([0, 0], [0, 0], -1.0, obs_block=np.zeros((2, 2))),
                                 np.ones(3), OBSERVED, N_TRUE)


# ======================================================================================
# observed-block agreement (the 3x3-vs-2x2 case that used to go silently NaN)
# ======================================================================================
def test_observed_block_perfect_recovery():
    J = _J3(extra_row=[0.0, 0.0], extra_col=[0.0, 0.0], diag_extra=-0.5)
    a = observed_block_agreement(J, J_TRUE_2x2, OBSERVED)
    assert a["n_observed"] == 2
    assert a["obs_sign_match_frac"] == pytest.approx(1.0)
    assert a["obs_sign_match_offdiag_frac"] == pytest.approx(1.0)
    assert a["obs_rel_fro_err"] == pytest.approx(0.0, abs=1e-12)
    assert a["obs_scale_ratio"] == pytest.approx(1.0)


def test_observed_block_scores_a_3x3_against_a_2x2_truth():
    """The Experiment B shape mismatch is scored, not NaN'd."""
    rec_block = np.array([[0.5, -0.9],     # signs match truth
                          [-0.4, -0.6]])   # J[1,0] sign FLIPPED vs truth (+1.1)
    J = _J3(extra_row=[0.02, 0.02], extra_col=[0.02, 0.02], diag_extra=-0.3,
            obs_block=rec_block)
    a = observed_block_agreement(J, J_TRUE_2x2, OBSERVED)
    assert a["obs_sign_match_frac"] == pytest.approx(3.0 / 4.0)
    assert a["obs_sign_match_offdiag_frac"] == pytest.approx(0.5)
    assert np.isfinite(a["obs_rel_fro_err"]) and a["obs_rel_fro_err"] > 0.0


def test_scale_free_error_separates_shape_from_magnitude():
    """A uniformly rescaled recovery: large raw error, ~zero error after best scaling."""
    J = _J3(extra_row=[0.0, 0.0], extra_col=[0.0, 0.0], diag_extra=-0.5,
            obs_block=0.25 * J_TRUE_2x2)
    a = observed_block_agreement(J, J_TRUE_2x2, OBSERVED)
    assert a["obs_sign_match_frac"] == pytest.approx(1.0)
    assert a["obs_rel_fro_err"] == pytest.approx(0.75)
    assert a["obs_scale_ratio"] == pytest.approx(4.0)
    assert a["obs_rel_fro_err_scaled"] == pytest.approx(0.0, abs=1e-12)


def test_observed_block_bad_inputs_raise():
    J = _J3([0, 0], [0, 0], -0.5)
    with pytest.raises(ValueError, match="outside the true Jacobian"):
        observed_block_agreement(J, J_TRUE_2x2, (0, 2))
    with pytest.raises(ValueError, match="identically zero"):
        observed_block_agreement(J, np.zeros((2, 2)), OBSERVED)


# ======================================================================================
# the run-index wrapper, via a stub model (no recovery run)
# ======================================================================================
class _StubModel:
    """Minimal stand-in for RNGRN: returns a fixed J and D via the same call surface."""

    def __init__(self, J, D):
        import torch
        self._J = torch.as_tensor(np.asarray(J, dtype=float))
        self.D = torch.as_tensor(np.asarray(D, dtype=float))
        self.N = int(self._J.shape[0])

    def jacobian(self, xstar, create_graph=True):
        return self._J


class _StubResult:
    def __init__(self, J, D, latent_fields=None, loss=0.123):
        self.model = _StubModel(J, D)
        self.xstar = np.ones(self.model.N) * 0.5
        self.latent_fields = latent_fields
        self.loss = loss


class _StubKey:
    def __init__(self, J, n_species_true):
        self.J = J
        self.n_species_true = n_species_true


def test_overparam_report_is_flat_and_carries_both_species_counts():
    J = _J3(extra_row=[1e-4, 1e-4], extra_col=[1e-4, 1e-4], diag_extra=-0.6)
    latent = np.full((1, 4, 4), 0.7)          # featureless spare latent field
    res = _StubResult(J, D_3, latent_fields=latent)
    row = overparam_report(res, _StubKey(J_TRUE_2x2, 2), OBSERVED)

    assert row["n_true"] == 2 and row["n_model"] == 3
    assert row["n_model_from_result"] == 3
    assert row["spare_species_inert"] is True
    assert row["block_status"] == "scored"
    assert row["obs_sign_match_frac"] == pytest.approx(1.0)
    assert row["latent_extra_std_max"] == pytest.approx(0.0, abs=1e-12)
    assert row["latent_status"] == "ok"
    assert "over-parameterised" in row["overparam_status"]
    for k, v in row.items():
        assert isinstance(v, (int, float, bool, str, type(None))), (k, type(v))


def test_overparam_report_flags_a_false_positive_third_gene():
    J = _J3(extra_row=[0.9, 0.1], extra_col=[1.4, 0.2], diag_extra=-0.6)
    row = overparam_report(_StubResult(J, D_3), _StubKey(J_TRUE_2x2, 2), OBSERVED)
    assert row["spare_species_inert"] is False
    assert row["max_abs_coupling_from_extra"] == pytest.approx(1.4)
    assert row["latent_status"].startswith("absent")


def test_overparam_report_rejects_an_inconsistent_answer_key():
    """J of the wrong size for n_species_true must raise, never silently NaN."""
    J = _J3([0, 0], [0, 0], -0.5)
    res = _StubResult(J, D_3)
    with pytest.raises(ValueError, match="internally inconsistent"):
        overparam_report(res, _StubKey(np.eye(3), 2), OBSERVED)
    with pytest.raises(ValueError, match="answer_key.J is None"):
        overparam_report(res, _StubKey(None, 2), OBSERVED)
    row = overparam_report(res, _StubKey(None, 2), OBSERVED, require_true_J=False)
    assert row["block_status"] == "skipped: answer_key.J is None"
    assert "obs_sign_match_frac" not in row


def test_scoring_init_imports_no_sibling_modules():
    """The scoring __init__ must stay import-light so two authors do not collide.

    Asserted structurally (ast on the source), not by attribute probing: a sibling
    already imported by another test would make a hasattr check pass vacuously.
    """
    import ast
    import pathlib

    import rngrn.scoring as pkg

    init = pathlib.Path(pkg.__file__)
    tree = ast.parse(init.read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == [], (
        f"rngrn/scoring/__init__.py must import nothing (found {len(imports)} import "
        "statements); submodules are imported explicitly by their users.")
    assert pkg.__doc__ is not None and "SCORING-SIDE" in pkg.__doc__


def test_overparam_is_not_imported_by_recovery_side():
    """Firewall direction: no recovery-side module may pull in this scoring module."""
    import ast
    import pathlib

    import rngrn

    src = pathlib.Path(rngrn.__file__).parent
    recovery_side = [
        "model.py", "observables.py", "recover.py",
        "losses/terms.py", "losses/total.py", "losses/weighting.py",
        "eval/rollout.py", "eval/numerics.py", "eval/dynamical.py",
        "eval/analysis.py", "eval/topology.py",
    ]
    for rel in recovery_side:
        text = (src / rel).read_text()
        names = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names += [mod] + [f"{mod}.{a.name}" for a in node.names]
        joined = " ".join(names)
        assert "scoring" not in joined, (
            f"FIREWALL BREACH: {rel} imports the scoring side ({joined!r}). "
            "rngrn.scoring.overparam reads the answer key and is scoring-only.")
