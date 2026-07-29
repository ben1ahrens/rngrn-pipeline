"""test_plausibility.py — unit tests for rngrn.scoring.plausibility and
rngrn.losses.terms.param_prior (unit 5, biological plausibility).

Uses a small temp YAML box (never the real configs/bio_box.yaml numbers) so the tests
assert against numbers WE control, and so a change to the shipped box cannot silently
break them. That temp-box pattern is also how test_yaml_is_only_source_of_numbers
proves the module carries no fallback box of its own.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from rngrn.scoring import plausibility as PLAUS


# --------------------------------------------------------------------------------------
# a small, self-contained box distinct from the real configs/bio_box.yaml
# --------------------------------------------------------------------------------------
TEST_BOX = {
    "alpha": {"low": 1.0, "high": 3.0, "source": "test fixture"},
    "delta": {"low": 2.0, "high": 4.0, "source": "test fixture"},
    "beta": {"low": None, "high": None, "source": "UNCITED"},
    "d_ratio": {"low": 2.0, "high": 20.0, "centre": 5.0, "spread": 1.0,
               "source": "test fixture"},
}


@pytest.fixture()
def box_path(tmp_path):
    p = tmp_path / "bio_box.yaml"
    p.write_text(yaml.safe_dump(TEST_BOX))
    return str(p)


@pytest.fixture()
def box(box_path):
    return PLAUS.load_box(box_path)


# --------------------------------------------------------------------------------------
# load_box / BoxRow
# --------------------------------------------------------------------------------------
def test_load_box_parses_bounds_and_uncited_rows(box):
    assert box["alpha"].low == 1.0 and box["alpha"].high == 3.0
    assert box["alpha"].cited and box["alpha"].bounded
    assert box["beta"].cited is False
    assert box["beta"].bounded is False


def test_load_box_requires_source(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"alpha": {"low": 1.0, "high": 2.0}}))
    with pytest.raises(ValueError, match="source"):
        PLAUS.load_box(str(p))


# --------------------------------------------------------------------------------------
# d_ratio_of — the mobile-pair ratio and immobile-node exclusion
# --------------------------------------------------------------------------------------
def test_d_ratio_of_is_largest_over_second_largest():
    D = np.array([0.5, 4.0, 1.0])
    # sorted: 0.5, 1.0, 4.0 -> largest/second-largest = 4.0/1.0
    assert PLAUS.d_ratio_of(D) == pytest.approx(4.0)


def test_d_ratio_of_ignores_a_near_immobile_node():
    """An immobile third species (D -> 0) must NOT move the ratio at all, however
    small it gets — it is excluded from the computation by construction, not merely
    down-weighted."""
    mobile_pair_ratio = PLAUS.d_ratio_of(np.array([2.0, 6.0, 1e-6]))
    for immobile in (1e-3, 1e-6, 1e-12):
        assert PLAUS.d_ratio_of(np.array([2.0, 6.0, immobile])) == pytest.approx(
            mobile_pair_ratio)


def test_d_ratio_of_rejects_fewer_than_two_species():
    with pytest.raises(ValueError):
        PLAUS.d_ratio_of(np.array([1.0]))


def test_d_ratio_of_rejects_nonpositive():
    with pytest.raises(ValueError):
        PLAUS.d_ratio_of(np.array([1.0, -0.5]))


# --------------------------------------------------------------------------------------
# plausibility_report — in-box scoring
# --------------------------------------------------------------------------------------
def test_all_parameters_inside_box_score_1(box_path):
    alpha = np.full((2, 2), 2.0)     # inside [1, 3]
    delta = np.full(2, 3.0)          # inside [2, 4]
    beta = np.full(2, 100.0)         # UNCITED: any value, never scored
    D = np.array([1.0, 5.0])         # ratio 5.0, inside [2, 20], equals centre

    out = PLAUS.plausibility_report(alpha, delta, beta, D, box_path=box_path)
    assert out["plausibility_alpha_in_box"] is True
    assert out["plausibility_delta_in_box"] is True
    assert out["plausibility_beta_in_box"] is None       # UNCITED, not scored
    assert out["plausibility_d_ratio_in_box"] is True
    assert out["plausibility_n_scored"] == 3              # alpha, delta, d_ratio only
    assert out["plausibility_score"] == pytest.approx(1.0)


def test_out_of_box_parameters_score_as_documented(box_path):
    alpha = np.full((2, 2), 50.0)    # outside [1, 3]
    delta = np.full(2, 3.0)          # inside [2, 4]
    beta = np.full(2, 100.0)         # UNCITED
    D = np.array([1.0, 5.0])         # inside

    out = PLAUS.plausibility_report(alpha, delta, beta, D, box_path=box_path)
    assert out["plausibility_alpha_in_box"] is False
    assert out["plausibility_delta_in_box"] is True
    assert out["plausibility_d_ratio_in_box"] is True
    # 2 of 3 scored rows in-box
    assert out["plausibility_score"] == pytest.approx(2.0 / 3.0)


def test_a_single_out_of_box_entry_fails_the_whole_parameter(box_path):
    """One runaway alpha[i,j] makes the whole parameter implausible, not just that
    entry — plausibility_report requires ALL entries in-box."""
    alpha = np.array([[2.0, 2.0], [2.0, 50.0]])
    delta = np.full(2, 3.0)
    beta = np.zeros(2)
    D = np.array([1.0, 5.0])
    out = PLAUS.plausibility_report(alpha, delta, beta, D, box_path=box_path)
    assert out["plausibility_alpha_in_box"] is False


def test_missing_row_raises(tmp_path):
    p = tmp_path / "incomplete.yaml"
    p.write_text(yaml.safe_dump({"alpha": {"low": 1.0, "high": 3.0, "source": "x"}}))
    with pytest.raises(KeyError):
        PLAUS.plausibility_report(np.ones(2), np.ones(2), np.ones(2), np.array([1.0, 2.0]),
                                  box_path=str(p))


# --------------------------------------------------------------------------------------
# the YAML is the ONLY source of the numbers — no hard-coded box in the module
# --------------------------------------------------------------------------------------
def test_yaml_is_only_source_of_the_numbers(tmp_path):
    """A box loaded from an arbitrary, hand-built YAML with numbers that appear nowhere
    in the real configs/bio_box.yaml must be exactly what plausibility_report uses —
    proving there is no fallback/hard-coded box inside the module overriding it."""
    weird_box = {
        "alpha": {"low": 111.0, "high": 222.0, "source": "arbitrary test value"},
        "delta": {"low": -50.0, "high": -10.0, "source": "arbitrary test value"},
        "beta": {"low": None, "high": None, "source": "UNCITED"},
        "d_ratio": {"low": 999.0, "high": 1000.0, "centre": 999.5, "spread": 0.1,
                   "source": "arbitrary test value"},
    }
    p = tmp_path / "weird.yaml"
    p.write_text(yaml.safe_dump(weird_box))

    # A value that would be OUT of box under any real bio_box.yaml row (alpha in
    # [0.5, 10]) but IS inside this arbitrary box ([111, 222]).
    out = PLAUS.plausibility_report(
        alpha=np.full((2, 2), 150.0), delta=np.full(2, -20.0),
        beta=np.zeros(2), D=np.array([1.0, 999.7]), box_path=str(p))
    assert out["plausibility_alpha_in_box"] is True      # would be False under bio_box.yaml
    assert out["plausibility_delta_in_box"] is True      # -20 IS in [-50, -10] (this box)
    assert out["plausibility_d_ratio_in_box"] is True    # ratio 999.7 in [999, 1000]


# --------------------------------------------------------------------------------------
# losses/terms.param_prior — the soft prior
# --------------------------------------------------------------------------------------
class _StubModel:
    """Minimal stand-in exposing exactly the properties param_prior reads (torch
    tensors, no gradient graph beyond theta_D/theta_alpha/theta_delta)."""
    def __init__(self, D, alpha, delta):
        self.theta_D = torch.tensor(D, dtype=torch.float64, requires_grad=True)
        self._alpha = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
        self._delta = torch.tensor(delta, dtype=torch.float64, requires_grad=True)

    @property
    def D(self):
        return self.theta_D  # already positive in tests, no exp() needed

    @property
    def alpha(self):
        return self._alpha

    @property
    def delta(self):
        return self._delta


def test_param_prior_is_zero_at_the_centre(box_path):
    from rngrn.losses.terms import param_prior
    # D chosen so the mobile-pair ratio is exactly the box's centre (5.0). alpha/delta
    # sit dead in the middle of their boxes, so the box hinges are near-zero too — the
    # softplus hinge is asymptotically, not exactly, zero away from the boundary (same
    # design as turing_hinges/anticollapse elsewhere in this file), hence the loose
    # tolerance rather than an exact 0.
    model = _StubModel(D=[1.0, 5.0], alpha=[[2.0]], delta=[3.0])
    loss, parts = param_prior(model, dratio_centre=5.0, dratio_spread=1.0, box_path=box_path)
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-4)
    assert parts["d_ratio"] == pytest.approx(5.0)
    assert parts["L_dratio"] == pytest.approx(0.0, abs=1e-12)  # the log-normal term IS exact


def test_param_prior_gradient_points_toward_the_centre(box_path):
    """Perturbing D away from the centre and taking a small gradient step on theta_D
    must move the ratio BACK toward the centre, not away from it."""
    from rngrn.losses.terms import param_prior
    model = _StubModel(D=[1.0, 50.0], alpha=[[2.0]], delta=[3.0])  # ratio 50, far above 5
    loss, _ = param_prior(model, dratio_centre=5.0, dratio_spread=1.0, box_path=box_path)
    loss.backward()
    grad = model.theta_D.grad
    assert grad is not None
    # gradient descent step: theta_D_new = theta_D - lr * grad
    lr = 1e-3
    with torch.no_grad():
        new_D = model.theta_D - lr * grad
    new_ratio = PLAUS.d_ratio_of(new_D.numpy())
    old_ratio = PLAUS.d_ratio_of(model.theta_D.detach().numpy())
    assert new_ratio < old_ratio   # moved down, toward the centre of 5.0


def test_param_prior_does_not_penalise_an_immobile_node(box_path):
    """A third, near-immobile species (very small D) must leave the prior identical to
    the two-species case with the same mobile pair — its magnitude must not matter."""
    from rngrn.losses.terms import param_prior
    two_species = _StubModel(D=[1.0, 5.0], alpha=[[2.0]], delta=[3.0])
    loss_2, _ = param_prior(two_species, dratio_centre=5.0, dratio_spread=1.0, box_path=box_path)

    three_species_a = _StubModel(D=[1.0, 5.0, 1e-4], alpha=[[2.0]], delta=[3.0])
    loss_3a, _ = param_prior(three_species_a, dratio_centre=5.0, dratio_spread=1.0,
                             box_path=box_path)

    three_species_b = _StubModel(D=[1.0, 5.0, 1e-9], alpha=[[2.0]], delta=[3.0])
    loss_3b, _ = param_prior(three_species_b, dratio_centre=5.0, dratio_spread=1.0,
                             box_path=box_path)

    assert float(loss_3a.detach()) == pytest.approx(float(loss_2.detach()), abs=1e-9)
    assert float(loss_3b.detach()) == pytest.approx(float(loss_2.detach()), abs=1e-9)


def test_param_prior_box_hinge_is_zero_inside_and_positive_outside(box_path):
    from rngrn.losses.terms import param_prior
    inside = _StubModel(D=[1.0, 5.0], alpha=[[2.0]], delta=[3.0])       # alpha in [1,3]
    _, parts_in = param_prior(inside, dratio_centre=5.0, box_path=box_path)
    assert parts_in.get("L_box_alpha", 0.0) == pytest.approx(0.0, abs=1e-4)

    outside = _StubModel(D=[1.0, 5.0], alpha=[[50.0]], delta=[3.0])     # alpha way > 3
    _, parts_out = param_prior(outside, dratio_centre=5.0, box_path=box_path)
    assert parts_out["L_box_alpha"] > 0.0


def test_param_prior_does_not_hinge_an_uncited_parameter(box_path):
    """beta has no bounds in the test box (UNCITED) and param_prior does not even read
    beta — confirm it never appears in the returned parts."""
    from rngrn.losses.terms import param_prior
    model = _StubModel(D=[1.0, 5.0], alpha=[[2.0]], delta=[3.0])
    _, parts = param_prior(model, dratio_centre=5.0, box_path=box_path)
    assert "L_box_beta" not in parts


def test_terms_and_scoring_box_loaders_agree_on_the_real_box():
    """losses/terms.py cannot import rngrn.scoring (firewall — see the note atop
    param_prior in that file) so it carries its own tiny YAML loader. Cross-check the
    two independent loaders against the real configs/bio_box.yaml so a future edit to
    one cannot silently drift from the other."""
    from rngrn.losses.terms import _load_box_bounds

    scoring_box = PLAUS.load_box("configs/bio_box.yaml")
    terms_box = _load_box_bounds("configs/bio_box.yaml")

    assert set(scoring_box) == set(terms_box)
    for name, row in scoring_box.items():
        expected = (row.low, row.high) if row.bounded else None
        assert terms_box[name] == expected, name
