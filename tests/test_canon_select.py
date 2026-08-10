"""Selection logic for the canonical Turing datasets.

Synthetic rows only — no payload.h5 is read here, so this file runs on a fresh clone where
the payloads (gitignored) are absent.

The gates and margins tested here exist because the morphology class boundaries they sit
against are NOT calibrated: `A > 0.55` appears exactly once in the whole repo with no
control behind it, and `phi < 0.34` sits on the Gaussian null. Admitting only high-margin,
label-stable samples is what makes the three canonical strata mean anything.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import canon_select as CS                                          # noqa: E402


def row(morph="spots", phi=0.20, aniso=0.05, cv=0.60, peak_bin=6):
    return {"morphology": morph, "area_frac": phi, "anisotropy": aniso,
            "cv": cv, "peak_bin": peak_bin}


# --------------------------------------------------------------------------------------
# margins
# --------------------------------------------------------------------------------------
def test_spots_margin_is_distance_below_the_area_fraction_cut():
    assert CS.class_margin("spots", phi=0.20, anisotropy=0.05) == pytest.approx(0.14)


def test_stripes_margin_is_the_binding_one_of_three_boundaries():
    # anisotropy 0.62 is 0.07 above the stripe cut; phi 0.40 is 0.06 above the spots cut.
    # The binding (smallest) margin is the phi one.
    assert CS.class_margin("stripes", phi=0.40, anisotropy=0.62) == pytest.approx(0.06)


def test_labyrinth_margin_is_bounded_by_the_anisotropy_cut_from_below():
    assert CS.class_margin("labyrinth", phi=0.50, anisotropy=0.45) == pytest.approx(0.10)


def test_a_sample_on_the_wrong_side_of_a_boundary_has_negative_margin():
    assert CS.class_margin("spots", phi=0.40, anisotropy=0.0) < 0


def test_unknown_morphology_raises_rather_than_scoring_it():
    with pytest.raises(ValueError, match="unknown morphology"):
        CS.class_margin("swirls", phi=0.5, anisotropy=0.5)


# --------------------------------------------------------------------------------------
# admission gates
# --------------------------------------------------------------------------------------
def test_gate_rejects_a_non_periodic_sample():
    """three_gene_qvar/sample_0032 has the LARGEST spots margin in the corpus (phi=0.032)
    precisely because it is nearly empty. peak_bin=1 is the only thing that catches it."""
    assert CS.passes_gates(row(phi=0.032, peak_bin=1)) is False


def test_gate_rejects_a_weak_contrast_sample():
    assert CS.passes_gates(row(cv=0.21)) is False


def test_gate_rejects_a_sample_on_the_wrong_side_of_its_own_boundary():
    assert CS.passes_gates(row(morph="spots", phi=0.40)) is False


def test_gate_accepts_a_clean_sample():
    assert CS.passes_gates(row()) is True


# --------------------------------------------------------------------------------------
# the periods-per-box draw — the anti-leak requirement
# --------------------------------------------------------------------------------------
def test_periods_are_distinct_within_a_dataset():
    """A constant p makes k* = p*2pi/L exact and recreates the domain-size leak."""
    p = CS.draw_periods("turing_spots", n=5, seed=2026)
    assert len(p) == 5
    assert len(set(p)) == 5
    assert all(x in CS.P_CHOICES for x in p)


def test_the_periods_draw_is_deterministic_and_dataset_specific():
    assert CS.draw_periods("turing_spots", 5, 2026) == CS.draw_periods("turing_spots", 5, 2026)
    assert CS.draw_periods("turing_spots", 5, 2026) != CS.draw_periods("turing_stripes", 5, 2026)


def test_period_range_keeps_every_sample_well_resolved_at_512():
    """px/wavelength = 512/p must stay in [16, 32] -- comfortably above the measured
    6 px/wavelength floor (docs/DECISIONS.md D15)."""
    for p in CS.P_CHOICES:
        assert 16 <= 512 / p <= 32


# --------------------------------------------------------------------------------------
# the candidate table
# --------------------------------------------------------------------------------------
def test_candidate_row_uid_is_stable_and_unique_per_system():
    r = {"source_dataset": "three_gene_qvar", "system_id": 7}
    assert CS.row_uid(r) == "three_gene_qvar:7"


def test_only_resimulatable_datasets_are_eligible():
    """The 127 legacy three_gene samples store no params_json and no sim_seed, so their
    kinetics are unrecoverable and they can never be re-simulated at a new resolution."""
    assert CS.ELIGIBLE_DATASETS == ("three_gene_qvar", "three_gene_multiL")


def test_table_rows_carry_every_field_the_selector_needs():
    required = {"source_dataset", "source_key", "system_id", "morphology", "area_frac",
                "anisotropy", "cv", "peak_bin", "margin", "k_star", "L", "uid"}
    assert required <= set(CS.CANDIDATE_FIELDS)
