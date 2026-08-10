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

import numpy as np
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
# --------------------------------------------------------------------------------------
# the domain is FIXED, so periodicity is emergent
# --------------------------------------------------------------------------------------
def test_the_domain_is_a_single_constant():
    """The central property. L is chosen without reference to any system's k*, so it cannot
    encode k* — the leak is structurally absent rather than merely small."""
    assert isinstance(CS.DOMAIN_L, float)
    assert CS.DOMAIN_L > 0


def test_the_selector_gives_every_sample_the_same_domain():
    t = _fake_table()
    sel = CS.select(t, _all_stable(t), per_class=4, n_tuning=2)
    Ls = {s["domain_L"] for d in sel["datasets"].values() for s in d["samples"]}
    assert Ls == {CS.DOMAIN_L}, f"L must be one constant across the corpus, got {Ls}"


def test_periods_are_emergent_and_generally_not_integers():
    """p = L*k*/(2pi) is read off the physics. If these came out integral it would mean
    something was still choosing them."""
    t = _fake_table()
    sel = CS.select(t, _all_stable(t), per_class=4, n_tuning=2)
    per = [s["periods_emergent"] for d in sel["datasets"].values() for s in d["samples"]]
    assert all(isinstance(x, float) for x in per)
    assert any(abs(x - round(x)) > 1e-6 for x in per), "no sample has a fractional period"


def test_periods_follow_kstar_exactly():
    """The defining relation: nothing mediates between the system's wavenumber and the
    number of periods in the image except the fixed box."""
    t = _fake_table()
    sel = CS.select(t, _all_stable(t), per_class=4, n_tuning=2)
    for d in sel["datasets"].values():
        for s in d["samples"]:
            assert s["periods_emergent"] == pytest.approx(
                CS.DOMAIN_L * s["k_star"] / (2 * np.pi))


def test_a_system_left_unresolved_by_the_fixed_domain_is_refused():
    """The fixed L must not silently produce an unmeasurable sample. The remedy named in
    the error is to change L for the WHOLE corpus, never per sample."""
    t = _fake_table()
    for r in t:
        r["k_star"] = 8.0          # wavelength 0.79 against L=300 -> ~380 periods
    with pytest.raises(ValueError, match="px/wavelength"):
        CS.select(t, _all_stable(t), per_class=4, n_tuning=2)


def test_the_fixed_domain_resolves_the_real_candidate_pool():
    """Guards the choice of DOMAIN_L itself against the measured wavelength range of the
    corpus (9.8 to 32.8 at the time of selection)."""
    for lam in (9.8, 32.8):
        periods = CS.DOMAIN_L / lam
        ppw = 512.0 / periods
        assert periods >= CS.PERIODS_MIN
        assert CS.PPW_MIN <= ppw <= CS.PPW_MAX


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


# --------------------------------------------------------------------------------------
# label stability across domain size
# --------------------------------------------------------------------------------------
def test_a_label_that_survives_every_probe_is_stable():
    assert CS.label_is_stable(row(morph="stripes"), ["stripes", "stripes"]) is True


def test_a_label_that_flips_at_another_box_size_is_not_stable():
    """Measured: ang_conc drifts 0.190 -> 0.074 across an 8x L span, enough to flip a
    class. A system that does that is not an exemplar of either class."""
    assert CS.label_is_stable(row(morph="stripes"), ["stripes", "labyrinth"]) is False


def test_stability_needs_at_least_one_probe():
    with pytest.raises(ValueError, match="no probe labels"):
        CS.label_is_stable(row(), [])


def test_probe_period_differs_from_the_samples_own_period():
    """Probing at the same p re-runs an identical simulation and proves nothing."""
    assert CS.stability_probe_p({"L": 78.0, "k_star": 0.2416}) != 3


def test_probe_period_stays_inside_the_generator_feasibility_window():
    """A p whose L falls outside [18, 220] makes simulate_and_classify raise -- this bit
    when the probe first ran (p=13 gave L=300.63 for a long-wavelength system)."""
    from gen_tg3 import feasible_periods
    for k in (0.10, 0.2416, 0.75, 1.5):
        p = CS.stability_probe_p({"L": 6 * 2 * np.pi / k, "k_star": k})
        assert p in feasible_periods(k)


def test_native_periods_recovers_the_generating_p():
    """L = p * 2pi/k*, so p = L*k*/(2pi). qvar/sample_0000 was generated at p=3."""
    assert CS.native_periods({"L": 78.01357861389891, "k_star": 0.24161891117478512}) == 3


# --------------------------------------------------------------------------------------
# which classes we actually ship
# --------------------------------------------------------------------------------------
def test_stripes_is_in_the_taxonomy_but_not_shipped():
    """MEASURED 2026-08-10: every stripes candidate in the re-simulatable corpus flips to
    labyrinth when only the box size changes, and no sample at p >= 11 is ever labelled
    stripes. The class is a small-box artefact, so it is not a canonical dataset."""
    assert "stripes" in CS.TAXONOMY_CLASSES
    assert "stripes" not in CS.CANONICAL_CLASSES
    assert CS.CANONICAL_CLASSES == ("spots", "labyrinth")


# --------------------------------------------------------------------------------------
# the selection driver
# --------------------------------------------------------------------------------------
def _fake_table():
    """8 synthetic systems: 4 per canonical class, decreasing margin, all gate-clean.

    Two per class carry system_ids that appear in CS.PREVIOUSLY_RUN (so they are eligible
    for the tuning slots) and two do not (so they can fill the held-out slots).
    """
    out = []
    # burned ids must exist in PREVIOUSLY_RUN; fresh ids must not.
    ids = {"spots": [0, 1, 100, 101], "labyrinth": [2, 3, 102, 103]}
    for cls, phi, an in (("spots", 0.10, 0.05), ("labyrinth", 0.50, 0.20)):
        for i, sid in enumerate(ids[cls]):
            r = row(morph=cls, phi=phi + (0.02 * i if cls == "spots" else 0.0),
                    aniso=an + (0.0 if cls == "spots" else 0.05 * i))
            # distinct k* per system, inside the corpus range 0.191-0.643, so the emergent
            # periods differ from one another the way real systems do
            r.update(source_dataset="three_gene_qvar", source_key=f"sample_{sid:04d}",
                     system_id=sid, k_star=0.22 + 0.07 * i, L=125.0)
            r["margin"] = CS.class_margin(cls, r["area_frac"], r["anisotropy"])
            r["uid"] = CS.row_uid(r)
            out.append(r)
    return out


def test_the_fixture_matches_the_real_previously_run_set():
    """Guards the fixture itself: if PREVIOUSLY_RUN changes, these tests must be updated
    rather than silently testing a different split."""
    t = _fake_table()
    burned = [r for r in t if r["uid"] in CS.PREVIOUSLY_RUN]
    assert len(burned) == 4, "fixture needs exactly 2 burned systems per class"


def _all_stable(table):
    return {r["uid"]: True for r in table}


def test_selection_returns_the_requested_count_per_class():
    t = _fake_table()
    sel = CS.select(t, _all_stable(t), per_class=4, n_tuning=2)
    for cls in CS.CANONICAL_CLASSES:
        assert len(sel["datasets"][f"turing_{cls}"]["samples"]) == 4


def test_unstable_systems_are_excluded_even_at_high_margin():
    t = _fake_table()
    stab = _all_stable(t)
    best = max((r for r in t if r["morphology"] == "spots"), key=lambda r: r["margin"])
    stab[best["uid"]] = False
    sel = CS.select(t, stab, per_class=3, n_tuning=1)
    assert best["uid"] not in {s["uid"] for s in sel["datasets"]["turing_spots"]["samples"]}


def test_no_sample_carries_an_imposed_period_count():
    """Regression guard. Earlier versions wrote an integer `periods_per_box` that WE chose;
    the whole point of the redesign is that no such field exists on this path."""
    t = _fake_table()
    sel = CS.select(t, _all_stable(t), per_class=4, n_tuning=2)
    for d in sel["datasets"].values():
        for s in d["samples"]:
            assert "periods_per_box" not in s


def test_split_roles_are_assigned_and_counted():
    t = _fake_table()
    sel = CS.select(t, _all_stable(t), per_class=4, n_tuning=2)
    roles = [s["role"] for s in sel["datasets"]["turing_spots"]["samples"]]
    assert roles.count("tuning") == 2
    assert roles.count("held_out") == 2


def test_selection_fails_loud_when_a_class_is_short():
    thin = [r for r in _fake_table() if r["morphology"] != "labyrinth"]
    with pytest.raises(ValueError, match="only 0 admissible"):
        CS.select(thin, _all_stable(thin), per_class=4, n_tuning=2)


def test_selection_is_deterministic():
    t = _fake_table()
    s = _all_stable(t)
    assert CS.select(t, s, 4, 2) == CS.select(t, s, 4, 2)
