"""test_td_figures.py — the figure contract for the training-data inspection notebook.

The owner asked for plots that carry "a colour bar and axis labels and tick marks", so
those three are not stylistic preferences here: they are the acceptance criteria, and this
file is what makes them verifiable rather than asserted. The existing report gallery
(``experiments/figures_report/stage0/f3_patterns_*.png``) has colorbars but NO axis labels
and NO ticks, which is the gap these helpers close.

Everything is tested against SYNTHETIC fields. No test here reads ``payload.h5`` — the
payloads are gitignored, so a test that needed one would fail on any fresh clone.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pytest                                                      # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import td_figures as TD                                            # noqa: E402


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def striped_field(n=64, periods=4, amp=1.0, offset=2.0):
    """A field with a KNOWN dominant wavenumber, so k* recovery is checkable."""
    x = np.arange(n)
    return offset + amp * np.sin(2 * np.pi * periods * x / n)[None, :] * np.ones((n, 1))


def flat_field(n=64, value=1.5):
    return np.full((n, n), value, dtype=float)


def fake_sample(n_species=3, n=64, L=40.0, morphology="stripes"):
    frame = np.stack([striped_field(n, periods=i + 3) for i in range(n_species)])
    return {
        "key": "sample_0000",
        "dataset_id": "fake_ds",
        "final_frame": frame,
        "L": L,
        "k_star": 2 * np.pi * 3 / L,
        "morphology": morphology,
        "grid": n,
        "n_species": n_species,
        "attrs": {"topology": "double_inhibitor", "cv0": 0.4},
    }


def colorbar_axes(fig):
    """The Axes matplotlib creates for a colorbar carry a non-None ``_colorbar`` owner."""
    return [ax for ax in fig.axes if getattr(ax, "_colorbar_info", None) is not None
            or ax.get_label() == "<colorbar>"]


# --------------------------------------------------------------------------------------
# plot_field_panel — the three acceptance criteria
# --------------------------------------------------------------------------------------
def test_field_panel_attaches_a_colorbar_with_a_label():
    fig, ax = plt.subplots()
    TD.plot_field_panel(ax, striped_field(), L=40.0, cbar_label="u  (concentration)")
    cbars = colorbar_axes(fig)
    assert len(cbars) == 1, "every field panel must carry exactly one colorbar"
    assert cbars[0].get_ylabel() == "u  (concentration)"
    plt.close(fig)


def test_field_panel_labels_both_axes_in_physical_units():
    fig, ax = plt.subplots()
    TD.plot_field_panel(ax, striped_field(), L=40.0)
    assert ax.get_xlabel().strip() != ""
    assert ax.get_ylabel().strip() != ""
    # the label must name the physical coordinate, not the pixel index
    assert "x" in ax.get_xlabel()
    assert "y" in ax.get_ylabel()
    plt.close(fig)


def test_field_panel_has_tick_marks_spanning_the_domain():
    L = 40.0
    fig, ax = plt.subplots()
    TD.plot_field_panel(ax, striped_field(), L=L)
    xt, yt = ax.get_xticks(), ax.get_yticks()
    assert len(xt) >= 3 and len(yt) >= 3, "ticks must be present on both axes"
    # ticks are in PHYSICAL units: they span [0, L], not [0, n_pixels]
    assert np.isclose(xt.min(), 0.0) and np.isclose(xt.max(), L)
    assert np.isclose(yt.min(), 0.0) and np.isclose(yt.max(), L)
    plt.close(fig)


def test_field_panel_extent_is_physical_not_pixel_indexed():
    """A 64x64 array on a domain of size 40 must be drawn on [0,40]^2, not [0,64]^2."""
    L = 40.0
    fig, ax = plt.subplots()
    im = TD.plot_field_panel(ax, striped_field(n=64), L=L)
    assert tuple(im.get_extent()) == (0.0, L, 0.0, L)
    plt.close(fig)


def test_field_panel_rejects_a_non_2d_field():
    """Fail loud (CLAUDE.md section 4) rather than silently drawing the wrong thing."""
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="2-D"):
        TD.plot_field_panel(ax, np.zeros((3, 8, 8)), L=10.0)
    plt.close(fig)


def test_field_panel_rejects_a_nonpositive_domain():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="L"):
        TD.plot_field_panel(ax, striped_field(), L=0.0)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# the RAPS panel — the QUANTITATIVE patterning check
# --------------------------------------------------------------------------------------
def test_raps_panel_recovers_a_known_wavenumber():
    """A field with 5 periods across L must peak at k = 2*pi*5/L."""
    L, periods = 40.0, 5
    fig, ax = plt.subplots()
    kstar = TD.plot_raps_panel(ax, striped_field(n=64, periods=periods), L=L)
    assert kstar == pytest.approx(2 * np.pi * periods / L, rel=0.15)
    assert ax.get_xlabel().strip() != "" and ax.get_ylabel().strip() != ""
    plt.close(fig)


def test_raps_panel_marks_the_reference_wavenumber_when_given_one():
    fig, ax = plt.subplots()
    TD.plot_raps_panel(ax, striped_field(), L=40.0, k_ref=0.47)
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert any("generator" in str(x) for x in labels), "the stored k* must be drawn for comparison"
    plt.close(fig)


# --------------------------------------------------------------------------------------
# patterning verdict — measured, not asserted (CLAUDE.md section 8)
# --------------------------------------------------------------------------------------
def test_patterning_verdict_flags_a_flat_field_as_unpatterned():
    v = TD.patterning_verdict(flat_field(), L=40.0)
    assert v["has_contrast"] is False
    assert v["cv"] == pytest.approx(0.0, abs=1e-9)


def test_patterning_verdict_flags_a_striped_field_as_patterned():
    v = TD.patterning_verdict(striped_field(amp=1.0, offset=2.0), L=40.0)
    assert v["has_contrast"] is True
    assert v["cv"] > 0.05
    assert v["k_star_obs"] > 0.0


def blob_field(n=96, n_blobs=3, width=4.0, seed=0):
    """A few isolated Gaussian blobs — HIGH contrast but NOT periodic.

    This is the failure mode the cv test alone cannot see, and the reason ``periodic``
    exists as a separate verdict. three_gene_qvar/sample_0032 is a real instance.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    out = np.full((n, n), 0.1)
    for _ in range(n_blobs):
        cx, cy = rng.integers(0, n, 2)
        out += 0.5 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width ** 2))
    return out


def test_a_periodic_field_has_contrast_and_is_periodic():
    v = TD.patterning_verdict(striped_field(n=96, periods=6), L=40.0)
    assert v["has_contrast"] is True
    assert v["periodic"] is True
    assert v["peak_bin"] >= TD.PEAK_BIN_MIN


def test_isolated_blobs_pass_the_cv_test_but_fail_the_periodicity_test():
    """The whole point: high contrast is not the same claim as a Turing pattern."""
    v = TD.patterning_verdict(blob_field(), L=180.0)
    assert v["has_contrast"] is True, "isolated blobs do have spatial contrast"
    assert v["periodic"] is False, "...but their spectrum has no interior peak"


def test_a_flat_field_is_neither():
    v = TD.patterning_verdict(flat_field(), L=40.0)
    assert v["has_contrast"] is False


def test_periods_across_box_matches_the_construction():
    """A field built with 6 periods across the box must measure ~6 periods across it."""
    v = TD.patterning_verdict(striped_field(n=96, periods=6), L=40.0)
    assert v["periods_across_box"] == pytest.approx(6.0, rel=0.2)


def test_the_contrast_key_does_not_collide_with_the_prereg_patterned_name():
    """`eval/rollout.py` already owns `patterned` for a DIFFERENT quantity (rollout
    amplitude vs max(1e-3, 0.02*|x*_0|)), and that one is pre-registered. Reusing the key
    here would invite reading a data-inspection number against a rollout rate. See
    docs/DECISIONS.md D-TDPLOT-1."""
    v = TD.patterning_verdict(striped_field(), L=40.0)
    assert "patterned" not in v
    assert "has_contrast" in v


def test_peak_bin_threshold_sits_in_the_measured_gap():
    """Calibrated on all 413 registered samples: one sample at bin 1, the rest at bin 3+.
    The threshold must fall in that empty gap, not cut through populated data."""
    assert TD.PEAK_BIN_MIN == 3


def test_patterning_verdict_uses_the_documented_cv_threshold():
    """The generator's own reject rule is cv < 0.05; the verdict must use the SAME number
    so 'patterned' here means what it meant at generation time."""
    assert TD.CV_PATTERNED_MIN == 0.05


# --------------------------------------------------------------------------------------
# gallery / detail figures
# --------------------------------------------------------------------------------------
def test_gallery_draws_one_panel_per_sample_each_with_a_colorbar():
    samples = [fake_sample() for _ in range(5)]
    fig = TD.gallery_figure(samples, ncols=3, title="fake")
    assert len(colorbar_axes(fig)) == 5
    plt.close(fig)


def test_gallery_panels_carry_labels_and_ticks_too():
    fig = TD.gallery_figure([fake_sample() for _ in range(2)], ncols=2, title="fake")
    field_axes = [ax for ax in fig.axes if ax.get_images()]
    assert len(field_axes) == 2
    for ax in field_axes:
        assert ax.get_xlabel().strip() != "" and ax.get_ylabel().strip() != ""
        assert len(ax.get_xticks()) >= 3 and len(ax.get_yticks()) >= 3
    plt.close(fig)


def test_gallery_rejects_an_empty_sample_list():
    with pytest.raises(ValueError, match="no samples"):
        TD.gallery_figure([], ncols=3, title="fake")


def test_detail_figure_has_one_panel_per_species_plus_a_spectrum():
    s = fake_sample(n_species=3)
    fig = TD.detail_figure(s)
    assert len(colorbar_axes(fig)) == 3, "one colorbar per species panel"
    field_axes = [ax for ax in fig.axes if ax.get_images()]
    assert len(field_axes) == 3
    plt.close(fig)


def test_detail_figure_handles_a_two_species_sample():
    fig = TD.detail_figure(fake_sample(n_species=2))
    assert len(colorbar_axes(fig)) == 2
    plt.close(fig)


# --------------------------------------------------------------------------------------
# re-simulation comparison
# --------------------------------------------------------------------------------------
def test_resim_figure_reports_the_reproduction_error():
    stored = striped_field(periods=4)
    fig, err = TD.resim_figure(stored=stored, resim=stored.copy(), L=40.0,
                               title="exact reproduction")
    assert err == pytest.approx(0.0, abs=1e-12)
    plt.close(fig)


def test_resim_figure_error_is_relative_and_nonzero_when_fields_differ():
    stored = striped_field(periods=4)
    fig, err = TD.resim_figure(stored=stored, resim=stored + 0.5, L=40.0, title="offset")
    assert err > 0.0
    plt.close(fig)


def test_resim_figure_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        TD.resim_figure(stored=striped_field(n=64), resim=striped_field(n=32), L=40.0,
                        title="mismatch")


# --------------------------------------------------------------------------------------
# which samples can actually be re-simulated
# --------------------------------------------------------------------------------------
def _resimulatable_sample():
    s = fake_sample()
    s["params"] = {"b": [1, 1, 1], "interaction_matrix": [[1, -1, -1], [1, 0, 0], [1, 0, 0]]}
    s["x_star"] = np.ones(3)
    s["attrs"] = {"sim_seed": 7, "periods_per_box": 5, "k_star": 0.4, "grid": 64}
    return s


def test_a_sample_with_kinetics_and_seed_is_resimulatable():
    assert TD.is_resimulatable(_resimulatable_sample()) is True


def test_a_sample_without_a_seed_is_not_resimulatable():
    """The initial condition is a SEEDED perturbation, so no seed means no reproduction."""
    s = _resimulatable_sample()
    del s["attrs"]["sim_seed"]
    assert TD.is_resimulatable(s) is False


def test_a_sample_without_kinetics_is_not_resimulatable():
    """three_gene_train/test/val store no params_json — their generator was never tracked."""
    s = _resimulatable_sample()
    s["params"] = None
    assert TD.is_resimulatable(s) is False


def test_corpus_summary_draws_three_labelled_panels():
    records = [{"dataset_id": f"ds{i % 2}", "cv": 0.3 + 0.01 * i, "k_star_obs": 0.4,
                "k_star": 0.41, "morphology": ["spots", "stripes"][i % 2]}
               for i in range(12)]
    fig = TD.corpus_summary_figure(records)
    assert len(fig.axes) == 3
    for ax in fig.axes:
        assert ax.get_title().strip() != ""
        assert ax.get_ylabel().strip() != ""
    plt.close(fig)


def test_corpus_summary_rejects_no_records():
    with pytest.raises(ValueError, match="no records"):
        TD.corpus_summary_figure([])


def test_resimulate_refuses_loudly_rather_than_guessing():
    s = _resimulatable_sample()
    s["params"] = None
    with pytest.raises(ValueError, match="cannot be re-simulated"):
        TD.resimulate(s)
