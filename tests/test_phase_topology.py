"""Morphology measured from the field, rather than read off an area-fraction threshold.

Synthetic patterns with a KNOWN morphology, so the measure is tested against ground truth
rather than against the generator's labels — which is the whole point, since those labels
are what this exists to check.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import phase_topology as PT                                        # noqa: E402


def _grid(n=192):
    y, x = np.mgrid[0:n, 0:n]
    return x, y


def spots_field(n=192, periods=8):
    """Round bright islands on a connected dark background."""
    x, y = _grid(n)
    f = np.cos(2 * np.pi * periods * x / n) + np.cos(2 * np.pi * periods * y / n)
    return (f > 1.0).astype(float) * 1.0 + 0.05 * f


def holes_field(n=192, periods=8):
    """Exactly the inverse: round dark voids in a connected bright matrix."""
    return 1.0 - spots_field(n, periods)


def labyrinth_field(n=192, periods=8, seed=0):
    """Interdigitated worms — neither phase is a set of round blobs.

    Band-passed noise thresholded at the median: the standard construction for a labyrinth,
    and the honest one. A naive `|sin(x)+sin(y)| < c` looks worm-like but is really a bright
    network pierced by round voids — this measure correctly calls that `holes`, which is how
    the first version of this fixture was caught.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, n))
    kx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    KR = np.sqrt(KX ** 2 + KY ** 2)
    band = np.exp(-((KR - periods) ** 2) / (2 * 1.2 ** 2))     # annulus at one wavenumber
    f = np.real(np.fft.ifft2(np.fft.fft2(w) * band))
    return (f > np.median(f)).astype(float) + 0.02 * f


def disconnected_worms_field(n=192, n_worms=30, seed=1):
    """Elongated bright domains, deliberately NOT touching each other.

    Many components AND low circularity — the configuration that breaks a
    connectivity-only rule, which would call this spots.
    """
    rng = np.random.default_rng(seed)
    y, x = _grid(n)
    out = np.zeros((n, n))
    for _ in range(n_worms):
        cx, cy = rng.integers(0, n, 2)
        th = rng.uniform(0, np.pi)
        dx, dy = x - cx, y - cy
        u = dx * np.cos(th) + dy * np.sin(th)
        v = -dx * np.sin(th) + dy * np.cos(th)
        out += np.exp(-(u ** 2) / (2 * 14.0 ** 2) - (v ** 2) / (2 * 1.6 ** 2))
    return out


# --------------------------------------------------------------------------------------
# the three classes, against known ground truth
# --------------------------------------------------------------------------------------
def test_round_bright_islands_read_as_spots():
    assert PT.measured_morphology(spots_field()) == "spots"


def test_round_dark_voids_read_as_holes():
    """The class the generator's classifier can essentially never emit."""
    assert PT.measured_morphology(holes_field()) == "holes"


def test_worms_read_as_labyrinth():
    assert PT.measured_morphology(labyrinth_field()) == "labyrinth"


def test_spots_and_holes_are_distinguished_from_each_other():
    """They have the same domain count and shape; only WHICH phase fragments differs. An
    area-fraction rule conflates them; this must not."""
    a = PT.measure(spots_field())
    b = PT.measure(holes_field())
    assert a["morphology_measured"] == "spots"
    assert b["morphology_measured"] == "holes"
    assert a["bright_domains"] > a["dark_domains"]
    assert b["dark_domains"] > b["bright_domains"]


# --------------------------------------------------------------------------------------
# why each ingredient is needed
# --------------------------------------------------------------------------------------
def test_shape_is_needed_because_worms_can_be_disconnected():
    """turing_labyrinth/sample_0000 has 36 disconnected bright domains and is plainly a
    labyrinth. Counting components alone calls that spots; circularity is what saves it."""
    m = PT.measure(disconnected_worms_field())
    assert m["bright_domains"] >= 4, "the worms here are genuinely disconnected"
    assert m["bright_circularity"] < PT.ROUND_CIRCULARITY_MIN
    assert m["morphology_measured"] == "labyrinth"


def test_periodic_components_stitch_across_the_boundary():
    """One stripe crossing the wrap is one domain, not two."""
    m = np.zeros((32, 32), bool)
    m[:, :3] = True
    m[:, -3:] = True
    assert PT.periodic_components(m) == 1


def test_periodic_components_still_separates_genuinely_distinct_domains():
    m = np.zeros((32, 32), bool)
    m[4:8, 4:8] = True
    m[20:24, 20:24] = True
    assert PT.periodic_components(m) == 2


def test_otsu_finds_the_gap_of_a_bimodal_field():
    f = np.concatenate([np.full(500, 0.1), np.full(500, 0.9)])
    assert 0.1 < PT.otsu_threshold(f.reshape(-1, 1) * np.ones((1, 2))) < 0.9


# --------------------------------------------------------------------------------------
# fail loud
# --------------------------------------------------------------------------------------
def test_a_fine_pattern_is_not_erased_by_the_speckle_floor():
    """Regression. A frame-relative floor deleted every domain of a fine spot lattice, so a
    clean pattern read as `labyrinth` with zero domains. Real instance:
    turing_spots/sample_0004, 36 periods across 512 px."""
    fine = spots_field(n=512, periods=36)
    m = PT.measure(fine)
    assert m["bright_domains"] > 100, "the spots must survive the speckle floor"
    assert m["morphology_measured"] == "spots"


def test_the_speckle_floor_tracks_the_pattern_wavelength():
    coarse = PT.measure(spots_field(n=512, periods=8))
    fine = PT.measure(spots_field(n=512, periods=36))
    assert coarse["min_domain_area_px"] > fine["min_domain_area_px"]
    assert coarse["morphology_measured"] == fine["morphology_measured"] == "spots"


def test_a_non_2d_field_raises():
    with pytest.raises(ValueError, match="2-D"):
        PT.measure(np.zeros((3, 8, 8)))


def test_measure_reports_the_evidence_not_just_the_verdict():
    """A labelling rule that cannot be audited is not usable in this repo."""
    m = PT.measure(spots_field())
    for k in ("bright_fraction", "bright_domains", "dark_domains",
              "bright_circularity", "dark_circularity"):
        assert k in m
