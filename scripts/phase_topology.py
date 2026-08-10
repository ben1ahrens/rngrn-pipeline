#!/usr/bin/env python
"""phase_topology.py — measure a pattern's morphology from the field itself.

WHY THIS EXISTS
---------------
The generator's ``classify`` (``scripts/gen_tg3.py``) assigns a morphology from the AREA
FRACTION above a fixed contrast threshold: ``phi < 0.34`` spots, ``phi > 0.66`` holes,
otherwise stripes-or-labyrinth by anisotropy. Measured against the 57 distinct
re-simulatable systems, that rule is reliable for ``spots`` and unreliable for the rest:

    stored label      -> spots   -> holes   -> labyrinth
    spots (29)             28         0            1
    labyrinth (17)          3         7            7
    stripes (11)            2         1            8

Only 41% of what it calls ``labyrinth`` is a labyrinth; most of the remainder is a HOLE
pattern — a connected bright matrix pierced by round dark voids, which is a distinct
morphology, not a variant of a labyrinth.

The reason the generator cannot see this is structural, and worth stating precisely because
an earlier version of ``docs/DECISIONS.md`` got it wrong. ``holes`` is NOT unreachable
physics: 7 of 57 systems produce hole patterns, several with negative skew. It is the LABEL
that is unreachable. Detecting holes needs ``phi > 0.66``, i.e. two thirds of all pixels
more than 0.4 SD above the mean — but a connected bright matrix has its own spread, so a
large fraction of the matrix itself falls below that line. The test can essentially never
fire, whatever the field does.

WHAT THIS MEASURES INSTEAD
--------------------------
Two properties of the phases, not one statistic of the histogram:

  * WHICH PHASE IS FRAGMENTED. Split at the Otsu threshold — the value maximising
    between-class variance, i.e. the natural boundary of a bimodal field. Spots fragment the
    bright phase and leave the dark one connected; holes do the reverse. (A median split
    cannot answer this: it forces a 50/50 partition by construction and cuts through the
    background.)
  * WHAT SHAPE THE DOMAINS ARE. Circularity ``4*pi*A/P^2`` is ~1 for a round blob and falls
    towards 0 for a worm. This term is essential: connectivity alone misreads a labyrinth as
    spots whenever its bright worms happen to be disconnected. That misfire was observed on
    ``turing_labyrinth/sample_0000``, which has 36 disconnected bright domains and is
    unmistakably a labyrinth by eye.

SCOPE, AND WHAT THIS DOES NOT DO
--------------------------------
This defines the morphology of the CANONICAL datasets. It does NOT relabel the corpus: the
413 registered samples keep their stored ``morphology`` attribute, so no existing number
changes meaning. See docs/DECISIONS.md D-CANON-5.

Data side. Reads fields only; imported by the canonical scripts and its own test.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# Domains smaller than this fraction of a WAVELENGTH SQUARED are speckle, not structure.
#
# Scaling by the pattern's own wavelength rather than by the frame is not a refinement, it
# is required. A frame-relative floor deletes the entire pattern of a fine field: at 512x512
# with 36 periods a spot is ~7 px across (area ~38 px), while 2e-4 of the frame is 52 px, so
# every domain is discarded and a clean spot lattice reads as `labyrinth` with ZERO domains.
# That is exactly what happened to turing_spots/sample_0004 before this was fixed.
MIN_DOMAIN_AREA_WAVELENGTH_FRAC = 0.02

# A domain is "round" above this circularity. Set from the measured separation on the
# canonical frames: round hole/spot domains score 0.94-1.43, labyrinth worms 0.29-0.75.
# 0.55 sits in that gap. UNCALIBRATED in the strict sense — it is read off a 10-sample
# separation, not a control — and is used for LABELLING, never as a pass condition.
ROUND_CIRCULARITY_MIN = 0.55

# A phase counts as fragmented at this many domains.
FRAGMENTED_MIN = 4

# ...and as a connected matrix at or below this many.
MATRIX_MAX = 2


def otsu_threshold(field, nbins=256):
    """Intensity that maximises between-class variance — the natural phase boundary."""
    f = np.asarray(field, float)
    hist, edges = np.histogram(f.ravel(), bins=nbins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(hist).astype(float)
    if w[-1] == 0:
        raise ValueError("empty field: cannot threshold")
    w /= w[-1]
    total = float(np.sum(hist * centres))
    if total == 0:
        raise ValueError("field sums to zero: cannot threshold")
    m = np.cumsum(hist * centres) / total
    denom = w * (1.0 - w)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (m[-1] * w - m) ** 2 / denom
    return float(centres[int(np.nanargmax(between))])


def periodic_components(mask):
    """Count connected components, stitching opposite edges.

    The generated frames are periodic by construction, so a domain crossing an edge is one
    domain, not two. Ignoring this inflates the count for exactly the large, sprawling
    domains that distinguish a labyrinth.
    """
    lab, n = ndimage.label(np.asarray(mask, bool))
    if n == 0:
        return 0
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for side_a, side_b in ((lab[0, :], lab[-1, :]), (lab[:, 0], lab[:, -1])):
        for a, b in zip(side_a, side_b):
            if a and b:
                union(int(a), int(b))
    return len({find(i) for i in range(1, n + 1)})


def domain_stats(mask, min_area):
    """(number of domains, median circularity) for domains above ``min_area``."""
    mask = np.asarray(mask, bool)
    lab, n = ndimage.label(mask)
    circs = []
    for i in range(1, n + 1):
        m = lab == i
        area = int(m.sum())
        if area < min_area:
            continue
        perim = int((m & ~ndimage.binary_erosion(m)).sum())
        if perim > 0:
            circs.append(4.0 * np.pi * area / (perim ** 2))
    return len(circs), (float(np.median(circs)) if circs else float("nan"))


def dominant_wavelength_px(field):
    """Dominant wavelength in PIXELS, straight from the field's own spectrum.

    Used only to set the speckle floor, so it needs the scale right, not sub-bin precision.
    """
    f = np.asarray(field, float)
    n = f.shape[0]
    F = np.abs(np.fft.fft2(f - f.mean())) ** 2
    kx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    KR = np.sqrt(KX ** 2 + KY ** 2)
    idx = np.clip(np.rint(KR).astype(int), 0, n // 2)
    power = np.bincount(idx.ravel(), weights=F.ravel(), minlength=n // 2 + 1)
    power[0] = 0.0
    mode = int(np.argmax(power))
    return float(n / mode) if mode > 0 else float(n)


def measure(field):
    """Full measurement behind the verdict, so a caller can see why."""
    f = np.asarray(field, float)
    if f.ndim != 2:
        raise ValueError(f"phase topology needs a 2-D field, got shape {f.shape}")
    lam_px = dominant_wavelength_px(f)
    min_area = max(6, int(MIN_DOMAIN_AREA_WAVELENGTH_FRAC * lam_px ** 2))
    bright = f > otsu_threshold(f)
    n_bright, c_bright = domain_stats(bright, min_area)
    n_dark, c_dark = domain_stats(~bright, min_area)
    bright_is_round = n_bright >= FRAGMENTED_MIN and c_bright > ROUND_CIRCULARITY_MIN
    dark_is_round = n_dark >= FRAGMENTED_MIN and c_dark > ROUND_CIRCULARITY_MIN
    if bright_is_round and n_dark <= MATRIX_MAX:
        morph = "spots"
    elif dark_is_round and n_bright <= MATRIX_MAX:
        morph = "holes"
    else:
        morph = "labyrinth"
    return {
        "morphology_measured": morph,
        "bright_fraction": float(bright.mean()),
        "bright_domains": int(n_bright),
        "dark_domains": int(n_dark),
        "bright_circularity": c_bright,
        "dark_circularity": c_dark,
        "periodic_bright_components": periodic_components(bright),
        "periodic_dark_components": periodic_components(~bright),
        "dominant_wavelength_px": lam_px,
        "min_domain_area_px": int(min_area),
    }


def measured_morphology(field):
    """``spots`` | ``holes`` | ``labyrinth``, measured from the field."""
    return measure(field)["morphology_measured"]
