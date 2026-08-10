#!/usr/bin/env python
"""canon_select.py — choose the systems that become the canonical Turing datasets.

Selection, not generation. Every decision here is deterministic given a seed, so the
canonical set is reproducible from the corpus rather than hand-picked.

WHY MARGIN AND NOT JUST THE LABEL
---------------------------------
The morphology class boundaries are uncalibrated. ``A > 0.55`` (the stripe test) appears
exactly ONCE in the whole repo, with no control and no test behind it, and ``phi < 0.34``
sits on the Gaussian null (P(z>0.4) = 0.3446), so it separates "positively skewed" from
"not" rather than "spots" from "not spots". Worse, the angular features behind ``A`` are
NOT L-invariant: a measured control drifts ``ang_conc`` 0.190 -> 0.074 across an 8x L span,
enough to flip a class label. So a sample sitting near a boundary carries a label that is
partly an artefact of its box size. Admitting only high-margin, label-stable samples is
what makes the three strata mean something.

WHY THE PERIODICITY GATE IS LOAD-BEARING, NOT A NICETY
------------------------------------------------------
Ranking by margin ALONE selects ``three_gene_qvar/sample_0032`` first among spots, because
its area fraction is 0.032 — the largest possible distance below the 0.34 cut. It is also
the one sample in the corpus that is not a Turing pattern at all: three isolated blobs,
species 1 and 2 flat to cv=0.002, and a monotonically decaying spectrum with no interior
peak. The ``peak_bin >= 3`` gate is the only thing between it and the canonical set.

FIREWALL POSITION
-----------------
Data side. This reads ``payload.h5`` — generating kinetics, the generator's ``k_star`` —
which CLAUDE.md section 5 forbids to recovery-side code. It lives in ``scripts/`` and is
imported only by ``canon_generate.py`` and its own test. It must never be imported from
``model.py``, ``observables.py``, ``recover.py``, ``losses/`` or ``eval/``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from gen_tg3 import stable_seed                                    # noqa: E402
from td_figures import PEAK_BIN_MIN                                # noqa: E402

# The generator's own class boundaries (scripts/gen_tg3.py::classify). Reproduced here so
# margins are measured against the SAME cuts that produced the stored labels; this module
# must not invent a second taxonomy.
PHI_SPOTS_MAX = 0.34
PHI_HOLES_MIN = 0.66
ANISO_STRIPES_MIN = 0.55

# Contrast floor. The generator rejects below 0.05 as "collapsed to homogeneous"; 0.30 is
# far above that, chosen so a canonical exemplar is unambiguous rather than merely legal.
CV_MIN = 0.30

# Periods-per-box for the 512 grid. px/wavelength = 512/p, so this range gives 16..32
# px/wavelength — every sample far above the 6 px/wavelength floor measured in D15 — while
# the k* half-bin precision 1/(2p) lands in 1.6%..3.1% (vs 8.3% on the legacy data).
P_CHOICES = tuple(range(16, 33))

CLASSES = ("spots", "stripes", "labyrinth")


def class_margin(morphology, phi, anisotropy):
    """Signed distance from the nearest boundary that defines this class.

    Positive means "inside the class with room to spare"; negative means the sample is
    actually on the wrong side of a cut and its stored label is wrong.
    """
    phi = float(phi)
    a = float(anisotropy)
    if morphology == "spots":
        return PHI_SPOTS_MAX - phi
    if morphology == "holes":
        return phi - PHI_HOLES_MIN
    if morphology == "stripes":
        return min(a - ANISO_STRIPES_MIN, phi - PHI_SPOTS_MAX, PHI_HOLES_MIN - phi)
    if morphology == "labyrinth":
        return min(ANISO_STRIPES_MIN - a, phi - PHI_SPOTS_MAX, PHI_HOLES_MIN - phi)
    raise ValueError(f"unknown morphology {morphology!r}; expected one of "
                     f"spots/holes/stripes/labyrinth")


def passes_gates(row):
    """Admission gates, applied before any ranking."""
    return bool(row["peak_bin"] >= PEAK_BIN_MIN
                and row["cv"] >= CV_MIN
                and class_margin(row["morphology"], row["area_frac"], row["anisotropy"]) > 0)


def draw_periods(dataset_id, n, seed):
    """``n`` DISTINCT periods-per-box for one dataset.

    Distinctness is the point: with a single p, ``k* = p*2pi/L`` inverts exactly and the
    domain size becomes the label again — the leak that made ``kstar_rel_err`` a gate
    rather than evidence on the legacy data. Seeded via SHA-256 so the draw is
    process-independent (Python salts ``hash()`` per process, which is why the legacy seeds
    do not reproduce their own screens).
    """
    rng = np.random.default_rng(stable_seed(f"{seed}:{dataset_id}"))
    return sorted(int(x) for x in rng.choice(P_CHOICES, size=n, replace=False))


# ======================================================================================
# the candidate table
# ======================================================================================
# Only these two datasets carry params_json AND sim_seed, which is what re-simulation at a
# new resolution requires. The 127 legacy three_gene_{train,val,test} samples carry
# neither — their generator lived in a gitignored tree and their kinetics are gone — and
# the *_classical_* families carry kinetics but no seed and are not 3-gene GRNs.
ELIGIBLE_DATASETS = ("three_gene_qvar", "three_gene_multiL")

CANDIDATE_FIELDS = ("source_dataset", "source_key", "system_id", "morphology", "area_frac",
                    "anisotropy", "cv", "peak_bin", "margin", "k_star", "L", "uid")


def row_uid(row):
    """Stable identity for a SYSTEM (kinetics), independent of which replicate we read."""
    return f"{row['source_dataset']}:{int(row['system_id'])}"


def candidate_table(datasets_root=None):
    """One row per distinct system across the eligible datasets.

    ``three_gene_multiL`` holds 23 systems x 4 domain sizes; we keep the first replicate as
    the representative and use the other three for the free label-stability check.
    """
    import td_figures as TD

    rows = []
    for ds in ELIGIBLE_DATASETS:
        seen = set()
        for s in TD.load_samples(ds, datasets_root):
            if not TD.is_resimulatable(s):
                raise ValueError(f"{ds}/{s['key']} is not re-simulatable but {ds} is listed "
                                 f"as eligible — the eligibility list is wrong")
            sid = int(s["attrs"]["system_id"])
            if sid in seen:
                continue
            seen.add(sid)
            v = TD.patterning_verdict(s["final_frame"][0], s["L"])
            r = {
                "source_dataset": ds,
                "source_key": s["key"],
                "system_id": sid,
                "morphology": str(s["morphology"]),
                "area_frac": float(s["attrs"]["area_frac"]),
                "anisotropy": float(s["attrs"]["anisotropy"]),
                "cv": float(v["cv"]),
                "peak_bin": int(v["peak_bin"]),
                "k_star": float(s["k_star"]),
                "L": float(s["L"]),
            }
            r["margin"] = class_margin(r["morphology"], r["area_frac"], r["anisotropy"])
            r["uid"] = row_uid(r)
            rows.append(r)
    return rows


# ======================================================================================
# label stability across domain size
# ======================================================================================
def native_periods(row):
    """Periods-per-box the sample was generated at: ``p = L*k*/(2*pi)``."""
    return int(round(float(row["L"]) * float(row["k_star"]) / (2 * np.pi)))


def stability_probe_p(row):
    """A second periods-per-box at which to re-simulate, well separated from the native one.

    Probing at the sample's own p would re-run an identical simulation and prove nothing.
    Stays inside the original generator's {3..14} window so the probe is a box size the
    screening actually certified as feasible for this system.
    """
    native = native_periods(row)
    candidates = [p for p in (native + 5, native - 5, native + 3, native - 3)
                  if 3 <= p <= 14 and p != native]
    if not candidates:
        raise ValueError(f"no valid stability probe for native p={native}")
    return candidates[0]


def label_is_stable(row, probe_labels):
    """Does the stored class survive every probe?"""
    if not probe_labels:
        raise ValueError("no probe labels supplied; stability cannot be asserted")
    return all(str(x) == str(row["morphology"]) for x in probe_labels)


def multiL_labels(system_id, datasets_root=None):
    """Labels of all four replicates of one multiL system.

    Free stability evidence: multiL already simulated each system at p in {4,7,10,13}, so
    for those 23 systems the probe costs no compute at all.
    """
    import td_figures as TD
    return [str(s["morphology"]) for s in TD.load_samples("three_gene_multiL", datasets_root)
            if int(s["attrs"]["system_id"]) == int(system_id)]
