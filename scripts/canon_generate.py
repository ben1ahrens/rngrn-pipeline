#!/usr/bin/env python
"""canon_generate.py — promote selected systems to 512x512 and write canonical payloads.

Generation, not selection. ``canon_select.py`` decides WHICH systems and at WHAT box size;
this module re-runs them at high resolution and writes the registry payload.

WHY 512 AND NOT 1024
--------------------
Recovery itself is nearly pixel-count-free (every loss term lives on NxN objects, not on
the image), so a big frame is cheap to FIT. The cost is the post-hoc ETDRK4 morphology
rollout, which scales purely with pixel count because its step count is reaction-rate
limited. Measured: ~45 s typical and ~17 min worst case at 512, against ~3 min typical and
~70+ min worst case at 1024 with ``eval/numerics._phi_contour`` peaking at 4-6 GB — on a
host that CLAUDE.md section 7a records as having been OOM-killed five times. Separately,
``eval/lgen_eval.grid_for_L`` refuses grids above 512 by design.

WHY THE PHYSICS IS IMPORTED, NOT REIMPLEMENTED
----------------------------------------------
Everything numerical comes from ``gen_tg3``. A second copy of the spectral-IMEX integrator
would drift from the one that produced every existing dataset, and then no number from the
canonical sets would be comparable with any number from the corpus.
"""
from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import gen_tg3 as G                                                # noqa: E402

GRID = 512
CV_EVERY = 200

# Saturation tolerance: the pattern must have stopped changing by the end of the run.
# UNCALIBRATED in the strict sense — it is a convergence tolerance, not a threshold
# separating two populations, and no control was run against it. It is enforced as a
# fail-loud gate rather than used to judge anything.
SATURATION_TOL = 0.01


def is_saturated(cv_trace, tol=SATURATION_TOL):
    """Had the pattern stopped changing by the end of the run?

    Compares the final cv against its value at 80% of the run. ``Tmax = 260`` was chosen
    for 96x96 boxes; a 512 box at p=32 holds far more pattern to organise, and a frame
    captured mid-coarsening is not a steady state.
    """
    cv_trace = np.asarray(cv_trace, float)
    if cv_trace.size < 5:
        raise ValueError(f"cv trace too short to judge saturation ({cv_trace.size} samples)")
    tail = cv_trace[int(0.8 * cv_trace.size):]
    final = float(tail[-1])
    return bool(abs(final - float(tail[0])) / max(abs(final), 1e-12) < tol)


def params_from_sample(sample, periods):
    """Rebuild the generator's parameter dict from a stored sample, at a new box size."""
    a = sample["attrs"]
    p = dict(sample["params"])
    p["_M"] = p["interaction_matrix"]
    p["x_star"] = np.asarray(sample["x_star"])
    p["k_star"] = float(a["k_star"])
    p["sim_seed"] = int(a["sim_seed"])
    p["periods_per_box"] = int(periods)
    return p


def probe_label(sample, periods, grid=96):
    """Re-simulate one sample at a different box size and return its morphology class.

    Cheap (96x96, ~20 s) because it exists only to test whether the LABEL is a property of
    the system or of its box. Returns ``"collapsed"`` rather than raising when the
    re-simulation fails to pattern — for stability purposes that is simply a label that
    differs from the stored one.
    """
    p = params_from_sample(sample, periods)
    out = G.simulate_and_classify(p, grid=grid, seed=p["sim_seed"])
    if out is None:
        return "collapsed"
    return str(out["morphology"])


# ======================================================================================
# the canonical payload
# ======================================================================================
def write_canonical_payload(records, out_path):
    """Write the canonical registry payload: final frame, cv trace, full provenance.

    Deliberately OMITS ``trajectory``. Nothing under ``src/rngrn/`` reads it — the loader
    records only ``frame_shape`` from ``observable_key`` — so at 512x512 it would cost 6x
    the storage for data no consumer touches.

    Deliberately KEEPS all three channels even though a real photograph gives one. That
    makes m=1 vs m=3 a controlled comparison on identical data rather than two datasets.

    ``L`` and ``k_star`` are mandatory: ``gate.from_registry`` raises rather than defaulting
    them, on purpose.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for idx, r in enumerate(records):
            p = r["params"]
            g = f.create_group(f"sample_{idx:04d}")
            g.create_dataset("final_frame", data=r["final"], compression="gzip")
            g.create_dataset("x_star", data=np.asarray(p["x_star"]))
            g.create_dataset("D", data=np.asarray(p["D"]))
            g.create_dataset("interaction_matrix", data=np.asarray(p["_M"]))
            g.create_dataset("jacobian", data=G.jac_fd(
                G.make_reaction(p["_M"], np.asarray(p["b"]), np.asarray(p["V"]),
                                np.asarray(p["mu"]), np.asarray(p["K"]), p["n"],
                                p["reaction"]), np.asarray(p["x_star"])))
            g.create_dataset("cv_trace", data=np.asarray(r["cv_trace"]))
            g.create_dataset("cv_times", data=np.asarray(r["cv_times"]))
            for k in ("L", "dx", "dt_sim", "grid", "cv0", "morphology", "wavelength",
                      "k_star_fft", "area_frac", "n_components", "anisotropy"):
                g.attrs[k] = r[k]
            g.attrs["k_star"] = float(p["k_star"])       # REQUIRED by gate.from_registry
            g.attrs["topology"] = p["topology"]
            g.attrs["reaction"] = p["reaction"]
            g.attrs["n"] = p["n"]
            g.attrs["sim_seed"] = int(p["sim_seed"])
            g.attrs["periods_per_box"] = int(p["periods_per_box"])
            g.attrs["system_id"] = int(r["system_id"])
            g.attrs["source_dataset"] = r["source_dataset"]
            g.attrs["source_key"] = r["source_key"]
            g.attrs["role"] = r["role"]
            g.attrs["params_json"] = json.dumps(
                {**{k: p[k] for k in G.PARAM_KEYS}, "topology": p["topology"],
                 "reaction": p["reaction"], "interaction_matrix": p["_M"]})
    return out_path
