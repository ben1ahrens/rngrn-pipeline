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

# px/wavelength band for the canonical sets. This — not an absolute L window — is the real
# resolution constraint: L enters the physics only as a unit (CLAUDE.md section 7c). The
# floor of 16 sits far above the 6.0 px/wavelength breakdown measured in D15; the ceiling
# of 32 keeps periods-per-box high enough that the k* half-bin precision stays at 1.6-3.1%.
PPW_MIN, PPW_MAX = 16.0, 32.0

# Domain-size window passed through to the generator. The default (18, 220) is tied to the
# 96x96 grid and every canonical sample exceeds it (measured L range 245.8 .. 794.0), which
# is correct rather than alarming: at 512 those give 17-30 px/wavelength. PPW_MIN/PPW_MAX
# above is what actually guards resolution here.
L_BOUNDS = (18.0, 2000.0)

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


# ======================================================================================
# the driver
# ======================================================================================
def _simulate_one(job):
    """Pool worker: re-simulate one selected system at the canonical resolution.

    Module-level and picklable. Returns the result dict plus the provenance fields the
    payload writer needs, or raises with the sample named.
    """
    import time
    s, src, grid = job
    p = params_from_sample(src, s["periods_per_box"])
    t0 = time.time()
    out = G.simulate_and_classify(p, grid=grid, n_traj=2, seed=p["sim_seed"],
                                  cv_every=CV_EVERY, l_bounds=L_BOUNDS)
    elapsed = time.time() - t0
    if out is None:
        raise RuntimeError(
            f"{s['uid']} diverged or collapsed at grid={grid}; it did not at 96 — "
            f"investigate before excluding it")
    if not is_saturated(out["cv_trace"]):
        raise RuntimeError(
            f"{s['uid']} had not saturated by Tmax at grid={grid} (cv still moving > "
            f"{SATURATION_TOL:.0%} over the last 20% of the run). Re-run with a longer "
            f"Tmax; do not ship a transient frame.")
    return {**out, "params": p, "source_dataset": s["source_dataset"],
            "source_key": s["source_key"], "system_id": s["system_id"],
            "role": s["role"], "_elapsed": elapsed, "_uid": s["uid"],
            "_label_before": s["morphology"]}


def build_dataset(ds_id, spec, source_root=None, out_root=None, grid=GRID, verbose=True,
                  procs=1, limit=None):
    """Re-simulate every system in one dataset spec at `grid` and write its payload.

    ``source_root`` is where the CORPUS is read from; ``out_root`` is where the new payload
    is written. They are separate so a smoke run can write to a scratch directory while
    still reading the real corpus.
    """
    from multiprocessing import Pool
    import td_figures as TD

    root = out_root or source_root or os.path.join(HERE, "..", "data", "datasets")
    samples = spec["samples"][:limit] if limit else spec["samples"]

    jobs = []
    for s in samples:
        ppw = grid / s["periods_per_box"]
        if not (PPW_MIN <= ppw <= PPW_MAX):
            raise ValueError(f"{ds_id}/{s['uid']}: {ppw:.1f} px/wavelength outside "
                             f"[{PPW_MIN}, {PPW_MAX}] — check the periods draw, not this gate")
        src = [x for x in TD.load_samples(s["source_dataset"], source_root)
               if x["key"] == s["source_key"]][0]
        jobs.append((s, src, grid))

    if procs > 1:
        with Pool(procs) as pool:
            results = pool.map(_simulate_one, jobs)
    else:
        results = [_simulate_one(j) for j in jobs]

    for r in results:
        if verbose:
            print(f"   {r['_uid']:24s} p={r['params']['periods_per_box']:3d} "
                  f"px/wl={grid / r['params']['periods_per_box']:5.1f} "
                  f"morph {r['_label_before']} -> {r['morphology']:10s} "
                  f"cv={r['cv0']:.2f} ({r['_elapsed']:.0f} s)")
    return write_canonical_payload(results, os.path.join(root, ds_id, "payload.h5"))


def main(argv=None):
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Generate the canonical Turing datasets.")
    ap.add_argument("--selection", default=os.path.join(HERE, "..", "data",
                                                        "canonical_selection.json"))
    ap.add_argument("--source-root", default=None, help="where the CORPUS is read from")
    ap.add_argument("--out-root", default=None, help="where the new payloads are written")
    ap.add_argument("--grid", type=int, default=GRID)
    ap.add_argument("--only", default=None, help="one dataset id, for restartability")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N samples only (debug)")
    a = ap.parse_args(argv)

    sel = json.load(open(a.selection))
    for ds_id, spec in sorted(sel["datasets"].items()):
        if a.only and ds_id != a.only:
            continue
        print(f"\n{ds_id} ({len(spec['samples'])} systems at {a.grid}x{a.grid}):", flush=True)
        t0 = time.time()
        path = build_dataset(ds_id, spec, a.source_root, a.out_root, a.grid,
                             procs=a.procs, limit=a.limit)
        print(f"  wrote {path} ({os.path.getsize(path)/1e6:.1f} MB) "
              f"in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
