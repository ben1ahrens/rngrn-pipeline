#!/usr/bin/env python
"""canon_annotate.py — write the MEASURED morphology into the canonical payloads.

The generator's stored ``morphology`` attribute is kept exactly as it was, so no existing
number changes meaning. Alongside it each sample gains ``morphology_measured`` and the
statistics behind it, so a consumer can see what the frame actually is rather than trusting
a label whose rule conflates holes with labyrinths (see docs/DECISIONS.md D-CANON-5).

This is an attribute write. It does not touch a single pixel and does not re-simulate.
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import phase_topology as PT                                        # noqa: E402
import gen_tg3 as G                                                # noqa: E402

CANONICAL = ("turing_spots", "turing_labyrinth")


def annotate(dataset_id, datasets_root):
    path = os.path.join(datasets_root, dataset_id, "payload.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no payload for {dataset_id} at {path}")
    rows = []
    with h5py.File(path, "r+") as f:
        for key in sorted(f.keys()):
            g = f[key]
            frame0 = np.asarray(g["final_frame"])[0].astype(float)
            m = PT.measure(frame0)
            for k, v in m.items():
                g.attrs[k] = v
            # Rewrite k_star_fft at FULL precision. It is a measurement of the stored frame,
            # so recomputing it is exact — no re-simulation involved. The generator rounded
            # it to 3 dp, which knocks `k_star_fft*L/2pi` off the half-integer bin grid by
            # ~2*0.0005*L/(2*pi); harmless at the legacy L <= 220, but 0.087 bins at the
            # L ~ 990 this corpus reaches, which broke test_kstar_fft_lands_on_the_half_bin_grid.
            g.attrs["k_star_fft"] = float(G.raps_dominant_k(frame0, float(g.attrs["L"])))
            rows.append((key, str(g.attrs["morphology"]), m))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-root",
                    default=os.path.join(HERE, "..", "data", "datasets"))
    a = ap.parse_args(argv)
    for ds in CANONICAL:
        print(f"\n{ds}")
        print(f"  {'sample':13s}{'stored':11s}{'measured':11s}{'bright n':>9s}"
              f"{'circ':>7s}{'dark n':>8s}{'circ':>7s}")
        for key, stored, m in annotate(ds, a.datasets_root):
            flag = "" if stored == m["morphology_measured"] else "   <- differs"
            print(f"  {key:13s}{stored:11s}{m['morphology_measured']:11s}"
                  f"{m['bright_domains']:9d}{m['bright_circularity']:7.2f}"
                  f"{m['dark_domains']:8d}{m['dark_circularity']:7.2f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
