"""measure_frame_bias.py — spec §3.2 precondition: frame-mean bias on the target.

REDESIGN_rngrn.md §3.2 pins the model's homogeneous fixed point to the observed frame
mean per channel (x*_i <- mean of observed channel i), on the strength of a legacy
measurement (exp12, `docs/STATE_OF_THE_SCIENCE.md` §2.8: `frame.mean()` tracks the true
x*_0 at Spearman rho=0.95, ratio median 0.921, n=127 `three_gene` samples). That figure
is from a DIFFERENT dataset family than the R2 training target. This script measures the
same ratio, per channel, on `turing_labyrinth/sample_0000` directly, so the R2 pinning
decision is not extrapolated from three_gene onto turing_labyrinth.

SCORING-SIDE BY CONSTRUCTION: it reads the AnswerKey's x_star (ground truth). It must
never be imported by recovery-side code — hence its module name `measure_frame_bias` is
listed in `tests/test_firewall.py::FORBIDDEN` (CLAUDE.md §5; scripts/ is outside the
completeness test's glob and is importable by bare name because the suite and notebooks
put scripts/ on sys.path).

Usage (payload.h5 reads are sandbox-denied — run unsandboxed):
    KMP_AFFINITY=disabled OMP_NUM_THREADS=1 .venv/bin/python scripts/measure_frame_bias.py

Writes experiments/redesign_r2/frame_bias/results/bias.json and prints a summary.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rngrn.data.gate import from_registry   # noqa: E402
from rngrn import observables as obs        # noqa: E402

DATASETS_ROOT = "data/datasets"
DATASET_ID = "turing_labyrinth"
SAMPLE_KEY = "sample_0000"
OUT_DIR = os.path.join("experiments", "redesign_r2", "frame_bias", "results")


def main() -> int:
    ri, key = from_registry(DATASETS_ROOT, DATASET_ID, SAMPLE_KEY, 3, (0, 1, 2))
    frame, L = np.asarray(ri.frame), float(ri.L)
    if key.x_star is None:
        raise RuntimeError(
            f"{DATASET_ID}/{SAMPLE_KEY}: AnswerKey.x_star is missing — cannot measure "
            "the frame-mean bias without the generator's true steady state.")
    x_star = np.asarray(key.x_star)
    if frame.shape[0] != x_star.shape[0]:
        raise RuntimeError(
            f"channel count mismatch: frame has {frame.shape[0]} observed channels, "
            f"x_star has {x_star.shape[0]}.")

    ratios = [float(frame[c].mean() / x_star[c]) for c in range(frame.shape[0])]
    kstar_obs = float(obs.kstar_of(frame[0], L))

    results = {
        "dataset": DATASET_ID,
        "sample": SAMPLE_KEY,
        "ratio_mean_over_xstar": ratios,
        "kstar_obs": kstar_obs,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "bias.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"measure_frame_bias — {DATASET_ID}/{SAMPLE_KEY} (L={L:.4f})")
    for c, r in enumerate(ratios):
        flag = "" if 0.8 <= r <= 1.25 else "  ** OUTSIDE [0.8, 1.25] **"
        print(f"  channel {c}: frame.mean() = {frame[c].mean():.6f}, "
              f"x_star = {x_star[c]:.6f}, ratio = {r:.6f}{flag}")
    print(f"  kstar_obs (channel 0) = {kstar_obs:.6f}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
