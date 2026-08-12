"""diag_fft_d4.py — D4: null-spread pilot (PLAN §3).

Question: does the Stage-0 held-out-band gate have discriminative content — do
ALTERNATIVE sign structures, fit to the same data under the same budget, produce
DIFFERENT held-out-band spectra? If the spread is ~zero the gate is vacuous and must be
redesigned before M1 builds machinery around it.

PILOT APPROXIMATION, stated: the alternative structures come from the D5 baseline seeds
(10 seeds -> 10 distinct sign structures, 4 of them Turing-unstable), fit by the CURRENT
baseline objective rather than by a frozen-gate B_train spectral fit (that machinery is
M1). This measures the null spread of "differently-wired models that all explain the
same two scalars" — a lower bound on the discriminative content the true frozen-gate
ensemble will see, because the spectral fit can only pull members closer in-band, not in
the held-out bands they never see.

FIREWALL: opens payload.h5 via data.gate.from_registry (observed frame only; AnswerKey
discarded unread) -> this module is in tests/test_firewall.py::FORBIDDEN.

Usage: .venv/bin/python scripts/diag_fft_d4.py [--n 256] [--runs experiments/diag_fft/d5/runs]
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rngrn.data.gate import from_registry        # noqa: E402
from rngrn.eval.rollout import simulate          # noqa: E402
from rngrn.model import RNGRN                    # noqa: E402
from rngrn import observables as obs             # noqa: E402

torch.set_default_dtype(torch.float64)

# D3-measured band edges (docs/DECISIONS.md D-FFT-9 closure 1)
B_LO, B_HI, B_HARM_HI = 0.60, 1.55, 3.0


def load_model(run_dir: str) -> tuple[RNGRN, dict]:
    ck = torch.load(os.path.join(run_dir, "checkpoints", "model.pt"),
                    map_location="cpu", weights_only=True)
    model = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    model.load_state_dict(ck["state_dict"])
    tr = json.load(open(os.path.join(run_dir, "results", "train_results.json")))
    return model, tr


def raps_shape(field: np.ndarray, L: float, kstar_obs: float,
               band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Band-restricted RAPS shape, normalised by the TRAIN band's total power (SPEC §4:
    normalisation pinned to B_train). Returns (k_centers_in_band, shape)."""
    k, p, _ = obs.raps(field, L)
    train = (k >= B_LO * kstar_obs) & (k <= B_HI * kstar_obs)
    norm = p[train].sum()
    m = (k >= band[0] * kstar_obs) & (k <= band[1] * kstar_obs) & (k > 0)
    return k[m], p[m] / norm


def spec_shape_dist(a: np.ndarray, b: np.ndarray) -> float:
    """SPEC §5 spec_shape form: sum of squared log-power differences."""
    return float(((np.log(a + 1e-300) - np.log(b + 1e-300)) ** 2).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="experiments/diag_fft/d5/runs")
    ap.add_argument("--n", type=int, default=256,
                    help="rollout grid (512 is the reference; cheaper grids are a pilot "
                         "approximation pending D2's grid-fidelity verdict)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/diag_fft/d4")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ri, _key = from_registry("data/datasets", "turing_labyrinth", "sample_0000",
                             3, (0, 1, 2))
    del _key      # AnswerKey quarantined: nothing below reads it
    frame = np.asarray(ri.frame, float)
    L = float(ri.L)
    kstar_obs = float(obs.kstar_of(frame[0], L))

    obs_train_k, obs_train = raps_shape(frame[0], L, kstar_obs, (B_LO, B_HI))
    obs_held_k, obs_held = raps_shape(frame[0], L, kstar_obs, (0.0, B_LO))
    obs_harm_k, obs_harm = raps_shape(frame[0], L, kstar_obs, (B_HI, B_HARM_HI))

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(a.runs, "*"))):
        if not os.path.isdir(run_dir):
            continue
        model, tr = load_model(run_dir)
        turing = bool(tr["metric"]["recovered_turing"])
        row = dict(run=os.path.basename(run_dir), turing=turing,
                   sign=tr["metric"].get("repro_sign_vector"))
        if turing:
            r = simulate(model, L=L, n=a.n, seed=a.seed, integrator="etdrk4_rfft",
                         early_stop=True, check_every=200)
            row.update(patterned=bool(r["patterned"]),
                       stopped_reason=r["stopped_reason"],
                       kstar_sim=float(r["kstar"]) if np.isfinite(r["kstar"]) else None,
                       seconds=r["seconds"])
            if r["patterned"]:
                f0 = r["fields"][0]
                _, sim_train = raps_shape(f0, L, kstar_obs, (B_LO, B_HI))
                _, sim_low = raps_shape(f0, L, kstar_obs, (0.0, B_LO))
                _, sim_harm = raps_shape(f0, L, kstar_obs, (B_HI, B_HARM_HI))
                # bin counts can differ between 512^2 frame and n^2 rollout ONLY if the
                # k-mask differs; same L => same dk => same bins inside the band.
                if len(sim_train) != len(obs_train):
                    raise RuntimeError(
                        f"band bin mismatch: sim {len(sim_train)} vs obs "
                        f"{len(obs_train)} — same L must give same bins")
                row.update(
                    d_train=spec_shape_dist(sim_train, obs_train),
                    d_low=spec_shape_dist(sim_low, obs_low_safe(sim_low, obs_held)),
                    d_harm=spec_shape_dist(sim_harm, obs_harm),
                )
        rows.append(row)
        print(row)

    held = [r for r in rows if r.get("d_harm") is not None]
    d_harm = [r["d_harm"] for r in held]
    d_train = [r["d_train"] for r in held]
    d_held = [r["d_harm"] + r["d_low"] for r in held]
    def spread(v):
        return (max(v) - min(v)) if len(v) >= 2 else float("nan")
    report = dict(kstar_obs=kstar_obs, L=L, n=a.n, b_lo=B_LO, b_hi=B_HI,
                  n_models=len(rows), n_patterned=len(held), rows=rows,
                  train_spread=spread(d_train), held_spread=spread(d_held),
                  harm_spread=spread(d_harm))
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nD4 pilot: {len(held)} patterned alternative structures; "
          f"train-band distances {sorted(round(x, 3) for x in d_train)}; "
          f"held-out distances {sorted(round(x, 3) for x in d_held)}; "
          f"held-out spread {report['held_spread']:.3f} "
          f"(vacuous-gate warning if ~0)")
    return 0


def obs_low_safe(sim_low: np.ndarray, obs_held: np.ndarray) -> np.ndarray:
    if len(sim_low) != len(obs_held):
        raise RuntimeError(f"B_low bin mismatch: sim {len(sim_low)} vs obs "
                           f"{len(obs_held)}")
    return obs_held


if __name__ == "__main__":
    sys.exit(main())
