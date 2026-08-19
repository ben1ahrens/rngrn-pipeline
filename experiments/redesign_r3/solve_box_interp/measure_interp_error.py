"""Measure the TARGET-INTERPOLATION ERROR of the adaptive commensurate solve box.

`docs/REDESIGN_rngrn.md` §4.3 states the gap this closes:

    "Interpolation error is *expected* far below the ~31 %/bin floor; it is UNMEASURED
     and is measured once per re-tile geometry at R3, alongside the FD check, before any
     spectral gradient is consumed."

This is that measurement (PLAN_redesign_R3 Task 11 Step 5). CPU only; no GPU, no training,
no solve. Run:

    .venv/bin/python experiments/redesign_r3/solve_box_interp/measure_interp_error.py

WHAT IS MEASURED. `solve_box.interpolate_targets` LINEARLY interpolates the observed
log-RAPS, sampled on the DATA box's bins (spacing dk_obs = 2*pi/L), onto the solve box's
B_train bin centres, which sit at k/k*_obs = (j+0.5)/p. The error is the difference between
that interpolant and the log-RAPS the observed spectrum would have AT those k. The true
value there is not observable — the data box does not resolve it — so two complementary
estimators are computed, both from the observed spectrum itself:

  (A) CURVATURE BOUND (primary, one-way). Linear interpolation of a twice-differentiable
      f on an interval of width h, at fraction t of the way across, has error
      |t(1-t)/2 * h^2 f''|, exactly (Lagrange remainder, with f'' at some interior point).
      h^2 f'' is estimated by the observed log-RAPS' second difference on the bracketing
      node. This is an ESTIMATE of the one-way error at each target position, using the
      only curvature information the data carries; it is exact for a quadratic and is the
      standard interpolation-error expression, not a proxy.

  (B) ROUND TRIP (cross-check, upper bound). Resample observed -> solve-box bins ->
      observed, and compare with the original. It applies the resampling TWICE, so it
      over-states the one-way error; it is reported because it uses no smoothness
      assumption at all.

Both are converted to a RELATIVE POWER error, |exp(err_log) - 1|, so they are directly
comparable to D3's ~31 %/bin RAPS estimation floor (`docs/DIAGNOSTICS_fft.md`).

FIREWALL. Only `final_frame` and the `L` attribute are read from `payload.h5` — the
RecoveryInput pair, `docs/SPEC_fourier_training.md` §7. `data.gate` is deliberately NOT
imported (it returns the AnswerKey alongside), and k*_obs is MEASURED from the frame by
`observables.raps`, never taken from the file's `k_star`/`k_star_fft` attributes. This file
lives under `experiments/`, which is not on `sys.path` for the test suite, so it is not
importable by bare top-level name the way `scripts/` modules are and does not need a
`tests/test_firewall.py::FORBIDDEN` entry (CLAUDE.md §5).
"""
from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import torch

from rngrn import solve_box as sb
from rngrn.losses.spectral import SpectralConfig, raps_torch
from rngrn.observables import raps

torch.set_default_dtype(torch.float64)

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent / "results"
DATASETS = ("turing_labyrinth", "turing_spots")
CFG = SpectralConfig()

#: D3's measured ~31 %/bin RAPS estimation floor (`docs/DIAGNOSTICS_fft.md`), the bar
#: §4.3 names for this measurement. NOT a threshold this file calibrates — it is quoted.
FLOOR = 0.31


def observed_spectrum(frame: np.ndarray, L: float):
    """(x_obs, log_power_obs, kstar_obs) for channel 0, with non-positive bins dropped.

    The spectrum is the TRAINING-side estimator (`losses.spectral.raps_torch`, F-D6-1 edges)
    so the measurement is of the interpolation the training path actually performs. k*_obs
    is `observables.raps`'s power-weighted centroid, the repo's k* measurement.
    """
    ch0 = torch.from_numpy(np.ascontiguousarray(frame[0], dtype=np.float64))
    k_centers, power = raps_torch(ch0, L)
    p = power.numpy()
    keep = p > 0.0                      # the DC bin is forced to 0; log is -inf there
    kstar_obs = float(raps(frame[0], L)[2])
    return k_centers[keep] / kstar_obs, np.log(p[keep]), kstar_obs


def curvature_estimate(x_obs, log_obs, x_want):
    """(A) per-target |t(1-t)/2 * h^2 f''| in log units, from the observed second difference.

    Assumes a uniform observed grid in x (it is: bins of width dk, scaled by 1/k*_obs).
    """
    h = float(np.mean(np.diff(x_obs)))
    d2 = np.diff(log_obs, 2)                          # h^2 f'' at nodes 1..n-2
    j = np.searchsorted(x_obs, x_want) - 1            # bracketing left node
    if j.min() < 1 or j.max() + 1 > len(x_obs) - 2:
        raise ValueError("target positions reach the ends of the observed grid, where no "
                         "second difference exists — widen the observed support")
    t = (x_want - x_obs[j]) / h
    return np.abs(t * (1.0 - t) / 2.0 * d2[j - 1])


def round_trip_error(x_obs, log_obs, x_want, target):
    """(B) resample back onto the observed bins inside the solve-box span, in log units."""
    inside = (x_obs >= x_want[0]) & (x_obs <= x_want[-1])
    back = np.interp(x_obs[inside], x_want, target)
    return np.abs(back - log_obs[inside])


def rel_power(err_log: np.ndarray) -> np.ndarray:
    """log-units error -> relative POWER error, comparable to D3's ~31 %/bin floor."""
    return np.abs(np.expm1(err_log))


def main() -> None:
    rows = []
    for ds in DATASETS:
        path = REPO / "data" / "datasets" / ds / "payload.h5"
        with h5py.File(path, "r") as f:
            for key in sorted(f.keys()):
                g = f[key]
                L = float(g.attrs["L"])
                x_obs, log_obs, kstar_obs = observed_spectrum(g["final_frame"][:], L)

                # k̂ is irrelevant to the SCALE-FREE bin positions (they are (j+0.5)/p
                # whatever k̂ is), so the ideal-converged k̂ = k*_obs is used; the assertion
                # below pins that independence rather than assuming it.
                box = sb.geometry(kstar_obs)
                x_want = sb.bin_centers_scalefree(box)[sb.band_bins(box)]
                target = sb.interpolate_targets(log_obs, x_obs * kstar_obs, kstar_obs, box)
                alt = sb.interpolate_targets(log_obs, x_obs * kstar_obs, kstar_obs,
                                             sb.geometry(kstar_obs * 1.37))
                assert torch.equal(target, alt), "target depends on k̂ — it must not"

                curv = rel_power(curvature_estimate(x_obs, log_obs, x_want))
                trip = rel_power(round_trip_error(x_obs, log_obs, x_want, target.numpy()))
                in_band = int(((x_obs >= CFG.b_lo) & (x_obs <= CFG.b_hi)).sum())
                rows.append(dict(
                    dataset=ds, sample=key, L=L, kstar_obs=kstar_obs,
                    data_periods=float(kstar_obs * L / (2.0 * np.pi)),
                    p=box["p"], n=box["n"], L_solve=box["L_solve"],
                    n_band_bins=int(len(x_want)),
                    # How many bins the DATA box puts in B_train. Above 7 the solve box is
                    # COARSER than the data box in the band, which the round trip pays for
                    # (it decimates and reconstructs); training never does that — it
                    # compares two 7-bin vectors.
                    n_band_bins_observed=in_band,
                    curvature_max_rel_power=float(curv.max()),
                    curvature_median_rel_power=float(np.median(curv)),
                    curvature_bins_over_floor=int((curv > FLOOR).sum()),
                    roundtrip_max_rel_power=float(trip.max()),
                    roundtrip_median_rel_power=float(np.median(trip)),
                ))

    summary = dict(
        n_samples=len(rows),
        datasets=list(DATASETS),
        p=sb.P_DEFAULT, n=sb.N_DEFAULT,
        band=[CFG.b_lo, CFG.b_hi],
        curvature_worst_rel_power=max(r["curvature_max_rel_power"] for r in rows),
        curvature_median_of_medians=float(np.median(
            [r["curvature_median_rel_power"] for r in rows])),
        roundtrip_worst_rel_power=max(r["roundtrip_max_rel_power"] for r in rows),
        roundtrip_median_of_medians=float(np.median(
            [r["roundtrip_median_rel_power"] for r in rows])),
        d3_estimation_floor_rel_power=FLOOR,
        curvature_bins_over_floor=sum(r["curvature_bins_over_floor"] for r in rows),
        band_bins_total=sum(r["n_band_bins"] for r in rows),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "interp_error.json").write_text(
        json.dumps(dict(summary=summary, rows=rows), indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    for r in rows:
        print(f"{r['dataset']}/{r['sample']}  periods={r['data_periods']:6.3f}  "
              f"curv max={r['curvature_max_rel_power']:8.2%} med="
              f"{r['curvature_median_rel_power']:8.2%}  "
              f"trip max={r['roundtrip_max_rel_power']:8.2%} med="
              f"{r['roundtrip_median_rel_power']:8.2%}")


if __name__ == "__main__":
    main()
