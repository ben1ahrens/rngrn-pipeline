"""diag_fft_d3.py — D3: spectral estimator floor and band edges for turing_labyrinth/sample_0000.

PLAN_fourier_training.md §3 D3 / SPEC §4 (bands, normalisation), §9.7 (patch control).
Pure measurement of the OBSERVED frame — no training, no GPU, no RNG. Everything here is
firewall-legal recovery-side arithmetic (statistics of the observed frame + L); this
script nevertheless opens payload.h5 via ``data.gate.from_registry``, so its module name
``diag_fft_d3`` MUST be listed in ``tests/test_firewall.py::FORBIDDEN`` (CLAUDE.md §5 —
the completeness test cannot see scripts/).

The AnswerKey returned by from_registry is discarded unread. k*_obs is re-measured from
the frame with observables.kstar_of, exactly as recover.py does.

Measures:
  1. RAPS of channel 0: full binned spectrum, k*_obs, bin width 2π/L, periods/box p.
  2. Contiguous spectrum support around the RAPS peak at 50%/10%/1% of peak
     (→ candidate b_lo/b_hi band edges, in units of k*_obs).
  3. Harmonic content: RAPS band power in B_harm = [1.5, 3.0]·k*_obs and
     B_low = (0, 0.5·k*_obs] (DC excluded) as fractions of B_train = [0.5, 1.5]·k*_obs,
     both RAPS-bin-summed and 2-D mode-summed; high-k noise floor from the top decade
     of k (k ≥ k_max/10) to answer "is B_harm measurable at 512², p=8?".
  4. Patch control (SPEC §9.7): four 256² quadrants, all 6 pairs —
     scoring.morphology.spectral_distance_2d (scripts MAY import scoring; only
     recovery-side modules may not) and the SPEC §5 spec_shape log-power distance
     restricted to [0.5, 1.5]·k*_obs, each quadrant's RAPS normalised to that band.
  5. Channels 1 and 2 (near-flat): RAPS peak, peak-vs-floor ratio, cv — is a k* even
     measurable there?

Usage (payload.h5 reads are denied inside the Bash sandbox — run unsandboxed):
    KMP_AFFINITY=disabled OMP_NUM_THREADS=1 .venv/bin/python scripts/diag_fft_d3.py

Writes experiments/diag_fft/d3/results.json and prints a summary. Deterministic.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rngrn.data.gate import from_registry            # noqa: E402
from rngrn import observables as obs                 # noqa: E402
from rngrn.scoring.morphology import spectral_distance_2d  # noqa: E402

DATASETS_ROOT = "data/datasets"
DATASET_ID = "turing_labyrinth"
SAMPLE_KEY = "sample_0000"
OUT_DIR = os.path.join("experiments", "diag_fft", "d3")

SUPPORT_FRACTIONS = (0.50, 0.10, 0.01)
B_TRAIN = (0.5, 1.5)      # provisional, SPEC §4 — this diagnostic tests those edges
B_HARM = (1.5, 3.0)
NOISE_FLOOR_TOP_DECADE = 10.0   # floor estimated from bins with k >= k_max / 10


# ---------------------------------------------------------------------- spectral helpers

def power2d(field: np.ndarray, L: float):
    """2-D power spectrum of a square field and the |k| grid, matching observables.raps
    conventions (mean removed, angular wavenumber rad/length)."""
    f = field - field.mean()
    N = f.shape[0]
    P = np.abs(np.fft.fft2(f)) ** 2
    kfreq = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    KX, KY = np.meshgrid(kfreq, kfreq)
    KR = np.sqrt(KX ** 2 + KY ** 2)
    return P, KR


def contiguous_support(kcent: np.ndarray, power: np.ndarray, frac: float, dk: float,
                       kstar: float) -> dict:
    """Contiguous run of bins around the RAPS peak with power >= frac * peak power.

    Bin 0 (DC) is excluded — raps() zeroes it. Edges reported both at bin centers and
    at half-bin-extended edges, absolute and in units of k*_obs."""
    pk = int(np.argmax(power))
    if pk < 1:
        raise RuntimeError("RAPS peak at the DC bin — no measurable spectral peak")
    thr = frac * power[pk]
    lo = pk
    while lo - 1 >= 1 and power[lo - 1] >= thr:
        lo -= 1
    hi = pk
    while hi + 1 <= len(power) - 1 and power[hi + 1] >= thr:
        hi += 1
    k_lo_c, k_hi_c = float(kcent[lo]), float(kcent[hi])
    k_lo_e, k_hi_e = k_lo_c - dk / 2.0, k_hi_c + dk / 2.0
    return {
        "frac_of_peak": frac,
        "bin_lo": lo, "bin_hi": hi, "n_bins": hi - lo + 1,
        "k_lo_center": k_lo_c, "k_hi_center": k_hi_c,
        "k_lo_edge": k_lo_e, "k_hi_edge": k_hi_e,
        "rel_lo_center": k_lo_c / kstar, "rel_hi_center": k_hi_c / kstar,
        "rel_lo_edge": k_lo_e / kstar, "rel_hi_edge": k_hi_e / kstar,
    }


def band_mask(k: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (k >= lo) & (k <= hi)


def raps_shape_distance(pa: np.ndarray, pb: np.ndarray, kcent: np.ndarray,
                        klo: float, khi: float) -> float:
    """SPEC §5 spec_shape form between two RAPS on the same bin grid, restricted to
    [klo, khi]: S = P / sum_band(P), distance = sum_band (log S_a - log S_b)^2, w=1."""
    m = band_mask(kcent, klo, khi)
    if not m.any():
        raise RuntimeError(f"band [{klo}, {khi}] contains no RAPS bins")
    a, b = pa[m], pb[m]
    if (a <= 0).any() or (b <= 0).any():
        raise RuntimeError("non-positive RAPS power inside the shape band — log undefined")
    Sa, Sb = a / a.sum(), b / b.sum()
    return float(np.sum((np.log(Sa) - np.log(Sb)) ** 2))


def band_powers(kcent: np.ndarray, power: np.ndarray, P2: np.ndarray, KR: np.ndarray,
                kstar: float) -> dict:
    """RAPS-bin-summed and 2-D mode-summed power in B_train, B_harm, B_low.

    RAPS bins are annulus MEANS (per-mode averages), so the bin-sum and the mode-sum
    weight k differently; both are reported. B_low excludes the DC bin/mode (raps()
    zeroes bin 0 already; the 2-D sum masks k > 0)."""
    bands = {
        "B_train": (B_TRAIN[0] * kstar, B_TRAIN[1] * kstar),
        "B_harm": (B_HARM[0] * kstar, B_HARM[1] * kstar),
        "B_low": (0.0, B_TRAIN[0] * kstar),
    }
    out = {}
    for name, (lo, hi) in bands.items():
        m1 = band_mask(kcent, lo, hi)
        if name == "B_low":
            m1 &= np.arange(len(kcent)) >= 1        # exclude DC bin
        m2 = (KR >= lo) & (KR <= hi) & (KR > 0)
        if not m1.any():
            raise RuntimeError(f"{name} contains no RAPS bins — band empty at this grid")
        out[name] = {
            "k_lo": lo, "k_hi": hi, "n_raps_bins": int(m1.sum()),
            "n_modes_2d": int(m2.sum()),
            "raps_sum": float(power[m1].sum()),
            "raps_peak_bin_value": float(power[m1].max()),
            "mode_sum_2d": float(P2[m2].sum()),
        }
    return out


def noise_floor(kcent: np.ndarray, power: np.ndarray) -> dict:
    """Estimator floor from the top decade of k: bins with k >= k_max / 10."""
    kmax = kcent[-1]
    m = kcent >= kmax / NOISE_FLOOR_TOP_DECADE
    m &= np.arange(len(kcent)) >= 1
    if m.sum() < 10:
        raise RuntimeError("top-decade floor window has <10 bins — grid too coarse")
    vals = power[m]
    return {
        "k_min_of_window": float(kcent[m][0]), "k_max": float(kmax),
        "n_bins": int(m.sum()),
        "median": float(np.median(vals)), "mean": float(vals.mean()),
        "p90": float(np.percentile(vals, 90)),
    }


# ------------------------------------------------------------------------- patch control

def quadrants(field: np.ndarray) -> list:
    n = field.shape[0]
    if n % 2 != 0:
        raise RuntimeError(f"field side {n} not even — cannot quadrant")
    h = n // 2
    return [field[:h, :h], field[:h, h:], field[h:, :h], field[h:, h:]]


def patch_control(field: np.ndarray, L: float, kstar: float) -> dict:
    """SPEC §9.7: pairwise distances between the four quadrants of one channel.

    Quadrants are (H/2)² on a box of extent L/2, so their RAPS bin width is 2·(2π/L);
    the shape band [0.5, 1.5]·k*_obs is in absolute rad/length and selects whichever
    quadrant bins fall inside it."""
    qs = quadrants(field)
    L_q = L / 2.0
    raps_q = [obs.raps(q, L_q) for q in qs]
    kcent_q = raps_q[0][0]
    d2d, dshape = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            d2d.append(spectral_distance_2d(qs[i], qs[j]))
            dshape.append(raps_shape_distance(raps_q[i][1], raps_q[j][1], kcent_q,
                                              B_TRAIN[0] * kstar, B_TRAIN[1] * kstar))
    m = band_mask(kcent_q, B_TRAIN[0] * kstar, B_TRAIN[1] * kstar)
    return {
        "n_pairs": len(d2d),
        "quadrant_raps_bin_width": float(2 * np.pi / L_q),
        "n_shape_band_bins": int(m.sum()),
        "spectral_distance_2d": {
            "pairs": [float(v) for v in d2d],
            "mean": float(np.mean(d2d)), "min": float(np.min(d2d)),
            "max": float(np.max(d2d)),
        },
        "raps_shape_distance": {
            "pairs": [float(v) for v in dshape],
            "mean": float(np.mean(dshape)), "min": float(np.min(dshape)),
            "max": float(np.max(dshape)),
        },
    }


# ------------------------------------------------------------------------------ per-channel

def channel_report(field: np.ndarray, L: float, kstar_ch0: float) -> dict:
    """RAPS peak, floor, cv for one channel; measurability of a k* there."""
    kcent, power, kstar = obs.raps(field, L)
    fl = noise_floor(kcent, power)
    pk = int(np.argmax(power))
    peak_over_floor = float(power[pk] / fl["median"]) if fl["median"] > 0 else float("inf")
    return {
        "kstar_of": float(kstar),
        "kstar_rel_to_ch0": float(kstar / kstar_ch0),
        "peak_bin": pk,
        "peak_k": float(kcent[pk]),
        "peak_power": float(power[pk]),
        "noise_floor": fl,
        "peak_over_floor_median": peak_over_floor,
        "mean": float(field.mean()),
        "std": float(field.std()),
        "cv": float(field.std() / field.mean()),
    }


# --------------------------------------------------------------------------------- main

def main() -> int:
    ri, _key = from_registry(DATASETS_ROOT, DATASET_ID, SAMPLE_KEY, 3, (0, 1, 2))
    del _key   # AnswerKey quarantined: nothing below reads it (firewall discipline)
    frame, L = np.asarray(ri.frame), float(ri.L)
    if frame.shape != (3, 512, 512):
        raise RuntimeError(f"expected (3, 512, 512) frame, got {frame.shape}")

    # ---- 1/2. channel 0 RAPS, k*_obs, support bands
    ch0 = frame[0]
    kcent, power, kstar = obs.raps(ch0, L)
    dk = 2 * np.pi / L
    p_periods = kstar / dk
    pk = int(np.argmax(power))
    support = [contiguous_support(kcent, power, f, dk, kstar) for f in SUPPORT_FRACTIONS]

    # ---- 3. band powers and noise floor
    P2, KR = power2d(ch0, L)
    bands = band_powers(kcent, power, P2, KR, kstar)
    fl = noise_floor(kcent, power)
    frac_harm_raps = bands["B_harm"]["raps_sum"] / bands["B_train"]["raps_sum"]
    frac_low_raps = bands["B_low"]["raps_sum"] / bands["B_train"]["raps_sum"]
    frac_harm_2d = bands["B_harm"]["mode_sum_2d"] / bands["B_train"]["mode_sum_2d"]
    frac_low_2d = bands["B_low"]["mode_sum_2d"] / bands["B_train"]["mode_sum_2d"]
    harm_peak_over_floor = bands["B_harm"]["raps_peak_bin_value"] / fl["median"]

    # ---- b_lo/b_hi proposal: the 10%-of-peak edges (half-bin-extended), SPEC §9.8
    s10 = next(s for s in support if s["frac_of_peak"] == 0.10)
    proposal = {
        "b_lo": s10["rel_lo_edge"], "b_hi": s10["rel_hi_edge"],
        "basis": "contiguous 10%-of-peak support, half-bin-extended edges, in k*_obs units",
        "alternatives": {
            "50pct": [s["rel_lo_edge"] for s in support if s["frac_of_peak"] == 0.50][0:1]
                     + [s["rel_hi_edge"] for s in support if s["frac_of_peak"] == 0.50][0:1],
            "1pct": [s["rel_lo_edge"] for s in support if s["frac_of_peak"] == 0.01][0:1]
                    + [s["rel_hi_edge"] for s in support if s["frac_of_peak"] == 0.01][0:1],
            "provisional_spec": list(B_TRAIN),
        },
    }

    # ---- 5. patch control, channel 0
    patch0 = patch_control(ch0, L, kstar)

    # ---- 6. channels 1, 2 (+ their patch controls, same band, for completeness)
    ch_reports, ch_patches = {}, {}
    for c in (1, 2):
        ch_reports[f"channel_{c}"] = channel_report(frame[c], L, kstar)
        ch_patches[f"channel_{c}"] = patch_control(frame[c], L, kstar)

    results = {
        "diagnostic": "D3",
        "dataset": f"{DATASET_ID}/{SAMPLE_KEY}",
        "grid": [512, 512], "L": L,
        "channel_0": {
            "kstar_obs": float(kstar),
            "bin_width_dk": float(dk),
            "kstar_in_bins_p": float(p_periods),
            "peak_bin": pk,
            "peak_power": float(power[pk]),
            "cv": float(ch0.std() / ch0.mean()),
            "raps_kcent": [float(v) for v in kcent],
            "raps_power": [float(v) for v in power],
            "support_bands": support,
            "band_edge_proposal": proposal,
            "band_powers": bands,
            "noise_floor_top_decade": fl,
            "fractions": {
                "harm_over_train_raps": float(frac_harm_raps),
                "low_over_train_raps": float(frac_low_raps),
                "harm_over_train_2d": float(frac_harm_2d),
                "low_over_train_2d": float(frac_low_2d),
                "harm_peak_bin_over_floor_median": float(harm_peak_over_floor),
            },
            "patch_control": patch0,
        },
        "channels_1_2": ch_reports,
        "channels_1_2_patch_control": ch_patches,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # ------------------------------------------------------------------ human summary
    print(f"D3 — spectral estimator floor and band edges — {DATASET_ID}/{SAMPLE_KEY}")
    print(f"  grid 512x512, L = {L:.4f}, dk = 2pi/L = {dk:.6f} rad/len")
    print(f"  channel 0: k*_obs = {kstar:.6f} rad/len  ->  p = {p_periods:.3f} periods/box"
          f"  (peak bin {pk})")
    print("  spectrum support around peak (contiguous, in units of k*_obs, edge-extended):")
    for s in support:
        print(f"    >= {s['frac_of_peak']*100:4.0f}% of peak: "
              f"[{s['rel_lo_edge']:.3f}, {s['rel_hi_edge']:.3f}]  "
              f"({s['n_bins']} bins, k in [{s['k_lo_center']:.4f}, {s['k_hi_center']:.4f}])")
    print(f"  PROPOSED b_lo, b_hi = {proposal['b_lo']:.3f}, {proposal['b_hi']:.3f} "
          f"(10%-of-peak edges; alternatives 50% "
          f"{proposal['alternatives']['50pct']}, 1% {proposal['alternatives']['1pct']}, "
          f"provisional {B_TRAIN})")
    print(f"  band power (RAPS-bin sums): B_train {bands['B_train']['raps_sum']:.4e}, "
          f"B_harm {bands['B_harm']['raps_sum']:.4e}, B_low {bands['B_low']['raps_sum']:.4e}")
    print(f"    B_harm/B_train = {frac_harm_raps:.4e} (RAPS) / {frac_harm_2d:.4e} (2-D modes)")
    print(f"    B_low /B_train = {frac_low_raps:.4e} (RAPS) / {frac_low_2d:.4e} (2-D modes)")
    print(f"  noise floor (top decade of k, {fl['n_bins']} bins): median "
          f"{fl['median']:.4e}, mean {fl['mean']:.4e}")
    print(f"    B_harm peak bin / floor median = {harm_peak_over_floor:.1f}x  ->  "
          f"{'MEASURABLE' if harm_peak_over_floor > 10 else 'MARGINAL/NOT measurable'}")
    p2d, psh = patch0["spectral_distance_2d"], patch0["raps_shape_distance"]
    print(f"  patch control ch0 (6 quadrant pairs, 256^2, {patch0['n_shape_band_bins']} "
          f"shape-band bins):")
    print(f"    spectral_distance_2d  mean {p2d['mean']:.4e}  min {p2d['min']:.4e}  "
          f"max {p2d['max']:.4e}")
    print(f"    raps shape distance   mean {psh['mean']:.4e}  min {psh['min']:.4e}  "
          f"max {psh['max']:.4e}")
    for c in (1, 2):
        r = ch_reports[f"channel_{c}"]
        print(f"  channel {c}: cv = {r['cv']:.4f}, RAPS peak at k = {r['peak_k']:.4f} "
              f"({r['kstar_rel_to_ch0']:.3f} k*_ch0 by kstar_of), "
              f"peak/floor = {r['peak_over_floor_median']:.1f}x")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
