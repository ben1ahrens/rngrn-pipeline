"""solve_box.py — the adaptive commensurate solve box (`docs/REDESIGN_rngrn.md` §4.3).

RECOVERY-SIDE. Everything here is a function of (a) the model's OWN current dispersion
argmax k̂(θ), (b) the OBSERVED frame's RAPS and its k*_obs — both already-legal recovery
quantities (`docs/SPEC_fourier_training.md` §7) — and (c) pure grid geometry. No
ground-truth quantity is read or imported; the only rngrn import is
`losses.spectral.SpectralConfig`, itself recovery-side, for the band edges.

WHAT THE BOX IS FOR. A training forward solve on an arbitrary box lands the model's
pattern on whatever wavenumber lattice that box happens to admit, so the model spectrum
and the observed spectrum are compared on two unrelated grids. §4.3's resolution is to
solve on a box COMMENSURATE with the model's own current pattern:

    L_solve = p * 2*pi / k̂(θ)

so the box holds exactly `p` periods of k̂. Its RAPS bin width is then dk = 2*pi/L_solve
= k̂/p, i.e. k̂ sits exactly on bin `p`'s scale, and bin `j`'s centre sits at

    x_j = (j + 0.5) * dk / k̂ = (j + 0.5) / p                     [scale-free, exact]

in k/k̂ units — independent of k̂. The observed log-RAPS is interpolated onto those same
scale-free positions read as k/k*_obs, which is what makes the comparison unit-free, and
is why band masks stay defined relative to k*_obs exactly as in `SPEC_fourier_training.md`
§4 (a band [b_lo, b_hi]*k*_obs is simply x in [b_lo, b_hi] here).

k̂(θ) is **DETACHED**: the geometry is not differentiated. `geometry` takes a float (or
floats a tensor, which detaches it) and returns plain python numbers, so no autograd path
can run from a loss back to k̂ through the box. `tests/test_solve_box.py` asserts both the
absence of a grad on the geometry and the absence of a grad PATH through a loss that
genuinely depends on `L_solve`.

MEASUREMENT STATUS (§4.3, stated plainly; updated at the Task 22 audit pass):

* The training-grid spectral fidelity at n = 96 is MEASURED at R3 (Task 15, paired-IC
  design, D-R3-4): 96^2 clears D3's ~31 %/bin estimation floor on every seed of every
  fixture, worst-case margin 1.85x
  (`experiments/redesign_r3/raps_fidelity/results/raps_fidelity_paired.json`). n = 96 was
  ruled kept for Phase II (D-R3-6); the scope caveats (three N=3 fixtures, p=8, one band,
  `ic_noise=0.01`) travel with the ruling.
* The target-interpolation error this module's `interpolate_targets` incurs is measured at
  R3 (Task 11 Step 5); the measurement and its method are recorded in
  `experiments/redesign_r3/solve_box_interp/`.

This module does NOT decide WHEN to re-tile during a run (it only answers `needs_retile`),
does NOT solve anything, and does NOT own the solver's grid — `forward.PatternSolver.retile`
/ `forward.BatchedPatternSolver.retile` apply a new geometry and clear the warm starts.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .losses.spectral import SpectralConfig

#: Periods of k̂ held by the solve box. SPEC-FIXED by `docs/REDESIGN_rngrn.md` §4.3:
#: the nearest integer to D3's MEASURED 8.36 data periods, and D1's verified 8-period
#: solve geometry (`docs/DIAGNOSTICS_fft.md`). Calibrators: D3 (data), D1 (gradient).
P_DEFAULT = 8

#: Solve-box grid points per side. SPEC-FIXED by §4.3 as a STARTING value: 12
#: px/wavelength at p = 8, D1's verified density *for the gradient*. Calibrator: D1.
#: MEASURED at this grid by Task 15's paired-IC 96^2-vs-512^2 RAPS band measurement
#: (D-R3-4): clears D3's ~31 %/bin floor on all seeds/fixtures, worst margin 1.85x;
#: ruled kept for Phase II by D-R3-6 (scope caveats travel with the ruling).
N_DEFAULT = 96


def _detached_float(val) -> float:
    """`val` as a python float, DETACHING it first if it is a tensor.

    The detach is explicit rather than relying on `float()`'s implicit one so that §4.3's
    "k̂(θ) is detached" is visible in the code, and so a caller passing a live
    `requires_grad` tensor does not trip torch's scalar-conversion warning on every
    re-tile.
    """
    return float(val.detach()) if torch.is_tensor(val) else float(val)


def geometry(kstar_hat, p: int = P_DEFAULT, n: int = N_DEFAULT) -> dict:
    """The commensurate solve box for the model's current dispersion argmax `kstar_hat`.

    Returns ``{"L_solve": p*2*pi/k̂, "n": n, "p": p}`` as plain python numbers. `kstar_hat`
    may be a tensor; it is floated, which DETACHES it — §4.3 requires the solve-box
    geometry not to be differentiated, and a python float cannot carry a grad_fn at all.
    """
    khat = _detached_float(kstar_hat)
    if not (math.isfinite(khat) and khat > 0.0):
        raise ValueError(
            f"kstar_hat must be a positive finite wavenumber, got {khat!r} — the solve box "
            "is L_solve = p*2*pi/kstar_hat and is undefined otherwise")
    p = int(p)
    n = int(n)
    if p < 1:
        raise ValueError(f"p must be a positive integer number of periods, got {p!r}")
    if n < 2:
        raise ValueError(f"n must be at least 2 grid points per side, got {n!r}")
    return {"L_solve": p * 2.0 * math.pi / khat, "n": n, "p": p}


def needs_retile(kstar_hat_now, kstar_hat_at_tile, p: int = P_DEFAULT) -> bool:
    """§4.3's hysteresis: re-tile only when k̂ has drifted more than HALF a solve-box bin.

    The current box was built at `kstar_hat_at_tile`, so its RAPS bin width is
    dk = 2*pi/L_solve = kstar_hat_at_tile/p. Half a bin is kstar_hat_at_tile/(2p), hence
    the relative test ``|Δk̂| / k̂_at_tile > 1/(2p)`` — a pure function of `p`. The
    reference is the k̂ the box was TILED at, not the current one: it is that box's bins
    the drift is measured against.
    """
    now = _detached_float(kstar_hat_now)
    at_tile = _detached_float(kstar_hat_at_tile)
    for name, val in (("kstar_hat_now", now), ("kstar_hat_at_tile", at_tile)):
        if not (math.isfinite(val) and val > 0.0):
            raise ValueError(f"{name} must be a positive finite wavenumber, got {val!r}")
    p = int(p)
    if p < 1:
        raise ValueError(f"p must be a positive integer number of periods, got {p!r}")
    return abs(now - at_tile) / at_tile > 1.0 / (2 * p)


def n_bins(box: dict) -> int:
    """Number of RAPS bins the solve box's grid produces.

    Computed with `losses/spectral.py::_raps_bins`'s OWN edge formula (integer arange
    times dk, `n_edges = ceil((|k|_max + dk)/dk)`, F-D6-1) on this box's (n, L_solve), so
    the count matches the binning the model spectrum is actually computed with.
    `tests/test_solve_box.py::test_bin_centers_match_the_spectral_raps_binning` is the
    drift test binding the two.
    """
    n, L = int(box["n"]), float(box["L_solve"])
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    KX, KY = np.meshgrid(k, k)
    dk = 2.0 * np.pi / L
    n_edges = int(np.ceil((np.sqrt(KX ** 2 + KY ** 2).max() + dk) / dk))
    return n_edges - 1


def bin_centers_scalefree(box: dict) -> np.ndarray:
    """The solve box's RAPS bin centres in SCALE-FREE units, i.e. k_centre / k̂.

    Exactly ``(j + 0.5) / p``: the box is commensurate, so dk = k̂/p and the j-th centre is
    (j+0.5)*dk. Written in that closed form rather than as `k_centers / khat` so the
    identity is exact in floating point and visibly independent of k̂ — the property the
    scale-free target relies on. The drift test in `tests/test_solve_box.py` binds it to
    `losses/spectral._raps_bins`'s physical centres.
    """
    return (np.arange(n_bins(box), dtype=np.float64) + 0.5) / int(box["p"])


def band_bins(box: dict, b_lo: float = SpectralConfig.b_lo,
              b_hi: float = SpectralConfig.b_hi) -> np.ndarray:
    """Indices of the solve box's RAPS bins inside the band ``[b_lo, b_hi] * k*_obs``.

    Bands stay defined relative to k*_obs (`SPEC_fourier_training.md` §4); in this module's
    scale-free coordinate that is simply ``b_lo <= x <= b_hi``, since x IS k/k*_obs on the
    target side. Defaults are B_train's CLOSED edges (D-FFT-9 closure 1), taken from
    `SpectralConfig` rather than restated, so the two cannot drift apart.
    """
    if not (b_lo > 0.0 and b_hi > b_lo):
        raise ValueError(f"band must satisfy 0 < b_lo < b_hi, got ({b_lo!r}, {b_hi!r})")
    x = bin_centers_scalefree(box)
    idx = np.nonzero((x >= b_lo) & (x <= b_hi))[0]
    if idx.size == 0:
        raise ValueError(
            f"band [{b_lo}, {b_hi}]*k*_obs contains no solve-box bins: with p={box['p']} "
            f"the bin centres are (j+0.5)/{box['p']}, spanning "
            f"[{x[0]:.4g}, {x[-1]:.4g}]")
    return idx


def interpolate_targets(log_raps_obs, k_obs, kstar_obs: float, box: dict,
                        b_lo: float = SpectralConfig.b_lo,
                        b_hi: float = SpectralConfig.b_hi) -> torch.Tensor:
    """Observed log-RAPS put onto the solve box's band bins, in k/k*_obs units (§4.3).

    `log_raps_obs` is the observed frame's log radially-averaged power at wavenumbers
    `k_obs` (physical, rad/length, on the DATA box). `kstar_obs` is the observed
    fundamental. The solve box's band bins sit at scale-free positions
    ``x = (j+0.5)/p``; the target for bin j is `log_raps_obs` LINEARLY INTERPOLATED at
    ``k = x * kstar_obs``. Returned in solve-box bin order, matching `band_bins(box)`.

    Called ONCE PER RE-TILE. Note it is in fact independent of k̂ — the scale-free bin
    centres are (j+0.5)/p whatever k̂ is — so re-interpolating on a re-tile is idempotent
    unless `p` or `n` changed. That is a property of the commensurate box, not a licence to
    skip the call; `p`/`n` are exactly what Task 15 may move.

    Fails loud (CLAUDE.md §4) rather than extrapolating or returning a NaN:

    * a non-finite entry raises — the DC RAPS bin is forced to zero, so its log is -inf;
      drop it (or any zeroed bin) before calling rather than letting it reach the band;
    * a band bin outside the observed k support raises — filling it would INVENT target
      power the data never measured.

    Interpolation error is UNMEASURED in the spec and is measured at R3; see the module
    docstring.
    """
    log_raps_obs = np.asarray(log_raps_obs, dtype=np.float64)
    k_obs = np.asarray(k_obs, dtype=np.float64)
    if log_raps_obs.ndim != 1 or k_obs.ndim != 1:
        raise ValueError(
            f"log_raps_obs and k_obs must be 1-D, got shapes {log_raps_obs.shape} and "
            f"{k_obs.shape}")
    if log_raps_obs.shape != k_obs.shape:
        raise ValueError(
            f"log_raps_obs and k_obs must have the same length, got "
            f"{log_raps_obs.shape[0]} and {k_obs.shape[0]}")
    if k_obs.shape[0] < 2:
        raise ValueError("need at least two observed RAPS bins to interpolate")
    if not (np.isfinite(log_raps_obs).all() and np.isfinite(k_obs).all()):
        raise ValueError(
            "log_raps_obs and k_obs must be finite everywhere — a zeroed RAPS bin (the DC "
            "bin is forced to zero) gives log = -inf; drop those bins before calling")
    if not np.all(np.diff(k_obs) > 0.0):
        raise ValueError("k_obs must be strictly increasing")
    kstar_obs = float(kstar_obs)
    if not (math.isfinite(kstar_obs) and kstar_obs > 0.0):
        raise ValueError(f"kstar_obs must be positive and finite, got {kstar_obs!r}")

    x_want = bin_centers_scalefree(box)[band_bins(box, b_lo, b_hi)]
    x_obs = k_obs / kstar_obs
    if x_want[0] < x_obs[0] or x_want[-1] > x_obs[-1]:
        raise ValueError(
            f"solve-box band bins span k/k*_obs in [{x_want[0]:.4g}, {x_want[-1]:.4g}] but "
            f"the observed RAPS only supports [{x_obs[0]:.4g}, {x_obs[-1]:.4g}] — bins "
            "outside the observed support cannot be filled by interpolation, and "
            "extrapolating would invent target power")
    return torch.from_numpy(np.interp(x_want, x_obs, log_raps_obs))
