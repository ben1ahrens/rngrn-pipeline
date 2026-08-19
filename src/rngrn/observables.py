"""
rngrn_observables.py
====================
Image-only observable extraction for single-frame RNGRN recovery.

Everything here is computed from the observed frame ALONE. No analytic quantity of any
ground-truth system (x*, J, sigma(k), or a generating coefficient) is ever read. This is the
Section-12 firewall in code form.

Observables (see observables_spec.md for the measured reliabilities):
  * k* via the radially averaged power spectrum (RAPS) with a power-weighted peak centroid.
    Measured reliable to ~10-15%, biased HIGH off onset (selected vs linear wavelength).
    Returned in physical angular units (rad / length), given the domain extent L.
  * morphology statistics: intensity skewness & kurtosis; angular concentration & entropy of
    the dominant Fourier ring. These separate spots / stripes / mixed / labyrinth.
  * the exact periodic Laplacian by spectral differentiation, for the full-RHS residual.

The RAPS and the numpy Laplacians are the measurement side. Two periodic-Laplacian operators
are provided: an FFT-based spectral one (`laplacian_spectral`) and a 5-point finite-difference
stencil (`laplacian_5point`), which agree to ~0.05% on smooth fields. `laplacian_torch`
implements the SAME 5-point stencil in torch, bit-identical to the numpy 5-point.

CORRECTED 2026-08-17: `laplacian_torch` does NOT match the M1 solver's operator. The
solver (`forward.py::make_spatial_F`) is spectral (rfft2), matching CLAUDE.md §7c's "no
discretised Laplacian" claim; `laplacian_torch` is a discretised 5-point stencil, used
ONLY by `losses/terms.py::stationarity_residual`, whose weight (`resid`) defaults to 0.0
(DEFAULT_WEIGHTS, settled OFF — see `losses/terms.py` line ~827). So at the default config
this stencil never runs in training at all; it is not "the solver's operator" on any path.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------------------
# k* : radially averaged power spectrum
# --------------------------------------------------------------------------------------
def raps(field: np.ndarray, L: float = 1.0):
    """Radially averaged power spectrum of a 2D field on a periodic square of extent L.

    Returns (k_centers, power, k_star) with k in physical ANGULAR wavenumber (rad/length).
    k_star is a power-weighted centroid over the peak bins (sub-bin precision).
    """
    f = field - field.mean()
    N = f.shape[0]
    F = np.abs(np.fft.fft2(f)) ** 2
    kfreq = np.fft.fftfreq(N, d=L / N) * 2 * np.pi          # rad/length
    KX, KY = np.meshgrid(kfreq, kfreq)
    KR = np.sqrt(KX ** 2 + KY ** 2)
    dk = 2 * np.pi / L                                       # fundamental
    kbins = np.arange(0, KR.max() + dk, dk)
    idx = np.clip(np.digitize(KR.ravel(), kbins) - 1, 0, len(kbins) - 2)
    power = np.bincount(idx, weights=F.ravel(), minlength=len(kbins) - 1)
    counts = np.bincount(idx, minlength=len(kbins) - 1)
    power = power / np.maximum(counts, 1)
    kcent = 0.5 * (kbins[:-1] + kbins[1:])
    power[0] = 0.0                                           # drop k=0
    pk = int(np.argmax(power))
    lo, hi = max(1, pk - 2), min(len(power), pk + 3)
    w = power[lo:hi]
    kstar = float(np.sum(kcent[lo:hi] * w) / (np.sum(w) + 1e-12))
    return kcent, power, kstar


def kstar_of(field: np.ndarray, L: float = 1.0) -> float:
    return raps(field, L)[2]


#: Cache of `raps_torch`'s binning, keyed (n, L, device, dtype). The binning depends on
#: nothing but the grid geometry, while building it costs an fftfreq/meshgrid/sqrt pass,
#: two host-to-device transfers, and a `KR.max()` device sync. Uncached that ran on EVERY
#: call — and `forward.relax_to_pattern_torch` calls it once per 500-step relax chunk, up
#: to 400 times per solve, where it dominated the detector (measured 2026-08-17: at 96^2
#: the uncached device path was SLOWER than pulling the frame back to the host). Entries
#: are one (n^2,) int64 index tensor plus two small vectors; a run touches one geometry.
_RAPS_TORCH_BINS: dict = {}


def _raps_torch_bins(n: int, L: float, device, dtype):
    """(idx, counts, kcent) for `raps_torch` — built once per (n, L, device, dtype).

    Built on the HOST in numpy, from the same fftfreq/meshgrid/arange expressions `raps`
    uses, then transferred once. Doing it here rather than on the device also removes the
    `KR.max().item()` sync that used to decide the bin count, and makes the edges
    bit-identical to `raps`'s on every device instead of depending on whether the device's
    `sqrt` rounds a lattice radius the way numpy's does.

    The bin assignment is `raps`'s OWN `np.digitize(...) - 1`, clipped — not an arithmetic
    stand-in for it. It was `floor(|k|/dk)` until 2026-08-19, justified by torch having no
    `digitize`; that justification ended when the binning moved to the host, and the two
    are NOT equal (D-OBS-1 in `docs/DECISIONS.md`). `tests/test_raps_torch_parity.py`
    pins this equality point for point, on a geometry that provably has the bin-edge
    lattice radii where the two can differ.
    """
    import torch
    key = (int(n), float(L), device, dtype)
    ent = _RAPS_TORCH_BINS.get(key)
    if ent is not None:
        return ent
    kfreq = np.fft.fftfreq(n, d=L / n) * 2 * np.pi              # rad/length
    KX, KY = np.meshgrid(kfreq, kfreq)
    KR = np.sqrt(KX ** 2 + KY ** 2)
    dk = 2 * np.pi / L                                          # fundamental
    kbins = np.arange(0, KR.max() + dk, dk)
    nbins = len(kbins) - 1
    which = np.clip(np.digitize(KR.ravel(), kbins) - 1, 0, nbins - 1)
    counts = np.maximum(np.bincount(which, minlength=nbins), 1).astype(np.float64)
    kcent = 0.5 * (kbins[:-1] + kbins[1:])
    ent = (torch.from_numpy(which).to(device),
           torch.from_numpy(counts).to(device=device, dtype=dtype),
           torch.from_numpy(kcent).to(device=device, dtype=dtype))
    _RAPS_TORCH_BINS[key] = ent
    return ent


def raps_torch(field, L: float = 1.0):
    """Torch parity port of `raps`, for a device-resident 2-D field. `field` stays on its
    own device throughout except for the final scalar; `k_centers`/`power` are returned as
    device tensors (real, same device/dtype as `field`) and `k_star` as a plain python
    float, mirroring `raps`'s (ndarray, ndarray, float) return.

    Same binning as `raps`, and the same CODE for it: bins of width `dk = 2*pi/L` starting
    at k=0 with `np.digitize(|k|, kbins) - 1` clipped, a power-weighted centroid over the 5
    bins around the peak, k=0 bin dropped before the peak search. Only the accumulation
    differs — `scatter_add_` in place of numpy's `bincount`.

    The bin assignment is built ONCE per (n, L, device, dtype) by `_raps_torch_bins` and
    cached, so a call does no host work, no host-to-device transfer and no `KR.max()`
    sync; only the FFT, the scatter and the peak `argmax` remain.

    PARITY PORT, PINNED by `tests/test_raps_torch_parity.py`, which asserts the bin
    assignment against `np.digitize` point for point and the power/k* against `raps` on a
    saturated 64^2 pattern and at the relax detector's own operating point (x* + 1e-2
    noise, n = 16/32/64/96). Bars are `1e-12` per bin and `1e-14` on k*, ~20x above the
    MEASURED round-off floor of 4.80e-14 / 4.50e-16 (2026-08-19, CPU float64) — that floor
    is FFT backend plus summation order, since a bin's power is an unordered sum of up to
    n^2 terms and `bincount` and `scatter_add_` add them in different orders.

    Until 2026-08-19 the binning here was `floor(|k|/dk)`, which is NOT equal to `digitize`
    on these grids: a lattice radius that is an exact multiple of `dk` sits exactly ON a
    bin edge, and `|k|/dk` can evaluate 1 ulp low there. Those radii are the ON-AXIS modes,
    which carry the pattern's dominant power, so the effect was not small — measured up to
    22.5% on a bin and 30.2% on k* at the detector's operating point. See D-OBS-1
    (`docs/DECISIONS.md`); same class as finding F-D6-1 (`docs/DIAGNOSTICS_fft.md`).
    `losses/spectral.py::raps_torch` remains a separate function with its own
    integer-arange edges (F-D6-1's fix) and is not called from here.

    Wired into `forward.relax_to_pattern_torch`'s saturation detector and, through the
    shared cached binning, into `forward._kstar_of_torch_batched`.
    """
    import torch
    f = field - field.mean()
    N = f.shape[0]
    idx, counts, kcent = _raps_torch_bins(N, L, f.device, f.dtype)
    F = torch.abs(torch.fft.fft2(f)) ** 2
    power = torch.zeros(kcent.shape[0], device=f.device,
                        dtype=f.dtype).scatter_add_(0, idx, F.reshape(-1))
    power = power / counts
    power[0] = 0.0                                              # drop k=0
    pk = int(torch.argmax(power).item())
    lo, hi = max(1, pk - 2), min(power.shape[0], pk + 3)
    w = power[lo:hi]
    kstar = float((kcent[lo:hi] * w).sum() / (w.sum() + 1e-12))
    return kcent, power, kstar


def kstar_of_torch(field, L: float = 1.0) -> float:
    """Torch parity port of `kstar_of`: `raps_torch(field, L)[2]`. See `raps_torch` — it
    shares that function's cached binning, so this costs one FFT, one scatter and two
    scalar syncs (the peak `argmax` and the returned k*), with no host rebuild."""
    return raps_torch(field, L)[2]


# --------------------------------------------------------------------------------------
# morphology statistics
# --------------------------------------------------------------------------------------
def morphology(field: np.ndarray):
    """Intensity and spectral-angular statistics that separate the morphology classes.

    Returns dict: skew, kurtosis, ang_conc, ang_entropy.
    """
    from scipy.stats import skew, kurtosis
    f = field.ravel()
    sk = float(skew(f))
    ku = float(kurtosis(f))
    ac, ae = _angular(field)
    return dict(skew=sk, kurtosis=ku, ang_conc=ac, ang_entropy=ae)


def _angular(field: np.ndarray, nbins: int = 18):
    fm = field - field.mean()
    N = fm.shape[0]
    F = np.abs(np.fft.fft2(fm)) ** 2
    kf = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kf, kf)
    KR = np.sqrt(KX ** 2 + KY ** 2)
    # locate dominant ring radius in pixel-frequency units
    kd = kstar_of(field, L=1.0) / (2 * np.pi)               # cycles across box == ring radius (px)
    ring = np.abs(KR - kd) < max(1.5, 0.15 * kd)
    ang = np.mod(np.arctan2(KY, KX)[ring], np.pi)
    p = F[ring]
    if p.sum() <= 0:
        return 0.0, 1.0
    hist, _ = np.histogram(ang, bins=nbins, range=(0, np.pi), weights=p)
    hist = hist / hist.sum()
    ang_conc = float(hist.max())
    ang_entropy = float(-np.sum(hist * np.log(hist + 1e-12)) / np.log(nbins))
    return ang_conc, ang_entropy


def classify(field: np.ndarray) -> str:
    """Coarse morphology label from the statistics. Auxiliary only."""
    m = morphology(field)
    if m["ang_conc"] > 0.7 and m["ang_entropy"] < 0.2:
        return "stripes"
    if abs(m["skew"]) > 1.0:
        return "spots"
    if m["ang_entropy"] > 0.8:
        return "labyrinth"
    return "mixed"


# --------------------------------------------------------------------------------------
# periodic Laplacian
# --------------------------------------------------------------------------------------
def laplacian_spectral(field: np.ndarray, L: float = 1.0) -> np.ndarray:
    """Exact periodic Laplacian by spectral differentiation (measurement side, numpy)."""
    N = field.shape[0]
    k = np.fft.fftfreq(N, d=L / N) * 2 * np.pi
    KX, KY = np.meshgrid(k, k)
    lap = np.fft.ifft2(-(KX ** 2 + KY ** 2) * np.fft.fft2(field)).real
    return lap


def laplacian_5point(field: np.ndarray, L: float = 1.0) -> np.ndarray:
    """5-point periodic Laplacian (matches the brief's stencil and the solver's operator)."""
    N = field.shape[0]
    dx = L / N
    return (np.roll(field, 1, 0) + np.roll(field, -1, 0) +
            np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field) / dx ** 2


def laplacian_torch(field, L=1.0):
    """5-point periodic Laplacian in torch, for use inside the differentiable objective.
    field: (H,W) or (C,H,W) tensor."""
    import torch
    if field.dim() == 2:
        field = field.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    N = field.shape[-1]
    dx = L / N
    lap = (torch.roll(field, 1, -2) + torch.roll(field, -1, -2) +
           torch.roll(field, 1, -1) + torch.roll(field, -1, -1) - 4 * field) / dx ** 2
    return lap.squeeze(0) if squeeze else lap
