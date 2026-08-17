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


def raps_torch(field, L: float = 1.0):
    """Torch parity port of `raps`, for a device-resident 2-D field. `field` stays on its
    own device throughout except for the final scalar; `k_centers`/`power` are returned as
    device tensors (real, same device/dtype as `field`) and `k_star` as a plain python
    float, mirroring `raps`'s (ndarray, ndarray, float) return.

    Same binning as `raps`: bins of width `dk = 2*pi/L` starting at k=0 (built from
    `np.fft.fftfreq`, then moved to `field`'s device — the on-device k-grid pattern
    `forward.py::make_spatial_F`/`translation_modes` already use), a power-weighted
    centroid over the 5 bins around the peak, k=0 bin dropped before the peak search.
    `np.digitize(x, kbins) - 1, clipped` is replaced by `floor(x / dk), clamped` — exactly
    equal for these bins (uniform width, starting at 0, x = |k| >= 0 always) — since
    torch has no digitize; accumulation is `scatter_add_` in place of numpy's `bincount`.

    PARITY PORT: numerically faithful by construction, but not yet PINNED against `raps` by
    a test (matching values on a shared field is the natural anchor) — do that before any
    caller relies on it. NOT wired into any caller here; forward.py owns that integration
    decision.
    """
    import torch
    f = field - field.mean()
    N = f.shape[0]
    device, dtype = f.device, f.dtype
    F = torch.abs(torch.fft.fft2(f)) ** 2
    kfreq_np = np.fft.fftfreq(N, d=L / N) * 2 * np.pi          # rad/length
    kfreq = torch.from_numpy(kfreq_np).to(device=device, dtype=dtype)
    KX, KY = torch.meshgrid(kfreq, kfreq, indexing="ij")
    KR = torch.sqrt(KX ** 2 + KY ** 2)
    dk = 2 * np.pi / L                                          # fundamental
    kbins_np = np.arange(0, float(KR.max().item()) + dk, dk)
    kbins = torch.from_numpy(kbins_np).to(device=device, dtype=dtype)
    nbins = kbins.shape[0] - 1
    idx = torch.clamp((KR.reshape(-1) / dk).floor().long(), 0, nbins - 1)
    Fflat = F.reshape(-1)
    power = torch.zeros(nbins, device=device, dtype=dtype).scatter_add_(0, idx, Fflat)
    counts = torch.zeros(nbins, device=device, dtype=dtype).scatter_add_(
        0, idx, torch.ones_like(Fflat))
    power = power / torch.clamp(counts, min=1.0)
    kcent = 0.5 * (kbins[:-1] + kbins[1:])
    power[0] = 0.0                                              # drop k=0
    pk = int(torch.argmax(power).item())
    lo, hi = max(1, pk - 2), min(power.shape[0], pk + 3)
    w = power[lo:hi]
    kstar = float((kcent[lo:hi] * w).sum() / (w.sum() + 1e-12))
    return kcent, power, kstar


def kstar_of_torch(field, L: float = 1.0) -> float:
    """Torch parity port of `kstar_of`: `raps_torch(field, L)[2]`. See `raps_torch`."""
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
