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
stencil (`laplacian_5point`), which agree to ~0.05% on smooth fields. For use inside the
differentiable objective, `laplacian_torch` implements the SAME 5-point stencil in torch
(matching the brief's stencil and the solver's operator), bit-identical to the numpy 5-point.
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
