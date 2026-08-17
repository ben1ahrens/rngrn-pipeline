"""losses/spectral.py — M1 spectral loss terms (Milestone M1, PLAN_fourier_training.md).

RECOVERY-SIDE (`tests/test_firewall.py::RECOVERY_SIDE`). Everything here is computed from
either (a) torch spectral/real-space statistics of `u_star`, the RECOVERY's own predicted
patterned steady state (produced elsewhere — this module never solves for it), or (b)
statistics of the OBSERVED frame passed in by the caller. No ground-truth quantity (x*,
J, sigma(k), a generating coefficient, or a payload attr) is read or imported. Imports are
limited to torch/numpy/stdlib.

PROVENANCE: the arithmetic (RAPS binning, the 2-D log-power block, the five term formulas)
is PORTED from `scripts/diag_fft_d1.py` (`torch_raps_power`, `spectral_block_torch`,
`make_loss_terms`), which passed D1's gradient-vs-finite-difference verification (worst rel
err 6.4e-8 vs tol 1e-4, `docs/DIAGNOSTICS_fft.md`). TWO changes from that reference, both
documented at their call site: (1) the RAPS bin edges are built as an INTEGER arange times
dk rather than `np.arange(0.0, kmax+dk, dk)`, per finding F-D6-1 (`docs/DIAGNOSTICS_fft.md`)
— a float-step arange can be 1 ulp off on knife-edge lattice points; the integer form
removes the risk while producing an identical edge count and identical values (verified
numerically for this port). (2) `spectral_block_torch` normalises by the CENTRAL BLOCK's
own power (`block.sum()`), not the whole spectrum's (`P.sum()`, the D1 prototype's choice)
— matching `scoring.morphology._spectral_block`/`rngrn.utils.spectral_block` exactly, which
the D1 prototype did not (see `spectral_block_torch`'s docstring).

SPEC references: `docs/SPEC_fourier_training.md` §4 (spectral representation, RAPS,
band/normalisation rules), §5 (the five loss-term definitions). Band edges are CLOSED at
b_lo=0.60, b_hi=1.55 (`docs/DECISIONS.md` D-FFT-9 closure 1). The leakage rule (SPEC §7,
§4): shape-term normalisers use ONLY `B_train` power of the TRAINING channel (channel 0),
never total power, never held-out bands, never other channels.

This module does NOT decide term weights, does NOT wire terms into `losses/total.py`, and
does NOT solve for `u_star` — those are other M1 units' work (PLAN §9 module change map).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpectralConfig:
    """Knobs for the M1 spectral terms. Every default's provenance is stated below; nothing
    here is invented for convenience (CLAUDE.md §10).

    b_lo, b_hi: B_train = [b_lo, b_hi] * k*_obs. CLOSED by measurement, D-FFT-9 closure 1
        (`docs/DECISIONS.md`): the >=1%-of-peak contiguous RAPS support of
        `turing_labyrinth/sample_0000` channel 0.
    channels: which channels are fitted. Stage 0 fits channel 0 only (SPEC §5), hence the
        default (0,).
    nblk: the central block size for the 2-D log-power comparator, matching
        `scoring.morphology._spectral_block`'s own default and `spectral_block_torch`'s
        D1-verified prototype.
    ignition_margin: UNCALIBRATED. Mirrors the margin default of
        `losses/terms.py::turing_hinges_split` (`margin=1e-3`, defined at that function's
        signature, losses/terms.py:186) — the same softplus-hinge margin already used to
        decide "Turing-unstable enough" for the existing hinge terms. This is a DIFFERENT
        gate from D-FFT-9 closure 2's real-space rollout amplitude floor
        (`eval/rollout.py`'s `pattern_floor`, used by the D2 diagnostic to separate
        saturated-Turing from decayed rollouts): `is_ignited` below reads the LINEAR
        stability diagnostic (`sig_max_pos`) that `losses/total.py::compute_terms` already
        computes every step, not a rollout amplitude. No measurement ties this specific
        value to spectral-term ignition timing yet; mark UNCALIBRATED until Stage 0.
    """
    b_lo: float = 0.60
    b_hi: float = 1.55
    channels: tuple[int, ...] = (0,)
    nblk: int = 24
    ignition_margin: float = 1e-3  # UNCALIBRATED — see docstring


# --------------------------------------------------------------------------------------
# spectral primitives
# --------------------------------------------------------------------------------------
#: Cache of the RAPS binning, keyed (n, L, device, dtype). The binning is pure grid
#: geometry — it depends on nothing that changes between training steps — while building
#: it costs an fftfreq/meshgrid/digitize/bincount pass on the HOST plus two H2D transfers.
#: Uncached that ran once per `raps_torch` call, i.e. twice per training step. Entries are
#: a handful of (n^2,) index tensors; a run touches one or two geometries.
_RAPS_BINS: dict = {}


def _raps_bins(n: int, L: float, device: torch.device, dtype: torch.dtype):
    """(idx, counts, k_centers) for `raps_torch`'s binning — built once per geometry.

    `idx` is the flat bin index of every |k| lattice point and `counts` the per-bin
    occupancy, both device-resident; `k_centers` is a read-only numpy array (it is shared
    by every caller, so it must not be written through)."""
    key = (int(n), float(L), device, dtype)
    ent = _RAPS_BINS.get(key)
    if ent is not None:
        return ent
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    KX, KY = np.meshgrid(k, k)
    KR = np.sqrt(KX ** 2 + KY ** 2)
    dk = 2.0 * np.pi / L
    n_edges = int(np.ceil((KR.max() + dk) / dk))
    kbins = np.arange(n_edges) * dk
    which = np.clip(np.digitize(KR.ravel(), kbins) - 1, 0, len(kbins) - 2)
    counts = np.bincount(which, minlength=len(kbins) - 1)
    k_centers = 0.5 * (kbins[:-1] + kbins[1:])
    k_centers.setflags(write=False)
    ent = (torch.from_numpy(which).to(device),
           torch.from_numpy(np.maximum(counts, 1)).to(device=device, dtype=dtype),
           k_centers)
    _RAPS_BINS[key] = ent
    return ent


def raps_torch(field: torch.Tensor, L: float) -> tuple[np.ndarray, torch.Tensor]:
    """Radially averaged power spectrum of one (H, W) channel, differentiable w.r.t. `field`.

    Bins bin-for-bin identically to `observables.raps` (dk = 2*pi/L, DC bin zeroed). The
    bin edges are built as an INTEGER arange times dk (`np.arange(n_edges) * dk`), not
    `np.arange(0.0, kmax+dk, dk)` — finding F-D6-1 (`docs/DIAGNOSTICS_fft.md`): a float-step
    arange can land 1 ulp off numpy's on knife-edge lattice points and flip which bin a
    boundary point falls into. `n_edges` is computed with the same `ceil` formula so the
    edge COUNT matches the numpy reference exactly (verified: identical for every (n, L)
    tested, including the p=8/512^2 canonical case).

    The binning itself is CACHED per (n, L, device, dtype) by `_raps_bins`, so a per-step
    call does no host work and no host-to-device transfer. The arithmetic is unchanged
    and the returned `k_centers` is the cached (read-only) array, not a fresh copy. This
    is deliberately NOT `observables.raps_torch`, whose bins come from a float-step arange
    and `floor` rather than F-D6-1's integer edges and `digitize`.
    """
    if field.shape[-2] != field.shape[-1]:
        # observables.raps assumes a square periodic box (one fftfreq grid for both axes);
        # a non-square field would bin silently wrongly there AND here. Refuse.
        raise ValueError(f"raps_torch: field must be square, got {tuple(field.shape)}")
    n = field.shape[-1]
    idx, counts, k_centers = _raps_bins(n, L, field.device, field.dtype)
    f = field - field.mean()
    P = torch.abs(torch.fft.fft2(f)) ** 2
    power = torch.zeros(k_centers.shape[0], dtype=field.dtype,
                        device=field.device).scatter_add_(0, idx, P.reshape(-1))
    power = power / counts
    power = power.clone()
    power[0] = 0.0
    return k_centers, power


def spectral_block_torch(field: torch.Tensor, nblk: int = 24) -> torch.Tensor:
    """Central (nblk, nblk) log-power block, differentiable w.r.t. `field`. Same arithmetic
    and ORDER (normalise by total power, THEN log1p, THEN sum-normalise) as
    `scoring.morphology._spectral_block` — verified to allclose 1e-12 by
    `tests/test_spectral_terms.py`.

    DEVIATION FROM `scripts/diag_fft_d1.py::spectral_block_torch`: that D1 prototype
    normalises by `P.sum()` (total power of the WHOLE spectrum, pre-slice). This function
    normalises by `block.sum()` (post-slice — the central block's own power), matching
    `scoring.morphology._spectral_block` and its side-neutral port `rngrn.utils.spectral_block`
    exactly (both use the block's own total). The diag script's choice was never checked
    against the comparator it names in its docstring; D1 only verified IFT-gradient
    consistency, which holds under either normalisation. This port uses the one that
    actually matches morphology's arithmetic, since `spec_aniso`'s whole point is to be a
    differentiable stand-in for that comparator."""
    f = field - field.mean()
    P = torch.abs(torch.fft.fftshift(torch.fft.fft2(f))) ** 2
    H, W = P.shape
    top, left = H // 2 - nblk // 2, W // 2 - nblk // 2
    block = P[top:top + nblk, left:left + nblk]
    total = block.sum()
    block = torch.log1p(block / total)
    return block / block.sum()


def band_mask(k_centers: np.ndarray, kstar_obs: float, lo: float, hi: float) -> np.ndarray:
    """Boolean mask of `k_centers` in `[lo, hi] * kstar_obs` (SPEC §4's B_train, or any other
    band by passing different lo/hi)."""
    return (k_centers >= lo * kstar_obs) & (k_centers <= hi * kstar_obs)


# --------------------------------------------------------------------------------------
# fixed targets from the observed frame
# --------------------------------------------------------------------------------------
def build_frame_targets(frame: np.ndarray, L: float, kstar_obs: float,
                        cfg: SpectralConfig, device=None) -> dict:
    """Statistics of the OBSERVED frame, computed ONCE. Every entry is a constant (a
    detached tensor or a float) — never something a term function differentiates through.

    Firewall: `frame` is the observed image, `L` and `kstar_obs` are already-legal recovery
    quantities (SPEC §7); nothing else is read. Fails loud (ValueError) rather than
    returning a NaN or an undefined target (CLAUDE.md §4).

    `device`: where the tensor-valued targets are PLACED. The statistics are always
    computed on the host (the frame is a numpy array, and this runs once), so the values
    do not depend on it; passing the model's device moves the one-off transfer here
    instead of paying it inside every term function on every training step. None keeps
    them on the CPU, which is what every existing caller gets.
    """
    if frame.ndim != 3:
        raise ValueError(f"frame must be (C, H, W); got shape {frame.shape}")

    ch0 = torch.from_numpy(np.ascontiguousarray(frame[0], dtype=np.float64))
    if float(ch0.std()) <= 0.0:
        raise ValueError(
            "frame channel 0 is constant — no spectral content to build a RAPS/block "
            "target from")

    k_centers, power = raps_torch(ch0, L)
    mask = band_mask(k_centers, kstar_obs, cfg.b_lo, cfg.b_hi)
    if not mask.any():
        raise ValueError(
            f"B_train=[{cfg.b_lo}, {cfg.b_hi}]*k*_obs (k*_obs={kstar_obs}) contains no RAPS "
            f"bins — k_centers span [{k_centers.min():.4g}, {k_centers.max():.4g}], "
            f"dk={k_centers[1] - k_centers[0]:.4g}")
    # Integer bin indices, not the boolean mask: `power[mask]` is data-dependent
    # indexing, whose output size CUDA cannot know without a device-to-host sync, and the
    # term functions index the band on every training step. torch.nonzero here (once) and
    # index_select there (per step) does the same selection, in the same bin order.
    band_idx = torch.from_numpy(np.nonzero(mask)[0])
    band_power = power.index_select(0, band_idx)
    band_sum = band_power.sum()
    if not torch.isfinite(band_sum) or float(band_sum) <= 0.0:
        raise ValueError(
            "B_train RAPS power is zero or non-finite for frame channel 0 — cannot build a "
            "normalised spec_shape target")
    raps_band_target = (band_power / band_sum).detach().to(device)
    band_idx = band_idx.to(device)

    block_target = spectral_block_torch(ch0, cfg.nblk).detach().to(device)
    if not torch.isfinite(block_target).all():
        raise ValueError("spectral block target for frame channel 0 has non-finite entries")

    log_mean: dict[int, float] = {}
    log_std: dict[int, float] = {}
    skew: dict[int, float] = {}
    for c in cfg.channels:
        if c >= frame.shape[0]:
            raise ValueError(f"channel {c} out of range for a frame with {frame.shape[0]} "
                             "channels")
        chan = torch.from_numpy(np.ascontiguousarray(frame[c], dtype=np.float64))
        if float(chan.std()) <= 0.0:
            raise ValueError(
                f"frame channel {c} is constant — amplitude/moment targets are undefined")
        m, s = chan.mean(), chan.std()
        if not float(m) > 0.0:
            raise ValueError(f"frame channel {c} has mean <= 0 ({float(m)!r}) — log_mean "
                             "target is undefined")
        v = chan - m
        sk = (v ** 3).mean() / (v ** 2).mean().clamp_min(1e-300) ** 1.5
        lm, ls, skv = float(torch.log(m)), float(torch.log(s)), float(sk)
        if not (np.isfinite(lm) and np.isfinite(ls) and np.isfinite(skv)):
            raise ValueError(
                f"channel {c} produced a non-finite target (log_mean={lm}, log_std={ls}, "
                f"skew={skv})")
        log_mean[c] = lm
        log_std[c] = ls
        skew[c] = skv

    return dict(k_centers=k_centers, kstar_obs=float(kstar_obs), L=float(L),
                raps_band_target=raps_band_target, band_idx=band_idx,
                block_target=block_target,
                log_mean=log_mean, log_std=log_std, skew=skew)


# --------------------------------------------------------------------------------------
# the five term functions — each (u_star, targets, cfg) -> (scalar tensor, parts dict)
# --------------------------------------------------------------------------------------
def spec_shape(u_star: torch.Tensor, targets: dict, cfg: SpectralConfig
              ) -> tuple[torch.Tensor, dict]:
    """Gain-invariant spectral shape on channel 0's B_train (SPEC §5, diag reference
    formula: `make_loss_terms::spec_shape`). Channel 0 only — `build_frame_targets` builds
    a single un-indexed `raps_band_target` for the training channel, matching Stage 0's
    "channel 0 only" fit (SPEC §5).

    The band selection reuses `targets["band_idx"]`, built by `build_frame_targets` from
    the same (n, L, k*_obs, b_lo, b_hi): the bin grid is a function of the geometry alone,
    and the shape check below is what pins that the two grids ARE the same. Rebuilding
    the boolean mask here per step would repeat that host work and index by a
    data-dependent mask, which forces a size sync on CUDA."""
    k_centers, power = raps_torch(u_star[0], targets["L"])
    if k_centers.shape != targets["k_centers"].shape:
        raise ValueError(
            "u_star's spatial grid does not match the observed frame's grid — RAPS bin "
            f"counts differ ({k_centers.shape[0]} vs {targets['k_centers'].shape[0]})")
    band_power = power.index_select(0, targets["band_idx"].to(power.device))
    norm = band_power.sum()
    # Fail loud, mirroring build_frame_targets' guard on the observed side: a solved
    # pattern with (near-)zero B_train power would otherwise hand Adam a silent NaN.
    if not torch.isfinite(norm) or float(norm) <= 0.0:
        raise ValueError(
            f"spec_shape: u_star channel 0 carries no finite power in B_train "
            f"(norm={float(norm):.3e}) — the solved pattern has no spectral content in "
            "the fitting band; this should have been caught by the pattern_floor gate.")
    s = band_power / norm
    t = targets["raps_band_target"].to(s.device)
    val = ((torch.log(s + 1e-300) - torch.log(t + 1e-300)) ** 2).sum()
    return val, dict(spec_shape_raw=float(val.detach()))


def spec_aniso(u_star: torch.Tensor, targets: dict, cfg: SpectralConfig
              ) -> tuple[torch.Tensor, dict]:
    """2-D block distance on channel 0 (SPEC §5). SQUARED form — the diag script's D1
    gradient check verified `((blk - target) ** 2).mean()`; note
    `scoring.morphology.spectral_distance_2d` (the comparator this is analogous to) reports
    RMSE, i.e. this term's square root, not the same scalar."""
    blk = spectral_block_torch(u_star[0], cfg.nblk)
    val = ((blk - targets["block_target"].to(blk.device)) ** 2).mean()
    return val, dict(spec_aniso_raw=float(val.detach()))


def spec_amp_mean(u_star: torch.Tensor, targets: dict, cfg: SpectralConfig
                  ) -> tuple[torch.Tensor, dict]:
    """Per-channel mean (DC) match in log space, summed over `cfg.channels` (SPEC §5)."""
    total = u_star.new_zeros(())
    parts: dict = {}
    for c in cfg.channels:
        m = u_star[c].mean()
        # Fail loud: nothing in the forward solve enforces positivity, and log() of a
        # non-positive mean would hand Adam a silent NaN (a defect, not a value).
        if not torch.isfinite(m) or float(m) <= 0.0:
            raise ValueError(
                f"spec_amp_mean: u_star channel {c} has non-positive/non-finite mean "
                f"({float(m):.3e}) — the solved pattern left the physical regime.")
        term = (torch.log(m) - targets["log_mean"][c]) ** 2
        total = total + term
        parts[f"spec_amp_mean_c{c}"] = float(term.detach())
    return total, parts


def spec_amp_fluct(u_star: torch.Tensor, targets: dict, cfg: SpectralConfig
                   ) -> tuple[torch.Tensor, dict]:
    """Per-channel fluctuation (std) match in log space, summed over `cfg.channels` (SPEC
    §5)."""
    total = u_star.new_zeros(())
    parts: dict = {}
    for c in cfg.channels:
        s = u_star[c].std()
        # Fail loud, same reason as spec_amp_mean: log() of a zero/non-finite std would
        # be a silent NaN. A zero-std (flat) channel here means the pattern_floor gate
        # upstream failed to do its job — raise, don't degrade.
        if not torch.isfinite(s) or float(s) <= 0.0:
            raise ValueError(
                f"spec_amp_fluct: u_star channel {c} has non-positive/non-finite std "
                f"({float(s):.3e}) — flat or invalid channel reached the spectral terms.")
        term = (torch.log(s) - targets["log_std"][c]) ** 2
        total = total + term
        parts[f"spec_amp_fluct_c{c}"] = float(term.detach())
    return total, parts


def real_moments(u_star: torch.Tensor, targets: dict, cfg: SpectralConfig
                 ) -> tuple[torch.Tensor, dict]:
    """Real-space skewness match, summed over `cfg.channels` (SPEC §5/§6 — restores the
    contrast-inversion structure the power spectrum discards). Verbatim from the diag
    script's `make_loss_terms::real_moments`, including the `clamp_min(1e-300)` guard on
    the variance in the skew denominator."""
    total = u_star.new_zeros(())
    parts: dict = {}
    for c in cfg.channels:
        v = u_star[c] - u_star[c].mean()
        skew = (v ** 3).mean() / (v ** 2).mean().clamp_min(1e-300) ** 1.5
        term = (skew - targets["skew"][c]) ** 2
        total = total + term
        parts[f"real_moments_c{c}"] = float(term.detach())
    return total, parts


_TERM_FNS = dict(spec_shape=spec_shape, spec_aniso=spec_aniso, spec_amp_mean=spec_amp_mean,
                 spec_amp_fluct=spec_amp_fluct, real_moments=real_moments)

# Public — the single source of truth for the five term names, so `losses/total.py`
# (history-stability NaN placeholders) and `recover.py` (the ignition/weight gate) cannot
# drift apart from `_TERM_FNS` (unit U4, M1 wiring).
SPECTRAL_TERM_KEYS = tuple(_TERM_FNS)


def spectral_terms(u_star: torch.Tensor, targets: dict, cfg: SpectralConfig
                   ) -> tuple[dict, dict]:
    """Evaluate all five terms. Returns (term_vals, parts) — term_vals keyed
    spec_shape/spec_aniso/spec_amp_mean/spec_amp_fluct/real_moments, each an UNWEIGHTED
    scalar tensor; parts is the merged flat-float diagnostics dict (CLAUDE.md §4: run-index
    rows are flat scalars built by dict.update, never `**`-expansion into a collision-prone
    literal)."""
    term_vals: dict = {}
    parts: dict = {}
    for name, fn in _TERM_FNS.items():
        val, p = fn(u_star, targets, cfg)
        term_vals[name] = val
        parts.update(p)
    return term_vals, parts


# --------------------------------------------------------------------------------------
# ignition gate
# --------------------------------------------------------------------------------------
def is_ignited(parts: dict, margin: float) -> bool:
    """True iff the steady state converged and the model is Turing-unstable beyond
    `margin` (`sig_max_pos`, the disjoint-support instability hinge diagnostic that
    `losses.terms.turing_hinges_split` puts in `parts` via `losses.total.compute_terms`).

    Spectral terms ignite on DETECTED patterning (CLAUDE.local.md) — they are OMITTED, not
    computed-and-zeroed, until this is True (mirrors the `resid_skipped` precedent in
    `losses/total.py`).

    Raises KeyError if `sig_max_pos` is absent: that means `split_hinges=False` was used
    (or hinges were skipped entirely), which is a misconfiguration for the spectral path,
    not a condition to silently treat as "not ignited".
    """
    if "sig_max_pos" not in parts:
        raise KeyError(
            "is_ignited requires parts['sig_max_pos'] (from "
            "losses.terms.turing_hinges_split, via losses.total.compute_terms with "
            "split_hinges=True). Its absence means split hinges are off for this run — a "
            "misconfiguration, not a silent 'not ignited'.")
    return bool(parts["ss_converged"]) and float(parts["sig_max_pos"]) > margin


# --------------------------------------------------------------------------------------
# BATCHED twins — B independent restarts, one stack of tensor ops
# --------------------------------------------------------------------------------------
# Each function below is its serial namesake with a leading member axis: `u_star` is
# (b, N, n, n) rather than (N, n, n), the returned term is a (b,) tensor rather than a 0-d
# one, and each `parts` entry is a (b,) numpy array under the SAME key the serial path uses.
# The arithmetic is unchanged term for term; only the reduction axes gain a leading b, so
# per-member values agree with the serial ones to floating-point associativity.
#
# THE FAIL-LOUD GUARDS ARE UNCHANGED IN SEVERITY. A non-positive mean or std still RAISES
# rather than degrading to a NaN — the serial contract — and a raise here loses the whole
# batch. That is the same reach a serial raise has: neither `recover`'s Adam loop nor
# `losses/total` catches anything but `SteadyStateError`, so a serial breach also aborts the
# entire `recover()` call. The message names the offending GLOBAL member ids.

def _members_label(members, rows) -> str:
    """Human-readable member ids for a fail-loud message; falls back to stack rows."""
    if members is None:
        return f"stack row(s) {list(rows)}"
    return f"batch member(s) {[int(members[r]) for r in rows]}"


def raps_torch_batched(fields: torch.Tensor, L: float) -> tuple[np.ndarray, torch.Tensor]:
    """`raps_torch` over a (b, H, W) stack. Returns (k_centers, power (b, n_bins)).

    Shares `_raps_bins`' cached F-D6-1 binning, so the batched and serial forms cannot drift
    on bin edges. The scatter accumulates along the bin axis of a (b, n_bins) buffer instead
    of a (n_bins,) one; on CUDA `scatter_add_` uses atomics and is therefore not
    bit-reproducible in EITHER form, batched or serial."""
    if fields.shape[-2] != fields.shape[-1]:
        raise ValueError(f"raps_torch_batched: fields must be square, got "
                         f"{tuple(fields.shape)}")
    b, n = fields.shape[0], fields.shape[-1]
    idx, counts, k_centers = _raps_bins(n, L, fields.device, fields.dtype)
    f = fields - fields.mean(dim=(-2, -1), keepdim=True)
    P = torch.abs(torch.fft.fft2(f)) ** 2
    power = torch.zeros(b, k_centers.shape[0], dtype=fields.dtype, device=fields.device)
    power = power.scatter_add_(1, idx.unsqueeze(0).expand(b, -1), P.reshape(b, -1))
    power = power / counts
    power = power.clone()
    power[:, 0] = 0.0
    return k_centers, power


def spectral_block_torch_batched(fields: torch.Tensor, nblk: int = 24) -> torch.Tensor:
    """`spectral_block_torch` over a (b, H, W) stack. Returns (b, nblk, nblk).

    Same arithmetic and same ORDER (normalise by the CENTRAL BLOCK's own power, then log1p,
    then sum-normalise); every reduction stops at the leading member axis."""
    f = fields - fields.mean(dim=(-2, -1), keepdim=True)
    P = torch.abs(torch.fft.fftshift(torch.fft.fft2(f), dim=(-2, -1))) ** 2
    H, W = P.shape[-2], P.shape[-1]
    top, left = H // 2 - nblk // 2, W // 2 - nblk // 2
    block = P[..., top:top + nblk, left:left + nblk]
    total = block.sum(dim=(-2, -1), keepdim=True)
    block = torch.log1p(block / total)
    return block / block.sum(dim=(-2, -1), keepdim=True)


def spec_shape_batched(u_star: torch.Tensor, targets: dict, cfg: SpectralConfig,
                       members=None) -> tuple[torch.Tensor, dict]:
    """`spec_shape` over a (b, N, n, n) stack. Returns ((b,), parts of (b,) arrays)."""
    k_centers, power = raps_torch_batched(u_star[:, 0], targets["L"])
    if k_centers.shape != targets["k_centers"].shape:
        raise ValueError(
            "u_star's spatial grid does not match the observed frame's grid — RAPS bin "
            f"counts differ ({k_centers.shape[0]} vs {targets['k_centers'].shape[0]})")
    band_power = power.index_select(1, targets["band_idx"].to(power.device))
    norm = band_power.sum(dim=-1, keepdim=True)
    bad = ~torch.isfinite(norm.reshape(-1)) | (norm.reshape(-1) <= 0.0)
    if bool(bad.any()):
        rows = torch.nonzero(bad).reshape(-1).tolist()
        raise ValueError(
            f"spec_shape_batched: u_star channel 0 carries no finite power in B_train for "
            f"{_members_label(members, rows)} — the solved pattern has no spectral content "
            "in the fitting band; this should have been caught by the pattern_floor gate.")
    s = band_power / norm
    t = targets["raps_band_target"].to(s.device).unsqueeze(0)
    val = ((torch.log(s + 1e-300) - torch.log(t + 1e-300)) ** 2).sum(dim=-1)
    return val, dict(spec_shape_raw=val.detach().cpu().numpy())


def spec_aniso_batched(u_star: torch.Tensor, targets: dict, cfg: SpectralConfig,
                       members=None) -> tuple[torch.Tensor, dict]:
    """`spec_aniso` over a (b, N, n, n) stack. SQUARED form, as serially."""
    blk = spectral_block_torch_batched(u_star[:, 0], cfg.nblk)
    tgt = targets["block_target"].to(blk.device).unsqueeze(0)
    val = ((blk - tgt) ** 2).mean(dim=(-2, -1))
    return val, dict(spec_aniso_raw=val.detach().cpu().numpy())


def spec_amp_mean_batched(u_star: torch.Tensor, targets: dict, cfg: SpectralConfig,
                          members=None) -> tuple[torch.Tensor, dict]:
    """`spec_amp_mean` over a (b, N, n, n) stack, summed over `cfg.channels`."""
    total = u_star.new_zeros(u_star.shape[0])
    parts: dict = {}
    for c in cfg.channels:
        m = u_star[:, c].mean(dim=(-2, -1))
        bad = ~torch.isfinite(m) | (m <= 0.0)
        if bool(bad.any()):
            rows = torch.nonzero(bad).reshape(-1).tolist()
            raise ValueError(
                f"spec_amp_mean_batched: u_star channel {c} has a non-positive/non-finite "
                f"mean for {_members_label(members, rows)} — the solved pattern left the "
                "physical regime.")
        term = (torch.log(m) - targets["log_mean"][c]) ** 2
        total = total + term
        parts[f"spec_amp_mean_c{c}"] = term.detach().cpu().numpy()
    return total, parts


def spec_amp_fluct_batched(u_star: torch.Tensor, targets: dict, cfg: SpectralConfig,
                           members=None) -> tuple[torch.Tensor, dict]:
    """`spec_amp_fluct` over a (b, N, n, n) stack, summed over `cfg.channels`."""
    total = u_star.new_zeros(u_star.shape[0])
    parts: dict = {}
    for c in cfg.channels:
        s = u_star[:, c].reshape(u_star.shape[0], -1).std(dim=-1)
        bad = ~torch.isfinite(s) | (s <= 0.0)
        if bool(bad.any()):
            rows = torch.nonzero(bad).reshape(-1).tolist()
            raise ValueError(
                f"spec_amp_fluct_batched: u_star channel {c} has a non-positive/non-finite "
                f"std for {_members_label(members, rows)} — flat or invalid channel reached "
                "the spectral terms.")
        term = (torch.log(s) - targets["log_std"][c]) ** 2
        total = total + term
        parts[f"spec_amp_fluct_c{c}"] = term.detach().cpu().numpy()
    return total, parts


def real_moments_batched(u_star: torch.Tensor, targets: dict, cfg: SpectralConfig,
                         members=None) -> tuple[torch.Tensor, dict]:
    """`real_moments` over a (b, N, n, n) stack, summed over `cfg.channels`."""
    total = u_star.new_zeros(u_star.shape[0])
    parts: dict = {}
    for c in cfg.channels:
        v = u_star[:, c] - u_star[:, c].mean(dim=(-2, -1), keepdim=True)
        skew = (v ** 3).mean(dim=(-2, -1)) / (
            (v ** 2).mean(dim=(-2, -1)).clamp_min(1e-300) ** 1.5)
        term = (skew - targets["skew"][c]) ** 2
        total = total + term
        parts[f"real_moments_c{c}"] = term.detach().cpu().numpy()
    return total, parts


_TERM_FNS_BATCHED = dict(spec_shape=spec_shape_batched, spec_aniso=spec_aniso_batched,
                         spec_amp_mean=spec_amp_mean_batched,
                         spec_amp_fluct=spec_amp_fluct_batched,
                         real_moments=real_moments_batched)
if tuple(_TERM_FNS_BATCHED) != SPECTRAL_TERM_KEYS:
    # Import-time, because a drift here would show up as a silently missing term rather
    # than an error: `losses/total.py` keys its NaN placeholders off SPECTRAL_TERM_KEYS.
    raise RuntimeError(
        f"batched spectral term table {tuple(_TERM_FNS_BATCHED)} does not match the serial "
        f"one {SPECTRAL_TERM_KEYS} — the two must stay in lockstep, in the same order")


def spectral_terms_batched(u_star: torch.Tensor, targets: dict, cfg: SpectralConfig,
                           members=None) -> tuple[dict, dict]:
    """`spectral_terms` over a (b, N, n, n) stack. Returns (term_vals of (b,) tensors,
    parts of (b,) numpy arrays). `members` labels the stack's rows with their global batch
    member ids, for fail-loud messages only."""
    term_vals: dict = {}
    parts: dict = {}
    for name, fn in _TERM_FNS_BATCHED.items():
        val, p = fn(u_star, targets, cfg, members=members)
        term_vals[name] = val
        parts.update(p)
    return term_vals, parts


def is_ignited_batched(parts: dict, margin: float) -> np.ndarray:
    """Per-member `is_ignited`: a (B,) bool array from the (B,) batched hinge diagnostics.

    COSTS NO EXTRA SYNC. `losses.terms.turing_hinges_split_batched` already materialises
    `sig_max_pos` on the host as a numpy array, and `compute_terms_batched` already
    materialises `ss_converged`; the ignition decision is therefore taken on quantities that
    have crossed the device boundary anyway. Selecting the ignited subset needs an integer
    index list, which is the one host-side quantity the batched solver cannot avoid.

    Raises KeyError if `sig_max_pos` is absent, for the same reason the serial function does:
    that means `split_hinges=False`, a misconfiguration for the spectral path, not a silent
    "not ignited"."""
    if "sig_max_pos" not in parts:
        raise KeyError(
            "is_ignited_batched requires parts['sig_max_pos'] (from "
            "losses.terms.turing_hinges_split_batched, via losses.total.compute_terms_batched "
            "with split_hinges=True). Its absence means split hinges are off for this run — a "
            "misconfiguration, not a silent 'not ignited'.")
    conv = np.asarray(parts["ss_converged"], dtype=bool)
    return conv & (np.asarray(parts["sig_max_pos"], dtype=float) > margin)


# --------------------------------------------------------------------------------------
# wiring context (unit U4, losses/total.py::compute_terms)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpectralContext:
    """What `losses.total` needs to attempt the spectral terms. Built by `recover.py`:
    `targets`/`cfg` are constructed ONCE per `recover()` call (the observed frame does not
    change across restarts); `solver` owns the warm-start state and is therefore per-run.

    `solver` is typed `object` rather than imported from `forward.py` to avoid this
    (recovery-side) module depending on the forward-solve module. The duck-typed interface
    depends on which assembler consumes the context, and only ONE of the two is ever called:

      * serial (`compute_terms`)  — `forward.PatternSolver`, one per restart (it owns that
        restart's warm start), called as `.solve() -> (u_star | None, reason)`;
      * batched (`compute_terms_batched`) — `forward.BatchedPatternSolver`, ONE for the whole
        batch (it owns per-MEMBER warm state keyed by global member index), called as
        `.solve_subset(members, xstar_batch) -> (u_stack | None, ok_members, reasons)`.

    A test stub satisfying whichever method its assembler calls is a legal `solver`.
    """
    solver: object
    targets: dict
    cfg: SpectralConfig
