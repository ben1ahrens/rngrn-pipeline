"""gate.py — THE FIREWALL GATE.

This is the single doorway between the answer-key side and the recovery side. It
takes a cached dataset (or a 3-gene HDF5 sample) and returns exactly two objects:

    RecoveryInput  : (frame, L, observed_idx)   -> handed to recovery. Nothing else.
    AnswerKey      : (x_star, J, kstar, kstar_fft, ...) -> handed to SCORING only.

`L` is on the RECOVERY side (it is the physical size of the image, readable off the
experiment, not a fact about the generator's kinetics) and every loader reads it from the
sample rather than from the caller's config — one config value cannot be correct for a
dataset whose samples span L = 18 to 208. A caller-supplied `L` is a cross-check only: it
warns and loses to the file. A sample missing `L` or `k_star` raises.

The contract enforced elsewhere (tests/test_firewall.py): no recovery-side module
(recover, train, losses, model, observables, optim, eval rollout) imports this
module's AnswerKey, rd_models, solver, or cache answer-key fields. gate.py itself
imports from the answer-key side, so gate.py is NOT a recovery-side module — it is
the boundary, called by the validation harness which sits astride the firewall.
"""
from __future__ import annotations
import json
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np


@dataclass(frozen=True)
class RecoveryInput:
    """EVERYTHING recovery is allowed to see. Deliberately minimal."""
    frame: np.ndarray            # (m, H, W) — only the OBSERVED channels
    L: float
    observed_idx: tuple          # which model indices these m channels correspond to
    N: int                       # assumed total species (a modelling choice, not truth)


@dataclass(frozen=True)
class AnswerKey:
    """The quarantined truth. Passed ONLY to scoring, NEVER to recovery.

    Two distinct wavenumbers are carried, and they are NOT interchangeable:

    * ``kstar``     — the LINEAR k*: argmax_k of sigma(k) from the generator's J and D.
                      This is the headline scoring target (``kstar_rel_err``), because it
                      is the like-for-like comparison against the recovered model's own
                      dispersion-relation argmax.
    * ``kstar_fft`` — the wavenumber MEASURED from the generated frame by FFT. A
                      finite-grid measurement of the realised pattern, so it disagrees with
                      ``kstar`` by a QUANTISATION offset of EITHER SIGN, not a systematic
                      bias: across the 287 registered samples ``k_star_fft * L / 2pi``
                      always lands on the half-integer FFT-bin grid, so the ratio
                      ``kstar_fft / kstar`` takes discrete values (measured range
                      0.417-1.583, median 1.083; 68% of samples above 1, but
                      ``three_gene_val`` has a median of 0.918, i.e. below). Per-sample
                      |ratio - 1| has median 0.084 and 90th percentile 0.250.
                      SECONDARY diagnostic only (``kstar_fft_rel_err``).

    Both are read verbatim from the sample's stored attributes (``k_star`` /
    ``k_star_fft``); neither is recomputed here.
    """
    x_star: Optional[np.ndarray]
    J: Optional[np.ndarray]
    kstar: Optional[float]
    kstar_fft: Optional[float]
    sigma_max: Optional[float]
    D: Optional[np.ndarray]
    coefficients: Optional[dict]
    n_species_true: int
    # unit 12 (L-generalisation). The label that says WHICH generator system a sample came
    # from, read verbatim from the sample's optional `system_id` attribute. It exists so
    # scoring can group the SAME system observed at SEVERAL domain sizes (the
    # three_gene_multiL layout) and ask whether recovery agreed with itself across them.
    # It is truth-side metadata and therefore quarantined here with the rest of the key --
    # recovery never sees it. None when the dataset carries no such attribute, which is the
    # case for every dataset registered before three_gene_multiL; absence is legitimate and
    # simply means no cross-L grouping is possible for that sample.
    system_id: Optional[str] = None


def _system_id(attrs):
    """Read the optional cross-L grouping label off a sample's attributes.

    HDF5 hands string attributes back as bytes or numpy str depending on how they were
    written, so normalise to a plain str. Returns None when absent -- see
    ``AnswerKey.system_id`` for why absence is legitimate rather than an error.
    """
    v = attrs.get("system_id")
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return str(v)


def _observe(frame_full, observed_idx):
    idx = list(observed_idx)
    return frame_full[idx].copy()


def _require_attr(attrs, name, where):
    """Read a required scalar sample attribute, or RAISE.

    House style is fail-loud: a sample with no domain size or no answer-key wavenumber
    cannot be scored on wavelength at all, and defaulting either one silently produces a
    number that looks like a result but is not one.
    """
    if name not in attrs:
        raise KeyError(
            f"{where}: required attribute {name!r} is missing. The gate will not default "
            f"it — the domain size and the answer-key wavenumber are per-sample facts, and "
            f"substituting a config value would make kstar_rel_err meaningless. Present "
            f"attributes: {sorted(map(str, attrs.keys()))}")
    return float(attrs[name])


def _resolve_L(attrs, L_arg, where):
    """The domain size is a property of the SAMPLE, not of the caller's config.

    Contract (documented because it is a real decision, not an implementation detail):

    * ``L_arg is None``            -> use the file's L. This is the normal path.
    * ``L_arg`` agrees with file   -> use the file's L, silently.
    * ``L_arg`` DISAGREES with file-> warn loudly and PREFER THE FILE. The caller's value
      is treated as a stale config default, never as an override of measured geometry;
      silently honouring it would rescale every wavenumber in the run.

    A missing file L raises (see ``_require_attr``) rather than falling back to ``L_arg``.
    """
    L_file = _require_attr(attrs, "L", where)
    if L_arg is not None and not np.isclose(float(L_arg), L_file, rtol=1e-6, atol=0.0):
        warnings.warn(
            f"{where}: caller passed L={float(L_arg)!r} but the sample stores "
            f"L={L_file!r}. PREFERRING THE FILE — the domain size is a property of the "
            f"sample. Remove L from the config to silence this.",
            UserWarning, stacklevel=3)
    return L_file


def from_cache(cache_root, dataset_hash, N, observed_idx):
    """Split a cached reference dataset into (RecoveryInput, AnswerKey).

    The cache's own schema stores the LINEAR k* under the answer-key group attribute
    ``kstar`` (written by ``rd_models.System.answer_key()``); it stores no FFT-measured
    wavenumber, so ``AnswerKey.kstar_fft`` is None on this path. ``L`` and ``kstar`` are
    both required — the generator always writes them, so their absence means a corrupt
    or foreign payload and must not be papered over.
    """
    d = os.path.join(cache_root, dataset_hash)
    with h5py.File(os.path.join(d, "payload.h5"), "r") as f:
        frame_full = f["final_frame"][:]
        L = _require_attr(dict(f.attrs), "L", f"cache/{dataset_hash}")
        n_true = int(f.attrs["n_species"])
        ak = f["answer_key"]
        key = AnswerKey(
            x_star=ak["x_star"][:] if "x_star" in ak else None,
            J=ak["J"][:] if "J" in ak else None,
            kstar=_require_attr(dict(ak.attrs), "kstar",
                                f"cache/{dataset_hash}/answer_key"),
            kstar_fft=None,   # the reference cache stores no FFT measurement
            sigma_max=float(ak.attrs.get("sigma_max")) if "sigma_max" in ak.attrs else None,
            D=ak["D"][:] if "D" in ak else None,
            coefficients=json.loads(ak.attrs["coefficients"]) if "coefficients" in ak.attrs else None,
            n_species_true=n_true,
            # the reference cache has no multi-L layout, so this is None in practice; read
            # it anyway so a future cache that grows the attribute is picked up for free.
            system_id=_system_id(dict(f.attrs)),
        )
    ri = RecoveryInput(frame=_observe(frame_full, observed_idx), L=L,
                       observed_idx=tuple(observed_idx), N=N)
    return ri, key


def from_registry(datasets_root, dataset_id, sample_key, N, observed_idx, L=None,
                  backend="jsonl"):
    """Split a REGISTERED dataset sample into (RecoveryInput, AnswerKey).

    Loads <datasets_root>/<dataset_id>/payload.h5, reads the manifest to know which
    HDF5 key is the observable vs which are answer-key quantities, and quarantines the
    latter. This is the load-by-name path for datasets generated/downloaded elsewhere.

    Per-sample geometry and wavenumbers come from the FILE, following the precedent set by
    ``from_cache``:

    * ``L``          — read from the sample's ``L`` attribute. ``L`` here is an optional
                       CROSS-CHECK, not an override: if it disagrees with the file the
                       call warns and prefers the file (see ``_resolve_L``). Every sample
                       in these datasets has its own domain size, so a single config value
                       cannot be right for more than one of them.
    * ``k_star``     — the linear answer-key wavenumber -> ``AnswerKey.kstar``.
    * ``k_star_fft`` — the FFT-measured wavenumber -> ``AnswerKey.kstar_fft`` (secondary).

    A sample missing ``L`` or ``k_star`` RAISES. ``k_star_fft`` is optional (it is a
    diagnostic, not the headline) and is None when absent.
    """
    from . import registry as reg
    man = reg.load_manifest(datasets_root, dataset_id)
    obs_key = man.get("observable_key", "final_frame")
    ak_keys = set(man.get("answer_key_keys", []))
    where = f"{dataset_id}/{sample_key}"
    with h5py.File(reg.payload_path(datasets_root, dataset_id), "r") as f:
        g = f[sample_key]
        attrs = dict(g.attrs)
        frame_full = g[obs_key][:]
        def _get(k):
            return g[k][:] if k in g and k in ak_keys else None
        J = _get("jacobian")
        xstar = _get("x_star")
        D = _get("D")
        inter = _get("interaction_matrix")
        L_used = _resolve_L(attrs, L, where)
        # NOTE the attribute is 'k_star' (underscore). An earlier version of this function
        # read 'kstar', which no payload carries, so AnswerKey.kstar was always None and
        # kstar_rel_err was always NaN. Do not "simplify" this name.
        kstar = _require_attr(attrs, "k_star", where)
        kstar_fft = float(attrs["k_star_fft"]) if "k_star_fft" in attrs else None
        sigma_max = float(attrs["sigma_max"]) if "sigma_max" in attrs else None
        n_true = frame_full.shape[0]
    coeffs = {"interaction_matrix": inter.tolist()} if inter is not None else None
    key = AnswerKey(x_star=xstar, J=J, kstar=kstar, kstar_fft=kstar_fft,
                    sigma_max=sigma_max, D=D,
                    coefficients=coeffs, n_species_true=n_true,
                    system_id=_system_id(attrs))
    ri = RecoveryInput(frame=_observe(frame_full, observed_idx), L=L_used,
                       observed_idx=tuple(observed_idx), N=N)
    return ri, key


def from_3gene_hdf5(path, sample_key, N, observed_idx, L=None):
    """Split a 3-gene dataset sample (train/val/test.h5) into (RecoveryInput, AnswerKey).

    The 3-gene contract: <sample>/final_frame (3,H,W) is the OBSERVABLE; jacobian,
    x_star, interaction_matrix, D are ANSWER-KEY quantities (quarantined here).

    These loose splits are the same bytes that get copied to
    ``data/datasets/three_gene_*/payload.h5``, so the same per-sample attribute contract
    applies: ``L`` and ``k_star`` are read from the sample and are REQUIRED; ``L`` passed
    by the caller is a cross-check that warns and loses to the file.
    """
    where = f"{os.path.basename(path)}/{sample_key}"
    with h5py.File(path, "r") as f:
        g = f[sample_key]
        attrs = dict(g.attrs)
        frame_full = g["final_frame"][:]
        J = g["jacobian"][:] if "jacobian" in g else None
        xstar = g["x_star"][:] if "x_star" in g else None
        D = g["D"][:] if "D" in g else None
        inter = g["interaction_matrix"][:] if "interaction_matrix" in g else None
        L_used = _resolve_L(attrs, L, where)
        kstar = _require_attr(attrs, "k_star", where)
        kstar_fft = float(attrs["k_star_fft"]) if "k_star_fft" in attrs else None
        sigma_max = float(attrs["sigma_max"]) if "sigma_max" in attrs else None
        n_true = frame_full.shape[0]
    key = AnswerKey(x_star=xstar, J=J, kstar=kstar, kstar_fft=kstar_fft,
                    sigma_max=sigma_max, D=D,
                    coefficients={"interaction_matrix": inter.tolist()} if inter is not None else None,
                    n_species_true=n_true, system_id=_system_id(attrs))
    ri = RecoveryInput(frame=_observe(frame_full, observed_idx), L=L_used,
                       observed_idx=tuple(observed_idx), N=N)
    return ri, key
