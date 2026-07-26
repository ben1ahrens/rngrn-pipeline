"""gate.py — THE FIREWALL GATE.

This is the single doorway between the answer-key side and the recovery side. It
takes a cached dataset (or a 3-gene HDF5 sample) and returns exactly two objects:

    RecoveryInput  : (frame, L, observed_idx)   -> handed to recovery. Nothing else.
    AnswerKey      : (x_star, J, kstar, ...)     -> handed to SCORING only.

The contract enforced elsewhere (tests/test_firewall.py): no recovery-side module
(recover, train, losses, model, observables, optim, eval rollout) imports this
module's AnswerKey, rd_models, solver, or cache answer-key fields. gate.py itself
imports from the answer-key side, so gate.py is NOT a recovery-side module — it is
the boundary, called by the validation harness which sits astride the firewall.
"""
from __future__ import annotations
import json
import os
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
    """The quarantined truth. Passed ONLY to scoring, NEVER to recovery."""
    x_star: Optional[np.ndarray]
    J: Optional[np.ndarray]
    kstar: Optional[float]
    sigma_max: Optional[float]
    D: Optional[np.ndarray]
    coefficients: Optional[dict]
    n_species_true: int


def _observe(frame_full, observed_idx):
    idx = list(observed_idx)
    return frame_full[idx].copy()


def from_cache(cache_root, dataset_hash, N, observed_idx):
    """Split a cached reference dataset into (RecoveryInput, AnswerKey)."""
    d = os.path.join(cache_root, dataset_hash)
    with h5py.File(os.path.join(d, "payload.h5"), "r") as f:
        frame_full = f["final_frame"][:]
        L = float(f.attrs["L"])
        n_true = int(f.attrs["n_species"])
        ak = f["answer_key"]
        key = AnswerKey(
            x_star=ak["x_star"][:] if "x_star" in ak else None,
            J=ak["J"][:] if "J" in ak else None,
            kstar=float(ak.attrs.get("kstar")) if "kstar" in ak.attrs else None,
            sigma_max=float(ak.attrs.get("sigma_max")) if "sigma_max" in ak.attrs else None,
            D=ak["D"][:] if "D" in ak else None,
            coefficients=json.loads(ak.attrs["coefficients"]) if "coefficients" in ak.attrs else None,
            n_species_true=n_true,
        )
    ri = RecoveryInput(frame=_observe(frame_full, observed_idx), L=L,
                       observed_idx=tuple(observed_idx), N=N)
    return ri, key


def from_registry(datasets_root, dataset_id, sample_key, N, observed_idx, L,
                  backend="jsonl"):
    """Split a REGISTERED dataset sample into (RecoveryInput, AnswerKey).

    Loads <datasets_root>/<dataset_id>/payload.h5, reads the manifest to know which
    HDF5 key is the observable vs which are answer-key quantities, and quarantines the
    latter. This is the load-by-name path for datasets generated/downloaded elsewhere.
    """
    from . import registry as reg
    man = reg.load_manifest(datasets_root, dataset_id)
    obs_key = man.get("observable_key", "final_frame")
    ak_keys = set(man.get("answer_key_keys", []))
    with h5py.File(reg.payload_path(datasets_root, dataset_id), "r") as f:
        g = f[sample_key]
        frame_full = g[obs_key][:]
        def _get(k):
            return g[k][:] if k in g and k in ak_keys else None
        J = _get("jacobian")
        xstar = _get("x_star")
        D = _get("D")
        inter = _get("interaction_matrix")
        kstar = float(g.attrs["kstar"]) if "kstar" in getattr(g, "attrs", {}) else None
        n_true = frame_full.shape[0]
    coeffs = {"interaction_matrix": inter.tolist()} if inter is not None else None
    key = AnswerKey(x_star=xstar, J=J, kstar=kstar, sigma_max=None, D=D,
                    coefficients=coeffs, n_species_true=n_true)
    ri = RecoveryInput(frame=_observe(frame_full, observed_idx), L=float(L),
                       observed_idx=tuple(observed_idx), N=N)
    return ri, key


def from_3gene_hdf5(path, sample_key, N, observed_idx, L):
    """Split a 3-gene dataset sample (train/val/test.h5) into (RecoveryInput, AnswerKey).

    The 3-gene contract: <sample>/final_frame (3,H,W) is the OBSERVABLE; jacobian,
    x_star, interaction_matrix, D are ANSWER-KEY quantities (quarantined here)."""
    with h5py.File(path, "r") as f:
        g = f[sample_key]
        frame_full = g["final_frame"][:]
        J = g["jacobian"][:] if "jacobian" in g else None
        xstar = g["x_star"][:] if "x_star" in g else None
        D = g["D"][:] if "D" in g else None
        inter = g["interaction_matrix"][:] if "interaction_matrix" in g else None
        n_true = frame_full.shape[0]
    key = AnswerKey(x_star=xstar, J=J, kstar=None, sigma_max=None, D=D,
                    coefficients={"interaction_matrix": inter.tolist()} if inter is not None else None,
                    n_species_true=n_true)
    ri = RecoveryInput(frame=_observe(frame_full, observed_idx), L=float(L),
                       observed_idx=tuple(observed_idx), N=N)
    return ri, key
