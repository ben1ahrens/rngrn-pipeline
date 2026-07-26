"""cache.py — the content-addressed dataset cache (Stage 1 core).

data/cache/<dataset_hash>/
    payload.h5      # frames + trajectory + the QUARANTINED answer key
    manifest.json   # hash, provenance, shapes, checksums
    spec.yaml       # human-readable generation spec

Idempotent: resolve spec -> hash -> return on hit, simulate only on miss. Writes
go to a temp dir and atomic-rename on success, so a half-written dataset never
looks complete. Training RESOLVES AND RECORDS a dataset id; it never generates.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import tempfile

import h5py
import numpy as np
import yaml

from .spec import DatasetSpec
from .solver import simulate_to_attractor
from ..utils import provenance


def _dataset_dir(cache_root: str, h: str) -> str:
    return os.path.join(cache_root, h)


def exists(cache_root: str, spec: DatasetSpec) -> bool:
    d = _dataset_dir(cache_root, spec.hash())
    return os.path.exists(os.path.join(d, "manifest.json"))


def _checksum(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def generate(cache_root: str, spec: DatasetSpec, system, overwrite: bool = False) -> str:
    """Generate the dataset for `spec` if absent; return its directory. Idempotent."""
    h = spec.hash()
    d = _dataset_dir(cache_root, h)
    if exists(cache_root, spec) and not overwrite:
        return d

    frame, L = simulate_to_attractor(
        system, resolution=spec.resolution, T_max=spec.T_max, dt=spec.dt,
        seed=spec.seed, noise_init=spec.noise_init)
    answer_key = system.answer_key()  # QUARANTINED — never read by recovery

    tmp = tempfile.mkdtemp(prefix="rngrn_ds_")
    try:
        with h5py.File(os.path.join(tmp, "payload.h5"), "w") as f:
            f.create_dataset("final_frame", data=frame, compression="gzip")
            f.attrs["L"] = L
            f.attrs["n_species"] = frame.shape[0]
            # answer key in a SEPARATE group, clearly named, never handed to recovery
            ak = f.create_group("answer_key")
            for k, v in answer_key.items():
                if isinstance(v, (list, tuple)):
                    ak.create_dataset(k, data=np.asarray(v, float))
                elif isinstance(v, dict):
                    ak.attrs[k] = json.dumps(v)
                else:
                    ak.attrs[k] = v
        manifest = dict(
            dataset_hash=h, spec=json.loads(spec_to_json(spec)),
            shapes=dict(final_frame=list(frame.shape)),
            checksums=dict(final_frame=_checksum(frame)),
            provenance=provenance(dict(solver="imex_split_fft")),
        )
        with open(os.path.join(tmp, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        with open(os.path.join(tmp, "spec.yaml"), "w") as fh:
            yaml.safe_dump(json.loads(spec_to_json(spec)), fh, sort_keys=False)
        os.makedirs(cache_root, exist_ok=True)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.replace(tmp, d)   # atomic promotion
    finally:
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
    return d


def spec_to_json(spec: DatasetSpec) -> str:
    import dataclasses
    dd = dataclasses.asdict(spec)
    dd["coefficients"] = sorted(map(list, spec.coefficients))
    return json.dumps(dd, sort_keys=True)


def load_frame(cache_root: str, dataset_hash: str):
    """Load ONLY (frame, L). Answer key is deliberately NOT returned here — see gate.py."""
    d = _dataset_dir(cache_root, dataset_hash)
    with h5py.File(os.path.join(d, "payload.h5"), "r") as f:
        frame = f["final_frame"][:]
        L = float(f.attrs["L"])
    return frame, L
