"""spec.py — the DatasetSpec and its content hash (dataset id).

A dataset's identity is a hash of the full spec that generates it. Same spec ->
same hash -> cache hit -> no regeneration. This is the first of the three IDs.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass(frozen=True)
class DatasetSpec:
    system: str                     # rd_systems registry key
    coefficients: tuple = ()        # ((name, value), ...) sorted — the generator params
    Du: float = 1.0
    Dv: float = 40.0
    L: float = 100.0
    resolution: int = 128
    T_max: float = 4000.0
    dt: float = 0.1
    n_save: int = 1
    seed: int = 0
    noise_init: float = 1e-2

    def canonical(self) -> str:
        d = asdict(self)
        d["coefficients"] = sorted(map(list, self.coefficients))
        return json.dumps(d, sort_keys=True)

    def hash(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:16]


def spec_from_config(data_cfg, system_obj) -> DatasetSpec:
    """Build a DatasetSpec from a DataConfig + an instantiated reference system."""
    coeffs = tuple(sorted(system_obj.coefficients().items()))
    Du, Dv = system_obj.D
    return DatasetSpec(system=data_cfg.system, coefficients=coeffs, Du=Du, Dv=Dv,
                       L=data_cfg.L, resolution=data_cfg.resolution,
                       T_max=data_cfg.T_max, dt=data_cfg.dt, seed=data_cfg.seed)
