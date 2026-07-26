"""sweep.py — the outer optimisation loop, built OUTWARD from fit().

A sweep enumerates config overrides (grid or explicit list), freezes one config
per point, calls fit() for each, and records to the run index. Seed-replicate
averaging is on by default (the inverse problem is non-identifiable; rank configs
by an aggregate over seeds, not one lucky init). Data-generation params are kept
OFF the tuning axis by construction — a sweep varies loss/train/solver, not the
target/diffusion/domain.
"""
from __future__ import annotations
import itertools
from copy import deepcopy

from ..config import Config, apply_overrides
from ..train import fit


# keys a sweep is ALLOWED to vary — everything under data.* is excluded on purpose
_TUNABLE_PREFIXES = ("loss.", "train.", "solver.", "model.form", "model.n_hill")


def _check_axis(key: str):
    if key.startswith("data."):
        raise ValueError(
            f"sweep axis '{key}' is a data-generation param — kept off the tuning axis "
            "by design (it defines WHICH problem; vary it on the benchmark axis instead).")


def grid_points(axes: dict) -> list[dict]:
    """axes: {dotted_key: [values]} -> list of override dicts (cartesian product)."""
    for k in axes:
        _check_axis(k)
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*[axes[k] for k in keys])]


def run_sweep(base_cfg: Config, axes: dict, seeds=(0,), runs_root="experiments",
              name="sweep", verbose=False) -> list[dict]:
    """Run fit() over the cartesian product of `axes` x `seeds`. Returns metric rows."""
    points = grid_points(axes)
    rows = []
    for pi, point in enumerate(points):
        overrides = [f"{k}={_as_literal(v)}" for k, v in point.items()]
        for seed in seeds:
            cfg = apply_overrides(deepcopy(base_cfg), overrides + [f"train.seed={seed}"])
            metric = fit(cfg, runs_root=runs_root,
                         run_id=None, verbose=verbose)
            metric.update(point); metric["seed"] = seed; metric["point_idx"] = pi
            rows.append(metric)
    return rows


def _as_literal(v):
    import json
    return json.dumps(v)
