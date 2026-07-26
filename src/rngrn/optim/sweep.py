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

# Experiment-ARM axes. These are NOT tuning knobs: they select which problem instance or
# which observation regime a run belongs to (e.g. rotating which channel is hidden, or
# stepping through dataset samples). They are legal sweep axes because an identifiability
# study must repeat the SAME config across arms — but they are deliberately named so a
# reader can tell an arm axis from a tuning axis in the run index.
#
# Data-GENERATION params (the ones that change the underlying system: system, L, resolution,
# T_max, dt, seed) remain forbidden — varying those means you are comparing different
# problems, not different methods.
_ARM_AXES = ("data.sample_key", "data.dataset_id", "model.observed_idx", "model.m", "model.N")

_FORBIDDEN_DATA = ("data.system", "data.L", "data.L_override",
                   "data.resolution", "data.T_max", "data.dt",
                   "data.seed", "data.cache_root", "data.datasets_root", "data.source")


def _check_axis(key: str):
    if key in _ARM_AXES:
        return                      # explicit experiment-arm axis — allowed
    if key in _FORBIDDEN_DATA or key.startswith("data."):
        raise ValueError(
            f"sweep axis '{key}' is a data-generation param — kept off the tuning axis "
            "by design (it defines WHICH problem, so varying it compares different problems "
            f"rather than different methods). Legal experiment-arm axes: {list(_ARM_AXES)}.")


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
