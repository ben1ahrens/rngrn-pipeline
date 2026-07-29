"""config.py — typed, hierarchical, hashable config-as-experiment.

Every knob that affects a result lives here, split by concern. A config is
loaded from YAML (base + override fragments), validated on load (unknown keys
raise), and FROZEN into each run's output dir. Two of the three orthogonal IDs
are derived here:
  * dataset id  — hash of a DatasetSpec (data/spec.py)
  * config id   — hash of the canonicalised full config (this module)
  * run id      — name_timestamp (io.py)

Dotted-key CLI overrides ("train.adam_lr=0.02") are supported via apply_overrides.
"""
from __future__ import annotations
import dataclasses as dc
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import yaml


# --------------------------------------------------------------------------------------
# per-concern config sections
# --------------------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Which frame(s) recovery trains on. On the ANSWER-KEY side of the firewall:
    the generation spec is here, but only (frame, L, observed_idx) reach recovery."""
    source: str = "reference"          # 'reference'|'cache'|'registry'|'hdf5_3gene'
    system: str = "schnakenberg"       # reference system name (rd_models registry)
    dataset_hash: Optional[str] = None # resolved content hash (filled by generate-data)
    cache_root: str = "data/cache"
    datasets_root: str = "data/datasets"  # the dataset REGISTRY (stored/downloaded sets)
    dataset_id: Optional[str] = None   # for source='registry': registered dataset name
    hdf5_path: Optional[str] = None    # for source='hdf5_3gene'
    sample_key: Optional[str] = None   # which sample group in the hdf5/registry payload
    # `L` is a GENERATOR parameter: it sizes the domain for source='reference' (and feeds
    # the DatasetSpec hash). It is NOT the domain size of a loaded sample — file-backed
    # sources (registry, cache, hdf5_3gene) read L from the sample itself, because each
    # sample has its own. Do not add `L:` to a file-backed config expecting it to apply.
    L: float = 100.0
    # Optional cross-check for file-backed sources: when set, the gate compares it to the
    # sample's stored L and warns loudly if they disagree — then uses the FILE's value.
    # Left None, no check is made. It is never an override of measured geometry.
    L_override: Optional[float] = None
    resolution: int = 128
    T_max: float = 4000.0      # generator horizon (answer-key side; OFF the tuning axis)
    dt: float = 0.1
    seed: int = 0


@dataclass
class ModelConfig:
    architecture: str = "gated_promoter"   # models registry key
    N: int = 2                              # total species
    m: int = 2                              # observed species (m<=N)
    form: str = "competitive"               # 'competitive' | 'nc1'
    n_hill: int = 2
    seed: int = 0
    observed_idx: Optional[list] = None     # which model indices the m rows map to
    nondim: bool = False                    # recover on the unit box x/L; see recover.py  # unit 12

    dispersion_backend: str = "eig"         # 'eig'|'cubic' ('cubic' exact for N==3 only) # unit 10

    init: str = "default"                   # 'default' | 'low_basal' -- see model.py (unit 2)

    d_init_from_kstar: bool = False         # opt-in D init from k*_obs, L-free  # unit B4


@dataclass
class LossConfig:
    """Weights and the weighting strategy. Weights are STARTING points — Claude Code
    tunes them (see TUNING.md). Strategy is a key into the weighting registry."""
    # starting weights (rngrn_design_doc.md §5): morphology ~0.1 per §5.6. NOTE morphology
    # is currently a non-differentiable post-hoc diagnostic (losses/terms.morphology_
    # consistency) and is NOT yet in the differentiable sum in losses/total.compute_terms,
    # so this weight is inert until Claude Code wires a differentiable morphology term.
    # `resid` defaults to 0.0 — SETTLED OFF, not merely untuned. exp06 swept pixel batch
    # {64,128,512} x weight {1,3,10}, 8 seeds per cell: all nine cells collapsed to 1/8
    # Turing seeds with best median k* error 11.8 %, against 0.4 % with the residual off.
    # `anchor` (2.0) is the frame-scale anchor promoted from exp05; see TUNING.md [TUNE].
    weights: dict = field(default_factory=lambda: dict(
        kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5, anchor=2.0, morphology=0.1,
        param_prior=0.0))                   # param_prior default 0.0: opt-in (unit 5)
    strategy: str = "fixed"                 # 'fixed' | 'scheduled' | 'ratio' | 'gradnorm' | 'ntk'
    tau: float = 0.12                        # k* tolerance band
    jac_floor: float = 1.0                   # anti-collapse ||J|| floor
    split_hinges: bool = True                # disjoint-support Turing hinges  # unit 1
    hinge_k_min_frac: float = 0.1            # instability hinge starts at this grid fraction  # unit 1
    staging_keys: list = field(default_factory=lambda: ["turing"])  # data-first staged terms  # unit 1
    staging_off_frac: float = 0.25           # staged weights held at 0 for this fraction  # unit 1
    staging_ramp_frac: float = 0.25          # then ramped 0->1 over this fraction  # unit 1
    detach_xstar: bool = False               # dispersion terms see x* as a constant  # unit 1
    dratio_centre: float = 7.5              # D-ratio prior centre, biological literature
                                             # value (Nodal/Lefty), NOT the generator's
                                             # ~135 median — see configs/bio_box.yaml # unit 5
    dratio_spread: float = 1.0              # D-ratio prior spread, natural-log units;
                                             # [TUNE], see configs/bio_box.yaml # unit 5
    bio_box_path: str = "configs/bio_box.yaml"  # source of every plausibility number # unit 5
    ratio_update_every: int = 50            # 'ratio' strategy: recompute cadence, in steps  # unit 13


@dataclass
class TrainConfig:
    n_restarts: int = 4
    adam_steps: int = 1500
    adam_lr: float = 0.05
    lbfgs_steps: int = 50
    grad_clip: float = 10.0
    seed: int = 0
    deterministic: bool = True              # unit 10: torch deterministic-algorithms mode


@dataclass
class SolverConfig:
    """Post-training rollout / stiff integration knobs."""
    integrator: str = "etdrk4"              # 'etdrk4' | 'imex_split' | 'bdf1_newton_krylov'
    n_grid: int = 128
    horizon_growth_times: float = 40.0      # T = this / sigma_max
    dt: Optional[float] = None              # None -> derived from fastest reaction rate
    noise: float = 1e-2
    robustness_samples: int = 200
    robustness_sigma_log: float = 0.1
    # ---- morphology rollout: fit() simulates the recovered model so morphology_match is
    # a real number instead of morphology_scored='target_only'. Measured 0.9-1.7 s per
    # 96x96/128x128 field; the grid and L come from the TARGET frame, not from n_grid,
    # because morphology is only comparable on a matching grid.
    morphology_rollout: bool = True          # unit 7
    morphology_integrator: str = "etdrk4_rfft"   # unit 7
    # 15000 is the WORST CASE, not the typical one: at the measured etdrk4_rfft step costs
    # (0.9 / 1.7 / 2.8 ms per step at 64 / 96 / 128) a fully-bound budget is 13 / 26 / 42 s.
    # 96x96 — the three_gene grid — therefore stays inside the 30 s target; a 128x128 target
    # would not, and would need this lowered. Typical runs are far shorter: a saturating
    # Turing model ends on the horizon at ~600 steps, and a collapsing stable one on the
    # collapse rule at ~1000 (both measured).
    morphology_max_steps: int = 15000        # unit 7
    morphology_early_stop: bool = True       # unit 7 — for the COLLAPSE stop; see rollout.py
    morphology_check_every: int = 200        # unit 7
    morphology_saturation_tol: float = 0.01  # unit 7 [TUNE] — uncalibrated stopping rule
    morphology_saturation_window: int = 5    # unit 7 [TUNE] — uncalibrated stopping rule


@dataclass
class TrackingConfig:
    backend: str = "noop"                   # 'noop' | 'wandb'
    project: str = "rngrn-inverse-turing"
    run_name: str = "run"
    index_backend: str = "jsonl"            # run index + dataset registry: 'jsonl' | 'sqlite'


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # ---- identity -----------------------------------------------------------------
    def canonical(self) -> str:
        """Canonical JSON (sorted keys) of the full config, for hashing."""
        return json.dumps(asdict(self), sort_keys=True, default=str)

    def config_id(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:12]

    def to_yaml(self, path: str):
        with open(path, "w") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False)


# --------------------------------------------------------------------------------------
# load / compose / override
# --------------------------------------------------------------------------------------
_SECTIONS = {f.name: f.type for f in dc.fields(Config)}


def _build_section(cls, d: dict):
    valid = {f.name for f in dc.fields(cls)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**d)


def from_dict(d: dict) -> Config:
    """Build a validated Config from a plain dict (fail loud on unknown keys)."""
    unknown = set(d) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Config: unknown top-level sections {sorted(unknown)}")
    kw = {}
    section_cls = {f.name: f.default_factory().__class__ for f in dc.fields(Config)}
    for name, cls in section_cls.items():
        kw[name] = _build_section(cls, d.get(name, {}) or {})
    return Config(**kw)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if (k in out and isinstance(out[k], dict)
                                            and isinstance(v, dict)) else v
    return out


def load_config(path: str) -> Config:
    """Load a YAML config with optional hierarchical composition via a top-level
    `_base_: [file1.yaml, ...]` key (merged in order, this file last / wins)."""
    import os
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    bases = raw.pop("_base_", [])
    merged: dict = {}
    for b in bases:
        bpath = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
        with open(bpath) as fh:
            merged = _deep_merge(merged, yaml.safe_load(fh) or {})
    merged = _deep_merge(merged, raw)
    return from_dict(merged)


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply dotted-key CLI overrides ('train.adam_lr=0.02'). Returns a new Config."""
    d = asdict(cfg)
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts = key.split(".")
        node = d
        for p in parts[:-1]:
            node = node[p]
        try:
            parsed = json.loads(val)
        except json.JSONDecodeError:
            parsed = val
        if parts[-1] not in node:
            raise KeyError(f"override targets unknown key: {key}")
        node[parts[-1]] = parsed
    return from_dict(d)
