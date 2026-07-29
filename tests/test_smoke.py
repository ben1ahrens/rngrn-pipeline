"""test_smoke.py — the seconds-long end-to-end dry run.

Exercises the WHOLE pipeline wired together at tiny scale (tiny grid, 1 restart, a
handful of steps) so a broken wire fails immediately. It does NOT check recovery
quality — that is the milestone tuning Claude Code does on CUDA.

  generate-data -> firewall gate -> recover -> score -> run index -> benchmark
  + lift-and-simulate rollout under ETDRK4
"""
import os
import tempfile

import numpy as np
import pytest

from rngrn.config import load_config, apply_overrides


CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")


def _tiny(cfg, form="competitive"):
    return apply_overrides(cfg, [
        "data.resolution=32", "data.T_max=5.0", "data.dt=0.05",
        "model.N=2", "model.m=2", f"model.form={form}",
        "train.n_restarts=1", "train.adam_steps=8", "train.lbfgs_steps=0",
        "solver.n_grid=32", "solver.robustness_samples=5",
    ])


def test_config_loads_and_hashes():
    cfg = load_config(os.path.join(CONFIGS, "milestone1_schnak.yaml"))
    assert cfg.model.N == 2 and cfg.data.system == "schnakenberg"
    assert len(cfg.config_id()) == 12


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_end_to_end_reference(tmp_path, form):
    from rngrn.train import fit
    cfg = _tiny(load_config(os.path.join(CONFIGS, "milestone1_schnak.yaml")), form=form)
    cfg = apply_overrides(cfg, [f"data.cache_root={str(tmp_path/'cache')}"])
    # keep the tiny generator fast: short horizon (dry-run only, not a converged pattern)
    metric = fit(cfg, runs_root=str(tmp_path / "experiments"))
    assert "kstar_model" in metric and "recovered_turing" in metric
    assert os.path.exists(tmp_path / "experiments" / "runs.jsonl")


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_benchmark_reads_index(tmp_path, form):
    from rngrn.train import fit
    from rngrn.optim.benchmark import build_table
    cfg = _tiny(load_config(os.path.join(CONFIGS, "milestone1_schnak.yaml")), form=form)
    cfg = apply_overrides(cfg, [f"data.cache_root={str(tmp_path/'cache')}"])
    fit(cfg, runs_root=str(tmp_path / "experiments"))
    table = build_table(runs_root=str(tmp_path / "experiments"))
    assert len(table) >= 1
    assert "kstar_rel_err_mean" in table[0]


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_rollout_etdrk4_runs(form):
    from rngrn.model import RNGRN
    from rngrn.eval import simulate
    m = RNGRN(N=2, form=form, seed=0)
    res = simulate(m, L=100.0, n=32, integrator="etdrk4", T=5.0, dt=0.01)
    assert "patterned" in res and "kstar" in res


def test_data_gen_idempotent(tmp_path):
    from rngrn.data.rd_models import build_system
    from rngrn.data.spec import spec_from_config
    from rngrn.data import cache
    cfg = _tiny(load_config(os.path.join(CONFIGS, "milestone1_schnak.yaml")))
    system = build_system(cfg.data.system, L=cfg.data.L)
    spec = spec_from_config(cfg.data, system)
    root = str(tmp_path / "cache")
    d1 = cache.generate(root, spec, system)
    mtime1 = os.path.getmtime(os.path.join(d1, "manifest.json"))
    d2 = cache.generate(root, spec, system)   # cache hit — no regeneration
    assert d1 == d2
    assert os.path.getmtime(os.path.join(d2, "manifest.json")) == mtime1
