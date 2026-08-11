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


def test_model_init_is_threaded_from_config_into_recover(monkeypatch):
    """unit C1 regression: `model.init` was recorded on the run-index `model_init` column
    and in frozen_config.yaml but never passed to recover(), so `-o model.init=low_basal`
    silently ran the DEFAULT init while the record asserted otherwise. A config value that
    reaches the audit trail but not the code is worse than one that does neither."""
    import rngrn.train as T
    seen = {}

    def _spy(ri, **kw):
        seen.update(kw)
        raise RuntimeError("stop after capturing kwargs")

    monkeypatch.setattr(T.R, "recover", _spy)
    cfg = _tiny(load_config(os.path.join(CONFIGS, "milestone1_schnak.yaml")))
    cfg = apply_overrides(cfg, ["model.init=low_basal"])
    with pytest.raises(RuntimeError, match="stop after capturing"):
        T.fit(cfg, runs_root=tempfile.mkdtemp())
    assert seen.get("init") == "low_basal", (
        "train.fit() did not hand model.init to recover(); the run index would record an "
        f"init the run never used. saw init={seen.get('init')!r}")


# ======================================================================================
# CALL-SITE GUARDS for the other two fragile kwargs in train.fit (D-EVID-16).
#
# `init=` above was guarded; `arm_id=` and `D=result.D_phys` were NOT. All three are
# single-line additions inside a long call or dict — exactly what CLAUDE.md §11 says a
# line-granularity merge drops silently, as it previously did to `resid`, the anchor
# weight and `model_seed`, with the suite green. Unit tests over synthetic rows cannot
# see it; only a spy on the real call site can.
# ======================================================================================
def test_morphology_rollout_is_handed_the_PHYSICAL_D(monkeypatch):
    """D-EVID-14 regression guard, at the call site.

    `_morphology_rollout` must pass `result.D_phys`, not fall back to `model.D`: on the
    nondim path the latter is D/L**2 while the rollout integrates on a box of physical
    size L, so diffusion is starved by L**2 (3600x at L=60) and the model silently reports
    unpatterned. tests/test_rollout.py drives `_morphology_rollout` with a stub that has no
    `D_phys` attribute, so it passes whether or not the kwarg is threaded.
    """
    import rngrn.train as T
    seen = {}

    def _spy(model, **kw):
        seen.update(kw)
        raise RuntimeError("stop after capturing kwargs")

    monkeypatch.setattr(T, "simulate", _spy)
    cfg = _tiny(load_config(os.path.join(CONFIGS, "milestone1_schnak.yaml")))
    cfg = apply_overrides(cfg, ["solver.morphology_rollout=true"])
    with pytest.raises(RuntimeError, match="stop after capturing"):
        T.fit(cfg, runs_root=tempfile.mkdtemp())
    assert "D" in seen, (
        "train._morphology_rollout did not pass D= to simulate(); on the non-dimensional "
        "path the rollout would integrate D/L**2 at physical L and report the model as "
        "unpatterned, with no error (docs/DECISIONS.md D-EVID-14)")


def test_fit_writes_arm_id_onto_the_run_index_row():
    """D-EVID-13 regression guard, end to end.

    `arm_id` is the seed-independent identity `optim.benchmark` groups on. Drop the line
    that writes it and every new row silently becomes a LEGACY row, grouped by the
    seed-dependent `config_id` — re-instating the dead cross-seed aggregation (`n_seeds`
    pinned at 1, `kstar_identifiability_std` always NaN) with the suite green.
    tests/test_benchmark_grouping.py injects `arm_id` into synthetic dicts, so it cannot
    catch this.
    """
    import json
    import rngrn.train as T
    runs = tempfile.mkdtemp()
    cfg = _tiny(load_config(os.path.join(CONFIGS, "milestone1_schnak.yaml")))
    assert T.fit(cfg, runs_root=runs) is not None

    rows = [json.loads(l) for l in open(os.path.join(runs, "runs.jsonl")) if l.strip()]
    assert rows, "fit() wrote no run-index row"
    assert rows[0].get("arm_id"), (
        "the run-index row carries no arm_id, so optim.benchmark would treat this run as a "
        "legacy row and never group it with its seed replicates (D-EVID-13)")
    assert rows[0]["arm_id"] == cfg.arm_id()
    assert rows[0]["arm_id"] != rows[0]["config_id"], (
        "arm_id must be the SEED-INDEPENDENT identity, distinct from config_id")
