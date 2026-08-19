"""test_determinism.py — recover() is bit-identical across two runs at the same seed.

Unit 10: determinism, seed plumbing, dispersion-backend recording.
"""
from __future__ import annotations
import numpy as np
import pytest

from rngrn import recover as R
from rngrn.data.gate import RecoveryInput
from rngrn.utils import set_deterministic


def _make_input(N=2, m=2, H=16, W=16, L=40.0, seed=0):
    rng = np.random.default_rng(seed)
    frame = rng.random((m, H, W)) * 0.5 + 0.5
    return RecoveryInput(frame=frame, L=L, observed_idx=tuple(range(m)), N=N)


def test_recover_bit_identical_same_seed():
    set_deterministic(True)
    ri = _make_input()
    kwargs = dict(form="competitive", n_restarts=1, adam_steps=5, adam_lr=0.05,
                  lbfgs_steps=0, seed=7, model_seed=3, verbose=False)
    r1 = R.recover(ri, **kwargs)
    r2 = R.recover(ri, **kwargs)
    J1 = r1.model.jacobian(
        __import__("torch").tensor(r1.xstar), create_graph=False).detach().numpy()
    J2 = r2.model.jacobian(
        __import__("torch").tensor(r2.xstar), create_graph=False).detach().numpy()
    assert np.array_equal(J1, J2), "recovered J must be bit-identical across runs at the same seed"
    assert np.array_equal(r1.params["KA"], r2.params["KA"])


def test_model_seed_controls_init_independent_of_train_seed():
    ri = _make_input()
    kwargs = dict(form="competitive", n_restarts=1, adam_steps=0, adam_lr=0.05,
                  lbfgs_steps=0, verbose=False)
    r1 = R.recover(ri, seed=1, model_seed=42, **kwargs)
    r2 = R.recover(ri, seed=2, model_seed=42, **kwargs)
    assert np.array_equal(r1.params["KA"], r2.params["KA"]), (
        "same model_seed with different train seed must yield identical init")


def test_restart_seeds_independent_across_run_seeds():
    """Unit B1: recover(seed=s) and recover(seed=s+1) must share NO restart init --
    the old `model_seed + r` scheme made adjacent run seeds a sliding window that
    shared n_restarts-1 of their n_restarts model inits (measured on
    three_gene_val/sample_0000: 8 seeds were ~4 independent draws). Check both that
    the derived integer seeds never collide across adjacent run seeds, and -- since
    recover() constructs each restart's RNGRN from exactly this seed -- that the
    actual init parameters (KA, adam_steps=0 so KA IS the init) never collide either."""
    n_restarts = 4
    for s in range(5, 8):
        seeds_s = [R._restart_seed(s, r) for r in range(n_restarts)]
        seeds_s1 = [R._restart_seed(s + 1, r) for r in range(n_restarts)]
        assert not (set(seeds_s) & set(seeds_s1)), (
            f"restart seeds for run seed={s} collide with run seed={s + 1}")

    from rngrn.model import RNGRN
    inits_5 = [RNGRN(N=2, form="competitive", seed=x).KA.detach().numpy() for x in
               (R._restart_seed(5, r) for r in range(n_restarts))]
    inits_6 = [RNGRN(N=2, form="competitive", seed=x).KA.detach().numpy() for x in
               (R._restart_seed(6, r) for r in range(n_restarts))]
    for i, a in enumerate(inits_5):
        for j, b in enumerate(inits_6):
            assert not np.array_equal(a, b), (
                f"restart {i} of seed=5 collided with restart {j} of seed=6 "
                "-- restart inits must be independent across adjacent run seeds")

    # a repeat of recover(seed=s) is bit-identical
    ri = _make_input()
    kwargs = dict(form="competitive", n_restarts=n_restarts, adam_steps=0, adam_lr=0.05,
                  lbfgs_steps=0, verbose=False)
    r1 = R.recover(ri, seed=5, model_seed=5, **kwargs)
    r2 = R.recover(ri, seed=5, model_seed=5, **kwargs)
    assert np.array_equal(r1.params["KA"], r2.params["KA"])


def test_dispersion_backend_cubic_rejected_for_wrong_N():
    ri = _make_input(N=2, m=2)
    try:
        R.recover(ri, form="competitive", n_restarts=1, adam_steps=0, lbfgs_steps=0,
                  seed=0, dispersion_backend="cubic")
        assert False, "expected ValueError for cubic backend with N != 3"
    except ValueError:
        pass


# --------------------------------------------------------------------------------------
# dispersion_backend='auto' resolution (R3 task 4, step 4). model.py:147-152 resolves
# 'auto' to 'cubic' when N==3 (exact, and measured 162x faster on CUDA per restart-step)
# and 'eig' otherwise (the general-N reference), AT CONSTRUCTION -- never lazily -- and
# `.dispersion_backend` is documented to always read the concrete value, never 'auto'.
# The only prior coverage (test_dispersion_backend_cubic_rejected_for_wrong_N above) is
# the explicit-cubic-at-wrong-N rejection; nothing pinned what 'auto' itself resolves to.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("N,expected", [(2, "eig"), (3, "cubic"), (4, "eig")])
def test_dispersion_backend_auto_resolves_by_N(N, expected):
    from rngrn.model import RNGRN
    m = RNGRN(N=N, form="competitive", seed=0, dispersion_backend="auto")
    assert m.dispersion_backend == expected, (
        f"N={N}: 'auto' resolved to {m.dispersion_backend!r}, expected {expected!r}")


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_dispersion_backend_auto_never_survives_construction(N):
    """`.dispersion_backend` must always read a CONCRETE value -- a caller that reads it
    back (BatchedRNGRN's member-compatibility check, model.py:422-427; recover.py's
    docstring) must never see 'auto' itself."""
    from rngrn.model import RNGRN
    m = RNGRN(N=N, seed=0, dispersion_backend="auto")
    assert m.dispersion_backend != "auto"
    assert m.dispersion_backend in ("eig", "cubic")


def test_batched_rngrn_from_seeds_resolves_auto_identically():
    """`BatchedRNGRN.from_seeds` constructs one `RNGRN` per seed with the SAME
    `dispersion_backend` argument (model.py:440-452) -- 'auto' must resolve to the same
    concrete backend a serial construction at the same N would get, for every member."""
    from rngrn.model import BatchedRNGRN, RNGRN
    for N, expected in [(2, "eig"), (3, "cubic"), (4, "eig")]:
        bm = BatchedRNGRN.from_seeds(N=N, seeds=[10, 11, 12], dispersion_backend="auto")
        assert bm.dispersion_backend == expected, (
            f"N={N}: BatchedRNGRN.from_seeds resolved 'auto' to "
            f"{bm.dispersion_backend!r}, expected {expected!r}")
        serial = RNGRN(N=N, seed=10, dispersion_backend="auto")
        assert bm.dispersion_backend == serial.dispersion_backend, (
            "batched and serial 'auto' resolution disagree at the same N")


# --------------------------------------------------------------------------------------
# train.seed must actually vary the recovery (regression: D-EVID-4)
# --------------------------------------------------------------------------------------
def test_base_config_does_not_pin_model_seed():
    """model.seed pinned in a base config makes train.seed a no-op for everything below it.

    configs/base.yaml used to set `model.seed: 0`. fit() passes it straight to recover() as
    model_seed, restart inits come from _restart_seed(model_seed, r) with per-restart
    generators, and nothing in the recovery path reads the global RNG that
    seed_everything(train.seed) touches -- so every run drew the SAME inits regardless of
    train.seed. A K-seed replicate was K identical recoveries and cross-seed
    topology_consistency, the project's primary metric, would have read 1.0 while measuring
    nothing. Asserted at the config level because that is where the damage was done.
    """
    import yaml
    with open("configs/base.yaml") as fh:
        raw = yaml.safe_load(fh)
    assert raw.get("model", {}).get("seed", "ABSENT") == "ABSENT", (
        "configs/base.yaml must NOT set model.seed -- it is an override, not a default. "
        "Pinning it makes train.seed a no-op for every config that composes this base.")
    from rngrn.config import ModelConfig
    assert ModelConfig().seed is None, "ModelConfig.seed must default to None (derive from train.seed)"


def test_train_seed_changes_the_recovery_and_repeats_are_exact(tmp_path):
    """Different train.seed -> different recovery. Same train.seed -> bit-identical.

    Both halves matter: the first is what makes a seed replicate meaningful, the second is
    what makes it reproducible. Kept cheap (few steps/restarts) -- this tests the SEEDING
    wiring, not convergence. Uses the registered three_gene_qvar payload rather than a
    reference system so nothing is generated here.
    """
    import os
    import pytest
    from rngrn.config import load_config, apply_overrides
    from rngrn.train import fit

    if not os.path.exists("data/datasets/three_gene_qvar/payload.h5"):
        pytest.skip("three_gene_qvar payload not provisioned (scripts/link_payloads.sh)")

    def run(seed, tag):
        cfg = load_config("configs/m3_registry.yaml")
        cfg = apply_overrides(cfg, [
            "data.dataset_id=three_gene_qvar", "data.sample_key=sample_0000",
            f"train.seed={seed}", "train.adam_steps=8", "train.n_restarts=2",
            "train.lbfgs_steps=0", "solver.morphology_rollout=false",
            "solver.save_plot_arrays=false"])
        return float(fit(cfg, runs_root=str(tmp_path / tag))["loss"])

    a0 = run(0, "a0")
    b1 = run(1, "b1")
    a0_again = run(0, "a0_again")

    assert a0 == a0_again, f"same train.seed must reproduce exactly, got {a0} vs {a0_again}"
    assert a0 != b1, (
        f"different train.seed must change the recovery, got seed0={a0} seed1={b1} -- "
        "train.seed is a no-op again (regression of docs/DECISIONS.md D-EVID-4)")
