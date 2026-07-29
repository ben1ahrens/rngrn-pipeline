"""test_determinism.py — recover() is bit-identical across two runs at the same seed.

Unit 10: determinism, seed plumbing, dispersion-backend recording.
"""
from __future__ import annotations
import numpy as np

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
