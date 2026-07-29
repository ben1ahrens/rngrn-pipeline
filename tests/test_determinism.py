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


def test_dispersion_backend_cubic_rejected_for_wrong_N():
    ri = _make_input(N=2, m=2)
    try:
        R.recover(ri, form="competitive", n_restarts=1, adam_steps=0, lbfgs_steps=0,
                  seed=0, dispersion_backend="cubic")
        assert False, "expected ValueError for cubic backend with N != 3"
    except ValueError:
        pass
