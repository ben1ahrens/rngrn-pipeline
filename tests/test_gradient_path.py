"""test_gradient_path.py — register item 8's PROMOTION: the unrolled path is now primary.

Owner ruling, 2026-08-19 (`docs/DECISIONS.md` D-R3-5). The pre-specified promotion condition
of `docs/REDESIGN_rngrn.md` §8 item 8 was met on both clauses by Task 14's FD A/B
(`experiments/redesign_r3/fd_ab/results/fd_ab.json`), so `train.gradient_path` defaults to
`"unrolled"` and the adjoint/IFT path is RETAINED as the A/B verification path.

`tests/test_stall_accounting.py` holds the ROUTING tests — which estimator one solve actually
earns — because they need that module's expensive relaxed fixture. This file holds everything
that needs no forward solve:

1. The A0 PIN. A0's behaviour must not change silently under a default flip
   (`docs/PLAN_redesign.md` Global Constraints: "A0 is untouchable"). It does not, and this
   MEASURES that rather than arguing it: A0's objective carries no spectral weight, so
   `recover()` builds no forward solver and never reads `gradient_path` at all.
2. The refusals — an unknown estimator, the vacuous `stall_switch` combination, and the
   batched combination the promoted path has no implementation for.
3. The counters staying LIVE on the promoted default (they are instrumentation now, not the
   router they were under Task 13) and reaching the run-index row.
4. `cfg.train.gradient_path` actually reaching `recover()` — the unit-C1 regression class.
"""
import os
import tempfile

import numpy as np
import pytest
import torch

import rngrn.recover as R
import rngrn.train as T
from rngrn.config import Config, apply_overrides, load_config
from rngrn.data.gate import RecoveryInput

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")

torch.set_default_dtype(torch.float64)

# The A0 baseline objective, quoted from `configs/base.yaml`'s `loss.weights` — the arm
# `docs/PLAN_redesign.md` calls untouchable. The five spectral weights at 0.0 are the whole
# of the A0 pin: at all-zero `recover()` computes use_spectral=False and builds no solver.
A0_WEIGHTS = dict(kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5, anchor=2.0,
                  morphology=0.1, param_prior=0.0, spec_shape=0.0, spec_aniso=0.0,
                  spec_amp_mean=0.0, spec_amp_fluct=0.0, real_moments=0.0)


def _make_input(N=2, m=2, H=16, W=16, L=40.0, seed=0):
    """`test_determinism.py::_make_input` verbatim — the cheapest real `recover()` input."""
    rng = np.random.default_rng(seed)
    frame = rng.random((m, H, W)) * 0.5 + 0.5
    return RecoveryInput(frame=frame, L=L, observed_idx=tuple(range(m)), N=N)


# ----------------------------------------------------------------------------------------
# 1. the A0 pin — MEASURED, not asserted
# ----------------------------------------------------------------------------------------
def test_the_a0_baseline_objective_carries_no_spectral_weight():
    """The structural reason A0 is immune, pinned at its source. If a future edit raises any
    of these five in `configs/base.yaml`, that is a DIFFERENT ARM on whichever estimator
    `train.gradient_path` selects — not a tweak to A0 — and this test says so first."""
    cfg = load_config(os.path.join(CONFIGS, "base.yaml"))
    from rngrn.losses.spectral import SPECTRAL_TERM_KEYS
    live = {k: cfg.loss.weights[k] for k in SPECTRAL_TERM_KEYS
            if float(cfg.loss.weights.get(k, 0.0)) != 0.0}
    assert live == {}, (
        f"configs/base.yaml now carries non-zero spectral weights {live} — the A0 baseline "
        f"arm would reach the forward solver and therefore the gradient-path default that "
        f"register item 8's promotion flipped (D-R3-5). A0 is untouchable "
        f"(docs/PLAN_redesign.md Global Constraints): pin this arm's gradient_path "
        f"explicitly and announce it, or keep the weights at 0.0.")


def test_a0_recovery_is_bit_identical_under_both_gradient_paths():
    """THE A0 PIN ITSELF. Two full `recover()` runs of the A0 objective, identical in every
    argument but the estimator, must agree BIT FOR BIT — because with no spectral weight the
    estimator is never reached. This is what licenses flipping the default at all."""
    ri = _make_input()
    kwargs = dict(form="competitive", weights=dict(A0_WEIGHTS), n_restarts=1, adam_steps=5,
                  adam_lr=0.05, lbfgs_steps=0, seed=7, model_seed=3, verbose=False)
    r_unrolled = R.recover(ri, gradient_path="unrolled", **kwargs)
    r_adjoint = R.recover(ri, gradient_path="adjoint", **kwargs)

    assert r_unrolled.loss == r_adjoint.loss, (
        "A0's loss moved when only the gradient estimator changed — the arm is no longer "
        "immune to the promotion's default flip (D-R3-5's A0 clause)")
    for key in ("KA", "KR", "alpha", "delta", "D"):
        assert np.array_equal(r_unrolled.params[key], r_adjoint.params[key]), (
            f"A0's recovered {key} is not bit-identical across the two gradient paths")
    assert np.array_equal(r_unrolled.xstar, r_adjoint.xstar)
    assert r_unrolled.kstar_model == r_adjoint.kstar_model
    # ... and no solve was attempted on either, which is WHY they agree.
    assert r_unrolled.n_ignited_solves == 0 and r_adjoint.n_ignited_solves == 0, (
        "an A0 run attempted a forward solve — it has no spectral weight, so it must not")


# ----------------------------------------------------------------------------------------
# 2. the refusals
# ----------------------------------------------------------------------------------------
def test_an_unknown_gradient_path_is_refused_loudly():
    with pytest.raises(ValueError, match="gradient_path must be one of"):
        R.recover(_make_input(), gradient_path="implicit", n_restarts=1, adam_steps=0,
                  lbfgs_steps=0)


def test_stall_switch_with_the_promoted_default_is_refused_rather_than_vacuous():
    """`stall_switch` is the fallback AWAY FROM an adjoint primary. With `"unrolled"` primary
    there is nothing to fall back to, and a caller who set it expected adjoint-primary — so
    silently honouring the default would hand them a run they did not ask for."""
    with pytest.raises(ValueError, match="meaningless with gradient_path='unrolled'"):
        R.recover(_make_input(), gradient_path="unrolled", stall_switch=True,
                  n_restarts=1, adam_steps=0, lbfgs_steps=0)


def test_the_promoted_default_refuses_a_batched_spectral_run():
    """`unrolled_relax` is SERIAL MODEL ONLY. Refuse rather than silently solving through the
    OTHER estimator, which would make the run non-comparable without saying so."""
    weights = dict(A0_WEIGHTS, spec_shape=1.0)
    with pytest.raises(ValueError, match="not supported with batched=True"):
        R.recover(_make_input(N=3, m=3), weights=weights, gradient_path="unrolled",
                  batched=True, n_restarts=1, adam_steps=0, lbfgs_steps=0)


def test_a_batched_run_with_no_spectral_weight_is_untouched_by_the_promotion():
    """The batched refusal is gated on `use_spectral` for A0's sake: every shipped config has
    zero spectral weights, so a batched A0 run must still be legal under the default."""
    r = R.recover(_make_input(N=2, m=2), weights=dict(A0_WEIGHTS), gradient_path="unrolled",
                  batched=True, n_restarts=2, adam_steps=2, adam_lr=0.05, lbfgs_steps=0,
                  seed=1, model_seed=1)
    assert r.gradient_path == "unrolled"
    assert r.n_ignited_solves == 0, "a batched run must not touch the serial stall counters"


# ----------------------------------------------------------------------------------------
# 3. the counters stay live on the promoted default
# ----------------------------------------------------------------------------------------
def test_uses_switch_solver_is_the_single_predicate_for_all_three_combinations():
    """`recover()` and `train._stall_columns` both ask this one function, so the run and the
    row it writes cannot disagree about whether the counters mean anything."""
    assert R.uses_switch_solver("unrolled", False) is True, (
        "the promoted default installs the switch-aware solver — the stall counters are "
        "instrumentation now (D-R3-5), not the router they were under Task 13")
    assert R.uses_switch_solver("adjoint", True) is True, "Task 13's adjoint-primary switch"
    assert R.uses_switch_solver("adjoint", False) is False, (
        "the pre-promotion combination goes to forward.PatternSolver, which has no stall "
        "accounting at all")
    with pytest.raises(ValueError, match="gradient_path must be one of"):
        R.uses_switch_solver("newton", False)


def test_the_run_index_row_carries_the_counters_and_the_estimator_on_the_default():
    """RE-TARGETED by the promotion: `_stall_columns` used to key off `stall_switch` alone,
    which since the flip would omit the counters from every DEFAULT run. It now keys off
    `uses_switch_solver`, and emits `gradient_path` alongside them — a gradient-derived
    number whose estimator is not recorded cannot be compared to one whose estimator is."""
    result = R.RecoveryResult(model=None, params={}, topology={}, xstar=np.zeros(3),
                              kstar_model=0.1, kstar_obs=0.1, loss=0.0, parts={},
                              n_ignited_solves=9, n_stalled_solves=4,
                              stall_switch_fraction=0.20, gradient_path="unrolled")
    live = R.uses_switch_solver("unrolled", False)
    row = dict(run_id="fake-run")
    row.update(T._stall_columns(result, live))
    row.update(run_id="fake-run")           # run identity re-applied last, as train.py does
    assert row == {"run_id": "fake-run", "n_ignited_solves": 9, "n_stalled_solves": 4,
                   "stall_switch_fraction": 0.20, "gradient_path": "unrolled"}
    assert isinstance(row["gradient_path"], str), "must be a flat scalar, not an enum/tensor"


def test_the_columns_stay_absent_on_the_pre_promotion_combination():
    result = R.RecoveryResult(model=None, params={}, topology={}, xstar=np.zeros(3),
                              kstar_model=0.1, kstar_obs=0.1, loss=0.0, parts={},
                              n_ignited_solves=9, n_stalled_solves=4,
                              gradient_path="adjoint")
    assert T._stall_columns(result, R.uses_switch_solver("adjoint", False)) == {}


# ----------------------------------------------------------------------------------------
# 4. config -> recover() threading (the unit-C1 regression class)
# ----------------------------------------------------------------------------------------
def test_the_config_default_is_the_promoted_unrolled_path():
    assert Config().train.gradient_path == "unrolled", (
        "register item 8's promotion made the unrolled path the DEFAULT primary estimator "
        "(owner ruling 2026-08-19, D-R3-5)")


def _tiny_cfg(**overrides):
    """`test_stall_accounting.py::_tiny_cfg` verbatim."""
    cfg = load_config(os.path.join(CONFIGS, "milestone1_schnak.yaml"))
    ov = [
        "data.resolution=32", "data.T_max=5.0", "data.dt=0.05",
        "model.N=2", "model.m=2",
        "train.n_restarts=1", "train.adam_steps=8", "train.lbfgs_steps=0",
        "solver.n_grid=32", "solver.robustness_samples=5",
    ] + [f"{k}={v}" for k, v in overrides.items()]
    return apply_overrides(cfg, ov)


@pytest.mark.parametrize("configured", ["unrolled", "adjoint"])
def test_gradient_path_is_threaded_from_config_into_recover(monkeypatch, configured):
    """A gradient estimator that reaches `frozen_config.yaml` but not `recover()` would make
    every row lie about which path produced its numbers."""
    seen = {}

    def _spy(ri, **kw):
        seen.update(kw)
        raise RuntimeError("stop after capturing kwargs")

    monkeypatch.setattr(T.R, "recover", _spy)
    cfg = _tiny_cfg(**{"train.gradient_path": configured})
    with pytest.raises(RuntimeError, match="stop after capturing"):
        T.fit(cfg, runs_root=tempfile.mkdtemp())
    assert seen.get("gradient_path") == configured, (
        f"train.fit() did not hand train.gradient_path to recover(); saw "
        f"{seen.get('gradient_path')!r}, configured {configured!r}")
