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
    assert r.n_ignited_solves == 0, "a batched run must not touch the serial stall counters"
    assert r.gradient_path is None, (
        "no solver was built (no spectral weight, and the batched path never enters the "
        "serial loop), so the run must name no estimator — fix-round-1 ruling")
    assert T._stall_columns(r) == {}


# ----------------------------------------------------------------------------------------
# 3. the counters stay live on the promoted default
# ----------------------------------------------------------------------------------------
def test_uses_switch_solver_answers_the_ROUTING_question_not_the_row_gate():
    """`recover()`'s ROUTING predicate: which solver class the serial loop would install. It
    is deliberately NOT the run-index gate — see the next three tests. It is True for the
    promoted default whether or not any spectral weight is on, because it answers "which
    solver WOULD be built", not "was one built"."""
    assert R.uses_switch_solver("unrolled", False) is True, (
        "the promoted default installs the switch-aware solver — the stall counters are "
        "instrumentation now (D-R3-5), not the router they were under Task 13")
    assert R.uses_switch_solver("adjoint", True) is True, "Task 13's adjoint-primary switch"
    assert R.uses_switch_solver("adjoint", False) is False, (
        "the pre-promotion combination goes to forward.PatternSolver, which has no stall "
        "accounting at all")
    with pytest.raises(ValueError, match="gradient_path must be one of"):
        R.uses_switch_solver("newton", False)


def test_the_row_carries_the_counters_and_the_estimator_when_a_solve_was_possible():
    """`_stall_columns` emits iff `recover()` actually BUILT the switch-aware solver, which it
    reports by setting `RecoveryResult.gradient_path` to a string rather than None.
    `gradient_path` rides alongside the counters and is gated identically."""
    result = R.RecoveryResult(model=None, params={}, topology={}, xstar=np.zeros(3),
                              kstar_model=0.1, kstar_obs=0.1, loss=0.0, parts={},
                              n_ignited_solves=9, n_stalled_solves=4,
                              stall_switch_fraction=0.20, gradient_path="unrolled")
    row = dict(run_id="fake-run")
    row.update(T._stall_columns(result))
    row.update(run_id="fake-run")           # run identity re-applied last, as train.py does
    assert row == {"run_id": "fake-run", "n_ignited_solves": 9, "n_stalled_solves": 4,
                   "stall_switch_fraction": 0.20, "gradient_path": "unrolled"}
    assert isinstance(row["gradient_path"], str), "must be a flat scalar, not an enum/tensor"


def test_the_columns_stay_absent_when_no_spectral_solve_was_possible():
    """THE FIX-ROUND-1 RULING, and the reason the row is not gated on `uses_switch_solver`.
    A run with no spectral weight — every A0 run, every config in `configs/` today — builds no
    solver and attempts no solve, so it must carry NO estimator columns at all.
    `uses_switch_solver("unrolled", False)` is True for such a run: gating on it would write
    0/0 and an estimator name, making the row indistinguishable from "spectral on, nothing
    ignited" and naming an estimator for a computation that never happened."""
    assert R.uses_switch_solver("unrolled", False) is True, "the looser predicate would fire"
    result = R.RecoveryResult(model=None, params={}, topology={}, xstar=np.zeros(3),
                              kstar_model=0.1, kstar_obs=0.1, loss=0.0, parts={})
    assert result.gradient_path is None, (
        "a RecoveryResult from a call with no spectral solve must not name an estimator")
    assert T._stall_columns(result) == {}, (
        "ABSENT, not 0/0 — the earlier ruling stands (controller, fix round 1)")


def test_an_a0_run_writes_no_estimator_columns_end_to_end():
    """The same rule through the REAL `recover()` rather than a hand-built result: the A0
    objective on the promoted default must leave `gradient_path` unset, so its row carries
    none of the four columns."""
    r = R.recover(_make_input(), weights=dict(A0_WEIGHTS), gradient_path="unrolled",
                  n_restarts=1, adam_steps=2, adam_lr=0.05, lbfgs_steps=0, seed=7,
                  model_seed=3)
    assert r.gradient_path is None, (
        "an A0 run attempted no spectral solve — naming an estimator would state a fact "
        "about a computation that never ran")
    assert T._stall_columns(r) == {}


def test_the_columns_stay_absent_on_the_pre_promotion_combination():
    """`gradient_path="adjoint"` with `stall_switch=False` goes to `forward.PatternSolver`,
    which has no stall accounting at all — so the field stays None there too."""
    result = R.RecoveryResult(model=None, params={}, topology={}, xstar=np.zeros(3),
                              kstar_model=0.1, kstar_obs=0.1, loss=0.0, parts={},
                              n_ignited_solves=9, n_stalled_solves=4)
    assert T._stall_columns(result) == {}


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


# ----------------------------------------------------------------------------------------
# 5. the IFT bridges on a PINNED model, and the adjoint tripwire (Task 22 audit)
# ----------------------------------------------------------------------------------------
# Both classes need no forward solve: the payload is synthetic and `solve_adjoint` is
# monkeypatched, so only the bridge's own backward — the F closure, the registered-theta
# gradient assembly, and the tripwire — actually runs.

def _serial_payload(model, n=12, L=40.0):
    """The dict `recover._spectral_solve_with_stall_switch` hands `PatternSolve.apply`."""
    from rngrn.eval.numerics import _spectral_k2
    u_star = torch.rand(model.N, n, n, dtype=torch.float64) + 0.5
    return dict(model=model, u_star=u_star, n=n, L=L, k2_full=_spectral_k2(n, L),
                D_np=model.D.detach().cpu().numpy(), gamma=1.0,
                k2h=None, k2_dev=None, D_dev=None)


def test_pattern_solve_backward_works_on_a_pinned_model(monkeypatch):
    """Task 22 audit (latent collision): the serial IFT bridge iterated `THETA_NAMES`, but a
    `pin_xstar` model registers no `theta_beta` — the old backward died with a raw
    `AttributeError` mid-step (RED confirmed against the pre-fix semantics). The bridge now
    iterates the REGISTERED list; the derived-beta chain rule needs no special handling
    because `make_spatial_F` reads `model.beta`, a property of the registered thetas."""
    import rngrn.forward as F
    m = F.RNGRN(N=2, seed=0, pin_xstar=[1.0, 1.0])
    assert not hasattr(m, "theta_beta")            # the precondition that made this RED
    monkeypatch.setattr(F, "solve_adjoint",
                        lambda *a, **kw: (torch.ones(2 * 12 * 12, dtype=torch.float64),
                                          1e-12))
    payload = _serial_payload(m)
    out = F.PatternSolve.apply(payload, *F._registered_theta_params(m))
    out.sum().backward()
    grads = [getattr(m, nm).grad for nm in F.theta_names_for(pinned=True)]
    assert all(g is not None for g in grads), "a registered theta received no gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_batched_pattern_solve_backward_works_on_a_pinned_model(monkeypatch):
    """The batched bridge had the identical `THETA_NAMES` iteration; same fix, same
    reasoning, member axis added."""
    import rngrn.forward as F
    from rngrn.model import BatchedRNGRN
    from rngrn.eval.numerics import _spectral_k2
    n, L, B = 12, 40.0, 2
    bm = BatchedRNGRN.from_seeds(N=2, seeds=[0, 1], pin_xstar=[1.0, 1.0])
    assert not hasattr(bm, "theta_beta")
    monkeypatch.setattr(F, "solve_adjoint",
                        lambda *a, **kw: (torch.ones(2 * n * n, dtype=torch.float64),
                                          1e-12))
    KX, KY = F._half_k_grids(n, L)
    k2h = torch.from_numpy(KX ** 2 + KY ** 2)
    idx = torch.arange(B)
    u_stack = torch.rand(B, 2, n, n, dtype=torch.float64) + 0.5
    payload = dict(model=bm, idx=idx, members=[0, 1], u_star=u_stack, n=n, L=L,
                   k2_full=_spectral_k2(n, L),
                   D_np=bm.D.detach().cpu().numpy(), gamma=np.ones(B),
                   k2h=k2h, k2_dev=None, D_dev=bm.D.detach())
    out = F.BatchedPatternSolve.apply(payload, *F._registered_theta_params(bm))
    out.sum().backward()
    grads = [getattr(bm, nm).grad for nm in F.theta_names_for(pinned=True)]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("which", ["serial", "batched"])
def test_adjoint_tripwire_raises_instead_of_handing_over_a_biased_gradient(
        monkeypatch, which):
    """The load-bearing fail-loud guard (D-FFT-10): an adjoint residual above
    `_ADJOINT_RESIDUAL_TRIPWIRE` must RAISE in the bridge backward, never reach Adam.
    Previously correct by inspection only (Task 22 numerics review, Low)."""
    import rngrn.forward as F
    from rngrn.model import BatchedRNGRN
    from rngrn.eval.numerics import _spectral_k2
    n, L = 12, 40.0
    bad = F._ADJOINT_RESIDUAL_TRIPWIRE * 10.0
    monkeypatch.setattr(F, "solve_adjoint",
                        lambda *a, **kw: (torch.ones(2 * n * n, dtype=torch.float64), bad))
    if which == "serial":
        m = F.RNGRN(N=2, seed=0)
        out = F.PatternSolve.apply(_serial_payload(m), *F._registered_theta_params(m))
    else:
        bm = BatchedRNGRN.from_seeds(N=2, seeds=[0, 1])
        KX, KY = F._half_k_grids(n, L)
        payload = dict(model=bm, idx=torch.arange(2), members=[0, 1],
                       u_star=torch.rand(2, 2, n, n, dtype=torch.float64) + 0.5,
                       n=n, L=L, k2_full=_spectral_k2(n, L),
                       D_np=bm.D.detach().cpu().numpy(), gamma=np.ones(2),
                       k2h=torch.from_numpy(KX ** 2 + KY ** 2), k2_dev=None,
                       D_dev=bm.D.detach())
        out = F.BatchedPatternSolve.apply(payload, *F._registered_theta_params(bm))
    with pytest.raises(RuntimeError, match="tripwire"):
        out.sum().backward()
