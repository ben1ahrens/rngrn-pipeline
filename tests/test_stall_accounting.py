"""test_stall_accounting.py — the two-path switch and its per-run counter (spec §4.3).

`docs/PLAN_redesign_R3.md` Task 13. Spec §4.3: "a per-run stall-rate counter"
on ignited-member solves that miss the **1e-9 Newton bar** on the commensurate box, and
"those members switch to the truncated-unrolled gradient path" (§4.2) instead of loosening
the convergence bar (F-D1-5 option (b), off the table — a pre-registered-class threshold).

Four contracts, in the order `docs/PLAN_redesign_R3.md` Task 13 states them:

1. The counter increments on a member that misses 1e-9 and not on one that meets it.
2. A stalled member's gradient comes from the unrolled path; a converged member's from the
   adjoint.
3. The "stall" event lands in `TrainingHistory` (R2 Task 9's event vocabulary — REUSED
   verbatim; no second event kind was added) with the right member and step.
4. The counter survives into the run index as a FLAT SCALAR — rows are built by
   `dict.update()`, never `**`-expansion (CLAUDE.md §4).

Plus two controller-mandated follow-ups (2026-08-19), against the REAL `train.py` code
paths rather than a hand-rolled stand-in:

5. `train._stall_columns(result, stall_switch)` — the exact helper `fit()`'s row-building
   calls — returns the three flat scalars from a `RecoveryResult` with a KNOWN stall count
   when `stall_switch=True`, and `{}` (the columns stay ABSENT from the row, not 0/NaN)
   when `False`.
6. `cfg.train.stall_switch` / `cfg.train.stall_switch_fraction` are threaded from `Config`
   into `recover()`'s call inside `train.fit()` — a call-site spy on `train.R.recover`, the
   same technique `test_smoke.py::test_model_init_is_threaded_from_config_into_recover`
   already uses for the `model.init` unit-C1 regression class this guards against too.

Contracts 1 and 3 are tested against `recover._account_for_stall` directly: it is pure
bookkeeping (no relax, no Newton), so a fake solver stub exercises it in milliseconds.

Contract 2 runs the real relax + real Newton on the tracked known-Turing fixture checkpoint
the rest of the M1/R3 suite already uses (`tests/test_forward_solve.py`,
`tests/test_unrolled_grad.py`) for the CONVERGED branch — proven to reach the 1e-9 bar on
this exact checkpoint/box by `test_forward_solve.py::base`, which solves it with
`PatternSolver.solve()` under the same defaults (default `newton_iter`).

The STALLED branch is forced by monkeypatching `forward.newton_polish` to report a
residual above the bar, rather than by starving the relax: MEASURED (during this task) that
this fixture's own relax already reaches ~1e-11 pre-Newton at the default
`max_chunks`/`chunk`, i.e. it does not stall naturally at any `n_iter`, and a shorter relax
just fails `relax_to_pattern_torch`'s OWN saturation detector (raises "no saturation")
before Newton is ever reached. Monkeypatching the one function whose return value the
switch's branch reads is the direct way to exercise "Newton reported > 1e-9" without either
side effect — and it never touches `CONVERGENCE_TOL` itself, which is read from
`forward.PatternSolver.CONVERGENCE_TOL` and never redefined here (F-D1-5 option (b) stays
off the table; `test_stall_switch_refuses_the_convergence_bar_as_a_parameter` below pins
that the switch has no `tol`-shaped parameter to loosen it through).
"""
import os
import pathlib
import tempfile

import numpy as np
import pytest
import torch

import rngrn.recover as R
import rngrn.train as T
from rngrn.config import apply_overrides, load_config
from rngrn.forward import PatternSolver
from rngrn.history import TrainingHistory
from rngrn.losses.terms import steady_state
from rngrn.model import RNGRN, THETA_NAMES

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")

torch.set_default_dtype(torch.float64)

REPO = pathlib.Path(__file__).resolve().parents[1]
CKPT = (REPO / "experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/"
        "checkpoints/model.pt")
N_GRID = 64          # same grid test_forward_solve.py's `base` fixture proves converges
PERIODS = 6.0
SEED = 0


def _load_ckpt() -> RNGRN:
    assert CKPT.exists(), (
        f"tracked fixture checkpoint missing: {CKPT} — it is committed to git; "
        f"a missing file means a broken checkout, not a skippable test")
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    m = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    m.load_state_dict(ck["state_dict"])
    return m


@pytest.fixture(scope="module")
def fixture():
    """The known-Turing checkpoint on a commensurate-style box: N=3, n=64, L chosen from
    the linear dispersion argmax exactly as `test_forward_solve.py::base` does (6 periods)
    — this is the SAME setup `PatternSolver.solve()` is already proven to converge on."""
    model = _load_ckpt()
    xs, ok = steady_state(model)
    assert ok, "steady state failed on the fixture checkpoint"
    J = model.jacobian(xs, create_graph=False).detach()
    kg = torch.linspace(1e-3, 10.0, 2000)
    sig = model.dispersion(xs, kg, J=J).detach()
    assert float(sig.max()) > 0, "fixture is not Turing-unstable"
    kstar_lin = float(kg[int(sig.argmax())])
    L = PERIODS * 2.0 * np.pi / kstar_lin
    return model, N_GRID, L


def _force_newton_stall(monkeypatch, residual=1.0):
    """Monkeypatch `forward.newton_polish` to report `residual` (>> 1e-9) regardless of
    input, so `_spectral_solve_with_stall_switch` takes its STALL branch deterministically.
    `_spectral_solve_with_stall_switch` imports `newton_polish` locally (`from .forward
    import ...`) INSIDE its own body, so patching `rngrn.forward.newton_polish` — the name
    the local import resolves at call time — is visible to it without needing to patch
    `recover`'s namespace at all."""
    import rngrn.forward as fwd
    real = fwd.newton_polish

    def fake(F_fn, u, modes_of, k2_full, D, gamma, n_iter=30, **kw):
        return u, residual   # never actually iterates; returns the field UNCHANGED

    monkeypatch.setattr(fwd, "newton_polish", fake)
    return real


# ----------------------------------------------------------------------------------------
# contracts 1 & 3: the counter and the history event (pure bookkeeping, no physics)
# ----------------------------------------------------------------------------------------
class _FakeSolver:
    """Duck-typed `_StallSwitchSolver`: only `.last_stalled` is read by `_account_for_stall`."""
    def __init__(self, stalled: bool):
        self.last_stalled = stalled


def test_stall_counter_increments_on_a_miss_and_not_on_a_meet():
    counts = [0, 0]
    R._account_for_stall(_FakeSolver(stalled=False), None, step=5, member=0, counts=counts)
    assert counts == [1, 0], "a MET bar must count as an ignited solve but NOT a stall"
    R._account_for_stall(_FakeSolver(stalled=True), None, step=6, member=0, counts=counts)
    assert counts == [2, 1], "a MISSED bar must count as both an ignited solve AND a stall"
    R._account_for_stall(_FakeSolver(stalled=False), None, step=7, member=1, counts=counts)
    assert counts == [3, 1], "a second converged member must not inflate the stall count"


def test_stall_event_lands_in_training_history_with_the_right_member_and_step():
    hist = TrainingHistory(every=1, total_steps=10, n_members=2, N=3)
    model = _load_ckpt()
    parts = dict(total=1.0, sig_max=0.2, sig_max_pos=0.2, kstar_model=0.4, rel_err=0.1,
                spec_ignited=1.0, ss_converged=True)
    hist.record_serial(0, 0, parts, model)   # to_arrays() needs >=1 scalar row to render

    counts = [0, 0]
    R._account_for_stall(_FakeSolver(stalled=True), hist, step=7, member=1, counts=counts)
    R._account_for_stall(_FakeSolver(stalled=False), hist, step=8, member=0, counts=counts)

    events = hist.to_arrays()["events"]
    got = {(int(e["step"]), int(e["member"]), str(e["kind"])) for e in events}
    assert got == {(7, 1, "stall")}, (
        "exactly one 'stall' event, at the stalled member's own (step, member) — the "
        "converged solve at step 8 must not appear, and 'stall' is REUSED, not a new kind")


# ----------------------------------------------------------------------------------------
# contract 2: the gradient path itself
# ----------------------------------------------------------------------------------------
def test_a_converged_member_gets_the_adjoint_path(fixture):
    model, n, L = fixture
    u, path, stalled, reason = R._spectral_solve_with_stall_switch(model, n, L, SEED)
    assert reason == "ok", f"expected a patterned solve, got reason={reason!r}"
    assert path == "adjoint" and not stalled, (
        "default newton_iter must reach the 1e-9 bar on this fixture — "
        "test_forward_solve.py::base proves PatternSolver.solve() does on the same setup")
    assert u.requires_grad
    grads = torch.autograd.grad(u.sum(), [getattr(model, nm) for nm in THETA_NAMES])
    for nm, g in zip(THETA_NAMES, grads):
        assert torch.isfinite(g).all(), f"non-finite adjoint gradient for {nm}"


def test_a_stalled_member_gets_the_unrolled_path_from_the_same_saturated_field(monkeypatch, fixture):
    model, n, L = fixture
    _force_newton_stall(monkeypatch)
    u, path, stalled, reason = R._spectral_solve_with_stall_switch(model, n, L, SEED)
    assert reason == "ok", f"expected a patterned solve, got reason={reason!r}"
    assert path == "unrolled" and stalled
    assert u.requires_grad
    grads = torch.autograd.grad(u.sum(), [getattr(model, nm) for nm in THETA_NAMES])
    for nm, g in zip(THETA_NAMES, grads):
        assert torch.isfinite(g).all(), f"non-finite unrolled gradient for {nm}"


def test_stall_switch_solver_wraps_the_same_decision_the_function_makes(monkeypatch, fixture):
    """`_StallSwitchSolver` is what `recover()`'s serial loop actually installs as
    `SpectralContext.solver`; this pins that its `.solve()` / `.last_path` / `.last_stalled`
    agree with `_spectral_solve_with_stall_switch` directly, on both branches."""
    model, n, L = fixture

    solver_ok = R._StallSwitchSolver(model, n, L, SEED)
    u, reason = solver_ok.solve()
    assert reason == "ok"
    assert solver_ok.last_path == "adjoint" and not solver_ok.last_stalled

    _force_newton_stall(monkeypatch)
    solver_stall = R._StallSwitchSolver(model, n, L, SEED)
    u2, reason2 = solver_stall.solve()
    assert reason2 == "ok"
    assert solver_stall.last_path == "unrolled" and solver_stall.last_stalled


def test_stall_switch_refuses_the_convergence_bar_as_a_parameter():
    """F-D1-5 option (b) — loosening the 1e-9 bar — stays off the table: the bar is not a
    parameter of the switch machinery at all, only the read-only class attribute it is
    compared against."""
    import inspect
    sig = inspect.signature(R._spectral_solve_with_stall_switch)
    assert "tol" not in sig.parameters and "convergence_tol" not in sig.parameters
    assert PatternSolver.CONVERGENCE_TOL == 1e-9


# ----------------------------------------------------------------------------------------
# contract 4: the counter is a flat scalar surviving a dict.update()-built row
# ----------------------------------------------------------------------------------------
def test_stall_counters_survive_a_dict_update_built_row_as_flat_scalars():
    """Mirrors `train.py`'s actual run-index convention (CLAUDE.md §4): rows are built by
    `dict.update()`, never `**`-expansion into a `dict()` literal, and run-identity keys are
    set last so they win any collision. `n_ignited_solves`/`n_stalled_solves` are new keys
    (no existing row column of either name), so no real collision is exercised here beyond
    demonstrating the update pattern itself."""
    result = R.RecoveryResult(model=None, params={}, topology={}, xstar=np.zeros(3),
                              kstar_model=0.1, kstar_obs=0.1, loss=0.0, parts={},
                              n_ignited_solves=7, n_stalled_solves=3)
    row = dict(run_id="fake-run")
    row.update(n_ignited_solves=result.n_ignited_solves,
              n_stalled_solves=result.n_stalled_solves)
    row.update(run_id="fake-run")   # run identity re-applied last, as train.py's rows do

    assert row["n_ignited_solves"] == 7 and row["n_stalled_solves"] == 3
    assert isinstance(row["n_ignited_solves"], int), "must be a flat scalar, not a tensor/np type"
    assert isinstance(row["n_stalled_solves"], int)
    assert not hasattr(row["n_ignited_solves"], "__len__"), "must not be collection-valued"


def test_stall_counters_default_to_zero_when_the_switch_is_off():
    """`stall_switch=False` (the default) must be bit-identical to every call before this
    task — including the counter reading 0/0 rather than something stale or None."""
    result = R.RecoveryResult(model=None, params={}, topology={}, xstar=np.zeros(3),
                              kstar_model=0.1, kstar_obs=0.1, loss=0.0, parts={})
    assert result.n_ignited_solves == 0
    assert result.n_stalled_solves == 0


# ----------------------------------------------------------------------------------------
# contract 5 (controller follow-up): the REAL train.py row-building helper, pinned
# ----------------------------------------------------------------------------------------
def test_stall_columns_are_flat_and_correct_from_a_result_with_a_known_stall():
    """`train._stall_columns` is the exact function `fit()`'s row-building calls
    (`row.update(_stall_columns(result, cfg.train.stall_switch))`). A "run with a known
    stall": 9 ignited solves, 4 of them stalls."""
    result = R.RecoveryResult(model=None, params={}, topology={}, xstar=np.zeros(3),
                              kstar_model=0.1, kstar_obs=0.1, loss=0.0, parts={},
                              n_ignited_solves=9, n_stalled_solves=4,
                              stall_switch_fraction=0.20)
    cols = T._stall_columns(result, stall_switch=True)
    assert cols == {"n_ignited_solves": 9, "n_stalled_solves": 4,
                    "stall_switch_fraction": 0.20}
    assert isinstance(cols["n_ignited_solves"], int) and isinstance(cols["n_stalled_solves"], int)
    assert isinstance(cols["stall_switch_fraction"], float)


def test_stall_columns_are_absent_not_zero_or_nan_when_the_switch_is_off():
    """Every run before this task, and every batched run (the switch is refused there) —
    the columns must not appear on the row at all, per the controller's "pick absent unless
    the row schema requires the key" ruling (`index.py`'s docstring: both the jsonl and the
    additive-sqlite backend tolerate a row missing a key)."""
    result = R.RecoveryResult(model=None, params={}, topology={}, xstar=np.zeros(3),
                              kstar_model=0.1, kstar_obs=0.1, loss=0.0, parts={},
                              n_ignited_solves=9, n_stalled_solves=4)   # non-zero on purpose:
    # even a result that DID stall must not leak into the row when the config said off.
    assert T._stall_columns(result, stall_switch=False) == {}


def test_stall_columns_survive_a_real_dict_update_row_build():
    """The helper's output actually merges into a row exactly as `fit()` uses it, run
    identity re-applied last as `train.py`'s rows do."""
    result = R.RecoveryResult(model=None, params={}, topology={}, xstar=np.zeros(3),
                              kstar_model=0.1, kstar_obs=0.1, loss=0.0, parts={},
                              n_ignited_solves=9, n_stalled_solves=4,
                              stall_switch_fraction=0.20)
    row = dict(run_id="fake-run")
    row.update(T._stall_columns(result, stall_switch=True))
    row.update(run_id="fake-run")
    assert row == {"run_id": "fake-run", "n_ignited_solves": 9, "n_stalled_solves": 4,
                   "stall_switch_fraction": 0.20}


# ----------------------------------------------------------------------------------------
# contract 6 (controller follow-up): config -> recover() threading, at the real call site
# ----------------------------------------------------------------------------------------
def _tiny_cfg(**overrides):
    cfg = load_config(os.path.join(CONFIGS, "milestone1_schnak.yaml"))
    ov = [
        "data.resolution=32", "data.T_max=5.0", "data.dt=0.05",
        "model.N=2", "model.m=2",
        "train.n_restarts=1", "train.adam_steps=8", "train.lbfgs_steps=0",
        "solver.n_grid=32", "solver.robustness_samples=5",
    ] + [f"{k}={v}" for k, v in overrides.items()]
    return apply_overrides(cfg, ov)


def test_stall_switch_and_fraction_are_threaded_from_config_into_recover(monkeypatch):
    """unit-C1-class regression guard (same technique as
    `test_smoke.py::test_model_init_is_threaded_from_config_into_recover`): a config value
    that reaches `frozen_config.yaml`/the run index but not `recover()` is worse than one
    that reaches neither -- `cfg.train.stall_switch` must actually change what the training
    call does, not just what the row later claims it did."""
    seen = {}

    def _spy(ri, **kw):
        seen.update(kw)
        raise RuntimeError("stop after capturing kwargs")

    monkeypatch.setattr(T.R, "recover", _spy)
    cfg = _tiny_cfg(**{"train.stall_switch": "true", "train.stall_switch_fraction": "0.15"})
    with pytest.raises(RuntimeError, match="stop after capturing"):
        T.fit(cfg, runs_root=tempfile.mkdtemp())
    assert seen.get("stall_switch") is True, (
        "train.fit() did not hand train.stall_switch to recover() -- the row would record "
        f"a switch state the run never used. saw stall_switch={seen.get('stall_switch')!r}")
    assert seen.get("stall_switch_fraction") == pytest.approx(0.15), (
        f"saw stall_switch_fraction={seen.get('stall_switch_fraction')!r}")


def test_stall_switch_defaults_to_false_when_not_overridden(monkeypatch):
    """The default config must thread `stall_switch=False` explicitly (not merely omit the
    kwarg and rely on recover()'s own default) -- so a reader of the spy trace can tell
    "off by config" from "kwarg forgotten" the same way unit C1 could not."""
    seen = {}

    def _spy(ri, **kw):
        seen.update(kw)
        raise RuntimeError("stop after capturing kwargs")

    monkeypatch.setattr(T.R, "recover", _spy)
    cfg = _tiny_cfg()
    with pytest.raises(RuntimeError, match="stop after capturing"):
        T.fit(cfg, runs_root=tempfile.mkdtemp())
    assert seen.get("stall_switch") is False
    assert seen.get("stall_switch_fraction") == pytest.approx(0.20)
