"""test_ignition_gating.py — unit U4 (M1 wiring): compute_terms/recover.py's spectral gate.

Covers the parts of the M1 spectral-term wiring test_losses.py and test_spectral_terms.py
do not: the batched refusal, the recover()-level misconfiguration raises, and the
history-stability trap (TrainingHistory._names freezes the scalar column set on the first
recorded step -- ignition is a training-time EVENT, so a run that ignites mid-training must
not change that set between steps).

CHEAPNESS RULE: no test here invokes the real `forward.PatternSolver.solve()` -- every
spectral solve is a stub. The real-solve path is covered by tests/test_forward_solve.py
(~50 s, not re-run here).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rngrn import observables as OBS
from rngrn.losses import total as LT
from rngrn.losses import spectral as S
from rngrn.losses.spectral import SpectralConfig, SpectralContext, build_frame_targets
from rngrn.losses.weighting import FixedWeighting
from rngrn.model import RNGRN, BatchedRNGRN

KGRID = torch.linspace(0.0, 3.0, 200, dtype=torch.float64)
_SPEC_KEYS = ("spec_shape", "spec_aniso", "spec_amp_mean", "spec_amp_fluct", "real_moments")
_BASE_KEYS = {"kstar", "turing", "resid", "anticollapse", "anchor"}


def _tiny_recovery_inputs(N=3, H=16):
    """Same tiny positive, non-constant field as test_losses.py's helper of the same name
    (kept local -- these two test files stay independent of each other's internals)."""
    torch.manual_seed(0)
    xx = torch.linspace(0, 2 * np.pi, H, dtype=torch.float64)
    field = 1.0 + 0.2 * torch.sin(3 * xx).reshape(1, -1) * torch.cos(3 * xx).reshape(-1, 1)
    frame = field.unsqueeze(0).repeat(N, 1, 1)
    return frame, 10.0, list(range(N))


def _model_at_steady_state(seed, N=3):
    from rngrn.losses import terms as T
    m = RNGRN(N=N, seed=seed)
    xs, conv = T.steady_state(m)
    if not conv:
        pytest.skip(f"seed {seed}: steady state did not converge; not this test's subject")
    return m, xs


def _kstar_obs(frame, L):
    """The frame's OWN measured k* (firewall-legal), not an arbitrary constant --
    build_frame_targets needs the B_train band to contain this fixture's one real peak."""
    return float(OBS.kstar_of(frame[0].numpy(), L=L))


def _different_synthetic_pattern(N=3, H=16):
    """A positive, non-constant field with DIFFERENT spatial structure than
    `_tiny_recovery_inputs` -- a stand-in "recovered pattern" whose spectral statistics
    actually differ from the observed-frame targets (unlike returning the frame itself,
    which trivially zeroes every term by construction)."""
    xx = torch.linspace(0, 2 * np.pi, H, dtype=torch.float64)
    field = 1.0 + 0.3 * torch.sin(5 * xx).reshape(1, -1) * torch.cos(2 * xx).reshape(-1, 1)
    return field.unsqueeze(0).repeat(N, 1, 1)


class _StubSpectralSolver:
    """Records call count (the cost guard) and returns a preset (u_star | None, reason)."""
    def __init__(self, result):
        self._result = result
        self.n_calls = 0

    def solve(self):
        self.n_calls += 1
        return self._result


class _StubBatchedSpectralSolver:
    """Batched twin of `_StubSpectralSolver`, per `losses.spectral.SpectralContext`'s
    duck-typed `solve_subset(members, xstar_batch) -> (u_stack | None, ok_members, reasons)`
    contract. Every member in `ok` resolves to `pattern`; every other member offered is
    reported failed under `fail_reasons[member]`."""
    def __init__(self, pattern, ok, fail_reasons):
        self._pattern = pattern
        self._ok = set(ok)
        self._fail_reasons = dict(fail_reasons)
        self.calls = []

    def solve_subset(self, members, xstar_batch):
        self.calls.append(list(members))
        ok_members = [m for m in members if m in self._ok]
        reasons = {m: self._fail_reasons[m] for m in members if m not in self._ok}
        u_stack = torch.stack([self._pattern] * len(ok_members)) if ok_members else None
        return u_stack, ok_members, reasons


# ---------------------------------------------------------------------------------
# (a) not ignited: keys absent, reason recorded, solver NOT called
# ---------------------------------------------------------------------------------
def test_not_ignited_omits_keys_and_never_calls_the_solver():
    """A margin forced above any reachable sig_max_pos stands in for 'non-Turing model' --
    is_ignited is a pure function of (parts['sig_max_pos'], margin), so this is an
    equivalent and much cheaper control than hunting for a genuinely stable fixture seed."""
    frame, L, obs_idx = _tiny_recovery_inputs()
    m, _ = _model_at_steady_state(seed=2)
    kstar_obs = _kstar_obs(frame, L)
    cfg = SpectralConfig(ignition_margin=1e9)
    targets = build_frame_targets(frame.numpy(), L, kstar_obs, cfg)
    solver = _StubSpectralSolver((frame, "ok"))
    ctx = SpectralContext(solver=solver, targets=targets, cfg=cfg)

    vals, parts = LT.compute_terms(m, frame, L, obs_idx, KGRID, kstar_obs, spectral=ctx)

    assert set(vals) == _BASE_KEYS
    assert solver.n_calls == 0, "the forward solve is expensive; must not run when not ignited"
    assert parts["spectral_skipped"] == "not_ignited"
    assert parts["spec_ignited"] == 0.0
    for k in _SPEC_KEYS:
        assert parts[f"L_{k}"] != parts[f"L_{k}"], f"L_{k} must be NaN, not absent"


# ---------------------------------------------------------------------------------
# (b) ignited + solved: five keys present, weighted total includes them
# ---------------------------------------------------------------------------------
def test_ignited_and_solved_terms_enter_the_weighted_total():
    frame, L, obs_idx = _tiny_recovery_inputs()
    m, _ = _model_at_steady_state(seed=2)
    kstar_obs = _kstar_obs(frame, L)
    cfg = SpectralConfig(ignition_margin=-1e9)   # forces ignition
    targets = build_frame_targets(frame.numpy(), L, kstar_obs, cfg)
    u_star = _different_synthetic_pattern()
    solver = _StubSpectralSolver((u_star, "ok"))
    ctx = SpectralContext(solver=solver, targets=targets, cfg=cfg)
    weights = dict(kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5, anchor=2.0,
                   spec_shape=1.0, spec_aniso=1.0, spec_amp_mean=1.0, spec_amp_fluct=1.0,
                   real_moments=1.0)
    strategy = FixedWeighting(weights)

    loss_with, parts_with = LT.total_loss(m, frame, L, obs_idx, KGRID, kstar_obs, strategy,
                                          spectral=ctx)
    loss_without, _ = LT.total_loss(m, frame, L, obs_idx, KGRID, kstar_obs, strategy,
                                    spectral=None)

    assert solver.n_calls == 1
    assert set(parts_with["weights_used"]) >= set(_SPEC_KEYS)
    for k in _SPEC_KEYS:
        assert np.isfinite(parts_with[f"L_{k}"])
    assert float(loss_with) != pytest.approx(float(loss_without)), (
        "the five spectral terms carry non-zero weight and a non-zero value here, so the "
        "weighted total must differ from the spectral=None total")


# ---------------------------------------------------------------------------------
# (c) ignited but the solve did not produce a pattern: skipped, reason recorded
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("reason", ["not_patterned", "solve_failed"])
def test_ignited_but_unsolved_is_skipped_with_the_solver_reason(reason):
    frame, L, obs_idx = _tiny_recovery_inputs()
    m, _ = _model_at_steady_state(seed=2)
    kstar_obs = _kstar_obs(frame, L)
    cfg = SpectralConfig(ignition_margin=-1e9)
    targets = build_frame_targets(frame.numpy(), L, kstar_obs, cfg)
    solver = _StubSpectralSolver((None, reason))
    ctx = SpectralContext(solver=solver, targets=targets, cfg=cfg)

    vals, parts = LT.compute_terms(m, frame, L, obs_idx, KGRID, kstar_obs, spectral=ctx)

    assert set(vals) == _BASE_KEYS
    assert solver.n_calls == 1
    assert parts["spectral_skipped"] == reason
    assert parts["spec_ignited"] == 1.0, "ignited (Turing-unstable beyond margin); the SOLVE failed"
    for k in _SPEC_KEYS:
        assert parts[f"L_{k}"] != parts[f"L_{k}"]


# ---------------------------------------------------------------------------------
# (d) batched + spectral: the combination is LEGAL (C1 repair, D-PERF-4) -- a batched
# solver is accepted and wired per member; a SERIAL solver is refused loudly, at the
# entry point, before any steady-state solve or Jacobian.
# ---------------------------------------------------------------------------------
def test_compute_terms_batched_accepts_a_batched_spectral_context():
    """The five spectral keys appear per member with the omitted-never-zeroed semantics
    `_apply_spectral_batched` documents (losses/total.py, unit b2/U4): a member the forward
    solve did not produce a pattern for gets an EXACT 0.0 loss contribution (so it cannot
    shift another member's gradient) and a NaN in the recorded `L_<key>` (so a reader can
    tell "not computed" from "zero loss") -- never the other way round."""
    frame, L, obs_idx = _tiny_recovery_inputs(N=3)
    kstar_obs = _kstar_obs(frame, L)
    cfg = SpectralConfig(ignition_margin=-1e9)   # forces ignition for every converged member
    targets = build_frame_targets(frame.numpy(), L, kstar_obs, cfg)
    bmodel = BatchedRNGRN.from_seeds(N=3, seeds=[200, 201])  # both converge (verified)
    pattern = _different_synthetic_pattern(N=3)
    # member 0 ignites and solves; member 1 ignites but the forward solve fails.
    solver = _StubBatchedSpectralSolver(pattern, ok=[0], fail_reasons={1: "solve_failed"})
    ctx = SpectralContext(solver=solver, targets=targets, cfg=cfg)
    weights = dict(kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5, anchor=2.0,
                   spec_shape=1.0, spec_aniso=1.0, spec_amp_mean=1.0, spec_amp_fluct=1.0,
                   real_moments=1.0)
    strategy = FixedWeighting(weights)

    term_vals, parts, conv = LT.compute_terms_batched(
        bmodel, frame, L, obs_idx, KGRID, kstar_obs, spectral=ctx)
    assert bool(conv[0]) and bool(conv[1]), "fixture seeds must converge for this test to hold"
    assert solver.calls == [[0, 1]], "both members ignited (margin=-1e9) -- both must be offered"
    assert list(parts["spec_ignited"]) == [1.0, 1.0]
    assert list(parts["spectral_skipped"]) == ["", "solve_failed"]
    assert list(parts["spec_computed"]) == [True, False]
    for k in _SPEC_KEYS:
        assert np.isfinite(float(term_vals[k][0])), f"{k}: solved member must be real"
        assert float(term_vals[k][1]) == 0.0, f"{k}: failed member's LOSS term must be exact 0"

    _, parts_lv, _ = LT.total_loss_batched(
        bmodel, frame, L, obs_idx, KGRID, kstar_obs, strategy, spectral=ctx)
    for k in _SPEC_KEYS:
        assert np.isfinite(parts_lv[f"L_{k}"][0]), f"L_{k}: solved member must be real"
        assert parts_lv[f"L_{k}"][1] != parts_lv[f"L_{k}"][1], (
            f"L_{k}: failed member's RECORD must be NaN, never a faked 0.0")


def test_compute_terms_batched_refuses_a_serial_solver():
    """A `forward.PatternSolver` (serial) exposes `.solve()`, not `.solve_subset()`. Handing
    one to the batched assembler is a newly plausible mistake now that both combinations are
    legal (REVIEW_gpu_optim_delta.md C1/§8) -- must be refused at the ENTRY POINT, before any
    steady-state solve or Jacobian, not surfaced as an AttributeError mid-step."""
    bmodel = BatchedRNGRN.from_seeds(N=3, seeds=[200, 201])
    frame, L, obs_idx = _tiny_recovery_inputs(N=3)
    kstar_obs = _kstar_obs(frame, L)
    cfg = SpectralConfig(ignition_margin=-1e9)
    targets = build_frame_targets(frame.numpy(), L, kstar_obs, cfg)
    serial_solver = _StubSpectralSolver((frame, "ok"))   # .solve(), no .solve_subset()
    ctx = SpectralContext(solver=serial_solver, targets=targets, cfg=cfg)

    with pytest.raises(ValueError, match="solve_subset"):
        LT.compute_terms_batched(bmodel, frame, L, obs_idx, KGRID, kstar_obs, spectral=ctx)

    assert serial_solver.n_calls == 0, "must refuse before any solve is attempted"


def test_compute_terms_batched_refuses_a_non_batched_model():
    """`model` must be a `model.BatchedRNGRN`: the deleted refusal used to make this check
    unreachable-but-implicit (it always fired before `model` was ever touched); now that
    `spectral` is legal, the check is explicit."""
    m = RNGRN(N=3, seed=0)
    with pytest.raises(ValueError, match="BatchedRNGRN"):
        LT.compute_terms_batched(m, None, None, None, None, None)


# ---------------------------------------------------------------------------------
# (d2) `_apply_spectral_batched` directly (Task 4, R3 review §5): the per-member
# ignite-or-omit contract at the unit it is actually implemented in, not only through
# `compute_terms_batched`'s end-to-end path above. Two things the (d) tests above do not
# reach: the EXACT-0-vs-NaN split in isolation (no real BatchedRNGRN needed), and the
# "abandoned" reason `active=` produces, which nothing under tests/ exercised before this.
# ---------------------------------------------------------------------------------
def test_apply_spectral_batched_exact_zero_for_a_member_the_solver_could_not_pattern():
    """A member the forward solve could not pattern gets an EXACT 0.0 entry in `term_vals`
    -- never omitted, never a NaN placeholder there (the NaN placeholder is a
    `total_loss_batched`-level concern, already pinned by
    `test_compute_terms_batched_accepts_a_batched_spectral_context` above) -- while its
    `spectral_skipped`/`spec_computed` record the failure. `term_vals` must exist as a
    concrete zero for EVERY member because it is what the batched loss sums over."""
    frame, L, obs_idx = _tiny_recovery_inputs(N=3)
    kstar_obs = _kstar_obs(frame, L)
    cfg = SpectralConfig(ignition_margin=-1e9)
    targets = build_frame_targets(frame.numpy(), L, kstar_obs, cfg)
    pattern = _different_synthetic_pattern(N=3)
    solver = _StubBatchedSpectralSolver(pattern, ok=[1], fail_reasons={0: "not_patterned"})
    ctx = SpectralContext(solver=solver, targets=targets, cfg=cfg)
    term_vals: dict = {}
    parts = dict(ss_converged=np.array([True, True]), sig_max_pos=np.array([5.0, 5.0]))
    xstar_pre = torch.zeros(2, 3, dtype=torch.float64)
    conv = torch.tensor([True, True])

    LT._apply_spectral_batched(term_vals, parts, ctx, xstar_pre, conv, active=None)

    assert solver.calls == [[0, 1]], "both members ignited (margin=-1e9) -- both offered"
    assert list(parts["spectral_skipped"]) == ["not_patterned", ""]
    assert list(parts["spec_computed"]) == [False, True]
    assert list(parts["spec_ignited"]) == [1.0, 1.0]
    ref_vals, _ = S.spectral_terms(pattern, targets, cfg)   # member 1's own true value
    for k in LT.SPECTRAL_TERM_KEYS:
        assert float(term_vals[k][0]) == 0.0, f"{k}: not-patterned member must be EXACT 0"
        assert float(term_vals[k][1]) == pytest.approx(float(ref_vals[k]), abs=1e-12), (
            f"{k}: patterned member's entry does not match its own computed value -- the "
            "index_copy scatter landed it in the wrong row or dropped it")


def test_apply_spectral_batched_abandoned_is_distinguished_from_not_ignited():
    """The `active` mask's "abandoned" reason must never be conflated with "not_ignited"
    (losses/total.py:222-225): an abandoned lane may well have been Turing-unstable when
    `recover` gave up on it, and the record should say WHY the solve was skipped, not imply
    a stability verdict nobody made. Two members, NEITHER of which reaches a solve, for two
    DIFFERENT reasons: member 0 is Turing-unstable but INACTIVE (`recover` abandoned it);
    member 1 is simply not Turing-unstable. If `active` were ignored (or ANDed in the wrong
    place), member 0 would read "not_ignited" exactly like member 1 and this test would
    fail on the `spectral_skipped` assertion below."""
    solver = _StubBatchedSpectralSolver(_different_synthetic_pattern(N=3), ok=[0, 1],
                                        fail_reasons={})
    cfg = SpectralConfig(ignition_margin=1e-3)
    ctx = SpectralContext(solver=solver, targets=None, cfg=cfg)
    term_vals: dict = {}
    parts = dict(ss_converged=np.array([True, True]),
                sig_max_pos=np.array([5.0, -5.0]))    # member 0 unstable, member 1 stable
    xstar_pre = torch.zeros(2, 3, dtype=torch.float64)
    conv = torch.tensor([True, True])
    active = torch.tensor([False, True])               # member 0 abandoned by the caller

    LT._apply_spectral_batched(term_vals, parts, ctx, xstar_pre, conv, active)

    assert list(parts["spectral_skipped"]) == ["abandoned", "not_ignited"]
    assert list(parts["spec_ignited"]) == [0.0, 0.0]
    assert list(parts["spec_computed"]) == [False, False]
    assert solver.calls == [], (
        "neither member reached a solve (one abandoned, one not ignited) -- the expensive "
        "forward solve must not run for either")
    for k in LT.SPECTRAL_TERM_KEYS:
        assert term_vals[k].tolist() == [0.0, 0.0], f"{k}: nothing solved, must be all-0"


# ---------------------------------------------------------------------------------
# (e) recover()-level misconfiguration raises
# ---------------------------------------------------------------------------------
def _tiny_recovery_input_np(N=3, H=16, L=10.0):
    from rngrn.data.gate import RecoveryInput
    xx = np.linspace(0, 2 * np.pi, H)
    field = 1.0 + 0.2 * np.sin(3 * xx).reshape(1, -1) * np.cos(3 * xx).reshape(-1, 1)
    frame = np.stack([field] * N, axis=0)
    return RecoveryInput(frame=frame, L=L, observed_idx=list(range(N)), N=N)


def _spectral_on_strategy():
    return FixedWeighting(dict(kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5,
                               anchor=2.0, spec_shape=1.0))


def test_recover_raises_on_split_hinges_off_with_a_spectral_weight():
    from rngrn import recover as R
    ri = _tiny_recovery_input_np()
    with pytest.raises(ValueError, match="split_hinges"):
        R.recover(ri, strategy=_spectral_on_strategy(), split_hinges=False,
                  adam_steps=0, n_restarts=1)


def test_recover_accepts_batched_with_a_spectral_weight():
    """PINS THE CURRENT CONTRACT (D-PERF-4, controller ruling 2026-08-19): batched=True
    combined with a non-zero spectral weight is LEGAL, not refused -- `recover.py:396-401`'s
    docstring states this in place ("the refusal is now DELETED because the solve gained a
    member axis"), and `_batched_restarts` wires a real `forward.BatchedPatternSolver` into
    the spectral context whenever a spectral weight is on. This test used to assert the
    OPPOSITE (`pytest.raises(ValueError, match="batched")`) without passing `lbfgs_steps`,
    which made it pass VACUOUSLY -- it tripped `recover()`'s unrelated LBFGS guard (whose
    message also happens to contain "batched"), never the thing its name claimed to test. Here
    `lbfgs_steps=0` is explicit so that guard cannot fire, and `adam_steps=0` keeps this a fast
    validation-layer check: the point is that NO ValueError fires, not that anything converges.

    NOT COVERED HERE -- Task 4 completed this, but NOT as a literal `recover()` call:
    `test_batched_forward_solve.py::
    test_total_loss_batched_matches_serial_with_a_spectral_weight_from_identical_warm_starts`
    explains why recover() itself is impractical to pin (its first ignited step is always a
    FRESH relax, and on CPU the serial and batched fresh relaxes are two independently-
    implemented integrators seeded through different RNGs -- test_batched_forward_solve.py's
    own module docstring: "their fresh relaxes differ by FFT backend and would confound
    it"). It instead drives `losses.total.total_loss` / `total_loss_batched` -- the exact
    assembler `recover()` calls per step -- from an IDENTICAL preset warm start on both
    sides, the same discipline that module already uses for its gradient checks, and closes
    the batched-vs-serial equivalence C1/C2 of docs/REVIEW_gpu_optim_delta.md wants. This
    test here only pins that
    the combination is ACCEPTED at the validation layer -- it asserts nothing about the
    recovered result's correctness.
    """
    from rngrn import recover as R
    ri = _tiny_recovery_input_np()
    result = R.recover(ri, strategy=_spectral_on_strategy(), batched=True,
                       lbfgs_steps=0, adam_steps=0, n_restarts=1)
    assert result is not None


def test_recover_raises_on_non_identity_observed_idx_with_a_spectral_weight():
    """Numerics review, 2026-08-12: the spectral terms compare u_star[c] to frame[c] with
    no observed_idx routing, so a permuted/partial observation would silently fit the
    WRONG species. recover() must refuse until routing exists."""
    from rngrn import recover as R
    from rngrn.data.gate import RecoveryInput
    ri3 = _tiny_recovery_input_np()
    # PERMUTED full observation (m == N, so the m<N hidden-channel guard stays out of the
    # way): channel 0 observes species 1, so identity fails on the fitted channel.
    ri = RecoveryInput(frame=ri3.frame, L=ri3.L, observed_idx=[1, 2, 0], N=3)
    with pytest.raises(ValueError, match="observed_idx"):
        R.recover(ri, strategy=_spectral_on_strategy(), adam_steps=0, n_restarts=1)


def test_recover_raises_on_adaptive_strategy_with_a_spectral_weight():
    """An adaptive strategy refreshes weights on a cadence; a term igniting between
    refreshes would contribute 0 then jump. Spectral runs require static weights (the
    param_prior rule)."""
    from rngrn import recover as R
    from rngrn.losses.weighting import RatioWeighting
    ri = _tiny_recovery_input_np()
    adaptive = RatioWeighting(dict(kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5,
                                   anchor=2.0, spec_shape=1.0))
    with pytest.raises(ValueError, match="static"):
        R.recover(ri, strategy=adaptive, adam_steps=0, n_restarts=1)


def test_recover_does_not_raise_with_every_spectral_weight_zero():
    """The default config's weights (unit U4: spec_* all 0.0, or absent) must never trip the
    new misconfiguration raises -- this is what 'nothing in the default config may trigger
    the forward solve' means in practice. lbfgs_steps=0/adam_steps=0/n_restarts=1 keeps this
    to one cheap steady-state Newton solve, same cost as any other recover() smoke test in
    this suite."""
    from rngrn import recover as R
    ri = _tiny_recovery_input_np()
    off_strategy = FixedWeighting(dict(kstar=1.0, turing=1.0, resid=0.0,
                                       anticollapse=0.5, anchor=2.0))
    R.recover(ri, strategy=off_strategy, split_hinges=False,
              adam_steps=0, lbfgs_steps=0, n_restarts=1)


# ---------------------------------------------------------------------------------
# history stability: ignition toggling mid-run must not change the recorded column set
# ---------------------------------------------------------------------------------
def test_history_survives_ignition_toggling_between_steps():
    """TrainingHistory._names (history.py:141-150) freezes the scalar column set on the
    FIRST recorded step and RAISES if a later step's key set differs. Ignition is a
    training-time EVENT (a run may cross the margin mid-training), so the five 'L_<key>'
    placeholders `losses.total._apply_spectral` writes on EVERY step -- real when computed,
    NaN when skipped -- are exactly what keeps this from raising. This test constructs the
    two parts dicts directly (synthetic), the way `_apply_spectral` would leave them; it
    does not exercise compute_terms itself (that is test_spectral_enumeration_contract_*
    in test_losses.py and the tests above)."""
    from rngrn.history import TrainingHistory

    hist = TrainingHistory(every=1, total_steps=2, n_members=1, N=3)
    common = dict(total=1.0, sig_max=0.1, sig_max_pos=0.2, kstar_model=0.3, rel_err=0.1,
                 ss_converged=True, weights_used={"kstar": 1.0},
                 L_kstar=0.1, L_turing=0.2, L_anticollapse=0.3, L_anchor=0.4)

    ignited_parts = dict(common, spec_ignited=1.0, L_spec_shape=0.5, L_spec_aniso=0.5,
                        L_spec_amp_mean=0.5, L_spec_amp_fluct=0.5, L_real_moments=0.5)
    nan = float("nan")
    skipped_parts = dict(common, spec_ignited=0.0, L_spec_shape=nan, L_spec_aniso=nan,
                        L_spec_amp_mean=nan, L_spec_amp_fluct=nan, L_real_moments=nan)

    m = RNGRN(N=3, seed=0)
    hist.record_serial(0, 0, ignited_parts, m)
    hist.record_serial(1, 0, skipped_parts, m)   # must NOT raise ValueError

    arrays = hist.to_arrays()
    names = list(arrays["hist_scalar_names"])
    assert "spec_ignited" in names
    for k in _SPEC_KEYS:
        assert f"L_{k}" in names
    # step 0 real, step 1 NaN, on the one recorded member's spec_ignited column
    idx = names.index("spec_ignited")
    assert arrays["hist_scalars"][0, 0, idx] == pytest.approx(1.0)
    assert arrays["hist_scalars"][1, 0, idx] == pytest.approx(0.0)
