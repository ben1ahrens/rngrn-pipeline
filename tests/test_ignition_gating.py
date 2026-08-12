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
from rngrn.losses.spectral import SpectralConfig, SpectralContext, build_frame_targets
from rngrn.losses.weighting import FixedWeighting
from rngrn.model import RNGRN

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
# (d) batched refuses spectral loudly
# ---------------------------------------------------------------------------------
def test_compute_terms_batched_refuses_spectral():
    """Mirrors the compute_resid refusal immediately above it in losses/total.py -- the
    check fires before the function ever touches `model`, so a placeholder is enough."""
    with pytest.raises(ValueError, match="spectral"):
        LT.compute_terms_batched(None, None, None, None, None, None, spectral=object())


def test_total_loss_batched_refuses_spectral():
    with pytest.raises(ValueError, match="spectral"):
        LT.total_loss_batched(None, None, None, None, None, None, None, spectral=object())


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


def test_recover_raises_on_batched_with_a_spectral_weight():
    from rngrn import recover as R
    ri = _tiny_recovery_input_np()
    with pytest.raises(ValueError, match="batched"):
        R.recover(ri, strategy=_spectral_on_strategy(), batched=True,
                  adam_steps=0, n_restarts=1)


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
