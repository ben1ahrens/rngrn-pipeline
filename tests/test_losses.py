"""test_losses.py — the promoted objective terms (unit 1).

Covers the three terms promoted from scripts/exp05_pixel_minibatch.py into the library:
split-support Turing hinges, the frame-scale anchor, and data-first staging. For each,
two things are asserted: (i) gradients reach theta, and (ii) the term's zero sits where
the design says it does.

The centrepiece is `test_old_hinges_collapse_onto_one_direction_split_do_not`, which
turns the MEASURED defect (docs/STATE_OF_THE_SCIENCE.md 2.2: at 60/60 random inits
argmax_k sigma(k) IS the k~0 grid point, so the shared-support hinges push one scalar
in two directions) into a failing-vs-passing distinction between the two functions.
"""
import numpy as np
import pytest
import torch

from rngrn.config import LossConfig
from rngrn.losses import terms as T
from rngrn.losses import total as LT
from rngrn.losses.weighting import DataFirstStaging, FixedWeighting, staging_factor
from rngrn.model import RNGRN

KGRID = torch.linspace(0.0, 3.0, 200, dtype=torch.float64)
SEEDS = list(range(8))


def _model_at_steady_state(seed, N=3):
    """A random-init model plus its own converged steady state. Skips inits whose Newton
    solve fails — those are a different failure mode and not what these tests are about."""
    m = RNGRN(N=N, seed=seed)
    xs, conv = T.steady_state(m)
    if not conv:
        pytest.skip(f"seed {seed}: steady state did not converge; not this test's subject")
    return m, xs


def _flat_grad(scalar, model, retain=True):
    g = torch.autograd.grad(scalar, list(model.parameters()), retain_graph=retain,
                            allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if gi is None else gi).reshape(-1)
                      for gi, p in zip(g, model.parameters())])


def _cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


class _StubDispersion:
    """Minimal model surface used by the hinges: jacobian() + dispersion(). Lets a test
    pin sigma(k) to an exact shape instead of hunting for a model that produces one."""

    def __init__(self, sig):
        self._sig = sig
        self.N = 2

    def jacobian(self, xstar, create_graph=True):
        return torch.zeros(2, 2, dtype=torch.float64)

    def dispersion(self, xstar, kgrid, J=None):
        return self._sig


# ---------------------------------------------------------------------------------
# the measured premise
# ---------------------------------------------------------------------------------
def test_argmax_dispersion_is_the_k0_grid_point_at_random_init():
    """The premise the split is built on (docs/STATE_OF_THE_SCIENCE.md 2.2). If this ever
    stops holding, the shared-support hinges stop being self-defeating and the split loses
    its justification — so it is asserted, not assumed."""
    at_zero = 0
    checked = 0
    for seed in SEEDS:
        m = RNGRN(N=3, seed=seed)
        xs, conv = T.steady_state(m)
        if not conv:
            continue
        checked += 1
        sig = m.dispersion(xs.detach(), KGRID)
        at_zero += int(torch.argmax(sig).item() == 0)
    assert checked > 0, "no seed produced a converged steady state"
    assert at_zero == checked, (
        f"argmax sigma(k) was the k~0 point on only {at_zero}/{checked} random inits; "
        "the split-hinge justification rests on this being all of them")


# ---------------------------------------------------------------------------------
# (a) split hinges
# ---------------------------------------------------------------------------------
def test_split_hinges_gradients_reach_theta():
    m, xs = _model_at_steady_state(seed=0)
    loss, parts = T.turing_hinges_split(m, T.steady_state_diff(m, xs), KGRID)
    g = _flat_grad(loss, m, retain=False)
    assert torch.isfinite(g).all()
    assert g.norm() > 0.0, "split hinges produced a zero gradient w.r.t. theta"
    assert set(parts) >= {"sig0", "sig_max", "sig_max_pos", "hinge_i_min"}


def test_split_hinges_are_near_zero_when_both_conditions_hold():
    """Design zero: sigma(0) < 0 (uniform-stable) AND max_{k>=k_min} sigma(k) > 0. The
    softplus is smooth, so "zero" means "negligible with margin", which is what a hinge
    promises."""
    sig = torch.full((200,), -1.0, dtype=torch.float64)
    sig[100] = 1.0                                    # interior instability, k >> k_min
    loss, parts = T.turing_hinges_split(_StubDispersion(sig), None, KGRID)
    assert float(loss) < 1e-3, f"hinges not satisfied where the design says zero: {float(loss)}"
    assert parts["sig_max_pos"] == pytest.approx(1.0)


def test_split_hinges_penalise_a_uniformly_stable_dispersion():
    """Control for the test above: no interior instability -> the instability hinge bites."""
    sig = torch.full((200,), -1.0, dtype=torch.float64)
    loss, _ = T.turing_hinges_split(_StubDispersion(sig), None, KGRID)
    assert float(loss) > 0.5, f"a fully stable dispersion should be penalised, got {float(loss)}"


def test_split_hinges_ignore_instability_below_k_min():
    """The whole point of the split: an unstable mode at k < k_min must NOT satisfy the
    instability hinge, because that mode is what the uniform hinge is pushing down."""
    sig = torch.full((200,), -1.0, dtype=torch.float64)
    sig[5] = 1.0                                      # index 5 < i_min = 20
    split_loss, _ = T.turing_hinges_split(_StubDispersion(sig), None, KGRID)
    shared_loss, _ = T.turing_hinges(_StubDispersion(sig), None, KGRID)
    assert float(split_loss) > 0.5, "split hinges credited a sub-k_min mode"
    assert float(shared_loss) < 1e-3, "shared-support hinges were expected to be fooled here"


def test_split_hinges_reject_a_k_min_frac_with_empty_support():
    with pytest.raises(ValueError, match="empty support"):
        T.turing_hinges_split(_StubDispersion(torch.zeros(200, dtype=torch.float64)),
                              None, KGRID, k_min_frac=1.0)


def test_old_hinges_collapse_onto_one_direction_split_do_not():
    """THE defect, made visible.

    At a random init argmax sigma is the k~0 point (asserted above), so in
    `turing_hinges` both `sig[0]` and `sig.max()` are the SAME scalar: the whole loss is
    a function of sigma(k~0) alone and its gradient is exactly parallel (or antiparallel)
    to grad sigma(k~0). One scalar, two opposing pushes, no escape route.

    `turing_hinges_split` draws its instability term from a disjoint part of the grid, so
    its gradient must carry a component that grad sigma(k~0) does not span.
    """
    checked = 0
    for seed in SEEDS:
        m = RNGRN(N=3, seed=seed)
        xs, conv = T.steady_state(m)
        if not conv:
            continue
        xd = xs.detach()
        sig = m.dispersion(xd, KGRID)
        if torch.argmax(sig).item() != 0:
            continue                                  # premise does not hold for this init
        checked += 1
        g_sig0 = _flat_grad(sig[0], m)

        old, _ = T.turing_hinges(m, xd, KGRID)
        new, _ = T.turing_hinges_split(m, xd, KGRID)
        c_old = abs(_cos(_flat_grad(old, m), g_sig0))
        c_new = abs(_cos(_flat_grad(new, m, retain=False), g_sig0))

        assert c_old > 0.999, (
            f"seed {seed}: shared-support hinge gradient was expected to be collinear with "
            f"grad sigma(k~0) (|cos| = {c_old:.6f})")
        assert c_new < 0.999, (
            f"seed {seed}: split hinge gradient is still collinear with grad sigma(k~0) "
            f"(|cos| = {c_new:.6f}) — the split bought nothing at this init")
    assert checked >= 4, f"only {checked} usable inits; the comparison is not meaningful"


# ---------------------------------------------------------------------------------
# (b) frame-scale anchor
# ---------------------------------------------------------------------------------
def test_frame_scale_anchor_is_exactly_zero_at_the_observed_scale():
    """Design zero: x*_i == obs_scale for every i."""
    obs_scale = 0.7
    xstar = torch.full((3,), obs_scale, dtype=torch.float64)
    loss, parts = T.frame_scale_anchor(xstar, obs_scale)
    assert float(loss) == pytest.approx(0.0, abs=1e-24)
    assert parts["obs_scale"] == pytest.approx(obs_scale)


def test_frame_scale_anchor_is_the_mean_square_log_ratio():
    xstar = torch.tensor([0.5, 2.0, 1.0], dtype=torch.float64)
    obs_scale = 1.0
    loss, _ = T.frame_scale_anchor(xstar, obs_scale)
    expected = np.mean([np.log(0.5) ** 2, np.log(2.0) ** 2, 0.0])
    assert float(loss) == pytest.approx(expected)


def test_frame_scale_anchor_gradients_reach_theta():
    m, xs = _model_at_steady_state(seed=1)
    xstar = T.steady_state_diff(m, xs)
    loss, _ = T.frame_scale_anchor(xstar, 0.5)
    g = _flat_grad(loss, m, retain=False)
    assert torch.isfinite(g).all()
    assert g.norm() > 0.0, "frame-scale anchor produced a zero gradient w.r.t. theta"


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_frame_scale_anchor_fails_loud_on_an_unusable_scale(bad):
    xstar = torch.ones(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="obs_scale"):
        T.frame_scale_anchor(xstar, bad)


# ---------------------------------------------------------------------------------
# (c) data-first staging
# ---------------------------------------------------------------------------------
def test_staging_factor_matches_the_exp05_schedule():
    """exp05 line 87: w = 0 for it < 0.25*steps, then min(1, (it-0.25*steps)/(0.25*steps))."""
    steps = 400
    for it in range(0, steps, 7):
        expected = 0.0 if it < 0.25 * steps else min(1.0, (it - 0.25 * steps) / (0.25 * steps))
        assert staging_factor(it, steps) == pytest.approx(expected)
    assert staging_factor(0, steps) == 0.0
    assert staging_factor(99, steps) == 0.0
    assert staging_factor(100, steps) == 0.0
    assert staging_factor(150, steps) == pytest.approx(0.5)
    assert staging_factor(200, steps) == pytest.approx(1.0)
    assert staging_factor(10_000, steps) == pytest.approx(1.0)


@pytest.mark.parametrize("kw", [dict(total_steps=0), dict(off_frac=-0.1),
                                dict(off_frac=1.5), dict(ramp_frac=0.0)])
def test_staging_factor_fails_loud_on_a_nonsense_schedule(kw):
    args = dict(step=0, total_steps=100, off_frac=0.25, ramp_frac=0.25)
    args.update(kw)
    with pytest.raises(ValueError):
        staging_factor(**args)


def test_staging_scales_only_the_named_terms_and_composes_with_the_inner_strategy():
    base = dict(kstar=1.0, turing=1.0, anchor=2.0)
    terms = {k: torch.tensor(1.0, dtype=torch.float64) for k in base}
    inner = FixedWeighting(base)
    staged = DataFirstStaging(inner, total_steps=400, keys=("turing",))

    total0, w0 = staged.combine(terms, step=0)
    assert w0["turing"] == 0.0
    assert w0["kstar"] == 1.0 and w0["anchor"] == 2.0
    assert float(total0) == pytest.approx(3.0)          # 1*1 + 0*1 + 2*1

    total_mid, w_mid = staged.combine(terms, step=150)
    assert w_mid["turing"] == pytest.approx(0.5)
    assert float(total_mid) == pytest.approx(3.5)

    total_end, w_end = staged.combine(terms, step=399)
    assert w_end["turing"] == pytest.approx(1.0)
    assert float(total_end) == pytest.approx(4.0)
    # unstaged: identical to the inner strategy at every step
    assert float(inner.combine(terms, step=0)[0]) == pytest.approx(4.0)


def test_staging_reports_the_weights_it_actually_applied():
    """weights_used must be post-staging, or the run log claims a weight that never ran."""
    terms = {"turing": torch.tensor(2.0, dtype=torch.float64)}
    staged = DataFirstStaging(FixedWeighting(dict(turing=1.0)), total_steps=100)
    total, w = staged.combine(terms, step=0)
    assert w["turing"] == 0.0
    assert float(total) == pytest.approx(0.0)


# ---------------------------------------------------------------------------------
# (d) assembly: what compute_terms now emits, and the resid default
# ---------------------------------------------------------------------------------
def _tiny_recovery_inputs(N=3, H=16):
    torch.manual_seed(0)
    xx = torch.linspace(0, 2 * np.pi, H, dtype=torch.float64)
    field = 1.0 + 0.2 * torch.sin(3 * xx).reshape(1, -1) * torch.cos(3 * xx).reshape(-1, 1)
    frame = field.unsqueeze(0).repeat(N, 1, 1)
    return frame, 10.0, list(range(N))


def test_compute_terms_emits_the_anchor_and_defaults_to_split_hinges():
    frame, L, obs_idx = _tiny_recovery_inputs()
    m, _ = _model_at_steady_state(seed=2)
    vals, parts = LT.compute_terms(m, frame, L, obs_idx, KGRID, 0.5)
    assert set(vals) == {"kstar", "turing", "resid", "anticollapse", "anchor"}
    assert "sig_max_pos" in parts, "compute_terms did not use the split hinges by default"
    assert parts["obs_scale"] == pytest.approx(float(frame.mean()))

    _, parts_old = LT.compute_terms(m, frame, L, obs_idx, KGRID, 0.5, split_hinges=False)
    assert "sig_max_pos" not in parts_old, "split_hinges=False did not select the old hinges"


def test_detach_xstar_is_a_live_knob():
    """The library differentiates through x*; exp05 does not. Whichever is right, the knob
    must actually change the gradient, or reporting which one a run used means nothing."""
    frame, L, obs_idx = _tiny_recovery_inputs()
    strategy = FixedWeighting(LossConfig().weights)
    grads = {}
    for detach in (False, True):
        m, _ = _model_at_steady_state(seed=3)
        loss, _ = LT.total_loss(m, frame, L, obs_idx, KGRID, 0.5, strategy,
                                detach_xstar=detach)
        grads[detach] = _flat_grad(loss, m, retain=False)
        assert grads[detach].norm() > 0.0
    assert not torch.allclose(grads[False], grads[True]), (
        "detach_xstar changed nothing; either it is not wired through or x* carries no "
        "gradient at this init")


def test_skipping_the_residual_omits_it_rather_than_faking_a_zero():
    """A skipped term must be ABSENT, not reported as 0.0 — otherwise a run log says the
    residual was satisfied when it was never evaluated."""
    frame, L, obs_idx = _tiny_recovery_inputs()
    m, _ = _model_at_steady_state(seed=4)
    vals, parts = LT.compute_terms(m, frame, L, obs_idx, KGRID, 0.5, compute_resid=False)
    assert "resid" not in vals
    assert "resid" not in parts
    assert parts["resid_skipped"] is True

    vals_on, parts_on = LT.compute_terms(m, frame, L, obs_idx, KGRID, 0.5)
    assert "resid" in vals_on and "resid_skipped" not in parts_on


def test_skipping_the_residual_leaves_the_total_unchanged_at_weight_zero():
    frame, L, obs_idx = _tiny_recovery_inputs()
    strategy = FixedWeighting(dict(kstar=1.0, turing=1.0, resid=0.0,
                                   anticollapse=0.5, anchor=2.0))
    totals = {}
    for skip in (False, True):
        m, _ = _model_at_steady_state(seed=5)
        loss, _ = LT.total_loss(m, frame, L, obs_idx, KGRID, 0.5, strategy,
                                compute_resid=not skip)
        totals[skip] = float(loss)
    assert totals[False] == pytest.approx(totals[True])


class _StubSpectralSolver:
    """Records how many times `.solve()` was called (the cost guard: it must be zero when
    spectral did not ignite) and returns a preset (u_star | None, reason)."""
    def __init__(self, result):
        self._result = result
        self.n_calls = 0

    def solve(self):
        self.n_calls += 1
        return self._result


_SPEC_KEYS = ("spec_shape", "spec_aniso", "spec_amp_mean", "spec_amp_fluct", "real_moments")


def test_spectral_enumeration_contract_ignited_vs_skipped():
    """unit U4 (M1 wiring). `spectral=None` (every other test in this file) reproduces the
    line-279 enumeration bit for bit -- untouched by this test. With a `SpectralContext`,
    the five spectral keys are PRESENT-IFF-COMPUTED: absence must never be mistaken for "the
    term was satisfied" (the same contract `test_skipping_the_residual_omits_it_rather_than_
    faking_a_zero` pins for `resid`). Ignition itself is forced via `ignition_margin`
    (+-1e9) rather than hunting for a genuinely Turing-unstable/stable model -- `is_ignited`
    is a pure function of `parts['sig_max_pos']` and the margin, so this is an equivalent,
    much cheaper control than picking fixture seeds."""
    from rngrn import observables as OBS
    from rngrn.losses.spectral import SpectralConfig, SpectralContext, build_frame_targets

    frame, L, obs_idx = _tiny_recovery_inputs()
    m, _ = _model_at_steady_state(seed=2)
    # the frame's OWN measured k* (firewall-legal, same call recover.py makes), not an
    # arbitrary constant -- build_frame_targets needs the B_train band to actually contain
    # this tiny fixture's one real spectral peak.
    kstar_obs = float(OBS.kstar_of(frame[0].numpy(), L=L))
    BASE = {"kstar", "turing", "resid", "anticollapse", "anchor"}

    # ---- ignited: margin forced below any possible sig_max_pos -----------------------
    ignite_cfg = SpectralConfig(ignition_margin=-1e9)
    targets = build_frame_targets(frame.numpy(), L, kstar_obs, ignite_cfg)
    solver = _StubSpectralSolver((frame, "ok"))
    ctx = SpectralContext(solver=solver, targets=targets, cfg=ignite_cfg)
    vals, parts = LT.compute_terms(m, frame, L, obs_idx, KGRID, kstar_obs, spectral=ctx)
    assert set(vals) == BASE | set(_SPEC_KEYS)
    assert solver.n_calls == 1
    assert "spectral_skipped" not in parts
    assert parts["spec_ignited"] == 1.0
    # compute_terms() alone does NOT write the real "L_<key>" values -- that loop lives in
    # total_loss (`for k, v in term_vals.items(): parts[f"L_{k}"] = ...`), so parts still
    # carries the NaN placeholders here; the real numbers are in `vals` (term_vals).
    for k in _SPEC_KEYS:
        assert torch.isfinite(vals[k])
        assert parts[f"L_{k}"] != parts[f"L_{k}"], (
            f"L_{k} is only overwritten by total_loss's term_vals loop, not by "
            "compute_terms alone")

    # ---- skipped: margin forced above any possible sig_max_pos -- solver NOT called --
    skip_cfg = SpectralConfig(ignition_margin=1e9)
    targets2 = build_frame_targets(frame.numpy(), L, kstar_obs, skip_cfg)
    solver2 = _StubSpectralSolver((frame, "ok"))
    ctx2 = SpectralContext(solver=solver2, targets=targets2, cfg=skip_cfg)
    vals2, parts2 = LT.compute_terms(m, frame, L, obs_idx, KGRID, kstar_obs, spectral=ctx2)
    assert set(vals2) == BASE
    assert solver2.n_calls == 0, "the forward solve is expensive; must not run when not ignited"
    assert parts2["spectral_skipped"] == "not_ignited"
    assert parts2["spec_ignited"] == 0.0
    for k in _SPEC_KEYS:
        assert parts2[f"L_{k}"] != parts2[f"L_{k}"], f"L_{k} must be NaN, not absent or 0.0"


def test_adaptive_strategies_never_qualify_for_the_residual_skip():
    """An adaptive strategy may move a weight off 0 later, so a base weight of 0 licenses
    nothing. This uses `ratio`: unit 13 made gradnorm/ntk raise at CONSTRUCTION rather than
    silently fall back to fixed weights, so they can no longer be instantiated here."""
    from rngrn.losses.weighting import (GradNormWeighting, NTKWeighting,
                                        RatioWeighting, ScheduledWeighting)
    w = dict(resid=0.0)
    assert FixedWeighting(w).static_weights is True
    assert ScheduledWeighting(w).static_weights is True
    assert RatioWeighting(w).static_weights is False
    # the wrapper inherits its inner strategy's answer, in both directions
    assert DataFirstStaging(FixedWeighting(w), total_steps=10).static_weights is True
    assert DataFirstStaging(RatioWeighting(w), total_steps=10).static_weights is False
    # and the unimplemented strategies fail loud rather than pretending to be adaptive
    for cls in (GradNormWeighting, NTKWeighting):
        with pytest.raises(NotImplementedError):
            cls(w)


def test_latent_fields_get_no_gradient_once_the_residual_is_off():
    """The defect that forces recover() to refuse hidden-channel runs at resid weight 0:
    the latent fields enter the objective through stationarity_residual and nothing else."""
    frame, L, _ = _tiny_recovery_inputs(N=3)
    frame = frame[:2]                                   # m=2, N=3
    latent = torch.nn.Parameter(frame.mean(0, keepdim=True).clone())
    m, _ = _model_at_steady_state(seed=6)
    strategy = FixedWeighting(dict(kstar=1.0, turing=1.0, resid=0.0,
                                   anticollapse=0.5, anchor=2.0))
    loss, _ = LT.total_loss(m, frame, L, [0, 1], KGRID, 0.5, strategy,
                            latent_fields=latent)
    loss.backward()
    assert latent.grad is not None
    assert float(latent.grad.abs().max()) == 0.0, (
        "latent fields now receive gradient from some other term — the recover() guard "
        "against hidden-channel runs at resid weight 0 can be relaxed")


def test_recover_refuses_hidden_channels_when_the_residual_cannot_contribute():
    from rngrn.data.gate import RecoveryInput
    from rngrn.recover import recover
    frame, L, _ = _tiny_recovery_inputs(N=3)
    ri = RecoveryInput(frame=frame[:2].numpy(), L=L, observed_idx=[0, 1], N=3)
    with pytest.raises(ValueError, match="zero gradient"):
        recover(ri, n_restarts=1, adam_steps=1, lbfgs_steps=0,
                weights=dict(kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5, anchor=2.0))


def test_stationarity_residual_weight_defaults_to_zero():
    """SETTLED OFF (exp06: 9 cells x 8 seeds, all collapsed). The term still exists."""
    w = LossConfig().weights
    assert w["resid"] == 0.0
    assert w["anchor"] == 2.0
    assert T.DEFAULT_WEIGHTS["resid"] == 0.0
    assert T.DEFAULT_WEIGHTS["anchor"] == 2.0
    assert callable(T.stationarity_residual)


def test_loss_config_staging_defaults_match_the_promoted_objective():
    cfg = LossConfig()
    assert cfg.split_hinges is True
    assert cfg.hinge_k_min_frac == pytest.approx(0.1)
    assert cfg.staging_keys == ["turing"]
    assert cfg.staging_off_frac == pytest.approx(0.25)
    assert cfg.staging_ramp_frac == pytest.approx(0.25)
    assert cfg.detach_xstar is False    # library behaviour preserved; see compute_terms


def test_composite_loss_includes_the_anchor_term():
    """Regression (merge damage, repaired 2026-08-12): `composite_loss` computed its
    loss/parts twice — the second block, added with param_prior, overwrote the first and
    silently dropped w['anchor']*L_s and the L_anchor part. The active path
    (losses/total.compute_terms) never had the defect; this pins the standalone reference
    form: the total must move by exactly w_anchor * L_anchor, and the part must be logged."""
    m, xs = _model_at_steady_state(0)
    g = torch.Generator().manual_seed(0)
    frame = torch.rand(3, 8, 8, generator=g, dtype=torch.float64) + 0.5
    loss0, parts0 = T.composite_loss(m, frame, L=10.0, observed_idx=(0, 1, 2),
                                     kgrid=KGRID, kstar_obs=1.0,
                                     weights=dict(anchor=0.0))
    loss2, parts2 = T.composite_loss(m, frame, L=10.0, observed_idx=(0, 1, 2),
                                     kgrid=KGRID, kstar_obs=1.0,
                                     weights=dict(anchor=2.0))
    assert "L_anchor" in parts2
    assert parts2["L_anchor"] == pytest.approx(parts0["L_anchor"])   # term itself unweighted
    assert parts2["L_anchor"] > 0.0    # a random init does not sit on the frame scale
    assert float(loss2) - float(loss0) == pytest.approx(2.0 * parts2["L_anchor"])
