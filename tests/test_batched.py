"""The BATCHED recovery path must be numerically equivalent to the SERIAL reference.

Unit b2. The serial path is the reference implementation and the default; the batched path
exists only to make a high step budget affordable on GPU. It is worth nothing if it is a
different computation, and this repository has already been bitten once by a closed form
that "looked plausible and was wrong by MAE ~2e+2" and was caught only by an exact
equivalence check against the reference (see tests/test_dispersion_cubic.py). These tests
hold that bar for the batch dimension:

  * reaction, Jacobian, steady state, dispersion and every loss term, batched vs B serial
    models built from the SAME seeds, to ~1e-10 or exactly;
  * one Adam step of B batched members vs B independent serial Adam steps, including the
    per-member gradient clipping that makes them independent;
  * the failure contract: a diverged member is FLAGGED, and the batch does not abort.

WHAT THESE TESTS DO NOT CLAIM. A long batched run is not bit-identical to its serial twin;
floating-point associativity differs and the optimiser amplifies it. The multi-step test
below MEASURES that drift rather than asserting it away.
"""
import numpy as np
import pytest
import torch

from rngrn.model import RNGRN, BatchedRNGRN
from rngrn.losses import terms as T
from rngrn.losses import total as LT
from rngrn.losses.weighting import FixedWeighting, DataFirstStaging
from rngrn import recover as R


N = 3
B = 6
SEED0 = 100
WEIGHTS = dict(kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5, anchor=2.0)


def _pair(form="competitive", backend="eig", init="default", b=B, seed0=SEED0):
    """B serial models and the batched model built from the SAME seeds.

    The seed list is built once and handed to BOTH sides, which is the point: from_seeds
    takes explicit seeds rather than a base-plus-offset rule (the offset rule lived here
    until unit B1 replaced `model_seed + r` with a stable hash of the pair, and a test that
    reimplemented the old rule would have silently stopped comparing like with like).
    """
    seeds = [seed0 + r for r in range(b)]
    serial = [RNGRN(N=N, form=form, seed=s, dispersion_backend=backend, init=init)
              for s in seeds]
    batched = BatchedRNGRN.from_seeds(N=N, seeds=seeds, form=form,
                                     dispersion_backend=backend, init=init)
    return serial, batched


def _frame(seed=0, H=24, W=24):
    """A deterministic synthetic striped frame: enough structure for kstar_of to find a k*."""
    g = torch.Generator().manual_seed(seed)
    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                          torch.arange(W, dtype=torch.float64), indexing="ij")
    f = 1.0 + 0.3 * torch.sin(2 * np.pi * 4 * x / W) * torch.cos(2 * np.pi * 3 * y / H)
    return (f + 0.01 * torch.rand(H, W, generator=g, dtype=torch.float64)).unsqueeze(0)


def _kgrid(kstar_obs, n=200):
    return torch.linspace(kstar_obs / 50.0 + 1e-3, max(2.0, 8.0 * kstar_obs), n,
                          dtype=torch.float64)


# --------------------------------------------------------------------------------------
# the shared reaction kernel
# --------------------------------------------------------------------------------------
def test_raw_theta_is_stacked_bit_exactly():
    """The batch is built from serial members, so the RAW parameters -- the only thing that
    is copied rather than recomputed -- must be bit-identical. Everything downstream is
    recomputed from these, and inherits float round-off (see the next test)."""
    serial, batched = _pair()
    for name in ("theta_s", "theta_g", "theta_alpha", "theta_delta", "theta_beta", "theta_D"):
        got = getattr(batched, name)
        for b, m in enumerate(serial):
            assert torch.equal(getattr(m, name).detach(), got[b].detach()), f"{name}/{b}"


# The measured floor on ALL batched-vs-serial agreement, and the reason none of these tests
# assert bit-equality: torch's ELEMENTWISE kernels are not shape-invariant in the last bit.
# softplus and sigmoid on a (3,3) tensor take the scalar path; on a (B,3,3) tensor they take
# the vectorised path, and the two differ by up to 1 ULP (measured 1.11e-16 absolute on
# both). Since KA/KR/alpha/delta/beta/D are all softplus/sigmoid/exp of theta, that 1 ULP is
# injected at the very first step and every later quantity inherits it. It is irreducible
# without giving up vectorisation, i.e. without giving up the entire point of the unit.
# 1e-13 is therefore ~1000x the round-off floor and still ~1e11 x tighter than the MAE ~2e+2
# error that the earlier bad cubic-dispersion implementations showed.
ULP_TOL = 1e-13


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_batched_reaction_matches_serial(form):
    """Unit b2 factored the reaction body into model._reaction_raw so serial and batched
    share ONE copy of the arithmetic; only the leading dim differs."""
    serial, batched = _pair(form=form)
    g = torch.Generator().manual_seed(7)
    x = torch.rand(B, N, generator=g, dtype=torch.float64) + 0.2
    got = batched.reaction(x)
    worst = max(float((m.reaction(x[b]) - got[b]).detach().abs().max())
                for b, m in enumerate(serial))
    assert worst < ULP_TOL, f"batched reaction differs from serial by {worst:.3e}"


def test_batched_constrained_params_match_serial():
    serial, batched = _pair()
    worst = 0.0
    for name in ("KA", "KR", "alpha", "beta", "delta", "D", "s", "gate"):
        got = getattr(batched, name)
        for b, m in enumerate(serial):
            ref = getattr(m, name).detach()
            worst = max(worst, float(((ref - got[b].detach()).abs()
                                      / (ref.abs() + 1e-30)).max()))
    assert worst < 1e-15, f"constrained parameters differ by relative {worst:.3e}"


def test_batched_reaction_rejects_wrong_shape():
    _, batched = _pair()
    with pytest.raises(ValueError, match="expects x of shape"):
        batched.reaction(torch.ones(B, 5, N, dtype=torch.float64))


# --------------------------------------------------------------------------------------
# Jacobian (autodiff both sides)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_batched_jacobian_matches_serial(form):
    serial, batched = _pair(form=form)
    g = torch.Generator().manual_seed(11)
    x = torch.rand(B, N, generator=g, dtype=torch.float64) + 0.2
    J = batched.jacobian(x, create_graph=False)
    assert J.shape == (B, N, N)
    worst = 0.0
    for b, m in enumerate(serial):
        ref = m.jacobian(x[b], create_graph=False)
        worst = max(worst, float((ref - J[b]).abs().max()))
    assert worst < 1e-12, f"batched Jacobian differs from serial autodiff by {worst:.3e}"


def test_batched_jacobian_is_differentiable_in_theta():
    """The standing model.py contract: J must stay differentiable w.r.t. theta, so the
    physics terms can push the parameters through it."""
    _, batched = _pair()
    x = torch.ones(B, N, dtype=torch.float64)
    J = batched.jacobian(x, create_graph=True)
    assert J.requires_grad
    J.sum().backward()
    for name in ("theta_s", "theta_g", "theta_alpha", "theta_delta"):
        gr = getattr(batched, name).grad
        assert gr is not None and torch.isfinite(gr).all(), name
        assert float(gr.abs().sum()) > 0.0, name
    # theta_beta shifts f by a constant and theta_D does not enter the reaction at all, so
    # neither has a path to J. Their gradients reach theta only via the dispersion terms.
    assert batched.theta_beta.grad is None and batched.theta_D.grad is None


def test_batched_jacobian_create_graph_false_is_detached():
    _, batched = _pair()
    J = batched.jacobian(torch.ones(B, N, dtype=torch.float64), create_graph=False)
    assert not J.requires_grad


# --------------------------------------------------------------------------------------
# dispersion
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["eig", "cubic"])
@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_batched_dispersion_matches_serial(backend, form):
    serial, batched = _pair(form=form, backend=backend)
    g = torch.Generator().manual_seed(13)
    x = torch.rand(B, N, generator=g, dtype=torch.float64) + 0.2
    kg = torch.linspace(0.01, 3.0, 120, dtype=torch.float64)
    got = batched.dispersion(x, kg)
    assert got.shape == (B, len(kg))
    worst = 0.0
    for b, m in enumerate(serial):
        ref = m.dispersion(x[b], kg)
        worst = max(worst, float((ref - got[b]).detach().abs().max()))
    assert worst < 1e-10, f"{backend} batched dispersion differs by {worst:.3e}"


def test_batched_cubic_rejects_non_three_species():
    m = RNGRN(N=2, form="competitive", seed=0)
    bm = BatchedRNGRN([m, RNGRN(N=2, form="competitive", seed=1)])
    bm.dispersion_backend = "cubic"      # RNGRN's constructor refuses N!=3 up front
    with pytest.raises(ValueError, match="exact for N=3 only"):
        bm.dispersion(torch.ones(2, 2, dtype=torch.float64),
                      torch.linspace(0.1, 1.0, 5, dtype=torch.float64))


# --------------------------------------------------------------------------------------
# steady state
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_batched_steady_state_matches_serial(form):
    """Every member's x* and its converged flag must match the serial damped Newton."""
    serial, batched = _pair(form=form, b=12, seed0=200)
    xs, conv = T.steady_state_batched(batched)
    assert xs.shape == (12, N) and conv.shape == (12,)
    worst, n_conv = 0.0, 0
    for b, m in enumerate(serial):
        ref, rconv = T.steady_state(m)
        assert bool(conv[b]) == bool(rconv), f"member {b} converged flag differs"
        rel = float(((ref - xs[b]).abs() / (ref.abs() + 1e-30)).max())
        worst = max(worst, rel)
        n_conv += int(rconv)
    assert n_conv > 0, "no member converged; the test would be vacuous"
    assert worst < 1e-10, f"batched x* differs from serial by relative {worst:.3e}"


def test_batched_steady_state_residual_is_actually_solved():
    """Not merely 'agrees with serial' — f(x*) must be at the Newton tolerance."""
    _, batched = _pair(b=12, seed0=200)
    xs, conv = T.steady_state_batched(batched)
    r = torch.linalg.norm(batched.reaction(xs), dim=-1)
    assert bool((r[conv] < 1e-10).all()), f"converged members' |f(x*)| = {r[conv].tolist()}"


def test_batched_steady_state_diff_matches_serial():
    serial, batched = _pair(b=12, seed0=200)
    xs, conv = T.steady_state_batched(batched)
    pol, ok = T.steady_state_diff_batched(batched, xs)
    assert bool(ok.all())
    worst = 0.0
    for b, m in enumerate(serial):
        if not bool(conv[b]):
            continue
        ref = T.steady_state_diff(m, xs[b])
        worst = max(worst, float((ref - pol[b]).abs().max()))
    assert worst < 1e-12, f"differentiable polish differs by {worst:.3e}"


def test_one_diverged_member_does_not_abort_the_batch():
    """The fail-loud contract becomes a PER-MEMBER FLAG: a member whose steady-state solve
    fails must be reported as such, with the SAME verdict the serial solver gives it, while
    the other members are still solved."""
    _, batched = _pair(b=4, seed0=200)
    with torch.no_grad():
        # delta -> ~0 with a large saturating production pushes the root out past where the
        # damped Newton can reach from x0 = ones in 100 iterations.
        batched.theta_delta[2] = -40.0
        batched.theta_alpha[2] = 40.0

    # multistart OFF is the legacy solver, and there member 2 genuinely fails. That is the
    # case this test exists for: the fail-loud contract becomes a PER-MEMBER FLAG, so the
    # other three must still be solved rather than taken down with it.
    xs, conv = T.steady_state_batched(batched, multistart=False)
    assert not bool(conv[2]), "the sabotaged member should be flagged as non-converged"
    assert bool(conv[[0, 1, 3]].all()), "healthy members must still converge"
    assert torch.isfinite(xs).all(), "a failed member must not return NaN/inf"
    _, serial_conv = T.steady_state(batched.member(2), multistart=False)
    assert bool(serial_conv) is False, "serial and batched must agree on the FAILURE too"

    # multistart ON, the default: unit B3's analytic bracket RESCUES this member. The point
    # is parity — the batched solver must reach the same verdict as the serial one, or the
    # batched path would abandon restarts the serial path recovers (and batched nc1, which
    # depends entirely on that rescue, would still be untrainable).
    xs_ms, conv_ms = T.steady_state_batched(batched)
    _, serial_ms = T.steady_state(batched.member(2))
    assert bool(conv_ms[2]) is bool(serial_ms), \
        "batched and serial must agree on the multistart verdict, member by member"
    assert bool(conv_ms.all()), "with multistart on, the bracket rescues the sabotaged member"
    assert torch.isfinite(xs_ms).all()


# --------------------------------------------------------------------------------------
# loss terms and the total
# --------------------------------------------------------------------------------------
def _serial_terms(m, frame, kgrid, kstar_obs):
    tv, parts = LT.compute_terms(m, frame, 1.0, [0, 1, 2], kgrid, kstar_obs,
                                 strict=True, compute_resid=False)
    return tv, parts


@pytest.mark.parametrize("backend", ["eig", "cubic"])
def test_batched_terms_match_serial_term_by_term(backend):
    serial, batched = _pair(backend=backend, b=12, seed0=200)
    frame = _frame()
    from rngrn import observables as obs
    kstar_obs = obs.kstar_of(frame[0].numpy(), L=1.0)
    kgrid = _kgrid(kstar_obs)
    tv_b, parts_b, conv = LT.compute_terms_batched(
        batched, frame, 1.0, [0, 1, 2], kgrid, kstar_obs)
    worst = {}
    n = 0
    for b, m in enumerate(serial):
        if not bool(conv[b]):
            continue
        tv_s, parts_s = _serial_terms(m, frame, kgrid, kstar_obs)
        assert set(tv_s) == set(tv_b), "batched and serial term sets differ"
        for k in tv_s:
            d = abs(float(tv_s[k]) - float(tv_b[k][b]))
            worst[k] = max(worst.get(k, 0.0), d)
        pm = LT.parts_member(parts_b, b)
        for k in ("sig0", "sig_max", "sig_max_pos", "jac_norm", "kstar_model"):
            worst[k] = max(worst.get(k, 0.0), abs(float(parts_s[k]) - float(pm[k])))
        n += 1
    assert n >= 6, f"only {n} members converged; test too weak"
    for k, d in worst.items():
        assert d < 1e-10, f"term/diagnostic {k!r} differs from serial by {d:.3e} (all: {worst})"


def test_batched_total_loss_matches_serial():
    serial, batched = _pair(b=12, seed0=200)
    frame = _frame()
    from rngrn import observables as obs
    kstar_obs = obs.kstar_of(frame[0].numpy(), L=1.0)
    kgrid = _kgrid(kstar_obs)
    strat_b = FixedWeighting(WEIGHTS)
    loss_v, parts_b, conv = LT.total_loss_batched(
        batched, frame, 1.0, [0, 1, 2], kgrid, kstar_obs, strat_b, step=0)
    worst, n = 0.0, 0
    for b, m in enumerate(serial):
        if not bool(conv[b]):
            continue
        ref, _ = LT.total_loss(m, frame, 1.0, [0, 1, 2], kgrid, kstar_obs,
                               FixedWeighting(WEIGHTS), step=0, compute_resid=False)
        worst = max(worst, abs(float(ref) - float(loss_v[b])))
        n += 1
    assert n >= 6
    assert worst < 1e-10, f"batched total loss differs by {worst:.3e}"


def test_active_mask_is_inert_without_a_spectral_context():
    """`active=` must be a no-op when `spectral is None` — values AND column set.

    `recover._batched_restarts` passes `active=alive` unconditionally, where Phase A wrote
    `active=alive if spec_cfg is not None else None` (review M9). The guard was dropped
    because the mask is read only inside `losses.total._apply_spectral_batched`, which runs
    only when a `SpectralContext` was supplied. That is an argument about the code; this is
    the measurement. If it were wrong the damage would be silent in exactly the way
    `history.TrainingHistory._names` punishes — a batched run's frozen scalar column set
    would depend on whether the caller happened to pass a liveness mask.
    """
    _, batched = _pair(b=4, seed0=200)
    frame = _frame()
    from rngrn import observables as obs
    kstar_obs = obs.kstar_of(frame[0].numpy(), L=1.0)
    kgrid = _kgrid(kstar_obs)
    alive = torch.tensor([True, False, True, False])

    def _run(active):
        loss, parts, conv = LT.total_loss_batched(
            batched, frame, 1.0, [0, 1, 2], kgrid, kstar_obs, FixedWeighting(WEIGHTS),
            step=0, active=active)
        return loss.detach().numpy(), parts, conv.numpy()

    loss_n, parts_n, conv_n = _run(None)
    loss_a, parts_a, conv_a = _run(alive)

    assert set(parts_a) == set(parts_n), (
        "passing active= changed the parts column set: "
        f"only-with-mask={set(parts_a) - set(parts_n)}, "
        f"only-without={set(parts_n) - set(parts_a)}")
    assert np.array_equal(loss_a, loss_n)
    assert np.array_equal(conv_a, conv_n)


def test_batched_total_loss_refuses_the_residual():
    _, batched = _pair(b=2, seed0=200)
    frame = _frame()
    with pytest.raises(ValueError, match="stationarity residual"):
        LT.compute_terms_batched(batched, frame, 1.0, [0, 1, 2],
                                 _kgrid(0.5), 0.5, compute_resid=True)


def test_batched_staging_strategy_composes():
    """DataFirstStaging must apply the SAME schedule batched (it only ever forms
    sum(weight * term), which is shape-agnostic)."""
    serial, batched = _pair(b=8, seed0=200)
    frame = _frame()
    from rngrn import observables as obs
    kstar_obs = obs.kstar_of(frame[0].numpy(), L=1.0)
    kgrid = _kgrid(kstar_obs)
    step = 60      # inside the ramp for total_steps=200, off_frac=ramp_frac=0.25
    loss_v, _, conv = LT.total_loss_batched(
        batched, frame, 1.0, [0, 1, 2], kgrid, kstar_obs,
        DataFirstStaging(FixedWeighting(WEIGHTS), total_steps=200), step=step)
    worst, n = 0.0, 0
    for b, m in enumerate(serial):
        if not bool(conv[b]):
            continue
        ref, _ = LT.total_loss(m, frame, 1.0, [0, 1, 2], kgrid, kstar_obs,
                               DataFirstStaging(FixedWeighting(WEIGHTS), total_steps=200),
                               step=step, compute_resid=False)
        worst = max(worst, abs(float(ref) - float(loss_v[b])))
        n += 1
    assert n >= 4
    assert worst < 1e-10, f"staged batched loss differs by {worst:.3e}"


# --------------------------------------------------------------------------------------
# optimiser equivalence: one batched step == B independent serial steps
# --------------------------------------------------------------------------------------
def test_per_member_grad_clip_matches_torch_per_model():
    """recover._clip_grad_norm_per_member must reproduce torch.nn.utils.clip_grad_norm_
    applied to each member alone. Clipping the JOINT norm would couple the members."""
    serial, batched = _pair(b=5, seed0=300)
    g = torch.Generator().manual_seed(21)
    for p in batched.parameters():
        p.grad = torch.rand(p.shape, generator=g, dtype=torch.float64) * 40.0
    for b, m in enumerate(serial):
        for name, p in m.named_parameters():
            p.grad = getattr(batched, name).grad[b].clone()
    R._clip_grad_norm_per_member(list(batched.parameters()), 10.0, 5)
    for b, m in enumerate(serial):
        torch.nn.utils.clip_grad_norm_(list(m.parameters()), 10.0)
        for name, p in m.named_parameters():
            d = float((p.grad - getattr(batched, name).grad[b]).abs().max())
            assert d < 1e-14, f"member {b} {name} clipped grad differs by {d:.3e}"


def _serial_adam_steps(m, frame, kgrid, kstar_obs, n_steps, lr=0.05, clip=10.0,
                       total_steps=None):
    opt = torch.optim.Adam(list(m.parameters()), lr=lr)
    strategy = FixedWeighting(WEIGHTS)
    if total_steps:
        strategy = DataFirstStaging(strategy, total_steps=total_steps)
    for step in range(n_steps):
        opt.zero_grad()
        loss, _ = LT.total_loss(m, frame, 1.0, [0, 1, 2], kgrid, kstar_obs, strategy,
                               step=step, compute_resid=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(m.parameters()), clip)
        opt.step()
    return m


def _batched_adam_steps(bm, frame, kgrid, kstar_obs, n_steps, lr=0.05, clip=10.0,
                        total_steps=None):
    params = list(bm.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    strategy = FixedWeighting(WEIGHTS)
    if total_steps:
        strategy = DataFirstStaging(strategy, total_steps=total_steps)
    for step in range(n_steps):
        opt.zero_grad()
        loss_v, _, conv = LT.total_loss_batched(bm, frame, 1.0, [0, 1, 2], kgrid,
                                                kstar_obs, strategy, step=step)
        assert bool(conv.all()), "test fixture assumed every member stays converged"
        loss_v.sum().backward()
        R._clip_grad_norm_per_member(params, clip, bm.B)
        opt.step()
    return bm


@pytest.mark.parametrize("n_steps", [1, 5])
def test_batched_adam_step_equals_serial_adam_step(n_steps):
    """B batched Adam steps must give what B independent serial Adam runs give. This is the
    claim that makes a batched result comparable to a serial one at all."""
    serial, batched = _pair(b=6, seed0=400)
    frame = _frame()
    from rngrn import observables as obs
    kstar_obs = obs.kstar_of(frame[0].numpy(), L=1.0)
    kgrid = _kgrid(kstar_obs)
    _batched_adam_steps(batched, frame, kgrid, kstar_obs, n_steps)
    worst = 0.0
    for b, m in enumerate(serial):
        _serial_adam_steps(m, frame, kgrid, kstar_obs, n_steps)
        for name, p in m.named_parameters():
            worst = max(worst, float((p.detach() - getattr(batched, name)[b].detach())
                                     .abs().max()))
    assert worst < 1e-9, f"{n_steps} batched Adam step(s) drift from serial by {worst:.3e}"


def test_member_extraction_round_trips():
    _, batched = _pair(b=4, seed0=500)
    for b in range(4):
        m = batched.member(b)
        for name in ("theta_s", "theta_g", "theta_alpha", "theta_delta", "theta_beta",
                     "theta_D"):
            assert torch.equal(getattr(m, name).detach(), getattr(batched, name)[b].detach())
        assert (m.N, m.form, m.n_hill) == (batched.N, batched.form, batched.n_hill)


def test_batched_constructor_rejects_mismatched_members():
    with pytest.raises(ValueError, match="must share"):
        BatchedRNGRN([RNGRN(N=3, form="competitive", seed=0),
                      RNGRN(N=3, form="nc1", seed=1)])


# --------------------------------------------------------------------------------------
# recover() end to end
# --------------------------------------------------------------------------------------
class _RI:
    """Minimal stand-in for data.gate.RecoveryInput — (frame, L, observed_idx, N) only."""
    def __init__(self, frame, L, N):
        self.frame = frame
        self.L = L
        self.N = N
        self.observed_idx = list(range(N))


def _ri():
    f = _frame()
    return _RI(np.repeat(f.numpy(), 3, axis=0), 40.0, 3)


def test_recover_batched_matches_recover_serial():
    """The whole recover() call, batched vs serial, on the same seeds and step budget."""
    kw = dict(form="competitive", n_restarts=4, adam_steps=5, lbfgs_steps=0, seed=7,
              staging_keys=(), weights=WEIGHTS)
    ser = R.recover(_ri(), **kw)
    bat = R.recover(_ri(), batched=True, **kw)
    assert abs(ser.loss - bat.loss) < 1e-9, f"serial {ser.loss!r} vs batched {bat.loss!r}"
    assert abs(ser.kstar_model - bat.kstar_model) < 1e-9
    for key in ("KA", "KR", "alpha", "delta", "beta", "D"):
        d = float(np.abs(ser.params[key] - bat.params[key]).max())
        assert d < 1e-9, f"recovered {key} differs by {d:.3e}"
    assert len(bat.restarts) == len(ser.restarts) == 4
    assert np.abs(ser.xstar - bat.xstar).max() < 1e-9


def test_recover_batched_refuses_lbfgs():
    with pytest.raises(ValueError, match="lbfgs_steps=0"):
        R.recover(_ri(), n_restarts=2, adam_steps=1, lbfgs_steps=5, batched=True,
                  staging_keys=(), weights=WEIGHTS)


def test_recover_batched_refuses_the_residual():
    w = dict(WEIGHTS); w["resid"] = 1.0
    with pytest.raises(ValueError, match="stationarity residual"):
        R.recover(_ri(), n_restarts=2, adam_steps=1, lbfgs_steps=0, batched=True,
                  staging_keys=(), weights=w)


def test_recover_batched_default_is_off():
    """The switch must default to the serial path so no pre-existing number changes method."""
    import inspect
    assert inspect.signature(R.recover).parameters["batched"].default is False
    from rngrn.config import TrainConfig
    assert TrainConfig().batched is False
    assert TrainConfig().device == "cpu"
