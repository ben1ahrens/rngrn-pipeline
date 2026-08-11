"""test_science.py — cheap correctness anchors (seconds, CPU).

These check the science invariants the pipeline depends on, not recovery success
(that is the milestone work Claude Code runs on CUDA):
  * autodiff Jacobian matches finite differences
  * general-N dispersion agrees with the 2x2 closed form away from the crossing
  * gradients flow to theta
  * the mu->0 lift reproduces the QSS reaction
  * ETDRK4 stays finite where split-step IMEX collapses under stiff diffusion
  * the reference answer keys reproduce the documented k*

Most invariants are parametrized over both regulation forms (competitive, nc1) — unit 6
enablement: nc1 is implemented (model.py, eval/rollout.py, eval/dynamical.py) but was
never exercised by a test before this file was updated.
"""
import numpy as np
import pytest
import torch

from rngrn.model import RNGRN

FORMS = ["competitive", "nc1"]


@pytest.mark.parametrize("form", FORMS)
def test_positivity_and_param_count(form):
    m = RNGRN(N=2, form=form, seed=0)
    assert torch.all(m.KA >= 0) and torch.all(m.KR >= 0) and torch.all(m.D > 0)
    assert torch.allclose(m.KA + m.KR, m.s, atol=1e-6)
    n_params = sum(p.numel() for p in m.parameters())
    assert n_params == 3 * m.N**2 + 3 * m.N


@pytest.mark.parametrize("form", FORMS)
def test_autodiff_jacobian_matches_fd(form):
    m = RNGRN(N=2, form=form, seed=1)
    x = torch.tensor([1.0, 0.9])
    J = m.jacobian(x, create_graph=False).detach().numpy()
    eps = 1e-6; f0 = m.reaction(x).detach().numpy(); Jfd = np.zeros((2, 2))
    for j in range(2):
        xp = x.clone(); xp[j] += eps
        Jfd[:, j] = (m.reaction(xp).detach().numpy() - f0) / eps
    assert np.max(np.abs(J - Jfd)) < 1e-5


@pytest.mark.parametrize("form", FORMS)
def test_general_vs_2x2_dispersion(form):
    m = RNGRN(N=2, form=form, seed=2)
    x = torch.tensor([1.0, 0.9])
    J = m.jacobian(x)
    kg = torch.linspace(0.05, 5.0, 60)
    sig_gen = m.dispersion(x, kg, J=J).detach().numpy()
    sig_2x2 = m.dispersion_2x2(J, kg).detach().numpy()
    # agree away from the discriminant crossing (real-eig region)
    ok = np.abs(sig_gen - sig_2x2) < 0.05
    assert ok.mean() > 0.7


@pytest.mark.parametrize("form", FORMS)
def test_gradients_flow_to_theta(form):
    m = RNGRN(N=2, form=form, seed=3)
    x = torch.tensor([1.0, 0.9])
    sig = m.dispersion(x, torch.linspace(0.05, 5.0, 40))
    loss = sig.max()
    loss.backward()
    assert m.theta_D.grad is not None and torch.any(m.theta_D.grad != 0)


@pytest.mark.parametrize("form", FORMS)
def test_mu_zero_lift_reduces_to_qss(form):
    from rngrn.eval.dynamical import lift_check
    m = RNGRN(N=2, form=form, seed=4)
    diff = lift_check(m, [1.0, 0.9], mu=1e-4, n=6000)
    assert diff < 1e-2, f"lift did not reduce to QSS: {diff}"


@pytest.mark.parametrize("form", FORMS)
def test_rollout_reaction_np_matches_model_reaction(form):
    """eval/rollout.py builds a SEPARATE numpy reaction for the lifted-PDE integrator
    (model.reaction is torch-only, pointwise). A silent divergence between the two would
    corrupt every rollout without a training-time signal (unit 6: nc1's numpy branch,
    eval/rollout.py lines ~29-34, had never been checked against the torch branch)."""
    from rngrn.eval.rollout import _reaction_np_builder
    m = RNGRN(N=3, form=form, seed=7)
    x = torch.tensor([1.0, 0.8, 1.2])
    f_torch = m.reaction(x).detach().numpy()
    reaction_np = _reaction_np_builder(m)
    X = np.broadcast_to(x.numpy()[:, None, None], (3, 4, 4)).copy()
    f_np = reaction_np(X)[:, 0, 0]
    assert np.max(np.abs(f_np - f_torch)) < 1e-10


@pytest.mark.parametrize("N", [2, 3])
def test_nc1_steady_state_converges(N):
    """nc1's multiplicative repression veto prod_j (1 - theta_R_ij) can in principle drive
    production to zero, which could stall the damped-Newton steady-state solve. Measure
    convergence at default random init across seeds rather than assume it (unit 6: this
    path was never exercised — a grep for nc1 previously returned 0 hits outside
    model.py/config.py)."""
    from rngrn.losses.terms import steady_state
    n_seeds = 20
    failures = []
    for seed in range(n_seeds):
        m = RNGRN(N=N, form="nc1", seed=seed)
        _, converged = steady_state(m)
        if not converged:
            failures.append(seed)
    assert not failures, (
        f"nc1 steady_state failed to converge on N={N} seeds {failures} "
        f"({len(failures)}/{n_seeds})")


def test_etdrk4_finite_under_stiff_diffusion():
    """The pitfall-8.1 guard: on stiff diffusion, split-step IMEX collapses to flat while
    ETDRK4 stays finite and non-trivial. We only require ETDRK4 to stay finite here."""
    from rngrn.eval.numerics import integrate_etdrk4
    N, n, L = 2, 32, 1.0
    D = np.array([1.0, 200.0])                 # very stiff on a unit box
    rng = np.random.default_rng(0)
    X0 = np.ones((N, n, n)) + 0.1 * rng.standard_normal((N, n, n))
    def reaction(X):                            # mild linear reaction
        return np.stack([0.5 * X[1] - X[0], -0.3 * X[1]])
    X, blew = integrate_etdrk4(X0, D, reaction, n, L, dt=0.05, nsteps=50)
    assert not blew and np.all(np.isfinite(X))


def test_reference_answer_keys():
    """Answer keys reproduce documented k* (validation-only quantities)."""
    from rngrn.data.rd_models import GiererMeinhardt, Schnakenberg
    gm = GiererMeinhardt().answer_key()
    sc = Schnakenberg().answer_key()
    assert abs(gm["kstar"] - 0.5224) < 0.05, gm["kstar"]
    assert abs(sc["kstar"] - 4.2059) < 0.2, sc["kstar"]


# ======================================================================================
# turing_ok must use the STRICT uniform-stability criterion (D-EVID-11)
#
# It used to test tr(J) < 0. A negative trace does NOT imply stability: the sum of the
# eigenvalues can be negative while one of them is positive. Combined with a default
# k-grid starting at 1e-3 — where sigma(k) is still essentially sigma(0) — BOTH halves of
# the Turing test could be satisfied by a system that is simply uniformly unstable, and the
# reported k* was then the grid floor rather than a real mode.
# ======================================================================================
def test_turing_ok_rejects_a_uniformly_unstable_system():
    """The counterexample. tr(J) < 0 but max Re eig(J) > 0 — not Turing, not anything."""
    from rngrn.eval.analysis import turing_ok

    J = np.diag([0.5, -1.0, -1.0])          # trace -1.5, but an eigenvalue at +0.5
    D = np.array([1.0, 10.0, 20.0])
    assert np.trace(J) < 0                                   # the loose test passes...
    assert np.linalg.eigvals(J).real.max() > 0               # ...on an unstable system

    ok, info = turing_ok(J, D)
    assert ok is False, "a uniformly unstable system is not Turing-unstable"
    assert info["stable_uniform"] is False
    # the old loose verdict stays visible rather than being silently dropped
    assert info["turing_loose"] is True
    assert info["stable_uniform_loose"] is True


def test_turing_ok_accepts_the_reference_turing_systems():
    """The fix must not reject genuine Turing systems — both references still pass."""
    from rngrn.data.rd_models import GiererMeinhardt, Schnakenberg
    from rngrn.eval.analysis import turing_ok

    for cls in (GiererMeinhardt, Schnakenberg):
        ak = cls().answer_key()
        ok, info = turing_ok(np.asarray(ak["J"], float), np.asarray(ak["D"], float))
        assert ok is True, f"{cls.__name__} must remain Turing-unstable"
        assert info["stable_uniform"] is True
        assert info["kstar"] > 1e-3, "k* must be a real mode, not the grid floor"
        assert abs(info["kstar"] - ak["kstar"]) / ak["kstar"] < 0.15, (
            cls.__name__, info["kstar"], ak["kstar"])


def test_turing_ok_kstar_comes_from_k_greater_than_zero():
    """k=0 is the uniform mode; it can never BE the structured instability."""
    from rngrn.data.rd_models import GiererMeinhardt
    from rngrn.eval.analysis import turing_ok

    ak = GiererMeinhardt().answer_key()
    kgrid = np.concatenate([[0.0], np.linspace(1e-3, 10.0, 500)])
    ok, info = turing_ok(np.asarray(ak["J"], float), np.asarray(ak["D"], float), kgrid=kgrid)
    assert ok is True
    assert info["kstar"] > 0.0, "k*=0 would mean the uniform mode was selected"


def test_turing_ok_agrees_with_the_robustness_cloud_criterion():
    """One definition of 'Turing', not two.

    `_perturb_cloud` has always used the strict criterion (analysis.py:86) and excluded
    k=0 from the structured max (analysis.py:92). `turing_ok` did neither, so the run
    index's `recovered_turing` and its `turing_volume_*` columns were answering different
    questions. A zero-noise cloud must now agree with turing_ok exactly.
    """
    from rngrn.data.rd_models import GiererMeinhardt
    from rngrn.eval.analysis import turing_ok, _perturb_cloud

    for J, D, expected in [
        (np.asarray(GiererMeinhardt().answer_key()["J"], float),
         np.asarray(GiererMeinhardt().answer_key()["D"], float), True),
        (np.diag([0.5, -1.0, -1.0]), np.array([1.0, 10.0, 20.0]), False),
    ]:
        ok, _ = turing_ok(J, D)
        res = _perturb_cloud(J, D, 0.0, np.random.default_rng(0), 8)   # sigma=0 -> no noise
        assert ok is expected
        assert (res["frac_strict"] > 0.5) is expected, (ok, res["frac_strict"])


def test_low_basal_init_gains_the_positive_diagonal_but_NOT_turing_reachability():
    """RE-MEASURED 2026-08-04 after the turing_ok correction (docs/DECISIONS.md D-EVID-11).

    This test previously asserted `low_basal_frac > 0.5` on `turing_ok`, and passed only
    because `turing_ok` tested tr(J) < 0. Under the strict criterion the low-basal init is
    **0.0% Turing-unstable at init, not 82%** — every one of the 206/398 draws that used to
    pass was uniformly UNSTABLE with k* pinned to the grid floor. The 82% figure in
    `docs/STATE_OF_THE_SCIENCE.md` §10 could not be reproduced under any definition; the
    closest is the 51.8% loose artefact.

    What survives, and is the honest motivation for the init: a positive Jacobian diagonal
    is a PREREQUISITE for Turing instability (all 88/88 true systems have one), the default
    init never produces one, and low_basal does. That separation is real and is what this
    test now pins.

    Measured at N=3 over 400 seeds, denominator = converged inits:

        init        converged   any positive J diagonal   STRICT Turing
        default        400          0/400  = 0.000          0/400 = 0.000
        low_basal      398        114/398  = 0.286          0/398 = 0.000

    The strict-Turing assertions are a REGRESSION GUARD: if someone restores the loose
    criterion, low_basal jumps back to ~0.52 and this test fails loudly.
    """
    from rngrn.losses.terms import steady_state
    from rngrn.eval.analysis import turing_ok

    def measure(init, n_seeds=400):
        n_conv = n_posdiag = n_strict = 0
        for seed in range(n_seeds):
            m = RNGRN(N=3, seed=seed, init=init)
            xstar, converged = steady_state(m)
            if not converged:
                continue
            J = m.jacobian(xstar, create_graph=False).detach().numpy()
            if not np.all(np.isfinite(J)):
                continue
            n_conv += 1
            n_posdiag += bool(np.any(np.diag(J) > 0))
            n_strict += bool(turing_ok(J, m.D.detach().numpy())[0])
        return n_posdiag / n_conv, n_strict / n_conv

    default_posdiag, default_strict = measure("default")
    low_basal_posdiag, low_basal_strict = measure("low_basal")

    # THE REAL EFFECT: low_basal buys the positive diagonal; default never gets one.
    assert default_posdiag == 0.0, default_posdiag
    assert low_basal_posdiag > 0.20, low_basal_posdiag
    assert low_basal_posdiag - default_posdiag > 0.20, (default_posdiag, low_basal_posdiag)

    # WHAT IT DOES NOT BUY: neither init is Turing-unstable at init under the strict
    # criterion. Stated as plainly as the effect above (CLAUDE.md §8).
    assert default_strict == 0.0, default_strict
    assert low_basal_strict == 0.0, (
        f"low_basal strict-Turing fraction is {low_basal_strict}, expected 0.0. If this "
        "rose, either turing_ok regressed to the loose tr(J) < 0 criterion or the init "
        "genuinely changed — check which before quoting any number.")


# ======================================================================================
# unit B3 — steady-state multistart. See docs/STATE_OF_THE_SCIENCE.md section 12.
# ======================================================================================
# Raw theta of the EXACT model at which nc1 training first lost its steady state:
# two_gene_classical_val/sample_0000, form=nc1, recover seed 0, restart 0, Adam step 880
# (adam_steps=2000, lbfgs off, base.yaml weights). Pinned to full float64 precision so
# the trap is reproducible without re-running 880 optimiser steps.
_B3_FOLD_TRAP_THETA = dict(
    theta_s=[[-0.8579053127688827, -0.14159688303303303],
             [-0.8489986984171967, -2.5543531779230033]],
    theta_g=[[4.195863311957089, -2.400018593667104],
             [3.670857357483652, 0.7057125421643792]],
    theta_alpha=[[4.970613527968563, -2.8231293440533394],
                 [4.464141747772455, -0.7576471383612259]],
    theta_delta=[0.3922929532313248, 1.0686497089179792],
    theta_beta=[-2.989599001251903, -1.0612067383851242],
    theta_D=[-3.910108871704899, 1.2235323634059783],
)


def _b3_fold_trap_model():
    m = RNGRN(N=2, form="nc1", seed=0)
    with torch.no_grad():
        for name, value in _B3_FOLD_TRAP_THETA.items():
            getattr(m, name).copy_(torch.tensor(value, dtype=torch.float64))
    return m


_B3_SWEEP_CACHE = {}


def _b3_sweep():
    """Walk a deterministic ensemble ONCE (the failing solves are the expensive ones) and
    record what all three B3 tests below need. Cached because the file is meant to run in
    seconds.

    The ensemble reaches the parameter region training actually visits: the random init has
    alpha ~ O(1), but by the time nc1 loses its steady state alpha has grown to ~5, so
    theta_alpha is boosted by 5. In that region the SINGLE-SEED solve fails often enough
    for the bit-identity pin to be non-vacuous."""
    if _B3_SWEEP_CACHE:
        return _B3_SWEEP_CACHE
    from rngrn.losses.terms import steady_state, steady_state_bracket
    rows = []
    for form in FORMS:
        for N in (2, 3):
            for seed in range(12):
                m = RNGRN(N=N, form=form, seed=seed)
                with torch.no_grad():
                    g = torch.Generator().manual_seed(1000 + seed)
                    m.theta_alpha += 5.0 + torch.randn(N, N, generator=g) * 0.3
                    m.theta_s += torch.randn(N, N, generator=g) * 0.5
                    m.theta_beta -= 1.0
                x_legacy, conv_legacy = steady_state(m, multistart=False)
                x_multi, conv_multi = steady_state(m, multistart=True)
                lo, hi = steady_state_bracket(m)
                rows.append(dict(
                    form=form, N=N, seed=seed,
                    conv_legacy=bool(conv_legacy), conv_multi=bool(conv_multi),
                    max_abs_diff=float((x_legacy - x_multi).detach().abs().max()),
                    in_bracket=bool(torch.all(x_multi.detach() >= lo - 1e-9)
                                    and torch.all(x_multi.detach() <= hi + 1e-9)),
                    resid=float(torch.linalg.norm(m.reaction(x_multi))),
                ))
    _B3_SWEEP_CACHE["rows"] = rows
    return _B3_SWEEP_CACHE


def test_b3_multistart_is_bit_identical_wherever_the_legacy_solve_converged():
    """THE REGRESSION PIN. terms.steady_state's multistart is a pure rescue path: its first
    attempt is the pre-B3 algorithm verbatim and short-circuits on success, so every call
    that converged before must return a BIT-IDENTICAL x* (difference exactly 0.0, not
    'small'). This is what makes the fix safe to have on by default and what keeps every
    competitive number recorded before B3 comparable."""
    rows = _b3_sweep()["rows"]
    legacy = [r for r in rows if r["conv_legacy"]]
    bad = [r for r in legacy if not (r["conv_multi"] and r["max_abs_diff"] == 0.0)]
    assert not bad, f"multistart changed a converged x*: {bad[:4]}"
    assert len(legacy) > 0
    # non-vacuous: the ensemble must actually contain legacy failures for the pin to mean
    # anything, and multistart must actually rescue them.
    rescued = [r for r in rows if not r["conv_legacy"] and r["conv_multi"]]
    assert rescued, "ensemble contains no legacy failure — the pin is vacuous"


def test_b3_multistart_rescues_every_stressed_model():
    """Companion to the pin above: on the same ensemble the multistart solve converges
    everywhere, i.e. the legacy failures were a globalisation artefact of the single
    x0 = ones seed and not an absence of a steady state."""
    rows = _b3_sweep()["rows"]
    failures = [(r["form"], r["N"], r["seed"], r["resid"]) for r in rows if not r["conv_multi"]]
    assert not failures, f"multistart still failed on {len(failures)} models: {failures[:8]}"


def test_b3_nc1_newton_fold_trap_is_rescued():
    """The measured nc1 defect, pinned. At this exact theta the single-seed damped Newton
    is trapped on the fold det J = 0 at x = [0.7456, 0.7464]: |f| = 1.68e-2 while
    sigma_min(J) = 1.7e-6, so the Newton step explodes and the line search collapses to
    lam ~ 1e-9; the relaxation fallback re-starts from the same seed and stalls at
    |f| = 1.3e-2. The reaction nevertheless has one well-conditioned positive root."""
    from rngrn.losses.terms import steady_state
    m = _b3_fold_trap_model()

    trap = torch.tensor([0.745589, 0.746393], dtype=torch.float64)
    J_trap = m.jacobian(trap, create_graph=False).detach()
    sv = torch.linalg.svdvals(J_trap)
    assert float(torch.linalg.norm(m.reaction(trap))) < 2e-2      # a near-zero of f ...
    assert float(sv[-1]) < 1e-5                                    # ... on a near-fold of J

    _, conv_legacy = steady_state(m, multistart=False)
    assert not conv_legacy, "the pinned trap no longer reproduces the legacy failure"

    x, conv = steady_state(m, multistart=True)
    assert conv
    assert float(torch.linalg.norm(m.reaction(x))) < 1e-8
    assert np.allclose(x.detach().numpy(), [0.06111129, 0.2228224], atol=1e-6)
    J = m.jacobian(x, create_graph=False).detach().numpy()
    assert np.linalg.cond(J) < 10.0            # the recovered root is benign, not singular


@pytest.mark.parametrize("form", FORMS)
def test_b3_steady_state_bracket_contains_the_root(form):
    """terms.steady_state_bracket claims lo_i = beta_i/delta_i <= x*_i <=
    (beta_i + sum_j alpha_ij)/delta_i for BOTH forms, because 0 <= prod_i <= sum_j alpha_ij.
    The multistart seeds are those two bounds and their geometric mean, so the claim has to
    hold or the seeds are not a bracket."""
    rows = [r for r in _b3_sweep()["rows"] if r["form"] == form and r["conv_multi"]]
    assert len(rows) >= 20, len(rows)
    outside = [(r["N"], r["seed"]) for r in rows if not r["in_bracket"]]
    assert not outside, f"root outside the claimed bracket for {form}: {outside}"
