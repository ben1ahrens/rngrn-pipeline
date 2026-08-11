"""test_lifted.py — the QSS lift as a state-space object (eval/lifted.py).

Everything downstream in Stage 0b — the finite-mu robustness volumes and the 64x64 lifted
fields — rests on four claims, and these are them:

  1. the lifted fixed point IS the QSS fixed point, at EVERY mu (mu moves stability, never
     the steady state);
  2. sigma_full(k, mu) -> sigma_QSS(k) as mu -> 0, at the O(mu) rate a singular
     perturbation should give;
  3. J_full(mu) is J_full(1) with the gate rows scaled by 1/mu — EXACTLY, which is what
     makes a mu sweep cost one autodiff instead of one per point;
  4. the gate substep is EXACT at frozen x, so a finite-mu run cannot report a dead pattern
     because the integrator damped it.

Cheap: N=2 or N=3, seconds on CPU. Both regulation forms throughout.
"""
import numpy as np
import pytest
import torch

from rngrn.eval import lifted, rollout
from rngrn.losses.terms import steady_state
from rngrn.model import RNGRN

FORMS = ["competitive", "nc1"]
MUS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def _model_and_xstar(form, N=3, seed=3):
    m = RNGRN(N=N, form=form, n_hill=2, seed=seed)
    xs, ok = steady_state(m)
    assert ok, f"{form}: no steady state to lift about"
    return m, xs.detach().cpu().numpy()


@pytest.mark.parametrize("form", FORMS)
def test_gates_qss_reproduce_the_qss_reaction(form):
    """The lift's fixed-point gates, fed back through the production term, must give the
    reaction model.py already computes algebraically. If this drifts, the lift is a
    DIFFERENT model and nothing else here means anything."""
    m, xstar = _model_and_xstar(form)
    GA, GR = lifted.gates_qss(m, xstar)
    f_lift = (m.beta.detach().numpy() + lifted.production_from_gates(m, GA, GR)
              - m.delta.detach().numpy() * xstar)
    f_qss = m.reaction(torch.as_tensor(xstar)).detach().numpy()
    assert np.max(np.abs(f_lift - f_qss)) < 1e-12


@pytest.mark.parametrize("form", FORMS)
def test_fixed_point_is_mu_independent(form):
    """CLAIM 1. Setting dG/dt = 0 recovers the QSS algebra, so z* does not move with mu."""
    m, xstar = _model_and_xstar(form)
    for mu in MUS + [1.0, 1e3]:
        assert lifted.fixed_point_residual(m, xstar, mu) < 1e-7, mu


@pytest.mark.parametrize("form", FORMS)
def test_dispersion_converges_to_qss_first_order(form):
    """CLAIM 2. The comparison is restricted to the k where the SLOW branch is what the
    maximum picks out (|sigma_QSS| <= 10). Beyond it the QSS branch runs to -D_min k^2 while
    the lifted gate branch sits at -1/mu and takes over — a genuine structural difference
    between the two systems, not an error, and decades outside the physical band."""
    m, xstar = _model_and_xstar(form)
    sq, _ = lifted.qss_dispersion(m, xstar)
    sel = np.abs(sq) <= 10.0
    assert sel.sum() > 20
    errs = []
    for mu in MUS:
        sl, _ = lifted.lifted_dispersion(m, xstar, mu)
        errs.append(float(np.max(np.abs(sl[sel] - sq[sel]))))
    errs = np.array(errs)
    assert errs[-1] < 1e-3, errs
    assert np.all(np.diff(errs) < 0), errs                       # monotone in mu
    slope = np.polyfit(np.log10(MUS), np.log10(errs), 1)[0]
    assert 0.85 < slope < 1.15, f"expected O(mu), measured slope {slope}"


@pytest.mark.parametrize("form", FORMS)
def test_rescale_mu_matches_autodiff(form):
    """CLAIM 3."""
    m, xstar = _model_and_xstar(form)
    J1 = lifted.lifted_jacobian(m, xstar, 1.0)
    for mu in (1e-3, 0.1, 7.3):
        Jd = lifted.lifted_jacobian(m, xstar, mu)
        rel = np.max(np.abs(lifted.rescale_mu(J1, m.N, mu) - Jd)) / np.max(np.abs(Jd))
        assert rel < 1e-12, (mu, rel)


@pytest.mark.parametrize("form", FORMS)
def test_gate_step_is_exact(form):
    """CLAIM 4: the closed-form gate update against a brute-force explicit integration of
    the same ODE at frozen x. The competitive branch is the one at risk — its gates are
    coupled within a row through `free`, and the update is a rank-one matrix exponential."""
    m, xstar = _model_and_xstar(form)
    N = m.N
    rng = np.random.default_rng(0)
    X = np.clip(xstar[:, None, None] * (1 + 0.3 * rng.standard_normal((N, 3, 3))), 1e-9, None)
    GA0, GR0 = lifted.gates_qss(m, xstar)
    GA = np.broadcast_to(GA0[:, :, None, None], (N, N, 3, 3)).copy() * 0.3
    GR = np.broadcast_to(GR0[:, :, None, None], (N, N, 3, 3)).copy() * 0.7
    mu, dt, nsub = 1e-2, 0.05, 40000
    GAe, GRe = lifted.gate_step_exact(m, X, GA, GR, dt, mu)
    ga, gr, h = GA.copy(), GR.copy(), dt / nsub
    xn = np.clip(X, 0, None) ** m.n_hill
    ua = m.KA.detach().numpy()[:, :, None, None] * xn[None]
    ur = m.KR.detach().numpy()[:, :, None, None] * xn[None]
    for _ in range(nsub):
        if form == "competitive":
            free = 1.0 - ga.sum(1) - gr.sum(1)
            ga, gr = (ga + h * (ua * free[:, None] - ga) / mu,
                      gr + h * (ur * free[:, None] - gr) / mu)
        else:
            ga, gr = ga + h * (ua * (1 - ga) - ga) / mu, gr + h * (ur * (1 - gr) - gr) / mu
    assert max(np.max(np.abs(GAe - ga)), np.max(np.abs(GRe - gr))) < 1e-5


@pytest.mark.parametrize("form", FORMS)
def test_batched_jacobian_matches_serial(form):
    """The vmapped cloud Jacobian against the serial autodiff one, on the same parameters.
    The finite-mu volumes are computed entirely on the batched path."""
    m, xstar = _model_and_xstar(form)
    g = lambda a: a.detach().numpy()[None]
    J1b, N = lifted.lifted_jacobian_batch(g(m.KA), g(m.KR), g(m.alpha), g(m.beta),
                                          g(m.delta), xstar[None], form, m.n_hill)
    assert N == m.N
    J1 = lifted.lifted_jacobian(m, xstar, 1.0)
    assert np.max(np.abs(J1b[0] - J1)) < 1e-9


@pytest.mark.parametrize("form", FORMS)
def test_lifted_verdict_reduces_to_qss_verdict(form):
    """At mu -> 0 the lifted STRICT Turing verdict and k* must agree with the reduced (QSS)
    Jacobian's, on the same k-grid. This is the verdict-level version of the dispersion
    convergence, and it is the quantity the robustness volumes actually count."""
    m, xstar = _model_and_xstar(form)
    J1 = lifted.lifted_jacobian(m, xstar, 1.0)
    vl = lifted.verdicts_from_J(lifted.rescale_mu(J1, m.N, 1e-8)[None],
                                m.D.detach().numpy()[None])
    Jq = m.jacobian(torch.as_tensor(xstar), create_graph=False).detach().numpy()
    vq = lifted.verdicts_from_J(Jq[None], m.D.detach().numpy()[None])
    assert bool(vl["turing_strict"][0]) == bool(vq["turing_strict"][0])
    assert bool(vl["stable_uniform"][0]) == bool(vq["stable_uniform"][0])
    assert abs(vl["sig_max_pos"][0] - vq["sig_max_pos"][0]) < 1e-4


@pytest.mark.parametrize("form", FORMS)
def test_lifted_pde_reproduces_the_qss_field_at_small_mu(form):
    """CLAIM 4, spatially: at mu = 1e-6 the lifted PDE must reproduce the QSS rollout of the
    SAME model at the same L, seed, noise and dt. If it does not, a finite-mu "the pattern
    died" result would be an integrator artefact, which is the failure mode this whole unit
    has to rule out. Small grid so this stays a few seconds."""
    m, xstar = _model_and_xstar(form, N=2, seed=5)
    L, n = 30.0, 16
    rq = rollout.simulate(m, L=L, n=n, integrator="etdrk4_rfft", seed=0, xstar=xstar)
    rl = lifted.simulate_lifted(m, L=L, mu=1e-6, n=n, seed=0, xstar=xstar)
    assert rl["dt"] == pytest.approx(rq["dt"])
    assert rl["nsteps"] == rq["nsteps"]
    scale = max(1e-30, float(np.abs(rq["fields"]).max()))
    assert np.max(np.abs(rl["fields"] - rq["fields"])) / scale < 1e-3
