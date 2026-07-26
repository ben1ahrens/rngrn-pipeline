"""test_science.py — cheap correctness anchors (seconds, CPU).

These check the science invariants the pipeline depends on, not recovery success
(that is the milestone work Claude Code runs on CUDA):
  * autodiff Jacobian matches finite differences
  * general-N dispersion agrees with the 2x2 closed form away from the crossing
  * gradients flow to theta
  * the mu->0 lift reproduces the QSS reaction
  * ETDRK4 stays finite where split-step IMEX collapses under stiff diffusion
  * the reference answer keys reproduce the documented k*
"""
import numpy as np
import torch

from rngrn.model import RNGRN


def test_positivity_and_param_count():
    m = RNGRN(N=2, form="competitive", seed=0)
    assert torch.all(m.KA >= 0) and torch.all(m.KR >= 0) and torch.all(m.D > 0)
    assert torch.allclose(m.KA + m.KR, m.s, atol=1e-6)
    n_params = sum(p.numel() for p in m.parameters())
    assert n_params == 3 * m.N**2 + 3 * m.N


def test_autodiff_jacobian_matches_fd():
    m = RNGRN(N=2, seed=1)
    x = torch.tensor([1.0, 0.9])
    J = m.jacobian(x, create_graph=False).detach().numpy()
    eps = 1e-6; f0 = m.reaction(x).detach().numpy(); Jfd = np.zeros((2, 2))
    for j in range(2):
        xp = x.clone(); xp[j] += eps
        Jfd[:, j] = (m.reaction(xp).detach().numpy() - f0) / eps
    assert np.max(np.abs(J - Jfd)) < 1e-5


def test_general_vs_2x2_dispersion():
    m = RNGRN(N=2, seed=2)
    x = torch.tensor([1.0, 0.9])
    J = m.jacobian(x)
    kg = torch.linspace(0.05, 5.0, 60)
    sig_gen = m.dispersion(x, kg, J=J).detach().numpy()
    sig_2x2 = m.dispersion_2x2(J, kg).detach().numpy()
    # agree away from the discriminant crossing (real-eig region)
    ok = np.abs(sig_gen - sig_2x2) < 0.05
    assert ok.mean() > 0.7


def test_gradients_flow_to_theta():
    m = RNGRN(N=2, seed=3)
    x = torch.tensor([1.0, 0.9])
    sig = m.dispersion(x, torch.linspace(0.05, 5.0, 40))
    loss = sig.max()
    loss.backward()
    assert m.theta_D.grad is not None and torch.any(m.theta_D.grad != 0)


def test_mu_zero_lift_reduces_to_qss():
    from rngrn.eval.dynamical import lift_check
    m = RNGRN(N=2, seed=4)
    diff = lift_check(m, [1.0, 0.9], mu=1e-4, n=6000)
    assert diff < 1e-2, f"lift did not reduce to QSS: {diff}"


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
