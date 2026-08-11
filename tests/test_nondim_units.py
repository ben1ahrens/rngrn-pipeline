"""test_nondim_units.py — the physical-D contract on the non-dimensional path (D-EVID-14).

`recover(nondim=True)` optimises on the unit box, where the learned parameter is
D_hat = D / L**2. `recover()` returns the model UNCHANGED, so `result.model.D` holds that
dimensionless value while `result.D_phys` holds the physical one. Any consumer that reads
`result.model.D` and then applies it at the PHYSICAL length scale is off by L**2 — at
L = 60 that is a factor of 3600.

MEASURED on the branch's own Turing fixture (L = 60):

    turing_volume_*   physical D : 1.000 / 0.995 / 0.835 / 0.595
                      nondim   D : 0.000 / 0.005 / 0.010 / 0.015

i.e. a maximally robust circuit reported as maximally fragile — a WRONG NUMBER, not a
crash. These tests pin the contract: every physics consumer takes D explicitly, and
`score_recovery` / the morphology rollout feed it `D_phys`.

No committed config sets `nondim: true` today (all 13 tracked run rows carry nondim=False),
so no recorded result is affected — but PREREGISTRATION §3.5b is the natural trigger, and
this must be correct before that runs.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_rollout import turing_model, FIXTURE_L          # noqa: E402

from rngrn.eval.analysis import robustness_volumes, robustness_cloud   # noqa: E402
from rngrn.losses.terms import steady_state                            # noqa: E402
from rngrn.validate import score_recovery                              # noqa: E402


def _fixture():
    m = turing_model()
    xs, _ = steady_state(m)
    return m, xs.detach().numpy(), m.D.detach().numpy().copy()


class _NondimResult:
    """A RecoveryResult as `recover(nondim=True)` returns one: model.D is D/L**2,
    D_phys is the physical value."""

    def __init__(self, model, xstar, D_phys, L):
        self.model = model
        self.xstar = xstar
        self.D_phys = np.asarray(D_phys, float)
        self.L = float(L)
        self.nondim = True
        self.kstar_model = 0.626
        self.latent_fields = None
        # the model itself carries the DIMENSIONLESS parameter
        with torch.no_grad():
            model.theta_D.copy_(torch.log(torch.as_tensor(self.D_phys / self.L ** 2)))


class _Key:
    def __init__(self, J, D):
        self.J = np.asarray(J, float)
        self.n_species_true = J.shape[0]
        self.kstar = 0.626
        self.x_star = None
        self.sigma_max = None
        self.D = np.asarray(D, float)
        self.coefficients = None


# ======================================================================================
# the physics helpers must accept D explicitly
# ======================================================================================
def test_robustness_volumes_accepts_an_explicit_D():
    m, xs, D_phys = _fixture()
    nondim_D = D_phys / FIXTURE_L ** 2

    from_model = robustness_volumes(m, xstar=xs)
    explicit = robustness_volumes(m, xstar=xs, D=D_phys)
    assert explicit == from_model, "explicit D equal to model.D must change nothing"

    scaled = robustness_volumes(m, xstar=xs, D=nondim_D)
    assert scaled != from_model, "D must actually be used, not ignored"


def test_robustness_cloud_accepts_an_explicit_D():
    m, xs, D_phys = _fixture()
    assert robustness_cloud(m, xstar=xs, D=D_phys)["frac_turing"] == pytest.approx(
        robustness_cloud(m, xstar=xs)["frac_turing"])


def test_simulate_accepts_an_explicit_D():
    """The rollout integrates D lap(x) + f(x); D must be the PHYSICAL one at physical L."""
    from rngrn.eval.rollout import simulate
    m, xs, D_phys = _fixture()
    kw = dict(L=FIXTURE_L, n=32, seed=0, noise=1e-2, xstar=xs, max_steps=200)

    same = simulate(m, D=D_phys, **kw)
    assert np.isfinite(same["sig_max"])

    # the dimensionless D at the physical length scale cannot pattern — 3600x too small
    starved = simulate(m, D=D_phys / FIXTURE_L ** 2, **kw)
    assert starved["sig_max"] < same["sig_max"], "D must reach the integrator"


# ======================================================================================
# and the scorers must FEED them the physical value
# ======================================================================================
def test_score_recovery_reports_robustness_on_physical_D():
    """THE DEFECT. A nondim result must give the same turing_volume_* as the physical one."""
    m_phys, xs, D_phys = _fixture()
    J = m_phys.jacobian(torch.as_tensor(xs), create_graph=False).detach().numpy()
    physical = robustness_volumes(m_phys, xstar=xs, D=D_phys)

    res = _NondimResult(turing_model(), xs, D_phys, FIXTURE_L)
    assert not np.allclose(res.model.D.detach().numpy(), D_phys), "fixture precondition"

    out = score_recovery(res, _Key(J, D_phys), observed_idx=[0, 1, 2], L=FIXTURE_L)
    for k, v in physical.items():
        assert out[k] == pytest.approx(v), (
            f"{k}: nondim path reported {out[k]}, physical value is {v} — score_recovery "
            f"is reading model.D (dimensionless) instead of result.D_phys")

    # and the failure this prevents is not subtle
    wrong = robustness_volumes(res.model, xstar=xs)
    assert wrong["turing_volume_10pct"] < 0.1 < physical["turing_volume_10pct"]


def test_overparam_report_uses_physical_D():
    """`D_observed_max` is a reported diagnostic; on the nondim path it was 1/L**2 off."""
    from rngrn.scoring import overparam as OVER
    m_phys, xs, D_phys = _fixture()
    res = _NondimResult(turing_model(), xs, D_phys, FIXTURE_L)
    key = _Key(np.ones((2, 2)), D_phys[:2])

    rep = OVER.overparam_report(res, key, [0, 1])
    assert rep["D_observed_max"] == pytest.approx(float(np.abs(D_phys[:2]).max())), (
        "overparam_report must read result.D_phys, not the dimensionless model.D")
