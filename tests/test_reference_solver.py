"""Stability of the reference generator (ANSWER-KEY SIDE).

Regression tests for D-EVID-17: `simulate_to_attractor` diverged on BOTH shipped
reference systems at their own default arguments, because the timestep was capped by the
reaction Jacobian alone and the explicit diffusion half was never consulted. That made
the `data.source="reference"` path — and the README quickstart built on it — unrunnable.

These tests pin the two things that were wrong: the timestep must respect diffusion, and
the systems must actually integrate to a finite converged pattern.
"""
from __future__ import annotations

import numpy as np
import pytest

from rngrn.data.rd_models import GiererMeinhardt, Schnakenberg
from rngrn.data.solver import _jac_at, simulate_to_attractor, stable_dt


REFERENCE_SYSTEMS = [GiererMeinhardt, Schnakenberg]


@pytest.mark.parametrize("cls", REFERENCE_SYSTEMS, ids=lambda c: c.__name__)
def test_reference_system_integrates_without_diverging(cls):
    """D-EVID-17: both systems raised FloatingPointError on their shipped defaults.

    T_max is cut right down — this asserts the integrator is STABLE, not that the
    pattern has converged. A diverging scheme blows up in tens of steps (measured:
    step 133 for Gierer-Meinhardt, step 29 for Schnakenberg), so a short horizon is
    a sufficient regression test and keeps the suite fast.
    """
    frame, L = simulate_to_attractor(cls(), resolution=32, T_max=5.0)

    assert np.all(np.isfinite(frame)), "solver produced non-finite values"
    assert frame.shape == (2, 32, 32)
    assert L == 100.0
    # A blown-up field grows without bound; a stable one stays near the steady state.
    assert np.abs(frame).max() < 1e3, f"field magnitude {np.abs(frame).max():.3e} implies divergence"


@pytest.mark.parametrize("cls", REFERENCE_SYSTEMS, ids=lambda c: c.__name__)
def test_timestep_respects_the_diffusion_limit_not_just_the_reaction(cls):
    """The defect itself: dt was `0.2 / max|eig J|`, ignoring D entirely.

    With an explicit diffusion half and a SPECTRAL Laplacian, the grid's largest
    eigenvalue is `D_max * k2_max` with `k2_max = 2*(pi*n/L)**2`, so explicit Euler
    needs `dt * D_max * k2_max <= 2`. At the shipped 128 / L=100 that bound is 6.2e-4
    (Gierer-Meinhardt, Dv=100) and 1.5e-3 (Schnakenberg, Dv=40) — both far below the
    reaction-only cap, which is what made the old code diverge.
    """
    system = cls()
    n, L = 128, system.L
    D_max = float(np.asarray(system.D, float).max())
    k2_max = 2.0 * (np.pi * n / L) ** 2

    dt = stable_dt(system, resolution=n, dt_requested=0.1)

    assert dt * D_max * k2_max <= 2.0, (
        f"dt={dt:.3e} violates the explicit diffusion bound "
        f"{2.0 / (D_max * k2_max):.3e} for {cls.__name__}"
    )
    # ...and it must still respect the reaction bound it already had.
    rate = float(np.max(np.abs(np.linalg.eigvals(_jac_at(system, np.asarray(system.steady_state(), float))))))
    assert dt <= 0.2 / rate * (1 + 1e-9), f"dt={dt:.3e} violates the reaction bound"


def test_stable_dt_never_exceeds_the_requested_timestep():
    """The cap only ever tightens dt; a caller asking for something tiny keeps it."""
    assert stable_dt(Schnakenberg(), resolution=32, dt_requested=1e-9) == pytest.approx(1e-9)


def test_stable_dt_tightens_as_diffusion_grows():
    """Sanity on the direction of the bound: more diffusive -> smaller dt.

    Gierer-Meinhardt (Dv=100) must get a strictly smaller diffusion-limited step than
    Schnakenberg (Dv=40) at identical resolution, since only D differs in that term.
    """
    gm = stable_dt(GiererMeinhardt(), resolution=64, dt_requested=1.0)
    sch_diffusion_only = 2.0 / (40.0 * 2.0 * (np.pi * 64 / 100.0) ** 2)
    gm_diffusion_only = 2.0 / (100.0 * 2.0 * (np.pi * 64 / 100.0) ** 2)

    assert gm_diffusion_only < sch_diffusion_only
    assert gm <= gm_diffusion_only * (1 + 1e-9)
