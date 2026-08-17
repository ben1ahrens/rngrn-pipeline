"""test_numerics_contract.py — numerics module contract tests.

Ensures the stub functions raise as documented rather than silently delegating.
"""
import pytest
from rngrn.eval import numerics


def test_bdf1_stub_raises_rather_than_silently_delegating():
    """D-REDESIGN spec §5.2: a stub that silently returns another integrator's
    result is the D-EVID-2 defect class. It must raise until implemented.
    """
    with pytest.raises(NotImplementedError, match="bdf1_newton_krylov"):
        numerics.INTEGRATORS["bdf1_newton_krylov"](
            None, None, None, n=16, L=10.0, dt=0.1, nsteps=1
        )
