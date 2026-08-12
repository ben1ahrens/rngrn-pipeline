"""test_etdrk4_torch.py — the torch ETDRK4 port (rngrn.etdrk4_torch).

Re-pins in the suite what diagnostic D2 measured once: the torch integrator is
numerically equivalent to the numpy reference `eval/numerics.integrate_etdrk4_rfft`
(D2 measured max |delta| 1.1e-13 over 100 steps; asserted at 1e-12 here, D2's own
tolerance). CUDA equivalence is asserted looser (1e-9) — cuFFT and pocketfft are
different FFT backends, so bit-equality is not the contract there; same-physics is.
"""
import pathlib

import numpy as np
import pytest
import torch

from rngrn.etdrk4_torch import (_torch_reaction_builder, integrate_etdrk4_rfft_torch,
                                torch_half_coeffs)
from rngrn.eval.numerics import integrate_etdrk4_rfft
from rngrn.eval.rollout import _reaction_np_builder
from rngrn.losses.terms import steady_state
from rngrn.model import RNGRN

torch.set_default_dtype(torch.float64)

REPO = pathlib.Path(__file__).resolve().parents[1]
CKPT = (REPO / "experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/"
        "checkpoints/model.pt")


def _fixture_model() -> RNGRN:
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    m = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    m.load_state_dict(ck["state_dict"])
    return m


def _equivalence_setup(n=48, L=100.0):
    model = _fixture_model()
    xs, ok = steady_state(model)
    assert ok
    xstar = xs.detach().numpy()
    J = model.jacobian(xs, create_graph=False).detach()
    dt = 0.2 / float(torch.linalg.eigvals(J).abs().max())
    rng = np.random.default_rng(42)
    X0 = xstar[:, None, None] + 1e-2 * rng.standard_normal((model.N, n, n))
    D = model.D.detach().numpy()
    Xnp, blew = integrate_etdrk4_rfft(X0.copy(), D, _reaction_np_builder(model),
                                      n, L, dt, 100)
    assert not blew
    return model, X0, D, n, L, dt, Xnp


def test_torch_integrator_matches_numpy_on_cpu():
    """D2's torch_equivalence, re-pinned: same IC, 100 steps, max |delta| <= 1e-12."""
    model, X0, D, n, L, dt, Xnp = _equivalence_setup()
    dev = torch.device("cpu")
    with torch.no_grad():
        coeffs = torch_half_coeffs(D, n, L, dt, dev)
        Xt, blew = integrate_etdrk4_rfft_torch(
            torch.from_numpy(X0.copy())[None], _torch_reaction_builder(model, dev),
            n, dt, 100, coeffs)
    assert not blew
    delta = float(np.max(np.abs(Xt[0].numpy() - Xnp)))
    assert delta <= 1e-12, f"torch/numpy ETDRK4 diverged: max |delta| = {delta:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_torch_integrator_matches_numpy_on_cuda():
    """Same physics on the GPU: 1e-9 tolerance (different FFT backend, not bit-equal —
    D3 measured the FFT round-off floor around 1e-16 of peak, so 1e-9 over 100 steps is
    backend noise, not a scheme difference)."""
    model, X0, D, n, L, dt, Xnp = _equivalence_setup()
    dev = torch.device("cuda")
    with torch.no_grad():
        coeffs = torch_half_coeffs(D, n, L, dt, dev)
        Xt, blew = integrate_etdrk4_rfft_torch(
            torch.from_numpy(X0.copy())[None].to(dev),
            _torch_reaction_builder(model, dev), n, dt, 100, coeffs)
    assert not blew
    delta = float(np.max(np.abs(Xt[0].cpu().numpy() - Xnp)))
    assert delta <= 1e-9, f"CUDA/numpy ETDRK4 diverged: max |delta| = {delta:.2e}"


def test_nc1_form_is_refused_loudly():
    """The reaction port covers the competitive form only (as in D2); a half-ported nc1
    would poison the equivalence contract, so it must refuse, not guess."""
    m = RNGRN(N=2, form="nc1", seed=0)
    with pytest.raises(NotImplementedError, match="competitive"):
        _torch_reaction_builder(m, torch.device("cpu"))
