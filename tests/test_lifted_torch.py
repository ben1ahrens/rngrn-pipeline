"""test_lifted_torch.py — the GPU port of the lifted spatial integrator.

ONE CLAIM, PINNED HERE: `lifted_torch.simulate_lifted_torch` on the CPU reproduces
`lifted.simulate_lifted` to round-off on the SAME model, grid, dt, step count and seed.
That is what makes the GPU run a faster way to compute the same number rather than a
second, differently-behaved integrator — and it is checked on BOTH regulation forms,
because the gate substep's competitive branch (a rank-one matrix exponential) and its nc1
branch (diagonal) are different code.

The IC noise is drawn with numpy and transferred, deliberately: an equivalent torch draw
would be statistically identical and bit-DIFFERENT, which would turn an exact equivalence
check into a fuzzy one.

CUDA equivalence is asserted looser (1e-9, as tests/test_etdrk4_torch.py does): cuFFT and
pocketfft are different FFT backends, so bit-equality is not the contract there.
"""
import numpy as np
import pytest
import torch

from rngrn.eval import lifted, lifted_torch
from rngrn.eval.ladder import draw_models

torch.set_default_dtype(torch.float64)

# 400 steps of dt = 5e-4 on a 32x32 box: small enough to be seconds on CPU, long enough
# that a wrong gate substep or a mis-sequenced Strang half-step cannot hide in round-off.
KW = dict(L=20.0, n=32, mu=1e-3, dt=5e-4, T=0.2, seed=5)


def _pair(form, seed, device="cpu"):
    m = draw_models(n=1, form=form, seed=seed)[0]
    a = lifted.simulate_lifted(m, **KW)
    b = lifted_torch.simulate_lifted_torch(m, device=device, **KW)
    return m, a, b


def _initial_field(model):
    """The IC both integrators start from, so a test can prove the run actually MOVED."""
    from rngrn.losses.terms import steady_state
    xs, _ = steady_state(model)
    xstar = xs.detach().cpu().numpy()
    rng = np.random.default_rng(KW["seed"])
    return xstar[:, None, None] + 1e-2 * rng.standard_normal((model.N, KW["n"], KW["n"]))


@pytest.mark.parametrize("form,seed", [("competitive", 23), ("nc1", 23)])
def test_torch_lifted_matches_numpy_on_cpu(form, seed):
    m, a, b = _pair(form, seed)
    assert a["nsteps"] == 400 and b["nsteps"] == 400
    assert not a["blew_up"] and not b["blew_up"]
    assert a["nsteps_run"] == b["nsteps_run"] == 400
    # GUARD AGAINST A VACUOUS PASS. Two integrators that both did nothing would agree
    # perfectly, so the displacement from the IC must dwarf the disagreement.
    moved = float(np.abs(a["fields"] - _initial_field(m)).max())
    delta = float(np.abs(b["fields"] - a["fields"]).max())
    assert moved > 1e-3, f"{form}: the reference run barely moved ({moved:.3g})"
    assert delta < 1e-6 * moved, f"{form}: delta {delta:.3g} vs displacement {moved:.3g}"
    np.testing.assert_allclose(b["fields"], a["fields"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(b["GA"], a["GA"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(b["GR"], a["GR"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("form,seed", [("competitive", 23), ("nc1", 23)])
def test_torch_lifted_reports_the_same_summary(form, seed):
    """The scalar return contract — the fields are only half of it."""
    _, a, b = _pair(form, seed)
    assert set(b) == set(a)
    assert b["stopped_reason"] == a["stopped_reason"] == "horizon"
    assert b["patterned"] == a["patterned"]
    assert b["dt"] == a["dt"] and b["n"] == a["n"] and b["L"] == a["L"]
    for key in ("sig_max", "horizon_rate", "amplitude", "kstar"):
        np.testing.assert_allclose(b[key], a[key], rtol=1e-8, atol=1e-12,
                                   err_msg=f"summary key {key!r} disagrees")


def test_record_every_returns_the_same_frames():
    m = draw_models(n=1, form="competitive", seed=23)[0]
    kw = dict(KW, record_every=50)
    a = lifted.simulate_lifted(m, **kw)
    b = lifted_torch.simulate_lifted_torch(m, device="cpu", **kw)
    assert len(b["frames"]) == len(a["frames"]) == 8
    for fb, fa in zip(b["frames"], a["frames"]):
        np.testing.assert_allclose(fb, fa, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_torch_lifted_on_cuda_matches_numpy():
    """Same physics on the GPU. Looser than the CPU bar: cuFFT != pocketfft."""
    _, a, b = _pair("competitive", 23, device="cuda")
    assert not b["blew_up"]
    np.testing.assert_allclose(b["fields"], a["fields"], rtol=1e-9, atol=1e-11)
