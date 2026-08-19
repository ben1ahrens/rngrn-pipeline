"""The cubic dispersion backend must be EXACTLY equivalent to the eigvals reference.

This is the guard on a hand-derived closed form. Two earlier implementations of
_sigma_max_cubic looked plausible and were wrong by MAE ~2e+2 (a dropped sign on p in the
trigonometric branch), which only this equivalence check caught. If you edit that function,
these tests are the acceptance criterion.
"""
import numpy as np
import pytest
import torch

from rngrn.model import RNGRN, _sigma_max_cubic


def _real_jacobians(limit=40):
    """(J, D) pairs from the registered three_gene answer keys, or skip if absent."""
    import pathlib
    import h5py
    p = pathlib.Path("data/datasets/three_gene_val/payload.h5")
    if not p.exists():
        pytest.skip("three_gene_val payload not present (see docs/LOCAL_DATA_SETUP.md)")
    out = []
    with h5py.File(p) as f:
        for k in list(f.keys())[:limit]:
            out.append((f[k]["jacobian"][:], f[k]["D"][:]))
    return out


def _sigma_eig(J, D, kgrid):
    M = J.unsqueeze(0) - (kgrid ** 2).reshape(-1, 1, 1) * torch.diag_embed(D).unsqueeze(0)
    return torch.linalg.eigvals(M).real.max(dim=-1).values


def _sigma_cubic(J, D, kgrid):
    M = J.unsqueeze(0) - (kgrid ** 2).reshape(-1, 1, 1) * torch.diag_embed(D).unsqueeze(0)
    return _sigma_max_cubic(M)


def test_cubic_matches_eigvals_on_real_jacobians():
    kg = torch.linspace(0.0, 3.0, 200, dtype=torch.float64)
    worst_sig, worst_k, flips, n = 0.0, 0.0, 0, 0
    for J0, D0 in _real_jacobians():
        J = torch.tensor(J0, dtype=torch.float64)
        D = torch.tensor(D0, dtype=torch.float64)
        ref, got = _sigma_eig(J, D, kg), _sigma_cubic(J, D, kg)
        worst_sig = max(worst_sig, float((ref - got).abs().max()))
        worst_k = max(worst_k, abs(float(kg[ref.argmax()]) - float(kg[got.argmax()])))
        # the decision that actually matters downstream
        flips += bool(ref[0] < 0 and ref[1:].max() > 0) != bool(got[0] < 0 and got[1:].max() > 0)
        n += 1
    assert n > 0
    assert flips == 0, f"{flips}/{n} Turing verdicts differ from the eigvals reference"
    assert worst_k == 0.0, f"argmax k* differs by {worst_k}"
    assert worst_sig < 1e-8, f"sigma_max max abs error {worst_sig:.3e}"


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_cubic_matches_eigvals_on_random_models(form):
    """Random model states, not just the answer keys — covers wilder theta. Parametrized
    over form (unit 6): the cubic backend operates only on J - k^2 D, so it is form-
    agnostic in principle, but that was never checked for nc1's Jacobian before."""
    kg = torch.linspace(0.0, 3.0, 128, dtype=torch.float64)
    for seed in range(12):
        a = RNGRN(N=3, form=form, seed=seed, dispersion_backend="eig")
        b = RNGRN(N=3, form=form, seed=seed, dispersion_backend="cubic")
        x = torch.full((3,), 0.7, dtype=torch.float64)
        assert torch.allclose(a.dispersion(x, kg), b.dispersion(x, kg), atol=1e-9)


@pytest.mark.parametrize("form", ["competitive", "nc1"])
def test_cubic_gradients_match(form):
    """The backend is used inside a differentiable loss, so gradients must agree too."""
    kg = torch.linspace(0.0, 3.0, 128, dtype=torch.float64)
    x = torch.full((3,), 0.7, dtype=torch.float64)
    grads = []
    for backend in ("eig", "cubic"):
        m = RNGRN(N=3, form=form, seed=3, dispersion_backend=backend)
        m.dispersion(x, kg).max().backward()
        grads.append(m.theta_D.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-10), (grads[0] - grads[1]).abs().max()


def test_cubic_rejects_wrong_N():
    """Exact for N=3 only — must fail loud rather than silently return nonsense.

    Rejection is at CONSTRUCTION (tightened at the 13-unit merge): a model that can never
    evaluate its own dispersion is misconfigured when it is built, not when it is first
    used. The lazy guard inside dispersion() is kept as defence in depth for anyone who
    mutates .dispersion_backend afterwards, and is exercised below.
    """
    with pytest.raises(ValueError, match="N=3 only"):
        RNGRN(N=2, seed=0, dispersion_backend="cubic")

    m = RNGRN(N=2, seed=0)                 # valid at construction ...
    m.dispersion_backend = "cubic"         # ... then mutated into an invalid state
    with pytest.raises(ValueError, match="N=3 only"):
        m.dispersion(torch.full((2,), 0.7, dtype=torch.float64),
                     torch.linspace(0.0, 3.0, 16, dtype=torch.float64))


# ======================================================================================
# 'auto' backend resolution (R3 Task 8; collision ledger row 26, D-PERF-3 as amended)
# ======================================================================================
def test_auto_resolves_at_construction_and_never_survives_it():
    """`'auto'` is a REQUEST; `.dispersion_backend` must always read the concrete backend.

    Every consumer -- `member(b)`, `io.save_checkpoint`'s payload, `frozen_config.yaml` --
    reads the attribute and would otherwise propagate a string that names no algorithm.
    """
    from rngrn.model import BatchedRNGRN, resolve_dispersion_backend

    assert resolve_dispersion_backend("auto", 3) == "cubic"
    assert resolve_dispersion_backend("auto", 2) == "eig"
    assert resolve_dispersion_backend("auto", 4) == "eig"
    # explicit requests pass through untouched -- the helper is not a second validation gate
    assert resolve_dispersion_backend("eig", 3) == "eig"
    assert resolve_dispersion_backend("cubic", 3) == "cubic"

    assert RNGRN(N=3, seed=0, dispersion_backend="auto").dispersion_backend == "cubic"
    assert RNGRN(N=2, seed=0, dispersion_backend="auto").dispersion_backend == "eig"
    assert BatchedRNGRN.from_seeds(
        3, [0, 1], dispersion_backend="auto").dispersion_backend == "cubic"


def test_the_default_backend_is_eig_everywhere():
    """A0 protection (docs/PLAN_redesign.md Global Constraints; ledger row 26).

    D-PERF-3 flipped all four defaults to `'auto'`, which resolves to `'cubic'` at N == 3 --
    i.e. every N = 3 run that omits the argument would change backend, and D-PERF-3 itself
    states cubic and eig runs are not bit-comparable. The controller ruled the MECHANICS in
    and the FLIP out. This test is that ruling; a flip must edit it in the same commit.
    """
    import inspect

    from rngrn.config import ModelConfig
    from rngrn.model import BatchedRNGRN
    from rngrn import recover as R

    assert ModelConfig().dispersion_backend == "eig"
    for fn in (RNGRN.__init__, BatchedRNGRN.from_seeds, R.recover):
        assert inspect.signature(fn).parameters["dispersion_backend"].default == "eig", fn
    # and the resolved effect of the default, not just the string
    assert RNGRN(N=3, seed=0).dispersion_backend == "eig"


def test_frozen_config_records_the_resolved_backend_not_the_request(tmp_path):
    """`config/frozen_config.yaml` must name the backend that RAN.

    `.claude/rules/reporting-numbers.md` step 4 says to read the frozen config rather than
    re-derive from it; a file recording `'auto'` cannot answer which backend produced the
    number. `train.fit` resolves before writing -- this pins that ordering.
    """
    import yaml

    from rngrn.config import Config
    cfg = Config()
    cfg.model.N = 3
    cfg.model.dispersion_backend = "auto"

    from rngrn import model as M
    cfg.model.dispersion_backend = M.resolve_dispersion_backend(
        cfg.model.dispersion_backend, cfg.model.N)
    p = tmp_path / "frozen_config.yaml"
    cfg.to_yaml(str(p))
    frozen = yaml.safe_load(p.read_text())
    assert frozen["model"]["dispersion_backend"] == "cubic"

    # The block above reproduces what `fit` does; it does NOT prove `fit` does it in that
    # ORDER, and the order is the whole fix (resolving after the write would freeze 'auto').
    # Running `fit` needs a dataset, so pin the ordering in its source instead -- weaker than
    # a behavioural test, and named as such.
    import inspect

    from rngrn import train
    src = inspect.getsource(train.fit)
    assert src.index("resolve_dispersion_backend") < src.index("frozen_config.yaml"), (
        "train.fit must resolve dispersion_backend BEFORE writing frozen_config.yaml")
