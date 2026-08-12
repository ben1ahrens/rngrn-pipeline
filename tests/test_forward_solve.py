"""test_forward_solve.py — the M1 forward/adjoint module (rngrn.forward).

Four contracts, per the M1 plan:

1. IFT-vs-FD regression on the ported chain (the load-bearing one). Tolerance 5e-3 is a
   REGRESSION TRIPWIRE against port breakage — a wrong adjoint, a dropped correction, or
   a sign error produces O(1) errors — NOT a re-acceptance of D1. D1's acceptance
   (worst rel err 6.4e-8 spectral / 8.1e-8 real_moments vs tol 1e-4 at 96^2,
   10 directions x 5 terms; the 64^2 probe's worst was 2.8e-5) stands, recorded in
   experiments/diag_fft/d1/results.json and re-runnable via scripts/diag_fft_d1.py.
   Here: 64^2, one direction, two term forms, eps in {1e-4, 1e-5} best-of.
2. One adjoint solve per backward(): all terms consuming one PatternSolve output share a
   single minimal-norm adjoint solve (D-FFT-10 machinery is per-step, not per-term).
3. The PatternSolver warm-start state machine transitions (mocked internals — the
   transitions are the contract, not the numerics, which contract 1 covers).
4. Translation modes of a converged pattern are near-null for the linearisation
   (finding F-D1-2: measured ~8e-4 at 64^2; asserted < 1e-2).
"""
import pathlib
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rngrn.forward as fwd
from rngrn import observables as obs
from rngrn.forward import THETA_NAMES, PatternSolver
from rngrn.losses.terms import steady_state
from rngrn.model import RNGRN

torch.set_default_dtype(torch.float64)

REPO = pathlib.Path(__file__).resolve().parents[1]
CKPT = (REPO / "experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/"
        "checkpoints/model.pt")
N_GRID = 64
PERIODS = 6.0
SEED = 0


def _load_ckpt() -> RNGRN:
    assert CKPT.exists(), (
        f"tracked fixture checkpoint missing: {CKPT} — it is committed to git; "
        f"a missing file means a broken checkout, not a skippable test")
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    assert not ck.get("nondim"), "fixture must be a dimensional run"
    m = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    m.load_state_dict(ck["state_dict"])
    return m


@pytest.fixture(scope="module")
def base():
    """The known-Turing fixture solved to its patterned state once for the module."""
    model = _load_ckpt()
    xs, ok = steady_state(model)
    assert ok, "steady state failed on the fixture checkpoint"
    J = model.jacobian(xs, create_graph=False).detach()
    kg = torch.linspace(1e-3, 10.0, 2000)
    sig = model.dispersion(xs, kg, J=J).detach()
    assert float(sig.max()) > 0, "fixture is not Turing-unstable"
    kstar_lin = float(kg[int(sig.argmax())])
    L = PERIODS * 2.0 * np.pi / kstar_lin

    solver = PatternSolver(model, N_GRID, L, SEED)
    t0 = time.perf_counter()
    u, reason = solver.solve()
    base_seconds = time.perf_counter() - t0
    assert reason == "ok", f"base solve did not pattern: {reason}"
    assert u.requires_grad, "PatternSolver must return the differentiable-through-IFT u*"
    kstar_pat = float(obs.kstar_of(u[0].detach().numpy(), L=L))
    return SimpleNamespace(model=model, solver=solver, u=u, L=L,
                           kstar=kstar_pat, base_seconds=base_seconds)


# ---------------------------------------------------------------- local term closures
# Deliberately NOT imported from losses/spectral.py (separate unit): the arithmetic is
# ported from scripts/diag_fft_d1.py::make_loss_terms so this test pins rngrn.forward
# alone. Any fixed target is valid for a gradient check.

def _torch_raps_power(field: torch.Tensor, L: float):
    n = field.shape[-1]
    f = field - field.mean()
    P = torch.abs(torch.fft.fft2(f)) ** 2
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    KX, KY = np.meshgrid(k, k)
    KR = np.sqrt(KX**2 + KY**2)
    dk = 2.0 * np.pi / L
    kbins = np.arange(0.0, KR.max() + dk, dk)
    which = np.clip(np.digitize(KR.ravel(), kbins) - 1, 0, len(kbins) - 2)
    counts = np.bincount(which, minlength=len(kbins) - 1)
    idx = torch.from_numpy(which)
    power = torch.zeros(len(kbins) - 1, dtype=field.dtype).scatter_add_(0, idx,
                                                                        P.reshape(-1))
    power = power / torch.from_numpy(np.maximum(counts, 1)).to(field.dtype)
    power = power.clone()
    power[0] = 0.0
    return 0.5 * (kbins[:-1] + kbins[1:]), power


def _make_terms(u_base: torch.Tensor, L: float, kstar: float):
    """spec_shape-form (band-limited log-RAPS) and spec_amp_fluct-form (log-std) closures
    with FIXED targets perturbed off the base pattern's own statistics, so gradients at
    the base point are nonzero."""
    rng = np.random.default_rng(7)
    kc, p = _torch_raps_power(u_base[0], L)
    mask_np = (kc >= 0.5 * kstar) & (kc <= 1.5 * kstar)
    mask = torch.from_numpy(mask_np)
    t_raw = (p[mask] / p[mask].sum()).detach()
    t_band = t_raw * torch.from_numpy(np.exp(0.2 * rng.standard_normal(len(t_raw))))
    t_band = (t_band / t_band.sum()).detach()
    t_logstd = [float(torch.log(u_base[c].std())) + 0.1 for c in range(u_base.shape[0])]

    def spec_shape(u):
        _, pw = _torch_raps_power(u[0], L)
        s = pw[mask] / pw[mask].sum()
        return ((torch.log(s + 1e-300) - torch.log(t_band + 1e-300)) ** 2).sum()

    def amp_fluct(u):
        return sum((torch.log(u[c].std()) - t_logstd[c]) ** 2
                   for c in range(u.shape[0]))

    return spec_shape, amp_fluct


def _flat_direction(model: RNGRN, seed: int) -> np.ndarray:
    sizes = [getattr(model, nm).numel() for nm in THETA_NAMES]
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(sum(sizes))
    return v / np.linalg.norm(v)


def _perturbed_model(v: np.ndarray, scale: float) -> RNGRN:
    m2 = _load_ckpt()
    off = 0
    with torch.no_grad():
        for nm in THETA_NAMES:
            p = getattr(m2, nm)
            dv = torch.from_numpy(v[off:off + p.numel()]).reshape(p.shape)
            p.add_(scale * dv)
            off += p.numel()
    return m2


# ---------------------------------------------------------------- 1. IFT vs FD

def test_ift_gradient_matches_finite_differences(base):
    """Regression tripwire (5e-3), not a re-acceptance — see module docstring."""
    spec_shape, amp_fluct = _make_terms(base.u.detach(), base.L, base.kstar)
    params = [getattr(base.model, nm) for nm in THETA_NAMES]

    loss = spec_shape(base.u) + amp_fluct(base.u)
    g = torch.autograd.grad(loss, params)
    g_flat = torch.cat([gi.reshape(-1) for gi in g]).numpy()
    v = _flat_direction(base.model, SEED + 1)
    d_ift = float(g_flat @ v)

    rel_errs = {}
    t0 = time.perf_counter()
    for eps in (1e-4, 1e-5):
        vals = {}
        for sgn in (+1.0, -1.0):
            m2 = _perturbed_model(v, sgn * eps)
            s2 = PatternSolver(m2, N_GRID, base.L, SEED)
            s2._warm = base.u.detach().clone()   # FD points: Newton-ONLY warm start
            u2, reason = s2.solve()
            assert reason == "ok", f"FD solve failed at eps={eps} sgn={sgn}: {reason}"
            drift = float((u2.detach() - base.u.detach()).norm()
                          / base.u.detach().norm())
            assert drift < 0.2, f"FD solve left the base branch (drift {drift:.2f})"
            vals[sgn] = float(spec_shape(u2).detach() + amp_fluct(u2).detach())
        d_fd = (vals[+1.0] - vals[-1.0]) / (2.0 * eps)
        rel_errs[eps] = abs(d_fd - d_ift) / max(abs(d_fd), abs(d_ift), 1e-300)
    fd_seconds = time.perf_counter() - t0

    best = min(rel_errs.values())
    print(f"\n[timing] base solve {base.base_seconds:.1f}s, FD sweep {fd_seconds:.1f}s; "
          f"rel errs {rel_errs} (best {best:.2e})")
    assert best <= 5e-3, (
        f"IFT-vs-FD relative error {best:.2e} exceeds the 5e-3 regression tripwire "
        f"(per-eps: {rel_errs}) — the port has broken the verified chain; compare "
        f"against scripts/diag_fft_d1.py")


# ---------------------------------------------------------------- 2. one adjoint/step

def test_one_adjoint_solve_per_backward(base, monkeypatch):
    calls = []
    orig = fwd.solve_adjoint

    def spy(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(fwd, "solve_adjoint", spy)
    u, reason = base.solver.solve()          # warm re-solve, fresh graph
    assert reason == "ok"
    spec_shape, amp_fluct = _make_terms(u.detach(), base.L, base.kstar)
    for p in base.model.parameters():
        p.grad = None
    (spec_shape(u) + amp_fluct(u)).backward()
    assert len(calls) == 1, (
        f"expected exactly ONE adjoint solve for a combined loss, got {len(calls)} — "
        f"terms must share the accumulated cotangent at the u* node")
    for nm in THETA_NAMES:
        p = getattr(base.model, nm)
        assert p.grad is not None, f"{nm} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{nm} gradient is not finite"
    for p in base.model.parameters():
        p.grad = None


# ---------------------------------------------------------------- 3. state machine

@pytest.fixture()
def machine():
    """A solver whose numerics are mocked per-test: state transitions are the contract."""
    model = _load_ckpt()
    xs, ok = steady_state(model)
    assert ok
    xstar = xs.detach().numpy()
    n = 16
    solver = PatternSolver(model, n, 10.0, 0)
    flat = torch.from_numpy(np.broadcast_to(xstar[:, None, None],
                                            (model.N, n, n)).copy())
    wavy = flat.clone()
    xg = torch.linspace(0, 2 * np.pi, n)
    wavy[0] = wavy[0] + 1.0 * torch.sin(xg)[None, :]
    return SimpleNamespace(solver=solver, flat=flat, wavy=wavy)


def _no_relax(*args, **kwargs):
    raise AssertionError("relax_to_pattern must not be called on this path")


def test_warm_success_skips_the_relax(machine, monkeypatch):
    monkeypatch.setattr(fwd, "relax_to_pattern", _no_relax)
    monkeypatch.setattr(fwd, "newton_polish",
                        lambda F_fn, u0, *a, **k: (u0, 1e-12))
    machine.solver._warm = machine.wavy.clone()
    u, reason = machine.solver.solve()
    assert reason == "ok" and u is not None
    assert machine.solver._warm is not None


def test_not_patterned_resets_the_warm_state(machine, monkeypatch):
    monkeypatch.setattr(fwd, "newton_polish",
                        lambda F_fn, u0, *a, **k: (u0, 1e-12))
    machine.solver._warm = machine.flat.clone()   # converged but homogeneous
    u, reason = machine.solver.solve()
    assert u is None and reason == "not_patterned"
    assert machine.solver._warm is None, (
        "a homogeneous field must not survive as a warm start — Newton from it "
        "re-converges homogeneous forever")


def test_relax_failure_reports_solve_failed_and_clears_warm(machine, monkeypatch):
    monkeypatch.setattr(fwd, "relax_to_pattern",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("blew up")))
    machine.solver._warm = None
    u, reason = machine.solver.solve()
    assert u is None and reason == "solve_failed"
    assert machine.solver._warm is None


def test_failed_warm_newton_falls_back_to_fresh_relax(machine, monkeypatch):
    relaxed = []
    polish_results = iter([(machine.wavy.clone(), 1.0),      # warm Newton: stalled
                           (machine.wavy.clone(), 1e-12)])   # post-relax Newton: fine

    def fake_relax(*args, **kwargs):
        relaxed.append(1)
        return machine.wavy.numpy()

    monkeypatch.setattr(fwd, "relax_to_pattern", fake_relax)
    monkeypatch.setattr(fwd, "newton_polish",
                        lambda F_fn, u0, *a, **k: next(polish_results))
    machine.solver._warm = machine.wavy.clone()
    u, reason = machine.solver.solve()
    assert reason == "ok" and len(relaxed) == 1


# ---------------------------------------------------------------- 4. near-null modes

def test_translation_modes_are_near_null(base):
    """F-D1-2: on the grid the translations are NEAR-null (~8e-4 at 64^2), which is both
    why they cannot be projected away (D-FFT-10) and why Newton needs the 2x2 subspace
    correction. Asserted an order looser than measured."""
    u = base.u.detach()
    F_fn = fwd.make_spatial_F(base.model, N_GRID, base.L)
    tx, ty = fwd.translation_modes(u, N_GRID, base.L)
    rng = np.random.default_rng(3)
    v = torch.from_numpy(rng.standard_normal(tuple(u.shape)))
    _, Av = torch.func.jvp(F_fn, (u,), (v / v.norm(),))
    ref = float(Av.detach().norm())
    for name, t in (("tx", tx), ("ty", ty)):
        _, At = torch.func.jvp(F_fn, (u,), (t / t.norm(),))
        ratio = float(At.detach().norm()) / ref
        assert ratio < 1e-2, f"{name}: ||A t||/||A v|| = {ratio:.2e} — not near-null"
