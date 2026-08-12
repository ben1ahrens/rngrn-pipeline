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


# ---------------------------------------------------------------- torch LSMR parity

def test_torch_lsmr_matches_scipy_on_a_well_conditioned_system():
    """The CUDA path's `_lsmr_torch` is a port of scipy's LSMR recurrences; on a
    well-conditioned overdetermined system (unique LS solution) the two must agree to
    machine precision. Near-singular agreement is pinned by the wrapper test below."""
    from scipy.sparse.linalg import lsmr
    rng = np.random.default_rng(0)
    A = rng.standard_normal((60, 40))
    b = rng.standard_normal(60)
    x_sp = lsmr(A, b, atol=1e-13, btol=1e-13, maxiter=5000)[0]
    At, bt = torch.from_numpy(A), torch.from_numpy(b)
    x_t = fwd._lsmr_torch(lambda v: At @ v, lambda v: At.T @ v, bt, maxiter=5000)
    assert float(np.max(np.abs(x_t.numpy() - x_sp))) < 1e-8


def test_minnorm_solve_torch_matches_scipy_on_a_near_singular_operator():
    """`_minnorm_solve_t` must mirror `_minnorm_solve` (D-FFT-10 semantics) including
    the refinement/stall behaviour on a NEAR-SINGULAR operator — the regime the real
    linearisation lives in (F-D1-2's near-null translations)."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((60, 40))
    S = A.T @ A
    S[0] *= 1e-8
    S[:, 0] *= 1e-8                      # a near-null direction, like a pinned phase
    b = rng.standard_normal(40)
    St = torch.from_numpy(S)
    x1, r1 = fwd._minnorm_solve(lambda v: S @ v, lambda v: S.T @ v, lambda v: v,
                                b, 40, tol=1e-10, maxiter=5000)
    x2, r2 = fwd._minnorm_solve_t(lambda v: St @ v, lambda v: St.T @ v, lambda v: v,
                                  torch.from_numpy(b), tol=1e-10, maxiter=5000)
    assert abs(r1 - r2) < 1e-12, f"stall residuals differ: scipy {r1} vs torch {r2}"
    assert float(np.max(np.abs(x2.numpy() - x1))) < 1e-8


def test_adjoint_solve_torch_path_matches_scipy_path(base):
    """The full `solve_adjoint` through `_minnorm_solve_t` (forced by moving nothing —
    exercised on CPU tensors via the torch primitives directly) must agree with the
    scipy path on a real linearisation. Uses the module's base pattern at 64^2 with a
    random translation-orthogonal rhs, the D-FFT-10 operating regime."""
    u0 = base.u.detach()
    F_fn = fwd.make_spatial_F(base.model, N_GRID, base.L)
    k2_full = fwd._spectral_k2(N_GRID, base.L)
    D_np = base.model.D.detach().numpy()
    J = base.model.jacobian(steady_state(base.model)[0],
                            create_graph=False).detach()
    gamma = float(torch.linalg.eigvals(J).abs().max())
    rng = np.random.default_rng(5)
    rhs = torch.from_numpy(rng.standard_normal(tuple(u0.shape)))
    # scipy path (device dispatch picks it for CPU tensors)
    lam_sp, res_sp = fwd.solve_adjoint(F_fn, u0, rhs, k2_full, D_np, gamma)
    # torch path, forced explicitly through the same closures the CUDA branch builds
    _, vjp_fn = torch.func.vjp(F_fn, u0)
    Nsp, n, _ = u0.shape
    D_t = torch.from_numpy(D_np)
    k2_t = torch.from_numpy(k2_full)

    def AT_t(x):
        (out,) = vjp_fn(x.reshape(u0.shape))
        return out.detach().reshape(-1)

    def Amv_t(x):
        _, out = torch.func.jvp(F_fn, (u0,), (x.reshape(u0.shape),))
        return out.detach().reshape(-1)

    def Mmv_t(x):
        vh = torch.fft.fft2(x.reshape(Nsp, n, n), dim=(-2, -1))
        vh = vh / (gamma + D_t[:, None, None] * k2_t[None])
        return torch.real(torch.fft.ifft2(vh, dim=(-2, -1))).reshape(-1)

    lam_t, res_t = fwd._minnorm_solve_t(AT_t, Amv_t, Mmv_t, rhs.reshape(-1))
    assert res_sp <= 1e-10 and res_t <= 1e-10, (res_sp, res_t)
    rel = float((lam_t - lam_sp.reshape(-1)).norm() / lam_sp.norm())
    assert rel < 1e-6, f"torch vs scipy adjoint solutions differ by {rel:.2e}"


# ---------------------------------------------------------------- warm_mode="relax"

def test_warm_relax_mode_rerelaxes_from_the_warm_state(machine, monkeypatch):
    """In warm_mode='relax' a present warm state goes through a SHORT re-relax (X0 = the
    warm state, budget warm_max_chunks) and then Newton — never Newton-only, never a
    fresh from-noise relax."""
    solver = PatternSolver(machine.solver.model, machine.solver.n, machine.solver.L, 0,
                           warm_mode="relax")
    calls = []

    def fake_relax(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("X0") is not None, "warm relax must start from the warm state"
        assert kwargs.get("max_chunks") == solver.warm_max_chunks
        return machine.wavy.numpy()

    monkeypatch.setattr(fwd, "relax_to_pattern", fake_relax)
    monkeypatch.setattr(fwd, "newton_polish", lambda F_fn, u0, *a, **k: (u0, 1e-12))
    solver._warm = machine.wavy.clone()
    u, reason = solver.solve()
    assert reason == "ok" and len(calls) == 1


def test_warm_relax_budget_exhaustion_falls_back_to_fresh(machine, monkeypatch):
    """A warm re-relax that cannot saturate within warm_max_chunks (RuntimeError) is not
    a failure: the solver falls back to exactly one FRESH relax, as the approved state
    machine requires."""
    solver = PatternSolver(machine.solver.model, machine.solver.n, machine.solver.L, 0,
                           warm_mode="relax")
    fresh_calls = []

    def fake_relax(*args, **kwargs):
        if kwargs.get("X0") is not None:
            raise RuntimeError("no saturation in the warm budget")
        fresh_calls.append(1)
        return machine.wavy.numpy()

    monkeypatch.setattr(fwd, "relax_to_pattern", fake_relax)
    monkeypatch.setattr(fwd, "newton_polish", lambda F_fn, u0, *a, **k: (u0, 1e-12))
    solver._warm = machine.wavy.clone()
    u, reason = solver.solve()
    assert reason == "ok" and len(fresh_calls) == 1


def test_warm_mode_is_validated_at_construction(machine):
    with pytest.raises(ValueError, match="warm_mode"):
        PatternSolver(machine.solver.model, machine.solver.n, machine.solver.L, 0,
                      warm_mode="hot")


# ---------------------------------------------------------------- CUDA path

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


@needs_cuda
def test_cuda_fresh_solve_matches_cpu_on_translation_invariant_stats(base):
    """The CUDA solve (torch relax + torch LSMR Newton) must land on the same pattern
    branch as the CPU solve, compared translation-invariantly: channel-0 amplitude
    within 1%, k* within one radial bin (the repo's k* tolerance)."""
    model_c = _load_ckpt().to("cuda")
    solver = PatternSolver(model_c, N_GRID, base.L, SEED)
    t0 = time.perf_counter()
    u_c, reason = solver.solve()
    dt_s = time.perf_counter() - t0
    assert reason == "ok", f"CUDA solve did not pattern: {reason}"
    amp_cpu = float(base.u.detach()[0].std())
    amp_gpu = float(u_c.detach()[0].std())
    k_gpu = float(obs.kstar_of(u_c.detach()[0].cpu().numpy(), L=base.L))
    one_bin = 2.0 * np.pi / base.L
    print(f"\n[timing] CUDA fresh solve at {N_GRID}^2: {dt_s:.1f}s "
          f"(CPU base was {base.base_seconds:.1f}s)")
    assert abs(amp_gpu - amp_cpu) / amp_cpu < 0.01, (amp_cpu, amp_gpu)
    assert abs(k_gpu - base.kstar) <= one_bin, (base.kstar, k_gpu, one_bin)


@needs_cuda
def test_ift_gradient_matches_finite_differences_on_cuda():
    """The FD regression tripwire (5e-3, NOT a re-acceptance — see module docstring) on
    the CUDA path: torch relax, torch-LSMR Newton and adjoint. One direction, two terms,
    warm Newton-only FD points, exactly like the CPU test."""
    model = _load_ckpt().to("cuda")
    xs, ok = steady_state(model)
    assert ok
    J = model.jacobian(xs, create_graph=False).detach()
    kg = torch.linspace(1e-3, 10.0, 2000, device="cuda")
    sig = model.dispersion(xs, kg, J=J).detach()
    kstar_lin = float(kg[int(sig.argmax())])
    L = PERIODS * 2.0 * np.pi / kstar_lin

    solver = PatternSolver(model, N_GRID, L, SEED)
    u, reason = solver.solve()
    assert reason == "ok", reason
    kstar_pat = float(obs.kstar_of(u.detach()[0].cpu().numpy(), L=L))
    spec_shape, amp_fluct = _make_terms(u.detach().cpu(), L, kstar_pat)

    params = [getattr(model, nm) for nm in THETA_NAMES]
    loss = spec_shape(u.cpu()) + amp_fluct(u.cpu())
    g = torch.autograd.grad(loss, params)
    g_flat = torch.cat([gi.reshape(-1) for gi in g]).cpu().numpy()
    v = _flat_direction(model, SEED + 1)
    d_ift = float(g_flat @ v)

    eps = 1e-4
    vals = {}
    for sgn in (+1.0, -1.0):
        m2 = _load_ckpt().to("cuda")
        off = 0
        with torch.no_grad():
            for nm in THETA_NAMES:
                p = getattr(m2, nm)
                dv = torch.from_numpy(v[off:off + p.numel()]).reshape(p.shape).to("cuda")
                p.add_(sgn * eps * dv)
                off += p.numel()
        s2 = PatternSolver(m2, N_GRID, L, SEED)
        s2._warm = u.detach().clone()
        u2, reason = s2.solve()
        assert reason == "ok", f"CUDA FD solve failed at sgn={sgn}: {reason}"
        vals[sgn] = float(spec_shape(u2.detach().cpu()) + amp_fluct(u2.detach().cpu()))
    d_fd = (vals[+1.0] - vals[-1.0]) / (2.0 * eps)
    rel = abs(d_fd - d_ift) / max(abs(d_fd), abs(d_ift), 1e-300)
    print(f"\n[cuda-fd] rel err {rel:.2e}")
    assert rel <= 5e-3, f"CUDA IFT-vs-FD relative error {rel:.2e} exceeds the tripwire"


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
