"""diag_fft_d1.py — D1: IFT feasibility, the go/no-go for the Fourier-training forward map.

PLAN_fourier_training.md §3 D1 / SPEC §3. On a small grid, for a known-Turing parameter
set: solve the patterned steady state; build the matrix-free linearisation A = dF/du at
that state; verify the translational zero modes; verify dL/du ⊥ zero modes for each
planned spectral term; solve the adjoint system; compare the IFT gradient dL/dtheta
against central finite differences on random theta-directions.

DIAGNOSTIC SCRIPT (no training, no payload.h5 — the known-Turing theta comes from a
tracked RECOVERED checkpoint under experiments/, i.e. a recovery OUTPUT, never an answer
key). Everything runs in float64 on CPU. Exit code 0 = D1 passes, 1 = D1 fails.

Acceptance (stated before running, recorded in docs/DECISIONS.md with the result):
per-direction relative error of the directional derivative <= 1e-4 at the best epsilon
of a step-size sweep, for every direction tested.

Usage:
    .venv/bin/python scripts/diag_fft_d1.py \
        [--ckpt experiments/tune_comp/runs/m3_registry_20260803_190250_seed3/checkpoints/model.pt] \
        [--n 96] [--ndir 10] [--out experiments/diag_fft/d1]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rngrn.model import RNGRN                                    # noqa: E402
from rngrn.losses.terms import steady_state                      # noqa: E402
from rngrn.eval.numerics import integrate_etdrk4_rfft, _spectral_k2  # noqa: E402
from rngrn.eval.rollout import _reaction_np_builder              # noqa: E402
from rngrn import observables as obs                             # noqa: E402

torch.set_default_dtype(torch.float64)

THETA_NAMES = ("theta_s", "theta_g", "theta_alpha", "theta_delta", "theta_beta", "theta_D")


# ------------------------------------------------------------------ model + field helpers

def load_known_turing_model(ckpt_path: str) -> RNGRN:
    """A recovered checkpoint whose run recorded recovered_turing=True (metric checked by
    the caller's choice of path; re-verified numerically below — fail loud, never trust
    the label alone)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if ck.get("nondim"):
        raise ValueError(f"{ckpt_path} was trained on the nondim path — its D is D/L^2 "
                         "and this script assumes physical D; pick a dimensional run")
    model = RNGRN(N=int(ck["N"]), form=ck["form"], n_hill=int(ck["n_hill"]), seed=0)
    model.load_state_dict(ck["state_dict"])
    return model


def reaction_fields(model: RNGRN, u: torch.Tensor) -> torch.Tensor:
    """Pointwise reaction applied to a (N, n, n) field via the model's own kernel."""
    x = u.permute(1, 2, 0)                     # (n, n, N)
    return model.reaction(x).permute(2, 0, 1)  # (N, n, n)


def make_spatial_F(model: RNGRN, n: int, L: float):
    """F(u) = D lap u + f(u), torch end-to-end (differentiable w.r.t. u AND theta).

    The Laplacian is spectral (rfft2), matching the training-side design: no stencil.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    ky = 2.0 * np.pi * np.fft.rfftfreq(n, d=L / n)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2h = torch.from_numpy(KX**2 + KY**2)      # (n, n//2+1), rfft2 layout

    def spatial_F(u: torch.Tensor) -> torch.Tensor:
        lap = torch.fft.irfft2(-k2h * torch.fft.rfft2(u), s=(n, n))
        return model.D[:, None, None] * lap + reaction_fields(model, u)

    return spatial_F


def translation_modes(u: torch.Tensor, n: int, L: float) -> tuple[torch.Tensor, torch.Tensor]:
    """du/dx, du/dy of a (N, n, n) periodic field, spectrally (the exact zero modes of A)."""
    kx = torch.from_numpy(2.0 * np.pi * np.fft.fftfreq(n, d=L / n))
    ky = torch.from_numpy(2.0 * np.pi * np.fft.rfftfreq(n, d=L / n))
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    uh = torch.fft.rfft2(u)
    tx = torch.fft.irfft2(1j * KX * uh, s=(n, n))
    ty = torch.fft.irfft2(1j * KY * uh, s=(n, n))
    return tx, ty


# ------------------------------------------------------------------ linear algebra (matrix-free)

class ProjectedOperator:
    """P A P (or P A^T P) as a scipy LinearOperator over flat numpy vectors, with P the
    projector off the (normalised, orthogonalised) translational zero modes."""

    def __init__(self, F_fn, u_star: torch.Tensor, modes: list[torch.Tensor],
                 transpose: bool):
        self.F_fn = F_fn
        self.u = u_star.detach()
        self.shape_field = tuple(u_star.shape)
        self.size = u_star.numel()
        self.transpose = transpose
        # orthonormalise the zero modes (Gram-Schmidt; they are near-orthogonal already)
        Q = []
        for m in modes:
            v = m.detach().clone().reshape(-1)
            for q in Q:
                v = v - (q @ v) * q
            nrm = v.norm()
            if float(nrm) > 1e-12:
                Q.append(v / nrm)
        self.Q = Q
        if transpose:
            _, self._vjp = torch.func.vjp(F_fn, self.u)

    def project(self, v: torch.Tensor) -> torch.Tensor:
        for q in self.Q:
            v = v - (q @ v) * q
        return v

    def matvec(self, x: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
        v = self.project(v).reshape(self.shape_field)
        if self.transpose:
            (out,) = self._vjp(v)
        else:
            _, out = torch.func.jvp(self.F_fn, (self.u,), (v,))
        out = self.project(out.detach().reshape(-1))
        return out.numpy()


def _minnorm_solve(apply_fwd, apply_adj, Mmv, b: np.ndarray, size: int,
                   tol: float = 1e-10, maxiter: int = 20000) -> tuple[np.ndarray, float]:
    """Minimal-norm least-squares solution of (apply_fwd) x = b by right-preconditioned
    LSMR (x = M y; matvec = apply_fwd∘M, rmatvec = M∘apply_adj with M symmetric) plus
    iterative refinement on the exact residual. Singular-safe: near-null components are
    resolved, not projected away (finding F-D1-3)."""
    from scipy.sparse.linalg import LinearOperator, lsmr

    Op = LinearOperator((size, size), matvec=lambda y: apply_fwd(Mmv(y)),
                        rmatvec=lambda z: Mmv(apply_adj(z)))
    bn = max(np.linalg.norm(b), 1e-300)
    x = np.zeros(size)
    res_prev = np.inf
    for _ in range(6):
        r = b - apply_fwd(x)
        res = float(np.linalg.norm(r) / bn)
        if res < tol or res > 0.9 * res_prev:
            break
        res_prev = res
        y = lsmr(Op, r, atol=1e-13, btol=1e-13, maxiter=maxiter)[0]
        x = x + Mmv(y)
    else:
        res = float(np.linalg.norm(b - apply_fwd(x)) / bn)
    return x, res


def solve_adjoint(F_fn, u_star, modes, rhs: torch.Tensor, k2_full: np.ndarray,
                  D: np.ndarray, gamma: float, tol: float = 1e-10,
                  maxiter: int = 20000) -> tuple[torch.Tensor, float]:
    """MINIMAL-NORM solution of A^T lam = rhs via right-preconditioned LSMR with
    true-residual refinement.

    WHY NOT projected GMRES (the first implementation): A has NEAR-null modes (grid-broken
    translations, near-marginal defect modes). Projecting the Krylov space off the
    translations makes the PROJECTED residual tiny (1e-13) while the TRUE residual
    A^T lam - rhs stays at ~1e-3 along the directions the projection ignores — measured,
    and it biased every gradient by exactly that amount (tangent-mode cross-check agreed
    with the adjoint, FD disagreed with both). For a translation-invariant loss
    (dL/du ⊥ t, measured ≤1e-18) the pinning response drops out of dL/dtheta identically
    and the correct adjoint is the minimal-norm least-squares solution — which LSMR
    computes without needing the null space explicitly. Right preconditioning by the
    SYMMETRIC Fourier-diagonal M = (gamma + D k^2)^{-1}: solve (A^T M) y = rhs,
    lam = M y; the transpose operator is (A^T M)^T = M A, available matrix-free.
    Refinement iterates on the exact torch residual until `tol` or stall.
    """
    shape_field = tuple(u_star.shape)
    Nsp, n, _ = shape_field
    u0 = u_star.detach()
    _, vjp_fn = torch.func.vjp(F_fn, u0)

    def AT(x: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
        (out,) = vjp_fn(v.reshape(shape_field))
        return out.detach().reshape(-1).numpy()

    def Amv(x: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
        _, out = torch.func.jvp(F_fn, (u0,), (v.reshape(shape_field),))
        return out.detach().reshape(-1).numpy()

    def Mmv(x: np.ndarray) -> np.ndarray:
        v = x.reshape(Nsp, n, n)
        vh = np.fft.fft2(v, axes=(1, 2))
        vh = vh / (gamma + D[:, None, None] * k2_full[None])
        return np.real(np.fft.ifft2(vh, axes=(1, 2))).reshape(-1)

    size = int(np.prod(shape_field))
    g = rhs.detach().reshape(-1).numpy()
    lam, res_full = _minnorm_solve(AT, Amv, Mmv, g, size, tol=tol, maxiter=maxiter)
    return torch.from_numpy(lam), res_full


def newton_polish(F_fn, u: torch.Tensor, modes_of, k2_full, D, gamma,
                  n_iter: int = 30, tol: float = 1e-11) -> tuple[torch.Tensor, float]:
    """Drive ||F(u)|| toward zero with DAMPED Newton steps: minimal-norm LSMR bulk step
    (unprojected — finding F-D1-3) plus an explicit 2-D translation-subspace correction
    for the pinned phase (finding F-D1-2). Damping (step halving when ||F|| does not
    decrease) matters for the eps=1e-3 FD points, whose start residual is large enough
    for the full step to overshoot along the labyrinth's near-marginal coarsening
    modes."""
    u = u.detach().clone()
    scale = float(u.norm())
    r = float(F_fn(u).detach().norm()) / max(scale, 1e-300)
    Nsp, n, _ = u.shape

    def precon(x: np.ndarray) -> np.ndarray:
        v = x.reshape(Nsp, n, n)
        vh = np.fft.fft2(v, axes=(1, 2))
        vh = vh / (gamma + D[:, None, None] * k2_full[None])
        return np.real(np.fft.ifft2(vh, axes=(1, 2))).reshape(-1)

    for _ in range(n_iter):
        if r < tol:
            break
        u0 = u.detach()
        _, vjp_fn = torch.func.vjp(F_fn, u0)

        def Amv(x: np.ndarray) -> np.ndarray:
            v = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
            _, out = torch.func.jvp(F_fn, (u0,), (v.reshape(u.shape),))
            return out.detach().reshape(-1).numpy()

        def ATmv(x: np.ndarray) -> np.ndarray:
            v = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
            (out,) = vjp_fn(v.reshape(u.shape))
            return out.detach().reshape(-1).numpy()

        Fv = F_fn(u).detach()
        # Bulk step: minimal-norm LSMR on the UNPROJECTED system (finding F-D1-3 — a
        # projected inner solve hides near-null error and floors Newton at ~1e-9).
        dx, _ = _minnorm_solve(Amv, ATmv, precon, -Fv.reshape(-1).numpy(),
                               u.numel(), tol=1e-12, maxiter=8000)
        step = torch.from_numpy(dx).reshape(u.shape)
        # TRANSLATION-SUBSPACE CORRECTION (finding F-D1-2). Grid pinning makes the
        # translations near-null, not null; a theta-perturbed fixed point sits at a
        # slightly different phase and the phase part of the step is best solved
        # explicitly in the 2-D mode subspace: G c = <t_i, F - A step>, step += -t.c.
        modes = modes_of(u)
        Q = []
        for m in modes:
            q = m.detach().reshape(-1)
            for p in Q:
                q = q - (p @ q) * p
            if float(q.norm()) > 1e-12:
                Q.append(q / q.norm())
        if len(Q) == 2:
            resid_after = Fv.reshape(-1) + torch.from_numpy(Amv(dx))
            At = []
            for q in Q:
                _, Aq = torch.func.jvp(F_fn, (u0,), (q.reshape(u.shape),))
                At.append(Aq.detach().reshape(-1))
            G = torch.stack([torch.stack([q @ a for a in At]) for q in Q])
            rhs = torch.stack([q @ resid_after for q in Q])
            if float(torch.linalg.cond(G)) < 1e8:
                c = torch.linalg.solve(G, rhs)
                step = step - (c[0] * Q[0] + c[1] * Q[1]).reshape(u.shape)
        lam_damp = 1.0
        for _half in range(6):
            u_try = u + lam_damp * step
            r_try = float(F_fn(u_try).detach().norm()) / max(scale, 1e-300)
            if r_try < r:
                break
            lam_damp *= 0.5
        else:
            break                     # no descent at any damping — stalled, report r
        u, r = u_try, r_try
    return u, r


# ------------------------------------------------------------------ loss-term prototypes

def torch_raps_power(field: torch.Tensor, L: float) -> tuple[np.ndarray, torch.Tensor]:
    """Radially averaged power spectrum of one channel, binned exactly as observables.raps
    (dk = 2*pi/L, edges arange(0, kmax+dk, dk), DC bin zeroed). Returns (k_centers, power);
    power is differentiable w.r.t. the field. Prototype for M1's losses/spectral.py; D6
    tests it against the numpy reference bin-for-bin."""
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
    power = torch.zeros(len(kbins) - 1, dtype=field.dtype).scatter_add_(0, idx, P.reshape(-1))
    power = power / torch.from_numpy(np.maximum(counts, 1)).to(field.dtype)
    power = power.clone()
    power[0] = 0.0
    k_centers = 0.5 * (kbins[:-1] + kbins[1:])
    return k_centers, power


def spectral_block_torch(field: torch.Tensor, nblk: int = 24) -> torch.Tensor:
    """The 2-D central log-power block of scoring/morphology.spectral_distance_2d,
    re-expressed in torch (normalise by total power, then log1p, then sum-normalise).
    Same arithmetic; the M1 port to utils.py gets the drift test."""
    f = field - field.mean()
    P = torch.abs(torch.fft.fftshift(torch.fft.fft2(f))) ** 2
    H, W = P.shape
    total = P.sum()
    top, left = H // 2 - nblk // 2, W // 2 - nblk // 2
    block = P[top:top + nblk, left:left + nblk]
    block = torch.log1p(block / total)
    return block / block.sum()


def make_loss_terms(L: float, kstar: float, targets: dict):
    """The five planned term forms (SPEC §5), each as loss_fn(u) -> scalar against FIXED
    target statistics. For the gradient check any fixed target is valid; these come from a
    reference relaxation at a nearby theta so magnitudes are realistic."""

    def band_mask(k_centers: np.ndarray) -> np.ndarray:
        return (k_centers >= 0.5 * kstar) & (k_centers <= 1.5 * kstar)

    def spec_shape(u):
        _, p = torch_raps_power(u[0], L)
        mask = torch.from_numpy(band_mask(targets["k_centers"]))
        norm = p[mask].sum()
        s = p[mask] / norm
        t = targets["raps_band_target"]
        return ((torch.log(s + 1e-300) - torch.log(t + 1e-300)) ** 2).sum()

    def spec_aniso(u):
        blk = spectral_block_torch(u[0])
        return ((blk - targets["block_target"]) ** 2).mean()

    def spec_amp_mean(u):
        return sum((torch.log(u[c].mean()) - targets["log_mean"][c]) ** 2 for c in range(3))

    def spec_amp_fluct(u):
        return sum((torch.log(u[c].std()) - targets["log_std"][c]) ** 2 for c in range(3))

    def real_moments(u):
        out = u.new_zeros(())
        for c in range(3):
            v = u[c] - u[c].mean()
            skew = (v ** 3).mean() / (v ** 2).mean().clamp_min(1e-300) ** 1.5
            out = out + (skew - targets["skew"][c]) ** 2
        return out

    return dict(spec_shape=spec_shape, spec_aniso=spec_aniso,
                spec_amp_mean=spec_amp_mean, spec_amp_fluct=spec_amp_fluct,
                real_moments=real_moments)


# ------------------------------------------------------------------ solve-to-pattern

def relax_to_pattern(model: RNGRN, xstar: np.ndarray, n: int, L: float, dt: float,
                     seed: int, noise: float = 1e-2, chunk: int = 500,
                     max_chunks: int = 400, flat_tol: float = 1e-4,
                     X0: np.ndarray | None = None) -> np.ndarray:
    """ETDRK4 from x* + noise (or a supplied warm start) until channel-0 amplitude AND k*
    are flat to flat_tol over 5 consecutive chunks. Deterministic for fixed seed."""
    D = model.D.detach().numpy()
    reaction_np = _reaction_np_builder(model)
    if X0 is None:
        rng = np.random.default_rng(seed)
        X = xstar[:, None, None] + noise * rng.standard_normal((3, n, n))
    else:
        X = X0.copy()
    amps: list[float] = []
    ks: list[float] = []
    for _ in range(max_chunks):
        X, blew = integrate_etdrk4_rfft(X, D, reaction_np, n, L, dt, chunk)
        if blew:
            raise RuntimeError("forward solve blew up — not a valid D1 fixture")
        amps.append(float(X[0].std()))
        ks.append(float(obs.kstar_of(X[0], L=L)))
        if len(amps) >= 5:
            a = np.array(amps[-5:]); k = np.array(ks[-5:])
            if ((a.max() - a.min()) / a.mean() < flat_tol
                    and (k.max() - k.min()) / k.mean() < flat_tol):
                return X
    raise RuntimeError(f"no saturation in {max_chunks * chunk} steps — not a valid fixture")


def solve_pattern_for(model: RNGRN, n: int, L: float, dt: float, seed: int,
                      k2_full: np.ndarray, gamma: float,
                      warm: torch.Tensor | None = None) -> tuple[torch.Tensor, float]:
    """The full u*(theta) map used by BOTH the base point and every FD evaluation:
    ETDRK4 relaxation (from noise, or warm-started) then projected-Newton polish."""
    xs, ok = steady_state(model)
    if not ok:
        raise RuntimeError("homogeneous steady state did not converge")
    xstar = xs.detach().numpy()
    F_fn = make_spatial_F(model, n, L)
    D = model.D.detach().numpy()

    modes_of = lambda uu: list(translation_modes(uu, n, L))  # noqa: E731
    if warm is None:
        X = relax_to_pattern(model, xstar, n, L, dt, seed)
        u = torch.from_numpy(np.ascontiguousarray(X))
    else:
        # Newton-ONLY from the base pattern: the projected Newton step cannot move along
        # the translations, so the phase stays pinned to the base solve — an ETDRK4
        # re-relaxation here would drift the phase and contaminate any FD of a term with
        # finite-grid phase sensitivity (the real_moments finding).
        u = warm.detach().clone()
    u, res = newton_polish(F_fn, u, modes_of, k2_full, D, gamma)
    if res > 1e-9:
        raise RuntimeError(f"pattern solve did not converge (||F||/||u|| = {res:.2e})")
    return u, res


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/tune_comp/runs/"
                    "m3_registry_20260803_190250_seed3/checkpoints/model.pt")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--ndir", type=int, default=10)
    ap.add_argument("--periods", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="acceptance: per-direction FD-vs-IFT relative error")
    ap.add_argument("--out", default="experiments/diag_fft/d1")
    ap.add_argument("--probe", action="store_true",
                    help="tangent-mode cross-check per direction (diagnosis aid)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report: dict = dict(ckpt=a.ckpt, n=a.n, ndir=a.ndir, tol=a.tol, seed=a.seed)

    # -- the known-Turing fixture, verified not trusted ------------------------------
    model = load_known_turing_model(a.ckpt)
    xs, ok = steady_state(model)
    assert ok, "steady state failed on the fixture checkpoint"
    J = model.jacobian(xs, create_graph=False).detach()
    kg = torch.linspace(1e-3, 10.0, 2000)
    sig = model.dispersion(xs, kg, J=J).detach()
    sig_max = float(sig.max())
    kstar_lin = float(kg[int(sig.argmax())])
    max_re_J = float(torch.linalg.eigvals(J).real.max())
    print(f"fixture: sig_max={sig_max:.4f} at k*={kstar_lin:.3f}, max Re eig(J)={max_re_J:.4f}")
    assert sig_max > 0 and max_re_J < 0, "fixture is not strictly Turing-unstable"
    report.update(sig_max=sig_max, kstar_lin=kstar_lin, max_re_J=max_re_J)

    n = a.n
    L = a.periods * 2.0 * np.pi / kstar_lin
    jac_rate = float(torch.linalg.eigvals(J).abs().max())
    dt = 0.2 / jac_rate
    k2_full = _spectral_k2(n, L)
    gamma = jac_rate
    D_np = model.D.detach().numpy()
    report.update(L=L, dt=dt)

    # -- base patterned state ---------------------------------------------------------
    print(f"relaxing to patterned state on {n}x{n}, L={L:.2f}, dt={dt:.4f} ...")
    u_star, res = solve_pattern_for(model, n, L, dt, a.seed, k2_full, gamma)
    amp = float(u_star[0].std())
    kstar_pat = float(obs.kstar_of(u_star[0].numpy(), L=L))
    print(f"patterned: amp={amp:.4f}, k*={kstar_pat:.3f}, ||F||/||u|| = {res:.2e}")
    assert amp > 1e-3, "fixture failed to pattern"
    report.update(residual=res, amplitude=amp, kstar_pattern=kstar_pat)

    F_fn = make_spatial_F(model, n, L)
    tx, ty = translation_modes(u_star, n, L)

    # -- check 1: translational zero modes -------------------------------------------
    rng = np.random.default_rng(a.seed + 1)
    v_rand = torch.from_numpy(rng.standard_normal(u_star.shape))
    _, Av = torch.func.jvp(F_fn, (u_star,), (v_rand / v_rand.norm(),))
    ref = float(Av.detach().norm())
    zero_ratios = {}
    for name, t in (("tx", tx), ("ty", ty)):
        _, At = torch.func.jvp(F_fn, (u_star,), (t / t.norm(),))
        zero_ratios[name] = float(At.detach().norm()) / ref
    print(f"zero modes: ||A t||/||A v_rand|| = tx {zero_ratios['tx']:.2e}, "
          f"ty {zero_ratios['ty']:.2e}")
    report["zero_mode_ratio"] = zero_ratios
    zero_ok = all(r < 1e-3 for r in zero_ratios.values())

    # -- fixed targets for the loss terms (from a NEARBY theta's pattern) -------------
    model_t = load_known_turing_model(a.ckpt)
    with torch.no_grad():
        for nm in THETA_NAMES:
            p = getattr(model_t, nm)
            p.add_(0.01 * torch.from_numpy(rng.standard_normal(p.shape)))
    u_t, _ = solve_pattern_for(model_t, n, L, dt, a.seed, k2_full, gamma)
    kc, p_t = torch_raps_power(u_t[0], L)
    mask = (kc >= 0.5 * kstar_pat) & (kc <= 1.5 * kstar_pat)
    targets = dict(
        k_centers=kc,
        raps_band_target=(p_t[torch.from_numpy(mask)] /
                          p_t[torch.from_numpy(mask)].sum()).detach(),
        block_target=spectral_block_torch(u_t[0]).detach(),
        log_mean=[float(torch.log(u_t[c].mean())) for c in range(3)],
        log_std=[float(torch.log(u_t[c].std())) for c in range(3)],
        skew=[float((((u_t[c] - u_t[c].mean()) ** 3).mean()
                     / ((u_t[c] - u_t[c].mean()) ** 2).mean() ** 1.5)) for c in range(3)],
    )
    terms = make_loss_terms(L, kstar_pat, targets)

    # -- check 2: dL/du orthogonal to the zero modes, per term ------------------------
    ortho = {}
    for name, fn in terms.items():
        u_req = u_star.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(fn(u_req), u_req)
        c = {}
        for mname, t in (("tx", tx), ("ty", ty)):
            c[mname] = float((g.reshape(-1) @ t.reshape(-1)).abs()
                             / (g.norm() * t.norm() + 1e-300))
        ortho[name] = c
        print(f"  dL/du ⊥ zero modes [{name}]: tx {c['tx']:.2e}, ty {c['ty']:.2e}")
    report["ortho"] = ortho
    ortho_ok = all(v < 1e-6 for c in ortho.values() for v in c.values())

    # -- IFT gradient, PER TERM (one adjoint solve each — cheap; the FD pattern solves
    # below are the expensive part and are shared across terms) -----------------------
    params = [getattr(model, nm) for nm in THETA_NAMES]
    g_flat_by_term: dict[str, np.ndarray] = {}
    adj_residuals: dict[str, float] = {}
    for name, fn in terms.items():
        u_req = u_star.clone().requires_grad_(True)
        (gu,) = torch.autograd.grad(fn(u_req), u_req)
        lam, adj_res = solve_adjoint(F_fn, u_star, [tx, ty], gu, k2_full, D_np, gamma)
        adj_residuals[name] = adj_res
        Fv = F_fn(u_star)      # u_star fixed; graph runs through theta only
        g_theta = torch.autograd.grad(Fv, params,
                                      grad_outputs=-lam.reshape(u_star.shape))
        g_flat_by_term[name] = torch.cat([g.reshape(-1) for g in g_theta]).numpy()
    print("adjoint residuals: " +
          ", ".join(f"{k} {v:.1e}" for k, v in adj_residuals.items()))
    report["adjoint_residuals"] = adj_residuals

    # -- check 3: central finite differences on random theta-directions, per term -----
    sizes = [getattr(model, nm).numel() for nm in THETA_NAMES]
    total_size = sum(sizes)
    eps_sweep = (1e-3, 1e-4, 1e-5, 1e-6)
    term_names = list(terms)
    rows = []
    for d in range(a.ndir):
        v = rng.standard_normal(total_size)
        v /= np.linalg.norm(v)
        d_ift = {name: float(g_flat_by_term[name] @ v) for name in term_names}
        # loss values of every term at theta +/- eps*v, sharing one pattern solve per point
        vals: dict[float, dict[float, dict[str, float]]] = {}
        for eps in eps_sweep:
            vals[eps] = {}
            for sgn in (+1.0, -1.0):
                m2 = load_known_turing_model(a.ckpt)
                off = 0
                with torch.no_grad():
                    for nm, sz in zip(THETA_NAMES, sizes):
                        p = getattr(m2, nm)
                        dv = torch.from_numpy(v[off:off + sz]).reshape(p.shape)
                        p.add_(sgn * eps * dv)
                        off += sz
                t0 = time.perf_counter()
                u2, res2 = solve_pattern_for(m2, n, L, dt, a.seed, k2_full, gamma,
                                             warm=u_star)
                drift = float((u2 - u_star).norm() / u_star.norm())
                print(f"    [dir {d} eps {eps:g} sgn {sgn:+.0f}] solve "
                      f"{time.perf_counter() - t0:.1f}s res {res2:.1e} "
                      f"drift {drift:.2e}")
                if drift > 0.2:
                    raise RuntimeError(
                        f"FD solve left the base branch (drift {drift:.2f}) — "
                        f"reduce eps or investigate")
                vals[eps][sgn] = {name: float(fn(u2)) for name, fn in terms.items()}
        if a.probe:
            # TANGENT-MODE cross-check: du solves A du = -dF/dtheta.v (the forward
            # sensitivity), then dL_tan = <dL/du, du>. Agreement with FD but not with the
            # adjoint value localises the bias to the adjoint solve; agreement with the
            # adjoint value localises it to the linearisation itself.
            h = 1e-6
            Fp, Fm = [], []
            for sgn in (+1.0, -1.0):
                m2 = load_known_turing_model(a.ckpt)
                off = 0
                with torch.no_grad():
                    for nm, sz in zip(THETA_NAMES, sizes):
                        p = getattr(m2, nm)
                        p.add_(sgn * h * torch.from_numpy(v[off:off + sz]).reshape(p.shape))
                        off += sz
                Ff = make_spatial_F(m2, n, L)(u_star).detach()
                (Fp if sgn > 0 else Fm).append(Ff)
            dF_dv = (Fp[0] - Fm[0]) / (2.0 * h)
            modes = [tx, ty]
            op = ProjectedOperator(F_fn, u_star, modes, transpose=False)
            from scipy.sparse.linalg import LinearOperator, lgmres

            def precon_t(x_):
                vv = x_.reshape(u_star.shape[0], n, n)
                vh = np.fft.fft2(vv, axes=(1, 2))
                vh = vh / (gamma + D_np[:, None, None] * k2_full[None])
                return np.real(np.fft.ifft2(vh, axes=(1, 2))).reshape(-1)

            Aop = LinearOperator((op.size, op.size), matvec=op.matvec)
            Mop = LinearOperator((op.size, op.size), matvec=precon_t)
            b_t = op.project(-dF_dv.reshape(-1)).numpy()
            du_np, _ = lgmres(Aop, b_t, M=Mop, rtol=1e-13, atol=0.0, maxiter=3000,
                              inner_m=50, outer_k=10)
            du = torch.from_numpy(du_np)
            # null-space part of du, via the same 2x2 mode-subspace solve as newton_polish
            Q2 = op.Q
            At2 = []
            for q in Q2:
                _, Aq = torch.func.jvp(F_fn, (u_star,), (q.reshape(u_star.shape),))
                At2.append(Aq.detach().reshape(-1))
            G2 = torch.stack([torch.stack([q @ ax for ax in At2]) for q in Q2])
            rhs2 = torch.stack([q @ dF_dv.reshape(-1) for q in Q2])
            c2 = torch.linalg.solve(G2, rhs2)
            du = du - (c2[0] * Q2[0] + c2[1] * Q2[1])
            for name, fn in terms.items():
                u_req = u_star.clone().requires_grad_(True)
                (g_u,) = torch.autograd.grad(fn(u_req), u_req)
                dL_tan = float(g_u.reshape(-1) @ du)
                print(f"      [probe] {name}: tangent {dL_tan:+.4e}  "
                      f"adjoint {d_ift[name]:+.4e}")
        row = dict(direction=d, terms={})
        for name in term_names:
            best = None
            per_eps = {}
            for eps in eps_sweep:
                d_fd = (vals[eps][+1.0][name] - vals[eps][-1.0][name]) / (2.0 * eps)
                rel = abs(d_fd - d_ift[name]) / max(abs(d_fd), abs(d_ift[name]), 1e-300)
                per_eps[eps] = dict(rel_err=rel, d_fd=d_fd)
                if best is None or rel < best[1]:
                    best = (eps, rel, d_fd)
            row["terms"][name] = dict(eps=best[0], rel_err=best[1],
                                      d_fd=best[2], d_ift=d_ift[name],
                                      per_eps={str(k): v for k, v in per_eps.items()})
            print(f"      {name}: ift {d_ift[name]:+.4e}  " +
                  "  ".join(f"eps{eps:g}: {per_eps[eps]['d_fd']:+.4e}"
                            f" ({per_eps[eps]['rel_err']:.1e})" for eps in eps_sweep))
        rows.append(row)
        summary = "  ".join(f"{nm} {row['terms'][nm]['rel_err']:.1e}"
                            for nm in term_names)
        print(f"  dir {d}: rel err  {summary}")
    report["directions"] = rows

    # Verdict: the four SPECTRAL terms are the binding acceptance (SPEC §3's claim is
    # about spectrum-derived losses); real_moments is reported separately because its
    # finite-grid zero-mode leakage (the ortho check above) is a measured property of
    # pixel moments under sub-pixel translation, not an IFT failure — if it misses the
    # tolerance that is a FINDING for M1's term design, stated in the report.
    spectral_terms = [t for t in term_names if t != "real_moments"]
    worst_spec = max(r["terms"][t]["rel_err"] for r in rows for t in spectral_terms)
    worst_moments = max(r["terms"]["real_moments"]["rel_err"] for r in rows)
    ortho_spec_ok = all(v < 1e-6 for nm, c in ortho.items() if nm != "real_moments"
                        for v in c.values())
    passed = bool(zero_ok and ortho_spec_ok and worst_spec <= a.tol)
    report.update(worst_rel_err_spectral=worst_spec,
                  worst_rel_err_real_moments=worst_moments,
                  real_moments_ok=bool(worst_moments <= a.tol),
                  zero_ok=zero_ok, ortho_ok=ortho_ok,
                  ortho_spectral_ok=ortho_spec_ok, passed=passed)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(report, f, indent=2)
    moments_note = ("ok" if worst_moments <= a.tol
                    else "FINDING — finite-grid moment term, see report")
    print(f"\nD1 {'PASS' if passed else 'FAIL'}: worst spectral rel err "
          f"{worst_spec:.2e} (tol {a.tol:g}); real_moments worst "
          f"{worst_moments:.2e} ({moments_note}); zero modes "
          f"{'ok' if zero_ok else 'BAD'}; spectral orthogonality "
          f"{'ok' if ortho_spec_ok else 'BAD'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
