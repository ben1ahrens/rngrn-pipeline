"""etdrk4_torch.py — the batched torch ETDRK4 integrator, ported from diagnostic D2.

RECOVERY-SIDE. Ported verbatim from ``scripts/diag_fft_d2.py`` (functions
``_phi_contour_torch``, ``_torch_reaction_builder``, ``torch_half_coeffs``,
``integrate_etdrk4_rfft_torch``), whose CPU output was measured bit-equivalent to the
numpy reference ``eval/numerics.integrate_etdrk4_rfft`` at max |delta| 1.1e-13 over 100
steps (D2 ``torch_equivalence``, `experiments/diag_fft/d2/results_full.json`), and whose
CUDA throughput was measured at 3.25 ms/step at 512^2 fp64 on this machine
(`experiments/diag_fft/gpu_probe/results.json`). The port is the GPU leg of the M1
forward solve (`rngrn.forward.PatternSolver(device="cuda")`).

Same scheme and same contour integral as the numpy original. The one departure is the
blow-up check: the numpy original tests `isfinite` every step, which on CUDA is a blocking
sync per step; here it runs ONCE per call, after the step loop (see
``integrate_etdrk4_rfft_torch`` for why that is equivalent for the flag, and what it
gives up). The 3.25 ms/step figure above was measured WITH the per-step check, so it is
an upper bound on the current path, not a description of it.
``tests/test_etdrk4_torch.py`` re-pins the numpy equivalence in the suite.

Nothing here reads the observed frame or any answer-key quantity: inputs are the model's
own parameters and caller-supplied grid geometry.
"""
from __future__ import annotations

import numpy as np
import torch

from .eval.numerics import _spectral_k2_half
from .model import RNGRN


def _phi_contour_torch(Lop: torch.Tensor, dt: float, M: int = 32):
    """Torch port of numerics._phi_contour: same contour integral, complex128.

    Lop is the real (N, n, n//2+1) half-spectrum linear operator; returns the six
    real-valued ETDRK4 coefficient tensors on Lop's device.
    """
    E = torch.exp(dt * Lop)
    E2 = torch.exp(dt * Lop / 2.0)
    theta = torch.pi * (torch.arange(1, M + 1, dtype=torch.float64,
                                     device=Lop.device) - 0.5) / M
    r = torch.exp(1j * theta)                                  # complex128 roots circle
    LR = (dt * Lop)[..., None].to(torch.complex128) + r
    Q = dt * torch.real(torch.mean((torch.exp(LR / 2.0) - 1.0) / LR, dim=-1))
    f1 = dt * torch.real(torch.mean(
        (-4.0 - LR + torch.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, dim=-1))
    f2 = dt * torch.real(torch.mean(
        (2.0 + LR + torch.exp(LR) * (-2.0 + LR)) / LR**3, dim=-1))
    f3 = dt * torch.real(torch.mean(
        (-4.0 - 3.0 * LR - LR**2 + torch.exp(LR) * (4.0 - LR)) / LR**3, dim=-1))
    return E, E2, Q, f1, f2, f3


def _torch_reaction_builder(model: RNGRN, device: torch.device):
    """Batched reaction for (B, N, n, n) fields — competitive form only (as in D2).

    Same arithmetic as rollout._reaction_np_builder's competitive branch, in torch,
    parameters DETACHED (the relax is gradient-free by design; gradients come from the
    IFT adjoint, never from backprop through the integrator). nc1 is refused loudly
    rather than half-ported: a silent wrong port would poison the equivalence check.
    A CUDA spectral run on an nc1 model therefore fails loud at relax time.
    """
    if model.form != "competitive":
        raise NotImplementedError(
            f"torch ETDRK4 reaction supports form='competitive' only, got "
            f"{model.form!r} (ported from diagnostic D2, which needed only the "
            f"fixture's form; port nc1 with its own equivalence check before using it)")
    KA = model.KA.detach().to(device)
    KR = model.KR.detach().to(device)
    alpha = model.alpha.detach().to(device)
    beta = model.beta.detach().to(device)
    delta = model.delta.detach().to(device)
    n_h = model.n_hill

    def reaction_t(X: torch.Tensor) -> torch.Tensor:           # (B, N, n, n)
        xn = torch.clamp(X, min=0.0) ** n_h
        denom = 1.0 + torch.einsum("ij,bjxy->bixy", KA + KR, xn)
        prod = torch.einsum("ij,bjxy->bixy", alpha * KA, xn) / denom
        return beta[None, :, None, None] + prod - delta[None, :, None, None] * X

    return reaction_t


def torch_half_coeffs(D: np.ndarray, n: int, L: float, dt: float,
                      device: torch.device):
    """ETDRK4 half-spectrum coefficients as torch tensors (mirrors _cached_half_coeffs)."""
    k2 = torch.from_numpy(_spectral_k2_half(n, L)).to(device)
    Lop = torch.stack([-float(D[i]) * k2 for i in range(len(D))])
    return _phi_contour_torch(Lop, dt)


def integrate_etdrk4_rfft_torch(X0: torch.Tensor, reaction_t, n: int, dt: float,
                                nsteps: int, coeffs) -> tuple[torch.Tensor, bool]:
    """Line-for-line torch port of numerics.integrate_etdrk4_rfft, batched over a
    leading IC dimension: X0 is (B, N, n, n), coefficients broadcast over B.

    Returns (X, blew_up). The blow-up check is ONE `isfinite` per CALL, after the step
    loop — NOT the numpy original's per-step check, which costs a blocking device sync
    per step on CUDA at a cost that does not shrink as the step's own FLOPs (n^2 log n)
    do, so it dominates at the small training geometries. Every operation in the step is
    linear or an FFT over the whole field, so a non-finite value cannot be erased once
    it appears: end-of-call detection is equivalent FOR THE BOOLEAN, which is all any
    caller uses. What is lost is WHICH step blew up — the returned field is the state
    after `nsteps`, not the state at the first non-finite step. Callers chunk their
    calls (the forward relax runs 500 steps at a time), so detection granularity is one
    chunk, unchanged.
    """
    E, E2, Q, f1, f2, f3 = coeffs

    def Nfun(vhat):
        X = torch.fft.irfft2(vhat, s=(n, n), dim=(-2, -1))
        return torch.fft.rfft2(reaction_t(X), dim=(-2, -1))

    v = torch.fft.rfft2(X0, dim=(-2, -1))
    for _ in range(nsteps):
        Nv = Nfun(v)
        a = E2 * v + Q * Nv
        Na = Nfun(a)
        b = E2 * v + Q * Na
        Nb = Nfun(b)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = Nfun(c)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
    blew = not bool(torch.isfinite(v).all())
    return torch.fft.irfft2(v, s=(n, n), dim=(-2, -1)), blew
