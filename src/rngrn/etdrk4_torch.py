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


def _torch_reaction_builder(model: RNGRN, device: torch.device,
                            differentiable: bool = False):
    """Batched reaction for (B, N, n, n) fields — competitive form only (as in D2).

    Same arithmetic as rollout._reaction_np_builder's competitive branch, in torch. By
    default the parameters are DETACHED: the forward relax behind the IFT/adjoint path is
    gradient-free by design, and its gradients come from the adjoint, never from backprop
    through the integrator. ``differentiable=True`` keeps them attached, which is what the
    §4.2 truncated-unrolled path (`rngrn.unrolled`) backpropagates through — the parameter
    tensors are then built ONCE here and shared by every step of the segment, so the
    theta -> KA/alpha/... subgraph is not rebuilt (or recomputed under checkpointing)
    per step. nc1 is refused loudly rather than half-ported: a silent wrong port would
    poison the equivalence check. A CUDA spectral run on an nc1 model therefore fails loud
    at relax time.
    """
    if model.form != "competitive":
        raise NotImplementedError(
            f"torch ETDRK4 reaction supports form='competitive' only, got "
            f"{model.form!r} (ported from diagnostic D2, which needed only the "
            f"fixture's form; port nc1 with its own equivalence check before using it)")
    keep = (lambda t: t) if differentiable else (lambda t: t.detach())
    KA = keep(model.KA).to(device)
    KR = keep(model.KR).to(device)
    alpha = keep(model.alpha).to(device)
    beta = keep(model.beta).to(device)
    delta = keep(model.delta).to(device)
    n_h = model.n_hill

    def reaction_t(X: torch.Tensor) -> torch.Tensor:           # (B, N, n, n)
        xn = torch.clamp(X, min=0.0) ** n_h
        denom = 1.0 + torch.einsum("ij,bjxy->bixy", KA + KR, xn)
        prod = torch.einsum("ij,bjxy->bixy", alpha * KA, xn) / denom
        return beta[None, :, None, None] + prod - delta[None, :, None, None] * X

    return reaction_t


def torch_half_coeffs(D, n: int, L: float, dt: float, device: torch.device):
    """ETDRK4 half-spectrum coefficients as torch tensors (mirrors _cached_half_coeffs).

    `D` may be a numpy array (the gradient-free path: the coefficients are then constants)
    or a torch tensor. A TENSOR is used as given — if it carries a graph, so do the six
    returned coefficients, which is the ONLY route by which theta_D reaches the field in the
    §4.2 unrolled path: D enters ETDRK4 through the linear operator -D k^2 and nowhere else.
    `dt` stays a float in both cases; the timestep is set from the detached |eig(J)|_max
    timescale and is not differentiated.
    """
    k2 = torch.from_numpy(_spectral_k2_half(n, L)).to(device)
    if torch.is_tensor(D):
        if D.dim() != 1:
            raise ValueError(
                f"D must be the (N,) diffusion vector of ONE model, got shape "
                f"{tuple(D.shape)} — a BatchedRNGRN's (B, N) D would be flattened into B*N "
                f"fake species here; use forward._half_coeffs_batched for the member axis")
        Lop = -D.to(device).reshape(-1)[:, None, None] * k2
    else:
        Lop = torch.stack([-float(D[i]) * k2 for i in range(len(D))])
    return _phi_contour_torch(Lop, dt)


def integrate_etdrk4_rfft_torch(X0: torch.Tensor, reaction_t, n: int, dt: float,
                                nsteps: int, coeffs,
                                check_blowup: bool = True) -> tuple[torch.Tensor, bool]:
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

    CROSS-BACKEND PARITY BREAK on a blow-up (D-PERF-6): `eval.numerics.integrate_etdrk4_rfft`
    (the numpy original) returns the field AT THE FIRST NON-FINITE STEP, while this function
    returns the field AFTER ALL `nsteps`. On a blow-up the two backends therefore return
    numerically different arrays, not just a different step index — this is a real deviation
    from the "same trajectory up to FFT-backend round-off" claim `relax_to_pattern_torch`
    otherwise makes (forward.py), not merely lost diagnostic granularity. It does not affect
    correctness on non-blowing-up trajectories, which is all `tests/test_etdrk4_torch.py`
    currently pins.

    `check_blowup=False` skips even that one check and returns ``blew=False`` unconditionally
    — for the §4.2 unrolled path, whose blocks are re-entered by gradient checkpointing (so
    the check would run twice per block) and which does its own single finiteness check on
    the segment's output. `blew` is then NOT a statement about the field; only a caller that
    checks finiteness itself may pass it.
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
    blew = (not bool(torch.isfinite(v).all())) if check_blowup else False
    return torch.fft.irfft2(v, s=(n, n), dim=(-2, -1)), blew
