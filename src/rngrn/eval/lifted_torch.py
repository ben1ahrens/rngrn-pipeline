"""lifted_torch.py — the GPU port of `eval/lifted.py`'s lifted spatial integrator.

RECOVERY-SIDE. Consumes the model's own parameters and caller-supplied grid geometry only;
it reads no observed frame and no ground-truth quantity, exactly like `eval/lifted.py` and
`etdrk4_torch.py`.

WHAT THIS IS. `lifted.simulate_lifted` integrates the FULL lifted PDE by Strang splitting —
gates dt/2 exact at frozen x, x over dt by ETDRK4 with the gates frozen, gates dt/2 exact.
It is numpy, and its per-step cost is dominated by two things that both batch well on a GPU:
the rfft2/irfft2 pair on an (N, n, n) field and the elementwise gate algebra on an
(N, N, n, n) array. At 512^2 with N=3 the gate array alone is 18.9 MB and is touched twice
per step, which is why the CPU path stops being affordable well before the grid does.

WHAT IS THE SAME, DELIBERATELY.
  * The step policy is `lifted.step_policy`, CALLED, not re-derived — same dt, same horizon,
    same step budget from the same inputs.
  * The x substep is `etdrk4_torch.integrate_etdrk4_rfft_torch` with coefficients from
    `etdrk4_torch.torch_half_coeffs`, i.e. the SAME torch ETDRK4 the M1 forward solve uses,
    already pinned bit-equivalent to `eval/numerics.integrate_etdrk4_rfft` by
    `tests/test_etdrk4_torch.py`. This module supplies its own reaction closure (production
    from the FROZEN gates) because `etdrk4_torch._torch_reaction_builder` builds the QSS
    reaction, which is precisely the thing the lift undoes.
  * The initial condition is drawn with **numpy**, from `np.random.default_rng(seed)`, and
    transferred. A torch draw would be statistically identical and bit-different, which
    would downgrade `tests/test_lifted_torch.py`'s equivalence check from exact to fuzzy for
    no gain.

WHAT IS NOT THE SAME. Nothing in the arithmetic. On CUDA the FFT backend is cuFFT rather
than pocketfft, so CUDA agreement is round-off-level rather than bit-level — the same
contract `etdrk4_torch` already carries.

BOTH REGULATION FORMS are ported. `gate_step_exact_torch` mirrors `lifted.gate_step_exact`
branch for branch, including the S -> 0 limit of (e^{-S tau} - 1)/S; getting that wrong
would show up as a damped pattern at small mu, which is the one artefact this whole scheme
exists to rule out.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..etdrk4_torch import integrate_etdrk4_rfft_torch, torch_half_coeffs
from .lifted import _np_params, gates_qss, step_policy


def gate_step_exact_torch(p, X, GA, GR, dt, mu, KA, KR):
    """Advance the gates EXACTLY over dt at frozen x — torch port of
    `lifted.gate_step_exact`. X: (N,n,n), GA/GR: (N,N,n,n), KA/KR: (N,N) on X's device."""
    xn = torch.clamp(X, min=0.0) ** p["n_hill"]                      # (N,n,n) over j
    ua = KA[:, :, None, None] * xn[None]                             # (i,j,n,n)
    ur = KR[:, :, None, None] * xn[None]
    if p["form"] != "competitive":
        ea = torch.exp(-(1.0 + ua) * (dt / mu))
        er = torch.exp(-(1.0 + ur) * (dt / mu))
        GAi = ua / (1.0 + ua)
        GRi = ur / (1.0 + ur)
        return GAi + (GA - GAi) * ea, GRi + (GR - GRi) * er
    S = ua.sum(1) + ur.sum(1)                                        # (i,n,n)
    tau = dt / mu
    e1 = float(np.exp(-tau))
    # (e^{-S tau} - 1)/S, taken to its limit -tau as S -> 0
    big = S > 1e-12
    c = torch.where(big, (torch.exp(-S * tau) - 1.0) / torch.where(big, S,
                                                                  torch.ones_like(S)),
                    torch.full_like(S, -tau))
    denom = 1.0 + S                                                  # (i,n,n)
    GAinf = ua / denom[:, None]
    GRinf = ur / denom[:, None]
    dA = GA - GAinf
    dR = GR - GRinf
    proj = dA.sum(1) + dR.sum(1)                                     # 1^T (w - w_inf)
    GAn = GAinf + e1 * (dA + c[:, None] * ua * proj[:, None])
    GRn = GRinf + e1 * (dR + c[:, None] * ur * proj[:, None])
    return GAn, GRn


def _reaction_from_gates_torch(p, GA, GR, alpha):
    if p["form"] == "competitive":
        return torch.einsum("ij,ijxy->ixy", alpha, GA)
    return torch.einsum("ij,ijxy->ixy", alpha, GA) * torch.prod(1.0 - GR, dim=1)


def simulate_lifted_torch(model, L, mu, n=64, T=None, dt=None, seed=0, noise=1e-2,
                          xstar=None, max_steps=6_000_000, record_every=0, device="cuda"):
    """`lifted.simulate_lifted` on a torch device. Same signature plus `device`, same return
    contract (numpy arrays out), same `stopped_reason` vocabulary.

    Fails loud rather than falling back: an unavailable device raises from torch, and no
    silent CPU substitution is attempted — a "GPU" number that quietly ran on the CPU is
    exactly the kind of claim CLAUDE.md §8 forbids.
    """
    dev = torch.device(device)
    p = _np_params(model)
    N = p["N"]
    D = p["D"]
    if xstar is None:
        from ..losses.terms import steady_state
        xs, _ = steady_state(model)
        xstar = xs.detach().cpu().numpy()
    xstar = np.asarray(xstar, float).reshape(N)

    dt, sig_max, horizon_rate, nsteps, hit_budget = step_policy(
        model, xstar, D, L, n, dt, T, max_steps)

    # numpy IC draw, then transferred — see the module docstring.
    rng = np.random.default_rng(seed)
    X0 = xstar[:, None, None] + noise * rng.standard_normal((N, n, n))
    GA0, GR0 = gates_qss(model, xstar)

    t = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64), device=dev)
    X = t(X0)
    GA = t(np.broadcast_to(GA0[:, :, None, None], (N, N, n, n)))
    GR = t(np.broadcast_to(GR0[:, :, None, None], (N, N, n, n)))
    KA, KR, alpha = t(p["KA"]), t(p["KR"]), t(p["alpha"])
    beta = t(p["beta"])[:, None, None]
    delta = t(p["delta"])[:, None, None]
    coeffs = torch_half_coeffs(D, n, L, dt, dev)

    gates = [GA, GR]           # closed over by reaction_t, rebound each step

    def reaction_t(Xr):
        return beta + _reaction_from_gates_torch(p, gates[0], gates[1], alpha) - delta * Xr

    frames = []
    t0 = time.perf_counter()
    blew_up = False
    done = 0
    with torch.no_grad():
        for step in range(nsteps):
            gates[0], gates[1] = gate_step_exact_torch(
                p, X, gates[0], gates[1], 0.5 * dt, mu, KA, KR)
            Xb, blew_up = integrate_etdrk4_rfft_torch(
                X[None], reaction_t, n, dt, 1, coeffs)
            X = Xb[0]
            if blew_up or not bool(torch.isfinite(X).all()):
                blew_up = True
                break
            gates[0], gates[1] = gate_step_exact_torch(
                p, X, gates[0], gates[1], 0.5 * dt, mu, KA, KR)
            done = step + 1
            if record_every and (step % record_every == 0):
                frames.append(X.cpu().numpy())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    Xn = X.cpu().numpy()
    out = dict(fields=Xn, GA=gates[0].cpu().numpy(), GR=gates[1].cpu().numpy(),
               mu=float(mu), dt=float(dt), nsteps=nsteps, nsteps_run=done,
               sig_max=sig_max, horizon_rate=float(horizon_rate),
               seconds=float(seconds), blew_up=bool(blew_up), L=float(L), n=n,
               frames=frames)
    if blew_up:
        out.update(kstar=np.nan, patterned=False, amplitude=np.nan,
                   stopped_reason="blew_up")
        return out
    from .. import observables as obs
    amp = float(Xn[0].std())
    pattern_floor = max(1e-3, 0.02 * abs(xstar[0]))
    out.update(kstar=(obs.kstar_of(Xn[0], L=L) if amp > 1e-6 else np.nan),
               patterned=bool(amp > pattern_floor), amplitude=amp,
               stopped_reason=("step_budget" if hit_budget else "horizon"))
    return out
