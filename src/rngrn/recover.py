"""recover.py — per-frame variational recovery (RECOVERY SIDE).

Optimise the RNGRN raw parameters theta for ONE frame by gradient descent on the
composite objective: Adam for global exploration, optional LBFGS polish. RANDOM
init only, multiple restarts, keep the best by total loss. For m<N, latent fields
for the unobserved channels are additional optimisation variables.

FIREWALL: the ONLY input is a data.gate.RecoveryInput = (frame, L, observed_idx, N).
No analytic x*, J, sigma, coefficient, or truth-derived init. k*_obs comes from the
frame's own FFT. This module MUST NOT import rd_models / solver / cache / AnswerKey.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import RNGRN
from . import observables as obs
from .losses import total as LT
from .losses.total import SteadyStateError
from .losses.weighting import build_strategy, FixedWeighting


class FreeScaleLatent(torch.nn.Module):
    """Free-scale latent parameterisation for m<N unobserved channels, adopted from
    Matas-Gil & Endres (arXiv:2309.06339 / iScience 2024, CDIMA experimental case): each
    unobserved channel is a TRAINABLE affine map of the observed frame renormalised to
    [0,1], u_c = W*kappa_c + gamma_c, with kappa/gamma optimised jointly with the model.
    W is derived from the OBSERVED frame only (firewall-legal)."""
    def __init__(self, W, n_channels, dtype):
        super().__init__()
        self.register_buffer("W", W)
        self.kappa = torch.nn.Parameter(torch.ones(n_channels, 1, 1, dtype=dtype))
        self.gamma = torch.nn.Parameter(torch.zeros(n_channels, 1, 1, dtype=dtype))

    def forward(self):
        return self.W * self.kappa + self.gamma


@dataclass
class RecoveryResult:
    model: object
    params: dict
    topology: dict
    xstar: np.ndarray
    kstar_model: float
    kstar_obs: float
    loss: float
    parts: dict
    restarts: list = field(default_factory=list)
    latent_fields: object = None


def _kgrid_for(kstar_obs, n=400, span=8.0, device=None, dtype=torch.float64):
    kmax = max(2.0, span * kstar_obs)
    return torch.linspace(kstar_obs / 50.0 + 1e-3, kmax, n, device=device, dtype=dtype)


def _topology(model):
    KA = model.KA.detach().cpu().numpy(); KR = model.KR.detach().cpu().numpy()
    s = model.s.detach().cpu().numpy(); g = model.gate.detach().cpu().numpy()
    sign = np.where(g > 0.5, 1, -1)
    sign = np.where(s > 0.05 * s.max(), sign, 0)
    return dict(sign=sign, magnitude=s, gate=g, KA=KA, KR=KR)


def recover(recovery_input, form="competitive", strategy=None, weights=None,
            tau=0.12, jac_floor=1.0, n_restarts=4, adam_steps=1500, adam_lr=0.05,
            lbfgs_steps=50, grad_clip=10.0, seed=0, verbose=False, device=None):
    """Recover a GRN from one RecoveryInput. Returns the best RecoveryResult.

    strategy: a WeightingStrategy instance (default FixedWeighting(weights or defaults)).
    """
    ri = recovery_input
    dev = torch.device(device) if device is not None else torch.device("cpu")
    frame = torch.tensor(np.asarray(ri.frame, dtype=float), device=dev)
    L, N, observed_idx = ri.L, ri.N, list(ri.observed_idx)
    m = frame.shape[0]
    if strategy is None:
        base = weights or dict(kstar=1.0, turing=1.0, resid=0.3, anticollapse=0.5, morphology=0.0)
        strategy = FixedWeighting(base)

    kstar_obs = obs.kstar_of(frame[0].detach().cpu().numpy(), L=L)   # firewall: FFT of the observed image
    kgrid = _kgrid_for(kstar_obs, device=dev)

    best = None; restart_log = []
    for r in range(n_restarts):
        model = RNGRN(N=N, form=form, seed=seed + r).to(dev)
        latent_module = None
        if m < N:
            obs_mean = frame.mean(0)                     # (H, W), observed frame only
            lo, hi = obs_mean.min(), obs_mean.max()
            W = (obs_mean - lo) / (hi - lo + 1e-12)       # renormalised to [0,1]
            latent_module = FreeScaleLatent(W.unsqueeze(0), N - m, frame.dtype).to(dev)
        params = list(model.parameters()) + (list(latent_module.parameters()) if latent_module is not None else [])

        opt = torch.optim.Adam(params, lr=adam_lr)
        failed = False
        for step in range(adam_steps):
            opt.zero_grad()
            latent = latent_module() if latent_module is not None else None
            try:
                loss, parts = LT.total_loss(model, frame, L, observed_idx, kgrid, kstar_obs,
                                            strategy, step=step, latent_fields=latent,
                                            tau=tau, jac_floor=jac_floor, strict=True)
            except SteadyStateError:
                # fail-loud honoured: this init cannot form a valid steady state — abandon
                # the restart rather than optimise against a meaningless x*.
                if verbose:
                    print(f"  restart {r} step {step}: steady state diverged; skipping restart")
                failed = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            if verbose and step % 300 == 0:
                print(f"  restart {r} step {step}: total={parts['total']:.3f} "
                      f"kstar_m={parts.get('kstar_model', float('nan')):.2f} "
                      f"sig_max={parts.get('sig_max', float('nan')):.3f}")
        if failed:
            restart_log.append(dict(restart=r, total=float("inf"), steady_state_failed=True))
            continue

        if lbfgs_steps:
            lopt = torch.optim.LBFGS(params, max_iter=lbfgs_steps, line_search_fn="strong_wolfe")
            def closure():
                lopt.zero_grad()
                latent = latent_module() if latent_module is not None else None
                loss, _ = LT.total_loss(model, frame, L, observed_idx, kgrid, kstar_obs,
                                        strategy, step=adam_steps, latent_fields=latent,
                                        tau=tau, jac_floor=jac_floor)
                loss.backward(); return loss
            try:
                lopt.step(closure)
            except Exception:
                pass

        with torch.no_grad():
            latent = latent_module() if latent_module is not None else None
            loss, parts = LT.total_loss(model, frame, L, observed_idx, kgrid, kstar_obs,
                                        strategy, step=adam_steps, latent_fields=latent,
                                        tau=tau, jac_floor=jac_floor)
        restart_log.append(dict(restart=r, total=float(loss), sig_max=parts.get("sig_max"),
                                kstar_model=parts.get("kstar_model"), rel_err=parts.get("rel_err")))
        if best is None or float(loss) < best[0]:
            from .losses.terms import steady_state
            xs, _ = steady_state(model)
            best = (float(loss), model, parts, xs.detach().cpu().numpy(),
                    latent.detach().cpu().numpy() if latent is not None else None)

    if best is None:
        # every random init failed to form a valid steady state. Fail loud to the caller —
        # this is a signal to widen the init distribution or revisit the model, not to
        # silently return a garbage recovery.
        raise SteadyStateError(
            f"all {n_restarts} restarts failed to converge to a valid steady state; "
            "no recovery produced. Widen model init scales or check the frame/observed_idx.")
    loss, model, parts, xstar, latent_np = best
    P = dict(KA=model.KA.detach().cpu().numpy(), KR=model.KR.detach().cpu().numpy(),
             alpha=model.alpha.detach().cpu().numpy(), delta=model.delta.detach().cpu().numpy(),
             beta=model.beta.detach().cpu().numpy(), D=model.D.detach().cpu().numpy())
    return RecoveryResult(model=model, params=P, topology=_topology(model), xstar=xstar,
                          kstar_model=parts.get("kstar_model", float("nan")), kstar_obs=kstar_obs,
                          loss=loss, parts=parts, restarts=restart_log, latent_fields=latent_np)
