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
from .losses.weighting import build_strategy, FixedWeighting, DataFirstStaging


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
            lbfgs_steps=50, grad_clip=10.0, seed=0, verbose=False, device=None,
            split_hinges=True, hinge_k_min_frac=0.1, staging_keys=("turing",),
            staging_off_frac=0.25, staging_ramp_frac=0.25, detach_xstar=False):
    """Recover a GRN from one RecoveryInput. Returns the best RecoveryResult.

    strategy: a WeightingStrategy instance (default FixedWeighting(weights or defaults)).

    split_hinges / hinge_k_min_frac / detach_xstar: passed through to losses.total (see
    compute_terms). staging_keys/off_frac/ramp_frac: data-first staging — the named terms
    are held at weight 0 for the first `off_frac` of `adam_steps`, then ramped to their
    configured weight over the next `ramp_frac`. Pass staging_keys=() to disable.
    """
    ri = recovery_input
    dev = torch.device(device) if device is not None else torch.device("cpu")
    frame = torch.tensor(np.asarray(ri.frame, dtype=float), device=dev)
    L, N, observed_idx = ri.L, ri.N, list(ri.observed_idx)
    m = frame.shape[0]
    if strategy is None:
        base = weights or dict(kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5,
                               anchor=2.0, morphology=0.0)
        strategy = FixedWeighting(base)
    if staging_keys:
        strategy = DataFirstStaging(strategy, total_steps=adam_steps, keys=staging_keys,
                                    off_frac=staging_off_frac, ramp_frac=staging_ramp_frac)
    # The stationarity residual is 45 % of a step (measured, 96x96 N=3: 9.39 vs 5.15 ms)
    # and its weight now defaults to 0. Skip it ONLY when it provably cannot contribute:
    # the strategy's weights must be a static function of base/step (adaptive strategies
    # may raise a weight of 0 later, so they never qualify) and its base weight must be 0.
    compute_resid = not (getattr(strategy, "static_weights", False)
                         and float(strategy.base.get("resid", 0.0)) == 0.0)
    if m < N and not compute_resid:
        # MEASURED, not inferred: the latent fields enter the objective through
        # stationarity_residual and nothing else, so at resid weight 0 their gradient is
        # exactly 0.0 (checked at N=3, m=2). Optimising them would be theatre and the
        # returned latent_fields would be their random init dressed up as a recovery.
        # Fail loud rather than emit a meaningless hidden-channel result.
        raise ValueError(
            f"m={m} < N={N} (hidden channels) but the stationarity residual has weight 0, "
            "and it is the ONLY term the latent fields enter. They would receive zero "
            "gradient and the recovered latent_fields would be the init. Set "
            "loss.weights.resid > 0 for hidden-channel runs, or add a term that sees the "
            "latent fields. NOTE exp06 measured the residual as harmful to Turing recovery "
            "(9/9 swept cells collapsed), so hidden-channel recovery currently has no "
            "known-good objective — that is an open problem, not a config mistake.")
    term_kw = dict(split_hinges=split_hinges, hinge_k_min_frac=hinge_k_min_frac,
                   detach_xstar=detach_xstar, compute_resid=compute_resid)

    kstar_obs = obs.kstar_of(frame[0].detach().cpu().numpy(), L=L)   # firewall: FFT of the observed image
    kgrid = _kgrid_for(kstar_obs, device=dev)

    best = None; restart_log = []
    for r in range(n_restarts):
        model = RNGRN(N=N, form=form, seed=seed + r).to(dev)
        latent = None
        if m < N:
            base_field = frame.mean(0, keepdim=True)
            lat0 = base_field.repeat(N - m, 1, 1) * (0.8 + 0.4 * torch.rand(N - m, 1, 1, device=dev, dtype=frame.dtype))
            latent = torch.nn.Parameter(lat0.clone())
        params = list(model.parameters()) + ([latent] if latent is not None else [])

        opt = torch.optim.Adam(params, lr=adam_lr)
        failed = False
        for step in range(adam_steps):
            opt.zero_grad()
            try:
                loss, parts = LT.total_loss(model, frame, L, observed_idx, kgrid, kstar_obs,
                                            strategy, step=step, latent_fields=latent,
                                            tau=tau, jac_floor=jac_floor, strict=True,
                                            **term_kw)
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
                loss, _ = LT.total_loss(model, frame, L, observed_idx, kgrid, kstar_obs,
                                        strategy, step=adam_steps, latent_fields=latent,
                                        tau=tau, jac_floor=jac_floor, **term_kw)
                loss.backward(); return loss
            try:
                lopt.step(closure)
            except Exception:
                pass

        try:
            with torch.no_grad():
                loss, parts = LT.total_loss(model, frame, L, observed_idx, kgrid, kstar_obs,
                                            strategy, step=adam_steps, latent_fields=latent,
                                            tau=tau, jac_floor=jac_floor, **term_kw)
        except SteadyStateError:
            # Same condition as a mid-training failure, so handle it the same way. Without
            # this the FINAL scoring pass was unguarded: a single restart that ended on
            # parameters with no valid steady state aborted the whole recovery, discarding
            # every other restart, instead of being logged and skipped.
            if verbose:
                print(f"  restart {r}: steady state diverged at final scoring; skipping restart")
            restart_log.append(dict(restart=r, total=float("inf"), steady_state_failed=True,
                                    failed_at="final_eval"))
            continue
        restart_log.append(dict(restart=r, total=float(loss), sig_max=parts.get("sig_max"),
                                sig_max_pos=parts.get("sig_max_pos"),
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
