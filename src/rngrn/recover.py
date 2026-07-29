"""recover.py — per-frame variational recovery (RECOVERY SIDE).

Optimise the RNGRN raw parameters theta for ONE frame by gradient descent on the
composite objective: Adam for global exploration, optional LBFGS polish. RANDOM
init only, multiple restarts, keep the best by total loss. For m<N, latent fields
for the unobserved channels are additional optimisation variables.

FIREWALL: the ONLY input is a data.gate.RecoveryInput = (frame, L, observed_idx, N).
No analytic x*, J, sigma, coefficient, or truth-derived init. k*_obs comes from the
frame's own FFT. This module MUST NOT import rd_models / solver / cache / AnswerKey.

L IS LEGAL-BUT-LOADED. The gate is entitled to hand recovery the domain size, and this
module uses it (for the Laplacian's dx and for the units of the k-grid). But on the
datasets registered here the generator chose L = clip(6*2pi/k*_true, 18, 220), so
k*_true is recoverable from L alone for 94.8% of the 287 registered samples. Using L is
within the firewall; treating any L-derived number as EVIDENCE of recovery is not. See
docs/STATE_OF_THE_SCIENCE.md.

NON-DIMENSIONAL PATH (`nondim=True`, unit 12). Recovery can run in rescaled coordinates
x_hat = x / L, i.e. on the unit box. Then the learned object is (J, D/L**2) and the
dispersion lives in the dimensionless wavenumber k_hat = k*L (rad per box; q = k_hat/2pi
is periods per box). Because sigma is invariant under (k, D) -> (k*L, D/L**2), the
non-dimensional problem is EXACTLY the dimensional one with the length unit removed --
and the optimisation then never sees L at all, so the recovered network is independent of
domain size BY CONSTRUCTION. That is a design decision, not a measurement: L is degenerate
with the recovered parameter vector (feed a grid of true extent L but label it sL and every
second derivative scales by 1/s**2, so the residual is satisfied by D -> D*s**2), so
"generalises across L" cannot mean anything stronger without breaking the degeneracy.

DEFAULT IS THE DIMENSIONAL PATH (`nondim=False`), unchanged, so every pre-existing number
stays comparable. `RecoveryResult` reports PHYSICAL quantities on both paths.

KNOWN LIMITATION OF THE NON-DIMENSIONAL ARM -- READ BEFORE COMPARING THE TWO PATHS.
model.py inits theta_D ~ N(0, 0.5), i.e. D starts at median 1.00 (measured over 2000
seeds). That scale was never chosen for either path, and the two paths are off from the
target in OPPOSITE directions. Measured on the 19 three_gene_val samples (L = 40..139):

    dimensional : generator D has median 28.8   -> init starts  29x TOO SMALL
    unit box    : generator D/L**2 median 4.97e-3 -> init starts 202x TOO LARGE

(This is a statement about the numerical scale the optimiser starts from, NOT a claim that
the generator's D values are biologically viable -- they are not; see the priors work.)
So a head-to-head "dimensional vs nondim" result at a fixed step budget is CONFOUNDED by
the init, and any such comparison must say so. The obvious repair -- shifting theta_D by
-2*log(L) on the nondim path -- is NOT applied here, deliberately: it would put L back
inside the optimisation and destroy the exact cross-L invariance that is the entire point
of the arm. The L-free repair is to set the D init from the frame's own k*_obs
(D ~ |J|/k*_obs**2, firewall-clean), but that would change the DIMENSIONAL path too and
therefore invalidate every recorded number, so it belongs with the priors/init work and
not here. Left explicitly open. See TUNING.md.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import RNGRN
from . import observables as obs
from .losses import total as LT
from .losses.total import SteadyStateError
from .losses.weighting import build_strategy, FixedWeighting, DataFirstStaging


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
    """Outcome of one recover() call.

    `kstar_model`, `kstar_obs` and `params["D"]` are PHYSICAL (rad/length, length**2/time)
    on BOTH paths, so a consumer never has to know which path ran. The raw learned
    quantities -- which are dimensionless when nondim=True -- are carried separately as
    `params["D_model"]` / `kstar_model_nondim`.

    ONE EXCEPTION, stated so nobody trips over it: `parts` is the raw loss-term dict handed
    back by losses/total.py, so `parts["kstar_model"]` and `parts["kstar_obs"]` are in the
    units the OBJECTIVE ran in (rad/box when nondim=True). Use the top-level fields, not
    `parts`, for anything that is compared across runs. `restarts` HAS been converted.
    """
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
    # ---- unit 12: L-generalisation bookkeeping (appended; all have defaults) ---------
    nondim: bool = False           # did recovery run on the unit box (x_hat = x/L)?
    L: float = float("nan")        # the domain size the frame was labelled with
    D_phys: object = None          # physical diffusivities, = D_model * L**2 when nondim
    kstar_model_nondim: float = float("nan")   # k*_model * L  (rad per box)
    q_model: float = float("nan")              # k*_model * L / 2pi (PERIODS per box)


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
            staging_off_frac=0.25, staging_ramp_frac=0.25, detach_xstar=False,
            nondim=False, model_seed=None, dispersion_backend="eig", init="default"):
    """Recover a GRN from one RecoveryInput. Returns the best RecoveryResult.

    strategy: a WeightingStrategy instance (default FixedWeighting(weights or defaults)).

    split_hinges / hinge_k_min_frac / detach_xstar: passed through to losses.total (see
    compute_terms). staging_keys/off_frac/ramp_frac: data-first staging — the named terms
    are held at weight 0 for the first `off_frac` of `adam_steps`, then ramped to their
    configured weight over the next `ramp_frac`. Pass staging_keys=() to disable.

    nondim:   False (DEFAULT, unchanged behaviour) optimises in physical coordinates, so
              the Laplacian uses dx = L/n_grid and the k-grid is in rad/length.
              True optimises on the unit box x_hat = x/L: the Laplacian uses dx = 1/n_grid
              and the k-grid is in rad/box, so the learned diffusivity is D/L**2 and the
              learned dispersion is a function of k*L only. The optimisation is then
              literally independent of L (nothing else in the objective reads it), which is
              the whole content of the claim "the recovered network generalises across L".
              Reported quantities are converted back to physical either way.

    model_seed: seeds the model's random raw-parameter init (per-restart offset by r).
        Defaults to `seed` when not given, for backward compatibility.

    dispersion_backend: 'eig' (any N, the reference) | 'cubic' (exact for N<=3 ONLY).

    init: 'default' | 'low_basal' -- model raw-parameter init strategy (see model.py).
        Defaults to 'default' (OFF); callers opt in explicitly.
    """
    ri = recovery_input
    model_seed = seed if model_seed is None else model_seed
    dev = torch.device(device) if device is not None else torch.device("cpu")
    frame = torch.tensor(np.asarray(ri.frame, dtype=float), device=dev)
    L, N, observed_idx = ri.L, ri.N, list(ri.observed_idx)
    if not (np.isfinite(L) and L > 0):
        raise ValueError(f"RecoveryInput.L must be a positive finite domain size, got {L!r}")
    m = frame.shape[0]
    if strategy is None:
        base = weights or dict(kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5,
                               anchor=2.0, morphology=0.0)
        strategy = FixedWeighting(base)
    # adam_steps == 0 is a legitimate call (init-only / determinism checks). Staging is
    # defined as a fraction OF adam_steps, so it is meaningless there and
    # weighting.staging_factor rightly refuses total_steps=0 — skip the wrapper instead.
    if staging_keys and adam_steps > 0:
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

    # The length unit the objective is written in. nondim=True sets it to the box itself,
    # which is an exact change of variables: obs.kstar_of and obs.laplacian_torch are both
    # exactly homogeneous in L (k scales as 1/L, the Laplacian as 1/L**2), so no
    # approximation is introduced here -- only the units of D and k change.
    L_model = 1.0 if nondim else L
    kstar_obs = obs.kstar_of(frame[0].detach().cpu().numpy(), L=L_model)  # firewall: FFT of the observed image
    kgrid = _kgrid_for(kstar_obs, device=dev)

    best = None; restart_log = []
    for r in range(n_restarts):
        model = RNGRN(N=N, form=form, seed=model_seed + r, init=init,
                      dispersion_backend=dispersion_backend).to(dev)
        latent_module = None
        latent = None
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
                loss, parts = LT.total_loss(model, frame, L_model, observed_idx, kgrid, kstar_obs,
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
                latent = latent_module() if latent_module is not None else None
                loss, _ = LT.total_loss(model, frame, L_model, observed_idx, kgrid, kstar_obs,
                                        strategy, step=adam_steps, latent_fields=latent,
                                        tau=tau, jac_floor=jac_floor, **term_kw)
                loss.backward(); return loss
            try:
                lopt.step(closure)
            except Exception:
                pass

        try:
            with torch.no_grad():
                latent = latent_module() if latent_module is not None else None
                loss, parts = LT.total_loss(model, frame, L_model, observed_idx, kgrid, kstar_obs,
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

    # ---- back to physical units ------------------------------------------------------
    # Under x_hat = x/L the learned quantities are k_hat = k*L and D_hat = D/L**2, so the
    # inverse map is exactly k = k_hat/L and D = D_hat*L**2. `scale` is 1.0 on the
    # dimensional path, which makes every line below a no-op there -- deliberately, so the
    # default path is bit-identical to the pre-nondim code.
    scale = L if nondim else 1.0
    D_model = model.D.detach().cpu().numpy()
    D_phys = D_model * (scale ** 2)
    kstar_model_raw = parts.get("kstar_model", float("nan"))
    kstar_model_phys = kstar_model_raw / scale
    kstar_model_hat = kstar_model_raw if nondim else kstar_model_raw * L
    kstar_obs_phys = kstar_obs / scale
    # per-restart diagnostics were logged in MODEL units; put them on the same physical
    # scale as the headline so a reader never has to ask which units a restart row is in.
    restart_log = [dict(e, kstar_model=e["kstar_model"] / scale)
                   if e.get("kstar_model") is not None else e
                   for e in restart_log]
    P = dict(KA=model.KA.detach().cpu().numpy(), KR=model.KR.detach().cpu().numpy(),
             alpha=model.alpha.detach().cpu().numpy(), delta=model.delta.detach().cpu().numpy(),
             beta=model.beta.detach().cpu().numpy(),
             D=D_phys,        # PHYSICAL on both paths (identical to D_model when nondim=False)
             D_model=D_model)  # the raw learned parameter, dimensionless when nondim=True
    return RecoveryResult(model=model, params=P, topology=_topology(model), xstar=xstar,
                          kstar_model=kstar_model_phys, kstar_obs=kstar_obs_phys,
                          loss=loss, parts=parts, restarts=restart_log, latent_fields=latent_np,
                          nondim=bool(nondim), L=float(L), D_phys=D_phys,
                          kstar_model_nondim=kstar_model_hat,
                          q_model=kstar_model_hat / (2.0 * math.pi))
