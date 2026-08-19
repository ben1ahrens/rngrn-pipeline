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

D INIT, unit B4 (defect 2) -- opt-in via `d_init_from_kstar` / ModelConfig.d_init_from_kstar.
model.py's "default" init used to set theta_D ~ N(0, 0.5) unconditionally, i.e. D starts at
median 1.00 (measured over 2000 seeds) regardless of L or path. That scale was never chosen
for either path, and the two paths were off from the target in OPPOSITE directions.
Measured on the 19 three_gene_val samples (L = 40..139):

    dimensional : generator D has median 28.8    -> init started  29x TOO SMALL
    unit box    : generator D/L**2 median 4.97e-3 -> init started 202x TOO LARGE

(This is a statement about the numerical scale the optimiser starts from, NOT a claim that
the generator's D values are biologically viable -- they are not; see the priors work.)
So a head-to-head "dimensional vs nondim" result at a fixed step budget was CONFOUNDED by
the init. The repair is NOT to shift theta_D by -2*log(L) on the nondim path: that would put
L back inside the optimisation and destroy the exact cross-L invariance that is the entire
point of the arm. Instead theta_D is shifted by -2*log(k*_obs) -- k*_obs is an IMAGE
observable (already a legal, firewall-clean recovery input) computed in the OBJECTIVE's own
units on both paths (rad/length dimensional, rad/box nondim), so the same formula is
automatically correct on both paths without ever reading L directly. This is OPT-IN
(default False) because it changes the raw-parameter init distribution and therefore every
number recorded with the old default -- see docs/TUNING.md and the PR body for the measured
before/after D-ratio comparison.
"""
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import RNGRN, BatchedRNGRN
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


def _restart_seed(model_seed, r):
    """Deterministic per-(model_seed, restart) init seed, independent of neighbouring
    model_seeds (unit B1). Prior scheme was `model_seed + r`, a sliding window: run
    seed s and run seed s+1 shared n_restarts-1 of their n_restarts model inits.
    Reuse the stable-hash approach from scripts/exp11_robustness_baseline.py's
    stable_seed() -- NOT builtin hash(), which Python salts per process
    (PYTHONHASHSEED) and would make this non-reproducible across processes."""
    h = hashlib.blake2b(f"{model_seed!r}|{r!r}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % 2 ** 32


def _kgrid_for(kstar_obs, n=400, span=8.0, device=None, dtype=torch.float64):
    # unit B4 (defect 1): the floor used to be an ABSOLUTE 2.0 rad/length, which is not
    # scale-free -- on the dimensional path it silently dominates whenever kstar_obs < 0.25
    # (L > 150.8 on the registered datasets, 11/287 samples, 3.8%), pinning the grid to the
    # wrong band instead of tracking the observed wavenumber. The only job of the floor is
    # to keep the grid non-degenerate when kstar_obs is itself ~0 (a patternless frame), so
    # it is now defined relative to kmin -- expressed in kstar_obs's own units, whatever
    # they are (rad/length dimensional, rad/box nondim) -- instead of a fixed constant.
    kmin = kstar_obs / 50.0 + 1e-3
    kmax = max(span * kstar_obs, 2.0 * kmin)
    return torch.linspace(kmin, kmax, n, device=device, dtype=dtype)


def _topology(model):
    KA = model.KA.detach().cpu().numpy(); KR = model.KR.detach().cpu().numpy()
    s = model.s.detach().cpu().numpy(); g = model.gate.detach().cpu().numpy()
    sign = np.where(g > 0.5, 1, -1)
    sign = np.where(s > 0.05 * s.max(), sign, 0)
    return dict(sign=sign, magnitude=s, gate=g, KA=KA, KR=KR)


def _clip_grad_norm_per_member(params, max_norm, B):
    """Clip each BATCH MEMBER's gradient norm independently, in place. Returns the (B,)
    pre-clip norms.

    THIS IS NOT COSMETIC. torch.nn.utils.clip_grad_norm_ on stacked (B, ...) parameters
    clips the JOINT norm over all B members, so one member with a large gradient would shrink
    every other member's step — the batched members would stop being independent and a
    batched result would not be comparable to a serial one. This reproduces torch's own rule
    (coef = max_norm / (norm + 1e-6), clamped to <= 1) per member, which makes one Adam step
    over the stack exactly B independent Adam steps.
    """
    sq = None
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.reshape(B, -1)
        s = (g * g).sum(dim=-1)
        sq = s if sq is None else sq + s
    if sq is None:
        return None
    nrm = torch.sqrt(sq)
    coef = torch.clamp(max_norm / (nrm + 1e-6), max=1.0)
    for p in params:
        if p.grad is None:
            continue
        p.grad.mul_(coef.reshape(B, *([1] * (p.grad.dim() - 1))))
    return nrm


def _batched_restarts(N, form, restart_seeds, init, dispersion_backend,
                      frame, L_model, observed_idx, kgrid, kstar_obs, strategy,
                      adam_steps, adam_lr, grad_clip, tau, jac_floor, dev, verbose,
                      term_kw, kstar_obs_init=None, history=None):
    """Run all restarts SIMULTANEOUSLY as one batched optimisation (unit b2).

    `restart_seeds` is the EXPLICIT list of per-member init seeds, one per restart. It is
    passed in rather than derived here so the seed policy lives in exactly one place
    (recover._restart_seed, unit B1) and the batched and serial paths cannot drift apart.
    An earlier form of this function took `model_seed` and used `seed0 + r` internally,
    which would have reintroduced the sliding-window seed dependence B1 removed.

    `kstar_obs_init` is the unit-B4 D-init wavenumber, or None to keep the legacy D init.
    It must be in the SAME units the objective runs in (rad/length dimensional, rad/box
    nondim) — passing the physical k* on the nondim path would put L back into the init and
    destroy the cross-L invariance that path exists for.

    Returns (best, restart_log) in exactly the shape recover()'s serial loop produces, so the
    tail of recover() (unit conversion, RecoveryResult assembly) is shared and cannot drift.
    `best` is (loss, serial RNGRN of the winning member, member parts dict, x* array, None).

    EQUIVALENCE, and where it stops. The loss is a sum over members of per-member terms, so
    each member's gradient is exactly its own; Adam is elementwise; clipping is per member.
    One batched step is therefore B independent serial steps up to floating-point
    associativity. It is NOT bit-identical over a long run: a ~1e-16 difference in a step is
    amplified by the optimiser, so a 1500-step batched restart lands near, not on, its serial
    twin. tests/test_batched.py measures both the single-step agreement and the multi-step
    drift; treat batched and serial as the SAME METHOD but not the same random draw.

    DEAD MEMBERS. A member whose steady state stops converging is killed for good (the serial
    path abandons that restart on the first SteadyStateError) and excluded from the summed
    loss from then on, so it contributes no gradient. It keeps occupying its lane in the batch
    -- reindexing the batch mid-run would change nothing about the answer and would make the
    seed-to-lane mapping unauditable -- so a run where most members die costs the same as one
    where none do.
    """
    restart_seeds = list(restart_seeds)
    B = len(restart_seeds)
    bmodel = BatchedRNGRN.from_seeds(N=N, form=form, seeds=restart_seeds,
                                    dispersion_backend=dispersion_backend, init=init,
                                    kstar_obs=kstar_obs_init).to(dev)
    params = list(bmodel.parameters())
    opt = torch.optim.Adam(params, lr=adam_lr)
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    died_at = [None] * B
    # Per-member death step, kept on-device. Updated every step by pure tensor ops (no host
    # sync); only read back to host on the cadence below. -1 means "still alive".
    died_at_step = torch.full((B,), -1, dtype=torch.long, device=dev)
    LIVENESS_SYNC_EVERY = 25   # see the comment at the sync point below
    loss_kw = dict(tau=tau, jac_floor=jac_floor, **term_kw)

    for step in range(adam_steps):
        opt.zero_grad()
        loss_vec, parts, conv = LT.total_loss_batched(
            bmodel, frame, L_model, observed_idx, kgrid, kstar_obs, strategy,
            step=step, **loss_kw)
        newly_dead = alive & ~conv                               # tensor op, no host sync
        died_at_step = torch.where(newly_dead, torch.full_like(died_at_step, step), died_at_step)
        alive = alive & conv                                     # tensor op, no host sync
        is_sync_step = (step % LIVENESS_SYNC_EVERY == 0) or (step == adam_steps - 1)
        if is_sync_step:
            # ONE D2H sync here does the work that used to cost TWO every single step
            # (bool(newly_dead.any()) and bool(alive.any())). A member already stops
            # contributing gradient the step it dies (`alive` masks the loss below, updated
            # tensor-side every step), so cadencing only the host bookkeeping means: the
            # early break can fire up to LIVENESS_SYNC_EVERY-1 steps late, and per-death
            # logging can batch up to that many steps late (D-PERF-7). NOT "harmless" in the
            # stronger sense a prior version of this comment claimed: during those extra
            # steps `total` is an exact zero tensor WITH A GRAPH once every member is dead,
            # so the fresh gradient contribution is zero, but Adam's momentum/velocity state
            # from prior real gradients keeps decaying and being applied -- it DOES move the
            # parameters. What is actually true is narrower: no REPORTED number depends on
            # it, because `final_alive` is all-False for such a batch and every member logs
            # `steady_state_failed` regardless of where its parameters drifted. Separately,
            # `verbose` at such a step computes `float(loss_vec[alive].mean())` over an
            # empty boolean selection, which is NaN, not an error.
            died_at_step_host = died_at_step.tolist()
            still_alive = False
            for b, died_step in enumerate(died_at_step_host):
                if died_step < 0:
                    still_alive = True
                elif died_at[b] is None:
                    died_at[b] = died_step
                    if history is not None:
                        history.record_death(b, died_step)
                    if verbose:
                        print(f"  member {b} step {died_step}: steady state diverged; "
                              "member abandoned")
            if not still_alive:
                break
        # Recorded BEFORE opt.step(), so the parameters in the trace are the ones that
        # produced the loss in the same row. `alive` is passed so an abandoned lane stays
        # NaN instead of logging frozen parameters nobody is optimising any more.
        if history is not None and history.should_record(step):
            history.record_batched(step, parts, bmodel, alive=alive)
        total = torch.where(alive, loss_vec, torch.zeros_like(loss_vec)).sum()
        total.backward()
        _clip_grad_norm_per_member(params, grad_clip, B)
        opt.step()
        if verbose and step % 300 == 0:
            print(f"  batched step {step}: alive={int(alive.sum())}/{B} "
                  f"mean_total={float(loss_vec[alive].mean()):.3f}")

    with torch.no_grad():
        loss_vec, parts, conv = LT.total_loss_batched(
            bmodel, frame, L_model, observed_idx, kgrid, kstar_obs, strategy,
            step=adam_steps, **loss_kw)
    final_alive = alive & conv
    # The FINAL parameters -- the ones that get checkpointed -- are always in the trace,
    # whatever the stride (TrainingHistory.should_record).
    if history is not None:
        history.record_batched(adam_steps, parts, bmodel, alive=final_alive)

    best, restart_log = None, []
    for b in range(B):
        if not bool(final_alive[b]):
            entry = dict(restart=b, total=float("inf"), steady_state_failed=True)
            if bool(alive[b]):
                entry["failed_at"] = "final_eval"
            restart_log.append(entry)
            continue
        pm = LT.parts_member(parts, b)
        lb = float(loss_vec[b])
        # No `lbfgs_error` key here (D-PERF-8): this batched path never runs an LBFGS
        # polish per member, unlike the serial restart loop below, which records one.
        # The two paths therefore emit different column sets into the run index.
        restart_log.append(dict(restart=b, total=lb, sig_max=pm.get("sig_max"),
                                sig_max_pos=pm.get("sig_max_pos"),
                                kstar_model=pm.get("kstar_model"),
                                rel_err=pm.get("rel_err")))
        if best is None or lb < best[0]:
            member = bmodel.member(b)
            from .losses.terms import steady_state
            xs, _ = steady_state(member)     # SERIAL reference x* for the reported winner
            best = (lb, member, pm, xs.detach().cpu().numpy(), None)
    return best, restart_log


def recover(recovery_input, form="competitive", strategy=None, weights=None,
            tau=0.12, jac_floor=1.0, n_restarts=4, adam_steps=1500, adam_lr=0.05,
            lbfgs_steps=50, grad_clip=10.0, seed=0, verbose=False, device=None,
            split_hinges=True, hinge_k_min_frac=0.1, staging_keys=("turing",),
            staging_off_frac=0.25, staging_ramp_frac=0.25, detach_xstar=False,
            nondim=False, model_seed=None, dispersion_backend="eig", init="default",
            d_init_from_kstar=False,   # unit b4
            batched=False,             # unit b2
            dratio_centre=7.5, dratio_spread=1.0,          # unit 5, biological prior
            bio_box_path="configs/bio_box.yaml",           # unit 5
            spectral_b_lo=0.60, spectral_b_hi=1.55,        # unit U4, D-FFT-9 closure 1
            spectral_channels=(0,), spectral_nblk=24,      # unit U4
            spectral_ignition_margin=1e-3,                 # unit U4, UNCALIBRATED
            history=None):             # plottable training trajectory
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

    model_seed: seeds the model's random raw-parameter init. Each restart r draws its
        init seed from the stable hash of (model_seed, r) -- see _restart_seed() --
        so distinct (model_seed, r) pairs give independent draws and repeats are exact.
        Defaults to `seed` when not given, for backward compatibility.

    dispersion_backend: 'eig' (the DEFAULT, any N, the reference) | 'cubic' (exact for
        N == 3 ONLY -- model.py raises ValueError for any other N, including N=2; it is
        not "N<=3") | 'auto' (resolves to 'cubic' when N == 3, else 'eig'). Resolution
        happens in RNGRN.__init__; the model's .dispersion_backend always reads the
        concrete backend, never 'auto'. The default is deliberately NOT 'auto': cubic and
        eig runs are not bit-comparable (D-PERF-3).

    init: 'default' | 'low_basal' -- model raw-parameter init strategy (see model.py).
        Defaults to 'default' (OFF); callers opt in explicitly.

    d_init_from_kstar: unit B4 (defect 2), OFF by default. When True and init='default',
        theta_D is shifted so D starts at median 1/k*_obs**2 (in the objective's own units)
        instead of median 1.0, on BOTH paths -- see the module docstring. Changes recorded
        D / D-ratio numbers for any run that opts in; leaves everything else bit-identical.

    batched: False (DEFAULT, unchanged serial behaviour) runs the restarts one at a time.
        True optimises ALL `n_restarts` restarts simultaneously as one batched computation
        (unit b2, model.BatchedRNGRN), which is what makes a high step budget on GPU
        affordable -- a serial step is B tiny kernel launches, a batched step is one B-wide
        launch. SAME OBJECTIVE, SAME METHOD, NOT THE SAME ARITHMETIC ORDER: agreement with
        the serial path is ~1e-12 for one step, not bit-exact over a whole run (see
        _batched_restarts). Requires lbfgs_steps=0, m==N, and a static weighting strategy
        with resid weight 0; each of those raises rather than quietly degrading.

    spectral_b_lo / spectral_b_hi / spectral_channels / spectral_nblk /
    spectral_ignition_margin: unit U4 (M1 spectral terms, losses/spectral.py) knobs for
        `SpectralConfig`. Only consulted when `strategy.base` carries a non-zero weight for
        at least one of the five spectral terms (spec_shape/spec_aniso/spec_amp_mean/
        spec_amp_fluct/real_moments) -- the same implicit gate `resid`/`param_prior` already
        use, not a separate enabled flag. When active: `build_frame_targets` runs ONCE (the
        observed frame does not change across restarts), a `forward.PatternSolver` is built
        PER RESTART (it owns per-restart warm-start state) seeded like the model init
        (`_restart_seed`), and both are threaded into `losses.total.total_loss` as a
        `losses.spectral.SpectralContext`. RAISES if a non-zero spectral weight is combined
        with `batched=True` (no batched forward solve) or `split_hinges=False` (no
        `sig_max_pos` -- no ignition signal). The forward solve is expensive (3-9 s/solve
        at 64^2 -- unrecorded test timing; forward.py module docstring); every default
        here keeps it OFF.

    history: an optional `history.TrainingHistory`. When given, the per-step loss terms, the
        live weights, and the CONSTRAINED physical parameters of EVERY member are recorded at
        the recorder's own thinned cadence, on both the serial and the batched path. The
        recorder is passed IN and read back by the caller rather than returned on the
        RecoveryResult, because it is an optional side-record and not part of the recovery's
        answer -- and because this module must not depend on it: nothing here imports
        `history`, it only calls methods on whatever object it is handed. None (the default)
        records nothing; the recorder itself only reads parameters under `no_grad`, so a
        recorded run and an unrecorded one produce the same numbers (verified bit-identical on
        the tracked `m3_registry_20260730_005701` recovery, see
        docs/LGEN_TRANSFER_FIRST_RESULT.md).
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
    # The biological-plausibility prior (unit 5) is opt-in the SAME way the residual is
    # opt-out: it is computed only when it can contribute, i.e. the strategy's weights are
    # a static function of base/step AND its base weight is non-zero. Passing None omits
    # the key from term_vals entirely, so every number recorded before this was wired in
    # is reproduced bit-for-bit. Before this, loss.weights.param_prior was a NO-OP on this
    # path -- losses/total never called terms.param_prior -- so a run asking for the prior
    # silently trained without it.
    use_param_prior = (getattr(strategy, "static_weights", False)
                       and float(strategy.base.get("param_prior", 0.0)) != 0.0)
    if float(strategy.base.get("param_prior", 0.0)) != 0.0 and not use_param_prior:
        raise ValueError(
            "loss.weights.param_prior > 0 requires a strategy with static weights "
            f"(got {type(strategy).__name__}). An adaptive strategy reads a term's VALUE "
            "to set its weight, and the prior would then be re-weighted by its own "
            "magnitude rather than by the recorded bio_box prior strength.")

    # unit U4 (M1 spectral terms). Gated on the strategy's BASE weights, exactly like
    # resid/param_prior above -- not a separate enabled flag. MISCONFIGURATION RAISES:
    # both conditions below would otherwise make the spectral terms silently unavailable
    # rather than loudly refused.
    from .losses.spectral import SPECTRAL_TERM_KEYS  # unit U4
    use_spectral = any(float(strategy.base.get(k, 0.0)) != 0.0 for k in SPECTRAL_TERM_KEYS)
    if use_spectral and batched:
        raise ValueError(
            "a spectral weight is non-zero but batched=True (unit U4): forward.PatternSolver "
            "owns per-restart warm-start state, which has no batched form, and the batched "
            "reaction does not broadcast to the per-pixel fields the forward solve needs. "
            "Run the serial path (batched=False) for spectral runs.")
    if use_spectral and not split_hinges:
        raise ValueError(
            "a spectral weight is non-zero but split_hinges=False (unit U4): spectral "
            "ignition (losses.spectral.is_ignited) reads parts['sig_max_pos'], which only "
            "losses.terms.turing_hinges_split (split_hinges=True) produces. Set "
            "split_hinges=True, or zero every spec_*/real_moments weight.")
    # Numerics review, 2026-08-12: the spectral terms compare u_star[c] (solver SPECIES c)
    # against targets from frame[c] (frame CHANNEL c) with no observed_idx routing, so
    # they are correct ONLY when channel c observes species c on every fitted channel.
    # Anything else (partial observation, permuted observed_idx) would silently fit the
    # WRONG species -- refuse loudly instead. Routing u_star[observed_idx[c]] is deferred
    # until a non-identity spectral run actually exists (Stage 2+); building it now would
    # be untestable speculation.
    if use_spectral and any(observed_idx[c] != c for c in spectral_channels):
        raise ValueError(
            f"a spectral weight is non-zero but observed_idx={observed_idx} is not the "
            f"identity on the fitted spectral channels {tuple(spectral_channels)}: the "
            "spectral terms compare u_star[c] against frame[c] directly and would fit the "
            "wrong species. Identity mapping only, until observed_idx routing lands.")
    # Same wrinkle class as param_prior above: an adaptive strategy refreshes its weights
    # only every update_every steps, so a term that IGNITES mid-run would contribute 0
    # until the next refresh and then jump. Require static weights for spectral runs.
    if use_spectral and not getattr(strategy, "static_weights", False):
        raise ValueError(
            "a spectral weight is non-zero but the strategy is adaptive "
            f"({type(strategy).__name__}): adaptive weights refresh on a cadence, so a "
            "term igniting between refreshes would silently contribute 0 and then jump. "
            "Spectral runs require a static-weight strategy (same rule as param_prior).")

    # The bio box is READ FROM DISK by losses.terms.param_prior whenever `box` is None --
    # a file open and a yaml.safe_load on EVERY step of every restart. It is a static config
    # file, so load it once here and pass the parsed bounds down instead of the path.
    param_prior_kw = None
    if use_param_prior:
        from .losses.terms import _load_box_bounds
        param_prior_kw = dict(dratio_centre=dratio_centre, dratio_spread=dratio_spread,
                              box=_load_box_bounds(bio_box_path))
    term_kw = dict(split_hinges=split_hinges, hinge_k_min_frac=hinge_k_min_frac,
                   detach_xstar=detach_xstar, compute_resid=compute_resid,
                   param_prior_kw=param_prior_kw)

    # The length unit the objective is written in. nondim=True sets it to the box itself,
    # which is an exact change of variables: obs.kstar_of and obs.laplacian_torch are both
    # exactly homogeneous in L (k scales as 1/L, the Laplacian as 1/L**2), so no
    # approximation is introduced here -- only the units of D and k change.
    L_model = 1.0 if nondim else L
    kstar_obs = obs.kstar_of(frame[0].detach().cpu().numpy(), L=L_model)  # firewall: FFT of the observed image
    kgrid = _kgrid_for(kstar_obs, device=dev)

    # Two per-STEP host<->device round trips the loss terms were paying for quantities that
    # are FIXED for this whole call: the frame's mean (the scale anchor's target, a
    # device->host sync of a frame that never changes) and the k-grid index k*_obs
    # interpolates into (a host->device copy of the key plus a search). Resolve both once
    # and thread them through term_kw; losses/total defaults them to the old per-step form
    # for any other caller. Same values, same terms -- see losses/total.compute_terms.
    term_kw["obs_scale"] = float(frame.mean())
    term_kw["kstar_idx"] = int(
        torch.searchsorted(kgrid, torch.as_tensor(float(kstar_obs), device=kgrid.device))
        .clamp(1, len(kgrid) - 1))

    # unit U4: the observed-frame spectral targets are FIXED across every restart (the
    # frame does not change), so build them ONCE here. `spec_cfg`/`spec_targets` stay None
    # when `use_spectral` is False, which is the whole default config.
    spec_cfg = spec_targets = None
    if use_spectral:
        from .losses.spectral import SpectralConfig, build_frame_targets
        spec_cfg = SpectralConfig(b_lo=spectral_b_lo, b_hi=spectral_b_hi,
                                  channels=tuple(spectral_channels), nblk=spectral_nblk,
                                  ignition_margin=spectral_ignition_margin)
        spec_targets = build_frame_targets(frame.detach().cpu().numpy(), L_model, kstar_obs,
                                           spec_cfg)

    best = None; restart_log = []
    if batched:
        # unit b2. Every restriction below is refused loudly because each one would otherwise
        # make a batched number quietly non-comparable to a serial one:
        #   * LBFGS keeps ONE history and ONE line search, so over stacked parameters it is a
        #     single joint quasi-Newton solve, NOT B independent polishes. There is no way to
        #     batch it that preserves the serial meaning.
        #   * the stationarity residual has no batched form (per-pixel states; see
        #     losses/total.compute_terms_batched), which also rules out m<N -- the latent
        #     fields enter the objective through the residual and nothing else.
        #   * value-reading strategies (RatioWeighting) reduce a term to a float and cannot
        #     see B of them.
        if lbfgs_steps:
            raise ValueError(
                f"batched=True requires lbfgs_steps=0 (got {lbfgs_steps}). LBFGS over the "
                "stacked parameters is one joint solve sharing a line search and a curvature "
                "history across members, not B independent polishes, so its result would not "
                "be comparable to a serial restart's. Run Adam batched and polish serially.")
        if compute_resid:
            raise ValueError(
                "batched=True requires the stationarity residual to be off (weights.resid=0 "
                "with a static strategy). The batched reaction takes one state vector per "
                "member, not per-pixel states, so there is no batched residual.")
        if m < N:
            raise ValueError(
                f"batched=True does not support hidden channels (m={m} < N={N}): the latent "
                "fields enter only the stationarity residual, which has no batched form.")
        # Seeds are derived with _restart_seed, exactly as the serial loop below does, so the
        # two paths start from the SAME B inits and B1's seed independence holds on both.
        # (unit b2 originally used model_seed + r here, which would have silently
        # reintroduced the sliding-window defect B1 removed.)
        best, restart_log = _batched_restarts(
            N, form, [_restart_seed(model_seed, r) for r in range(n_restarts)], init,
            dispersion_backend, frame, L_model,
            observed_idx, kgrid, kstar_obs, strategy, adam_steps, adam_lr, grad_clip,
            tau, jac_floor, dev, verbose, term_kw,
            kstar_obs_init=kstar_obs if d_init_from_kstar else None, history=history)
    # the serial loop is skipped entirely when the batched path ran; it stays the REFERENCE
    # implementation and the default, so no pre-existing number changes method.
    for r in range(0 if batched else n_restarts):
        model = RNGRN(N=N, form=form, seed=_restart_seed(model_seed, r), init=init,
                      dispersion_backend=dispersion_backend,
                      kstar_obs=kstar_obs if d_init_from_kstar else None).to(dev)
        # unit U4: a fresh PatternSolver per restart (it owns per-restart warm-start state,
        # so restart r's Newton warm start must never leak into restart r+1's). Seeded like
        # the model init so a spectral run is reproducible the same way the init is.
        spectral_ctx = None
        if use_spectral:
            from .forward import PatternSolver
            from .losses.spectral import SpectralContext
            # warm_mode="relax" is the TRAINING policy (GPU-port unit, 2026-08-12): the
            # losses are translation-invariant so a warm re-relax's phase drift is
            # harmless here, and Newton-only warm starts measured pathological at
            # Adam-scale theta displacement (5030 s vs fresh 938 s at 96^2). The
            # "newton" default on PatternSolver itself remains the FD-instrumentation
            # contract. Device is derived from the model (already .to(dev) above).
            solver = PatternSolver(model, n=frame.shape[-1], L=L_model,
                                   seed=_restart_seed(model_seed, r),
                                   warm_mode="relax")
            spectral_ctx = SpectralContext(solver=solver, targets=spec_targets, cfg=spec_cfg)
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
                                            spectral=spectral_ctx, **term_kw)
            except SteadyStateError:
                # fail-loud honoured: this init cannot form a valid steady state — abandon
                # the restart rather than optimise against a meaningless x*.
                if verbose:
                    print(f"  restart {r} step {step}: steady state diverged; skipping restart")
                if history is not None:
                    history.record_death(r, step)
                failed = True
                break
            # BEFORE opt.step(): the recorded parameters are the ones that produced this loss.
            if history is not None and history.should_record(step):
                history.record_serial(step, r, parts, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            if verbose and step % 300 == 0:
                print(f"  restart {r} step {step}: total={parts['total']:.3f} "
                      f"kstar_m={parts.get('kstar_model', float('nan')):.2f} "
                      f"sig_max={parts.get('sig_max', float('nan')):.3f}")
        if failed:
            # `failed_at_step` (unit B3): WHICH Adam step lost the steady state. Diagnosing
            # the nc1 instability needed exactly this and the log did not have it — the
            # failures cluster late (step 779-1188 on the measured nc1 trajectories), so a
            # run that dies at step 900 of 2000 is a different animal from one that dies at
            # init, and the pre-B3 log could not tell them apart.
            restart_log.append(dict(restart=r, total=float("inf"), steady_state_failed=True,
                                    failed_at="train", failed_at_step=step))
            continue

        lbfgs_error = None
        if lbfgs_steps:
            lopt = torch.optim.LBFGS(params, max_iter=lbfgs_steps, line_search_fn="strong_wolfe")
            def closure():
                lopt.zero_grad()
                latent = latent_module() if latent_module is not None else None
                loss, _ = LT.total_loss(model, frame, L_model, observed_idx, kgrid, kstar_obs,
                                        strategy, step=adam_steps, latent_fields=latent,
                                        tau=tau, jac_floor=jac_floor,
                                        spectral=spectral_ctx, **term_kw)
                loss.backward(); return loss
            try:
                lopt.step(closure)
            except Exception as e:
                # The LBFGS polish is optional refinement (Adam already produced a usable
                # `model`), so a failed polish does not abort the restart -- but the failure
                # must not vanish silently either: this is where the FIRST async CUDA error
                # of a GPU run would previously have been swallowed by a bare `except: pass`.
                # Recorded on the restart log (below) so it lands in the results rather than
                # only in a log line no one reads.
                lbfgs_error = f"{type(e).__name__}: {e}"
                if verbose:
                    print(f"  restart {r}: LBFGS polish failed ({lbfgs_error}); "
                          "keeping the pre-LBFGS (Adam) parameters")

        try:
            with torch.no_grad():
                latent = latent_module() if latent_module is not None else None
                loss, parts = LT.total_loss(model, frame, L_model, observed_idx, kgrid, kstar_obs,
                                            strategy, step=adam_steps, latent_fields=latent,
                                            tau=tau, jac_floor=jac_floor,
                                            spectral=spectral_ctx, **term_kw)
        except SteadyStateError:
            # Same condition as a mid-training failure, so handle it the same way. Without
            # this the FINAL scoring pass was unguarded: a single restart that ended on
            # parameters with no valid steady state aborted the whole recovery, discarding
            # every other restart, instead of being logged and skipped.
            if verbose:
                print(f"  restart {r}: steady state diverged at final scoring; skipping restart")
            if history is not None:
                history.record_death(r, adam_steps)
            restart_log.append(dict(restart=r, total=float("inf"), steady_state_failed=True,
                                    failed_at="final_eval"))
            continue

        # The FINAL parameters -- post-LBFGS, the ones that get checkpointed -- are always in
        # the trace, whatever the stride (TrainingHistory.should_record).
        if history is not None:
            history.record_serial(adam_steps, r, parts, model)
        restart_log.append(dict(restart=r, total=float(loss), sig_max=parts.get("sig_max"),
                                sig_max_pos=parts.get("sig_max_pos"),
                                kstar_model=parts.get("kstar_model"), rel_err=parts.get("rel_err"),
                                lbfgs_error=lbfgs_error))
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
    # ---- back to the host ------------------------------------------------------------
    # `device` is a knob for the OPTIMISATION only; RecoveryResult is a host-side object.
    # Every consumer works in numpy on the CPU, and validate.score_recovery in particular
    # probes the Jacobian with `torch.as_tensor(result.xstar)` -- a CPU tensor -- which
    # raised "Expected all tensors to be on the same device" against a CUDA model. That was
    # unreachable before unit b2 only because nothing ever passed recover(device=...). The
    # copy is exact and a no-op on the default CPU path, so no recorded number changes.
    model = model.to("cpu")

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
