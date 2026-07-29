"""rollout.py — lift-and-simulate: does the recovered circuit actually pattern?

Training constrains the reaction pointwise; it does NOT simulate the network.
This integrates the full lifted PDE from x* + small noise and measures whether a
pattern forms and at what k*. The integrator is pluggable (numerics.INTEGRATORS);
ETDRK4 is the stiff-safe default. The timestep/horizon are growth-rate-aware:
diffusion is handled inside the integrator (not a CFL limit), dt is set by the
fastest reaction rate, and the horizon is ~40x the LINEAR RATE timescale.

WHY THIS FILE WAS MADE CHEAPER (unit 7)
---------------------------------------
The morphology comparison — whether the recovered model reproduces the target's
pattern, the owner's primary criterion — needs a simulated field, and it had never once
been computed because the rollout cost minutes. Measured on this branch, CPU,
OMP_NUM_THREADS=1, N=3:

    step cost              96x96  3.06 ms (full-FFT ETDRK4)  ->  1.70 ms (rfft ETDRK4)
    planned step count     127904 - 200000 steps for six random-init N=3 models
    implied wall time      ~6.5 - 10 min per field

The step count, not the step cost, was the problem, and its cause was a sign bug:
`sig_max = max(sigd.max(), 1e-3)` floors a NEGATIVE dispersion maximum at 1e-3, so a
model whose uniform state is linearly STABLE (sigd.max() = -0.42 for all six seeds
measured) was handed a horizon of T = 40/1e-3 = 40000 time units — 40 000 growth times
of a growth that does not exist. Three changes follow, none of which touches the
integrator contract:

  1. the horizon rate is |sigd.max()|, not max(sigd.max(), 1e-3). Unstable: 40 growth
     times, as before. Stable: 40 DECAY times of the slowest mode, which is the correct
     "long enough to be sure nothing happens" scale. `sig_max` in the result is now the
     SIGNED maximum (what its name says); the magnitude actually used for the horizon is
     reported separately as `horizon_rate`. The two agree whenever the old value was
     meaningful, i.e. whenever the model is actually unstable.
  2. two optional mid-run stopping rules, enabled together by `early_stop=True`, which
     drives the run in chunks of `check_every` steps:
       * `_saturated` — the pattern's amplitude and dominant wavenumber have stopped
         moving. MEASURED HONESTLY: once (1) is in place this one does NOT fire on the
         Turing fixture in tests/test_rollout.py. The amplitude goes flat by ~step 200 of
         609, but k* keeps creeping as the labyrinth coarsens, and the rule (deliberately)
         requires both. It is kept for models with much longer horizons.
       * `_collapsed` — the field has decayed so far below the `patterned` threshold that
         the verdict can no longer flip. THIS is the one that pays: a near-marginal stable
         recovered model (sig_max = -0.0098, measured on three_gene_val/sample_0000) would
         otherwise spend the entire 20000-step budget integrating noise downwards.
  3. `max_steps` replaces the hardcoded 200000 clip, and `stopped_reason` records which
     bound actually ended the run, so a truncated rollout can never be misread as a
     converged one.

Chunked driving re-enters the integrator every `check_every` steps. The ETDRK4 phi
coefficients depend only on (D, n, L, dt), so numerics caches them; with that cache,
chunked and one-call driving of the same 609-step rollout differ by -1.8% (64x64) and
+1.5% (96x96), best of three — i.e. by nothing.

RESULTING COST, measured on the Turing fixture, CPU, OMP_NUM_THREADS=1, N=3, 609 steps:
    64x64    0.73 s (etdrk4)   0.54 s (etdrk4_rfft)
    96x96    1.51 s            0.94 s
    128x128  3.02 s            1.71 s
against ~6.5-10 min before, for the same default settings.
"""
from __future__ import annotations
import time

import numpy as np
import torch

from .. import observables as obs
from .numerics import INTEGRATORS


def _reaction_np_builder(model):
    KA = model.KA.detach().cpu().numpy(); KR = model.KR.detach().cpu().numpy()
    alpha = model.alpha.detach().cpu().numpy(); beta = model.beta.detach().cpu().numpy()
    delta = model.delta.detach().cpu().numpy(); n_h = model.n_hill
    form = model.form

    def reaction_np(X):  # X: (N,n,n)
        xn = np.clip(X, 0, None) ** n_h
        if form == 'competitive':
            denom = 1.0 + np.einsum('ij,jxy->ixy', KA + KR, xn)
            prod = np.einsum('ij,ij,jxy->ixy', alpha, KA, xn) / denom
        else:
            thA = KA[:, :, None, None] * xn[None] / (1 + KA[:, :, None, None] * xn[None])
            thR = KR[:, :, None, None] * xn[None] / (1 + KR[:, :, None, None] * xn[None])
            act = np.einsum('ij,ijxy->ixy', alpha, thA)
            veto = np.prod(1 - thR, axis=1)
            prod = act * veto
        return beta[:, None, None] + prod - delta[:, None, None] * X
    return reaction_np


def _saturated(amps, kstars, tol, window):
    """Has the pattern stopped changing? THE STOPPING RULE — read this before trusting a
    rollout that ended with stopped_reason='saturated'.

    `amps` and `kstars` are the observed amplitude (std of channel 0) and dominant
    wavenumber recorded at every check. Saturation is declared when the last `window`
    checks satisfy BOTH

        (max(a) - min(a)) / max(mean(a), tiny)  <  tol        # amplitude is flat
        (max(k) - min(k)) / max(mean(k), tiny)  <  tol        # the mode has stopped moving

    Both conditions are required because either alone is a known false positive: the
    amplitude is flat during the induction period before the instability takes off (a is
    the initial noise level and barely moves), and k* is flat throughout a run whose
    amplitude is still growing exponentially.

    [TUNE] `tol` and `window` are NOT calibrated and NOT measured to be useful. On the
    Turing fixture in tests/test_rollout.py this rule never fires at the defaults
    (tol=0.01, window=5, check_every=200): the amplitude is flat from ~step 200 of 609 but
    k* keeps creeping as the labyrinth coarsens, so the conjunction is never satisfied and
    the 40-growth-time horizon ends the run first. Scanning tol at check_every=100 it fires
    only at window=3 (step 600 at tol=0.01, step 400 at tol=0.05 on a 64x64 grid), i.e.
    barely before the horizon. The failure mode to watch for is a too-loose tol stopping
    during the induction period and reporting an unpatterned field; it shows up as
    stopped_reason='saturated' together with patterned=False.
    """
    if len(amps) < window:
        return False
    a = np.asarray(amps[-window:], float)
    k = np.asarray(kstars[-window:], float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(k))):
        return False
    a_span = (a.max() - a.min()) / max(abs(a.mean()), 1e-300)
    k_span = (k.max() - k.min()) / max(abs(k.mean()), 1e-300)
    return bool(a_span < tol and k_span < tol)


def _collapsed(amps, sig_max, level, window):
    """Has the field decayed so far that 'unpatterned' is already locked in?

    THE SECOND STOPPING RULE, and the one that actually pays. A recovered model whose
    uniform state is linearly STABLE cannot grow a pattern out of the initial noise, but a
    NEARLY marginal one takes forever to say so: an e2e run measured sig_max = -0.0098,
    giving T = 40/0.0098 = 4076 time units at dt = 0.095, i.e. 42900 steps — 28 s of
    integrating a field down to 1.9e-12, at which point its "morphology" is float noise.

    Three conditions, all required:
      * sig_max <= 0 — no mode grows, so the linear prediction is monotone decay. With a
        positive sig_max the field may still take off and this rule must never fire.
      * the last `window` amplitudes are all below `level`, which is a fixed fraction of
        the very threshold `patterned` is decided by. Once the amplitude is an order of
        magnitude under that threshold the verdict patterned=False cannot flip.
      * those amplitudes are non-increasing, which rules out a non-normal transient that
        dips below `level` on its way up.

    This does not change any verdict; it reaches the same patterned=False sooner. What it
    does NOT do is decide whether an unpatterned model should count as a morphology
    mismatch — that is a metric definition, and it is left to the caller.
    """
    if sig_max > 0 or len(amps) < window:
        return False
    a = np.asarray(amps[-window:], float)
    if not np.all(np.isfinite(a)):
        return False
    return bool(np.all(a < level) and np.all(np.diff(a) <= 0.0))


def simulate(model, L, n=128, T=None, dt=None, seed=0, noise=1e-2, xstar=None,
             integrator="etdrk4", horizon_growth_times=40.0, record_kstar=True,
             max_steps=200000, early_stop=False, check_every=200,
             saturation_tol=0.01, saturation_window=5, collapse_margin=0.1):
    """Integrate d x/dt = D lap(x) + f(x) from x* + noise. Returns a result dict.

    max_steps : hard bound on the number of steps taken, whatever the horizon implies.
        A run that hits it reports stopped_reason='step_budget' — it was TRUNCATED, and
        its final field is not a statement about the model's attractor.
    early_stop : drive the integration in chunks of `check_every` steps and stop once
        either `_saturated` (the pattern stopped moving) or `_collapsed` (the field decayed
        past the point where patterned=False could flip) fires. Off by default so the
        fixed-horizon behaviour is what a caller gets unless it asks.
    saturation_tol, saturation_window : the saturation rule's two knobs. [TUNE] — see
        `_saturated` for what they mean and why they are not calibrated.
    collapse_margin : the collapse rule fires below this fraction of the amplitude
        threshold `patterned` itself is decided by. See `_collapsed`.

    Result keys added by unit 7: `nsteps_run` (steps actually taken; `nsteps` remains the
    PLANNED count), `stopped_reason` in {'horizon', 'saturated', 'collapsed',
    'step_budget', 'blew_up'},
    `horizon_rate` (the magnitude used to set T), `seconds` and `checks`. `sig_max` is now
    the SIGNED dispersion maximum — negative means the uniform state is linearly stable and
    no pattern can grow from infinitesimal noise — and is reported even when the run blew up.
    """
    if integrator not in INTEGRATORS:
        raise KeyError(f"unknown integrator '{integrator}'; have {sorted(INTEGRATORS)}")
    if early_stop and check_every < 1:
        raise ValueError(f"check_every must be >= 1 when early_stop is on; got {check_every}")
    if early_stop and saturation_window < 2:
        raise ValueError(
            f"saturation_window must be >= 2 (a span over one sample is always 0, so a "
            f"window of 1 would declare saturation at the first check); got "
            f"{saturation_window}")

    rng = np.random.default_rng(seed)
    N = model.N
    D = model.D.detach().cpu().numpy()
    if xstar is None:
        from ..losses.terms import steady_state
        xs, _ = steady_state(model); xstar = xs.detach().cpu().numpy()
    xstar = np.asarray(xstar, float).reshape(N)

    # growth-rate-aware dt and horizon
    xs_t = torch.tensor(xstar, device=model.device, dtype=model.dtype)
    Jn = model.jacobian(xs_t, create_graph=False).detach().cpu().numpy()
    kg = np.linspace(1e-3, 2 * np.pi * (n // 2) / L, 2000)
    sigd = np.array([np.max(np.real(np.linalg.eigvals(Jn - kk**2 * np.diag(D)))) for kk in kg])
    # SIGNED maximum of the dispersion relation. Positive: the fastest-growing mode's
    # growth rate. Negative: the slowest DECAY rate, i.e. the uniform state is linearly
    # stable. Both are legitimate rate scales for a horizon; flooring the negative case at
    # a fixed 1e-3 (as this did before unit 7) is not — it invents a 40000-time-unit
    # horizon for a model that relaxes in ~2.
    sig_max = float(sigd.max())
    horizon_rate = max(abs(sig_max), 1e-12)
    jac_rate = float(np.max(np.abs(np.linalg.eigvals(Jn))))
    if dt is None: dt = 0.2 / (jac_rate + 1e-9)
    if T is None:  T = horizon_growth_times / horizon_rate
    nsteps = int(np.clip(T / dt, 200, max_steps))
    hit_budget = (T / dt) > max_steps

    X = xstar[:, None, None] + noise * rng.standard_normal((N, n, n))
    reaction_np = _reaction_np_builder(model)
    step = INTEGRATORS[integrator]
    # The amplitude above which the final field is called `patterned`. Defined here so the
    # collapse rule below is expressed in the SAME units as the verdict it anticipates.
    pattern_floor = max(1e-3, 0.02 * abs(xstar[0]))

    t0 = time.perf_counter()
    blew_up = False
    done = 0
    amps: list[float] = []
    kstars: list[float] = []
    early_reason = None
    if not early_stop:
        X, blew_up = step(X, D, reaction_np, n, L, dt, nsteps)
        done = nsteps
    else:
        # Chunk re-entry is free: numerics caches the ETDRK4 coefficients on (D, n, L, dt).
        while done < nsteps:
            chunk = min(check_every, nsteps - done)
            X, blew_up = step(X, D, reaction_np, n, L, dt, chunk)
            done += chunk
            if blew_up:
                break
            amps.append(float(X[0].std()))
            kstars.append(float(obs.kstar_of(X[0], L=L)))
            if _collapsed(amps, sig_max, collapse_margin * pattern_floor, saturation_window):
                early_reason = "collapsed"
                break
            if _saturated(amps, kstars, saturation_tol, saturation_window):
                early_reason = "saturated"
                break
    seconds = time.perf_counter() - t0

    common = dict(integrator=integrator, nsteps=nsteps, nsteps_run=done, dt=float(dt),
                  sig_max=sig_max, horizon_rate=float(horizon_rate),
                  seconds=float(seconds), checks=len(amps))
    if blew_up:
        return dict(fields=X, kstar=np.nan, patterned=False, amplitude=np.nan,
                    blew_up=True, stopped_reason="blew_up", **common)

    amp = float(X[0].std())
    ks = obs.kstar_of(X[0], L=L) if (record_kstar and amp > 1e-6) else np.nan
    patterned = amp > pattern_floor
    stopped_reason = (early_reason if early_reason is not None else
                      "step_budget" if hit_budget else "horizon")
    return dict(fields=X, kstar=ks, patterned=bool(patterned), amplitude=amp,
                blew_up=False, stopped_reason=stopped_reason, **common)
