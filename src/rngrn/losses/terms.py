"""
rngrn_objective.py
==================
Composite objective for single-frame RNGRN recovery. Reference implementation.

Terms (see rngrn_design_doc.md Section 5). Every term is differentiable w.r.t. the model's raw
parameters theta. Nothing here reads any analytic quantity of a ground-truth system: the steady
state is the MODEL's own, k* is the FFT MEASUREMENT of the frame, the Turing conditions are
inequalities on the model's own J and D. This is the Section-12 firewall in code.

  steady_state          f(x*)=0 by damped Newton with relaxation fallback (never the frame mean)
  turing_hinges         softplus hinges on the model's Turing conditions at x* (general-N via dispersion)
  turing_hinges_split   the same two conditions on DISJOINT k-support (the promoted default)
  frame_scale_anchor    log-scale anchor of x* to the frame's own mean intensity
  kstar_anchor          soft tolerance-band penalty |k*_model - k*_FFT| / k*_FFT
  stationarity_residual full RHS D lap(x) + f(x) = 0 on observed channels (latent inferred if m<N)
  anticollapse          margin penalty excluding the f==0, D==0 trivial minimum
  morphology_consistency weak regulariser matching simulated to observed morphology (optional)
  param_prior           soft biological-plausibility prior (D-ratio log-normal + box hinges,
                         unit 5) — reads only configs/bio_box.yaml and the model's own
                         parameters, never an answer key
  composite_loss        weighted sum + a dict of the individual parts for logging
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from .. import observables as obs

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------------------
# 5.5  model-derived homogeneous steady state
# --------------------------------------------------------------------------------------
def _damped_newton(model, x0, tol, max_iter):
    """The damped-Newton core with the positivity guard. ARITHMETIC IS THE LEGACY ONE,
    lifted verbatim out of `steady_state` so it can be re-run from more than one seed;
    do not "improve" the line search here — `steady_state`'s bit-identity guarantee
    rests on this being unchanged. Returns (last iterate, converged)."""
    x = x0.clone()
    for _ in range(max_iter):
        fx = model.reaction(x)
        if torch.linalg.norm(fx).item() < tol:
            return x, True
        J = model.jacobian(x, create_graph=False)
        try:
            step = torch.linalg.solve(J, fx)
        except RuntimeError:
            return x, False
        # damped Newton with positivity guard
        lam = 1.0
        for _ in range(30):
            xn = x - lam * step
            if (xn > 0).all() and torch.linalg.norm(model.reaction(xn)).item() < torch.linalg.norm(fx).item():
                break
            lam *= 0.5
        x = torch.clamp(x - lam * step, min=1e-9)
    return x, False


def steady_state_bracket(model):
    """Componentwise bracket [lo, hi] containing EVERY positive root of f (both forms).

    At any root, x_i = (beta_i + prod_i(x)) / delta_i, and the production term is bounded
    0 <= prod_i <= sum_j alpha_ij:

      nc1          prod_i = (sum_j alpha_ij thetaA_ij) * prod_j (1 - thetaR_ij), and both
                   thetaA_ij in [0,1) and the veto product in (0,1].
      competitive  prod_i = sum_j alpha_ij KA_ij x_j^n / (1 + sum_j (KA_ij+KR_ij) x_j^n),
                   a weighted average of the alpha_ij with weights summing to < 1.

    Hence lo_i = beta_i/delta_i <= x*_i <= (beta_i + sum_j alpha_ij)/delta_i = hi_i.
    lo is exactly the steady state with production switched OFF, hi with it SATURATED.

    Reads only the model's own parameters -- no frame, no answer key. Returns (lo, hi).
    """
    lo = model.beta / model.delta
    hi = (model.beta + model.alpha.sum(dim=-1)) / model.delta
    return lo.detach(), hi.detach()


def steady_state(model, x0=None, tol=1e-10, max_iter=100, relax_steps=2000, relax_dt=1e-2,
                 multistart=True):
    """Solve f(x*) = 0 for the model's OWN homogeneous steady state.

    Damped Newton on the reaction; if it fails (early training / wild theta) fall back to a short
    forward relaxation of dx/dt = f(x) from a positive init. NEVER returns the frame mean.

    Returns (xstar detached-safe tensor of shape (N,), converged: bool).
    The returned xstar carries grad (Newton is differentiable through the reaction), so physics
    terms evaluated at xstar are differentiable w.r.t. theta.

    MULTISTART (unit B3, default True) -- a SOLVER fix, not a model or objective change.
    The single fixed seed x0 = ones is a GLOBALISATION failure: measured on a real nc1
    training trajectory (two_gene_classical_val/sample_0000, seed 0, restart 0, Adam step
    880), the Newton iterate from ones is trapped on the fold det J = 0 at
    x = [0.7456, 0.7464], where |f| = 1.68e-2, sigma_min(J) = 1.70e-6 and cond(J) = 1.28e6,
    so the Newton direction explodes (|step| 100 -> 5e3) and the line search collapses to
    lam ~ 1e-9. The relaxation fallback restarts from the SAME seed and crawls in the same
    near-fold bottleneck, ending at |f| = 1.3e-2 >> 1e-4. Yet the reaction has EXACTLY ONE
    positive root, x* = [0.06111, 0.22282], which is well-conditioned (cond 1.98) and stable
    (eig J = -0.703, -1.355): Newton from beta/delta reaches it to |f| = 2.0e-15. The old
    code therefore reported "no steady state" for a model whose steady state exists and is
    benign, and recover.py's fail-loud then threw the whole restart away.

    So: attempt 1 is the LEGACY path verbatim (Newton from x0, then the relaxation fallback
    from x0). Only if that fails do we re-run Newton from the bracket seeds of
    `steady_state_bracket` (lo, hi and their geometric mean). Consequence, and the reason
    this is safe to have on by default: ANY call that converged before converges to the
    BIT-IDENTICAL xstar now, because the first attempt is byte-for-byte the old algorithm
    and short-circuits on success (pinned by tests/test_science.py). What changes is only
    the previously-unconverged cases, which used to raise SteadyStateError and lose the
    restart. `multistart=False` restores the exact legacy behaviour for reproducing
    pre-B3 numbers.
    """
    N = model.N
    if x0 is None:
        x0 = torch.ones(N, device=model.device, dtype=model.dtype)
    # ---- attempt 1: the legacy path, unchanged ----------------------------------------
    x, converged = _damped_newton(model, x0, tol, max_iter)
    if converged:
        return x, True
    # relaxation fallback
    x = (x0.clone() if (x0 > 0).all() else torch.ones(N, device=model.device, dtype=model.dtype))
    dt = relax_dt
    for _ in range(relax_steps):
        x = torch.clamp(x + dt * model.reaction(x), min=1e-9)
    converged = torch.linalg.norm(model.reaction(x)).item() < 1e-4
    if converged or not multistart:
        return x, converged
    # ---- attempt 2..4: Newton from the analytic bracket (unit B3) ----------------------
    lo, hi = steady_state_bracket(model)
    for seed in (lo, hi, torch.sqrt(lo * hi)):
        xs, ok = _damped_newton(model, seed, tol, max_iter)
        if ok:
            return xs, True
    # nothing found: return the legacy fallback state and its (False) verdict, so a caller
    # that inspects the state sees exactly what the pre-B3 code would have handed it.
    return x, False


def steady_state_diff(model, xstar_init):
    """Re-solve one differentiable Newton polish step from a converged xstar so the returned
    state carries gradient w.r.t. theta (implicit-function differentiation, one Newton step is
    enough at the root). Used when the physics terms need d xstar / d theta."""
    x = xstar_init.detach()
    fx = model.reaction(x)
    J = model.jacobian(x, create_graph=True)
    xstar = x - torch.linalg.solve(J, fx)   # differentiable in theta via fx and J
    return xstar


# --------------------------------------------------------------------------------------
# 5.2  Turing-regime hinges
# --------------------------------------------------------------------------------------
def _softplus_hinge(violation, beta=10.0):
    """Zero when the condition holds with margin; smooth positive as it is approached/violated.
    `violation` should be > 0 when the condition is VIOLATED."""
    return F.softplus(violation * beta) / beta


def turing_hinges(model, xstar, kgrid, margin=1e-3):
    """SUPERSEDED shared-support hinges. Kept because docs and scripts/exp02 reference it as
    the control arm; `turing_hinges_split` is the default in losses/total.compute_terms.

    Softplus hinges on the Turing conditions. General-N: uses the differentiable dispersion,
    requiring sigma(0) < 0 (uniform-stable) and max_k sigma(k) > 0 (structured-unstable).

    KNOWN DEFECT (measured, scripts/exp02, 60/60 random inits): argmax_k sigma(k) IS the
    kgrid[0] point at random init, so `sig.max()` and `sig[0]` are the SAME scalar and the
    two hinges push it in opposite directions. See docs/STATE_OF_THE_SCIENCE.md 2.2.
    """
    J = model.jacobian(xstar, create_graph=True)
    sig = model.dispersion(xstar, kgrid, J=J)              # (K,)
    # sigma at (near) k=0: stability to uniform perturbations -> want sig0 < 0
    sig0 = sig[0]
    # max growth over the WHOLE grid (this is the defect: it can be sig0 itself)
    sig_max_pos = sig.max()
    L_uniform = _softplus_hinge(sig0 + margin)             # penalise sig0 >= -margin
    L_unstable = _softplus_hinge(-(sig_max_pos - margin))  # penalise sig_max <= margin
    return L_uniform + L_unstable, dict(sig0=float(sig0.detach()),
                                        sig_max=float(sig_max_pos.detach()))


def turing_hinges_split(model, xstar, kgrid, margin=1e-3, k_min_frac=0.1):
    """Turing hinges on DISJOINT k-support. The promoted default (see total.compute_terms).

    Same two conditions as `turing_hinges`, but the instability hinge maximises only over
    k >= kgrid[i_min] with i_min = max(1, int(k_min_frac * len(kgrid))), so it can never be
    evaluated at the same grid point as the uniform-stability hinge. One scalar therefore
    never receives opposing gradients.

    PROVENANCE: ported verbatim from scripts/exp02_objective_fix.py::turing_hinges_split
    (identical in scripts/exp05_pixel_minibatch.py::split_hinges). Measured there over
    40 seeds x 400 steps, N=3, WITH the frame-scale anchor: 38/40 converged, 36.8 % Turing,
    against 37/40 converged and 0 % Turing for the shared-support form. Neither fix works
    alone (docs/STATE_OF_THE_SCIENCE.md 2.1). That measurement was made by the experiment
    scripts, not by this function; this is the promotion of the same arithmetic.

    TWO GRID DIFFERENCES FROM THE EXPERIMENTS, both real, neither measured for its effect:
      * kgrid[0] is TREATED AS the uniform (k=0) mode, but recover._kgrid_for starts the
        grid at kstar_obs/50 rather than exactly 0 (the experiments used exactly 0), so
        sig0 here is sigma(k*_obs/50). The gap is O(D k^2) at that k.
      * `k_min_frac` is a fraction of the GRID INDEX, not of k*, so where the instability
        hinge actually starts depends on the grid's span. Measured on three_gene_val
        sample_0000 (kstar_obs = 0.4320): recover._kgrid_for puts k_min at
        0.822 * kstar_obs, the exp05 grid at 0.698 * kstar_obs. The library therefore
        excludes a wider band below k* than the experiments did. Whether the floor should
        instead be defined relative to kstar_obs is an open call, deliberately NOT made
        here — see TUNING.md.
    """
    K = len(kgrid)
    i_min = max(1, int(k_min_frac * K))
    if i_min >= K:
        raise ValueError(
            f"k_min_frac={k_min_frac} leaves no k>=k_min grid points (i_min={i_min}, K={K}); "
            "the instability hinge would have empty support")
    J = model.jacobian(xstar, create_graph=True)
    sig = model.dispersion(xstar, kgrid, J=J)              # (K,)
    sig0 = sig[0]
    sig_pos = sig[i_min:].max()                            # strictly k > kgrid[0]
    L_uniform = _softplus_hinge(sig0 + margin)             # penalise sig0 >= -margin
    L_unstable = _softplus_hinge(-(sig_pos - margin))      # penalise sig_pos <= margin
    # `sig_max` keeps its ORIGINAL meaning (max over the whole grid) so the diagnostic
    # logged by recover.py/restart_log does not silently change definition; `sig_max_pos`
    # is the new quantity this hinge actually uses.
    return L_uniform + L_unstable, dict(sig0=float(sig0.detach()),
                                        sig_max=float(sig.max().detach()),
                                        sig_max_pos=float(sig_pos.detach()),
                                        hinge_i_min=int(i_min))


# --------------------------------------------------------------------------------------
# frame-scale anchor
# --------------------------------------------------------------------------------------
def frame_scale_anchor(xstar, obs_scale, floor=1e-6):
    """Anchor the model's steady state to the frame's own mean intensity, in log space:

        L = mean_i ( log(obs_scale) - log(x*_i) )^2

    FIREWALL: `obs_scale` must be `frame.mean()` — a statistic of the OBSERVED image. No
    ground-truth x* is involved. Callers on the recovery side pass the frame they already
    have; nothing else is legal here.

    WHAT IT IS AND IS NOT. exp12 measured Spearman rho = +0.950 between `frame.mean()` and
    the true x*_0, with the ratio median 0.921 — i.e. frame.mean() is a BIASED but strongly
    monotone image-only estimator of x*_0, not an unbiased measurement of it. This term is
    therefore a SCALE REGULARISER, not a fit to a known value: its job is to keep x* in the
    sub-saturated Hill regime where 1 + KA x^n < n can hold, not to pin x* to a truth.

    PROVENANCE: ported from scripts/exp05_pixel_minibatch.py::fit (line 86) and
    scripts/exp02_objective_fix.py. Measured in exp02, 40 seeds x 400 steps, N=3: adding it
    took convergence from 2/40 to 38/40 (with split hinges). Its weight (2.0 in the
    experiments) is supplied by the weighting strategy here, NOT baked in, and has never
    been swept — see TUNING.md.

    Returns (scalar loss, parts). The zero of this term is x*_i == obs_scale for every i.
    """
    s = float(obs_scale)
    if not (s > 0.0) or not np.isfinite(s):
        # fail loud: log of a non-positive / non-finite scale is meaningless, and a silent
        # clamp here would turn a broken frame into a plausible-looking loss.
        raise ValueError(f"frame_scale_anchor: obs_scale must be finite and > 0, got {obs_scale!r}")
    d = np.log(s) - torch.log(xstar.clamp_min(floor))
    L = (d ** 2).mean()
    return L, dict(obs_scale=s, xstar_mean=float(xstar.detach().mean()))


# --------------------------------------------------------------------------------------
# 5.1  k* soft-anchor (tolerance band)
# --------------------------------------------------------------------------------------
def _sigma_at(sig, kgrid, k):
    """Differentiable linear interpolation of sigma(kgrid) at scalar k.

    `device=kgrid.device` is load-bearing, not defensive (unit b2): without it this line
    built the search key on the CPU while `kgrid` was on the GPU, and torch.searchsorted
    raised "Expected all tensors to be on the same device". That made the ENTIRE recovery
    objective unrunnable on CUDA -- measured: recover(device='cuda') crashed here on the
    first step, on the serial path, before unit b2 existed. Fixing it changes no CPU value
    (the index and the interpolation are identical); it only removes a hard crash.
    """
    idx = torch.searchsorted(kgrid, torch.as_tensor(float(k), device=kgrid.device)
                             ).clamp(1, len(kgrid) - 1)
    k0, k1 = kgrid[idx - 1], kgrid[idx]
    s0, s1 = sig[idx - 1], sig[idx]
    t = (k - k0) / (k1 - k0 + 1e-12)
    return s0 + t * (s1 - s0)


def kstar_anchor(model, xstar, kgrid, kstar_obs, tau=0.12, temp=60.0):
    """Make the OBSERVED wavenumber the dominant (fastest-growing) mode of the model's dispersion.

    Rather than estimate a k*_model by a (broad-band-unreliable) soft-argmax and match it, penalise
    directly the gap between the dispersion's soft-maximum and its value at k*_obs:

        L = logsumexp(temp * sigma) / temp  -  sigma(k*_obs)

    This is >= 0 and zero exactly when k*_obs is the argmax of sigma. A broad, weakly-selective band
    yields a small penalty (honest: selection is weak), a sharp peak away from k*_obs a large one.
    The true argmax is reported (detached) as kstar_model for logging/scoring.
    """
    sig = model.dispersion(xstar, kgrid, J=None)
    lse = torch.logsumexp(sig * temp, dim=0) / temp        # smooth max of sigma
    sig_obs = _sigma_at(sig, kgrid, kstar_obs)
    L = torch.clamp(lse - sig_obs, min=0.0)
    kstar_model = float(kgrid[torch.argmax(sig)].detach())
    r = abs(kstar_model - kstar_obs) / (kstar_obs + 1e-9)
    return L, dict(kstar_model=kstar_model, kstar_obs=float(kstar_obs), rel_err=float(r))


# --------------------------------------------------------------------------------------
# 5.3  full-RHS stationarity residual
# --------------------------------------------------------------------------------------
def stationarity_residual(model, fields, L, observed_idx, latent_fields=None):
    """mean [ D_i lap(x_i) + f_i(x) ]^2 over observed channels & all grid points.

    fields: (m, H, W) observed channels (torch). observed_idx: list mapping the m rows to model
    indices. latent_fields: (N-m, H, W) inferred fields for the unobserved channels, or None when
    m == N. f is evaluated on the FULL reconstructed state vector at every pixel.
    """
    N = model.N
    m, H, W = fields.shape
    # assemble full state (N, H, W)
    full = torch.zeros(N, H, W, dtype=fields.dtype, device=fields.device)
    for row, idx in enumerate(observed_idx):
        full[idx] = fields[row]
    if latent_fields is not None:
        lat_idx = [i for i in range(N) if i not in observed_idx]
        for row, idx in enumerate(lat_idx):
            full[idx] = latent_fields[row]
    # pointwise reaction over all pixels: reshape to (P, N)
    x = full.permute(1, 2, 0).reshape(-1, N)               # (P, N)
    fx = model.reaction(x).reshape(H, W, N).permute(2, 0, 1)  # (N,H,W)
    lap = obs.laplacian_torch(full, L=L)                   # (N,H,W)
    D = model.D.reshape(N, 1, 1)
    rhs = D * lap + fx                                     # (N,H,W)
    # residual on observed channels (latent channels enforced separately if desired)
    Dlap = D * lap
    # RELATIVE residual: dimensionless, O(1) by construction. Normalise the squared full-RHS
    # residual by the mean square magnitude of its own components (diffusion + reaction), so the
    # term measures "how far from cancelling" on a 0..~1 scale rather than in raw rate units.
    def _rel(idx):
        num = (rhs[idx] ** 2).mean()
        den = (Dlap[idx] ** 2).mean() + (fx[idx] ** 2).mean() + 1e-9
        return num / den
    loss = _rel(observed_idx)
    parts = dict(resid=float(loss.detach()))
    if latent_fields is not None:
        lat_idx = [i for i in range(N) if i not in observed_idx]
        loss_lat = _rel(lat_idx)
        loss = loss + loss_lat
        parts['resid_latent'] = float(loss_lat.detach())
    return loss, parts


# --------------------------------------------------------------------------------------
# 5.4  anti-collapse
# --------------------------------------------------------------------------------------
def anticollapse(model, xstar, jac_floor=1.0, amp_floor=None, sim_field=None):
    """Exclude the f==0, D==0 trivial minimum. Default: keep the Jacobian Frobenius norm above a
    floor (the reaction must have non-trivial local dynamics). If a lifted-simulation field is
    supplied, additionally floor its amplitude so a flat relaxation is penalised."""
    J = model.jacobian(xstar, create_graph=True)
    jn = torch.linalg.norm(J)
    L = _softplus_hinge(jac_floor - jn)                    # penalise ||J|| < floor
    parts = dict(jac_norm=float(jn.detach()))
    if sim_field is not None and amp_floor is not None:
        amp = sim_field.std()
        L = L + _softplus_hinge(amp_floor - amp)
        parts['sim_amp'] = float(amp.detach() if torch.is_tensor(amp) else amp)
    return L, parts


# --------------------------------------------------------------------------------------
# 5.6  morphology consistency (weak, optional)
# --------------------------------------------------------------------------------------
def morphology_consistency(sim_field_np, obs_field_np):
    """Match simulated to observed morphology (skew sign + angular concentration). numpy in,
    scalar out. Weak regulariser; not differentiable (used as a post-hoc penalty / diagnostic)."""
    ms = obs.morphology(sim_field_np); mo = obs.morphology(obs_field_np)
    d = (np.sign(ms['skew']) - np.sign(mo['skew'])) ** 2 + (ms['ang_conc'] - mo['ang_conc']) ** 2
    return float(d), dict(sim=ms, obs=mo)


# --------------------------------------------------------------------------------------
# 5.7  biological-plausibility soft prior (unit 5)
# --------------------------------------------------------------------------------------
# NOTE ON THE DUPLICATED BOX LOADER: rngrn.scoring.plausibility (the SCORING-side
# module, which offers `load_box`/`BoxRow`/`d_ratio_of`) cannot be imported here.
# tests/test_permutation_scoring.py, test_morphology_scoring.py and
# test_overparam_scoring.py each assert that NO recovery-side module (this file
# included) imports `rngrn.scoring` at all — the scoring package as a whole is
# treated as answer-key-adjacent, not just the specific forbidden names in
# tests/test_firewall.py. So this file loads configs/bio_box.yaml independently
# below, rather than sharing scoring/plausibility.py's loader. The definitions are
# kept intentionally tiny and are exercised by tests/test_plausibility.py against
# BOTH this loader and the scoring-side one to catch drift between the two.
def _load_box_bounds(path):
    """Read configs/bio_box.yaml -> {name: (low, high) or None}. None means the row is
    UNCITED or carries no bounds — never a fabricated (low, high) pair."""
    import yaml
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    out = {}
    for name, row in raw.items():
        if not isinstance(row, dict) or "source" not in row:
            raise ValueError(f"{path}: row {name!r} must be a mapping with a `source`")
        low, high = row.get("low"), row.get("high")
        cited = row["source"] != "UNCITED"
        out[name] = (float(low), float(high)) if (cited and low is not None and high is not None) else None
    return out


def param_prior(model, dratio_centre=7.5, dratio_spread=1.0, box=None,
                box_path="configs/bio_box.yaml"):
    """Soft biological-plausibility prior on the model's OWN parameters. Recovery-side
    (reads no answer-key quantity; tests/test_firewall.py covers this file).

    Two summed components:

    1. Log-normal D-ratio term: w * (log D_ratio - log dratio_centre)**2 / (2 *
       dratio_spread**2), where D_ratio is the two MOST-MOBILE species' diffusivity
       ratio (largest / second-largest — see scoring/plausibility.py::d_ratio_of for
       the same definition used at scoring time: it excludes the single smallest D by
       construction, so a near-immobile node, docs/ROBUSTNESS_MEASUREMENT.md §4.4, is
       never penalised).
       `dratio_centre` default 7.5 is the LITERATURE value (measured Nodal/Lefty
       ratio) per the user decision on record: priors are centred on biologically
       viable literature values, NOT on the synthetic generators' own draws (~8-250,
       median ~135) — see configs/bio_box.yaml and docs/STATE_OF_THE_SCIENCE.md §11.
    2. Soft box hinges (one-sided softplus) pulling `alpha` and `delta` back toward
       [low, high] from configs/bio_box.yaml whenever an entry strays outside. `beta`
       has no cited box there (UNCITED) and is deliberately NOT hinged — hinging an
       invented bound would fabricate a provenance the box explicitly refuses to claim.

    The RETURNED loss is UNWEIGHTED; the caller applies loss.weights.param_prior
    (config.py, default 0.0 — this term is opt-in, its effect measurable rather than
    assumed).

    Returns (loss: 0-d torch tensor, parts: dict of floats for logging).
    """
    box = _load_box_bounds(box_path) if box is None else box

    D = model.D
    sorted_D, _ = torch.sort(D)
    if sorted_D.numel() < 2:
        raise ValueError(f"param_prior needs N>=2 species, got {sorted_D.numel()}")
    lo, hi = sorted_D[-2], sorted_D[-1]
    log_ratio = torch.log(hi) - torch.log(lo)
    log_centre = float(np.log(dratio_centre))
    L_dratio = (log_ratio - log_centre) ** 2 / (2.0 * dratio_spread ** 2)

    L_box = torch.zeros((), dtype=D.dtype, device=D.device)
    parts = dict(d_ratio=float((hi / lo).detach()), L_dratio=float(L_dratio.detach()))
    for name, value in (("alpha", model.alpha), ("delta", model.delta)):
        bounds = box.get(name)
        if bounds is None:
            continue  # UNCITED or unbounded row: no fabricated hinge (bio_box.yaml)
        low, high = bounds
        term = (_softplus_hinge(low - value) + _softplus_hinge(value - high)).sum()
        L_box = L_box + term
        parts[f"L_box_{name}"] = float(term.detach())

    loss = L_dratio + L_box
    parts["L_param_prior"] = float(loss.detach())
    return loss, parts


# ======================================================================================
# unit b2 — BATCHED term variants: B independent members, one set of tensor ops
# ======================================================================================
# Every function below is the SAME arithmetic as its serial namesake with a leading batch
# dimension B, and returns a (B,) tensor instead of a 0-d one. Diagnostics come back as
# (B,) numpy arrays under the SAME key names, so losses/total.parts_member(parts, b) can
# reconstruct exactly the per-member dict the serial path logs.
#
# NOT batched, deliberately: `stationarity_residual` (needs per-pixel states, which would
# require broadcasting the parameters to (B,1,N,N); its default weight is 0 and
# total.compute_terms_batched raises rather than pretend) and `morphology_consistency`
# (non-differentiable numpy diagnostic). `param_prior` IS batched, as `param_prior_batched`
# below: it reads only model.D / .alpha / .delta, all of which carry the leading B.

def steady_state_batched(model, x0=None, tol=1e-10, max_iter=100,
                         relax_steps=2000, relax_dt=1e-2, multistart=True):
    """Batched damped Newton for f(x*)=0 on B independent members. Returns (x*, converged).

    x*: (B, N) DETACHED. converged: (B,) bool.

    A LINE-BY-LINE translation of the serial `steady_state`, not a re-derivation:

      * convergence is tested at the TOP of each iteration, exactly as serially, and a
        member that passes is FROZEN (masked out) rather than returned early. Its x* is the
        x it held at that check -- the same tensor the serial call would have returned.
      * the damped line search halves lambda per member and stops at the FIRST accepted
        lambda; a member that never accepts in 30 halvings ends on lambda = 0.5**30, which
        is what the serial loop does when its inner `for` runs to completion.
      * a singular Jacobian is caught per member via torch.linalg.solve_ex's info code
        (torch.linalg.solve would raise for the WHOLE batch because one member went bad),
        mirroring the serial `except RuntimeError: break` -> relaxation fallback.
      * FAIL-LOUD BECOMES PER MEMBER. The serial contract is that losses/total raises
        SteadyStateError when x* does not converge. Here one diverged member must not abort
        the other B-1, so non-convergence is returned as a per-member FLAG. It is still
        never silently ignored: recover's batched loop kills that member for good and logs
        it as steady_state_failed, exactly as a serial restart would have been abandoned.

    TWO HONEST DIFFERENCES, both value-preserving:
      * the whole solve runs under no_grad. Serially x* carries a graph that
        `steady_state_diff` immediately throws away by detaching, so the graph is pure cost.
      * the relaxation fallback runs on the FULL batch whenever ANY member needs it, and the
        results are then selected by mask. 2000 relaxation steps for one straggler is the
        price of not indexing a sub-batch; it only happens on inits Newton could not solve.
    """
    B, N = model.B, model.N
    dev, dt_ = model.device, model.dtype
    if x0 is None:
        x0 = torch.ones(B, N, device=dev, dtype=dt_)

    def _attempt(x_init, relax=True):
        """Damped Newton from x_init, then optionally the relaxation fallback.

        Returns (x (B,N), converged (B,)). With relax=True this is verbatim the
        pre-multistart body, which is what makes attempt 1 bit-identical to the legacy
        solver. relax=False is Newton only — used for the bracket attempts, see below."""
        x = x_init.clone()
        active = torch.ones(B, dtype=torch.bool, device=dev)
        broke = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_iter):
            fx = model.reaction(x)
            nrm = torch.linalg.norm(fx, dim=-1)                  # (B,)
            active = active & (nrm >= tol)
            if not bool(active.any()):
                break
            J = model.jacobian(x, create_graph=False)
            step, info = torch.linalg.solve_ex(J, fx.unsqueeze(-1))
            step = step.squeeze(-1)
            bad = (info != 0) & active
            if bool(bad.any()):
                broke = broke | bad
                active = active & ~bad
                step = torch.where(bad.unsqueeze(-1), torch.zeros_like(step), step)
                if not bool(active.any()):
                    break
            lam = torch.ones(B, device=dev, dtype=dt_)
            accept = torch.zeros(B, dtype=torch.bool, device=dev)
            for _ in range(30):
                xn = x - lam.unsqueeze(-1) * step
                ok = (xn > 0).all(dim=-1) & (
                    torch.linalg.norm(model.reaction(xn), dim=-1) < nrm)
                accept = accept | ok
                lam = torch.where(accept, lam, lam * 0.5)
                if bool((accept | ~active).all()):
                    break
            x_new = torch.clamp(x - lam.unsqueeze(-1) * step, min=1e-9)
            x = torch.where(active.unsqueeze(-1), x_new, x)

        needs_relax = (active | broke) & (True if relax else False)
        converged = ~(active | broke)
        if relax and bool(needs_relax.any()):
            xr = torch.where((x_init > 0).all(dim=-1, keepdim=True), x_init,
                             torch.ones_like(x_init))
            # EARLY EXIT, and it is worth 100x mid-training. The relaxation runs on the
            # whole batch (indexing out a sub-batch would need a sub-model, since the
            # parameters are per-member), and it used to run all `relax_steps` of them
            # unconditionally. That was tolerable when the fallback was rare, but once
            # multistart made it reachable up to four times per call it dominated
            # everything: MEASURED on CUDA at B=64, a training step cost 110-170 ms for
            # the first ~25 Adam steps and then 10,500-17,600 ms once the parameters
            # drifted into a region where attempt 1 fails for a few members -- about 66
            # minutes for a 400-step run. Checking convergence periodically and stopping
            # when every member that NEEDS the relaxation has converged is value-preserving
            # (it only ever stops once the answer has stopped changing for those members)
            # and removes the blowup.
            check_every = max(1, int(relax_steps // 40))
            for i in range(relax_steps):
                xr = torch.clamp(xr + relax_dt * model.reaction(xr), min=1e-9)
                if (i + 1) % check_every == 0:
                    done = torch.linalg.norm(model.reaction(xr), dim=-1) < 1e-4
                    if bool((done | ~needs_relax).all()):
                        break
            conv_r = torch.linalg.norm(model.reaction(xr), dim=-1) < 1e-4
            x = torch.where(needs_relax.unsqueeze(-1), xr, x)
            converged = torch.where(needs_relax, conv_r, converged)
        return x, converged

    with torch.no_grad():
        # ORDER: Newton(x0) -> Newton(bracket) -> relaxation, LAST RESORT ONLY.
        #
        # The legacy order was Newton(x0) -> relaxation -> Newton(bracket). Keeping that
        # here made batched training 1.3 s/step and rising on real data (MEASURED: 18.7 s
        # at 20 steps, 124.1 s at 100), because the relaxation is 2000 whole-batch reaction
        # evaluations and fires often mid-training once the parameters are hard --
        # especially with d_init_from_kstar, which starts D near 1/k*^2.
        #
        # Reordered because the relaxation is both the MOST expensive step and the LEAST
        # accurate one: it accepts at |f| < 1e-4, where Newton from the analytic bracket
        # reaches |f| ~ 1e-15 (B3 measured 2.0e-15 from beta/delta on the case that
        # defeated x0 = ones). Trying the cheap, tighter solver before the expensive, looser
        # one is strictly better on both axes.
        #
        # WHAT THIS CHANGES, stated because it is not nothing: a member that BOTH the
        # bracket and the relaxation could solve now returns the bracket's root instead of
        # the relaxation's. That is a tighter root, not a different steady state - both
        # solve f(x*) = 0 for the same reaction - but it is not bit-identical to the legacy
        # value, so a run whose x* came from the relaxation fallback will differ in the last
        # digits. Members solved by Newton(x0) - the overwhelming majority, and every
        # member at random init - are untouched. The relaxation still runs for anything
        # neither Newton pass can solve, so the CONVERGENCE SET is unchanged or larger,
        # never smaller. Recorded in docs/DECISIONS.md.
        # multistart=False is the LEGACY solver exactly: Newton(x0) then the relaxation,
        # nothing else. Kept bit-for-bit so pre-B3 numbers can be reproduced on demand and
        # so a test can still exhibit a genuine per-member failure.
        if not multistart:
            return _attempt(x0, relax=True)

        x, converged = _attempt(x0, relax=False)
        if not bool(converged.all()):
            lo0, hi0 = steady_state_bracket(model)
            for seed in (lo0, hi0, torch.sqrt(lo0 * hi0)):
                xs, ok = _attempt(seed.expand_as(x0).clone(), relax=False)
                take = ok & ~converged
                if bool(take.any()):
                    x = torch.where(take.unsqueeze(-1), xs, x)
                    converged = converged | take
                if bool(converged.all()):
                    break
        if not bool(converged.all()):
            xs, ok = _attempt(x0, relax=True)      # the legacy relaxation, last resort
            take = ok & ~converged
            if bool(take.any()):
                x = torch.where(take.unsqueeze(-1), xs, x)
                converged = converged | take
    return x, converged


def steady_state_diff_batched(model, xstar_init):
    """Batched one-step differentiable Newton polish. Returns (x* (B,N), ok (B,) bool).

    Same arithmetic as the serial `steady_state_diff`, with one addition forced by batching:
    the serial version calls torch.linalg.solve, which RAISES on a singular J. Batched, that
    raise would take down every member because of one. `solve_ex` gives a per-member info
    code instead; a member whose J is singular gets a ZERO polish step and ok=False, and the
    caller must fold that into its non-convergence mask. It is not a silent fallback: a
    member with no invertible Jacobian at its own root has no usable differentiable steady
    state, which is precisely the condition the serial path raises on."""
    x = xstar_init.detach()
    fx = model.reaction(x)
    J = model.jacobian(x, create_graph=True)
    step, info = torch.linalg.solve_ex(J, fx.unsqueeze(-1))
    ok = info == 0
    step = torch.where(ok.reshape(-1, 1, 1), step, torch.zeros_like(step))
    return x - step.squeeze(-1), ok


def turing_hinges_batched(model, xstar, kgrid, margin=1e-3):
    """Batched SUPERSEDED shared-support hinges (control arm). Returns ((B,), parts)."""
    J = model.jacobian(xstar, create_graph=True)
    sig = model.dispersion(xstar, kgrid, J=J)                # (B,K)
    sig0 = sig[:, 0]
    sig_max_pos = sig.max(dim=1).values
    L = _softplus_hinge(sig0 + margin) + _softplus_hinge(-(sig_max_pos - margin))
    return L, dict(sig0=_np(sig0), sig_max=_np(sig_max_pos))


def turing_hinges_split_batched(model, xstar, kgrid, margin=1e-3, k_min_frac=0.1):
    """Batched disjoint-support Turing hinges (the promoted default). Returns ((B,), parts).

    Identical to `turing_hinges_split` term for term; `i_min` depends only on the k-grid,
    which is shared across members, so it is one integer for the whole batch."""
    K = len(kgrid)
    i_min = max(1, int(k_min_frac * K))
    if i_min >= K:
        raise ValueError(
            f"k_min_frac={k_min_frac} leaves no k>=k_min grid points (i_min={i_min}, K={K}); "
            "the instability hinge would have empty support")
    J = model.jacobian(xstar, create_graph=True)
    sig = model.dispersion(xstar, kgrid, J=J)                # (B,K)
    sig0 = sig[:, 0]
    sig_pos = sig[:, i_min:].max(dim=1).values
    L = _softplus_hinge(sig0 + margin) + _softplus_hinge(-(sig_pos - margin))
    return L, dict(sig0=_np(sig0), sig_max=_np(sig.max(dim=1).values),
                   sig_max_pos=_np(sig_pos), hinge_i_min=int(i_min))


def param_prior_batched(model, dratio_centre=7.5, dratio_spread=1.0, box=None,
                        box_path="configs/bio_box.yaml"):
    """Batched twin of `param_prior`. Returns ((B,), parts of (B,) numpy arrays).

    Term for term the same arithmetic as the serial version: the D-ratio log-normal on the
    two MOST-MOBILE species (so a near-immobile node is never penalised) plus the soft box
    hinges on alpha and delta. `beta` is UNCITED in configs/bio_box.yaml and is not hinged
    here either. The only change is that every reduction stops at the leading batch
    dimension instead of collapsing to a scalar.
    """
    box = _load_box_bounds(box_path) if box is None else box
    D = model.D                                          # (B, N)
    if D.shape[-1] < 2:
        raise ValueError(f"param_prior needs N>=2 species, got {D.shape[-1]}")
    sorted_D, _ = torch.sort(D, dim=-1)
    lo, hi = sorted_D[..., -2], sorted_D[..., -1]        # (B,)
    log_ratio = torch.log(hi) - torch.log(lo)
    L_dratio = (log_ratio - float(np.log(dratio_centre))) ** 2 / (2.0 * dratio_spread ** 2)

    L_box = torch.zeros_like(L_dratio)
    parts = dict(d_ratio=_np(hi / lo), L_dratio=_np(L_dratio))
    for name, value in (("alpha", model.alpha), ("delta", model.delta)):
        bounds = box.get(name)
        if bounds is None:
            continue  # UNCITED or unbounded row: no fabricated hinge (bio_box.yaml)
        low, high = bounds
        term = _softplus_hinge(low - value) + _softplus_hinge(value - high)
        term = term.reshape(term.shape[0], -1).sum(dim=-1)      # (B,)
        L_box = L_box + term
        parts[f"L_box_{name}"] = _np(term)

    loss = L_dratio + L_box
    parts["L_param_prior"] = _np(loss)
    return loss, parts


def frame_scale_anchor_batched(xstar, obs_scale, floor=1e-6):
    """Batched log-scale anchor. xstar (B,N), obs_scale a SCALAR frame statistic shared by
    every member (all members fit the same frame). Returns ((B,), parts)."""
    s = float(obs_scale)
    if not (s > 0.0) or not np.isfinite(s):
        raise ValueError(f"frame_scale_anchor: obs_scale must be finite and > 0, got {obs_scale!r}")
    d = np.log(s) - torch.log(xstar.clamp_min(floor))
    L = (d ** 2).mean(dim=-1)
    return L, dict(obs_scale=s, xstar_mean=_np(xstar.detach().mean(dim=-1)))


def kstar_anchor_batched(model, xstar, kgrid, kstar_obs, tau=0.12, temp=60.0):
    """Batched k* soft-anchor. Returns ((B,), parts).

    `searchsorted` runs on the shared k-grid, so the interpolation index is one scalar for
    the batch -- the same index the serial `_sigma_at` computes."""
    sig = model.dispersion(xstar, kgrid, J=None)              # (B,K)
    lse = torch.logsumexp(sig * temp, dim=1) / temp
    idx = int(torch.searchsorted(kgrid, torch.as_tensor(float(kstar_obs), device=kgrid.device))
              .clamp(1, len(kgrid) - 1))
    k0, k1 = kgrid[idx - 1], kgrid[idx]
    s0, s1 = sig[:, idx - 1], sig[:, idx]
    t = (kstar_obs - k0) / (k1 - k0 + 1e-12)
    sig_obs = s0 + t * (s1 - s0)
    L = torch.clamp(lse - sig_obs, min=0.0)
    kstar_model = kgrid[torch.argmax(sig, dim=1)].detach()
    r = (kstar_model - float(kstar_obs)).abs() / (float(kstar_obs) + 1e-9)
    return L, dict(kstar_model=_np(kstar_model), kstar_obs=float(kstar_obs), rel_err=_np(r))


def anticollapse_batched(model, xstar, jac_floor=1.0):
    """Batched anti-collapse Jacobian-norm floor. Returns ((B,), parts).

    The optional `sim_field` amplitude floor of the serial term is NOT carried: no caller in
    the library passes it (recover.py never does), so batching it would be dead code."""
    J = model.jacobian(xstar, create_graph=True)
    jn = torch.linalg.matrix_norm(J, ord="fro")               # (B,) — Frobenius, per member
    L = _softplus_hinge(jac_floor - jn)
    return L, dict(jac_norm=_np(jn))


def _np(t):
    """Detach a (B,) tensor to a numpy array for the per-member diagnostics dict."""
    return t.detach().cpu().numpy()


# --------------------------------------------------------------------------------------
# composite
# --------------------------------------------------------------------------------------
# resid defaults to 0.0: SETTLED OFF. exp06 swept batch {64,128,512} x weight {1,3,10},
# 8 seeds each; ALL NINE cells collapsed to 1/8 Turing seeds, best median k* error 11.8 %
# against 0.4 % with the residual off. The term is kept, not deleted — it is the only
# pixel-level term and a future arm may re-enable it — but it is off by default.
DEFAULT_WEIGHTS = dict(kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5,
                       anchor=2.0, morphology=0.0,
                       param_prior=0.0)   # param_prior default 0.0: opt-in (unit 5)


def composite_loss(model, frame, L, observed_idx, kgrid, kstar_obs,
                   weights=None, latent_fields=None, tau=0.12, jac_floor=1.0,
                   k_min_frac=0.1,
                   dratio_centre=7.5, dratio_spread=1.0, bio_box_path="configs/bio_box.yaml"):
    """The weighted objective. Returns (scalar loss, parts dict).

    frame: (m,H,W) observed channels (torch). kstar_obs: measured FFT k* of the frame (float).

    Uses the same terms as losses/total.compute_terms (split hinges + frame-scale anchor).
    losses/total.py is the assembler recover.py actually calls; this one is the standalone
    reference form.
    """
    w = dict(DEFAULT_WEIGHTS);
    if weights: w.update(weights)
    xstar, conv = steady_state(model)
    xstar = steady_state_diff(model, xstar)                # differentiable polish
    L_k, p_k = kstar_anchor(model, xstar, kgrid, kstar_obs, tau=tau)
    L_t, p_t = turing_hinges_split(model, xstar, kgrid, k_min_frac=k_min_frac)
    L_r, p_r = stationarity_residual(model, frame, L, observed_idx, latent_fields)
    L_a, p_a = anticollapse(model, xstar, jac_floor=jac_floor)
    L_s, p_s = frame_scale_anchor(xstar, float(frame.mean()))
    L_p, p_p = param_prior(model, dratio_centre=dratio_centre, dratio_spread=dratio_spread,
                           box_path=bio_box_path)
    # Merge damage repaired 2026-08-12: this function previously computed loss/parts TWICE —
    # the second block (added with param_prior) overwrote the first and silently dropped
    # w['anchor']*L_s and the L_anchor/**p_s entries. The active path (losses/total.py)
    # never had the defect; tests/test_losses.py pins this form against it now.
    loss = (w['kstar']*L_k + w['turing']*L_t + w['resid']*L_r
            + w['anticollapse']*L_a + w['anchor']*L_s + w['param_prior']*L_p)
    parts = dict(total=float(loss), ss_converged=conv,
                 L_kstar=float(L_k), L_turing=float(L_t), L_resid=float(L_r),
                 L_anti=float(L_a), L_anchor=float(L_s),
                 **p_k, **p_t, **p_r, **p_a, **p_s, **p_p)
    return loss, parts
