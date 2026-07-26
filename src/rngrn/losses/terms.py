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
  kstar_anchor          soft tolerance-band penalty |k*_model - k*_FFT| / k*_FFT
  stationarity_residual full RHS D lap(x) + f(x) = 0 on observed channels (latent inferred if m<N)
  anticollapse          margin penalty excluding the f==0, D==0 trivial minimum
  morphology_consistency weak regulariser matching simulated to observed morphology (optional)
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
def steady_state(model, x0=None, tol=1e-10, max_iter=100, relax_steps=2000, relax_dt=1e-2):
    """Solve f(x*) = 0 for the model's OWN homogeneous steady state.

    Damped Newton on the reaction; if it fails (early training / wild theta) fall back to a short
    forward relaxation of dx/dt = f(x) from a positive init. NEVER returns the frame mean.

    Returns (xstar detached-safe tensor of shape (N,), converged: bool).
    The returned xstar carries grad (Newton is differentiable through the reaction), so physics
    terms evaluated at xstar are differentiable w.r.t. theta.
    """
    N = model.N
    if x0 is None:
        x0 = torch.ones(N) * 1.0
    x = x0.clone()
    for _ in range(max_iter):
        fx = model.reaction(x)
        if torch.linalg.norm(fx).item() < tol:
            return x, True
        J = model.jacobian(x, create_graph=False)
        try:
            step = torch.linalg.solve(J, fx)
        except RuntimeError:
            break
        # damped Newton with positivity guard
        lam = 1.0
        for _ in range(30):
            xn = x - lam * step
            if (xn > 0).all() and torch.linalg.norm(model.reaction(xn)).item() < torch.linalg.norm(fx).item():
                break
            lam *= 0.5
        x = torch.clamp(x - lam * step, min=1e-9)
    # relaxation fallback
    x = (x0.clone() if (x0 > 0).all() else torch.ones(N))
    dt = relax_dt
    for _ in range(relax_steps):
        x = torch.clamp(x + dt * model.reaction(x), min=1e-9)
    converged = torch.linalg.norm(model.reaction(x)).item() < 1e-4
    return x, converged


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
    """Softplus hinges on the Turing conditions. General-N: uses the differentiable dispersion,
    requiring sigma(0) < 0 (uniform-stable) and max_{k>0} sigma(k) > 0 (structured-unstable).
    For N=2 this reduces to the classical four inequalities, but the dispersion form is general."""
    J = model.jacobian(xstar, create_graph=True)
    sig = model.dispersion(xstar, kgrid, J=J)              # (K,)
    k0 = kgrid[0]
    # sigma at (near) k=0: stability to uniform perturbations -> want sig0 < 0
    sig0 = sig[0]
    # max growth at k>0 -> want > 0
    sig_max_pos = sig.max()
    L_uniform = _softplus_hinge(sig0 + margin)             # penalise sig0 >= -margin
    L_unstable = _softplus_hinge(-(sig_max_pos - margin))  # penalise sig_max <= margin
    return L_uniform + L_unstable, dict(sig0=float(sig0.detach()),
                                        sig_max=float(sig_max_pos.detach()))


# --------------------------------------------------------------------------------------
# 5.1  k* soft-anchor (tolerance band)
# --------------------------------------------------------------------------------------
def _sigma_at(sig, kgrid, k):
    """Differentiable linear interpolation of sigma(kgrid) at scalar k."""
    idx = torch.searchsorted(kgrid, torch.as_tensor(float(k))).clamp(1, len(kgrid) - 1)
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
    full = torch.zeros(N, H, W, dtype=fields.dtype)
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
# composite
# --------------------------------------------------------------------------------------
DEFAULT_WEIGHTS = dict(kstar=1.0, turing=1.0, resid=0.3, anticollapse=0.5, morphology=0.0)


def composite_loss(model, frame, L, observed_idx, kgrid, kstar_obs,
                   weights=None, latent_fields=None, tau=0.12, jac_floor=1.0):
    """The weighted objective. Returns (scalar loss, parts dict).

    frame: (m,H,W) observed channels (torch). kstar_obs: measured FFT k* of the frame (float).
    """
    w = dict(DEFAULT_WEIGHTS); 
    if weights: w.update(weights)
    xstar, conv = steady_state(model)
    xstar = steady_state_diff(model, xstar)                # differentiable polish
    L_k, p_k = kstar_anchor(model, xstar, kgrid, kstar_obs, tau=tau)
    L_t, p_t = turing_hinges(model, xstar, kgrid)
    L_r, p_r = stationarity_residual(model, frame, L, observed_idx, latent_fields)
    L_a, p_a = anticollapse(model, xstar, jac_floor=jac_floor)
    loss = w['kstar']*L_k + w['turing']*L_t + w['resid']*L_r + w['anticollapse']*L_a
    parts = dict(total=float(loss), ss_converged=conv,
                 L_kstar=float(L_k), L_turing=float(L_t), L_resid=float(L_r), L_anti=float(L_a),
                 **p_k, **p_t, **p_r, **p_a)
    return loss, parts
