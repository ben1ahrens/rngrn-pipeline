"""lifted.py — UNDO the quasi-steady-state reduction and ask whether the pattern survives.

WHICH `mu` THIS IS. In this codebase `mu` names two unrelated things (recorded as an
ambiguity in docs/STATE_OF_THE_SCIENCE.md §713-714):
  (a) the QSS lift's FAST-VARIABLE TIMESCALE in eval/dynamical.py::lift_check — the time
      constant of transcription-factor/promoter binding relative to protein turnover;
  (b) the generator's degradation rate `mu[i]` in scripts/gen_tg3.py, which is the RNGRN's
      `delta` (that is what scripts/stage0_bio_viability.py swept).
THIS MODULE IS ENTIRELY ABOUT (a). Nothing here touches `delta`.

WHAT THE REDUCTION ASSUMES. model.py::_reaction_raw is the QSS algebra: promoter occupancy
is assumed to equilibrate INSTANTANEOUSLY with the TF concentrations, so the gates GA, GR
never appear as state variables. The lift makes them explicit,

    competitive:  free_i = 1 - sum_j GA_ij - sum_j GR_ij
                  dGA_ij/dt = (KA_ij x_j^n free_i - GA_ij) / mu
                  dGR_ij/dt = (KR_ij x_j^n free_i - GR_ij) / mu
                  prod_i    = sum_j alpha_ij GA_ij
    nc1:          dGA_ij/dt = (KA_ij x_j^n (1 - GA_ij) - GA_ij) / mu
                  dGR_ij/dt = (KR_ij x_j^n (1 - GR_ij) - GR_ij) / mu
                  prod_i    = (sum_j alpha_ij GA_ij) * prod_j (1 - GR_ij)
    both:         dx_i/dt   = D_i lap(x_i) + beta_i + prod_i - delta_i x_i

which is exactly eval/dynamical.py::lift_check's algebra, promoted from a fixed-x
consistency check to a state-space object.

THE ONE FACT EVERYTHING ELSE RESTS ON. Setting dGA/dt = dGR/dt = 0 recovers the algebraic
QSS relations EXACTLY, so the lifted fixed point is (x*, GA_qss(x*), GR_qss(x*)) for EVERY
mu: mu changes stability and dynamics, never the steady state. `fixed_point_residual`
measures it; tests/test_lifted.py pins it.

ONLY x DIFFUSES. Promoter states are bound to DNA, so the lifted diffusion matrix is
diag(D_1..D_N, 0, ..., 0) — the gate block has no spatial coupling at all.
"""
from __future__ import annotations

import numpy as np
import torch


# ======================================================================================
# state packing:  z = [ x (N) | GA.ravel() (N^2) | GR.ravel() (N^2) ]   -> N + 2 N^2
# ======================================================================================
def state_dim(N: int) -> int:
    return N + 2 * N * N


def unpack(z, N):
    z = np.asarray(z, float)
    return z[:N], z[N:N + N * N].reshape(N, N), z[N + N * N:].reshape(N, N)


def pack(x, GA, GR):
    return np.concatenate([np.asarray(x, float).ravel(),
                           np.asarray(GA, float).ravel(),
                           np.asarray(GR, float).ravel()])


def _np_params(model):
    return dict(
        KA=model.KA.detach().cpu().numpy(), KR=model.KR.detach().cpu().numpy(),
        alpha=model.alpha.detach().cpu().numpy(), beta=model.beta.detach().cpu().numpy(),
        delta=model.delta.detach().cpu().numpy(), n_hill=model.n_hill, form=model.form,
        D=model.D.detach().cpu().numpy(), N=model.N)


# ======================================================================================
# QSS gate occupancies — the lifted fixed point's gate block
# ======================================================================================
def gates_qss(model, x):
    """(GA, GR) solving dGA/dt = dGR/dt = 0 at the given x. These are the occupancies the
    QSS reaction has ALREADY substituted away; `reaction_from_gates(gates_qss(x))` equals
    model.reaction(x) to round-off, for both forms."""
    p = _np_params(model)
    x = np.asarray(x, float)
    xn = np.clip(x, 0.0, None) ** p["n_hill"]
    if p["form"] == "competitive":
        # free_i = 1 - sum_j (GA+GR)_ij with GA = KA x^n free  =>  free = 1/(1 + sum(KA+KR)x^n)
        free = 1.0 / (1.0 + ((p["KA"] + p["KR"]) * xn[None, :]).sum(1, keepdims=True))
        return p["KA"] * xn[None, :] * free, p["KR"] * xn[None, :] * free
    ua = p["KA"] * xn[None, :]
    ur = p["KR"] * xn[None, :]
    return ua / (1.0 + ua), ur / (1.0 + ur)


def production_from_gates(model, GA, GR):
    p = _np_params(model)
    if p["form"] == "competitive":
        return (p["alpha"] * GA).sum(1)
    return (p["alpha"] * GA).sum(1) * np.prod(1.0 - GR, axis=1)


def lifted_state(model, xstar):
    """The lifted fixed point z* = (x*, GA_qss(x*), GR_qss(x*)). mu-INDEPENDENT."""
    GA, GR = gates_qss(model, xstar)
    return pack(xstar, GA, GR)


# ======================================================================================
# the lifted right-hand side (torch, so the Jacobian is autodiff — never hand-derived)
# ======================================================================================
def lifted_rhs_torch(model, z, mu):
    """f_lift(z) for a single (non-spatial) lifted state. Torch so `lifted_jacobian` can
    autodiff it, matching model.py's contract that Jacobians are never hand-derived."""
    N = model.N
    x = z[:N]
    GA = z[N:N + N * N].reshape(N, N)
    GR = z[N + N * N:].reshape(N, N)
    xn = torch.clamp(x, min=0.0) ** model.n_hill
    KA, KR = model.KA, model.KR
    if model.form == "competitive":
        free = 1.0 - GA.sum(1, keepdim=True) - GR.sum(1, keepdim=True)
        dGA = (KA * xn.unsqueeze(0) * free - GA) / mu
        dGR = (KR * xn.unsqueeze(0) * free - GR) / mu
        prod = (model.alpha * GA).sum(1)
    else:
        dGA = (KA * xn.unsqueeze(0) * (1.0 - GA) - GA) / mu
        dGR = (KR * xn.unsqueeze(0) * (1.0 - GR) - GR) / mu
        prod = (model.alpha * GA).sum(1) * torch.prod(1.0 - GR, dim=1)
    dx = model.beta + prod - model.delta * x
    return torch.cat([dx, dGA.reshape(-1), dGR.reshape(-1)])


def fixed_point_residual(model, xstar, mu):
    """max |f_lift(z*)| at the QSS-gate state. Must be ~0 at EVERY mu — that is the claim
    that the lift moves stability and not the steady state."""
    z = torch.tensor(lifted_state(model, xstar), device=model.device, dtype=model.dtype)
    with torch.no_grad():
        return float(lifted_rhs_torch(model, z, mu).abs().max())


def lifted_jacobian(model, xstar, mu):
    """The full (N + 2N^2)-square Jacobian of the lifted system at the fixed point."""
    z = torch.tensor(lifted_state(model, xstar), device=model.device, dtype=model.dtype)
    J = torch.autograd.functional.jacobian(
        lambda zz: lifted_rhs_torch(model, zz, mu), z, create_graph=False, vectorize=True)
    return J.detach().cpu().numpy()


def lifted_diffusion(model):
    """diag(D_1..D_N, 0, ..., 0) as a vector: only x diffuses, gates are DNA-bound."""
    D = model.D.detach().cpu().numpy()
    return np.concatenate([D, np.zeros(2 * model.N * model.N)])


def rescale_mu(J1, N, mu):
    """J_full(mu) from J_full(1) by scaling the GATE ROWS by 1/mu.

    EXACT, not an approximation, and it is what makes a mu sweep free. mu enters the lifted
    RHS only as an overall 1/mu on the 2N^2 gate equations; the x equations do not contain
    it at all. So every gate ROW of the Jacobian carries exactly one factor 1/mu and the x
    rows carry none. tests/test_lifted.py pins this against a fresh autodiff Jacobian.
    """
    J = np.array(J1, float, copy=True)
    J[..., N:, :] /= mu
    return J


# ======================================================================================
# the extended dispersion relation
# ======================================================================================
# k-grid: the same LOG-spaced 6.5-decade grid stage0_bio_viability.py uses, for exactly the
# same reason (a near-immobile node's branch only turns over at k ~ 1e3). k=0 is prepended
# and EXCLUDED from the k>0 instability search.
KGRID = np.concatenate([[0.0], np.logspace(-3.0, 3.5, 500)])


def lifted_dispersion(model, xstar, mu, kgrid=KGRID, J=None):
    """sigma_full(k, mu) = max Re eig(J_full - k^2 diag(D_ext)), plus the IMAGINARY part
    of the leading eigenvalue at each k.

    Returns (sigma, omega) with omega = |Im| of the eigenvalue attaining the maximum. A
    non-zero omega at the maximiser means the leading mode OSCILLATES — a wave/oscillatory
    instability, which is NOT the stationary Turing pattern this project claims.
    """
    if J is None:
        J = lifted_jacobian(model, xstar, mu)
    Dext = lifted_diffusion(model)
    k2 = np.asarray(kgrid, float) ** 2
    M = J[None] - k2[:, None, None] * np.diag(Dext)[None]
    ev = np.linalg.eigvals(M)                     # (K, dim)
    i = np.argmax(ev.real, axis=-1)
    lead = ev[np.arange(len(k2)), i]
    return lead.real.copy(), np.abs(lead.imag)


def qss_dispersion(model, xstar, kgrid=KGRID):
    """sigma_QSS(k) from the N x N reduced Jacobian — the mu -> 0 limit the lift must
    reproduce. Same k-grid, same convention as `lifted_dispersion`."""
    xs = torch.as_tensor(np.asarray(xstar, float), device=model.device, dtype=model.dtype)
    J = model.jacobian(xs, create_graph=False).detach().cpu().numpy()
    D = model.D.detach().cpu().numpy()
    k2 = np.asarray(kgrid, float) ** 2
    ev = np.linalg.eigvals(J[None] - k2[:, None, None] * np.diag(D)[None])
    i = np.argmax(ev.real, axis=-1)
    lead = ev[np.arange(len(k2)), i]
    return lead.real.copy(), np.abs(lead.imag)


def turing_verdict_lifted(model, xstar, mu, kgrid=KGRID, tol=1e-9, J1=None):
    """STRICT Turing verdict on the LIFTED system.

        stable_uniform : max Re eig(J_full) < 0            (the k=0 mode, ALL 21 branches)
        unstable_k     : max_{k>0} sigma_full(k) > tol

    Deliberately NOT eval/analysis.py::turing_ok's trace test: Stage 0 measured the trace
    criterion overcounting by 64x, every extra draw being a UNIFORM instability rather than
    a pattern. Here the trace of J_full is dominated by the -1/mu gate diagonal and would
    be negative at ANY mu, so the trace test would report "stable" for a system that has
    already gone Hopf. It must not be used.

    `J1` is the lifted Jacobian at mu=1, supplied by a caller sweeping mu so the autodiff
    is done once and `rescale_mu` (exact) produces the rest.
    """
    J = lifted_jacobian(model, xstar, mu) if J1 is None else rescale_mu(J1, model.N, mu)
    sig, om = lifted_dispersion(model, xstar, mu, kgrid, J=J)
    ev0 = np.linalg.eigvals(J)
    max_re0 = float(ev0.real.max())
    # the k=0 leading eigenvalue's frequency: non-zero => the uniform mode goes Hopf, not
    # a simple real-eigenvalue exchange of stability.
    i0 = int(np.argmax(ev0.real))
    i = 1 + int(np.argmax(sig[1:]))
    return dict(
        mu=float(mu), max_re_eig_J=max_re0, omega_uniform=float(abs(ev0[i0].imag)),
        stable_uniform=bool(max_re0 < 0.0),
        sig_max_pos=float(sig[i]), kstar=float(kgrid[i]), omega_at_kstar=float(om[i]),
        unstable_k=bool(sig[i] > tol),
        turing_strict=bool(max_re0 < 0.0 and sig[i] > tol),
        oscillatory=bool(om[i] > 1e-9))


# ======================================================================================
# mu_crit — the largest mu at which the system is still STRICTLY Turing
# ======================================================================================
def mu_critical(model, xstar, lo=1e-6, hi=1e4, per_decade=8, tol_rel=0.02, kgrid=KGRID,
                max_iter=60):
    """The upper edge of the Turing interval that CONTAINS the mu -> 0 limit.

    THE VERDICT IS NOT MONOTONE IN mu AND A PLAIN BISECTION IS WRONG. Measured on
    harvest survivor nc1/mobile3 draw 0: strictly Turing for mu <= 1e-2, NOT Turing over
    mu ~ 1e-1..1e1 (the uniform state goes unstable, and OSCILLATORILY — omega = 0.75),
    then strictly Turing AGAIN for mu >= 1e2, where the frozen gates themselves act as the
    slow non-diffusing species. A bisection seeded on [1e-6, 1e3] sees Turing at both ends
    and reports "never loses it", which is false.

    So: SCAN a log grid (`per_decade` points per decade) from `lo` up, find the FIRST point
    that is not strictly Turing, then bisect that bracket in log10(mu) to `tol_rel`. The
    scan also records whether the verdict comes BACK at larger mu (`reentrant`), because a
    re-entrant band is a real feature of this system and hiding it would be a lie by
    omission.

    Returns mu_lo (last Turing), mu_hi (first not-Turing), their geometric mean as the
    point estimate, and WHICH condition fails at mu_hi: 'uniform' (the homogeneous state
    went unstable — a Hopf/delay route if omega != 0) or 'band' (the k>0 band closed).
    """
    J1 = lifted_jacobian(model, xstar, 1.0)
    grid = 10.0 ** np.arange(np.log10(lo), np.log10(hi) + 1e-12, 1.0 / per_decade)
    verds = [turing_verdict_lifted(model, xstar, float(m), kgrid, J1=J1) for m in grid]
    ok = np.array([v["turing_strict"] for v in verds], bool)
    if not ok[0]:
        return dict(status="not_turing_at_lo", mu_lo=float(lo), mu_hi=float(lo),
                    mu_crit=float("nan"), failure="none", n_iter=0, reentrant=False,
                    scan_mu=grid, scan_ok=ok, verdict_lo=verds[0])
    if ok.all():
        return dict(status="turing_at_hi", mu_lo=float(hi), mu_hi=float("inf"),
                    mu_crit=float("inf"), failure="none", n_iter=0, reentrant=False,
                    scan_mu=grid, scan_ok=ok, verdict_hi=verds[-1])
    j = int(np.argmax(~ok))                       # first False
    a, b = float(grid[j - 1]), float(grid[j])
    it = 0
    while it < max_iter and (np.log10(b) - np.log10(a)) > np.log10(1.0 + tol_rel):
        m = 10.0 ** (0.5 * (np.log10(a) + np.log10(b)))
        if turing_verdict_lifted(model, xstar, m, kgrid, J1=J1)["turing_strict"]:
            a = m
        else:
            b = m
        it += 1
    v_a = turing_verdict_lifted(model, xstar, a, kgrid, J1=J1)
    v_b = turing_verdict_lifted(model, xstar, b, kgrid, J1=J1)
    failure = ("uniform" if not v_b["stable_uniform"] else
               "band" if not v_b["unstable_k"] else "unknown")
    return dict(status="bracketed", mu_lo=a, mu_hi=b,
                mu_crit=float(10.0 ** (0.5 * (np.log10(a) + np.log10(b)))),
                failure=failure, n_iter=it, reentrant=bool(ok[j:].any()),
                scan_mu=grid, scan_ok=ok, verdict_lo=v_a, verdict_hi=v_b)


# ======================================================================================
# BATCHED lifted Jacobians — one vmapped autodiff pass over a whole perturbation cloud
# ======================================================================================
def _rhs_flat(KA, KR, alpha, beta, delta, z, form, n_hill):
    """Single-member lifted RHS at mu = 1, a pure function of tensors so torch.func can
    vmap/jacrev it. Same arithmetic as `lifted_rhs_torch` with mu = 1."""
    N = beta.shape[-1]
    x = z[:N]
    GA = z[N:N + N * N].reshape(N, N)
    GR = z[N + N * N:].reshape(N, N)
    xn = torch.clamp(x, min=0.0) ** n_hill
    if form == "competitive":
        free = 1.0 - GA.sum(1, keepdim=True) - GR.sum(1, keepdim=True)
        dGA = KA * xn.unsqueeze(0) * free - GA
        dGR = KR * xn.unsqueeze(0) * free - GR
        prod = (alpha * GA).sum(1)
    else:
        dGA = KA * xn.unsqueeze(0) * (1.0 - GA) - GA
        dGR = KR * xn.unsqueeze(0) * (1.0 - GR) - GR
        prod = (alpha * GA).sum(1) * torch.prod(1.0 - GR, dim=1)
    dx = beta + prod - delta * x
    return torch.cat([dx, dGA.reshape(-1), dGR.reshape(-1)])


def gates_qss_arrays(KA, KR, x, form, n_hill):
    """`gates_qss` for raw arrays, batched over leading dims. KA/KR (...,N,N), x (...,N)."""
    KA = np.asarray(KA, float)
    KR = np.asarray(KR, float)
    xj = (np.clip(np.asarray(x, float), 0.0, None) ** n_hill)[..., None, :]
    if form == "competitive":
        free = 1.0 / (1.0 + ((KA + KR) * xj).sum(-1, keepdims=True))
        return KA * xj * free, KR * xj * free
    ua, ur = KA * xj, KR * xj
    return ua / (1.0 + ua), ur / (1.0 + ur)


def lifted_jacobian_batch(KA, KR, alpha, beta, delta, xstar, form, n_hill):
    """(B, dim, dim) lifted Jacobians at mu = 1, by vmapped autodiff (never hand-derived).
    All inputs numpy: KA/KR/alpha (B,N,N), beta/delta/xstar (B,N). Rescale the gate rows by
    1/mu (`rescale_mu`) for any other mu. Returns (J1, N)."""
    t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)
    bt = t(beta)
    B = bt.shape[0]
    GA, GR = gates_qss_arrays(KA, KR, xstar, form, n_hill)
    z = torch.cat([t(xstar), t(GA).reshape(B, -1), t(GR).reshape(B, -1)], dim=1)
    f = lambda ka, kr, al, be, de, zz: _rhs_flat(ka, kr, al, be, de, zz, form, n_hill)
    J = torch.func.vmap(torch.func.jacrev(f, argnums=5))(
        t(KA), t(KR), t(alpha), bt, t(delta), z)
    return J.detach().cpu().numpy(), int(bt.shape[1])


# ======================================================================================
# FINITE-mu ROBUSTNESS — the headline measurement
# ======================================================================================
# Coarser than KGRID (500 -> 120 points, still ~18 per decade over the same 6.5 decades)
# because the cloud evaluates it on B x K matrices of side N + 2N^2. That resolves k* to
# ~4%, which is irrelevant to a Turing VERDICT (a band either contains a positive sigma or
# it does not) — scripts/stage0b_mu.py's `--kgrid-check` re-runs a cell on the full
# 500-point grid and reports the volume difference.
KGRID_CLOUD = np.concatenate([[0.0], np.logspace(-3.0, 3.5, 120)])

# --------------------------------------------------------------------------------------
# WHERE BIOLOGY SITS ON THE mu AXIS.  mu = (TF-promoter binding timescale) / (protein
# production-degradation timescale), both in the SAME units, so mu is dimensionless and the
# model's own time unit cancels. Full derivation and citations: docs/TIMESCALE_MU.md §2.
#
#   NUMERATOR — TF residence time on specific DNA, live-cell single-molecule tracking:
#     Chen J. et al. (2014) "Single-molecule dynamics of enhanceosome assembly in embryonic
#     stem cells", Cell 156:1274-1285 — Sox2 dwells 12.0-14.6 s on specific target DNA,
#     0.75-0.9 s on non-specific chromatin. Tens of seconds is the standard order across
#     factors (Gebhardt et al. 2013; Chen et al. 2014). Take 1-30 s as the defensible span,
#     the low end being fast/non-specific-dominated binding and the high end long-dwelling
#     factors.
#   DENOMINATOR — protein MEAN lifetime in patterning tissue:
#     Muller P. et al. (2012) "Differential diffusivity of Nodal and Lefty underlies a
#     reaction-diffusion patterning system", Science 336:721-724 — measured clearance
#     half-lives of Cyclops, Squint, Lefty1, Lefty2 in live zebrafish embryos span 95-218
#     min, i.e. mean lifetimes t_1/2/ln2 = 137-315 min = 8.2e3-1.9e4 s. (Schwanhausser et
#     al. (2011) Nature 473:337-342 give a median protein half-life of 46 h in cultured
#     NIH3T3, an order of magnitude LONGER; using it would push mu_bio ~10x DOWN, so the
#     developmental anchor is the conservative — larger mu — choice.)
#   MODEL TIME UNIT — configs/bio_box.yaml's cited delta row is [0.4, 5.0], and delta is the
#     degradation rate in model units, so one model time unit is (0.4 to 5) protein
#     lifetimes. mu in MODEL units therefore carries that factor as well.
#
#     mu_bio = t_residence / (delta_model * t_lifetime)
#            = [1 s / (5.0 * 1.9e4 s), 30 s / (0.4 * 8.2e3 s)] = [1.1e-5, 9.2e-3]
#     central (10 s, delta = 1, t_1/2 = 160 min):  10 / (1 * 1.39e4) = 7.2e-4
#
# This is an ORDER-OF-MAGNITUDE band, not a point, and it is drawn on every figure as a
# band. It is CITED, not UNCALIBRATED, but its width is three decades and must stay visible.
# --------------------------------------------------------------------------------------
MU_BIO_LO, MU_BIO_CENTRAL, MU_BIO_HI = 1.1e-5, 7.2e-4, 9.2e-3


def draw_mu_bio(rng, n, lo=MU_BIO_LO, hi=MU_BIO_HI):
    """n log-uniform draws across the biological mu band — mu treated as the uncertain
    parameter it is, rather than as a known constant."""
    return 10.0 ** rng.uniform(np.log10(lo), np.log10(hi), n)


# The pre-registered bars these volumes are read against (docs/PREREGISTRATION.md §3.2).
PREREG_BARS = {"4p8pct": 0.95, "10pct": 0.90}
# Same levels as eval/analysis.py::ROBUSTNESS_SIGMA_LEVELS (4.8% is Tica et al.'s measured
# experimental parameter CV between biological repeats), so the numbers line up by level.
CLOUD_SIGMA_LEVELS = {"1pct": 0.01, "4p8pct": 0.048, "10pct": 0.10, "20pct": 0.20}


def draw_param_cloud(model, sigma_log, rng, n_samples):
    """n_samples lognormal multiplicative perturbations of the model's PHYSICAL KINETIC
    parameters (KA, KR, alpha, beta, delta, D), each entry with its own factor.

    WHY NOT eval/analysis.py::_draw_JD_cloud. That routine perturbs the entries of the
    REDUCED Jacobian J directly. The lifted Jacobian is not a free matrix — it is BUILT from
    (KA, KR, alpha, beta, delta) and x*, so a perturbed reduced J corresponds to no lifted
    system and cannot be lifted at all. The perturbation therefore has to act one level
    down, on the kinetic parameters, with x* and the Jacobian RE-DERIVED per draw. That is
    Tica et al.'s own perturbation model, which analysis.py's docstring already names as the
    alternative to its own.

    The two properties that made _draw_JD_cloud correct are preserved by construction: every
    factor is lognormal hence strictly positive, and every perturbed quantity is strictly
    positive under the model's link functions, so no sign flips and no structural zero is
    created. What is NOT preserved is comparability of the ABSOLUTE volume to the QSS tables
    in docs/ROBUSTNESS_MEASUREMENT.md §4.2, which used the other scheme. That is why every
    curve carries its OWN mu -> 0 limit as its baseline: along a curve the perturbation model
    is fixed and ONLY mu varies, so the mu-dependence is isolated even where the absolute
    level is not comparable to the older tables.
    """
    g = lambda a: np.asarray(a.detach().cpu().numpy(), float)
    KA, KR = g(model.KA), g(model.KR)
    alpha, beta = g(model.alpha), g(model.beta)
    delta, D = g(model.delta), g(model.D)
    f = lambda shape: rng.lognormal(0.0, sigma_log, size=(n_samples,) + shape)
    return dict(KA=KA[None] * f(KA.shape), KR=KR[None] * f(KR.shape),
                alpha=alpha[None] * f(alpha.shape), beta=beta[None] * f(beta.shape),
                delta=delta[None] * f(delta.shape), D=D[None] * f(D.shape))


def cloud_xstar(model, cloud):
    """x*, convergence flag and the batched model for every draw, by the batched Newton
    (losses.terms.steady_state_batched). The model is returned so the caller can also take
    the REDUCED (QSS) Jacobian of the same draws — that is the mu -> 0 baseline every
    finite-mu volume is read against."""
    from ..losses.terms import steady_state_batched
    from ..model import BatchedRNGRN, RNGRN

    B = cloud["beta"].shape[0]
    proto = RNGRN(N=model.N, form=model.form, n_hill=model.n_hill, seed=0)
    bm = BatchedRNGRN([proto] * B)
    s = cloud["KA"] + cloud["KR"]
    inv_sp = lambda a: torch.log(torch.expm1(torch.as_tensor(a, dtype=torch.float64)))
    with torch.no_grad():
        bm.theta_alpha.copy_(inv_sp(cloud["alpha"]))
        bm.theta_delta.copy_(inv_sp(cloud["delta"]))
        bm.theta_beta.copy_(inv_sp(cloud["beta"]))
        bm.theta_D.copy_(torch.log(torch.as_tensor(cloud["D"], dtype=torch.float64)))
        bm.theta_s.copy_(inv_sp(s))
        bm.theta_g.copy_(torch.logit(torch.as_tensor(cloud["KA"] / s, dtype=torch.float64)))
    xs, conv = steady_state_batched(bm)
    return xs.detach().cpu().numpy(), np.asarray(conv.cpu().numpy(), bool), bm


def verdicts_from_J(J, D, kgrid=KGRID_CLOUD, osc_tol=1e-9, tol=1e-9, block=32):
    """Strict Turing verdict for a BATCH of lifted Jacobians. J (B,dim,dim), D (B,N).

    `stationary` = turing_strict AND the leading k>0 mode has zero imaginary part. A
    Turing-unstable mode with Im != 0 is an oscillatory/wave instability — a travelling
    wave, NOT the stationary pattern this project claims to recover — so it is reported
    separately and never folded into the headline volume.
    """
    B, dim, _ = J.shape
    N = D.shape[1]
    Dext = np.zeros((B, dim))
    Dext[:, :N] = D
    k2 = np.asarray(kgrid, float) ** 2
    K = len(k2)
    idx = np.arange(dim)
    out = {k: np.empty(B) for k in ("sig_max_pos", "kstar", "omega_at_kstar",
                                    "max_re_eig_J", "omega_uniform")}
    for a in range(0, B, block):
        b = min(a + block, B)
        Dm = np.zeros((b - a, K, dim, dim))
        Dm[:, :, idx, idx] = Dext[a:b, None, :] * k2[None, :, None]
        ev = np.linalg.eigvals(J[a:b, None] - Dm)                      # (b-a, K, dim)
        li = np.argmax(ev.real, axis=-1)
        lead = np.take_along_axis(ev, li[..., None], axis=-1)[..., 0]  # (b-a, K)
        out["max_re_eig_J"][a:b] = lead.real[:, 0]
        out["omega_uniform"][a:b] = np.abs(lead.imag[:, 0])
        i = 1 + np.argmax(lead.real[:, 1:], axis=1)
        r = np.arange(b - a)
        out["sig_max_pos"][a:b] = lead.real[r, i]
        out["omega_at_kstar"][a:b] = np.abs(lead.imag[r, i])
        out["kstar"][a:b] = np.asarray(kgrid, float)[i]
    out["stable_uniform"] = out["max_re_eig_J"] < 0.0
    out["unstable_k"] = out["sig_max_pos"] > tol
    out["turing_strict"] = out["stable_uniform"] & out["unstable_k"]
    out["oscillatory"] = out["turing_strict"] & (out["omega_at_kstar"] > osc_tol)
    out["stationary"] = out["turing_strict"] & ~out["oscillatory"]
    return out


def robustness_vs_mu(model, mus, sigma_log=0.10, n_samples=200, seed=0,
                     kgrid=KGRID_CLOUD, mu_draw=None):
    """THE HEADLINE MEASUREMENT: strict Turing VOLUME of the LIFTED system vs mu.

    One parameter cloud is drawn ONCE and every mu is evaluated on the SAME draws, so the
    curve isolates mu: the only thing changing along it is the gate timescale. Each draw's
    lifted Jacobian is built once at mu = 1 and rescaled exactly (`rescale_mu`), so a mu
    point costs one eigen-scan and no extra autodiff.

    A draw whose x* does not converge is a FAILED draw: it keeps its place in the
    denominator with turing_strict = False and is never dropped.

    `mu_draw` (length n_samples) gives each draw its OWN mu, e.g. log-uniform across the
    biological band. It adds the `mu_drawn` row, which is the honest answer to "is it robust
    when mu is uncertain TOO": a fixed-mu column understates the exposure, because mu is
    itself a parameter known only to an order of magnitude.
    """
    rng = np.random.default_rng(seed)
    cloud = draw_param_cloud(model, sigma_log, rng, n_samples)
    xstar, conv, bm = cloud_xstar(model, cloud)
    J1, N = lifted_jacobian_batch(cloud["KA"], cloud["KR"], cloud["alpha"], cloud["beta"],
                                  cloud["delta"], xstar, model.form, model.n_hill)
    D = cloud["D"]

    def _row(Jmu):
        v = verdicts_from_J(Jmu, D, kgrid)
        keep = v["turing_strict"] & conv
        lost = conv & ~v["turing_strict"]
        return dict(
            frac_turing=float(keep.mean()), n_turing=int(keep.sum()),
            frac_stationary=float((v["stationary"] & conv).mean()),
            frac_oscillatory=float((v["oscillatory"] & conv).mean()),
            frac_fail_uniform=float((lost & ~v["stable_uniform"]).mean()),
            frac_fail_band=float((lost & v["stable_uniform"] & ~v["unstable_k"]).mean()),
            kstar_mean=float(v["kstar"][keep].mean()) if keep.any() else float("nan"),
            kstar_std=float(v["kstar"][keep].std()) if keep.any() else float("nan"))

    rows = [dict(mu=float(mu), **_row(rescale_mu(J1, N, mu)))
            for mu in np.asarray(mus, float)]

    # THE mu -> 0 BASELINE, on the SAME draws: the REDUCED (QSS) Jacobian's own volume.
    # Every finite-mu number above is read against this one, never against the older
    # docs/ROBUSTNESS_MEASUREMENT.md tables (different perturbation model — see
    # draw_param_cloud). The lifted volume must converge to it as mu -> 0; if it does not,
    # the lift is wrong, not the physics.
    Jq = bm.jacobian(torch.as_tensor(xstar, dtype=torch.float64),
                     create_graph=False).detach().cpu().numpy()
    qss = _row(Jq)

    drawn = None
    if mu_draw is not None:
        md = np.asarray(mu_draw, float).reshape(-1)
        if len(md) != n_samples:
            raise ValueError(
                f"mu_draw must have n_samples={n_samples} entries, got {len(md)}")
        Jmu = J1.copy()
        Jmu[:, N:, :] /= md[:, None, None]
        drawn = dict(mu_lo=float(md.min()), mu_hi=float(md.max()),
                     mu_median=float(np.median(md)), **_row(Jmu))

    return dict(mus=np.asarray(mus, float), rows=rows, mu_drawn=drawn, qss=qss,
                n=int(n_samples), n_converged=int(conv.sum()), sigma_log=float(sigma_log),
                seed=int(seed), form=model.form)


# ======================================================================================
# SPATIAL INTEGRATION of the lifted PDE — Strang splitting with an EXACT gate substep
# ======================================================================================
# THE STIFFNESS. The gates relax at rate 1/mu, so an explicit scheme needs dt << mu and a
# mu = 1e-3 run costs ~1000x the QSS one. But at FROZEN x the gate block is LINEAR in G, so
# it can be integrated EXACTLY over a step — unconditionally stable at any dt, and exact in
# the mu -> 0 limit (where it simply sets G = G_qss(x), recovering the QSS scheme).
#
#   nc1:          dG/dt = (u - (1+u) G)/mu, u = K x^n, diagonal.
#                 G <- G_inf + (G - G_inf) exp(-(1+u) dt/mu),  G_inf = u/(1+u).
#   competitive:  per ROW i the 2N gates w = [GA_i | GR_i] obey
#                 dw/dt = (a - (I + a 1^T) w)/mu,   a = [KA_i x^n | KR_i x^n].
#                 A = I + a 1^T is a RANK-ONE update of the identity, whose matrix
#                 exponential is closed-form:
#                     exp(-A tau) = e^{-tau} ( I + ((e^{-S tau} - 1)/S) a 1^T ),  S = sum a
#                 (eigenvalue 1 on the 1-orthogonal subspace, 1+S along a), and
#                     w_inf = A^{-1} a = a / (1 + S)
#                 which IS the QSS occupancy. So the competitive gate substep is exact too,
#                 with no linear solve and no matrix exponential routine.
#
# An explicit or damped-implicit gate treatment would produce "the pattern died at finite
# mu" as an ARTEFACT, which is the most dangerous failure mode available here. Exactness in
# G removes that risk structurally; `dt_convergence` in scripts/stage0b_mu.py demonstrates
# it empirically by halving dt.


def gate_step_exact(model, X, GA, GR, dt, mu):
    """Advance the gates EXACTLY over dt at frozen x. X: (N,n,n), GA/GR: (N,N,n,n)."""
    p = _np_params(model)
    xn = np.clip(X, 0.0, None) ** p["n_hill"]                        # (N,n,n) over j
    ua = p["KA"][:, :, None, None] * xn[None]                        # (i,j,n,n)
    ur = p["KR"][:, :, None, None] * xn[None]
    if p["form"] != "competitive":
        ea = np.exp(-(1.0 + ua) * (dt / mu))
        er = np.exp(-(1.0 + ur) * (dt / mu))
        GAi = ua / (1.0 + ua)
        GRi = ur / (1.0 + ur)
        return GAi + (GA - GAi) * ea, GRi + (GR - GRi) * er
    S = ua.sum(1) + ur.sum(1)                                        # (i,n,n)
    tau = dt / mu
    e1 = np.exp(-tau)
    # (e^{-S tau} - 1)/S, taken to its limit -tau as S -> 0
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(S > 1e-12, (np.exp(-S * tau) - 1.0) / np.where(S > 1e-12, S, 1.0), -tau)
    denom = 1.0 + S                                                  # (i,n,n)
    GAinf = ua / denom[:, None]
    GRinf = ur / denom[:, None]
    dA = GA - GAinf
    dR = GR - GRinf
    proj = dA.sum(1) + dR.sum(1)                                     # 1^T (w - w_inf)
    GAn = GAinf + e1 * (dA + c[:, None] * ua * proj[:, None])
    GRn = GRinf + e1 * (dR + c[:, None] * ur * proj[:, None])
    return GAn, GRn


def _reaction_from_gates_np(model, GA, GR):
    p = _np_params(model)
    if p["form"] == "competitive":
        return np.einsum("ij,ijxy->ixy", p["alpha"], GA)
    return (np.einsum("ij,ijxy->ixy", p["alpha"], GA)
            * np.prod(1.0 - GR, axis=1))


def step_policy(model, xstar, D, L, n, dt, T, max_steps):
    """dt, horizon and step budget for a lifted spatial run.

    eval/rollout.py::simulate's growth-rate-aware policy, evaluated on the QSS Jacobian, so
    the lifted and QSS runs of the same model use the SAME dt and horizon. Extracted from
    `simulate_lifted` (unchanged arithmetic) so `lifted_torch.simulate_lifted_torch` picks
    the identical step sequence from the identical inputs — two copies of this block would
    make a CPU/GPU disagreement look like a numerics defect when it was a policy drift.

    Returns (dt, sig_max, horizon_rate, nsteps, hit_budget).
    """
    xs_t = torch.tensor(np.asarray(xstar, float), device=model.device, dtype=model.dtype)
    Jn = model.jacobian(xs_t, create_graph=False).detach().cpu().numpy()
    kg = np.linspace(1e-3, 2 * np.pi * (n // 2) / L, 2000)
    sigd = np.array([np.max(np.real(np.linalg.eigvals(Jn - kk ** 2 * np.diag(D))))
                     for kk in kg])
    sig_max = float(sigd.max())
    horizon_rate = max(abs(sig_max), 1e-12)
    jac_rate = float(np.max(np.abs(np.linalg.eigvals(Jn))))
    if dt is None:
        dt = 0.2 / (jac_rate + 1e-9)
    if T is None:
        T = 40.0 / horizon_rate
    nsteps = int(np.clip(T / dt, 200, max_steps))
    return float(dt), sig_max, float(horizon_rate), nsteps, bool((T / dt) > max_steps)


def simulate_lifted(model, L, mu, n=64, T=None, dt=None, seed=0, noise=1e-2, xstar=None,
                    max_steps=6_000_000, record_every=0):
    """Integrate the FULL lifted spatial system on an n x n periodic grid.

    Strang splitting, one step:  gates dt/2 (exact, frozen x)  ->  x over dt by ETDRK4 with
    the gates frozen  ->  gates dt/2 (exact). The x substep reuses eval/numerics.py's
    cached-coefficient ETDRK4 (rfft), i.e. the SAME diffusion treatment eval/rollout.py
    uses, so the QSS and lifted runs differ only in how the gates are handled.

    dt and T follow eval/rollout.py::simulate's growth-rate-aware policy on the QSS
    Jacobian, so the lifted and QSS runs of the same model use the SAME dt and horizon and
    are directly comparable. `stopped_reason` is 'horizon', 'step_budget' or 'blew_up'; a
    step_budget run was TRUNCATED and its field is not a statement about the attractor.
    """
    import time

    from .numerics import integrate_etdrk4_rfft

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

    rng = np.random.default_rng(seed)
    X = xstar[:, None, None] + noise * rng.standard_normal((N, n, n))
    GA0, GR0 = gates_qss(model, xstar)
    GA = np.broadcast_to(GA0[:, :, None, None], (N, N, n, n)).copy()
    GR = np.broadcast_to(GR0[:, :, None, None], (N, N, n, n)).copy()

    delta = p["delta"][:, None, None]
    beta = p["beta"][:, None, None]
    frames = []
    gates = [GA, GR]           # closed over by reaction_np, rebound each step

    def reaction_np(Xr):
        return beta + _reaction_from_gates_np(model, gates[0], gates[1]) - delta * Xr

    t0 = time.perf_counter()
    blew_up = False
    done = 0
    for step in range(nsteps):
        gates[0], gates[1] = gate_step_exact(model, X, gates[0], gates[1], 0.5 * dt, mu)
        X, blew_up = integrate_etdrk4_rfft(X, D, reaction_np, n, L, dt, 1)
        if blew_up or not np.all(np.isfinite(X)):
            blew_up = True
            break
        gates[0], gates[1] = gate_step_exact(model, X, gates[0], gates[1], 0.5 * dt, mu)
        done = step + 1
        if record_every and (step % record_every == 0):
            frames.append(X.copy())
    seconds = time.perf_counter() - t0
    GA, GR = gates

    out = dict(fields=X, GA=GA, GR=GR, mu=float(mu), dt=float(dt), nsteps=nsteps,
               nsteps_run=done, sig_max=sig_max, horizon_rate=float(horizon_rate),
               seconds=float(seconds), blew_up=bool(blew_up), L=float(L), n=n,
               frames=frames)
    if blew_up:
        out.update(kstar=np.nan, patterned=False, amplitude=np.nan,
                   stopped_reason="blew_up")
        return out
    from .. import observables as obs
    amp = float(X[0].std())
    pattern_floor = max(1e-3, 0.02 * abs(xstar[0]))
    out.update(kstar=(obs.kstar_of(X[0], L=L) if amp > 1e-6 else np.nan),
               patterned=bool(amp > pattern_floor), amplitude=amp,
               stopped_reason=("step_budget" if hit_budget else "horizon"))
    return out
