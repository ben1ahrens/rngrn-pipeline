"""ladder.py — PROVENANCE: ladder scaffolding, validated by tests/test_lift_ladder.py.

Library home for every rung of the lift-validation ladder (docs/REDESIGN_rngrn.md §5.3,
V0-V4): each rung is `vN_invariants(...)`, run against draws from `draw_models`, and each
licenses the next. This module currently implements V0 only; later tasks add v1-v4 here.

V0 — algebraic invariants (§5.3): the lifted fixed point equals the QSS fixed point at
EVERY mu (`lifted.fixed_point_residual`); `rescale_mu` reproduces a fresh autodiff Jacobian
(`lifted.lifted_jacobian`); and the QSS gate occupancies fed through `production_from_gates`
reproduce `model.reaction` exactly (`lifted.gates_qss`). All three are proved once,
algebraically, in eval/lifted.py's own module docstring; V0 only extends the *sample size*
the claim is checked over. Reuses eval/lifted.py's functions verbatim -- no algebra is
re-derived here.

CONTROLLER RULING (ledger 2026-08-17, on Task 2's decision point). The single-model 1e-7
fixed-point-residual bar (tests/test_lifted.py, one hand-picked seed per form) does NOT hold
over a 20-draw low_basal population at N=3 (measured: competitive 2.56e-7, nc1 1.27e-6 --
see task-2-report.md). Two distinct causes were found, and the ruling addresses each
differently:
  (a) some low_basal draws are accepted by losses.terms.steady_state's own `converged` flag
      via its LOOSE relaxation-fallback path (||f(x*)|| < 1e-4), not the tight Newton path
      (||f(x*)|| < 1e-10, losses/terms.py:36-59,129). The lifted fixed point's mu-independence
      is proved for an EXACT root of the reaction; a loose x* measures steady_state's own
      solver quality, not the lift algebra. RULING: `draw_models` now filters to
      Newton-tight draws only (||f(x*)|| <= 1e-10), reporting the exclusion rate as a
      first-class deliverable rather than a hidden filter (see `ModelDraws.exclusion_rate`).
  (b) even at a Newton-tight x*, `fixed_point_residual` still grows like C/mu at the smallest
      tested mu: eval/lifted.py independently reconstructs the gate normalisation `free` two
      ways (gates_qss's closed form vs lifted_rhs_torch's `1 - GA.sum - GR.sum`), algebraically
      identical but not bit-identical, so their float64 difference (~1e-13-1e-16) gets divided
      by mu inside the residual. This is a real property of the EXISTING eval/lifted.py code,
      not a defect, and not something V0 re-derives or patches.

RULING ROUND 2 (ledger 2026-08-17, review of round 1's fix). Round 1's single amplification
constant A was measured as the max of `residual(mu) * mu / eps` over the WHOLE (model, mu)
grid, including the flat-in-mu region -- so it was FLOOR-dominated (the x-block round-off
floor, mechanism (b)'s "flat" part below, sets the max of that ratio at the LARGEST mu, not
the smallest), producing a bound ~5 orders of magnitude too loose exactly where V1 (mu in
{1e-7, 1e-6, 1e-5}) needs it tight. Two DISTINCT residual components, honestly measured
separately, replace the single A:
  - `residual_floor_F` = max fixed-point residual over mu in the FLAT region ({1e-2, 1, 1e2}
    in the standard mu list) -- this is the x-BLOCK round-off floor: `model.reaction(x*)`
    itself, mu-INDEPENDENT (it does not touch the gate rows at all).
  - `amplification_C` = max of `residual(mu) * mu / eps` over mu in the GATE-BLOCK
    amplification region ({1e-6, 1e-4} in the standard mu list) -- the C/mu mechanism proper,
    isolated from the floor by only sampling the region where it actually dominates.
tests/test_lift_ladder.py freezes both, per form, into
`max_residual(mu) <= max(1e-9, 10 * F_FORM, 10 * C_FORM * eps / mu)`, tight in both regimes.

V1 -- linear: eigenvalue continuation of all N + 2N^2 branches (§5.3): the N SLOW branches
converge to the QSS eigenvalues at empirical order ~1 in mu; the 2N^2 FAST branches sit at
~-(1+u)/mu with Re < 0; the lifted k* lands within one grid point of the QSS k*. Branches
are tracked by EIGENVECTOR OVERLAP between adjacent mu (`match_by_overlap`, Hungarian
assignment on |v_i^H v_j|), never by sorting real parts -- branch crossings are exactly what
the re-entrant mu-band is made of, and Re-sorting silently swaps identities through one. See
`v1_continuation` for the two aggregation choices it makes and what it reports about them.

V2 -- temporal, 0-D (§5.3): the N + 2N^2 dimensional WELL-MIXED lifted ODE (the lifted PDE
with the diffusion term dropped) against a trusted stiff reference, scipy Radau at
rtol 1e-10 / atol 1e-12. Two claims, one function each:
  (i)  `v2_qss_limit` -- the lifted trajectory converges to the QSS N-dim ODE trajectory as
       mu -> 0, at empirical order ~1 in mu. BOTH sides are integrated by the same Radau
       reference, so this measures the LIFT and not a stepper.
  (ii) `v2_temporal` -- the Strang stepper `simulate_lifted` uses reproduces that reference
       under dt-halving at observed order ~2 where dt <~ mu, with a sup-norm error at the
       horizon below 0.1 x pattern_floor. The 0-D stepper calls the SAME exact gate substep
       (`lifted.gate_step_exact`) and the same gate-to-production map
       (`lifted._reaction_from_gates_np`) that `simulate_lifted` calls, on 1x1 grids, so what
       V2 measures is the production stepper and not a 0-D re-implementation of it. The only
       substitution is the x substep: ETDRK4-on-rfft becomes plain RK4, because in 0-D there
       is no Laplacian to treat exponentially (CLAUDE.md §7c).
  ORDER DEGRADES FOR dt >> mu BY CONSTRUCTION and that is not a failure: the exact gate
  substep buys unconditional STABILITY at any dt, not second-order accuracy. V2's order claim
  is scoped to dt <~ mu and `dt_over_mu_max` is returned so a reader can see the scope was
  respected.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

from ..losses.terms import steady_state
from ..model import RNGRN
from . import analysis, lifted

# Newton-tight acceptance bar for draw_models (ruling, ledger 2026-08-17): matches
# losses/terms.py::_damped_newton's own convergence tolerance (tol=1e-10) on the SAME norm
# (torch.linalg.norm, Euclidean 2-norm) it uses internally, so "Newton-tight" here means
# exactly what steady_state's fast path itself would have accepted -- not an independently
# invented threshold.
NEWTON_TIGHT_TOL = 1e-10

# float64 machine epsilon, used to define `amplification_C` (ruling round 2, ledger
# 2026-08-17).
EPS64 = 2.220446049250313e-16

# The two mu regions ruling round 2 (ledger 2026-08-17) splits `fixed_point_residual` into,
# matching the standard mu list [1e-6, 1e-4, 1e-2, 1.0, 1e2] this ladder rung is measured on:
# mu >= FLOOR_REGION_MU_MIN is the flat, x-block-round-off-dominated region (residual_floor_F);
# mu <= AMPLIFICATION_REGION_MU_MAX is the gate-block 1/mu-amplified region (amplification_C).
# A mu strictly between the two (not present in the standard list) contributes to neither.
FLOOR_REGION_MU_MIN = 1e-2
AMPLIFICATION_REGION_MU_MAX = 1e-4


class ModelDraws(list):
    """list[RNGRN] returned by draw_models(), carrying its own sampling diagnostics.

    `exclusion_rate` (ruling, ledger 2026-08-17): of the draws losses.terms.steady_state
    ITSELF marked converged=True, the fraction that were nonetheless excluded here for being
    loose (||f(x*)|| > NEWTON_TIGHT_TOL, i.e. accepted only via steady_state's looser
    relaxation-fallback threshold of 1e-4, not its Newton path's 1e-10). NaN if draw_models
    never saw a steady_state-converged draw (e.g. n=0).
    """
    exclusion_rate: float = float("nan")


def draw_models(n: int, form: str, seed: int, N: int = 3) -> ModelDraws:
    """n RNGRN(N, form, init="low_basal", seed=...) models with a NEWTON-TIGHT steady state:
    ||model.reaction(x*)|| <= NEWTON_TIGHT_TOL (1e-10, the 2-norm losses.terms.steady_state's
    own Newton path uses internally). STRICTER than steady_state's own `converged` flag,
    which also accepts its looser relaxation-fallback path (up to 1e-4) -- excluded here per
    the controller ruling (ledger 2026-08-17, module docstring): the lift's algebraic
    invariants presume an EXACT fixed point, so a loose x* would measure solver quality, not
    the lift.

    Deterministic given (n, form, seed): seeds are tried in the fixed sequence
    seed, seed+1, seed+2, ... regardless of which draws are kept, loose-excluded, or hard
    failures, so the same call always returns the same models. A hard failure (steady_state
    itself does not converge) and a loose exclusion (converged, but not Newton-tight) both
    count toward the same fail-loud budget: raises after 10*n such draws rather than ever
    returning a shortfall silently.

    Returns a `ModelDraws` (list[RNGRN] + `.exclusion_rate`, see its docstring) -- the
    exclusion rate this call measured is a deliverable, read back out by `v0_invariants`.
    """
    models = ModelDraws()
    n_failures = 0
    max_failures = 10 * n
    n_steady_state_converged = 0   # denominator: draws steady_state itself accepted
    n_loose = 0                    # numerator: of those, how many were not Newton-tight
    s = seed
    while len(models) < n:
        m = RNGRN(N=N, form=form, init="low_basal", seed=s)
        s += 1
        xs, converged = steady_state(m)
        if not converged:
            n_failures += 1
        else:
            n_steady_state_converged += 1
            resid = float(torch.linalg.norm(m.reaction(xs)).detach())
            if resid <= NEWTON_TIGHT_TOL:
                models.append(m)
            else:
                n_loose += 1
                n_failures += 1
        if n_failures >= max_failures and len(models) < n:
            raise RuntimeError(
                f"draw_models: {n_failures} unsuccessful draws (limit "
                f"{max_failures} = 10*n; {n_loose} of those were steady_state-converged but "
                f"not Newton-tight) while collecting n={n} form={form!r} models starting "
                f"from seed={seed}; only {len(models)} collected. This is a shortfall, not "
                "a partial result -- raising rather than returning it.")
    models.exclusion_rate = (
        n_loose / n_steady_state_converged if n_steady_state_converged else float("nan"))
    return models


def v0_invariants(models: list, mus: list[float]) -> dict:
    """V0 rung: algebraic invariants of the lifted system, over every model and every mu.

    For each model: the gate-occupancy invariant (mu-independent) is checked once; the
    fixed-point-residual and rescale_mu invariants are checked at EVERY mu in `mus`. Every
    value returned is the WORST (max) observed across the whole (model, mu) grid -- this is
    a floor measurement, not a spot check, per docs/REDESIGN_rngrn.md §5.3 V0.

    Returns {"max_fixed_point_residual", "max_rescale_mu_err", "max_gate_qss_err",
    "n_models", "exclusion_rate", "residual_floor_F", "amplification_C"}.
    `max_rescale_mu_err` is RELATIVE (max abs diff / max abs of the fresh Jacobian), matching
    tests/test_lifted.py::test_rescale_mu_matches_autodiff -- the gate rows of J_full scale
    as 1/mu, so an ABSOLUTE tolerance would not be comparable across the mu range in `mus`
    (a 1e-6 relative slip at mu=1e-6 is a 1.0-scale absolute one). `max_gate_qss_err` is
    absolute, matching tests/test_lifted.py::test_gates_qss_reproduce_the_qss_reaction (both
    f_lift and f_qss are O(x*), no scale sweep involved).

    CONTROLLER RULING ROUND 2 (ledger 2026-08-17, module docstring has the full rationale):
    `exclusion_rate` is read back from `models.exclusion_rate` if `models` came from
    `draw_models` (a `ModelDraws`); NaN for a plain list, since the rate is meaningless
    without draw_models's own bookkeeping of steady_state-converged-but-loose draws.
    `fixed_point_residual(mu)` has TWO honestly-distinct components, and round 2 replaced
    round 1's single floor-dominated `amplification_A` with one measurement per component:
      - `residual_floor_F` = max residual over mu in the FLAT region (mu >=
        FLOOR_REGION_MU_MIN) -- the x-BLOCK round-off floor, i.e. how far
        `model.reaction(x*)` itself sits from zero; mu-INDEPENDENT, because the gate rows of
        the lifted RHS do not appear in this component at all.
      - `amplification_C` = max of `residual(mu) * mu / EPS64` over mu in the
        GATE-AMPLIFICATION region (mu <= AMPLIFICATION_REGION_MU_MAX) -- the C/mu mechanism
        proper (task-2-report.md §4 mechanism (a): eval/lifted.py's two independent,
        not-bit-identical reconstructions of the gate normalisation `free`), isolated from
        the floor by only sampling the region where it actually dominates.
    Both are MEASUREMENTS this function reports, not bars it enforces; the caller
    (tests/test_lift_ladder.py) freezes both, per form, into a bound tight in both regimes.
    """
    max_fp_residual = 0.0
    max_rescale_err = 0.0
    max_gate_err = 0.0
    residual_floor_F = 0.0
    amplification_C = 0.0
    for m in models:
        xs, converged = steady_state(m)
        if not converged:
            raise RuntimeError(
                "v0_invariants: received a model with no converged steady state -- "
                "draw_models should already have filtered this out")
        xstar = xs.detach().cpu().numpy()

        GA, GR = lifted.gates_qss(m, xstar)
        # Mirrors eval/lifted.py::lifted_rhs_torch's `dx = model.beta + prod - model.delta *
        # x` line (numpy here vs torch there, since this feeds a numpy f_qss comparison
        # below) -- a shared helper is deferred to Task 7's single-sourcing decision rather
        # than introduced here as a one-off (Minor, code review round 2).
        f_lift = (m.beta.detach().cpu().numpy() + lifted.production_from_gates(m, GA, GR)
                  - m.delta.detach().cpu().numpy() * xstar)
        f_qss = m.reaction(
            torch.as_tensor(xstar, device=m.device, dtype=m.dtype)).detach().cpu().numpy()
        max_gate_err = max(max_gate_err, float(np.max(np.abs(f_lift - f_qss))))

        J1 = lifted.lifted_jacobian(m, xstar, 1.0)
        for mu in mus:
            resid = lifted.fixed_point_residual(m, xstar, mu)
            max_fp_residual = max(max_fp_residual, resid)
            if mu >= FLOOR_REGION_MU_MIN:
                residual_floor_F = max(residual_floor_F, resid)
            if mu <= AMPLIFICATION_REGION_MU_MAX:
                amplification_C = max(amplification_C, resid * mu / EPS64)
            Jd = lifted.lifted_jacobian(m, xstar, mu)
            Jr = lifted.rescale_mu(J1, m.N, mu)
            rel = float(np.max(np.abs(Jr - Jd))) / max(float(np.max(np.abs(Jd))), 1e-300)
            max_rescale_err = max(max_rescale_err, rel)

    exclusion_rate = getattr(models, "exclusion_rate", float("nan"))
    return dict(max_fixed_point_residual=max_fp_residual,
               max_rescale_mu_err=max_rescale_err,
               max_gate_qss_err=max_gate_err,
               n_models=len(models),
               exclusion_rate=exclusion_rate,
               residual_floor_F=residual_floor_F,
               amplification_C=amplification_C)


# ======================================================================================
# V1 — eigenvalue continuation of all N + 2N^2 branches as mu -> 0
# ======================================================================================
def default_kgrid(model):
    """The k band V1 scans: eval/analysis.py's own `_TURING_KGRID`, linspace(1e-3, 50, 4000).

    Reused verbatim rather than reinvented so a V1 k* is directly comparable to every k*
    `analysis.turing_ok` has ever reported. That band is ABSOLUTE (rad/length) and
    eval/analysis.py:17-23 labels it [UNCALIBRATED] -- adequate for the registered L range
    (18..220 -> k* 0.17..2.09) and not scale-free. V1 inherits that caveat rather than
    silently fixing it, since a different band would make the k* offset non-comparable.

    `model` is accepted so call sites read as "the grid for this model" and so a scale-free
    band could replace this one without touching them; the grid returned today does NOT
    depend on it. It contains no k = 0 -- `v1_continuation` rejects a grid that does.
    """
    return analysis._TURING_KGRID


def match_by_overlap(Va, Vb):
    """Maximum-|overlap| pairing of Va's columns to Vb's columns, independently at each k.

    Va (K, d, na), Vb (K, d, nb) with na <= nb. Returns (K, na) int: `pair[k, i]` is the
    column of Vb[k] paired with Va[k, :, i]. The pairing maximises the TOTAL overlap
    sum_i |Va[k,:,i]^H Vb[k,:,pair[k,i]]| (Hungarian assignment,
    scipy.optimize.linear_sum_assignment), so it is a global matching rather than a greedy
    per-column argmax -- greedy would let one strongly-overlapping Vb column be claimed twice
    and leave a branch unpaired.

    COLUMNS ARE USED AS GIVEN -- this function does not normalise them, because the caller's
    scaling is load-bearing in both of V1's two uses:
      - adjacent-mu tracking passes both sides normalised to unit 2-norm, so the overlap is a
        pure |cos| between branch directions;
      - QSS-to-lifted slow-branch identification passes the N-dim QSS eigenvector against the
        x-BLOCK of the unit-norm 21-dim lifted eigenvector, deliberately NOT renormalised: a
        fast branch's x-block has norm O(mu), so leaving it unnormalised makes ||v_x|| itself
        discriminate slow from fast. Renormalising it would throw that away and let a fast
        branch win on a chance 3-D alignment.
    """
    Va = np.asarray(Va)
    Vb = np.asarray(Vb)
    if Va.ndim != 3 or Vb.ndim != 3 or Va.shape[0] != Vb.shape[0] or Va.shape[1] != Vb.shape[1]:
        raise ValueError(f"match_by_overlap needs (K, d, na) and (K, d, nb) with the same K "
                         f"and d; got {Va.shape} and {Vb.shape}")
    if Va.shape[2] > Vb.shape[2]:
        raise ValueError(f"match_by_overlap needs na <= nb; got na={Va.shape[2]} "
                         f"nb={Vb.shape[2]}")
    ov = np.abs(np.einsum("kia,kib->kab", Va.conj(), Vb))
    pair = np.empty(ov.shape[:2], int)
    for t in range(ov.shape[0]):
        rows, cols = linear_sum_assignment(-ov[t])
        pair[t, rows] = cols
    return pair


def v1_continuation(model, kgrid, mus=(1e-7, 1e-6, 1e-5)) -> dict:
    """V1 rung: continue all N + 2N^2 lifted eigenvalue branches to the mu -> 0 QSS limit.

    THE METHOD, in the order it matters:

    1. One autodiff lifted Jacobian at mu = 1 (`lifted.lifted_jacobian`), then `rescale_mu`
       -- EXACT, not an approximation -- for every mu in the triplet. No mu costs a second
       autodiff pass.
    2. At the SMALLEST mu, the N slow branches are IDENTIFIED by matching the QSS
       eigenvectors against the x-block of the lifted eigenvectors (`match_by_overlap`, see
       its docstring for why the x-block is left unnormalised). Smallest mu first because
       that is where the slow/fast separation is widest.
    3. That labelling is then PROPAGATED to each larger mu by a full 21-branch overlap
       assignment between ADJACENT mu values. Never by sorting real parts: at mu = 1e-5 the
       tracked slow set differs from "the N largest Re" at a large fraction of k on real
       draws (measured 44% on competitive seed 101 draw 1), so Re-sorting would swap branch
       identities through exactly the crossings the re-entrant band is made of.
    4. e(mu, k, i) = |lambda_slow_i(mu, k) - lambda_qss_i(k)|; the per-(k, branch) order is
       the least-squares slope of log e against log mu over the triplet.

    TWO AGGREGATION CHOICES, both reported so neither is hidden:

    (a) `slow_branch_order` is the MEAN of the per-(k, branch) slopes over the k-points where
        the two-timescale separation actually holds, `max(mus) * max(D) * k^2 < 1` -- i.e.
        where the slow relaxation rate D k^2 is slower than the gate rate 1/mu. This is the
        condition under which "order 1 in mu" is a claim at all, not a tuned threshold: above
        it the x-branch and the gate branches genuinely mix, `e` is not O(mu), and the min
        QSS-to-lifted eigenvector overlap collapses (on competitive seed 101 draw 1 it is
        0.996 at mu = 1e-7 and 0.030 at mu = 1e-5 -- a real property of the lift, not a
        defect).
        `slow_branch_order_all_k` is the same mean over EVERY k, `frac_k_separated` the
        fraction kept, `max_mu_D_k2` the worst value of that product on the grid. Measured
        over 16 draws (both forms, seeds 7/101/202/303, n=2 each), 2026-08-17: separated
        0.94..1.07, all-k 0.98..1.15 -- so the choice moves the estimate by <0.1 and both
        sides of it clear the brief's 0.35 band.
    (b) `max_slow_err[mu]` is the max over ALL k and all N slow branches, unmasked -- a floor
        measurement, like V0's. At the largest mu it is therefore dominated by the
        separation-broken large-k tail (measured 3.46e-2 vs 458 on two different draws at
        mu = 1e-5) and is NOT a pure O(mu) quantity there. Read it with `max_mu_D_k2`.

    THE k* CHECK IS USUALLY VACUOUS ON THIS POPULATION, AND SAYS SO. `draw_models` samples
    init="low_basal", and eval/analysis.py::turing_ok records that 0 of 398 such N=3 draws are
    strictly Turing -- so sigma(k) is typically monotone DECREASING and both the QSS and the
    lifted k* pin to the grid FLOOR, making `kstar_grid_offset == 0` true by construction
    rather than by branch tracking. `kstar_at_grid_floor` reports exactly that, so a reader
    never mistakes the one for the other. It is False on the minority of draws whose
    sigma_qss(k) has an interior maximum (measured 2 of 30 draws over both forms and seeds
    7/101/202/303/404, e.g. nc1 seed 202 draw 0 at grid index 3, k = 0.0385) -- those are the
    draws where the offset is a real measurement, and it is 0 there too.

    Returns, all MEASUREMENTS rather than bars (§5.3 V1: "absolute error constants
    UNCALIBRATED -- measured and recorded, not invented"):
      slow_branch_order, slow_branch_order_all_k, slow_order_min, slow_order_max
      kstar_grid_offset (int; max over mu of |argmax_k sigma_lift - argmax_k sigma_qss|)
      kstar_qss, kstar_lifted (dict mu -> k), kstar_at_grid_floor (bool; see above)
      fast_branches_stable (bool; Re lambda < 0 for every fast branch at every k and mu)
      max_fast_re, min_fast_mu_product (min over mu,k,branch of |Re lambda_fast| * mu; ~1+u,
        so >= 1 is the -(1+u)/mu claim), max_slow_err (dict mu -> max |lambda diff|)
      min_qss_overlap (the weakest slow-branch identification made, at the smallest mu)
      n_resort_mismatches, frac_k_separated, max_mu_D_k2, n_slow_branches, n_fast_branches
    """
    mus = np.sort(np.asarray(mus, float))
    if len(mus) < 2:
        raise ValueError(f"v1_continuation needs >=2 mu values to fit a log-log slope; "
                         f"got {len(mus)}")
    if np.any(mus <= 0.0):
        raise ValueError(f"v1_continuation needs strictly positive mu; got {mus.tolist()}")
    kgrid = np.asarray(kgrid, float)
    if kgrid.ndim != 1 or kgrid.size == 0:
        raise ValueError(f"v1_continuation needs a non-empty 1-D kgrid; got {kgrid.shape}")
    if np.any(kgrid <= 0.0):
        raise ValueError(
            "v1_continuation needs a kgrid of strictly positive k. k = 0 is the UNIFORM "
            "mode, not a wavenumber of the structured band, so including it would make "
            "`kstar_grid_offset` an offset against the uniform mode. Use "
            "ladder.default_kgrid(model), or lifted.KGRID[1:].")

    xs, converged = steady_state(model)
    if not converged:
        raise RuntimeError("v1_continuation: model has no converged steady state -- "
                           "draw_models should already have filtered this out")
    xstar = xs.detach().cpu().numpy()
    N = model.N
    K = len(kgrid)
    kk = np.arange(K)[:, None]
    k2 = kgrid ** 2

    # --- the mu -> 0 limit the lift must reproduce: the REDUCED (QSS) N x N problem --------
    D = model.D.detach().cpu().numpy()
    Jq = model.jacobian(xs, create_graph=False).detach().cpu().numpy()
    w_qss, V_qss = np.linalg.eig(Jq[None] - k2[:, None, None] * np.diag(D)[None])
    V_qss = V_qss / np.linalg.norm(V_qss, axis=1, keepdims=True)
    i_qss = int(np.argmax(w_qss.real.max(axis=1)))

    # --- one autodiff Jacobian, exact rescaling for every mu (lifted.rescale_mu) -----------
    J1 = lifted.lifted_jacobian(model, xstar, 1.0)
    Dext = np.diag(lifted.lifted_diffusion(model))

    errs = []
    kstar_lifted = {}
    kstar_offset = 0
    max_fast_re = -np.inf
    min_fast_mu_product = np.inf
    n_resort_mismatches = 0
    min_qss_overlap = np.nan
    slow = None
    w_prev = V_prev = None
    for j, mu in enumerate(mus):
        M = lifted.rescale_mu(J1, N, mu)[None] - k2[:, None, None] * Dext[None]
        w, V = np.linalg.eig(M)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)

        i_lift = int(np.argmax(w.real.max(axis=1)))
        kstar_lifted[float(mu)] = float(kgrid[i_lift])
        kstar_offset = max(kstar_offset, abs(i_lift - i_qss))

        if slow is None:
            # step 2: identify the slow branches against the QSS eigenvectors, x-block only
            # and deliberately NOT renormalised (see match_by_overlap).
            slow = match_by_overlap(V_qss, V[:, :N, :])
            ov = np.abs(np.einsum("kia,kib->kab", V_qss.conj(), V[:, :N, :]))
            min_qss_overlap = float(ov[kk, np.arange(N)[None, :], slow].min())
        else:
            # step 3: propagate by a full 21-branch adjacent-mu overlap assignment.
            pair = match_by_overlap(V_prev, V)
            n_resort_mismatches += _n_resort_mismatches(w_prev, w, pair)
            slow = pair[kk, slow]

        errs.append(np.abs(w[kk, slow] - w_qss))

        fast = np.ones(w.shape, bool)
        fast[kk, slow] = False
        max_fast_re = max(max_fast_re, float(w.real[fast].max()))
        min_fast_mu_product = min(min_fast_mu_product,
                                  float(np.abs(w.real[fast]).min() * mu))
        w_prev, V_prev = w, V

    errs = np.array(errs)                                    # (n_mu, K, N)
    if not np.all(errs > 0.0):
        raise RuntimeError(
            "v1_continuation: a slow-branch error is exactly 0, so its log-log slope is "
            "undefined. Dropping it would bias the order estimate, so this raises instead. "
            f"n_zero={int((errs == 0.0).sum())} of {errs.size}.")
    slopes = np.polyfit(np.log(mus), np.log(errs).reshape(len(mus), -1), 1)[0].reshape(K, N)

    # (a): the region where the two-timescale separation the order-1 claim rests on holds.
    mu_D_k2 = mus.max() * D.max() * k2
    separated = mu_D_k2 < 1.0
    if not separated.any():
        raise RuntimeError(
            f"v1_continuation: max(mus)*max(D)*k^2 >= 1 over the WHOLE kgrid (min "
            f"{mu_D_k2.min():.3g}), so there is no k at which the slow/fast separation holds "
            f"and 'order 1 in mu' is not a claim about any point on this grid. Lower mu or "
            f"restrict the kgrid rather than reading the returned order.")

    return dict(
        slow_branch_order=float(slopes[separated].mean()),
        slow_branch_order_all_k=float(slopes.mean()),
        slow_order_min=float(slopes.min()), slow_order_max=float(slopes.max()),
        kstar_grid_offset=int(kstar_offset),
        kstar_qss=float(kgrid[i_qss]), kstar_lifted=kstar_lifted,
        kstar_at_grid_floor=bool(i_qss == 0),
        fast_branches_stable=bool(max_fast_re < 0.0),
        max_fast_re=max_fast_re, min_fast_mu_product=min_fast_mu_product,
        max_slow_err={float(mu): float(errs[j].max()) for j, mu in enumerate(mus)},
        min_qss_overlap=min_qss_overlap,
        n_resort_mismatches=int(n_resort_mismatches),
        frac_k_separated=float(separated.mean()), max_mu_D_k2=float(mu_D_k2.max()),
        n_slow_branches=int(N), n_fast_branches=int(2 * N * N))


def _n_resort_mismatches(w_a, w_b, pair):
    """How many k the overlap assignment `pair` DISAGREES with descending-Re relabelling at.

    A diagnostic, not a criterion: it is the count of k where the Re-sorting shortcut V1
    rejects would have tracked different branches than the eigenvector overlap did. It is an
    UPPER bound on genuine crossings -- at N=3 the 18 fast branches are near-degenerate in Re
    (all ~-(1+u)/mu with u spread only over the KA/KR range), so argsort ties break
    arbitrarily and inflate the count. The unambiguous demonstration that overlap and
    Re-sorting differ is the synthetic crossing in
    tests/test_lift_ladder.py::test_match_by_overlap_beats_real_part_sorting_on_a_crossing.
    """
    order_a = np.argsort(-w_a.real, axis=1)
    order_b = np.argsort(-w_b.real, axis=1)
    resort = np.empty_like(order_a)
    np.put_along_axis(resort, order_a, order_b, axis=1)
    return int((pair != resort).any(axis=1).sum())


# ======================================================================================
# V2 — the 0-D (well-mixed) lifted ODE: Radau reference, Strang order
# ======================================================================================
# THE REFERENCE. scipy Radau (3-stage RadauIIA, L-stable, fully implicit) at rtol 1e-10 /
# atol 1e-12, the tolerance docs/REDESIGN_rngrn.md §5.3 V2 names. It is given an ANALYTIC
# Jacobian, by autodiff through `lifted.lifted_rhs_torch` (`_lifted_jac_0d`) rather than
# hand-derived, matching eval/lifted.py's own contract; without it Radau finite-differences a
# 21x21 Jacobian and both the cost and the Newton convergence at small mu degrade.
# WHETHER THE REFERENCE IS ACTUALLY RESOLVING THE STIFFNESS IS MEASURED, NOT ASSUMED: every V2
# call also integrates at a TIGHTER tolerance (1e-12/1e-14) and reports
# `radau_self_err` = sup |reference - tighter|, and `radau_ok` requires that to sit at least
# 1/V2_RADAU_MARGIN below the smallest Strang error the reference is being used to judge -- so a
# reference too coarse to judge its subject fails LOUDLY instead of silently setting the floor.
# THE COMPARATOR MUST BE THE TIGHTER SOLVE, NOT A LOOSER ONE. Measured while building this:
# comparing against rtol 1e-8 instead bounds the 1e-8 solution's error, which is ~100x the
# reference's, so it over-states `radau_self_err` by ~2 orders and rejects references that are
# in fact perfectly adequate. |ref - tighter| bounds the REFERENCE's error, which is the
# quantity `radau_ok` is about.
V2_RADAU_RTOL, V2_RADAU_ATOL = 1e-10, 1e-12
V2_RADAU_CHECK_RTOL, V2_RADAU_CHECK_ATOL = 1e-12, 1e-14
V2_RADAU_MARGIN = 0.1

# Output grid: 20 equal intervals on [0, T]. Errors are compared at these times only, and the
# grid is INDEPENDENT of dt, so every dt in a halving triplet is judged at exactly the same
# times (the alternative -- comparing at each stepper's own times -- would fold an
# interpolation error into the order estimate).
V2_N_OUT = 20

# The initial condition, deliberately ON the slow manifold: x0 = V2_X0_SCALE * x*, gates at
# their QSS occupancy FOR x0 (`lifted.lifted_state`). There is therefore no initial fast layer
# of width mu, and the gate dynamics that remain are the O(mu) lag as x relaxes back toward x*
# -- which is exactly the regime `simulate_lifted` runs in (it starts the gates at
# gates_qss(x*) with x perturbed off x*). A layer-laden start (e.g. G = 0) would make the
# sup-norm error report the layer rather than the trajectory.
V2_X0_SCALE = 1.5


def _lifted_rhs_0d(z, mu, p):
    """The 0-D lifted RHS in numpy -- the same vector field as `lifted.lifted_rhs_torch`
    (which is already the non-spatial RHS; there is no diffusion term in it to drop).

    WHY A MIRROR EXISTS AT ALL. Radau makes tens of thousands of function evaluations per
    solve and the torch version measures ~45 us per call, i.e. minutes per V2 case. This is a
    transcription for speed, not new algebra, and it is PINNED to the torch original by
    `v2_rhs_mirror_error` /
    tests/test_lift_ladder.py::test_v2_numpy_rhs_mirrors_lifted_rhs_torch (max relative
    deviation over random states and mu, both forms). `p` is `lifted._np_params(model)`,
    hoisted out of the inner loop.
    """
    N = p["N"]
    x = z[:N]
    GA = z[N:N + N * N].reshape(N, N)
    GR = z[N + N * N:].reshape(N, N)
    xn = np.clip(x, 0.0, None) ** p["n_hill"]
    ua = p["KA"] * xn[None, :]
    ur = p["KR"] * xn[None, :]
    if p["form"] == "competitive":
        free = 1.0 - GA.sum(1, keepdims=True) - GR.sum(1, keepdims=True)
        dGA = (ua * free - GA) / mu
        dGR = (ur * free - GR) / mu
        prod = (p["alpha"] * GA).sum(1)
    else:
        dGA = (ua * (1.0 - GA) - GA) / mu
        dGR = (ur * (1.0 - GR) - GR) / mu
        prod = (p["alpha"] * GA).sum(1) * np.prod(1.0 - GR, axis=1)
    dx = p["beta"] + prod - p["delta"] * x
    return np.concatenate([dx, dGA.ravel(), dGR.ravel()])


def _lifted_jac_0d(model, z, mu):
    """d f_lift / d z at an ARBITRARY z, by autodiff through `lifted.lifted_rhs_torch`.

    `lifted.lifted_jacobian` does the same thing but only at the QSS fixed point; Radau needs
    it along the trajectory. Autodiff, never hand-derived (eval/lifted.py's contract). Called
    once per Radau Jacobian update (tens to hundreds of times per solve), so the torch cost is
    irrelevant here -- unlike in the RHS, which is why only the RHS has a numpy mirror.
    """
    zt = torch.as_tensor(np.asarray(z, float), device=model.device, dtype=model.dtype)
    J = torch.autograd.functional.jacobian(
        lambda zz: lifted.lifted_rhs_torch(model, zz, mu), zt, create_graph=False,
        vectorize=True)
    return J.detach().cpu().numpy()


def _qss_rhs_0d(model, x, p):
    """The 0-D QSS (reduced, N-dim) RHS, assembled from eval/lifted.py's own numpy pieces:
    beta + production_from_gates(gates_qss(x)) - delta * x. V0 measures exactly this against
    `model.reaction` (`max_gate_qss_err`, ~1e-16), so it is the same vector field."""
    GA, GR = lifted.gates_qss(model, x)
    return (p["beta"] + lifted.production_from_gates(model, GA, GR)
            - p["delta"] * np.asarray(x, float))


def _qss_jac_0d(model, x):
    """The reduced N x N Jacobian at an arbitrary x, via `model.jacobian` (autodiff)."""
    xt = torch.as_tensor(np.asarray(x, float), device=model.device, dtype=model.dtype)
    return model.jacobian(xt, create_graph=False).detach().cpu().numpy()


def _radau(fun, y0, T, t_out, jac, rtol, atol):
    """solve_ivp(method="Radau") on [0, T], sampled at `t_out`. Returns (y (n_out+1, dim), sol).
    Raises on failure rather than returning a partial trajectory (CLAUDE.md §4, fail loud)."""
    sol = solve_ivp(fun, (0.0, float(T)), np.asarray(y0, float), method="Radau",
                    t_eval=np.asarray(t_out, float), jac=jac, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"V2 Radau reference failed: {sol.message!r} "
                           f"(rtol={rtol}, atol={atol}, T={T}, nfev={sol.nfev})")
    return sol.y.T.copy(), sol


def v2_initial_state(model, xstar=None):
    """(x0, z0) for V2: x0 = V2_X0_SCALE * x*, gates at gates_qss(x0). See V2_X0_SCALE."""
    if xstar is None:
        xs, converged = steady_state(model)
        if not converged:
            raise RuntimeError("v2_initial_state: model has no converged steady state -- "
                               "draw_models should already have filtered this out")
        xstar = xs.detach().cpu().numpy()
    x0 = V2_X0_SCALE * np.asarray(xstar, float).reshape(model.N)
    return x0, lifted.lifted_state(model, x0)


def _rk4_reaction(X, prod, beta, delta, h):
    """One plain RK4 step of the 0-D reaction dx/dt = beta + prod - delta*x with the gates
    (hence `prod`) FROZEN -- the x half of the Strang step. Frozen gates make the field affine,
    so `prod` is evaluated once per step; this is RK4 on that field, nothing else."""
    c = beta + prod
    f = lambda Y: c - delta * Y
    k1 = f(X)
    k2 = f(X + 0.5 * h * k1)
    k3 = f(X + 0.5 * h * k2)
    k4 = f(X + h * k3)
    return X + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def strang_trajectory_0d(model, z0, mu, T, dt, p=None, n_out=V2_N_OUT):
    """The 0-D Strang trajectory at `t_out = linspace(0, T, n_out+1)`, as (n_out+1, dim).

    ONE STEP, identical in structure to `lifted.simulate_lifted`: gates dt/2 EXACT at frozen x
    (`lifted.gate_step_exact`) -> x over dt with the gates frozen -> gates dt/2 exact. The
    fields are carried as 1x1 spatial grids precisely so the production stepper's own helpers
    (`gate_step_exact`, `_reaction_from_gates_np`) are called VERBATIM; the only substitution is
    ETDRK4 -> RK4 for the x substep, there being no Laplacian in 0-D (CLAUDE.md §7c).

    dt IS SNAPPED so that T is hit exactly and every output time is a step boundary:
    `steps_per_out = round(T/dt/n_out)`, `n_steps = steps_per_out * n_out`,
    `dt_used = T/n_steps`. The snapped value is returned and it is the one `v2_temporal` fits
    the order against, so an inexact halving cannot bias the slope.

    Returns (traj, dt_used, n_steps).
    """
    if p is None:
        p = lifted._np_params(model)
    N = p["N"]
    if dt <= 0.0 or T <= 0.0:
        raise ValueError(f"strang_trajectory_0d needs dt > 0 and T > 0; got dt={dt}, T={T}")
    steps_per_out = max(1, int(round(T / dt / n_out)))
    n_steps = steps_per_out * n_out
    h = float(T) / n_steps

    x, GA, GR = lifted.unpack(z0, N)
    X = x[:, None, None].copy()
    GA = GA[:, :, None, None].copy()
    GR = GR[:, :, None, None].copy()
    beta = p["beta"][:, None, None]
    delta = p["delta"][:, None, None]

    traj = np.empty((n_out + 1, lifted.state_dim(N)))
    traj[0] = np.asarray(z0, float)
    for s in range(n_steps):
        GA, GR = lifted.gate_step_exact(model, X, GA, GR, 0.5 * h, mu)
        X = _rk4_reaction(X, lifted._reaction_from_gates_np(model, GA, GR), beta, delta, h)
        GA, GR = lifted.gate_step_exact(model, X, GA, GR, 0.5 * h, mu)
        if (s + 1) % steps_per_out == 0:
            traj[(s + 1) // steps_per_out] = np.concatenate(
                [X.ravel(), GA.reshape(-1), GR.reshape(-1)])
    if not np.all(np.isfinite(traj)):
        raise RuntimeError(f"strang_trajectory_0d blew up at mu={mu}, dt={h}, T={T} "
                           f"(dt/mu = {h / mu:.3g}) -- non-finite state.")
    return traj, h, n_steps


def v2_temporal(model, mu: float, T: float, dts: list[float]) -> dict:
    """V2 rung (ii): the 0-D Strang stepper against the Radau reference, and its ORDER.

    `err(dt)` is the sup-norm over BOTH the output times and all N + 2N^2 components of
    |z_strang - z_radau| -- a floor measurement over the whole trajectory, like V0's and V1's,
    not a spot check at the horizon. `strang_order` is the mean over consecutive (coarse, fine)
    pairs of `log(err_c/err_f) / log(dt_c/dt_f)`; that reduces to the briefed
    `log2(err(dt)/err(dt/2))` for an exact halving and stays correct when
    `strang_trajectory_0d` snaps dt to divide T (see its docstring).

    `sup_err_at_horizon` is the FINEST dt's sup-norm error over all components at t = T. It is
    the quantity §5.3 V2 bars at 0.1 x pattern_floor; taking the max over all N + 2N^2
    components rather than over x alone is the conservative reading of "sup-norm", and
    `sup_err_x_at_horizon` reports the x-only value beside it since pattern_floor is an
    x-amplitude.

    SCOPE. The order-2 claim holds where dt <~ mu. `dt_over_mu_max` reports the coarsest dt/mu
    actually used so a reader can check the scope was respected; at dt >> mu the exact gate
    substep still keeps the integration STABLE (that is what it is for) but the splitting order
    drops, which is a known property of the scheme and not a defect.

    Returns strang_order, strang_order_pairs, sup_err_at_horizon, sup_err_x_at_horizon,
    sup_err (dict dt_used -> err), n_steps (dict dt_used -> steps), dts_used, dt_over_mu_max,
    radau_ok, radau_self_err, radau_ref_margin, radau_nfev, radau_njev, radau_nlu, mu, T.
    """
    dts = sorted(float(d) for d in dts)[::-1]          # coarse -> fine
    if len(dts) < 2:
        raise ValueError(f"v2_temporal needs >=2 dt values to fit an order; got {len(dts)}")
    if dts[-1] <= 0.0:
        raise ValueError(f"v2_temporal needs strictly positive dt; got {dts}")

    p = lifted._np_params(model)
    _, z0 = v2_initial_state(model)
    t_out = np.linspace(0.0, float(T), V2_N_OUT + 1)
    fun = lambda t, z: _lifted_rhs_0d(z, mu, p)
    jac = lambda t, z: _lifted_jac_0d(model, z, mu)

    ref, sol = _radau(fun, z0, T, t_out, jac, V2_RADAU_RTOL, V2_RADAU_ATOL)
    tighter, _ = _radau(fun, z0, T, t_out, jac, V2_RADAU_CHECK_RTOL, V2_RADAU_CHECK_ATOL)
    radau_self_err = float(np.max(np.abs(ref - tighter)))

    errs, hs, nsteps, horizon, horizon_x = [], [], [], [], []
    for dt in dts:
        traj, h, ns = strang_trajectory_0d(model, z0, mu, T, dt, p=p)
        d = np.abs(traj - ref)
        errs.append(float(d.max()))
        horizon.append(float(d[-1].max()))
        horizon_x.append(float(d[-1, :model.N].max()))
        hs.append(h)
        nsteps.append(ns)

    if not all(e > 0.0 for e in errs):
        raise RuntimeError(
            "v2_temporal: a Strang error is exactly 0, so its log-log slope is undefined "
            f"(errs={errs}). Round-off has swallowed the truncation error -- raise T or "
            "coarsen dt rather than reading the returned order.")
    pairs = [float(np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1]))
             for i in range(len(hs) - 1)]

    radau_ok = bool(sol.success and np.all(np.isfinite(ref))
                    and radau_self_err <= V2_RADAU_MARGIN * min(errs))
    return dict(
        strang_order=float(np.mean(pairs)), strang_order_pairs=pairs,
        sup_err_at_horizon=horizon[-1], sup_err_x_at_horizon=horizon_x[-1],
        sup_err={h: e for h, e in zip(hs, errs)},
        n_steps={h: n for h, n in zip(hs, nsteps)},
        dts_used=hs, dt_over_mu_max=float(max(hs) / mu),
        radau_ok=radau_ok, radau_self_err=radau_self_err,
        radau_ref_margin=(float(min(errs) / radau_self_err) if radau_self_err > 0.0
                          else float("inf")),
        radau_nfev=int(sol.nfev), radau_njev=int(sol.njev), radau_nlu=int(sol.nlu),
        mu=float(mu), T=float(T))


def v2_qss_limit(model, mus, T: float = 5.0) -> dict:
    """V2 rung (i): the lifted trajectory converges to the QSS N-dim trajectory as mu -> 0.

    BOTH sides are integrated by the SAME Radau reference (the lifted N + 2N^2 dim system at
    each mu, the reduced N-dim system once), so the measured gap is a property of the LIFT and
    carries no stepper error. The initial condition is on the slow manifold for both
    (x0 = 1.5 x*, gates at gates_qss(x0)), so there is no initial layer and the gap is the
    O(mu) tracking lag -- the quantity the order claim is about.

    `qss_gap[mu]` is the sup-norm over output times and the N x-components of
    |x_lift - x_qss|. Only the x block is compared: the gate block's mu -> 0 counterpart is the
    ALGEBRAIC QSS relation, which V0 already checks (`max_gate_qss_err`), not a trajectory.
    `qss_order` is the least-squares slope of log gap against log mu -- a MEASUREMENT (§5.3:
    absolute error constants UNCALIBRATED), reported here, not enforced.

    Returns qss_order, qss_gap (dict mu -> sup err), qss_gap_rel (the same divided by
    max|x_qss|), radau_ok, radau_nfev (dict mu -> nfev, the QSS solve under key 0.0), T.
    """
    mus = np.sort(np.asarray(mus, float))
    if len(mus) < 2:
        raise ValueError(f"v2_qss_limit needs >=2 mu values to fit a log-log slope; "
                         f"got {len(mus)}")
    if np.any(mus <= 0.0):
        raise ValueError(f"v2_qss_limit needs strictly positive mu; got {mus.tolist()}")

    p = lifted._np_params(model)
    N = model.N
    x0, z0 = v2_initial_state(model)
    t_out = np.linspace(0.0, float(T), V2_N_OUT + 1)

    xq, solq = _radau(lambda t, x: _qss_rhs_0d(model, x, p), x0, T, t_out,
                      lambda t, x: _qss_jac_0d(model, x),
                      V2_RADAU_RTOL, V2_RADAU_ATOL)
    scale = max(float(np.max(np.abs(xq))), 1e-300)

    gaps = {}
    nfev = {0.0: int(solq.nfev)}
    ok = bool(solq.success)
    for mu in mus:
        zl, soll = _radau(lambda t, z: _lifted_rhs_0d(z, float(mu), p), z0, T, t_out,
                          lambda t, z: _lifted_jac_0d(model, z, float(mu)),
                          V2_RADAU_RTOL, V2_RADAU_ATOL)
        gaps[float(mu)] = float(np.max(np.abs(zl[:, :N] - xq)))
        nfev[float(mu)] = int(soll.nfev)
        ok = ok and bool(soll.success) and bool(np.all(np.isfinite(zl)))

    g = np.array([gaps[float(m)] for m in mus])
    if not np.all(g > 0.0):
        raise RuntimeError(
            f"v2_qss_limit: a lifted-vs-QSS gap is exactly 0, so its log-log slope is "
            f"undefined (gaps={gaps}). Raise T rather than reading the returned order.")
    return dict(qss_order=float(np.polyfit(np.log(mus), np.log(g), 1)[0]),
                qss_gap=gaps, qss_gap_rel={k: v / scale for k, v in gaps.items()},
                radau_ok=ok, radau_nfev=nfev, T=float(T))


def v2_rhs_mirror_error(model, mus=(1e-6, 1e-3, 1.0), n_states: int = 25,
                        seed: int = 0) -> float:
    """Max RELATIVE deviation of `_lifted_rhs_0d` from `lifted.lifted_rhs_torch`.

    The numpy mirror exists only for Radau's speed (see `_lifted_rhs_0d`); this is what pins it
    to the torch original rather than trusting the transcription. RELATIVE because the gate
    rows carry a 1/mu factor, so an absolute deviation is not comparable across the mu range.
    States are the QSS state at a random x with the gate block randomly rescaled, so both the
    `free`/`1-G` gate normalisations and the production terms are exercised away from the
    fixed point.
    """
    p = lifted._np_params(model)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(int(n_states)):
        x = rng.uniform(0.05, 2.0, model.N)
        z = lifted.lifted_state(model, x)
        z[model.N:] *= rng.uniform(0.5, 1.5, z.size - model.N)
        zt = torch.as_tensor(z, device=model.device, dtype=model.dtype)
        for mu in mus:
            with torch.no_grad():
                b = lifted.lifted_rhs_torch(model, zt, float(mu)).cpu().numpy()
            a = _lifted_rhs_0d(z, float(mu), p)
            worst = max(worst, float(np.max(np.abs(a - b)))
                        / max(float(np.max(np.abs(b))), 1e-300))
    return worst
