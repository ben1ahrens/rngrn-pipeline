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
"""
from __future__ import annotations

import numpy as np
import torch
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
