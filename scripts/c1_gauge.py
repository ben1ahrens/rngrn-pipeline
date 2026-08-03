"""c1_gauge.py — WHICH DIRECTIONS IN J DOES THE OBJECTIVE ACTUALLY CONSTRAIN?

Criterion 3.1 is measured (docs/C1_COMPETITIVE_TUNING.md sec 7.2) to be a SEPARATE failure
from the Turing rate: on legacy data all 8 seeds pattern and agree on k* to 1.5 %, and
`topology_consistency` is still 0.125 with 8 distinct sign structures. This script asks the
mechanistic question that follows -- not "how do we raise the rate" but "what about J is the
objective even able to see".

THE ARGUMENT, which BLOCK 1 then verifies numerically rather than asserting.

Every J-dependent term in the trained objective is one of:

  losses/terms.py::kstar_anchor          -> J only via sigma(k) = max Re eig(J - k^2 D)
  losses/terms.py::turing_hinges_split   -> J only via sigma(k)
  losses/terms.py::anticollapse          -> softplus_hinge(jac_floor - ||J||_F)

(`frame_scale_anchor` sees x* only, `param_prior` sees D/alpha/delta only, and `resid` has
weight 0 and is not even computed -- losses/total.py::compute_terms_batched.)

The anticollapse term is a ONE-SIDED hinge at jac_floor = 1.0, so it is exactly flat, with
exactly zero gradient, everywhere ||J||_F > 1. In that region -- which is where every
recovered network in this project sits -- the objective is a function of sigma(k) ALONE.

D is diagonal. Therefore, for any k:

  (a) TRANSPOSE.            (J - k^2 D)^T = J^T - k^2 D, and a matrix and its transpose have
                            the same spectrum. So J and J^T give IDENTICAL sigma(k) at every
                            k. The objective cannot distinguish a network from the one with
                            every regulatory edge REVERSED.
  (b) DIAGONAL SIMILARITY.  For S = diag(s), s > 0: S(J - k^2 D)S^-1 = S J S^-1 - k^2 D,
                            because a diagonal S commutes with the diagonal D. Same spectrum
                            at every k. So J_ij -> (s_i/s_j) J_ij is invisible. That is an
                            (N-1)-parameter CONTINUOUS gauge freedom (2 parameters at N=3).
  (c) NODE PERMUTATION.     J -> P J P^T with D -> P D P^T. Already measured by the PERM
                            block of scripts/c1_analyse.py.

A dimension count that agrees: sigma(k) for all k fixes the characteristic polynomial of
J - k^2 D, i.e. 9 coefficients in k^2 at N=3, of which 7 are independent functions of J
(the 3 diagonal entries, the 3 2x2 principal minors, and det J). 9 entries - 7 constraints
= a 2-dimensional degenerate manifold -- exactly the dimension of the diagonal-similarity
group modulo its scalar centre. So (a)-(c) are not merely SOME degeneracies; together with
the count they are the WHOLE of the objective's blindness in J.

WHY THIS MATTERS FOR THE PRE-REGISTERED STATISTIC. `topology_consistency` zeroes entries
with |J_ij| < rtol * max|J| and then compares sign matrices ENTRYWISE. Transpose changes
which entry carries a sign. Diagonal similarity changes the MAGNITUDES -- so it changes
which entries fall under the rtol threshold -- while leaving every sign unchanged. Both
therefore inflate `n_distinct_structures` without the seeds having found different physics.

BLOCK 2 measures how much of the measured 3.1 failure each symmetry accounts for, by
recomputing the identical modal-fraction statistic after quotienting by a growing group:

  raw  ->  +permutation  ->  +permutation,transpose  ->  +permutation,transpose,balancing

`balancing` is scipy.linalg.matrix_balance, which picks a canonical representative of the
diagonal-similarity orbit (Osborne's algorithm equalises row and column norms). It is a
gauge FIXING, not a rescue.

*** NONE OF BLOCK 2's QUOTIENTED COLUMNS IS CRITERION 3.1. ***
PREREGISTRATION section 3.1's statistic is the RAW modal fraction at rtol 0.05, read against
0.75, together with the size-matched cross-target control and the >= 0.25 gap. That is the
number reported against the bar; it is the `raw` column here and it is printed first. These
columns say how much of the failure is estimator gauge and how much is genuinely different
recovered physics. Every level is reported WITH its own cross-target control, because a
quotient that lifts the within-target number and the cross-target number equally has
explained nothing -- it has just made the statistic coarser.

Usage:  python scripts/c1_gauge.py [cell ...]
        (filter to cells sharing one runs-root/dataset; see c1_analyse.py's note)
"""
import itertools
import json
import random
import sys
from collections import Counter

import numpy as np
from scipy.linalg import matrix_balance

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from c1_analyse import RTOLS, load_cells, results_of, sign_key  # noqa: E402

N_CROSS_DRAWS = 4000


# ======================================================================================
# BLOCK 1 — verify the claimed symmetries numerically, on real recovered (J, D)
# ======================================================================================
def sigma(J, D, kgrid):
    """max Re eig(J - k^2 D) over the k-grid — the ONLY channel through which the trained
    objective sees J (see the module docstring)."""
    return np.array([max(np.linalg.eigvals(J - k * k * np.diag(D)).real) for k in kgrid])


def block_verify(reps, n_show=3):
    print("\n=== BLOCK 1 — are the claimed symmetries EXACT? (max |delta sigma(k)| over the grid) ===")
    print("    A symmetry is only interesting if it leaves sigma(k) invariant to machine")
    print("    precision. Measured on REAL recovered (J, D), not on synthetic matrices.")
    hdr = (f"{'cell':<20}{'sample':<12}{'seed':>5}{'||J||_F':>9}{'transpose':>12}"
           f"{'diag-sim':>12}{'perm':>10}{'sigma scale':>13}")
    print(hdr)
    print("-" * len(hdr))
    kgrid = np.linspace(0.0, 2.0, 128)
    rng = np.random.default_rng(0)
    shown = 0
    for (cell, samp), r in sorted(reps.items()):
        for seed, d in results_of(r):
            J = np.asarray(d["recovered"]["J"], float)
            D = np.asarray(d["recovered"]["D_model"], float)
            s0 = sigma(J, D, kgrid)
            scale = float(np.abs(s0).max())
            d_t = float(np.abs(sigma(J.T, D, kgrid) - s0).max())
            s = np.exp(rng.normal(0, 1.0, size=J.shape[0]))          # a random gauge element
            d_s = float(np.abs(sigma(np.diag(s) @ J @ np.diag(1 / s), D, kgrid) - s0).max())
            p = [2, 0, 1]
            d_p = float(np.abs(sigma(J[np.ix_(p, p)], D[p], kgrid) - s0).max())
            print(f"{cell:<20}{samp:<12}{seed:>5}{np.linalg.norm(J):>9.3f}"
                  f"{d_t:>12.2e}{d_s:>12.2e}{d_p:>10.2e}{scale:>13.4f}")
            shown += 1
            if shown >= n_show:
                return


# ======================================================================================
# BLOCK 2 — the consistency ladder, each level with its own cross-target control
# ======================================================================================
def canon_key(J, rtol, transpose=False, balance=False):
    """Sign structure canonicalised over the requested symmetry group.

    Always quotients by the 6 node relabelings (the symmetry c1_analyse's PERM block
    already measured); `transpose` adds J -> J^T; `balance` gauge-fixes the diagonal
    similarity first, via scipy's Osborne balancing, before any thresholding — which is the
    point, since the rtol threshold is what the magnitude gauge corrupts.
    """
    if balance:
        try:
            J = matrix_balance(J, permute=False)[0]
        except Exception:
            pass                      # a singular/degenerate balance leaves J as it is
    cands = [J, J.T] if transpose else [J]
    n = J.shape[0]
    return min(sign_key(M[np.ix_(list(p), list(p))], rtol)
               for M in cands for p in itertools.permutations(range(n)))


LEVELS = [
    ("raw",        lambda J, t: sign_key(J, t)),                                # CRITERION 3.1
    ("+perm",      lambda J, t: canon_key(J, t)),
    ("+transp",    lambda J, t: canon_key(J, t, transpose=True)),
    ("+balance",   lambda J, t: canon_key(J, t, transpose=True, balance=True)),
]


def block_ladder(reps, rtol):
    groups = {}
    for (cell, samp), r in sorted(reps.items()):
        Js = [np.asarray(d["recovered"]["J"], float) for _, d in results_of(r)]
        if Js:
            groups[(cell, samp)] = Js
    print(f"\n=== BLOCK 2 — consistency ladder at sign-zero rtol {rtol} "
          f"({'THE PRE-REGISTERED TOLERANCE' if rtol == 0.05 else 'sensitivity'}) ===")
    print("    `raw` IS criterion 3.1. The rest are gauge diagnostics and no bar reads them.")
    hdr = f"{'cell':<20}{'sample':<12}{'K':>3}" + "".join(
        f"{n:>11}{'ctrl':>7}{'gap':>7}" for n, _ in LEVELS)
    print(hdr)
    print("-" * len(hdr))
    rng = random.Random(0)
    for key, Js in sorted(groups.items()):
        cell, samp = key
        row = f"{cell:<20}{samp:<12}{len(Js):>3}"
        for _, fn in LEVELS:
            keys = [fn(J, rtol) for J in Js]
            within = Counter(keys).most_common(1)[0][1] / len(keys)
            others = [k for k in groups if k[1] != samp]
            if len(others) >= 2:
                vals = []
                for _ in range(N_CROSS_DRAWS):
                    pick = others if len(others) <= len(Js) else rng.sample(others, len(Js))
                    mixed = [fn(rng.choice(groups[o]), rtol) for o in pick]
                    vals.append(Counter(mixed).most_common(1)[0][1] / len(mixed))
                cross = sum(vals) / len(vals)
                row += f"{within:>11.3f}{cross:>7.3f}{within - cross:>7.3f}"
            else:
                row += f"{within:>11.3f}{'-':>7}{'-':>7}"
        print(row)
    if len(set(k[1] for k in groups)) < 2:
        print("  (cross-target control needs >= 2 distinct samples; not computable yet)")


# ======================================================================================
# BLOCK 3 — is the pre-registered rtol landing in the middle of the entry distribution?
# ======================================================================================
RTOL_SWEEP = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50)


def block_threshold(reps):
    """Where the rtol cut sits relative to |J_ij|/max|J|, and how much the criterion moves.

    `topology_consistency` calls an entry an edge iff |J_ij| >= rtol * max|J|. That is a HARD
    cut, so entries sitting near it are assigned by numerical noise rather than by biology,
    and two seeds whose networks agree on every dynamically material edge can still be scored
    as two distinct structures. This block reports (i) where the entry-magnitude distribution
    actually sits relative to the cut and (ii) the criterion recomputed across a wide rtol
    sweep, so its stability as an estimator is visible.

    PREREGISTRATION section 3.1 already flags the tolerance UNCALIBRATED and requires
    0.02/0.05/0.10 to be reported. This measures HOW uncalibrated. It changes no bar: 0.05
    remains the tolerance the 0.75 bar is read at.
    """
    print("\n=== BLOCK 3 — the rtol cut against the entry-magnitude distribution ===")
    print("    A cut near the MEDIAN maximises the number of coin-flip entries.")
    hdr = (f"{'cell':<20}{'sample':<12}{'K':>3}{'q25':>8}{'q50':>8}{'q75':>8}"
           f"{'near_cut':>10}{'min':>7}{'max':>7}{'range':>7}")
    print(hdr)
    print("-" * len(hdr))
    for (cell, samp), r in sorted(reps.items()):
        Js = [np.asarray(d["recovered"]["J"], float) for _, d in results_of(r)]
        if not Js:
            continue
        rel = np.concatenate([np.abs(J).ravel() / np.abs(J).max() for J in Js])
        # "near the cut" = within a factor of 3 either side of rtol 0.05
        near = float(np.mean((rel > 0.05 / 3) & (rel < 0.05 * 3)))
        vals = [Counter([sign_key(J, t) for J in Js]).most_common(1)[0][1] / len(Js)
                for t in RTOL_SWEEP]
        print(f"{cell:<20}{samp:<12}{len(Js):>3}"
              f"{np.quantile(rel, .25):>8.4f}{np.quantile(rel, .50):>8.4f}"
              f"{np.quantile(rel, .75):>8.4f}{near:>10.3f}"
              f"{min(vals):>7.3f}{max(vals):>7.3f}{max(vals) - min(vals):>7.3f}")
    print("    topology_consistency across the rtol sweep " +
          "/".join(str(t) for t in RTOL_SWEEP) + ":")
    for (cell, samp), r in sorted(reps.items()):
        Js = [np.asarray(d["recovered"]["J"], float) for _, d in results_of(r)]
        if not Js:
            continue
        vals = [Counter([sign_key(J, t) for J in Js]).most_common(1)[0][1] / len(Js)
                for t in RTOL_SWEEP]
        print(f"      {cell:<20}{samp:<12}" + " ".join(f"{v:.3f}" for v in vals))


# ======================================================================================
# BLOCK 4 — the gauge-INVARIANT physics, which is what the objective can actually pin
# ======================================================================================
def invariants(J):
    """The diagonal-similarity invariants of J's off-diagonal structure, at N=3.

    Under J -> S J S^-1 with S = diag(s), J_ij -> (s_i/s_j) J_ij, so the invariant
    polynomials are the diagonal entries, the 2-cycle products J_ij*J_ji, and the two
    3-cycle products (which transpose swaps). Their SIGNS are the gauge-invariant analogue
    of a topology: sign(J_ii) is self-activation/self-repression, sign(J_ij*J_ji) is the
    feedback SENSE of a pair (mutual activation, mutual repression, or mixed), and the
    3-cycle sign is the sense of the loop. This is 3+3+1 = 7 sign bits against the raw
    statistic's 9, and 7 is exactly the number of independent functions of J that
    sigma(k) fixes.
    """
    d = np.sign(np.diag(J))
    p = np.sign([J[0, 1] * J[1, 0], J[0, 2] * J[2, 0], J[1, 2] * J[2, 1]])
    c = np.sign(J[0, 1] * J[1, 2] * J[2, 0])
    return d, p, np.array([c])


def block_invariant(reps):
    print("\n=== BLOCK 4 — DIAGNOSTIC: agreement on the GAUGE-INVARIANT sign content ===")
    print("    NOT criterion 3.1 and coarser than it (7 sign bits, not 9, and node identity")
    print("    is quotiented out). Read ONLY against its cross-target control: a statistic")
    print("    this coarse can agree by inductive bias alone.")
    hdr = (f"{'cell':<20}{'sample':<12}{'K':>3}{'diag':>10}{'pairs':>10}"
           f"{'3cycle':>10}{'all7':>10}{'ctrl':>7}{'gap':>7}")
    print(hdr)
    print("-" * len(hdr))
    groups = {}
    for (cell, samp), r in sorted(reps.items()):
        Js = [np.asarray(d["recovered"]["J"], float) for _, d in results_of(r)]
        if Js:
            groups[(cell, samp)] = Js

    def full(J):
        d, p, c = invariants(J)
        return tuple(np.sort(d)) + tuple(np.sort(p)) + tuple(c)

    rng = random.Random(0)
    for (cell, samp), Js in sorted(groups.items()):
        def modal(fn, JJ):
            return Counter(fn(J) for J in JJ).most_common(1)[0][1] / len(JJ)
        row = (f"{cell:<20}{samp:<12}{len(Js):>3}"
               f"{modal(lambda J: tuple(np.sort(invariants(J)[0])), Js):>10.3f}"
               f"{modal(lambda J: tuple(np.sort(invariants(J)[1])), Js):>10.3f}"
               f"{modal(lambda J: tuple(invariants(J)[2]), Js):>10.3f}"
               f"{modal(full, Js):>10.3f}")
        others = [k for k in groups if k[1] != samp]
        if len(others) >= 2:
            vals = []
            for _ in range(N_CROSS_DRAWS):
                pick = others if len(others) <= len(Js) else rng.sample(others, len(Js))
                mixed = [full(rng.choice(groups[o])) for o in pick]
                vals.append(Counter(mixed).most_common(1)[0][1] / len(mixed))
            cross = sum(vals) / len(vals)
            row += f"{cross:>7.3f}{modal(full, Js) - cross:>7.3f}"
        else:
            row += f"{'-':>7}{'-':>7}"
        print(row)


def main():
    reps = load_cells(sys.argv[1:] or None)
    if not reps:
        print("no non-empty reports found")
        return
    block_verify(reps)
    for t in RTOLS:
        block_ladder(reps, t)
    block_threshold(reps)
    block_invariant(reps)


if __name__ == "__main__":
    main()
