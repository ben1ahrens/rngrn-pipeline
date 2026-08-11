"""reproducibility.py — cross-seed topology reproducibility scoring (SCORING SIDE).

WHY THIS EXISTS
---------------
The user's stated #1 success metric for this project is not "did recovery match the
generator's parameters" (explicitly NOT a goal) but: for a SINGLE target system, does
the model consistently learn the SAME topology across seeds? No metric for that existed
before this module. `validate.score_recovery` scores one run against the answer key;
this module scores K runs (K seeds, one target) against EACH OTHER — it needs no answer
key at all, and is therefore usable even on datasets with no ground-truth J.

WHAT "SAME TOPOLOGY" MEANS HERE
--------------------------------
Topology == the SIGN STRUCTURE of the recovered Jacobian J (excitatory / inhibitory /
absent edge), not its magnitude — consistent with the rest of this codebase, which
scores sign/regime and deliberately not parameter values (see validate.py's module
docstring). Two seeds "agree on the topology" when their sign structures are entrywise
equal.

THE NEAR-ZERO ENTRY DECISION — read before trusting a number from this module
-------------------------------------------------------------------------------
A Jacobian entry at 1e-12 has a sign but no meaning, and if it is counted as a real
edge, comparing two independently-fit seeds will inflate apparent agreement (both seeds
"agree" the entry is some arbitrary tiny sign, when really both are silent on that
edge). This module therefore maps entries below a RELATIVE magnitude threshold to a
third state, 0 ("no edge"), before comparing.

`DEFAULT_SIGN_ZERO_RTOL = 0.05` (5% of the matrix's own largest |J| entry) is used —
the same magnitude as `scoring.overparam.DEFAULT_COUPLING_THRESHOLD` (also 0.05), and
for the same underlying reason: nothing in the recovery objective pins J's overall
scale (see permutation.py's `fro_rel_err` docstrings), so only a per-matrix RELATIVE
cut is meaningful, and 5% of the strongest recovered edge is a reasonable starting
guess for "this edge is real" versus fit noise.

*** THIS THRESHOLD IS UNCALIBRATED — [TUNE]. *** It has not been checked against real
recovered-J score distributions (none of the fits behind the 15/16-seeds/0%-Turing
numbers in docs/CODE_REALITY.md were scored with this module). It is exposed as a
parameter (`sign_zero_rtol`) on every function here specifically so it can be swept
without editing code. This is the THIRD documented-uncalibrated threshold in the
project (alongside `overparam.DEFAULT_COUPLING_THRESHOLD` and `validate._sign_structure`
/ `permutation.SIGN_ZERO_RTOL`'s 1e-9 near-machine-epsilon cut) — note it is
DELIBERATELY coarser than the 1e-9 cut used elsewhere: 1e-9 catches values that are
zero to floating-point round-off (e.g. after a Newton solve), while 0.05 is meant to
catch values that are numerically nonzero but too small, relative to the matrix's own
scale, to read as a structurally meaningful edge. They answer different questions and
are not interchangeable.

THE topology_consistency DESIGN CHOICE — read before quoting this number
--------------------------------------------------------------------------
Multiple single scalars are defensible: mean pairwise agreement (average entrywise
sign agreement over all seed PAIRS) and modal fraction (the fraction of seeds that
produced the single most common exact sign structure) measure related but different
things — pairwise agreement rewards "close" partial agreement between every pair even
if no two seeds are IDENTICAL, while modal fraction answers the user's literal
question ("does the model consistently learn THE SAME topology?") directly and is
much stricter (a single off-diagonal-sign seed scores 0 contribution to modal_fraction
even if it agrees with every other seed on 8 of 9 entries).

THIS MODULE DEFINES `topology_consistency := modal_fraction` because it is the
literal, strictest reading of the user's stated metric. `mean_pairwise_agreement` is
still computed and returned alongside it for context (a run can have low
topology_consistency but high mean_pairwise_agreement, which is worth seeing). This is
a metric-definition choice that BIASES WHAT RESULTS MEAN and could reasonably go the
other way — it is called out here, in the PR body, and in worker notes rather than
picked silently.

FIREWALL
--------
Scoring side. This module compares RECOVERED Jacobians to each other; it never reads
an answer key and has no ground-truth dependency at all. Deliberately free of any
import of rngrn.model/recover/losses/data, matching the convention in permutation.py
and overparam.py.

Status: SCAFFOLD. Unit-tested on synthetic matrices with known permutations and known
sign patterns (tests/test_reproducibility_scoring.py); not validated as a measure of
recovery quality on real recovered fits.
"""
from __future__ import annotations

import json
from collections import Counter
from statistics import pstdev
from typing import Sequence

import numpy as np

__all__ = [
    "DEFAULT_SIGN_ZERO_RTOL",
    "sign_structure",
    "pairwise_sign_agreement",
    "modal_sign_structure",
    "edge_consensus",
    "coefficient_of_variation",
    "reproducibility_report",
    "per_run_fields",
]

#: UNCALIBRATED — see module docstring "THE NEAR-ZERO ENTRY DECISION". Fraction of a
#: matrix's own largest |J| entry below which an entry is treated as structurally zero.
DEFAULT_SIGN_ZERO_RTOL: float = 0.05


def _as_square(J, name: str) -> np.ndarray:
    arr = np.asarray(J, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite entries; refusing to score")
    return arr


def sign_structure(J, sign_zero_rtol: float = DEFAULT_SIGN_ZERO_RTOL) -> np.ndarray:
    """Elementwise sign of ``J`` with near-zero entries (relative to ``J``'s own max
    |entry|) collapsed to 0. See module docstring for why the threshold is relative
    and uncalibrated. Idempotent: applying this to an array already valued in
    {-1, 0, 1} returns it unchanged (used by ``optim.benchmark`` to re-hydrate sign
    vectors already thresholded once by ``per_run_fields``).
    """
    if not np.isfinite(sign_zero_rtol) or sign_zero_rtol < 0:
        raise ValueError(f"sign_zero_rtol must be finite and >= 0; got {sign_zero_rtol}")
    arr = _as_square(J, "J")
    scale = float(np.abs(arr).max())
    return np.sign(np.where(np.abs(arr) > sign_zero_rtol * (scale + 1e-12), arr, 0.0))


def _validate_J_list(J_list: Sequence, sign_zero_rtol: float) -> list[np.ndarray]:
    K = len(J_list)
    if K < 2:
        raise ValueError(
            f"reproducibility scoring needs K >= 2 recovered Jacobians to compare "
            f"seeds against each other; got K={K}. A single seed has nothing to be "
            f"reproducible WITH — this is not the same as a score of 1.0.")
    signs = [sign_structure(J, sign_zero_rtol) for J in J_list]
    shape0 = signs[0].shape
    for i, s in enumerate(signs):
        if s.shape != shape0:
            raise ValueError(
                f"all K Jacobians must share one shape (same target, same N); "
                f"J_list[0] is {shape0}, J_list[{i}] is {s.shape}")
    return signs


def pairwise_sign_agreement(J_list: Sequence,
                            sign_zero_rtol: float = DEFAULT_SIGN_ZERO_RTOL) -> dict:
    """Pairwise sign-structure agreement across K recovered Jacobians on one target.

    Returns
    -------
    dict with keys
        agreement_matrix : (K, K) list of lists  entrywise agreement fraction per pair
                                                   (diagonal is 1.0 by construction — a
                                                   matrix agrees with itself)
        mean_agreement : float                    mean over the K*(K-1)/2 OFF-DIAGONAL
                                                   pairs only (excludes the trivial
                                                   diagonal self-agreement)
        K : int
        sign_zero_rtol : float                    threshold actually used (provenance)
    """
    signs = _validate_J_list(J_list, sign_zero_rtol)
    K = len(signs)
    mat = np.empty((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            mat[i, j] = float(np.mean(signs[i] == signs[j]))
    off_diag = [mat[i, j] for i in range(K) for j in range(K) if i != j]
    return {
        "agreement_matrix": mat.tolist(),
        "mean_agreement": float(np.mean(off_diag)),
        "K": K,
        "sign_zero_rtol": float(sign_zero_rtol),
    }


def modal_sign_structure(J_list: Sequence,
                         sign_zero_rtol: float = DEFAULT_SIGN_ZERO_RTOL) -> dict:
    """The single most common EXACT sign structure among K recovered Jacobians.

    Ties (more than one structure attaining the maximum count) are broken by
    lexicographically-smallest flattened structure, which makes the result
    deterministic but the choice among tied structures ARBITRARY — a tie means the
    seeds genuinely did not converge on one topology, and ``modal_fraction`` reports
    how weak that "mode" actually is.

    Returns
    -------
    dict with keys
        modal_structure : (N, N) list of lists of {-1, 0, 1}
        modal_fraction : float             (count of the modal structure) / K
        modal_count : int
        n_distinct_structures : int        how many distinct exact structures occurred
        K : int
    """
    signs = _validate_J_list(J_list, sign_zero_rtol)
    K = len(signs)
    n = signs[0].shape[0]
    keys = [tuple(int(v) for v in s.flatten()) for s in signs]
    counts = Counter(keys)
    best_count = max(counts.values())
    modal_key = min(k for k, c in counts.items() if c == best_count)
    return {
        "modal_structure": np.array(modal_key, dtype=int).reshape(n, n).tolist(),
        "modal_fraction": float(best_count) / K,
        "modal_count": int(best_count),
        "n_distinct_structures": len(counts),
        "K": K,
    }


def edge_consensus(J_list: Sequence,
                   sign_zero_rtol: float = DEFAULT_SIGN_ZERO_RTOL) -> dict:
    """Per-edge consensus: for each (i, j), the fraction of the K seeds agreeing on
    its sign (independently of whether the REST of the matrix also agrees — this is
    a weaker, entrywise-local statistic that ``modal_fraction`` deliberately is not).

    Returns
    -------
    dict with keys
        consensus_matrix : (N, N) list of lists   majority-sign fraction per entry,
                                                    in [1/3, 1.0] for K large (the
                                                    minimum possible majority share
                                                    among 3 sign classes)
        majority_sign_matrix : (N, N) list of lists of {-1, 0, 1}   the majority sign
                                                    per entry (ties broken toward 0,
                                                    i.e. "no consensus edge", then
                                                    toward the smaller signed value)
        K : int
    """
    signs = _validate_J_list(J_list, sign_zero_rtol)
    K = len(signs)
    n = signs[0].shape[0]
    stack = np.stack(signs, axis=0)  # (K, N, N)
    consensus = np.empty((n, n), dtype=float)
    majority = np.empty((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            counts = Counter(int(v) for v in stack[:, i, j])
            best = max(counts.values())
            # tie-break toward 0 first (no consensus), then toward the smaller sign
            candidates = sorted(k for k, c in counts.items() if c == best)
            chosen = 0 if 0 in candidates else candidates[0]
            consensus[i, j] = float(best) / K
            majority[i, j] = chosen
    return {
        "consensus_matrix": consensus.tolist(),
        "majority_sign_matrix": majority.tolist(),
        "K": K,
    }


def coefficient_of_variation(values: Sequence[float], name: str) -> float:
    """Population-CV (``pstdev / mean``) of a K-length sequence, K >= 2.

    Uses population (not sample) standard deviation for consistency with
    ``optim.benchmark``'s existing ``kstar_identifiability_std`` (also ``pstdev``).
    Raises rather than dividing by zero/negative — kstar and D-ratio are both
    strictly positive physical quantities, so a non-positive mean means the inputs
    were not what this function expects.
    """
    vals = [float(v) for v in values]
    if len(vals) < 2:
        raise ValueError(f"{name}: need K >= 2 values for a spread; got {len(vals)}")
    m = sum(vals) / len(vals)
    if not np.isfinite(m) or m <= 0.0:
        raise ValueError(f"{name}: mean must be finite and positive to define a "
                         f"coefficient of variation, got mean={m!r} over {vals!r}")
    return float(pstdev(vals) / m)


def reproducibility_report(J_list: Sequence, kstar_list: Sequence[float],
                           Dratio_list: Sequence[float],
                           sign_zero_rtol: float = DEFAULT_SIGN_ZERO_RTOL) -> dict:
    """Full cross-seed reproducibility report for K seeds recovered on ONE target.

    Parameters
    ----------
    J_list : sequence of K (N, N) array_like
        Recovered Jacobians, one per seed, all for the SAME target system.
    kstar_list : sequence of K float
        Recovered kstar_model, one per seed, same seed order as J_list.
    Dratio_list : sequence of K float
        Recovered D-ratio (max(D)/min(D), see ``per_run_fields``), same seed order.
    sign_zero_rtol : float
        See module docstring — UNCALIBRATED, [TUNE].

    Returns
    -------
    dict combining ``pairwise_sign_agreement``, ``modal_sign_structure`` and
    ``edge_consensus`` (each flattened into this dict, see their own docstrings for
    key meanings), plus:
        topology_consistency : float   == modal_fraction. See "THE topology_consistency
                                        DESIGN CHOICE" in the module docstring — this is
                                        a metric-definition decision, not a mechanical
                                        default.
        kstar_spread : float            coefficient_of_variation(kstar_list)
        Dratio_spread : float           coefficient_of_variation(Dratio_list)

    Raises
    ------
    ValueError
        K < 2 in any of J_list/kstar_list/Dratio_list, mismatched K across the three,
        non-square/non-finite J, mismatched N across J_list, or a non-positive mean
        kstar/D-ratio (see coefficient_of_variation).
    """
    K = len(J_list)
    if not (len(kstar_list) == K and len(Dratio_list) == K):
        raise ValueError(
            f"J_list, kstar_list, Dratio_list must have equal length (one seed each); "
            f"got {K}, {len(kstar_list)}, {len(Dratio_list)}")

    out: dict = {}
    out.update(pairwise_sign_agreement(J_list, sign_zero_rtol))
    out.update(modal_sign_structure(J_list, sign_zero_rtol))
    out.update(edge_consensus(J_list, sign_zero_rtol))
    out["topology_consistency"] = out["modal_fraction"]  # see module docstring
    out["kstar_spread"] = coefficient_of_variation(kstar_list, "kstar")
    out["Dratio_spread"] = coefficient_of_variation(Dratio_list, "Dratio")
    return out


def per_run_fields(J_rec, D_rec, kstar_model: float,
                   sign_zero_rtol: float = DEFAULT_SIGN_ZERO_RTOL) -> dict:
    """The PER-RUN pieces this module needs, for one recovered model, flattened for
    the run index (see io.py: run-index rows are flat scalars).

    Called from ``validate.score_recovery`` on EVERY run (independent of whether an
    answer key exists — this module never needs one). ``optim.benchmark`` later reads
    these fields back across the K runs for one target and calls
    ``reproducibility_report`` on them.

    D-ratio convention: ``max(D_rec) / min(D_rec)``, i.e. the ratio between the
    fastest- and slowest-diffusing recovered species (generalises the standard
    activator/inhibitor Dv/Du ratio to N > 2 species; for N == 2 it equals
    max(Du,Dv)/min(Du,Dv), i.e. it is order-invariant, unlike Dv/Du). Raises if any
    recovered D is non-positive — a diffusion coefficient must be positive.

    ``repro_sign_vector`` is stored as a JSON string (a flattened row-major list of
    {-1, 0, 1} of length N*N), following the existing precedent of stringifying a
    non-scalar for run-index storage (``validate.score_recovery``'s ``best_perm``).
    It round-trips via ``json.loads`` + ``np.reshape(..., (repro_N, repro_N))``, and
    ``sign_structure`` is idempotent on it (see that function's docstring), so it can
    be fed straight back into ``reproducibility_report`` as a "Jacobian".

    **AND THAT IDEMPOTENCE IS EXACTLY WHY THE RAW J IS ALSO STORED (D-EVID-12).**
    ``sign_structure`` is a no-op on an already-collapsed matrix for ANY
    ``sign_zero_rtol < 1``: the entries are 0 or ±1 and the scale is 1, so the comparison
    ``|arr| > rtol * scale`` never changes an outcome. Re-thresholding
    ``repro_sign_vector`` downstream therefore CANNOT change anything — which silently
    turned ``docs/PREREGISTRATION.md`` §3.1's committed 0.02 / 0.05 / 0.10 sensitivity
    sweep into three identical numbers under three different labels. The threshold is
    applied ONCE, here, and the information needed to apply a different one was being
    discarded at that moment. ``repro_J_vector`` preserves it: the raw Jacobian,
    uncollapsed, so any consumer can re-threshold honestly. It costs N*N floats (9 at
    N=3) as a JSON string, i.e. a flat scalar for run-index purposes.

    Returns
    -------
    dict with keys
        repro_J_vector : str           JSON list of floats, length N*N, row-major — the
                                        RAW Jacobian, so sign_zero_rtol can be re-applied
                                        downstream (D-EVID-12)
        repro_sign_vector : str        JSON list of ints, length N*N, row-major
        repro_N : int                  N, needed to reshape either vector back
        repro_kstar : float            == kstar_model, echoed under this module's
                                        naming for the benchmark aggregation to key on
        repro_D_ratio : float
        repro_sign_zero_rtol : float   threshold actually used (provenance)
    """
    D = np.asarray(D_rec, dtype=float)
    if D.ndim != 1 or D.size == 0:
        raise ValueError(f"D_rec must be a non-empty 1-D array of diffusion "
                         f"coefficients, got shape {D.shape}")
    if not np.all(np.isfinite(D)) or np.any(D <= 0.0):
        raise ValueError(f"D_rec must be finite and strictly positive; got {D.tolist()}")

    sign = sign_structure(J_rec, sign_zero_rtol)
    n = sign.shape[0]
    J_raw = np.asarray(J_rec, dtype=float)
    return {
        # the RAW Jacobian, so a DIFFERENT sign_zero_rtol can be applied later.
        # Re-thresholding repro_sign_vector cannot do this — see the docstring.
        "repro_J_vector": json.dumps([float(v) for v in J_raw.flatten()]),
        "repro_sign_vector": json.dumps([int(v) for v in sign.flatten()]),
        "repro_N": int(n),
        "repro_kstar": float(kstar_model),
        "repro_D_ratio": float(D.max() / D.min()),
        "repro_sign_zero_rtol": float(sign_zero_rtol),
    }
