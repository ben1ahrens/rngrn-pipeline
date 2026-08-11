"""lgen.py — L-GENERALISATION scoring (unit 12).

The requirement is "the recovered model must generalise across spatial domain sizes L".
That sentence needs a definition before it can be measured, because L is DEGENERATE with
the recovered parameter vector: feed a grid whose true extent is L but label it sL, and
every second derivative scales by 1/s**2, so the stationarity residual is satisfied
identically by D -> D*s**2. Nothing in a single frame can break that. Matas-Gil & Endres
(arXiv:2309.06339) never hit this because they hard-code one L via dx and never vary it;
their unit-square normalisation is commented out in their released source
(models/RBFPINNs_ChemPat.py). So the definition below is ours, and it is a decision:

    "generalises across L" == for ONE physical system imaged at SEVERAL domain sizes, the
    recovered PHYSICAL object agrees across those sizes.

The physical object has three L-independent handles, and this module scores all three:

  * the J SIGN STRUCTURE          — the topology, the thing the project actually wants.
  * the PHYSICAL D-RATIO          — max(D)/min(D). Dimensionless, hence identical whether
                                    recovery ran dimensionally or on the unit box.
  * the PHYSICAL k*               — the same system has the same physical wavelength
                                    regardless of how big a window you image it through.

Deliberately NOT scored here: the dimensionless wavenumber q = k*L/2pi (periods per box).
q is *not* invariant across L for a fixed system — it is precisely the thing that changes —
and on the currently registered three_gene data q is exactly 6.000 for all 127 samples
(the generator sets L = clip(6*2pi/k*, 18, 220)), so "recovering q" there is trivially
perfect and means nothing. q is recorded per run as a DIAGNOSTIC, never aggregated as a
success statistic.

--------------------------------------------------------------------------------------
DECISIONS LEFT OPEN ON PURPOSE (they would bias what a number means):
  1. The agreement statistic is entrywise MODAL agreement (below). Pairwise-mean agreement
     or a Hamming-distance-to-consensus would give different numbers on the same runs.
  2. The control is WITHIN-L across-seed agreement. Reading cross-L agreement against zero
     would be meaningless — seeds already disagree at a single L (docs/STATE_OF_THE_SCIENCE
     records 7/8 and 0/8 seed outcomes on held-out samples). Whether that is the RIGHT
     control, versus e.g. agreement between two disjoint seed halves at one L, is not
     settled here.
  3. NO PASS THRESHOLD is defined. This module reports agreement and its control; it does
     not declare a system "L-generalising".
Do not paper over these by picking a default later without saying so.
--------------------------------------------------------------------------------------

WIRING. `per_run_lgen_metrics` is called from validate.score_recovery (one out.update
line) so every run carries its own pieces into experiments/runs.jsonl. `lgen_table` then
aggregates the ledger across runs, exactly as optim/benchmark.py does for the other
cross-run tables. It is deliberately a separate entry point rather than an edit to
benchmark.py, so it can be adopted there in one import line without a merge conflict.
"""
from __future__ import annotations
import math
from collections import Counter, defaultdict
from statistics import mean, pstdev

import numpy as np

TWO_PI = 2.0 * math.pi

# Columns of the cross-L consistency table, in reading order.
LGEN_COLUMNS = [
    "system_id", "nondim", "form", "n_model", "n_runs", "n_L", "L_values",
    "sign_agree_cross_L", "sign_agree_within_L", "sign_agree_gap",
    "D_ratio_log10_std_cross_L", "D_ratio_log10_std_within_L",
    "kstar_log10_std_cross_L", "kstar_log10_std_within_L",
]


# --------------------------------------------------------------------------------------
# per-run pieces (recorded into the run index by validate.score_recovery)
# --------------------------------------------------------------------------------------
def sign_string(J, rel_tol: float = 1e-9) -> str:
    """Flatten a Jacobian's SIGN STRUCTURE to a comparable string of '+', '-', '0'.

    An entry counts as zero when |J_ij| <= rel_tol * max|J|, the same relative rule
    validate._sign_structure uses, so the two never disagree about what "no edge" means.
    A string (not an array) because the run index stores flat scalars only.

    Raises on a non-square, non-2-D or non-finite J. That is deliberate and it does abort
    the scoring of that run: a non-finite Jacobian at the recovered x* means
    `recovered_sig_max` and `recovered_turing` are already meaningless for that run, and
    house style is to raise rather than append a row that looks like a result. A silently
    reshaped J would compare unrelated entries across runs, which is worse still.
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"sign_string needs a square 2-D Jacobian, got shape {J.shape}")
    if not np.isfinite(J).all():
        raise ValueError("sign_string: Jacobian contains non-finite entries")
    thresh = rel_tol * (np.abs(J).max() + 1e-12)
    s = np.where(np.abs(J) > thresh, np.sign(J), 0.0).ravel()
    return "".join("+" if v > 0 else ("-" if v < 0 else "0") for v in s)


def d_ratio(D) -> float:
    """max(D)/min(D): the dimensionless diffusivity contrast.

    This is THE L-invariant of the diffusion vector — rescaling the domain multiplies every
    D_i by the same L**2, so the ratio is untouched. It is also the quantity Turing theory
    actually constrains (differential diffusivity), which is why it is preferred here over
    any individual D_i.

    Raises on a non-positive D: the model parameterises D = exp(theta) > 0, so a
    non-positive value means the caller passed something that is not a recovered D.
    """
    D = np.asarray(D, dtype=float).ravel()
    if D.size == 0:
        raise ValueError("d_ratio: empty diffusivity vector")
    if not np.isfinite(D).all() or (D <= 0).any():
        raise ValueError(f"d_ratio: diffusivities must be finite and positive, got {D!r}")
    return float(D.max() / D.min())


def per_run_lgen_metrics(result, J_rec, D_phys, system_id=None) -> dict:
    """The per-run pieces of the cross-L consistency metric. Flat scalars + strings.

    Parameters
    ----------
    result : RecoveryResult (or any object exposing the same attributes). Read for
        `L`, `nondim` and `q_model`; all three are looked up with getattr defaults so a
        duck-typed stand-in (the scoring tests use one) still scores.
    J_rec : the recovered Jacobian at the recovered x*, as validate already computed it.
        Passed in rather than recomputed — an autodiff Jacobian is not free, and two
        computations of it could silently drift apart.
    D_phys : the PHYSICAL diffusivities of the recovered model. On the non-dimensional
        path these are D_model * L**2; recover.py has already converted.
    system_id : the answer key's cross-L grouping label, or None. Truth-side metadata,
        used only for grouping at aggregation time — it reaches this module because
        scoring sits on the answer-key side of the firewall.

    `lgen_L` is recorded because the aggregation needs it, and L is legal for recovery to
    have seen. Note the recorded exposure: on the registered datasets the generator set
    L = clip(6*2pi/k*_true, 18, 220), so L encodes k*_true for 94.8% of the 287 samples.
    Nothing here is evidence of recovery on its own.
    """
    return {
        "lgen_system_id": system_id,
        "lgen_L": float(getattr(result, "L", float("nan"))),
        "lgen_nondim": bool(getattr(result, "nondim", False)),
        # DIAGNOSTIC ONLY — see the module docstring on why q is never a success statistic.
        "lgen_q_model": float(getattr(result, "q_model", float("nan"))),
        "lgen_D_ratio": d_ratio(D_phys),
        "lgen_J_sign": sign_string(J_rec),
    }


# --------------------------------------------------------------------------------------
# cross-run aggregation
# --------------------------------------------------------------------------------------
def modal_sign_agreement(sign_strings) -> float:
    """Entrywise agreement with the modal sign, averaged over Jacobian entries.

    For each position of the flattened sign structure, take the most common symbol across
    the supplied runs and score the fraction of runs carrying it; average over positions.
    1.0 means every run recovered the identical topology; 1/3 is the floor for a 3-symbol
    alphabet with runs spread evenly.

    ONE of several defensible statistics — see the module docstring, decision (1).
    Raises if the strings are not all the same length: comparing sign structures of
    different sizes compares unrelated entries.
    """
    strings = [s for s in sign_strings if isinstance(s, str) and s]
    if not strings:
        return float("nan")
    lengths = {len(s) for s in strings}
    if len(lengths) != 1:
        raise ValueError(
            f"modal_sign_agreement: sign structures have differing lengths {sorted(lengths)}; "
            "these come from different model sizes and must not be pooled")
    n = len(strings)
    per_entry = []
    for position in zip(*strings):
        per_entry.append(Counter(position).most_common(1)[0][1] / n)
    return float(mean(per_entry))


def _log10_spread(values) -> float:
    """Population stdev of log10(x) over strictly positive x; NaN if fewer than 2 usable.

    Log scale because both quantities aggregated this way (a diffusivity RATIO and a
    wavenumber) are multiplicative — a factor-of-2 disagreement should read the same
    whether the values are near 1 or near 100.
    """
    xs = [float(v) for v in values
          if isinstance(v, (int, float)) and v == v and float(v) > 0.0]
    if len(xs) < 2:
        return float("nan")
    return float(pstdev([math.log10(x) for x in xs]))


def _group_key(row):
    """Never pool arms. The dimensional and non-dimensional paths are different methods,
    and a different form or model size is a different experiment."""
    return (row.get("lgen_system_id"), bool(row.get("lgen_nondim")),
            row.get("form"), row.get("n_model"))


def lgen_consistency(rows) -> list[dict]:
    """Aggregate run-index rows into one cross-L consistency row per (system x arm).

    Rows with no `lgen_system_id` are EXCLUDED, not scored as failures: the datasets
    registered before three_gene_multiL carry no such label, so there is genuinely no
    grouping to do. Groups spanning fewer than 2 distinct L are excluded for the same
    reason — "consistency across L" is undefined at one L, and reporting it would let a
    single-L group masquerade as a passing cross-L result.

    Each returned row carries the cross-L statistic AND its within-L control. Read the two
    together: a low `sign_agree_cross_L` alongside an equally low `sign_agree_within_L`
    says seeds disagree at a fixed L and tells you nothing about L at all.
    `sign_agree_gap` = within - cross isolates the part attributable to the domain size.
    """
    groups = defaultdict(list)
    for r in rows:
        if r.get("lgen_system_id") is None:
            continue
        groups[_group_key(r)].append(r)

    table = []
    for (system_id, nondim, form, n_model), members in groups.items():
        by_L = defaultdict(list)
        for r in members:
            L = r.get("lgen_L")
            if isinstance(L, (int, float)) and L == L:
                by_L[round(float(L), 6)].append(r)
        if len(by_L) < 2:
            continue

        signs = [r.get("lgen_J_sign") for r in members]
        within_sign = [modal_sign_agreement([r.get("lgen_J_sign") for r in grp])
                       for grp in by_L.values() if len(grp) > 1]
        cross_sign = modal_sign_agreement(signs)
        within_mean = float(mean(within_sign)) if within_sign else float("nan")

        d_within = [_log10_spread([r.get("lgen_D_ratio") for r in grp])
                    for grp in by_L.values() if len(grp) > 1]
        k_within = [_log10_spread([r.get("kstar_model") for r in grp])
                    for grp in by_L.values() if len(grp) > 1]

        table.append(dict(
            system_id=system_id, nondim=nondim, form=form, n_model=n_model,
            n_runs=len(members), n_L=len(by_L),
            L_values=",".join(f"{L:g}" for L in sorted(by_L)),
            sign_agree_cross_L=cross_sign,
            sign_agree_within_L=within_mean,
            sign_agree_gap=(within_mean - cross_sign
                            if within_mean == within_mean else float("nan")),
            D_ratio_log10_std_cross_L=_log10_spread([r.get("lgen_D_ratio") for r in members]),
            D_ratio_log10_std_within_L=_mean_of_finite(d_within),
            kstar_log10_std_cross_L=_log10_spread([r.get("kstar_model") for r in members]),
            kstar_log10_std_within_L=_mean_of_finite(k_within),
        ))
    table.sort(key=lambda r: (str(r["system_id"]), bool(r["nondim"])))
    return table


def _mean_of_finite(values) -> float:
    xs = [v for v in values if isinstance(v, (int, float)) and v == v]
    return float(mean(xs)) if xs else float("nan")


def lgen_table(runs_root: str = "experiments", backend: str = "jsonl") -> list[dict]:
    """Read the append-only run index and build the cross-L consistency table.

    Same shape and contract as optim/benchmark.build_table: pure aggregation over the
    ledger, no re-running of anything.
    """
    from .. import io as IO
    return lgen_consistency(IO.read_run_index(runs_root, backend=backend))


def lgen_markdown(table: list[dict]) -> str:
    if not table:
        return ("_(no cross-L groups in the index: no run carried a `lgen_system_id`, or "
                "no system was recovered at more than one L)_")
    hdr = ("| " + " | ".join(LGEN_COLUMNS) + " |\n"
           + "|" + "---|" * len(LGEN_COLUMNS) + "\n")
    body = ""
    for row in table:
        body += "| " + " | ".join(
            (f"{row.get(c):.4g}" if isinstance(row.get(c), float) else str(row.get(c)))
            for c in LGEN_COLUMNS) + " |\n"
    return hdr + body
