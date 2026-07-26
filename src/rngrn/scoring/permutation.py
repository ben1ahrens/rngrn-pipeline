"""permutation.py — permutation-invariant Jacobian scoring (SCORING SIDE, SCAFFOLD).

WHY THIS EXISTS
---------------
A recovered RNGRN assigns its own, arbitrary, integer label to every species it
models. The observed channels are pinned by the gate: row ``r`` of the frame is model
species ``observed_idx[r]``, and it is the same physical species as true channel
``observed_idx[r]``. The UNOBSERVED (latent) species carry no such pin — nothing in
the recovery objective ties latent model index ``i`` to true channel ``i``. Comparing
J_rec to J_true entrywise therefore mixes a real structural error with a pure
relabelling artefact, and an entrywise ``sign_match_frac`` computed under the
identity labelling is not interpretable on its own.

This module scores the sign structure over the (small, exactly enumerable) set of
relabellings that are consistent with the pinned observed channels, and reports the
chosen relabelling alongside the score. Reporting the permutation is REQUIRED: a
score maximised over a hidden nuisance parameter is only interpretable if the
argmax is shown with it.

HONEST LIMITATION — read before trusting a "delta"
--------------------------------------------------
With the observed channels held fixed, the search space has exactly ``(N - m)!``
elements. For EXPERIMENT A (N = 3 model species, m = 2 observed) that is ``1! = 1``:
the identity is the ONLY admissible relabelling, alignment is a mathematical no-op,
and ``sign_match_frac_aligned == sign_match_frac_identity`` by construction. Aligning
does NOT rescue Experiment A's ~0.44 sign score — it only proves that that number is
not a labelling artefact of the hidden index. Alignment does real work from two
latent species upward (N - m >= 2), and in the ``free_observed=True`` mode, which
asks a different (weaker) question — see ``align_permutation``.

FIREWALL
--------
Scoring side. This module reads answer-key quantities (J_true, the true hidden
channel) and MUST NEVER be imported by a recovery-side module. It is deliberately
free of any dependency on ``rngrn.model``/``recover``/``losses`` so the import can
only ever run in the scoring direction.

Status: SCAFFOLD. Unit-tested on synthetic matrices with known permutations; not
validated as a measure of recovery quality on real data.
"""
from __future__ import annotations

import itertools
from typing import Iterable, NamedTuple, Optional, Sequence

import numpy as np

__all__ = [
    "AlignmentScore",
    "align_permutation",
    "latent_field_quality",
    "observed_subblock_score",
    "permuted_sign_match",
    "unobserved_indices",
]

# Relative magnitude below which a Jacobian entry counts as structurally zero.
# Matches the convention already used by validate._sign_structure so the aligned and
# unaligned numbers are directly comparable.
SIGN_ZERO_RTOL = 1e-9


class AlignmentScore(NamedTuple):
    """Result of :func:`align_permutation`.

    Unpacks in the order ``(perm, score, permuted_J, fro_rel_err, n_candidates)``, so
    the documented three-value contract ``perm, score, permuted_J, *_ = ...`` holds.

    Attributes
    ----------
    perm:
        Tuple of length N. ``perm[i]`` is the MODEL species index placed at aligned
        position ``i``; equivalently ``permuted_J = J_rec[np.ix_(perm, perm)]``. Under
        the default (observed-pinned) mode ``perm[o] == o`` for every observed index
        ``o``. This is the value that MUST be reported with the score.
    score:
        The maximised objective: fraction of the N**2 entries whose sign structure
        agrees between ``permuted_J`` and ``J_true`` (in [0, 1], higher is better).
    permuted_J:
        ``J_rec`` with rows and columns relabelled by ``perm`` — the matrix the caller
        should show next to ``J_true``.
    fro_rel_err:
        SECONDARY diagnostic only, never optimised:
        ``||permuted_J - J_true||_F / ||J_true||_F``. The project scores sign/regime,
        not parameter values, so this number is reported for context and is expected
        to be large even for a structurally correct recovery (overall scale of J is
        not pinned by the objective).
    n_candidates:
        Size of the enumerated search space. ``1`` means the score is not a maximum
        over anything and the alignment was a no-op (see module docstring).
    """

    perm: tuple
    score: float
    permuted_J: np.ndarray
    fro_rel_err: float
    n_candidates: int


def unobserved_indices(n: int, observed_idx: Iterable[int]) -> tuple:
    """Ascending tuple of the model indices in ``range(n)`` that are NOT observed.

    This is the same convention ``losses.terms.stationarity_residual`` and
    ``recover.recover`` use to lay out ``latent_fields``: row ``r`` of a
    ``(N - m, H, W)`` latent stack is model species ``unobserved_indices(N, obs)[r]``.
    """
    obs = tuple(int(i) for i in observed_idx)
    return tuple(i for i in range(int(n)) if i not in obs)


def _validate_observed_idx(observed_idx: Sequence[int], n: int, what: str) -> tuple:
    obs = tuple(int(i) for i in observed_idx)
    if len(obs) == 0:
        raise ValueError("observed_idx is empty; scoring needs at least one observed channel")
    if len(set(obs)) != len(obs):
        raise ValueError(f"observed_idx has duplicates: {obs}")
    for i in obs:
        if not 0 <= i < n:
            raise IndexError(
                f"observed index {i} is out of range for {what} of size {n}; "
                f"observed_idx={obs}")
    return obs


def _as_square(J, name: str) -> np.ndarray:
    arr = np.asarray(J, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite entries; refusing to score")
    return arr


def sign_structure(J) -> np.ndarray:
    """Elementwise sign of ``J`` with near-zero entries collapsed to 0.

    An entry counts as structurally zero when ``|J_ij| <= SIGN_ZERO_RTOL * max|J|``.
    Same rule as ``validate._sign_structure``, duplicated here rather than imported so
    this module stays independent of the module another agent is editing.
    """
    arr = _as_square(J, "J")
    scale = np.abs(arr).max()
    return np.sign(np.where(np.abs(arr) > SIGN_ZERO_RTOL * (scale + 1e-12), arr, 0.0))


def _fro_rel_err(A: np.ndarray, B: np.ndarray) -> float:
    """||A - B||_F / ||B||_F. Raises if B is the zero matrix (no scale to divide by)."""
    denom = float(np.linalg.norm(B))
    if denom == 0.0:
        raise ValueError(
            "reference matrix has zero Frobenius norm; relative error is undefined")
    return float(np.linalg.norm(A - B) / denom)


def _candidate_perms(n: int, observed: tuple, free_observed: bool):
    """Yield admissible relabellings of the model indices as length-n tuples."""
    if free_observed:
        yield from itertools.permutations(range(n))
        return
    latent_positions = unobserved_indices(n, observed)
    for assignment in itertools.permutations(latent_positions):
        perm = list(range(n))
        for pos, model_idx in zip(latent_positions, assignment):
            perm[pos] = model_idx
        yield tuple(perm)


def align_permutation(J_rec, J_true, observed_idx, *,
                      free_observed: bool = False) -> AlignmentScore:
    """Relabel the model's species indices to best match ``J_true``'s sign structure.

    WHAT IS OPTIMISED (exactly, no approximation): the fraction of the ``N**2``
    entries on which ``sign_structure(J_rec[perm][:, perm])`` equals
    ``sign_structure(J_true)``. The search is an exhaustive enumeration over the
    admissible permutations, so the returned permutation is a true global argmax of
    that objective — not a heuristic. Ties are broken by the SMALLER relative
    Frobenius error, and any remaining tie by lexicographic order of ``perm``, which
    makes the result deterministic. The Frobenius error is a tie-break and a reported
    diagnostic ONLY; it is never part of the primary objective, because this project
    scores sign/regime rather than parameter values.

    Parameters
    ----------
    J_rec:
        (N, N) recovered Jacobian, e.g. ``result.model.jacobian(result.xstar)``.
    J_true:
        (N, N) answer-key Jacobian. MUST have the same N as ``J_rec``: a full-matrix
        permutation comparison is only defined when model and truth have the same
        number of species. When they differ (EXPERIMENT B: model N=3, truth 2x2) this
        function raises — use :func:`observed_subblock_score`, which is well-defined
        for both experiments, instead of silently producing a NaN.
    observed_idx:
        Model indices of the observed channels, as carried by
        ``data.gate.RecoveryInput.observed_idx``. These are held FIXED at their own
        positions (``perm[o] == o``) because the gate pins observed model channel ``o``
        to true channel ``o``; only the latent indices are permuted.
    free_observed:
        Default False (the pinned mode described above). Set True to enumerate all
        ``N!`` relabellings, INCLUDING the observed ones. That answers a strictly
        weaker question — "is this structure present under any relabelling at all?" —
        and discards the gate's known observed-channel correspondence, so a score
        obtained this way is an upper bound and must be reported as such. Provided
        because the pinned search is degenerate when ``N - m == 1`` (see the module
        docstring), not because it is the recommended metric.

    Returns
    -------
    AlignmentScore
        Named tuple ``(perm, score, permuted_J, fro_rel_err, n_candidates)``.
        ``n_candidates == 1`` flags that no real maximisation happened.

    Raises
    ------
    ValueError
        Non-square input, non-finite entries, shape mismatch between J_rec and J_true,
        or a zero-norm ``J_true`` (relative Frobenius error undefined).
    IndexError
        An observed index out of range for N.
    """
    rec = _as_square(J_rec, "J_rec")
    true = _as_square(J_true, "J_true")
    if rec.shape != true.shape:
        raise ValueError(
            f"align_permutation needs matching shapes, got J_rec {rec.shape} vs "
            f"J_true {true.shape}. A whole-matrix permutation comparison is undefined "
            f"across a species-count mismatch (this is EXPERIMENT B's situation); call "
            f"observed_subblock_score(J_rec, J_true, observed_idx) instead.")
    n = rec.shape[0]
    observed = _validate_observed_idx(observed_idx, n, "J_rec")

    s_true = sign_structure(true)
    s_rec = sign_structure(rec)

    best: Optional[AlignmentScore] = None
    n_candidates = 0
    for perm in _candidate_perms(n, observed, free_observed):
        n_candidates += 1
        ix = np.ix_(perm, perm)
        score = float(np.mean(s_rec[ix] == s_true))
        fro = _fro_rel_err(rec[ix], true)
        if best is None or (score, -fro) > (best.score, -best.fro_rel_err):
            best = AlignmentScore(perm=perm, score=score, permuted_J=rec[ix].copy(),
                                  fro_rel_err=fro, n_candidates=0)
    if best is None:  # unreachable: itertools.permutations always yields >= 1 element
        raise RuntimeError("no candidate permutation was enumerated")
    return best._replace(n_candidates=n_candidates)


def permuted_sign_match(J_rec, J_true, observed_idx, *,
                        free_observed: bool = False) -> dict:
    """Flat metric dict: aligned vs unaligned sign agreement, plus the permutation.

    Reports BOTH numbers on purpose. ``sign_match_frac_identity`` is exactly what the
    pipeline reported before alignment existed (``validate.score_recovery``'s
    ``sign_match_frac``), so ``sign_match_delta`` states how much of that number was a
    latent-relabelling artefact. When ``n_permutations_searched == 1`` the delta is
    necessarily 0.0 and the identity labelling was already the only admissible one —
    report it that way rather than as evidence that alignment "worked".

    Keys
    ----
    best_perm : tuple                    argmax relabelling (report it with the score)
    sign_match_frac_aligned : float      maximised sign agreement, in [0, 1]
    sign_match_frac_identity : float     sign agreement under the identity labelling
    sign_match_delta : float             aligned - identity, >= 0 by construction
    fro_rel_err_aligned : float          secondary diagnostic under best_perm
    fro_rel_err_identity : float         secondary diagnostic under the identity
    n_permutations_searched : int        size of the enumerated search space
    observed_idx : tuple                 echoed for provenance
    free_observed : bool                 which question was asked (see align_permutation)
    """
    best = align_permutation(J_rec, J_true, observed_idx, free_observed=free_observed)
    rec = np.asarray(J_rec, dtype=float)
    true = np.asarray(J_true, dtype=float)
    ident = float(np.mean(sign_structure(rec) == sign_structure(true)))
    return {
        "best_perm": tuple(int(i) for i in best.perm),
        "sign_match_frac_aligned": float(best.score),
        "sign_match_frac_identity": ident,
        "sign_match_delta": float(best.score - ident),
        "fro_rel_err_aligned": float(best.fro_rel_err),
        "fro_rel_err_identity": _fro_rel_err(rec, true),
        "n_permutations_searched": int(best.n_candidates),
        "observed_idx": tuple(int(i) for i in observed_idx),
        "free_observed": bool(free_observed),
    }


def observed_subblock_score(J_rec, J_true, observed_idx) -> dict:
    """Compare ONLY the observed-species sub-block of J_rec and J_true.

    This is the one J comparison that is well-defined in BOTH experiments, because it
    never touches a latent/spare species: model species ``observed_idx[r]`` is pinned
    by the gate to true channel ``observed_idx[r]``, so the ``m x m`` sub-blocks
    ``J_rec[obs][:, obs]`` and ``J_true[obs][:, obs]`` describe the same physical
    interactions regardless of how many species each matrix has in total.

      * EXPERIMENT A (truth 3x3, model 3x3): the 2x2 observed block, with the hidden
        gene's row/column excluded.
      * EXPERIMENT B (truth 2x2, model 3x3): the whole truth vs the model's 2x2
        observed block. This replaces the previous silent NaN, which came from
        ``validate.py`` bailing on the 3x3-vs-2x2 shape mismatch.

    The same integer indexes both matrices, so every observed index must be a valid
    channel of the truth as well. If an observed index exceeds the truth's species
    count the function RAISES — that combination means the caller's observed/truth
    correspondence is not what this function assumes, and guessing would be worse
    than failing.

    Returns
    -------
    dict with keys
        sign_match_frac_observed : float   sign agreement over the m**2 block entries
        fro_rel_err_observed : float       ||block_rec - block_true||_F / ||block_true||_F
        n_observed : int                   m
        n_species_model : int              N of J_rec
        n_species_true : int               N of J_true
        observed_idx : tuple               indices used, for provenance

    Raises
    ------
    ValueError / IndexError
        Non-square or non-finite input; an observed index invalid for either matrix.
    """
    rec = _as_square(J_rec, "J_rec")
    true = _as_square(J_true, "J_true")
    obs = _validate_observed_idx(observed_idx, rec.shape[0], "J_rec")
    _validate_observed_idx(obs, true.shape[0], "J_true (answer key)")

    ix = np.ix_(obs, obs)
    block_rec, block_true = rec[ix], true[ix]
    return {
        "sign_match_frac_observed": float(
            np.mean(sign_structure(block_rec) == sign_structure(block_true))),
        "fro_rel_err_observed": _fro_rel_err(block_rec, block_true),
        "n_observed": len(obs),
        "n_species_model": int(rec.shape[0]),
        "n_species_true": int(true.shape[0]),
        "observed_idx": obs,
    }


def _as_field_stack(arr, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(f"{name} must have shape (H, W) or (k, H, W), got {a.shape}")
    if a.shape[0] == 0:
        raise ValueError(f"{name} contains no fields (shape {a.shape})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values; refusing to score")
    return a


# A field counts as spatially constant when its centred norm falls below this fraction of
# its own magnitude. A RELATIVE test is required: mean-removal of a genuinely constant
# array leaves ~1e-17 float round-off, so an exact `norm == 0.0` check silently passes a
# collapsed field through and returns a meaningless correlation of round-off against
# signal. That is precisely the kind of silent near-zero this project forbids.
CONSTANT_FIELD_RTOL = 1e-9


def _centred_or_raise(a: np.ndarray, name: str) -> np.ndarray:
    x = a.ravel() - a.mean()
    scale = float(np.abs(a).max())
    if float(np.linalg.norm(x)) <= CONSTANT_FIELD_RTOL * (scale * np.sqrt(a.size) + 1e-300):
        raise ValueError(
            f"{name} is spatially constant (zero variance to within "
            f"{CONSTANT_FIELD_RTOL:g} relative); Pearson correlation is undefined. A "
            f"collapsed latent field is a real recovery outcome — report it as collapse, "
            f"do not score it as r=0.")
    return x


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two flattened fields. Raises if either field is constant."""
    x = _centred_or_raise(a, "a field")
    y = _centred_or_raise(b, "a field")
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def _cosine_uncentered(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel(), b.ravel()
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx == 0.0 or ny == 0.0:
        raise ValueError("a field is identically zero; uncentered cosine is undefined")
    return float(np.dot(x, y) / (nx * ny))


def latent_field_quality(latent_fields, true_hidden_channel) -> dict:
    """SCORING-ONLY diagnostic: does the recovered latent field look like the hidden gene?

    *** FIREWALL WARNING — READ THIS ***
    ``true_hidden_channel`` is an ANSWER-KEY quantity. It is the concentration field of
    the species that recovery is not allowed to see. It MUST reach this function only
    AFTER recovery has finished, and it must NEVER be passed into recovery, an
    initialiser, a loss term, or anything reachable from ``rngrn.recover``. This module
    lives on the scoring side of the firewall precisely so that constraint is auditable
    (tests/test_firewall.py, tests/test_permutation_scoring.py). Leaking this field into
    the recovery path invalidates the entire identifiability experiment.

    Interpretation
    --------------
    ``pearson_r`` is computed on mean-removed fields, so it already measures agreement
    of the SPATIAL PATTERN alone and is blind to any additive offset or positive
    rescaling of the latent field — which is the right invariance here, since nothing
    in the recovery objective pins the latent field's level or units.
    ``cosine_uncentered`` is the same inner product WITHOUT mean removal; it is the
    number that also feels the offset, and it is reported so the difference between
    "right pattern, wrong level" and "right pattern and level" is visible.
    ``abs_pearson_r`` is reported because a sign-flipped latent field can still be
    structurally informative (an inhibitor recovered as an activator with an inverted
    field); a large ``abs_pearson_r`` with a negative ``pearson_r`` is a phase
    inversion, not a success — say which one it is when reporting.

    Parameters
    ----------
    latent_fields:
        ``(n_lat, H, W)`` (or ``(H, W)``) recovered latent field stack, i.e.
        ``RecoveryResult.latent_fields``. Row ``r`` is model species
        ``unobserved_indices(N, observed_idx)[r]``. ``None`` raises — a missing latent
        field means recovery ran with m == N and there is nothing to score. When
        ``n_lat > 1``, apply the relabelling from :func:`align_permutation` to the rows
        first, or read the full ``correlation_matrix`` and say which pairing you used.
    true_hidden_channel:
        ``(H, W)`` (or ``(n_hidden, H, W)``) true field(s) of the unobserved species.
        For EXPERIMENT B no third gene exists, so there is NO true hidden channel to
        pass: inertness of the spare species must be judged from the latent field's own
        amplitude/variance, not from this correlation.

    Returns
    -------
    dict with keys
        correlation_matrix : (n_lat, n_hidden) list of lists of Pearson r
        uncentered_cosine_matrix : (n_lat, n_hidden) list of lists
        pearson_r : float                  scalar shortcut, present only when 1x1
        abs_pearson_r : float              scalar shortcut, present only when 1x1
        cosine_uncentered : float          scalar shortcut, present only when 1x1
        best_pearson_r : float             entry of largest |r| in correlation_matrix
        best_pair : (int, int)             (latent row, hidden row) attaining it
        n_latent : int
        n_hidden : int
        latent_std : list[float]           per-row spatial std of the latent fields
        hidden_std : list[float]           per-row spatial std of the true fields

    Raises
    ------
    ValueError
        ``latent_fields is None``; wrong rank; non-finite values; an (H, W) grid
        mismatch between the recovered and true fields (this function will NOT
        interpolate or resample to make shapes agree — a grid mismatch means the two
        fields came from different setups and the comparison is meaningless); or a
        spatially constant field, for which correlation is undefined.
    """
    if latent_fields is None:
        raise ValueError(
            "latent_fields is None: recovery inferred no latent field (m == N), so "
            "there is nothing to compare against the hidden channel.")
    lat = _as_field_stack(latent_fields, "latent_fields")
    hid = _as_field_stack(true_hidden_channel, "true_hidden_channel")
    if lat.shape[1:] != hid.shape[1:]:
        raise ValueError(
            f"spatial grid mismatch: latent_fields {lat.shape[1:]} vs "
            f"true_hidden_channel {hid.shape[1:]}. Refusing to interpolate — fix the "
            f"grids upstream.")

    n_lat, n_hid = lat.shape[0], hid.shape[0]
    corr = np.empty((n_lat, n_hid), dtype=float)
    cos = np.empty((n_lat, n_hid), dtype=float)
    for i in range(n_lat):
        for j in range(n_hid):
            corr[i, j] = _pearson(lat[i], hid[j])
            cos[i, j] = _cosine_uncentered(lat[i], hid[j])

    flat_best = int(np.argmax(np.abs(corr)))
    bi, bj = np.unravel_index(flat_best, corr.shape)
    out = {
        "correlation_matrix": corr.tolist(),
        "uncentered_cosine_matrix": cos.tolist(),
        "best_pearson_r": float(corr[bi, bj]),
        "best_pair": (int(bi), int(bj)),
        "n_latent": n_lat,
        "n_hidden": n_hid,
        "latent_std": [float(lat[i].std()) for i in range(n_lat)],
        "hidden_std": [float(hid[j].std()) for j in range(n_hid)],
    }
    if n_lat == 1 and n_hid == 1:
        out["pearson_r"] = float(corr[0, 0])
        out["abs_pearson_r"] = float(abs(corr[0, 0]))
        out["cosine_uncentered"] = float(cos[0, 0])
    return out
