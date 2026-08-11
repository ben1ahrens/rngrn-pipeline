"""morphology.py — MORPHOLOGY as a first-class scored metric (SCORING SIDE).

WHY THIS EXISTS
---------------
The owner's stated primary criterion for a successful recovery is that the recovered
model reproduces the same dominant spatial mode and the same pattern MORPHOLOGY
(spots / stripes / labyrinth) as the target frame. Parameter agreement (D, production,
degradation) is explicitly NOT a target. Before this module, morphology was measured
nowhere: ``observables.morphology`` computed the four statistics and
``losses.terms.morphology_consistency`` compared two of them, but neither was wired into
scoring, so no run ever recorded whether the morphology matched.

This module supplies the scored quantities:

    morphology_vector(field)                 -> the 4 observables in a fixed order
    morphology_distance(a, b, scale=None)    -> z-scored distance in that 4-D space
    classify_morphology(field, bank)         -> (label, margin_to_runner_up, distances)
    spectral_distance_2d(a, b, n=24)         -> secondary 2-D log-power-spectrum diagnostic

WHY Z-SCORING IS NOT OPTIONAL
-----------------------------
The four features have incomparable scales. Measured over the 88 three_gene_train
samples, the per-feature standard deviations are

    skew 0.824 | kurtosis 1.335 | ang_conc 0.0947 | ang_entropy 0.0884

so an unweighted Euclidean distance is dominated by kurtosis and skew and is nearly
blind to the two angular features — which are precisely the ones that carry the
stripes signal (stripes mean ang_conc 0.410 vs 0.157 / 0.205 for spots / labyrinth).
Every distance here is therefore taken in units of the per-feature training SD.

PROVENANCE OF THE DEFAULT SCALES (DATA-DERIVED, NOT INHERITED FROM A DOC)
------------------------------------------------------------------------
``TRAIN_SCALE`` is the per-feature sample standard deviation (ddof=1) of
``morphology_vector`` over ALL 88 samples of the **three_gene_train** split
(``data/datasets/three_gene_train/payload.h5``, channel 0 of ``final_frame``), measured
2026-07-26 on this branch. ``TRAIN_MEAN`` is the matching mean, recorded for reference
only: a difference of z-scores cancels the mean exactly, so no distance or class call in
this module depends on it. The val and test splits were NOT used to fit the scales.

MEASURED SEPARABILITY (leave-one-out 1-NN over all 127 three_gene samples;
train+val+test pooled, nearest single member per class, z-scored 4-D space)

    all four stats, z-scored : 92.1% overall accuracy, 81.8% balanced
    all four stats, unscaled : 94.5% overall accuracy, 79.3% balanced
    skew alone               : 92.1% overall, 77.6% balanced
    kurtosis alone           : 66.9% overall, 47.2% balanced
    ang_conc alone           : 59.8% overall, 50.6% balanced
    ang_entropy alone        : 57.5% overall, 57.4% balanced
    majority-class baseline  : 47.2% (60 spots / 60 labyrinth / 7 stripes)

Read those two top lines together: unscaled distance wins on OVERALL accuracy purely by
giving up on stripes (43% stripes recall vs 57% z-scored), because 120 of the 127 samples
are spots or labyrinth. Balanced accuracy — which weights the three classes equally — is
the honest number, and z-scoring wins there. This module optimises for the stripes class
the owner cares about, not for the pooled hit rate.

HONEST LIMITATIONS — READ BEFORE TRUSTING A CLASS CALL
-----------------------------------------------------
1. **The stripes class is weakly characterised.** Only 7 of the 127 three_gene samples
   are labelled stripes (4 of the 88 in the train split). Its class statistics rest on a
   handful of examples, its measured recall is 57%, and no threshold in this module has
   been calibrated on enough stripes samples to be trustworthy. Treat a "stripes" call as
   a hypothesis.
2. **The margin is the guard.** ``classify_morphology`` returns the distance margin to the
   runner-up class alongside the label, because a nearest-neighbour label alone hides how
   close the call was. Measured over the 127 samples: median margin 0.491 for correct
   calls vs 0.139 for the 10 incorrect ones, and 80% of incorrect calls have margin < 0.5
   (vs 50% of correct ones). A small margin is therefore informative but NOT a reliable
   error detector on its own — do not turn it into a pass/fail threshold without
   calibrating on more data, particularly more stripes.
3. **Labels come from the dataset's own ``morphology`` attribute.** This module scores
   agreement with that labelling; it does not independently establish that the labelling
   is correct.
4. Nothing here is a differentiable loss. All of it is post-hoc scoring. A simulated field
   costs seconds per rollout, so the recovery objective is untouched.

FIREWALL
--------
Scoring side. This module runs only AFTER recovery. It imports ``rngrn.observables``,
which is recovery-side and deliberately image-only: ``observables.morphology`` reads the
frame and nothing else — no x*, no J, no sigma(k), no generating coefficient. The
firewall constrains the OTHER direction (recovery-side code must never reach the answer
key or the scoring package), so a scoring module reusing an image-only measurement is
safe, and reusing it is preferable to reimplementing the statistics and letting the two
copies drift. This module reads no answer-key quantity itself; the truth it is scored
against is the morphology LABEL, supplied by the caller from the dataset attributes.
"""
from __future__ import annotations

from typing import Mapping, NamedTuple, Optional, Sequence

import numpy as np

from ..observables import morphology as _observed_morphology

__all__ = [
    "FEATURE_ORDER",
    "MorphologyCall",
    "TRAIN_MEAN",
    "TRAIN_SCALE",
    "TRAIN_SCALE_PROVENANCE",
    "TRAIN_CLASS_MEANS",
    "build_reference_bank",
    "classify_morphology",
    "default_reference_bank",
    "morphology_distance",
    "morphology_vector",
    "spectral_distance_2d",
]


# Fixed feature order. Every vector, scale and bank in this module uses THIS order; the
# dict returned by observables.morphology is not order-guaranteed, so it is indexed by
# name here exactly once.
FEATURE_ORDER = ("skew", "kurtosis", "ang_conc", "ang_entropy")

TRAIN_SCALE_PROVENANCE = (
    "per-feature sample SD (ddof=1) of morphology_vector over all 88 samples of "
    "three_gene_train (data/datasets/three_gene_train/payload.h5, final_frame channel 0), "
    "measured 2026-07-26 on branch feature/spatial-mode-recovery; val/test unused"
)

# DATA-DERIVED (see TRAIN_SCALE_PROVENANCE). Order matches FEATURE_ORDER.
TRAIN_SCALE = np.array([0.8236810184, 1.334654765, 0.09471414586, 0.08840497827])

# Recorded for reference only — a z-score difference cancels the mean, so no distance or
# class call in this module uses it. Same provenance as TRAIN_SCALE.
TRAIN_MEAN = np.array([0.5175666184, -0.4982427759, 0.1903191692, 0.8765522631])

# Class means over the three_gene TRAIN split only (n=88: 42 labyrinth, 42 spots,
# 4 stripes), same provenance as TRAIN_SCALE. These back default_reference_bank(), which
# is a CENTROID bank and measurably weaker than a bank of real samples: on the 39 held-out
# val+test samples, these centroids score 82.1% accuracy (68.5% balanced) against 89.7%
# (74.1% balanced) for a bank of the 88 real train fields. Prefer build_reference_bank()
# on real fields whenever the dataset is reachable. The 'stripes' centroid is the mean of
# FOUR samples.
TRAIN_CLASS_MEANS = {
    "labyrinth": np.array([-0.1451465352, -1.344734822, 0.198265426, 0.8705182017]),
    "spots": np.array([1.235126773, 0.4420721615, 0.1534494105, 0.9086239347]),
    "stripes": np.array([-0.05832689816, -1.483383136, 0.4940159386, 0.603157356]),
}


class MorphologyCall(NamedTuple):
    """Result of a morphology class call.

    label   : nearest class in the z-scored 4-D feature space.
    margin  : distance to the runner-up class MINUS distance to `label`, in units of
              per-feature training SD. Always >= 0. Small means borderline; see
              limitation 2 in the module docstring for the measured distribution.
    distances : distance to every class in the bank, keyed by label.
    """
    label: str
    margin: float
    distances: dict


# --------------------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------------------
def _as_field(field, name: str) -> np.ndarray:
    """Validate a single 2-D square real field. Raises rather than coercing."""
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a single 2-D field (H, W); got shape {arr.shape}. Pass one "
            f"channel, e.g. frame[0], not the whole (m, H, W) stack.")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{name} must be square; got shape {arr.shape}. observables.morphology "
            f"builds its Fourier grid from shape[0] for BOTH axes, so a non-square field "
            f"would be silently mis-binned.")
    if arr.shape[0] < 4:
        raise ValueError(f"{name} is too small to have a spectrum: shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values ({np.sum(~np.isfinite(arr))} "
                         f"of {arr.size}); refusing to score a corrupt field")
    return arr


def morphology_vector(field) -> np.ndarray:
    """The four morphology observables of one field, as a fixed-order float array.

    Order is FEATURE_ORDER = (skew, kurtosis, ang_conc, ang_entropy). The statistics
    themselves are NOT reimplemented here: this delegates to observables.morphology so
    scoring and the (image-only) measurement side cannot drift apart.

    Invariance: skew and kurtosis are standardised moments and the angular features are
    computed on a sum-normalised power histogram, so the whole vector is invariant to an
    affine intensity change f -> a*f + b for a > 0. Measured max deviation over
    a in {0.1, 2.5, 3.0}, b in {0, -1, 5}: 3.2e-14.

    Raises ValueError on a non-2-D, non-square, or non-finite field, or if the resulting
    statistics are not finite (e.g. a constant field has no defined skewness).
    """
    arr = _as_field(field, "field")
    stats = _observed_morphology(arr)
    missing = [k for k in FEATURE_ORDER if k not in stats]
    if missing:
        raise KeyError(
            f"observables.morphology did not return {missing}; scoring expects "
            f"{list(FEATURE_ORDER)}. The measurement side changed — fix the order here "
            f"deliberately rather than reordering silently.")
    vec = np.array([float(stats[k]) for k in FEATURE_ORDER])
    if not np.all(np.isfinite(vec)):
        bad = {k: v for k, v in zip(FEATURE_ORDER, vec) if not np.isfinite(v)}
        raise ValueError(
            f"morphology statistics are not finite: {bad}. A field with (near-)zero "
            f"spatial variance has no defined skewness/kurtosis — this is an unpatterned "
            f"or collapsed field, not a morphology.")
    return vec


def _scale_vector(scale) -> np.ndarray:
    if scale is None:
        return TRAIN_SCALE.copy()
    s = np.asarray(scale, dtype=float).ravel()
    if s.shape != (len(FEATURE_ORDER),):
        raise ValueError(
            f"scale must have {len(FEATURE_ORDER)} entries in FEATURE_ORDER "
            f"{FEATURE_ORDER}; got shape {s.shape}")
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError(f"scale entries must be finite and > 0; got {s}")
    return s


# --------------------------------------------------------------------------------------
# distance
# --------------------------------------------------------------------------------------
def morphology_distance(field_a, field_b, scale: Optional[Sequence[float]] = None) -> float:
    """Euclidean distance between two fields in the z-scored 4-D morphology space.

    The returned number is in units of per-feature TRAINING standard deviation, so a
    distance of 1 means "the two fields differ by about one training SD, summed in
    quadrature over the four features". 0 means identical morphology statistics.

    scale : per-feature divisors in FEATURE_ORDER. Defaults to TRAIN_SCALE, which is
            DATA-DERIVED from the three_gene_train split (see TRAIN_SCALE_PROVENANCE and
            the module docstring). Pass your own when scoring a different dataset family —
            the three_gene scales are not claimed to transfer to two_gene.

    Both fields must be single 2-D square fields of the SAME shape; a shape mismatch
    raises (the features are shape-dependent through the Fourier grid, so comparing
    across grids would be a silent apples-to-oranges).
    """
    a = _as_field(field_a, "field_a")
    b = _as_field(field_b, "field_b")
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: field_a {a.shape} vs field_b {b.shape}. The angular "
            f"features depend on the Fourier grid, so morphology of two different grids "
            f"is not comparable — resample one field first, deliberately.")
    s = _scale_vector(scale)
    return float(np.linalg.norm((morphology_vector(a) - morphology_vector(b)) / s))


# --------------------------------------------------------------------------------------
# class call
# --------------------------------------------------------------------------------------
def build_reference_bank(fields_by_label: Mapping[str, Sequence]) -> dict:
    """Build a reference bank of morphology vectors from labelled REAL fields.

    fields_by_label : {label: iterable of 2-D fields}. Each field is passed through
                      morphology_vector, so all its validation applies.
    Returns {label: (n_label, 4) array} suitable for classify_morphology.

    Preferred over default_reference_bank(). Measured on the 39 held-out val+test samples
    with a bank built from the 88 train fields: 89.7% accuracy (74.1% balanced), vs 82.1%
    (68.5% balanced) for the baked-in centroids on the same held-out samples.
    """
    if not fields_by_label:
        raise ValueError("fields_by_label is empty; a reference bank needs >= 2 classes")
    bank = {}
    for label, fields in fields_by_label.items():
        vecs = [morphology_vector(f) for f in fields]
        if not vecs:
            raise ValueError(f"class '{label}' has no fields; refusing an empty class")
        bank[str(label)] = np.stack(vecs)
    if len(bank) < 2:
        raise ValueError(
            f"a reference bank needs >= 2 classes to make a class call with a margin; "
            f"got {sorted(bank)}")
    return bank


def default_reference_bank() -> dict:
    """The baked-in CENTROID bank: one vector per class, the three_gene_train class mean.

    Weaker than build_reference_bank() on real fields: 82.1% vs 89.7% accuracy on the 39
    held-out val+test samples (68.5% vs 74.1% balanced — see TRAIN_CLASS_MEANS), and its
    'stripes' centroid is the mean of just 4 samples. Provided so scoring works when the
    dataset is not reachable at scoring time; use the real-sample bank whenever it is.
    """
    return {k: v.reshape(1, -1).copy() for k, v in TRAIN_CLASS_MEANS.items()}


def _validate_bank(reference_bank) -> dict:
    if reference_bank is None:
        raise ValueError(
            "reference_bank is required; pass build_reference_bank(...) for a bank of "
            "real samples, or default_reference_bank() for the weaker baked-in centroids")
    if not isinstance(reference_bank, Mapping):
        raise TypeError(f"reference_bank must be a mapping label -> (n, 4) array; got "
                        f"{type(reference_bank).__name__}")
    if len(reference_bank) < 2:
        raise ValueError(
            f"a reference bank needs >= 2 classes to define a runner-up margin; got "
            f"{sorted(reference_bank)}")
    out = {}
    for label, vecs in reference_bank.items():
        arr = np.asarray(vecs, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != len(FEATURE_ORDER):
            raise ValueError(
                f"reference_bank['{label}'] must be (n, {len(FEATURE_ORDER)}) in "
                f"FEATURE_ORDER {FEATURE_ORDER}; got shape {np.shape(vecs)}")
        if arr.shape[0] == 0:
            raise ValueError(f"reference_bank['{label}'] is empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"reference_bank['{label}'] contains non-finite values")
        out[str(label)] = arr
    return out


def classify_morphology(field, reference_bank,
                        scale: Optional[Sequence[float]] = None) -> MorphologyCall:
    """Nearest-class morphology call in the z-scored 4-D space, WITH its margin.

    Distance to a class is the distance to its NEAREST member in the bank (so a bank of
    real samples is used as a 1-NN classifier, and a centroid bank degenerates to
    nearest-centroid). The margin to the runner-up class is returned alongside the label
    so a borderline call is visible instead of silently confident.

    field          : one 2-D square field.
    reference_bank : {label: (n, 4) vectors}, from build_reference_bank() (preferred) or
                     default_reference_bank(). >= 2 classes required.
    scale          : per-feature divisors; defaults to the DATA-DERIVED TRAIN_SCALE.

    Returns MorphologyCall(label, margin, distances).

    Measured (127 three_gene samples, leave-one-out, real-sample bank): 92.1% overall
    accuracy, 81.8% balanced, vs a 47.2% majority-class baseline. Per class: spots 96.7%,
    labyrinth 91.7%, stripes 57.1% — stripes has 7 samples in total, so its number is
    weak (module docstring, limitation 1).
    """
    bank = _validate_bank(reference_bank)
    s = _scale_vector(scale)
    z = morphology_vector(field) / s
    dists = {label: float(np.min(np.linalg.norm(arr / s - z, axis=1)))
             for label, arr in bank.items()}
    order = sorted(dists, key=dists.get)
    best, runner_up = order[0], order[1]
    return MorphologyCall(label=best,
                          margin=float(dists[runner_up] - dists[best]),
                          distances=dists)


# --------------------------------------------------------------------------------------
# secondary diagnostic: 2-D log-power-spectrum distance
# --------------------------------------------------------------------------------------
def _spectral_block(field: np.ndarray, n: int) -> np.ndarray:
    """Central low-frequency block of the normalised 2-D log power spectrum, flattened.

    Normalise-THEN-log ordering, deliberately (see spectral_distance_2d)."""
    f = field - field.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    H, W = P.shape
    if n < 4 or n > min(H, W):
        raise ValueError(
            f"n={n} is not a usable block size for a {H}x{W} field: need 4 <= n <= "
            f"{min(H, W)}")
    top, left = H // 2 - n // 2, W // 2 - n // 2
    block = P[top:top + n, left:left + n]
    total = block.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "the central spectral block carries no power — the field is (near-)constant, "
            "so it has no spatial mode to compare")
    block = np.log1p(block / total)          # scale-free FIRST, then compress
    total_log = block.sum()
    if total_log <= 0:
        raise ValueError("log-power block sums to zero; cannot normalise")
    return (block / total_log).ravel()


def spectral_distance_2d(field_a, field_b, n: int = 24) -> float:
    """RMSE between the two fields' normalised 2-D log power spectra (central n x n block).

    SECONDARY DIAGNOSTIC. The headline morphology metric is morphology_distance /
    classify_morphology; this exists as an independent, non-hand-picked view of the
    spatial mode that does not go through the four hand-chosen statistics.

    WHY 2-D AND NOT THE EXISTING 1-D observables.raps()
    ---------------------------------------------------
    raps() averages the power spectrum over rings of constant |k|. That average destroys
    the ANGULAR structure, which is exactly what separates stripes (power concentrated in
    one direction) from labyrinth (power spread over all directions) at the same dominant
    |k|. Measured as a leave-one-out 1-NN morphology classifier over the 127 three_gene
    samples: the radially-averaged 1-D spectrum reaches 54.3% accuracy (42.5% balanced),
    barely above the 47.2% majority baseline, while this 2-D block reaches 83.5% (75.7%
    balanced) at n=24. Keeping the angular information is worth ~29 accuracy points.

    NORMALISE-THEN-LOG, a deliberate deviation
    ------------------------------------------
    The power is normalised to sum 1 BEFORE log1p (then renormalised), not after. Taking
    log1p of raw power first makes the descriptor depend on absolute intensity: a 3x
    brightness rescale of the same pattern then moves it by 1.0e-3 in this metric — which
    would be a bug in a morphology diagnostic, since brightness is not morphology — and it
    also scores worse (70.1% vs 83.5% accuracy at n=24). With this ordering the same
    rescale moves it by 6.6e-18, i.e. exact to floating point.

    n : side of the central low-frequency block. Measured accuracy is flat-ish in n
        (79.5% at n=12/16, 83.5% at n=24, 84.3% at n=32); n=24 keeps the low-frequency
        modes that carry the pattern on the 96x96 three_gene grids without pulling in
        mostly-noise high frequencies. Not a calibrated optimum — it has been scanned over
        four values on one dataset family, nothing more.

    Both fields must be single 2-D square fields of the same shape; anything else raises.
    """
    a = _as_field(field_a, "field_a")
    b = _as_field(field_b, "field_b")
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: field_a {a.shape} vs field_b {b.shape}. Two spectra on "
            f"different grids have different frequency bins and are not comparable.")
    sa, sb = _spectral_block(a, n), _spectral_block(b, n)
    return float(np.sqrt(np.mean((sa - sb) ** 2)))
