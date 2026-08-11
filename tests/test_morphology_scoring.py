"""test_morphology_scoring.py — tests for rngrn.scoring.morphology.

Morphology (same dominant spatial mode / same pattern class) is the owner's PRIMARY
criterion for a successful recovery, so these tests are about whether the metric actually
separates the classes on REAL data, not only whether the arithmetic runs.

Four groups:
  1. class separation on the real three_gene samples, with the measured 92.1%-vs-47.2%
     leave-one-out separability PINNED so a regression in observables.morphology or in the
     z-scoring shows up as a test failure rather than a quietly worse number.
  2. invariance to an affine intensity rescale — brightness is not morphology.
  3. loud failure: shape mismatch, wrong rank, non-finite, degenerate bank.
  4. wiring into validate.score_recovery, and the firewall audit.

The dataset-backed tests SKIP (they do not fail) when data/datasets is not present, so the
suite still runs on a checkout without the local data; the separability pin is worthless
without the data and a false pass would be worse than a skip.
"""
import ast
import pathlib

import numpy as np
import pytest

from rngrn.scoring import morphology as M


DATASETS = pathlib.Path(__file__).resolve().parents[1] / "data" / "datasets"
SPLITS = ("three_gene_train", "three_gene_val", "three_gene_test")


# --------------------------------------------------------------------------------------
# fixtures: real samples, loaded once
# --------------------------------------------------------------------------------------
def _load_split(split):
    import h5py
    path = DATASETS / split / "payload.h5"
    if not path.exists():
        return None
    out = []
    with h5py.File(path, "r") as f:
        for key in sorted(f):
            g = f[key]
            if "morphology" not in g.attrs:
                raise KeyError(
                    f"{split}/{key} has no 'morphology' attribute; the class-separation "
                    f"tests score against that label and cannot proceed without it")
            out.append((str(g.attrs["morphology"]), g["final_frame"][0].astype(float)))
    return out


@pytest.fixture(scope="module")
def real_samples():
    """[(label, field)] over three_gene train+val+test, or skip if the data is absent."""
    splits = {s: _load_split(s) for s in SPLITS}
    if any(v is None for v in splits.values()):
        pytest.skip(f"three_gene payloads not present under {DATASETS}")
    return splits


@pytest.fixture(scope="module")
def pooled(real_samples):
    recs = [r for s in SPLITS for r in real_samples[s]]
    labels = np.array([l for l, _ in recs])
    vecs = np.stack([M.morphology_vector(f) for _, f in recs])
    return recs, labels, vecs


def _synthetic(kind, n=64, seed=0):
    """A crude synthetic pattern, for tests that must not depend on the dataset."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] * (2 * np.pi / n)
    if kind == "stripes":
        f = np.sin(4 * x)
    elif kind == "spots":
        f = np.maximum(np.sin(4 * x) * np.sin(4 * y), 0.0) ** 2
    elif kind == "noise":
        f = rng.standard_normal((n, n))
    else:
        raise ValueError(kind)
    return f + 0.01 * rng.standard_normal((n, n))


# --------------------------------------------------------------------------------------
# 1. class separation on real samples  (the point of the module)
# --------------------------------------------------------------------------------------
def test_train_scale_matches_the_split_it_claims(real_samples):
    """TRAIN_SCALE/TRAIN_MEAN must still be what three_gene_train actually measures.

    The docstring states these are DATA-DERIVED from that split. If observables.morphology
    changes, or the split changes, the claim becomes false — that must fail here rather
    than silently mis-scale every distance in the pipeline.
    """
    vecs = np.stack([M.morphology_vector(f) for _, f in real_samples["three_gene_train"]])
    assert vecs.shape == (88, 4), f"three_gene_train has {vecs.shape[0]} samples, expected 88"
    np.testing.assert_allclose(vecs.std(0, ddof=1), M.TRAIN_SCALE, rtol=1e-6)
    np.testing.assert_allclose(vecs.mean(0), M.TRAIN_MEAN, rtol=1e-6)


def test_class_means_separate_the_classes(real_samples):
    """The angular features must carry the stripes signal, and skew the spots signal.

    These are the two measured contrasts the metric relies on; if either collapses, the
    4-D space stops separating and every downstream morphology score is noise.
    """
    recs = real_samples["three_gene_train"]
    labels = np.array([l for l, _ in recs])
    vecs = np.stack([M.morphology_vector(f) for _, f in recs])
    mean = {c: vecs[labels == c].mean(0) for c in set(labels)}
    i_skew = M.FEATURE_ORDER.index("skew")
    i_conc = M.FEATURE_ORDER.index("ang_conc")
    # spots are strongly right-skewed; the other two are near-symmetric or left
    assert mean["spots"][i_skew] > 0.8
    assert mean["labyrinth"][i_skew] < 0.3
    assert mean["stripes"][i_skew] < 0.3
    # stripes concentrate spectral power in one direction; the others do not
    assert mean["stripes"][i_conc] > 1.5 * mean["labyrinth"][i_conc]
    assert mean["stripes"][i_conc] > 1.5 * mean["spots"][i_conc]


def _loo_nearest_member(vecs, labels, scale):
    """Leave-one-out 1-NN accuracy in the z-scored space, using the module's own metric
    semantics (distance to the nearest MEMBER of each class)."""
    z = vecs / np.asarray(scale)
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    classes = sorted(set(labels))
    pred = np.array([min(classes, key=lambda c: d[i][labels == c].min())
                     for i in range(len(z))])
    per_class = {c: float((pred[labels == c] == c).mean()) for c in classes}
    return float((pred == labels).mean()), per_class


def test_pinned_separability_of_the_four_stat_space(pooled):
    """PIN the measured separability so a regression is visible.

    Measured 2026-07-26 over all 127 three_gene samples (train+val+test pooled),
    leave-one-out nearest-member 1-NN in the z-scored 4-D space:

        overall accuracy 92.1%   (majority-class baseline 47.2%)
        balanced         81.8%
        per class: spots 96.7%, labyrinth 91.7%, stripes 57.1% (7 samples)

    Bounds are set a little below the measured values so ordinary numerical drift does not
    trip them, but a real degradation does. The BASELINE is asserted too — an accuracy
    number is meaningless without it.
    """
    _, labels, vecs = pooled
    assert len(labels) == 127, f"expected 127 pooled samples, got {len(labels)}"
    counts = {c: int((labels == c).sum()) for c in sorted(set(labels))}
    assert counts == {"labyrinth": 60, "spots": 60, "stripes": 7}, counts
    baseline = max(counts.values()) / len(labels)
    assert 0.47 < baseline < 0.48, baseline

    acc, per_class = _loo_nearest_member(vecs, labels, M.TRAIN_SCALE)
    balanced = float(np.mean(list(per_class.values())))
    assert acc >= 0.90, f"separability regressed: {acc:.3f} < 0.90 (measured 0.921)"
    assert acc > baseline + 0.35, f"acc {acc:.3f} not far enough above baseline {baseline:.3f}"
    assert balanced >= 0.78, f"balanced accuracy regressed: {balanced:.3f} (measured 0.818)"
    assert per_class["spots"] >= 0.93, per_class
    assert per_class["labyrinth"] >= 0.88, per_class
    # stripes is the weak class BY CONSTRUCTION (7 samples). Pinned loosely and low on
    # purpose: this asserts the class is not entirely lost, and documents that it is weak.
    assert per_class["stripes"] >= 0.42, per_class


def test_zscoring_beats_unscaled_on_balanced_accuracy(pooled):
    """The stated reason for z-scoring: it buys the stripes class.

    Unscaled distance wins on POOLED accuracy (94.5% vs 92.1%) by giving up on stripes,
    because 120 of 127 samples are spots or labyrinth. On BALANCED accuracy — the honest
    number when one class has 7 members — z-scoring wins (81.8% vs 79.3%), and its stripes
    recall is higher (57.1% vs 42.9%). If this inverts, the docstring's justification for
    z-scoring is false and must be rewritten.
    """
    _, labels, vecs = pooled
    _, z_per = _loo_nearest_member(vecs, labels, M.TRAIN_SCALE)
    _, raw_per = _loo_nearest_member(vecs, labels, np.ones(4))
    assert np.mean(list(z_per.values())) > np.mean(list(raw_per.values()))
    assert z_per["stripes"] > raw_per["stripes"]


def test_classify_morphology_beats_baseline_on_held_out(real_samples):
    """A train-only bank must generalise to val+test above the majority baseline.

    This is the honest generalisation check: the bank is built from three_gene_train ONLY
    and evaluated on the 39 val+test samples it never saw. Measured 89.7% accuracy.
    """
    train = real_samples["three_gene_train"]
    held = real_samples["three_gene_val"] + real_samples["three_gene_test"]
    classes = sorted({l for l, _ in train})
    bank = M.build_reference_bank({c: [f for l, f in train if l == c] for c in classes})
    pred = np.array([M.classify_morphology(f, bank).label for _, f in held])
    truth = np.array([l for l, _ in held])
    acc = float((pred == truth).mean())
    assert acc >= 0.82, f"held-out accuracy {acc:.3f} < 0.82 (measured 0.897)"
    assert acc > 0.48, "held-out accuracy is at or below the majority-class baseline"


def test_baked_in_centroids_match_the_train_split(real_samples):
    """TRAIN_CLASS_MEANS must be the three_gene_train class means it claims to be.

    default_reference_bank() is built from these, so a drift here silently degrades every
    class call made without an explicit bank.
    """
    recs = real_samples["three_gene_train"]
    labels = np.array([l for l, _ in recs])
    vecs = np.stack([M.morphology_vector(f) for _, f in recs])
    assert set(M.TRAIN_CLASS_MEANS) == set(labels), (
        f"bank classes {sorted(M.TRAIN_CLASS_MEANS)} != split classes {sorted(set(labels))}")
    for cls, claimed in M.TRAIN_CLASS_MEANS.items():
        np.testing.assert_allclose(vecs[labels == cls].mean(0), claimed, rtol=1e-6,
                                   err_msg=f"TRAIN_CLASS_MEANS['{cls}'] is stale")


def test_centroid_bank_is_weaker_than_the_real_sample_bank(real_samples):
    """The documented ordering between the two banks, pinned on held-out data.

    Measured on the 39 val+test samples: real-member bank 89.7% / 74.1% balanced;
    baked-in centroids 82.1% / 68.5%. default_reference_bank()'s docstring says it is the
    weaker option — if that inverts, the recommendation must be rewritten.
    """
    train = real_samples["three_gene_train"]
    held = real_samples["three_gene_val"] + real_samples["three_gene_test"]
    truth = np.array([l for l, _ in held])
    classes = sorted({l for l, _ in train})
    real_bank = M.build_reference_bank({c: [f for l, f in train if l == c] for c in classes})

    def acc(bank):
        pred = np.array([M.classify_morphology(f, bank).label for _, f in held])
        per = {c: float((pred[truth == c] == c).mean()) for c in classes if (truth == c).any()}
        return float((pred == truth).mean()), float(np.mean(list(per.values())))

    real_acc, real_bal = acc(real_bank)
    cent_acc, cent_bal = acc(M.default_reference_bank())
    assert real_acc > cent_acc, (
        f"real-member bank ({real_acc:.3f}) no longer beats the centroid bank "
        f"({cent_acc:.3f}); default_reference_bank()'s 'weaker option' claim is false")
    assert real_bal > cent_bal
    assert cent_acc >= 0.75, f"centroid bank regressed: {cent_acc:.3f} (measured 0.821)"


def test_margin_is_smaller_for_incorrect_calls(pooled):
    """The margin must carry information about how borderline a call is.

    Measured: median margin 0.491 for correct calls vs 0.139 for the 10 incorrect ones.
    The module only claims the medians separate — NOT that the margin is a reliable error
    detector — so that is all this asserts.
    """
    recs, labels, _ = pooled
    classes = sorted(set(labels))
    bank = M.build_reference_bank({c: [f for l, f in recs if l == c] for c in classes})
    correct, wrong = [], []
    for (truth, field) in recs:
        # leave the sample itself out of the bank so the nearest member is never itself
        loo = {c: np.stack([v for v in bank[c]
                            if not np.allclose(v, M.morphology_vector(field))])
               for c in classes}
        call = M.classify_morphology(field, loo)
        (correct if call.label == truth else wrong).append(call.margin)
    assert wrong, "no incorrect calls at all — the pin below cannot be evaluated"
    assert np.median(correct) > np.median(wrong)
    assert np.median(correct) > 2 * np.median(wrong)


def test_same_field_has_zero_distance(pooled):
    recs, _, _ = pooled
    field = recs[0][1]
    assert M.morphology_distance(field, field) == 0.0
    assert M.spectral_distance_2d(field, field) == 0.0


def test_within_class_distance_is_smaller_than_between_class(pooled):
    """The distance must order pairs the way the classes do, on average."""
    recs, labels, vecs = pooled
    z = vecs / M.TRAIN_SCALE
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    same = (labels[:, None] == labels[None, :]) & ~np.eye(len(labels), dtype=bool)
    diff = labels[:, None] != labels[None, :]
    assert d[same].mean() < d[diff].mean()


# --------------------------------------------------------------------------------------
# 2. invariance to intensity rescaling — brightness is not morphology
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [(3.0, 0.0), (0.1, 0.0), (1.0, 5.0), (2.5, -1.0),
                                 (100.0, 0.0)])
def test_morphology_vector_is_affine_invariant(a, b):
    field = _synthetic("spots", seed=1)
    np.testing.assert_allclose(M.morphology_vector(a * field + b),
                               M.morphology_vector(field), rtol=0, atol=1e-9)


@pytest.mark.parametrize("a,b", [(3.0, 0.0), (0.1, 0.0), (1.0, 5.0), (7.5, -2.0)])
def test_distances_are_zero_under_rescale(a, b):
    """A constant intensity rescale must move NEITHER metric.

    This is the property that fixed the spectral descriptor's normalisation order:
    log1p-then-normalise put a 3x rescale 1.0e-3 away from itself, which would let a
    brightness change masquerade as a morphology change. Measured now: 6.6e-18.
    """
    field = _synthetic("stripes", seed=2)
    assert M.morphology_distance(field, a * field + b) < 1e-9
    assert M.spectral_distance_2d(field, a * field + b) < 1e-12


def test_rescale_does_not_change_the_class_call(pooled):
    recs, _, _ = pooled
    bank = M.default_reference_bank()
    for _, field in recs[:20]:
        base = M.classify_morphology(field, bank)
        scaled = M.classify_morphology(4.0 * field - 3.0, bank)
        assert base.label == scaled.label
        assert abs(base.margin - scaled.margin) < 1e-9


# --------------------------------------------------------------------------------------
# 3. loud failure (house style: raise, never a silent fallback / zero / NaN)
# --------------------------------------------------------------------------------------
def test_shape_mismatch_raises_not_nan():
    a, b = _synthetic("spots", n=64), _synthetic("spots", n=32)
    with pytest.raises(ValueError, match="shape mismatch"):
        M.morphology_distance(a, b)
    with pytest.raises(ValueError, match="shape mismatch"):
        M.spectral_distance_2d(a, b)


def test_non_2d_field_raises():
    stack = np.stack([_synthetic("spots", n=32)] * 3)          # (3, 32, 32)
    with pytest.raises(ValueError, match="2-D"):
        M.morphology_vector(stack)
    with pytest.raises(ValueError, match="2-D"):
        M.morphology_distance(stack, stack)


def test_non_square_field_raises():
    with pytest.raises(ValueError, match="square"):
        M.morphology_vector(np.random.default_rng(0).standard_normal((32, 48)))


def test_non_finite_field_raises():
    field = _synthetic("spots", n=32)
    field[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        M.morphology_vector(field)
    field[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        M.morphology_vector(field)


def test_constant_field_raises_rather_than_returning_nan():
    """A collapsed / unpatterned field has no morphology. It must not score as 0 or NaN."""
    flat = np.full((32, 32), 2.5)
    with pytest.raises(ValueError):
        M.morphology_vector(flat)
    with pytest.raises(ValueError):
        M.spectral_distance_2d(flat, flat)


def test_bad_scale_raises():
    field = _synthetic("spots", n=32)
    for bad in (np.zeros(4), -np.ones(4), np.array([1.0, 2.0]), np.array([1, np.nan, 1, 1])):
        with pytest.raises(ValueError):
            M.morphology_distance(field, field, scale=bad)


def test_degenerate_bank_raises():
    field = _synthetic("spots", n=32)
    with pytest.raises(ValueError, match="required"):
        M.classify_morphology(field, None)
    with pytest.raises(ValueError, match="runner-up|>= 2"):
        M.classify_morphology(field, {"spots": np.zeros((1, 4))})
    with pytest.raises(ValueError, match="FEATURE_ORDER|shape"):
        M.classify_morphology(field, {"a": np.zeros((1, 3)), "b": np.zeros((1, 3))})
    with pytest.raises(ValueError):
        M.build_reference_bank({"spots": []})


def test_bad_spectral_block_size_raises():
    field = _synthetic("spots", n=32)
    for n in (0, 2, 33, 64):
        with pytest.raises(ValueError, match="block size"):
            M.spectral_distance_2d(field, field, n=n)


def test_feature_order_is_stable_and_matches_observables():
    """The fixed order is a contract: the scales and banks are indexed by it positionally."""
    from rngrn import observables as obs
    assert M.FEATURE_ORDER == ("skew", "kurtosis", "ang_conc", "ang_entropy")
    stats = obs.morphology(_synthetic("spots", n=32))
    assert set(M.FEATURE_ORDER) <= set(stats), (
        "observables.morphology no longer returns all four features scoring expects")
    assert len(M.TRAIN_SCALE) == len(M.TRAIN_MEAN) == len(M.FEATURE_ORDER)


# --------------------------------------------------------------------------------------
# 4. the 2-D spectral diagnostic vs the 1-D radial average
# --------------------------------------------------------------------------------------
def test_2d_spectrum_separates_better_than_radial_average(pooled):
    """WHY the diagnostic is 2-D: radial averaging destroys the stripes anisotropy.

    Measured leave-one-out 1-NN morphology accuracy over the 127 samples: 83.5% for this
    2-D block (n=24) vs 54.3% for the radially-averaged 1-D spectrum, against a 47.2%
    baseline. The 1-D average is barely above chance. This test asserts the ORDERING and a
    floor on the 2-D number, not the exact values.
    """
    from rngrn import observables as obs
    recs, labels, _ = pooled
    classes = sorted(set(labels))

    def loo(feats):
        d = np.sqrt(((feats[:, None, :] - feats[None, :, :]) ** 2).mean(-1))
        np.fill_diagonal(d, np.inf)
        pred = labels[d.argmin(1)]
        return float((pred == labels).mean())

    two_d = np.stack([M._spectral_block(f, 24) for _, f in recs])
    radial = []
    for _, f in recs:
        p = obs.raps(f, 1.0)[1][1:25]
        p = p / p.sum()
        p = np.log1p(p)
        radial.append(p / p.sum())
    acc_2d, acc_1d = loo(two_d), loo(np.stack(radial))
    assert acc_2d >= 0.78, f"2-D spectral separability regressed: {acc_2d:.3f} (measured 0.835)"
    assert acc_2d > acc_1d + 0.15, (
        f"2-D ({acc_2d:.3f}) no longer clearly beats the radial average ({acc_1d:.3f}); "
        f"the module's stated reason for being 2-D would be false")
    # the 2-D block is the SECONDARY diagnostic: it must not beat the 4-stat headline
    acc_stats, _ = _loo_nearest_member(
        np.stack([M.morphology_vector(f) for _, f in recs]), labels, M.TRAIN_SCALE)
    assert acc_stats > acc_2d, (
        "the 2-D spectral diagnostic now outperforms the 4-stat headline metric; the "
        "'secondary' framing needs revisiting rather than silently keeping the old order")


def test_spectral_block_is_normalised():
    block = M._spectral_block(_synthetic("stripes", n=64), 24)
    assert block.shape == (24 * 24,)
    assert np.isclose(block.sum(), 1.0)
    assert np.all(block >= 0)


def test_spectral_distance_orders_similar_before_dissimilar():
    stripes = _synthetic("stripes", n=64, seed=3)
    stripes2 = _synthetic("stripes", n=64, seed=4)
    noise = _synthetic("noise", n=64, seed=5)
    assert (M.spectral_distance_2d(stripes, stripes2)
            < M.spectral_distance_2d(stripes, noise))


# --------------------------------------------------------------------------------------
# 5. wiring into validate.score_recovery
# --------------------------------------------------------------------------------------
class _Key:
    """AnswerKey stand-in. Morphology does not read it, but score_recovery does."""
    def __init__(self, J, n_true, kstar=1.0):
        self.J = None if J is None else np.asarray(J, float)
        self.n_species_true = n_true
        self.kstar = kstar
        self.kstar_fft = None
        self.x_star = None
        self.sigma_max = None
        self.D = None
        self.coefficients = None


class _Result:
    def __init__(self, N=3, seed=0):
        from rngrn.model import RNGRN
        self.model = RNGRN(N=N, seed=seed)
        self.xstar = np.full(N, 1.0)
        self.kstar_model = 1.0
        self.latent_fields = None


def test_score_recovery_records_morphology_when_frames_given():
    from rngrn.validate import score_recovery
    tgt = _synthetic("spots", n=64, seed=6)
    mdl = _synthetic("spots", n=64, seed=7)
    out = score_recovery(_Result(), _Key(np.ones((3, 3)), 3), observed_idx=[0, 1],
                         target_frame=tgt, model_frame=mdl)
    assert out["morphology_scored"] == "compared"
    for key in ("morphology_distance", "morphology_match", "morphology_pred",
                "morphology_margin", "spectral_distance_2d"):
        assert key in out, f"score_recovery did not record {key}"
    assert isinstance(out["morphology_match"], bool)
    assert out["morphology_distance"] >= 0.0
    assert out["morphology_margin"] >= 0.0
    assert isinstance(out["morphology_pred"], str)


def test_score_recovery_omits_morphology_without_frames():
    """No frame at all -> no morphology keys, and an explicit reason. NOT a NaN."""
    from rngrn.validate import score_recovery
    out = score_recovery(_Result(), _Key(np.ones((3, 3)), 3), observed_idx=[0, 1])
    assert out["morphology_scored"] == "not_scored"
    assert "morphology_skipped_reason" in out
    assert "morphology_distance" not in out
    assert "morphology_match" not in out
    assert "morphology_pred_target" not in out


def test_score_recovery_scores_the_target_alone_without_a_rollout():
    """target_frame alone is FREE and must be scored; the COMPARISON keys must be absent.

    A caller that has not simulated the recovered model passes only the target. That must
    still record the target's morphology and must NOT invent the comparison keys.
    score_recovery never simulates anything itself — fit() does the rollout and hands the
    field in (see test_fit_records_morphology_end_to_end).
    """
    from rngrn.validate import score_recovery
    out = score_recovery(_Result(), _Key(np.ones((3, 3)), 3), observed_idx=[0, 1],
                         target_frame=_synthetic("spots", n=64, seed=11))
    assert out["morphology_scored"] == "target_only"
    assert out["morphology_pred_target"] in ("spots", "stripes", "labyrinth")
    assert out["morphology_margin_target"] >= 0.0
    assert "morphology_skew_target" in out
    for absent in ("morphology_distance", "morphology_match", "morphology_pred",
                   "spectral_distance_2d", "morphology_skew_model"):
        assert absent not in out, f"{absent} recorded without a model_frame"


def test_score_recovery_scores_morphology_even_without_true_J():
    """Morphology depends only on the fields, so the no_true_J arm must still get it.

    score_recovery returns early in that arm; morphology is recorded before the return.
    """
    from rngrn.validate import score_recovery
    tgt = _synthetic("stripes", n=64, seed=8)
    out = score_recovery(_Result(), _Key(None, None), observed_idx=[0, 1],
                         target_frame=tgt, model_frame=tgt)
    assert out["scoring_mode"] == "no_true_J"
    assert out["morphology_scored"] == "compared"
    assert out["morphology_match"] is True
    assert out["morphology_distance"] == 0.0


def test_score_recovery_accepts_a_channel_stack():
    """(m, H, W) frames are accepted; channel 0 is scored, matching recovery's k* channel."""
    from rngrn.validate import score_recovery
    field = _synthetic("spots", n=64, seed=9)
    stack = np.stack([field, _synthetic("noise", n=64, seed=10)])
    out = score_recovery(_Result(), _Key(None, None), target_frame=stack,
                         model_frame=field)
    assert out["morphology_distance"] == 0.0, (
        "channel 0 of the stack was not the field used")


def test_fit_records_morphology_end_to_end(tmp_path):
    """The wiring that matters: a real fit() run must come back with morphology COMPARED.

    Tiny scale — this checks the WIRE, not recovery quality (a dry run is never a finding).

    Until unit 7 this asserted morphology_scored == "target_only" and that
    morphology_distance was ABSENT, because a rollout cost 6.5-10 min per field. That cost
    was a horizon bug (eval/rollout.py); fit() now pays 0.9-1.7 s and the comparison is
    recorded on every run. The assertion is inverted deliberately: "target_only" is now the
    failure this test exists to catch.
    """
    import os
    from rngrn.config import load_config, apply_overrides
    from rngrn.train import fit
    configs = pathlib.Path(__file__).resolve().parents[1] / "configs"
    cfg = load_config(str(configs / "milestone1_schnak.yaml"))
    cfg = apply_overrides(cfg, [
        "data.resolution=32", "data.T_max=5.0", "data.dt=0.05",
        "model.N=2", "model.m=2",
        "train.n_restarts=1", "train.adam_steps=8", "train.lbfgs_steps=0",
        "solver.n_grid=32", "solver.robustness_samples=5",
        f"data.cache_root={tmp_path / 'cache'}",
    ])
    metric = fit(cfg, runs_root=str(tmp_path / "experiments"))
    assert metric["morphology_pred_target"] in ("spots", "stripes", "labyrinth")
    for key in M.FEATURE_ORDER:
        assert f"morphology_{key}_target" in metric
    assert metric["rollout_status"] in ("ok", "blew_up", "non_finite", "unpatterned")
    if metric["rollout_status"] != "ok":
        pytest.skip(f"the recovered model's rollout was {metric['rollout_status']}, which "
                    f"is a legitimate outcome for an 8-step fit; nothing to compare")
    assert metric["morphology_scored"] == "compared", (
        "fit() no longer supplies a simulated model field, so morphology_match — the "
        "primary criterion — would go uncomputed on every run again")
    assert "morphology_distance" in metric
    assert isinstance(metric["morphology_match"], bool)
    assert metric["rollout_grid"] == 32, "the model field must be on the TARGET's grid"


def test_score_recovery_raises_on_a_bad_frame_rank():
    from rngrn.validate import score_recovery
    bad = np.zeros((2, 3, 16, 16))
    with pytest.raises(ValueError, match="H, W"):
        score_recovery(_Result(), _Key(None, None), target_frame=bad,
                       model_frame=_synthetic("spots", n=16))


def test_score_recovery_propagates_a_shape_mismatch_loudly():
    """A grid mismatch between target and model field must raise, not score as 'far'."""
    from rngrn.validate import score_recovery
    with pytest.raises(ValueError, match="shape mismatch"):
        score_recovery(_Result(), _Key(None, None),
                       target_frame=_synthetic("spots", n=64),
                       model_frame=_synthetic("spots", n=32))


# --------------------------------------------------------------------------------------
# 6. firewall
# --------------------------------------------------------------------------------------
def _imports_of(path):
    tree = ast.parse(pathlib.Path(path).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names += [f"{mod}.{a.name}" for a in node.names] + [mod]
    return " ".join(names)


def test_morphology_scoring_does_not_import_the_recovery_engine():
    """Scoring side. It may reuse the image-only measurement (observables.morphology reads
    the frame and nothing else) but must not reach the model / optimiser / loss."""
    joined = _imports_of(M.__file__)
    for forbidden in ("rngrn.model", "rngrn.recover", "rngrn.losses", "rngrn.train",
                      ".model", ".recover", ".losses", "torch"):
        assert forbidden not in joined, (
            f"scoring/morphology.py imports '{forbidden}'; scoring-side code must not "
            f"pull in the recovery engine")


def test_morphology_scoring_reads_no_answer_key():
    """This module scores against a LABEL supplied by the caller, not the answer key."""
    joined = _imports_of(M.__file__)
    for forbidden in ("rd_models", "data.solver", "data.cache", "AnswerKey"):
        assert forbidden not in joined, (
            f"scoring/morphology.py imports '{forbidden}'")


def test_no_recovery_side_module_imports_the_morphology_scorer():
    """The direction that actually matters: recovery must not reach scoring.

    Mirrors tests/test_firewall.py's list rather than importing it, so the audits stay
    independent. In particular, wiring morphology into the differentiable loss would show
    up here — losses/terms.py importing this module would fail.
    """
    src_root = pathlib.Path(M.__file__).resolve().parents[1]
    recovery_side = [
        "model.py", "observables.py", "recover.py",
        "losses/terms.py", "losses/total.py", "losses/weighting.py",
        "eval/rollout.py", "eval/numerics.py", "eval/dynamical.py",
        "eval/analysis.py", "eval/topology.py",
    ]
    for relpath in recovery_side:
        path = src_root / relpath
        assert path.exists(), f"missing recovery-side module {relpath}"
        assert "scoring" not in _imports_of(path), (
            f"FIREWALL BREACH: {relpath} imports the scoring package")


def test_scoring_init_still_imports_nothing():
    """scoring/__init__ deliberately imports no submodule, so one failing scorer cannot
    mask another. Adding morphology.py must not have changed that."""
    import rngrn.scoring as pkg
    joined = _imports_of(pkg.__file__)
    assert "morphology" not in joined
    assert joined.strip() == "", f"scoring/__init__.py grew imports: {joined!r}"
