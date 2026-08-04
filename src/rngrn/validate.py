"""validate.py — score a recovery against the quarantined answer key.

Scoring priority (owner priority — MORPHOLOGY, then wavelength & regime, OVER analogous
parameters):
  0. morphology  : does the recovered model reproduce the target frame's pattern
                   MORPHOLOGY (same dominant spatial mode; spots / stripes / labyrinth)?
                   The owner's PRIMARY criterion. See scoring/morphology.py for the
                   metric, its data-derived scales, and its measured separability.
                   Recorded from EXPLICITLY PASSED fields (`target_frame`, `model_frame`)
                   and computed POST-HOC only — it never enters the differentiable loss,
                   because comparing a recovered model's morphology needs a simulated
                   field, and a rollout (0.9-1.7 s per field since unit 7) is many orders
                   of magnitude more expensive than one gradient step.
  1. wavelength  : recovered model k* vs answer-key k*.
                   HEADLINE  `kstar_fft_rel_err`  — against the FFT-MEASURED k*
                                                   (answer_key.kstar_fft), the wavenumber
                                                   actually measured off the image. This
                                                   REVERSED the 2026-07-26 decision on
                                                   2026-07-29 (docs/STATE_OF_THE_SCIENCE.md
                                                   line 499); it is quantised onto the FFT
                                                   bin grid so a non-zero floor is expected
                                                   even for a perfect recovery. Tune and
                                                   report on this.
                   SECONDARY `kstar_rel_err`     — against the LINEAR k* (answer_key.kstar),
                                                   the dispersion-relation comparison. A
                                                   property of the generating equations,
                                                   not of the image; diagnostic only, not a
                                                   target (see the note in score_recovery).
                   LEAK GATE — ONE CONTROL PER REFERENCE (D-EVID-7, corrected 2026-08-04;
                   this note previously claimed a single control "applies to BOTH columns",
                   which it did not). The LEGACY generators set L = clip(6*2*pi/k*, 18, 220),
                   so k*_true is identically 6*2*pi/L for 94.8% of the 287 legacy samples
                   and an image-blind predictor using L ALONE scores 1.4e-14% median error
                   on kstar_rel_err — better than recovery. Against the FFT reference the
                   SAME predictor scores several percent, because kstar_fft differs from
                   kstar by a measured median of 8.4%. Each column therefore carries the
                   control computed against its own denominator:
                       kstar_rel_err      <- trivial_kstar_err,     kstar_fft_bin_width
                       kstar_fft_rel_err  <- trivial_kstar_fft_err, kstar_fft_bin_width_rel_fft
                   Reading the headline against the linear control overstates the win by
                   orders of magnitude. NOTE also that `scripts/gen_tg3.py` draws
                   periods-per-box from 3..14, so the L = 6*lambda relation does NOT hold on
                   three_gene_qvar / three_gene_multiL and the bin-width resolution floor is
                   dataset-dependent there.
  2. regime      : does recovered J satisfy the Turing conditions?
  3. sign        : recovered J sign-structure vs answer-key J sign-structure
  4. robustness  : left to the analysis stage (eval.robustness_cloud), summarised here
  5. plausibility: recovered model's own parameters (alpha, delta, beta, D-ratio)
                   against configs/bio_box.yaml. Unconditional (no answer-key
                   dependence). See scoring/plausibility.py.

  5. L-generalis.: per-run pieces of the cross-L consistency metric (scoring/lgen.py).
                   Recorded on every run; the CROSS-L statistic itself only exists once
                   several runs of the same system at different L are in the index, so it
                   is aggregated there (scoring.lgen.lgen_table), not here.

This function is called by the harness AFTER recovery; it is the ONLY place the
answer key is read, and it is never imported by recovery-side modules. Parameter-
value agreement is deliberately NOT scored.
"""
from __future__ import annotations
import numpy as np

from .eval.analysis import turing_ok, robustness_volumes
from .scoring import morphology as MORPH
from .scoring import permutation as PERM
from .scoring import overparam as OVER
from .scoring import reproducibility as REPRO

from .scoring import plausibility as PLAUS

from .scoring import lgen as LGEN


def _sign_structure(J):
    J = np.asarray(J, float)
    return np.sign(np.where(np.abs(J) > 1e-9 * (np.abs(J).max() + 1e-12), J, 0.0))


def _rel_ref(answer_key, field):
    """Read a scalar reference wavenumber off the answer key as a float, or NaN if absent.

    NaN here means "this dataset carries no such reference", which is a legitimate state
    for the secondary FFT number (the reference cache stores no FFT measurement). It is NOT
    a swallowed failure: the gate raises when the *required* attributes are missing, so a
    NaN can only arrive from a source that genuinely has nothing to compare against.
    """
    v = getattr(answer_key, field, None)
    if v is None:
        return float("nan")
    v = float(v)
    if not np.isfinite(v):
        raise ValueError(f"answer_key.{field} is {v!r} — a non-finite reference wavenumber "
                         f"cannot be scored against")
    return v


def _rel_err(model_value, ref):
    """Relative error of the recovered k* against a reference, NaN when no reference."""
    if not np.isfinite(ref):
        return float("nan")
    if ref <= 0.0:
        raise ValueError(f"reference wavenumber must be positive, got {ref!r}")
    return float(abs(float(model_value) - ref) / ref)


def _first_channel(frame, name):
    """Take the single 2-D field a morphology comparison operates on.

    Accepts (H, W) or (m, H, W) and uses channel 0 of a stack — the same channel recovery
    itself measures k* from (recover.recover: obs.kstar_of(frame[0])), so the morphology
    score refers to the same observable. Raises on any other rank rather than guessing.
    """
    arr = np.asarray(frame, dtype=float)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 0:
            raise ValueError(f"{name} has zero channels; nothing to score")
        return arr[0]
    raise ValueError(
        f"{name} must be (H, W) or (m, H, W); got shape {arr.shape}. Morphology is a "
        f"single-field statistic — pass one frame, not a batch or a time series.")


def _morphology_metrics(target_frame, model_frame=None, reference_bank=None,
                        spectral_block=24) -> dict:
    """Morphology block. Returns a flat dict of scalars.

    Split by COST, deliberately:

    * target_frame alone is FREE — it is the observed image the run already loaded, so the
      target's own morphology class and four features are recorded on every run.
    * the COMPARISON (`morphology_distance`, `morphology_match`, `spectral_distance_2d`)
      additionally needs a field simulated from the recovered model. That used to cost
      6.5-10 minutes per field, which is why it had never once been computed; unit 7
      traced the cost to a horizon bug in eval/rollout.py and it is now 0.9-1.7 s
      (64x64 to 128x128, N=3, CPU, one thread). train.fit() therefore supplies the model
      field by default (solver.morphology_rollout). The comparison still happens only when
      the caller passes a field in — this function never simulates anything itself.

    `morphology_match` compares the two CLASS CALLS (target vs model field), which is the
    owner's criterion stated directly: same morphology class, not merely a small distance.
    The distance and both margins are reported alongside so a match can be read as
    confident or borderline; see scoring/morphology.py for the measured margin
    distribution and the weakness of the stripes class (7 of 127 samples).
    """
    tgt = _first_channel(target_frame, "target_frame")
    bank = MORPH.default_reference_bank() if reference_bank is None else reference_bank
    out = {"morphology_bank": ("default_centroids" if reference_bank is None
                               else "caller_supplied")}

    tgt_call = MORPH.classify_morphology(tgt, bank)
    out["morphology_pred_target"] = tgt_call.label
    out["morphology_margin_target"] = tgt_call.margin
    v_tgt = MORPH.morphology_vector(tgt)
    for i, key in enumerate(MORPH.FEATURE_ORDER):
        out[f"morphology_{key}_target"] = float(v_tgt[i])

    if model_frame is None:
        return out

    mdl = _first_channel(model_frame, "model_frame")
    out["morphology_distance"] = MORPH.morphology_distance(tgt, mdl)
    out["spectral_distance_2d"] = MORPH.spectral_distance_2d(tgt, mdl, n=spectral_block)
    mdl_call = MORPH.classify_morphology(mdl, bank)
    out["morphology_pred"] = mdl_call.label
    out["morphology_margin"] = mdl_call.margin
    out["morphology_match"] = bool(mdl_call.label == tgt_call.label)
    v_mdl = MORPH.morphology_vector(mdl)
    for i, key in enumerate(MORPH.FEATURE_ORDER):
        out[f"morphology_{key}_model"] = float(v_mdl[i])
    return out


def _leak_instrumentation(L, kstar_true, kstar_fft_true=float("nan")) -> dict:
    """The image-blind control for the k* numbers in this project — ONE PER REFERENCE.

    A predictor that never looks at the image — `k_trivial = 6*2*pi/L` — is the baseline
    every k* number must beat before it is evidence of recovery. Computing it here, from L
    alone, is what makes that comparison available on every row without a separate lookup.

    WHY THERE ARE TWO CONTROLS (D-EVID-7, 2026-08-04). This function used to return only
    the LINEAR-referenced control while the reported headline is `kstar_fft_rel_err`, which
    is measured against `answer_key.kstar_fft`. The legacy generators set
    L = clip(6*2*pi/k*, 18, 220), so `k_trivial` is exact against `kstar` to 1e-6 on those
    samples -- but `kstar_fft` differs from `kstar` by a measured median of 8.4%, so the
    same predictor scores several percent against the reference that actually gates the
    claim. A row showing `kstar_fft_rel_err = 0.075` beside `trivial_kstar_err = 1.2e-16`
    therefore read as a 14-order-of-magnitude win over a baseline that had never been
    measured on that column. Both columns are now emitted, each against its own denominator:

        trivial_kstar_err          controls  kstar_rel_err        (vs answer_key.kstar)
        trivial_kstar_fft_err      controls  kstar_fft_rel_err    (vs answer_key.kstar_fft)

    `trivial_kstar_err` keeps its exact former meaning so rows recorded before this change
    stay comparable.

    THE BIN WIDTHS ARE THE RESOLUTION FLOORS of their respective columns. One FFT bin is
    2*pi/L absolute; as a fraction of the reference it is 1/(periods-per-box). The legacy
    generators fix periods-per-box at 6, hence the familiar 16.7% and the pre-registered
    8.3% bar (half a bin). `scripts/gen_tg3.py` draws periods-per-box freely from 3..14, so
    on `three_gene_qvar` / `three_gene_multiL` the floor ranges over ~7%..33% and can EXCEED
    the pre-registered bar -- i.e. the bar can demand precision the estimator does not have.
    Read `kstar_fft_bin_width_rel_fft` before reading `kstar_fft_rel_err`.

    Every column is NaN rather than a fallback when its own reference is unavailable.
    Falling back to the other reference would reintroduce precisely the defect above.
    """
    nan = float("nan")
    out = {"trivial_kstar_err": nan, "kstar_fft_bin_width": nan,
           "trivial_kstar_fft_err": nan, "kstar_fft_bin_width_rel_fft": nan}
    if L is None:
        return out
    k_trivial = 6.0 * 2.0 * np.pi / float(L)
    bin_width = 2.0 * np.pi / float(L)
    if np.isfinite(kstar_true) and kstar_true > 0.0:
        out["trivial_kstar_err"] = float(abs(k_trivial - kstar_true) / kstar_true)
        out["kstar_fft_bin_width"] = float(bin_width / kstar_true)
    if np.isfinite(kstar_fft_true) and kstar_fft_true > 0.0:
        out["trivial_kstar_fft_err"] = float(abs(k_trivial - kstar_fft_true) / kstar_fft_true)
        out["kstar_fft_bin_width_rel_fft"] = float(bin_width / kstar_fft_true)
    return out


def score_recovery(result, answer_key, observed_idx=None, target_frame=None,
                   model_frame=None, morphology_bank=None,
                   spectral_block=24, L=None) -> dict:
    """Grade a RecoveryResult against an AnswerKey. Returns a flat metric dict.

    L : the domain size (gate.RecoveryInput.L) of the sample being scored, or None.
        Used ONLY to compute the image-blind LEAK-instrumentation columns below
        (`trivial_kstar_err`, `kstar_fft_bin_width`) from L alone — it never reaches the
        rel-err computation against the answer key. When None (the historical default,
        still used by any caller not yet updated to pass it) both leak columns are NaN
        rather than silently omitted, so their absence is visible in the row, not hidden.

    target_frame : the observed frame that was recovered from. (H, W) or (m, H, W);
        channel 0 is scored, the same channel recovery measures k* from. Given this alone,
        the TARGET's own morphology class and four features are recorded (free — no
        rollout) and `morphology_scored` is "target_only".
    model_frame : a field simulated from the RECOVERED model, e.g.
        eval.rollout.simulate(...)["fields"]. Supplying it adds the actual comparison
        (`morphology_distance`, `morphology_match`, `spectral_distance_2d`) and
        `morphology_scored` becomes "compared".
        Both frames are passed EXPLICITLY rather than pulled off `result` or a global: a
        RecoveryResult carries neither the observed frame nor a simulated field, so there
        is nothing to reach for, and a caller who has not paid for a rollout must not
        silently be charged for one or handed a fabricated field. `model_frame` must be on
        the SAME grid as `target_frame` — a mismatch raises rather than scoring as "far".
        When a frame is absent the corresponding keys are
        OMITTED (never NaN-filled), so an unscored morphology cannot be misread as a bad
        one; `morphology_scored` says which level was reached.
    morphology_bank : reference bank for the class call. Defaults to
        scoring.morphology.default_reference_bank() (baked-in three_gene_train centroids —
        the weaker option; see that module). Pass build_reference_bank(...) over real
        fields when the dataset is reachable at scoring time.
    spectral_block : block size n for the secondary spectral_distance_2d diagnostic.
    """
    out = {}

    # 0. morphology — the owner's PRIMARY criterion, and previously scored nowhere.
    #    Recorded FIRST, and before the no_true_J early return below, because it depends
    #    only on the two FIELDS: a dataset carrying no answer-key J still gets a
    #    morphology score. Post-hoc only; the differentiable loss is untouched.
    if target_frame is not None:
        out.update(_morphology_metrics(target_frame, model_frame,
                                       reference_bank=morphology_bank,
                                       spectral_block=spectral_block))
        # Three honest states, not a bool: "compared" (both fields), "target_only" (the
        # target's morphology recorded, no rollout paid for), "not_scored" (no frame).
        out["morphology_scored"] = "compared" if model_frame is not None else "target_only"
    else:
        # Not an error: scoring a run whose caller passed no frame is legitimate. Say so
        # explicitly rather than emitting a NaN that reads like a failed comparison.
        out["morphology_scored"] = "not_scored"
        out["morphology_skipped_reason"] = "no target_frame"

    # 1. wavelength — TWO references, and they are not interchangeable.
    #
    #    HEADLINE   kstar_fft_rel_err  : vs answer_key.kstar_fft, the wavenumber MEASURED
    #                                    off the generated frame by FFT — the quantity an
    #                                    inverse problem given only the image can actually
    #                                    be graded against. Owner decision 2026-07-29
    #                                    (docs/STATE_OF_THE_SCIENCE.md line 499), reversing
    #                                    the 2026-07-26 decision that made kstar_rel_err the
    #                                    headline. It is quantised onto the half-integer FFT
    #                                    bin grid, so a non-zero floor is expected even for a
    #                                    perfect recovery, and the offset has EITHER SIGN
    #                                    depending on the sample (median |kstar_fft/kstar -
    #                                    1| = 0.084 over the 287 registered samples, 90th pct
    #                                    0.250; the ratio's median is above 1 on most
    #                                    datasets but below 1 on three_gene_val). Tune and
    #                                    report on this.
    #    SECONDARY  kstar_rel_err      : vs answer_key.kstar, the LINEAR k* (argmax_k of
    #                                    sigma(k) from the generator's J, D). A property of
    #                                    the generating equations, not of the image — no
    #                                    experiment can observe it directly. Diagnostic
    #                                    only; do not tune on it and do not quote it as the
    #                                    headline.
    #
    #    LEAK GATE — read before trusting either column above. Every generator sets
    #    L = clip(6*2*pi/k*, 18, 220) (data/staging/tg3/generator.py), so k*_true is
    #    identically 6*2*pi/L to 1e-6 for 94.8% of all 287 registered samples (and
    #    periods-per-box is exactly 6.000 for all 127 three_gene samples). An image-blind
    #    predictor using L ALONE — never looking at the frame — scores 1.4e-14% median
    #    error on kstar_rel_err, beating measured recovery (3.14%). So no k* number here is
    #    interpretable at face value: `trivial_kstar_err` (the L-only predictor's relative
    #    error) is recorded on every row precisely so a reader of runs.jsonl can never see a
    #    k* error without simultaneously seeing what ignoring the image entirely would have
    #    scored. Treat any k* column as a GATE (regime / order-of-magnitude sanity), never
    #    as evidence of recovery, until it clears its own trivial baseline by a wide margin.
    #    Also see `kstar_fft_bin_width`: one FFT bin is 16.7% of k*, i.e. the configured
    #    tolerance loss.tau = 0.12 (0.72 of one bin) sits BELOW the FFT estimator's own
    #    resolution — see TUNING.md.
    out["kstar_model"] = float(result.kstar_model)
    out["kstar_true"] = _rel_ref(answer_key, "kstar")
    out["kstar_rel_err"] = _rel_err(result.kstar_model, out["kstar_true"])
    out["kstar_fft_true"] = _rel_ref(answer_key, "kstar_fft")
    out["kstar_fft_rel_err"] = _rel_err(result.kstar_model, out["kstar_fft_true"])

    # LEAK INSTRUMENTATION — computed from L and the answer-key references ONLY, never from
    # the recovered model, so it measures what an image-blind predictor would have scored,
    # not what recovery actually did. FOUR columns, PAIRED BY REFERENCE (D-EVID-7):
    #   trivial_kstar_err / kstar_fft_bin_width          control kstar_rel_err     (vs kstar)
    #   trivial_kstar_fft_err / kstar_fft_bin_width_rel_fft
    #                                                    control kstar_fft_rel_err (vs kstar_fft)
    # Reading the headline `kstar_fft_rel_err` against `trivial_kstar_err` is a CATEGORY
    # ERROR: the legacy generators set L = clip(6*2*pi/k*, 18, 220), so the predictor is
    # exact against kstar but several percent against kstar_fft, which is the reference the
    # claim is gated on. Each column is NaN when its own reference or L is unavailable —
    # never a fallback to the other reference, which is the defect this pairing fixes.
    out.update(_leak_instrumentation(L, out["kstar_true"], out["kstar_fft_true"]))

    # 2. regime — Turing conditions on the RECOVERED model
    J_rec = result.model.jacobian(
        __import__("torch").as_tensor(result.xstar), create_graph=False).detach().cpu().numpy()
    # PHYSICAL diffusivities. On the non-dimensional recovery path (recover(nondim=True))
    # `result.model.D` is D/L**2, which would put the Turing verdict on a different k scale
    # from every dimensional run; `result.D_phys` is the converted value and is bit-equal to
    # model.D on the dimensional path. getattr, because the scoring tests pass duck-typed
    # stand-ins that carry only (model, xstar, kstar_model).
    D_rec = getattr(result, "D_phys", None)
    D_rec = (result.model.D.detach().cpu().numpy() if D_rec is None
             else np.asarray(D_rec, dtype=float))
    ok, info = turing_ok(J_rec, D_rec)
    # `recovered_turing` is the STRICT verdict as of 2026-08-04 (D-EVID-11): uniformly
    # stable by max Re eig(J) < 0, structurally unstable at some k > 0. It previously used
    # tr(J) < 0, which a uniformly UNSTABLE system can satisfy. `recovered_turing_loose`
    # carries the superseded verdict so rows recorded either side of the change stay
    # readable against each other; per docs/DECISIONS.md D-BIO-3 the loose verdict is
    # reported, never led with. MEASURED: the two agree on all 12 tracked run rows that
    # store a Jacobian, so no recorded number moved — the correction bites at init
    # (low_basal: 51.8% loose vs 0.0% strict), not on recoveries.
    out["recovered_turing"] = bool(ok)
    out["recovered_turing_loose"] = bool(info["turing_loose"])
    out["recovered_sig_max"] = float(info["sig_max"])
    out["recovered_sig0"] = float(info["sig0"])          # max Re eig(J): the uniform mode
    out["recovered_tr0"] = float(info["tr0"])

    # 4. robustness — local Turing-volume fraction of the RECOVERED (J, D) under a
    #    log-normal parameter cloud, at Tica's four noise levels. See
    #    eval.analysis.robustness_volumes and docs/ROBUSTNESS_MEASUREMENT.md section 4
    #    for the perturbation model and the baseline to read this against.
    out.update(robustness_volumes(result.model, xstar=result.xstar))

    # unit 3 (repro-metric): per-run fields consumed by optim.benchmark's cross-seed
    # topology reproducibility aggregation (scoring/reproducibility.py). Computed
    # unconditionally — it needs no answer key — so it must land here, before the
    # no_true_J early return below, not after it.
    out.update(REPRO.per_run_fields(J_rec, D_rec, out["kstar_model"]))

    # 2.5 biological plausibility (unit 5) — the RECOVERED model's own parameters against
    #     configs/bio_box.yaml. Unconditional: reads no answer-key quantity, so it is
    #     recorded even when the run has no true J (the no_true_J early return below).
    out.update(PLAUS.plausibility_report(result.model.alpha, result.model.delta,
                                         result.model.beta, result.model.D))

    # 2b. L-generalisation per-run pieces (unit 12). Placed here, not after the sign block,
    #     because that block returns early in the `no_true_J` arm and these metrics do not
    #     depend on the true J — a dataset with no answer-key J still gets them.
    out.update(LGEN.per_run_lgen_metrics(result, J_rec, D_rec,
                                         getattr(answer_key, "system_id", None)))

    # 3. sign structure vs answer key — routed by ARM, never a silent NaN.
    #
    #    same-size J      -> permutation-aligned sign match (observed indices pinned).
    #    model N > true N -> no correct full J exists; score the spare species instead
    #                        (Experiment B) and compare the OBSERVED sub-block.
    #    The observed sub-block is the one comparison valid in every arm, so it is always
    #    reported when a true J is available.
    ak_J = getattr(answer_key, "J", None)
    n_true = getattr(answer_key, "n_species_true", None)
    n_model = int(J_rec.shape[0])
    out["n_true"] = n_true
    out["n_model"] = n_model

    # A missing true J can arrive as None OR as a 0-d/empty array (np.asarray(None) is a
    # 0-d object array, which is NOT None) — check the resulting shape, not just identity.
    ak_J = None if ak_J is None else np.asarray(ak_J, dtype=object)
    if ak_J is None or ak_J.ndim != 2 or ak_J.size == 0:
        out["sign_match_frac"] = float("nan")
        out["scoring_mode"] = "no_true_J"
        return out

    ak_J = np.asarray(ak_J, float)
    obs_idx = list(observed_idx) if observed_idx is not None else list(range(min(
        n_model, ak_J.shape[0])))

    # always-valid cross-arm comparison
    try:
        sub = PERM.observed_subblock_score(J_rec, ak_J, obs_idx)
        out["subblock_sign_match"] = sub["sign_match_frac_observed"]
        out["subblock_fro_rel_err"] = sub["fro_rel_err_observed"]
    except Exception as exc:                     # fail loud in the record, not silently
        out["subblock_sign_match"] = float("nan")
        out["subblock_error"] = f"{type(exc).__name__}: {exc}"

    if ak_J.shape == J_rec.shape:
        aligned = PERM.permuted_sign_match(J_rec, ak_J, obs_idx)
        out["sign_match_frac"] = aligned["sign_match_frac_aligned"]
        out["sign_match_frac_aligned"] = aligned["sign_match_frac_aligned"]
        out["sign_match_frac_identity"] = aligned["sign_match_frac_identity"]
        out["best_perm"] = str(aligned["best_perm"])
        out["n_permutations_searched"] = aligned["n_permutations_searched"]
        out["scoring_mode"] = "permutation_aligned"
    elif n_true is not None and n_model > n_true:
        # Experiment B: there is no correct full J. Score spare-species inertness.
        rep = OVER.overparam_report(result, answer_key, obs_idx)
        for k, v in rep.items():
            out.setdefault(k, v)
        out["sign_match_frac"] = float("nan")     # deliberately undefined in this arm
        out["scoring_mode"] = "overparameterised"
    else:
        out["sign_match_frac"] = float("nan")
        out["scoring_mode"] = f"shape_mismatch_{J_rec.shape[0]}x{ak_J.shape[0]}"

    return out
