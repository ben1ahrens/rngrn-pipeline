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
                   field and that costs a rollout (~seconds).
  1. wavelength  : recovered model k* vs answer-key k*.
                   HEADLINE  `kstar_rel_err`     — against the LINEAR k* (answer_key.kstar),
                                                   the like-for-like dispersion-relation
                                                   comparison. Tune and report on this.
                   SECONDARY `kstar_fft_rel_err` — against the FFT-MEASURED k*
                                                   (answer_key.kstar_fft). Diagnostic only;
                                                   it is quantised onto the FFT bin grid, so
                                                   it differs from the linear number by an
                                                   offset of EITHER SIGN (see the note in
                                                   score_recovery) and is not a target.
  2. regime      : does recovered J satisfy the Turing conditions?
  3. sign        : recovered J sign-structure vs answer-key J sign-structure
  4. robustness  : left to the analysis stage (eval.robustness_cloud), summarised here
  5. plausibility: recovered model's own parameters (alpha, delta, beta, D-ratio)
                   against configs/bio_box.yaml. Unconditional (no answer-key
                   dependence). See scoring/plausibility.py.

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
      additionally needs a field simulated from the recovered model. A default-horizon
      ETDRK4 rollout on a 96x96 grid measured ~4.2 ms/step, and the step count is derived
      from the model's own sigma_max (T = horizon_growth_times / sigma_max, dt from the
      fastest reaction rate). For an untrained N=3 model that came to ~128k steps, i.e.
      ~9 minutes for one field — and rollout.simulate clips nsteps at 200k, so ~14 min is
      the ceiling, not the typical case. The comparison happens only when the caller has
      already paid for that rollout and passes the field in.

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


def score_recovery(result, answer_key, observed_idx=None, target_frame=None,
                   model_frame=None, morphology_bank=None,
                   spectral_block=24) -> dict:
    """Grade a RecoveryResult against an AnswerKey. Returns a flat metric dict.

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
        is nothing to reach for, and a caller who has not paid for a rollout (measured
        ~9 min for one 96x96 field at the default horizon) must not silently be charged for one
        or handed a fabricated field. When a frame is absent the corresponding keys are
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
    #    HEADLINE   kstar_rel_err      : vs answer_key.kstar, the LINEAR k* (argmax_k of
    #                                    sigma(k) from the generator's J, D). Like-for-like
    #                                    against kstar_model, which is the same quantity
    #                                    computed from the RECOVERED J, D. This is the
    #                                    number to tune against and report.
    #    SECONDARY  kstar_fft_rel_err  : vs answer_key.kstar_fft, the wavenumber MEASURED
    #                                    off the generated frame by FFT. A diagnostic only:
    #                                    it compares a dispersion-relation prediction to a
    #                                    finite-grid Fourier measurement, which is quantised
    #                                    onto the half-integer FFT-bin grid. So a non-zero
    #                                    floor is expected even for a perfect recovery, and
    #                                    the offset has EITHER SIGN depending on the sample
    #                                    (median |kstar_fft/kstar - 1| = 0.084 over the 287
    #                                    registered samples, 90th pct 0.250; the ratio's
    #                                    median is above 1 on most datasets but below 1 on
    #                                    three_gene_val). Do not tune on it and do not quote
    #                                    it as the headline.
    out["kstar_model"] = float(result.kstar_model)
    out["kstar_true"] = _rel_ref(answer_key, "kstar")
    out["kstar_rel_err"] = _rel_err(result.kstar_model, out["kstar_true"])
    out["kstar_fft_true"] = _rel_ref(answer_key, "kstar_fft")
    out["kstar_fft_rel_err"] = _rel_err(result.kstar_model, out["kstar_fft_true"])

    # 2. regime — Turing conditions on the RECOVERED model
    J_rec = result.model.jacobian(
        __import__("torch").as_tensor(result.xstar), create_graph=False).detach().cpu().numpy()
    D_rec = result.model.D.detach().cpu().numpy()
    ok, info = turing_ok(J_rec, D_rec)
    out["recovered_turing"] = bool(ok)
    out["recovered_sig_max"] = float(info["sig_max"])
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
