"""validate.py — score a recovery against the quarantined answer key.

Scoring priority (owner priority — wavelength & regime OVER analogous parameters):
  1. wavelength  : recovered model k* vs answer-key k* (relative error)
  2. regime      : does recovered J satisfy the Turing conditions?
  3. sign        : recovered J sign-structure vs answer-key J sign-structure
  4. robustness  : left to the analysis stage (eval.robustness_cloud), summarised here

This function is called by the harness AFTER recovery; it is the ONLY place the
answer key is read, and it is never imported by recovery-side modules. Parameter-
value agreement is deliberately NOT scored.
"""
from __future__ import annotations
import numpy as np

from .eval.analysis import turing_ok
from .scoring import permutation as PERM
from .scoring import overparam as OVER


def _sign_structure(J):
    J = np.asarray(J, float)
    return np.sign(np.where(np.abs(J) > 1e-9 * (np.abs(J).max() + 1e-12), J, 0.0))


def score_recovery(result, answer_key, observed_idx=None) -> dict:
    """Grade a RecoveryResult against an AnswerKey. Returns a flat metric dict."""
    out = {}

    # 1. wavelength
    ak_kstar = getattr(answer_key, "kstar", None)
    out["kstar_model"] = float(result.kstar_model)
    out["kstar_true"] = float(ak_kstar) if ak_kstar is not None else float("nan")
    out["kstar_rel_err"] = (abs(result.kstar_model - ak_kstar) / (ak_kstar + 1e-9)
                            if ak_kstar is not None else float("nan"))

    # 2. regime — Turing conditions on the RECOVERED model
    J_rec = result.model.jacobian(
        __import__("torch").as_tensor(result.xstar), create_graph=False).detach().numpy()
    D_rec = result.model.D.detach().numpy()
    ok, info = turing_ok(J_rec, D_rec)
    out["recovered_turing"] = bool(ok)
    out["recovered_sig_max"] = float(info["sig_max"])
    out["recovered_tr0"] = float(info["tr0"])

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
