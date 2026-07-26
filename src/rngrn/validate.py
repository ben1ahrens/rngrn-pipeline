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


def _sign_structure(J):
    J = np.asarray(J, float)
    return np.sign(np.where(np.abs(J) > 1e-9 * (np.abs(J).max() + 1e-12), J, 0.0))


def score_recovery(result, answer_key) -> dict:
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

    # 3. sign structure vs answer key (only when the true J is 2x2-comparable in size)
    ak_J = getattr(answer_key, "J", None)
    if ak_J is not None and np.asarray(ak_J).shape == J_rec.shape:
        s_rec, s_true = _sign_structure(J_rec), _sign_structure(ak_J)
        out["sign_match_frac"] = float(np.mean(s_rec == s_true))
    else:
        out["sign_match_frac"] = float("nan")

    return out
