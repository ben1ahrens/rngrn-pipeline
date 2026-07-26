"""total.py — assemble the composite objective from named, source-tagged terms.

Each term is computed by losses/terms.py (repackaged from the SCAFFOLD objective). This
module wires them into a dict of UNWEIGHTED scalars, hands that to a weighting
strategy, and returns (total, parts). Fail-loud by default (strict=True, the default
everywhere and passed explicitly by recover.py): if the model-derived steady state
does not converge, compute_terms RAISES SteadyStateError rather than scoring the
physics terms against a meaningless x*. Only an explicit strict=False caller opts out
of the raise (e.g. a diagnostic that wants best-effort terms); no default path does.
"""
from __future__ import annotations
import torch

from . import terms as T


class SteadyStateError(RuntimeError):
    """Raised when the model-derived steady state fails to converge (fail-loud)."""


def compute_terms(model, frame, L, observed_idx, kgrid, kstar_obs,
                  latent_fields=None, tau=0.12, jac_floor=1.0, strict=True) -> tuple:
    """Return (terms_dict, parts_dict) of UNWEIGHTED loss terms + diagnostics.

    strict=True (default): raise SteadyStateError if the steady state did not converge,
    rather than scoring physics terms against a meaningless x*."""
    xstar, conv = T.steady_state(model)
    if not conv and strict:
        # fail loud: an unconverged steady state makes every physics term meaningless
        # (the design contract forbids substituting the frame mean). Raise rather than
        # score garbage. recover.py catches this per-restart and moves to the next init.
        raise SteadyStateError(
            "steady-state Newton did not converge; physics terms are undefined for this init")
    xstar = T.steady_state_diff(model, xstar)
    L_k, p_k = T.kstar_anchor(model, xstar, kgrid, kstar_obs, tau=tau)
    L_t, p_t = T.turing_hinges(model, xstar, kgrid)
    L_r, p_r = T.stationarity_residual(model, frame, L, observed_idx, latent_fields)
    L_a, p_a = T.anticollapse(model, xstar, jac_floor=jac_floor)
    term_vals = dict(kstar=L_k, turing=L_t, resid=L_r, anticollapse=L_a)
    parts = dict(ss_converged=conv, **p_k, **p_t, **p_r, **p_a)
    return term_vals, parts


def total_loss(model, frame, L, observed_idx, kgrid, kstar_obs, strategy,
               step=0, latent_fields=None, tau=0.12, jac_floor=1.0, strict=True) -> tuple:
    """Composite loss via a weighting strategy. Returns (scalar loss, parts).

    Raises SteadyStateError (from compute_terms) when strict and x* did not converge."""
    term_vals, parts = compute_terms(
        model, frame, L, observed_idx, kgrid, kstar_obs,
        latent_fields=latent_fields, tau=tau, jac_floor=jac_floor, strict=strict)
    loss, weights_used = strategy.combine(term_vals, step, model=model)
    parts["total"] = float(loss.detach())
    parts["weights_used"] = weights_used
    for k, v in term_vals.items():
        parts[f"L_{k}"] = float(v.detach())
    return loss, parts
