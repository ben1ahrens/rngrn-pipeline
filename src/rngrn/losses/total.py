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
                  latent_fields=None, tau=0.12, jac_floor=1.0, strict=True,
                  split_hinges=True, hinge_k_min_frac=0.1, detach_xstar=False,
                  compute_resid=True) -> tuple:
    """Return (terms_dict, parts_dict) of UNWEIGHTED loss terms + diagnostics.

    strict=True (default): raise SteadyStateError if the steady state did not converge,
    rather than scoring physics terms against a meaningless x*.

    split_hinges=True (default): use terms.turing_hinges_split, whose two hinges have
    disjoint k-support. False selects the superseded shared-support terms.turing_hinges,
    kept so it can be run as a control arm.

    detach_xstar: whether the DISPERSION-side terms (kstar, turing, anticollapse) see x*
    as a constant. False (default) matches the library as it stands — gradients flow
    through the steady state via steady_state_diff. True matches
    scripts/exp05_pixel_minibatch.py::fit, which passes `xs.detach()` to those terms while
    letting the scale anchor differentiate through x*. This is a REAL behavioural
    difference between the library and the experiment that produced the 36.8 % measurement,
    and it has not been isolated in an A/B; the default deliberately does not change
    library behaviour. The frame-scale anchor always sees the differentiable x*, under
    both settings, because that is the only path by which it can move theta.

    compute_resid=False: OMIT the stationarity residual from term_vals entirely rather
    than compute it and multiply by zero. Measured at 96x96 / N=3: a forward+backward step
    costs 9.39 ms with the residual computed and 5.15 ms without, i.e. 45 % of step time on
    a term whose default weight is now 0. The key is omitted, never faked — parts records
    `resid_skipped=True` so no downstream reader can mistake "not computed" for "zero".
    Callers must only pass False when the residual's weight is genuinely 0 (recover.py
    checks the strategy's base weight AND that the strategy's weights are static).
    """
    xstar, conv = T.steady_state(model)
    if not conv and strict:
        # fail loud: an unconverged steady state makes every physics term meaningless
        # (the design contract forbids substituting the frame mean). Raise rather than
        # score garbage. recover.py catches this per-restart and moves to the next init.
        raise SteadyStateError(
            "steady-state Newton did not converge; physics terms are undefined for this init")
    xstar = T.steady_state_diff(model, xstar)
    x_disp = xstar.detach() if detach_xstar else xstar
    L_k, p_k = T.kstar_anchor(model, x_disp, kgrid, kstar_obs, tau=tau)
    if split_hinges:
        L_t, p_t = T.turing_hinges_split(model, x_disp, kgrid, k_min_frac=hinge_k_min_frac)
    else:
        L_t, p_t = T.turing_hinges(model, x_disp, kgrid)
    L_a, p_a = T.anticollapse(model, x_disp, jac_floor=jac_floor)
    # FIREWALL: frame.mean() is a statistic of the observed image, nothing else.
    L_s, p_s = T.frame_scale_anchor(xstar, float(frame.mean()))
    term_vals = dict(kstar=L_k, turing=L_t, anticollapse=L_a, anchor=L_s)
    parts = dict(ss_converged=conv, **p_k, **p_t, **p_a, **p_s)
    if compute_resid:
        L_r, p_r = T.stationarity_residual(model, frame, L, observed_idx, latent_fields)
        term_vals["resid"] = L_r
        parts.update(p_r)
    else:
        parts["resid_skipped"] = True
    return term_vals, parts


def total_loss(model, frame, L, observed_idx, kgrid, kstar_obs, strategy,
               step=0, latent_fields=None, tau=0.12, jac_floor=1.0, strict=True,
               split_hinges=True, hinge_k_min_frac=0.1, detach_xstar=False,
               compute_resid=True) -> tuple:
    """Composite loss via a weighting strategy. Returns (scalar loss, parts).

    Raises SteadyStateError (from compute_terms) when strict and x* did not converge."""
    term_vals, parts = compute_terms(
        model, frame, L, observed_idx, kgrid, kstar_obs,
        latent_fields=latent_fields, tau=tau, jac_floor=jac_floor, strict=strict,
        split_hinges=split_hinges, hinge_k_min_frac=hinge_k_min_frac,
        detach_xstar=detach_xstar, compute_resid=compute_resid)
    loss, weights_used = strategy.combine(term_vals, step, model=model)
    parts["total"] = float(loss.detach())
    parts["weights_used"] = weights_used
    for k, v in term_vals.items():
        parts[f"L_{k}"] = float(v.detach())
    return loss, parts
