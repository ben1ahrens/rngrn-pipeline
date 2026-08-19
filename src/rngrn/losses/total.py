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
import numpy as np
import torch

from . import terms as T
from .spectral import (is_ignited, is_ignited_batched, spectral_terms,
                       spectral_terms_batched, SPECTRAL_TERM_KEYS)
from .term_registry import LOSS_TERMS
from ..model import BatchedRNGRN


class SteadyStateError(RuntimeError):
    """Raised when the model-derived steady state fails to converge (fail-loud)."""


def _apply_spectral(term_vals: dict, parts: dict, spectral) -> None:
    """Ignite-or-omit the five spectral terms (unit U4). `spectral` is a
    `losses.spectral.SpectralContext` or None (checked by the caller).

    HISTORY-STABILITY (history.py:141-150, `TrainingHistory._names`): the recorder freezes
    its scalar column set on the FIRST recorded step and RAISES if a later step's key set
    differs. Ignition is a training-time EVENT — a run may cross the margin mid-training —
    so, unlike `compute_resid`/`param_prior_kw` (fixed for a whole run), the five "L_<key>"
    entries below must exist in `parts` on EVERY step this function runs, real when
    computed and `float("nan")` when skipped (never faked as 0.0 — CLAUDE.md §4). They are
    written directly into `parts`, not derived from `term_vals`, so a skipped step still
    carries them; `total_loss`'s `for k, v in term_vals.items(): parts[f"L_{k}"] = ...` loop
    overwrites these placeholders with the real values on an ignited-and-solved step, since
    `term_vals` then contains the five keys and this function does not.
    """
    for k in SPECTRAL_TERM_KEYS:
        parts.setdefault(f"L_{k}", float("nan"))
    if not is_ignited(parts, spectral.cfg.ignition_margin):
        # Not ignited: no solve attempted — the forward solve is expensive (3-9 s/solve
        # at 64^2, unrecorded test timing; forward.py module docstring) and nothing gates
        # it but detected Turing instability.
        parts["spectral_skipped"] = "not_ignited"
        parts["spec_ignited"] = 0.0
        return
    parts["spec_ignited"] = 1.0
    u_star, reason = spectral.solver.solve()
    if reason != "ok":
        # Ignited (Turing-unstable beyond margin) but the forward solve did not produce a
        # usable pattern this step (relax/Newton failure, or converged homogeneous). Same
        # "omitted, not zeroed" contract as a not-ignited step.
        parts["spectral_skipped"] = reason
        return
    spec_vals, spec_parts = spectral_terms(u_star, spectral.targets, spectral.cfg)
    term_vals.update(spec_vals)
    parts.update(spec_parts)


def compute_terms(model, frame, L, observed_idx, kgrid, kstar_obs,
                  latent_fields=None, tau=0.12, jac_floor=1.0, strict=True,
                  split_hinges=True, hinge_k_min_frac=0.1, detach_xstar=False,
                  compute_resid=True, param_prior_kw=None, spectral=None,
                  obs_scale=None, kstar_idx=None) -> tuple:
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

    param_prior_kw: kwargs for terms.param_prior (dratio_centre, dratio_spread, box_path),
    or None to OMIT the biological-plausibility prior entirely. None is the default and
    reproduces every number recorded before this term was wired in: the key is absent from
    term_vals, so `strategy.combine`'s sum(weight * term) does not see it at all rather
    than adding a zero-weighted product. recover.py decides, from the strategy's base
    weight for 'param_prior', exactly as it does for the stationarity residual. Until this
    argument existed, loss.weights.param_prior was a NO-OP on the path recover.py runs:
    losses/terms.py::param_prior had no caller outside terms.composite_loss (the standalone
    reference form, which recover.py does not use), so a run configured with the prior on
    trained without it and said nothing.

    spectral: a `losses.spectral.SpectralContext`, or None (default) to OMIT the M1
    spectral terms entirely -- with spectral=None every existing number is bit-identical
    (unit U4, milestone M1). Not None: AFTER the hinge terms produce `parts` (so
    `sig_max_pos` exists), the five terms IGNITE on `losses.spectral.is_ignited` (detected
    patterning, not a config flag) -- see `_apply_spectral` above for the omitted-not-zeroed
    contract and the history-stability reason all five "L_<key>" placeholders are written
    every step this argument is not None.

    obs_scale / kstar_idx: two quantities that are FIXED for a whole recover() call and were
    being recomputed on every step -- `float(frame.mean())`, a device->host sync of a frame
    that never changes, and the k-grid index k*_obs interpolates into, a host->device copy
    and a search. None (the default) computes them here exactly as before, so every external
    caller and test is unaffected; recover.py resolves both once and passes them in.
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
    # ONE autodiff Jacobian for the three dispersion-side terms. They all evaluate J at the
    # SAME x_disp with create_graph=True, so this is the same tensor each of them used to
    # build for itself -- three identical `torch.autograd.functional.jacobian` calls per
    # step, of which two were pure waste.
    J = model.jacobian(x_disp, create_graph=True)
    L_k, p_k = T.kstar_anchor(model, x_disp, kgrid, kstar_obs, tau=tau, J=J, idx=kstar_idx)
    if split_hinges:
        L_t, p_t = T.turing_hinges_split(model, x_disp, kgrid, k_min_frac=hinge_k_min_frac,
                                         J=J)
    else:
        L_t, p_t = T.turing_hinges(model, x_disp, kgrid, J=J)
    L_a, p_a = T.anticollapse(model, x_disp, jac_floor=jac_floor, J=J)
    # FIREWALL: frame.mean() is a statistic of the observed image, nothing else.
    L_s, p_s = T.frame_scale_anchor(
        xstar, float(frame.mean()) if obs_scale is None else obs_scale)
    term_vals = dict(kstar=L_k, turing=L_t, anticollapse=L_a, anchor=L_s)
    parts = dict(ss_converged=conv, **p_k, **p_t, **p_a, **p_s)
    if compute_resid:
        L_r, p_r = T.stationarity_residual(model, frame, L, observed_idx, latent_fields)
        term_vals["resid"] = L_r
        parts.update(p_r)
    else:
        parts["resid_skipped"] = True
    if param_prior_kw is not None:
        L_p, p_p = T.param_prior(model, **param_prior_kw)
        term_vals["param_prior"] = L_p
        parts.update(p_p)
    if spectral is not None:
        _apply_spectral(term_vals, parts, spectral)
    return term_vals, parts


def total_loss(model, frame, L, observed_idx, kgrid, kstar_obs, strategy,
               step=0, latent_fields=None, tau=0.12, jac_floor=1.0, strict=True,
               split_hinges=True, hinge_k_min_frac=0.1, detach_xstar=False,
               compute_resid=True, param_prior_kw=None, spectral=None,
               obs_scale=None, kstar_idx=None) -> tuple:
    """Composite loss via a weighting strategy. Returns (scalar loss, parts).

    obs_scale / kstar_idx: per-call constants the caller may resolve once — see
    `compute_terms`; None reproduces the previous per-step computation exactly.

    Raises SteadyStateError (from compute_terms) when strict and x* did not converge."""
    term_vals, parts = compute_terms(
        model, frame, L, observed_idx, kgrid, kstar_obs,
        latent_fields=latent_fields, tau=tau, jac_floor=jac_floor, strict=strict,
        split_hinges=split_hinges, hinge_k_min_frac=hinge_k_min_frac,
        detach_xstar=detach_xstar, compute_resid=compute_resid,
        param_prior_kw=param_prior_kw, spectral=spectral,
        obs_scale=obs_scale, kstar_idx=kstar_idx)
    loss, weights_used = strategy.combine(term_vals, step, model=model)
    parts["total"] = float(loss.detach())
    parts["weights_used"] = weights_used
    for k, v in term_vals.items():
        parts[f"L_{k}"] = float(v.detach())
    return loss, parts


# ======================================================================================
# unit b2 — BATCHED assembler: B independent members, one forward/backward
# ======================================================================================
def _apply_spectral_batched(term_vals, parts, spectral, xstar_pre, conv, active) -> None:
    """Per-member ignite-or-omit for the five spectral terms — `_apply_spectral` with a
    member axis. `spectral.solver` is a `forward.BatchedPatternSolver`.

    IGNITION IS PER MEMBER. At a given Adam step only some of the B restarts are
    Turing-unstable beyond the margin, so only those are solved for; the ignited subset is
    handed to the solver as a list of GLOBAL member indices and comes back as a stack in the
    same order, with a reason for each member whose solve failed. The decision itself costs
    NO extra device sync — see `losses.spectral.is_ignited_batched`.

    OMITTED-NOT-ZEROED, per member, with one forced difference from the serial path and it
    is a difference of REPRESENTATION, not of arithmetic. Serially a skipped step omits the
    five keys from `term_vals` entirely, so `strategy.combine`'s sum(weight * term) never
    sees them. Batched, every member shares one `term_vals` dict, so the keys cannot be
    omitted for SOME members: a skipped member's entry is an exact CONSTANT 0.0, written by
    `index_copy` into a zero tensor. Its loss contribution is 0 (identical to omission) and
    its gradient contribution is 0 (the constant carries no graph), so a failed member can
    neither shift nor poison another member's optimisation. What must NOT become 0 is the
    RECORD, which is the whole point of the rule — so `parts["L_<key>"]` is NaN for every
    member that was not computed (applied in `total_loss_batched`, which would otherwise
    overwrite it with the placeholder zero), `parts["spectral_skipped"][b]` carries the
    reason, and `parts["spec_computed"][b]` says which members the terms are real for.

    `parts["spectral_skipped"]` is a (B,) object array; an EMPTY string means "not skipped",
    mirroring the serial path's absence of the key on a successful step.
    """
    B = int(xstar_pre.shape[0])
    ignited = is_ignited_batched(parts, spectral.cfg.ignition_margin)
    skipped = np.array(["not_ignited"] * B, dtype=object)
    if active is not None:
        live = np.asarray(
            active.detach().cpu().numpy() if hasattr(active, "detach") else active,
            dtype=bool)
        # An abandoned lane is recorded as such, not merged into "not_ignited": it may well
        # have been Turing-unstable when the caller gave up on it, and the record should say
        # why the solve was skipped rather than imply a stability verdict it did not make.
        skipped[~live] = "abandoned"
        ignited = ignited & live
    parts["spec_ignited"] = ignited.astype(float)
    computed = np.zeros(B, dtype=bool)
    zeros = xstar_pre.new_zeros(B)
    members = np.nonzero(ignited)[0].tolist()
    u_stack, ok_members, reasons = None, [], {}
    if members:
        u_stack, ok_members, reasons = spectral.solver.solve_subset(members, xstar_pre)
    for m, why in reasons.items():
        skipped[m] = why
    if ok_members:
        for m in ok_members:
            skipped[m] = ""
            computed[m] = True
        vals, spec_parts = spectral_terms_batched(u_stack, spectral.targets, spectral.cfg,
                                                  members=ok_members)
        idx = torch.as_tensor(ok_members, dtype=torch.long, device=zeros.device)
        for name, v in vals.items():
            term_vals[name] = zeros.index_copy(0, idx, v)
        for key, v in spec_parts.items():
            full = np.full(B, float("nan"))
            full[ok_members] = v
            parts[key] = full
    else:
        # Nothing solved this step: the five keys still have to EXIST with a stable column
        # set (history.py freezes its columns on the first recorded step), as a constant 0.
        for name in SPECTRAL_TERM_KEYS:
            term_vals[name] = zeros
    parts["spectral_skipped"] = skipped
    parts["spec_computed"] = computed


def compute_terms_batched(model, frame, L, observed_idx, kgrid, kstar_obs,
                          tau=0.12, jac_floor=1.0, split_hinges=True,
                          hinge_k_min_frac=0.1, detach_xstar=False,
                          compute_resid=False, param_prior_kw=None,
                          spectral=None, obs_scale=None, kstar_idx=None,
                          active=None) -> tuple:
    """Batched twin of `compute_terms`. Returns (term_vals, parts, converged).

    `model` is a model.BatchedRNGRN of B members; every term_vals entry is a (B,) tensor and
    every per-member diagnostic in `parts` is a (B,) numpy array under the SAME key the
    serial path uses (see `parts_member`).

    THE FAIL-LOUD CONTRACT IS RELOCATED, NOT WEAKENED. The serial `compute_terms(strict=True)`
    RAISES SteadyStateError when x* does not converge, because scoring physics terms against
    a meaningless x* is the failure mode the contract exists to prevent. Here a raise would
    destroy B-1 healthy members for one bad init, so non-convergence is returned as the third
    element, a (B,) bool mask. The caller MUST act on it: recover's batched loop kills a
    non-converged member permanently and logs it as steady_state_failed, which is exactly
    what a serial restart's SteadyStateError does. A caller that ignores the mask has broken
    the contract; nothing here can fix that for it, so there is no strict= flag pretending to.

    Non-converged members' x* is REPLACED BY ONES before the terms are evaluated. This is not
    a fallback value being scored: their loss is excluded from the optimised sum. It exists
    so a diverged member cannot inject NaN/inf into the shared graph and poison the other
    members' gradients. `parts['ss_converged']` records which members are real.

    `compute_resid` must be False. The stationarity residual needs per-pixel states, which
    the batched reaction does not broadcast to (model.BatchedRNGRN.reaction), so there is no
    batched residual to compute. Its default weight is 0 (exp06, settled off), so this costs
    the default path nothing -- but it is refused loudly rather than silently skipped.

    `spectral`: a `losses.spectral.SpectralContext` whose `solver` is a
    `forward.BatchedPatternSolver`, or None (default) to OMIT the M1 spectral terms. THIS
    USED TO BE UNCONDITIONALLY REFUSED; the refusal was deleted, not weakened: it said the
    forward solve had no batched form, which was true when it was written and is no longer.
    The solver now owns per-MEMBER warm state keyed by global member index, and the forward
    solve gets its own member axis (`forward.batched_reaction_fields` broadcasts the batched
    parameters to per-pixel fields, which `model.BatchedRNGRN.reaction` deliberately does
    not). See `_apply_spectral_batched` for the per-member ignite-or-omit contract.

    What the deleted refusal used to provide IMPLICITLY -- because it always fired before
    `spectral` (or `model`) was ever touched -- is restored EXPLICITLY here instead, at the
    entry point, before any steady-state solve or Jacobian: `spectral.solver` must expose
    `solve_subset` (the `forward.BatchedPatternSolver` contract), not `solve` (the serial
    `forward.PatternSolver` one). Handing the serial solver to this assembler is now a
    plausible mistake -- both combinations are legal -- and used to surface only as an
    `AttributeError` mid-step, inside `_apply_spectral_batched`, after the batch had already
    paid for a steady-state solve and three Jacobians (REVIEW_gpu_optim_delta.md C1/§8).

    `active`: an optional (B,) bool mask of members still being optimised, used ONLY to skip
    the forward solve for members `recover` has already abandoned. It cannot change any
    computed value — a dead member's loss is masked out of the optimised sum by the caller
    either way — it only stops a dead lane paying seconds per step for a solve nobody reads.
    None (the default) treats every member as active, which is what every non-spectral caller
    gets, unchanged.

    `obs_scale` / `kstar_idx`: the same per-call constants the serial `compute_terms` takes,
    with the same None-means-compute-it-here default.
    """
    if not isinstance(model, BatchedRNGRN):
        raise ValueError(
            f"compute_terms_batched requires a model.BatchedRNGRN (B independent members); "
            f"got {type(model).__name__}. Use losses.total.compute_terms for a single "
            "model.RNGRN.")
    if compute_resid:
        # message single-sourced from losses/term_registry.py (Task 8, R2 redesign) so the
        # registry's `resid` refusal_reason and this raise can never drift apart. `resid`
        # is the ONLY surviving single-sourced refusal here: the spectral one is retired
        # with the refusal itself (INTEGRATION_r3_collisions.md rows 8/28, §2.3).
        raise ValueError(LOSS_TERMS.get("resid").refusal_reason)
    if spectral is not None and not hasattr(spectral.solver, "solve_subset"):
        raise ValueError(
            f"compute_terms_batched requires spectral.solver to expose solve_subset(members, "
            f"xstar_batch) -> (u_stack, ok_members, reasons); got "
            f"{type(spectral.solver).__name__}, which looks like the SERIAL "
            "forward.PatternSolver (.solve() only, no member axis). Use "
            "forward.BatchedPatternSolver for the batched path -- see "
            "losses.spectral.SpectralContext's docstring for the two duck-typed contracts.")
    xstar, conv = T.steady_state_batched(model)
    # ones for the failed members: a poison guard for the SHARED graph, not a scored value.
    x_ok = torch.where(conv.unsqueeze(-1), xstar, torch.ones_like(xstar))
    xstar, polish_ok = T.steady_state_diff_batched(model, x_ok)
    # a singular J at the polish step is the same failure the serial path raises on
    conv = conv & polish_ok
    x_disp = xstar.detach() if detach_xstar else xstar
    # ONE batched Jacobian for the three dispersion-side terms -- see `compute_terms`.
    J = model.jacobian(x_disp, create_graph=True)
    L_k, p_k = T.kstar_anchor_batched(model, x_disp, kgrid, kstar_obs, tau=tau, J=J,
                                      idx=kstar_idx)
    if split_hinges:
        L_t, p_t = T.turing_hinges_split_batched(model, x_disp, kgrid,
                                                 k_min_frac=hinge_k_min_frac, J=J)
    else:
        L_t, p_t = T.turing_hinges_batched(model, x_disp, kgrid, J=J)
    L_a, p_a = T.anticollapse_batched(model, x_disp, jac_floor=jac_floor, J=J)
    # FIREWALL: frame.mean() is a statistic of the observed image, nothing else.
    L_s, p_s = T.frame_scale_anchor_batched(
        xstar, float(frame.mean()) if obs_scale is None else obs_scale)
    term_vals = dict(kstar=L_k, turing=L_t, anticollapse=L_a, anchor=L_s)
    parts = dict(ss_converged=conv.detach().cpu().numpy(), resid_skipped=True,
                 **p_k, **p_t, **p_a, **p_s)
    if param_prior_kw is not None:
        L_p, p_p = T.param_prior_batched(model, **param_prior_kw)
        term_vals["param_prior"] = L_p
        parts.update(p_p)
    if spectral is not None:
        _apply_spectral_batched(term_vals, parts, spectral, x_ok.detach(), conv, active)
    return term_vals, parts, conv


def total_loss_batched(model, frame, L, observed_idx, kgrid, kstar_obs, strategy,
                       step=0, tau=0.12, jac_floor=1.0, split_hinges=True,
                       hinge_k_min_frac=0.1, detach_xstar=False,
                       compute_resid=False, param_prior_kw=None, spectral=None,
                       obs_scale=None, kstar_idx=None, active=None) -> tuple:
    """Batched twin of `total_loss`. Returns (loss_vec (B,), parts, converged (B,)).

    The weighting strategy is applied UNCHANGED: `combine` only ever does
    sum(weight * term), which is shape-agnostic, so a (B,) term vector yields a (B,) total
    with exactly the weights the serial path would have used. Strategies that read a term's
    VALUE (RatioWeighting calls float() on it) cannot work batched and will raise on the
    vector -- recover's batched path rejects non-static strategies up front so the failure
    is a clear message rather than a torch cast error.

    The returned loss is a VECTOR, deliberately not pre-summed: only the caller knows which
    members are still alive, and summing a dead member's loss in would give it gradient.

    `active`: passed straight to `compute_terms_batched` — see there; it only skips the
    forward solve for members the caller has already abandoned.
    """
    term_vals, parts, conv = compute_terms_batched(
        model, frame, L, observed_idx, kgrid, kstar_obs, tau=tau, jac_floor=jac_floor,
        split_hinges=split_hinges, hinge_k_min_frac=hinge_k_min_frac,
        detach_xstar=detach_xstar, compute_resid=compute_resid,
        param_prior_kw=param_prior_kw, spectral=spectral,
        obs_scale=obs_scale, kstar_idx=kstar_idx, active=active)
    loss, weights_used = strategy.combine(term_vals, step, model=model)
    parts["total"] = loss.detach().cpu().numpy()
    parts["weights_used"] = weights_used
    computed = parts.get("spec_computed")
    for k, v in term_vals.items():
        arr = v.detach().cpu().numpy()
        if computed is not None and k in SPECTRAL_TERM_KEYS:
            # A spectral term's placeholder is an exact 0 so the loss sees an omission
            # (`_apply_spectral_batched`); the RECORD must not inherit that zero, or a
            # reader could not tell "not computed" from "zero loss". NaN, per CLAUDE.md §4.
            arr = np.where(computed, arr, float("nan"))
        parts[f"L_{k}"] = arr
    return loss, parts, conv


def parts_member(parts: dict, b: int) -> dict:
    """One member's slice of a batched `parts` dict, in the SHAPE the serial path returns.

    Per-member entries are (B,) arrays; shared entries (obs_scale, hinge_i_min, kstar_obs,
    weights_used, resid_skipped) are scalars and pass through. This is what lets recover.py
    hand a batched restart's diagnostics to the same consumers -- run-index rows are flat
    scalars, so `bool`/`float`/`int` conversion happens here, once.

    `spectral_skipped` is the one per-member entry that is NOT numeric: it is a (B,) object
    array of reason STRINGS (empty for a member that was not skipped), and it is passed
    through as a str so a row records the same value the serial path's `parts` carries.
    """
    out = {}
    for k, v in parts.items():
        if hasattr(v, "shape") and getattr(v, "ndim", 0) == 1:
            x = v[b]
            # `kind` exists on a numpy dtype and not on a torch one, so the getattr chain
            # both selects the string case and leaves any tensor entry on the old path.
            if getattr(getattr(v, "dtype", None), "kind", None) in ("O", "U", "S"):
                out[k] = str(x)
            elif x.dtype == bool:
                out[k] = bool(x)
            else:
                out[k] = float(x)
        else:
            out[k] = v
    return out
