"""r2_ignition_run.py — the R2 Phase-I population run (Task 16, docs/REDESIGN_rngrn.md §4.5).

WHAT IT DOES. Trains a BATCHED population of B members on one target frame with the
IGNITION TERMS ONLY, and runs the matched A0 control arm (the legacy objective, the same
member seeds, the same budget) beside it. Sweeps B so the throughput-and-RSS curve that
sets B and K for R3/R4 is a measurement rather than a guess (§4.5: "B = 512 is a starting
point, UNCALIBRATED").

TWO ARMS (plus two attribution ablations, off by default), ONE LOOP.

  * `r2`  — pinned (§3.2) + boxed (§3.3) `BatchedRNGRN`, objective
            `kstar_si` + `turing_hinges_split` + `param_prior` + `beta_hinge`.
            **No steady-state solve anywhere in the loop**: pinning MAKES x* the datum, so
            `terms.steady_state_batched` never runs and the per-step Newton fragility that
            §3.2 removes is genuinely gone rather than merely down-weighted.
  * `a0`  — the legacy objective, unchanged, through `losses.total.total_loss_batched` with
            D5's own weight dict. Its model is neither pinned nor boxed, so it does solve
            for x* every step. This is the CONTROL every `r2` number is read against
            (CLAUDE.md §8: an arm is read against its control, never against zero).

`--arms r2_nobox r2_noprior` adds the attribution ablations described in `run_arm`. They
exist because the first sweep measured 0 ignition in `r2` against a non-zero rate in `a0`,
and an arm-level difference with three simultaneous causes attributes nothing.

Both arms take their per-member init seeds from `recover._restart_seed(model_seed, r)` —
the one seed policy in the repo (unit B1) — so member r of the r2 arm and member r of the
a0 arm start from the same draw, and the B rungs are NESTED (the first 64 seeds of B=512
are the B=64 run's seeds).

FIREWALL. The training loop is answer-key-free: the only inputs are the observed frame, L,
and `kstar_obs` from the frame's own FFT. The answer key is touched in exactly one place,
`_score_against_answer_key`, AFTER training is finished, and only to produce the
D5-comparable `kstar_fft_rel_err` column (which is defined against `AnswerKey.kstar_fft` —
`validate.py`'s headline). That makes this module SCORING-SIDE, and it opens `payload.h5`,
so its name is on `tests/test_firewall.py::FORBIDDEN` (CLAUDE.md §5) — `scripts/` is
outside the completeness test's glob and is importable by bare name because the suite and
the notebooks put `scripts/` on `sys.path`.

EVIDENCE. Every number this writes is labelled with its run directory under
`--out`. Nothing here is a plumbing check: the defaults are the run
(`docs/PLAN_redesign.md` Task 16; `notebooks/redesign_pipeline.ipynb` launches it with
`--out experiments/redesign_r2/phase1` and nothing else).

Usage — ALWAYS under the memory guard (CLAUDE.md §7a):
    bash scripts/guarded_run.sh .venv/bin/python -u scripts/r2_ignition_run.py \
        --out experiments/redesign_r2/phase1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rngrn import observables as obs                            # noqa: E402
from rngrn import recover as R                                  # noqa: E402
from rngrn.data.gate import from_registry                       # noqa: E402
from rngrn.eval.analysis import turing_ok                       # noqa: E402
from rngrn.history import TrainingHistory                       # noqa: E402
from rngrn.losses import terms as T                             # noqa: E402
from rngrn.losses import total as LT                            # noqa: E402
from rngrn.losses.weighting import DataFirstStaging, FixedWeighting   # noqa: E402
from rngrn.model import BatchedRNGRN                            # noqa: E402
from rngrn.scoring.reproducibility import (                     # noqa: E402
    modal_sign_structure, pairwise_sign_agreement)
from rngrn.utils import provenance                              # noqa: E402

# ---------------------------------------------------------------------------------------
# Defaults. Every one of these is a decision; the UNCALIBRATED ones say so, here and in
# docs/DECISIONS.md (D-R2-1), per CLAUDE.md §10.
# ---------------------------------------------------------------------------------------
DATASET_ID = "turing_labyrinth"      # REDESIGN §1: the single R2/R4 target
SAMPLE_KEY = "sample_0000"
N_SPECIES = 3
FORM = "competitive"

# The ignition weights (REDESIGN §4.4). ALL FOUR ARE UNCALIBRATED — no sweep exists.
#   kstar_si    1.0  inherits `kstar`'s shipped weight verbatim; §4.4 defines kstar_si as
#                    the drop-in replacement for kstar in the redesign arm, so changing the
#                    magnitude at the same time would confound the swap with a re-weighting.
#   turing      1.0  unchanged from the shipped default and from D5's frozen config.
#   param_prior 1.0  §3.3/§4.4 "promoted to nonzero weight" from its shipped 0.0. No
#                    calibrated value exists; 1.0 is the neutral promotion, not a tuned one.
#   beta_hinge  1.0  born with this task (§3.2's beta>=0 constraint). No prior art at all.
R2_WEIGHTS = dict(kstar_si=1.0, turing=1.0, param_prior=1.0, beta_hinge=1.0)

# The A0 control's weights are D5's own, verbatim from
# experiments/diag_fft/d5/target_reports.jsonl's frozen_config — NOT re-derived, so the
# control really is the objective the baseline row was measured under. `morphology` is
# INERT (never in compute_terms/compute_terms_batched); it is carried at D5's 0.1 rather
# than the registry's 0.0 precisely so this dict is a quotation, not an edit.
A0_WEIGHTS = dict(kstar=1.0, turing=1.0, resid=0.0, anticollapse=0.5, anchor=2.0,
                  morphology=0.1, param_prior=0.0)

# configs/bio_box.yaml, §3.3's hard box. Loaded from the file at run time; these are the
# keys the box-sigmoid reparameterization accepts.
BOX_KEYS = ("alpha", "delta")

HINGE_MARGIN = 1e-3          # losses/terms.turing_hinges_split's own default
SIGN_ZERO_RTOL = 0.05        # D5's `reproducibility_sign_zero_rtol` — matched for comparability


# =======================================================================================
# helpers
# =======================================================================================
def _status_mb(key: str) -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return float(line.split()[1]) / 1024.0
    raise RuntimeError(f"/proc/self/status has no {key} line")


def _peak_rss_mb() -> float:
    """VmHWM — this process's PEAK resident set size, in MiB. The §7a guard's currency.

    CAVEAT, stated because it changes how the curve reads: VmHWM is a HIGH-WATER MARK for
    the whole PROCESS and the kernel never lowers it, so across a sweep run in one process
    it is MONOTONE — cell n's value includes every earlier cell's peak. The rungs are
    executed in ASCENDING B, so the increments are attributable to the larger B, but the
    absolute number is not "what this cell alone needed". `rss_mb_end` (VmRSS after the
    cell) and `cuda_max_alloc_mb` (reset per cell) are the per-cell figures."""
    return _status_mb("VmHWM")


def _mem_available_mb() -> float:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    raise RuntimeError("/proc/meminfo has no MemAvailable line")


def _load_box(path: str) -> dict:
    """configs/bio_box.yaml -> {'alpha': (lo, hi), 'delta': (lo, hi)}, dropping UNCITED rows.

    Reuses `losses.terms._load_box_bounds` so the box the MODEL is constructed with and the
    box `param_prior` hinges against are read by the same parser, from the same file."""
    bounds = T._load_box_bounds(path)
    boxes = {k: bounds[k] for k in BOX_KEYS if bounds.get(k) is not None}
    missing = [k for k in BOX_KEYS if k not in boxes]
    if missing:
        raise RuntimeError(
            f"{path}: no cited (low, high) for {missing}; REDESIGN §3.3 requires a hard box "
            "for alpha and delta. An UNCITED row cannot be turned into a box here.")
    return boxes


def _ignited(sig0: np.ndarray, sig_max_pos: np.ndarray) -> np.ndarray:
    """The per-member ignition flag used for EVENTS during training.

    The Turing conditions read off the quantities `turing_hinges_split_batched` already
    computes on the TRAINING k-grid: uniform mode stable and some k > k_min unstable. It is
    a proxy for `eval.analysis.turing_ok`, which is what the final verdict uses — the grids
    differ (recover._kgrid_for is anchored to k*_obs; turing_ok's is a fixed
    linspace(1e-3, 50, 4000)), so the two can disagree near the boundary. Only the final
    `turing_ok` verdict is reported as a Turing fraction.
    """
    return (sig0 < 0.0) & (sig_max_pos > HINGE_MARGIN)


def _invariant_block(bmodel, xstar, kstar_obs) -> tuple:
    """The §3.4 invariant set for every member: (sorted names, (B, Q) array).

    REDESIGN §3.4 names the reported uncertainty coordinates as "the SPEC §9.4 combinations
    d_ratio, α/δ, β/δ (β now derived, §3.2), and D·k*²/δ, plus per-edge promoter occupancy at
    the operating point KA_ij·x*ⱼⁿ/(1 + …) ∈ [0,1]". These are the TIME-GAUGE-INVARIANT
    coordinates: §3.2 fixes the time gauge only by convention (geomean(δ) ≡ 1), so a raw rate
    is not comparable across members while every ratio here is.

    The occupancy denominator is taken from `model._reaction_prod`'s own competitive branch
    (1 + Σⱼ (KA+KR)·xⱼⁿ) rather than re-derived, so the two cannot drift. `nc1` saturates per
    site instead and is refused rather than silently given the competitive formula.
    """
    if bmodel.form != "competitive":
        raise NotImplementedError(
            f"the §3.4 occupancy invariant is written for form='competitive'; got "
            f"{bmodel.form!r}, whose production term saturates per site (KAx/(1+KAx)). Add "
            "its own branch rather than reporting the competitive formula for it.")
    with torch.no_grad():
        N = bmodel.N
        KA, KR = bmodel.KA, bmodel.KR                                   # (B,N,N)
        alpha, delta, beta, D = bmodel.alpha, bmodel.delta, bmodel.beta, bmodel.D
        xn = torch.clamp(xstar, min=0.0) ** bmodel.n_hill               # (B,N)
        denom = 1.0 + ((KA + KR) * xn.unsqueeze(-2)).sum(dim=-1)        # (B,N)
        occ = (KA * xn.unsqueeze(-2)) / denom.unsqueeze(-1)             # (B,N,N)
        sorted_D, _ = torch.sort(D, dim=-1)
        cols = {"d_ratio": sorted_D[..., -1] / sorted_D[..., -2]}
        for i in range(N):
            cols[f"beta_over_delta[{i}]"] = beta[:, i] / delta[:, i]
            cols[f"Dk2_over_delta[{i}]"] = D[:, i] * (kstar_obs ** 2) / delta[:, i]
            for j in range(N):
                cols[f"alpha_over_delta[{i},{j}]"] = alpha[:, i, j] / delta[:, i]
                cols[f"occupancy[{i},{j}]"] = occ[:, i, j]
        names = sorted(cols)
        arr = torch.stack([cols[k] for k in names], dim=-1).detach().cpu().numpy()
    return names, arr


def _sign_structures(J: np.ndarray) -> list:
    """Hashable J sign structures, member by member, at D5's own `sign_zero_rtol`.

    Uses `scoring.reproducibility.sign_structure` — the SAME function that produced D5's
    `n_distinct_structures` = 10 — rather than `recover._topology`'s KA/KR-gate signs, which
    are a different quantity and would not be comparable.
    """
    from rngrn.scoring.reproducibility import sign_structure
    return [tuple(sign_structure(J[b], SIGN_ZERO_RTOL).ravel().astype(int).tolist())
            for b in range(J.shape[0])]


# =======================================================================================
# the training loop (one arm, one B)
# =======================================================================================
def run_arm(arm, B, *, frame_t, frame_np, L, kstar_obs, kgrid, box, seeds, steps, lr,
            grad_clip, device, backend, history_every, staging_off, staging_ramp,
            x_pin, verbose_every):
    """Train one (arm, B) population. Returns (bmodel, xstar, record, history).

    ARMS. `r2` and `a0` are the two the task defines. The `r2_no*` arms are ATTRIBUTION
    ablations added after the first sweep measured 0 ignition in `r2` against a non-zero
    rate in `a0`: each removes exactly ONE of `r2`'s departures from the legacy
    parameterization, so the effect can be attributed to a named constraint rather than left
    as an unexplained arm-level difference. They are not in the default `--arms`.
      r2_nobox   — pinned, NOT boxed (softplus alpha/delta), prior on.
      r2_noprior — pinned and boxed, param_prior weight 0.
    There is deliberately NO `r2_nopin` arm: removing the pin reinstates the per-step Newton
    solve AND deletes `beta_hinge` (which only a pinned model has), so it would be a
    different objective rather than the R2 arm minus one thing, and would attribute nothing.
    """
    if arm in ("r2", "r2_nobox", "r2_noprior"):
        bmodel = BatchedRNGRN.from_seeds(
            N=N_SPECIES, seeds=seeds, form=FORM, dispersion_backend=backend,
            init="default", pin_xstar=x_pin,
            param_boxes=(None if arm == "r2_nobox" else box)).to(device)
        base = dict(R2_WEIGHTS)
        if arm == "r2_noprior":
            base["param_prior"] = 0.0
    elif arm == "a0":
        bmodel = BatchedRNGRN.from_seeds(
            N=N_SPECIES, seeds=seeds, form=FORM, dispersion_backend=backend,
            init="default").to(device)
        base = dict(A0_WEIGHTS)
    else:
        raise ValueError(
            f"unknown arm {arm!r}; expected one of r2, a0, r2_nobox, r2_noprior")

    # The SAME staging both arms and D5 used: `turing` held at 0 for the first 25 % of the
    # budget, ramped over the next 25 % (recover()'s defaults, D5's frozen config).
    strategy = DataFirstStaging(FixedWeighting(base), total_steps=steps,
                                keys=("turing",), off_frac=staging_off, ramp_frac=staging_ramp)

    history = (TrainingHistory(every=history_every, total_steps=steps, n_members=B,
                               N=N_SPECIES) if history_every > 0 else None)

    params = list(bmodel.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    alive = torch.ones(B, dtype=torch.bool, device=device)
    was_ignited = np.zeros(B, dtype=bool)
    obs_scale = float(frame_t.mean())

    # x* is a CONSTANT for the pinned arm — this is the whole point of §3.2. It is built
    # once, outside the loop, and no Newton solve ever runs.
    x_pin_t = (torch.tensor(x_pin, dtype=torch.float64, device=device).expand(B, N_SPECIES)
               if arm != "a0" else None)
    prior_kw = dict(dratio_centre=7.5, dratio_spread=1.0, box=box)

    t0 = time.time()
    for step in range(steps + 1):          # +1: a final no-grad evaluation pass
        final = step == steps
        opt.zero_grad(set_to_none=True)
        if arm != "a0":
            loss_vec, parts, conv = _r2_terms(bmodel, x_pin_t, kgrid, kstar_obs,
                                              strategy, step, prior_kw)
        else:
            loss_vec, parts, conv = LT.total_loss_batched(
                bmodel, frame_t, L, list(range(N_SPECIES)), kgrid, kstar_obs, strategy,
                step=step, tau=0.12, jac_floor=1.0, split_hinges=True,
                hinge_k_min_frac=0.1, detach_xstar=False, compute_resid=False,
                param_prior_kw=None)

        newly_dead = alive & ~conv
        if bool(newly_dead.any()):
            for b in newly_dead.nonzero().flatten().tolist():
                if history is not None:
                    history.record_death(b, step)
            alive = alive & conv

        # ---- population events (§3.4/§4.5, history.EVENT_KINDS) ------------------------
        live_np = alive.detach().cpu().numpy()
        ign = _ignited(parts["sig0"], parts["sig_max_pos"]) & live_np
        if history is not None:
            for b in np.nonzero(ign & ~was_ignited)[0]:
                history.record_event(step, int(b), "ignition")
            for b in np.nonzero(~ign & was_ignited & live_np)[0]:
                history.record_event(step, int(b), "deignition")
        was_ignited = ign

        if history is not None and (history.should_record(step) or final):
            history.record_batched(step, parts, bmodel, alive=alive)
            # §3.4's invariant coordinates, on the same cadence. Recorded for LIVE members
            # only, so a dead lane stays NaN rather than carrying frozen values — the same
            # rule `record_batched(alive=...)` applies to the parameter trace.
            #
            # PINNED ARMS ONLY, and this is a real gap rather than a design choice: the
            # occupancy invariant is evaluated AT x*, which a pinned arm has for free but
            # which `losses.total.total_loss_batched` does not return for `a0`. Recovering
            # it would mean re-running `steady_state_batched` per recorded step, i.e.
            # multiplying the a0 arm's already-pathological solve cost. a0 therefore has a
            # parameter trace but no invariant trace; see docs/HANDOFF_redesign_r2.md.
            if x_pin_t is not None:
                inv_names, inv = _invariant_block(bmodel, x_pin_t, kstar_obs)
                for b in np.nonzero(live_np)[0]:
                    history.record_invariants(step, int(b),
                                              dict(zip(inv_names, inv[b].tolist())))

        if not bool(alive.any()):
            break
        if final:
            break
        total = torch.where(alive, loss_vec, torch.zeros_like(loss_vec)).sum()
        total.backward()
        R._clip_grad_norm_per_member(params, grad_clip, B)
        opt.step()
        if verbose_every and step % verbose_every == 0:
            print(f"  [{arm} B={B}] step {step}: alive={int(alive.sum())} "
                  f"ignited={int(ign.sum())} "
                  f"mean_total={float(loss_vec[alive].mean().detach()):.4f}",
                  flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0

    record = dict(
        arm=arm, B=B, steps=steps, wall_s=wall,
        s_per_step=wall / max(1, steps),
        member_steps_per_s=(B * steps) / wall,
        peak_rss_mb=_peak_rss_mb(),          # MONOTONE across the sweep — see _peak_rss_mb
        rss_mb_end=_status_mb("VmRSS"),      # resident set at the END of this cell
        mem_available_mb_after=_mem_available_mb(),
        cuda_max_alloc_mb=(torch.cuda.max_memory_allocated() / 2 ** 20
                           if device.type == "cuda" else float("nan")),
        n_alive=int(alive.sum()),
    )
    xstar = x_pin_t if arm != "a0" else None      # every r2-family arm is pinned
    return bmodel, xstar, record, history, alive


def _r2_terms(bmodel, xstar, kgrid, kstar_obs, strategy, step, prior_kw):
    """The R2 ignition objective. Returns (loss_vec (B,), parts, converged (B,)).

    Shaped exactly like `losses.total.total_loss_batched`'s return so the same caller code,
    the same `TrainingHistory.record_batched`, and the same death/alive bookkeeping serve
    both arms.

    `converged` is "this member's loss is finite". There is no steady-state solve to fail —
    that is what pinning buys — so the only way a member can die here is by producing a
    non-finite loss, which would poison the shared graph exactly as a diverged x* would.

    MEASURED INEFFICIENCY, deliberately not fixed here. `turing_hinges_split_batched` and
    `kstar_anchor_si_batched` each call `model.dispersion(...)`, which each rebuild the
    per-member Jacobian — so the dispersion is computed TWICE per step. Neither function
    accepts a precomputed J (`kstar_anchor_si_batched`'s signature has no `J` argument), so
    sharing it would mean editing a reviewed loss term, which this task does not do. The
    throughput numbers below are therefore the cost of the objective AS REVIEWED, and the
    obvious ~2x is an R3 optimisation, reported not taken.
    """
    L_t, p_t = T.turing_hinges_split_batched(bmodel, xstar, kgrid, margin=HINGE_MARGIN,
                                             k_min_frac=0.1)
    L_k, p_k = T.kstar_anchor_si_batched(bmodel, xstar, kgrid, kstar_obs)
    L_p, p_p = T.param_prior_batched(bmodel, **prior_kw)
    L_b = bmodel.beta_hinge()

    term_vals = dict(kstar_si=L_k, turing=L_t, param_prior=L_p, beta_hinge=L_b)
    loss, weights_used = strategy.combine(term_vals, step, model=bmodel)
    conv = torch.isfinite(loss)
    parts = dict(ss_converged=conv.detach().cpu().numpy(),
                 beta_min=T._np(bmodel.beta.min(dim=-1).values),
                 **p_k, **p_t, **p_p)
    parts["total"] = loss.detach().cpu().numpy()
    parts["weights_used"] = weights_used
    for k, v in term_vals.items():
        parts[f"L_{k}"] = v.detach().cpu().numpy()
    return loss, parts, conv


# =======================================================================================
# scoring (post-training, D5-comparable columns)
# =======================================================================================
def score_population(bmodel, xstar, alive, kgrid, kstar_obs, arm):
    """Per-member final outcomes + the D5-comparable population columns.

    FIREWALL-CLEAN: reads only the recovered models and `kstar_obs` (the observed frame's
    own FFT). The answer key enters separately, in `_score_against_answer_key`.
    """
    with torch.no_grad():
        if arm != "a0":                    # every r2-family arm is pinned
            xs = xstar
        else:
            xs, conv = T.steady_state_batched(bmodel)
            alive = alive & conv
        J = bmodel.jacobian(xs, create_graph=False).detach().cpu().numpy()
        D = bmodel.D.detach().cpu().numpy()
        sig = bmodel.dispersion(xs, kgrid, J=torch.as_tensor(J, device=bmodel.device))
        kstar_model = kgrid[torch.argmax(sig, dim=1)].detach().cpu().numpy()
    live = alive.detach().cpu().numpy()

    verdicts, infos = [], []
    for b in range(bmodel.B):
        if not live[b]:
            verdicts.append(False); infos.append(None); continue
        ok, info = turing_ok(J[b], D[b])
        verdicts.append(bool(ok)); infos.append(info)
    verdicts = np.asarray(verdicts, dtype=bool)

    live_idx = np.nonzero(live)[0]
    ign_idx = np.nonzero(verdicts)[0]
    signs = _sign_structures(J)

    def _pop(idx):
        if len(idx) == 0:
            return dict(n=0, n_distinct_structures=0, topology_consistency=float("nan"),
                        mean_agreement=float("nan"), modal_fraction=float("nan"))
        Js = [J[b] for b in idx]
        modal = modal_sign_structure(Js, SIGN_ZERO_RTOL)
        agree = pairwise_sign_agreement(Js, SIGN_ZERO_RTOL)
        return dict(n=int(len(idx)),
                    n_distinct_structures=int(modal["n_distinct_structures"]),
                    topology_consistency=float(modal["modal_fraction"]),
                    modal_fraction=float(modal["modal_fraction"]),
                    mean_agreement=float(agree["mean_agreement"]))

    kstar_live = kstar_model[live_idx] if len(live_idx) else np.array([])
    rel_obs = (np.abs(kstar_live - kstar_obs) / kstar_obs) if len(kstar_live) else np.array([])

    return dict(
        n_members=int(bmodel.B),
        n_alive=int(live.sum()),
        n_turing=int(verdicts.sum()),
        turing_frac=float(verdicts.sum()) / float(bmodel.B),
        turing_frac_of_alive=(float(verdicts.sum()) / float(live.sum())
                              if live.sum() else float("nan")),
        population_all=_pop(live_idx),
        population_ignited=_pop(ign_idx),
        kstar_rel_err_vs_obs_median=(float(np.median(rel_obs)) if len(rel_obs) else float("nan")),
        kstar_rel_err_vs_obs_mean=(float(np.mean(rel_obs)) if len(rel_obs) else float("nan")),
        n_distinct_structures_all=int(len(set(signs[b] for b in live_idx))) if len(live_idx) else 0,
        _kstar_model=kstar_model, _live=live, _turing=verdicts, _J=J, _D=D,
        _signs=signs,
    )


def _score_against_answer_key(kstar_model, live, turing, key):
    """THE ONLY ANSWER-KEY READ IN THIS FILE, and it happens after training has finished.

    `kstar_fft_rel_err` is `validate.py`'s HEADLINE k* column and is defined against
    `AnswerKey.kstar_fft` (the FFT-measured wavenumber recorded by the generator), which is
    what D5's median 0.977 is. It is NOT the same as the firewall-clean
    `|k*_model - k*_obs| / k*_obs` this script also reports: on this target
    `kstar_obs` = 0.283878 and `AnswerKey.kstar_fft` = 0.288677, a 1.66 % difference, so
    the two columns are close but not interchangeable. Both are reported, each labelled.
    """
    if key.kstar_fft is None:
        return dict(kstar_fft_rel_err_median=float("nan"),
                    kstar_fft_rel_err_mean=float("nan"),
                    kstar_fft=None, note="AnswerKey.kstar_fft absent for this sample")
    kf = float(key.kstar_fft)
    rel = np.abs(kstar_model - kf) / kf
    out = dict(kstar_fft=kf,
               kstar_fft_rel_err_median=float(np.median(rel[live])) if live.any() else float("nan"),
               kstar_fft_rel_err_mean=float(np.mean(rel[live])) if live.any() else float("nan"))
    if turing.any():
        out["kstar_fft_rel_err_median_ignited"] = float(np.median(rel[turing]))
    else:
        out["kstar_fft_rel_err_median_ignited"] = float("nan")
    return out


# =======================================================================================
# main
# =======================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="experiments/redesign_r2/phase1")
    ap.add_argument("--datasets-root", default="data/datasets")
    ap.add_argument("--dataset-id", default=DATASET_ID)
    ap.add_argument("--sample-key", default=SAMPLE_KEY)
    ap.add_argument("--bio-box", default="configs/bio_box.yaml")
    ap.add_argument("--B", type=int, nargs="+", default=[64, 128, 256, 512],
                    help="batch sizes to sweep (§4.5: B=512 is a STARTING POINT, UNCALIBRATED)")
    ap.add_argument("--arms", nargs="+", default=["r2", "a0"],
                    choices=("r2", "a0", "r2_nobox", "r2_noprior"),
                    help="r2/a0 are the task's two arms; r2_nobox / r2_noprior are the "
                         "attribution ablations (see run_arm's docstring)")
    ap.add_argument("--steps", type=int, default=1500, help="Adam steps (§4.5: ~1500)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--backend", default="cubic", choices=("eig", "cubic"))
    ap.add_argument("--history-every", type=int, default=100,
                    help="history stride; history is recorded for the LARGEST B only "
                         "(an npz per sweep rung would add tens of MB to a tracked tree)")
    ap.add_argument("--staging-off-frac", type=float, default=0.25)
    ap.add_argument("--staging-ramp-frac", type=float, default=0.25)
    ap.add_argument("--verbose-every", type=int, default=250)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="re-run a cell that already has results/run.json (default: skip it, "
                         "so an interrupted sweep resumes instead of recomputing)")
    ap.set_defaults(skip_existing=True)
    args = ap.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.backend == "eig" and device.type == "cuda":
        raise ValueError(
            "backend='eig' on CUDA is refused: torch.linalg.eigvals has no batched cuSOLVER "
            "kernel for small non-symmetric matrices (~700 us per matrix, FLAT in B), which "
            "measured 2500x worse than 'cubic' at B=8 (model.BatchedRNGRN docstring). Use "
            "--backend cubic on CUDA, or --device cpu.")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)

    # ---- the target ------------------------------------------------------------------
    ri, key = from_registry(args.datasets_root, args.dataset_id, args.sample_key,
                            N_SPECIES, tuple(range(N_SPECIES)))
    frame_np = np.asarray(ri.frame, dtype=float)
    L = float(ri.L)
    frame_t = torch.tensor(frame_np, device=device)
    kstar_obs = float(obs.kstar_of(frame_np[0], L=L))
    kgrid = R._kgrid_for(kstar_obs, device=device)

    # §3.2: x*_i <- the observed frame channel's mean. The estimator's bias ON THIS TARGET
    # was measured before R2 (experiments/redesign_r2/frame_bias/results/bias.json:
    # ratio mean/x*_true = 1.068 / 1.193 / 0.811 per channel), which is the precondition
    # §3.2 requires and the reason this is a pin rather than an extrapolation.
    x_pin = [float(frame_np[c].mean()) for c in range(N_SPECIES)]
    box = _load_box(args.bio_box)

    prov = provenance()
    print(f"target      {args.dataset_id}/{args.sample_key}  L={L:.5f}  "
          f"frame={frame_np.shape}", flush=True)
    print(f"kstar_obs   {kstar_obs:.6f}   x*_pin {x_pin}", flush=True)
    print(f"box         {box}", flush=True)
    print(f"device      {device}  backend={args.backend}  git={prov.get('git_revision')}",
          flush=True)

    max_B = max(args.B)
    runs, index_rows = [], []
    for B in sorted(args.B):
        seeds = [R._restart_seed(args.model_seed, r) for r in range(B)]
        for arm in args.arms:
            run_id = f"phase1_{arm}_B{B}"
            run_dir = os.path.join(args.out, run_id)
            os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "config"), exist_ok=True)
            if args.skip_existing and os.path.isfile(
                    os.path.join(run_dir, "results", "run.json")):
                print(f"\n=== {run_id} === already has results/run.json — SKIPPED", flush=True)
                continue
            keep_history = (B == max_B) and args.history_every > 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            print(f"\n=== {run_id} ===  MemAvailable={_mem_available_mb():.0f}MB", flush=True)

            bmodel, xstar, rec, history, alive = run_arm(
                arm, B, frame_t=frame_t, frame_np=frame_np, L=L, kstar_obs=kstar_obs,
                kgrid=kgrid, box=box, seeds=seeds, steps=args.steps, lr=args.lr,
                grad_clip=args.grad_clip, device=device, backend=args.backend,
                history_every=(args.history_every if keep_history else 0),
                staging_off=args.staging_off_frac, staging_ramp=args.staging_ramp_frac,
                x_pin=x_pin, verbose_every=args.verbose_every)

            sc = score_population(bmodel, xstar, alive, kgrid, kstar_obs, arm)
            ak = _score_against_answer_key(sc.pop("_kstar_model"), sc["_live"],
                                           sc["_turing"], key)
            J, D, signs = sc.pop("_J"), sc.pop("_D"), sc.pop("_signs")
            live, turing = sc.pop("_live"), sc.pop("_turing")

            # CLAUDE.md §4: built by dict.update() merge, NEVER by **-expansion into a
            # literal — `rec` and `sc` both carry `n_alive`, which a literal would reject
            # (it did, on the first smoke run) or, worse, silently pick one of. Run identity
            # is applied LAST and therefore wins on any collision.
            summary = {}
            summary.update(rec)
            summary.update(sc)
            summary.update(ak)
            summary.update(dict(
                run_id=run_id, run_dir=run_dir,
                dataset_id=args.dataset_id, sample_key=args.sample_key,
                L=L, kstar_obs=kstar_obs,
                x_pin=(x_pin if arm != "a0" else None), device=str(device),
                backend=args.backend, lr=args.lr, grad_clip=args.grad_clip,
                model_seed=args.model_seed,
                weights=_arm_weights(arm),
                staging=dict(keys=["turing"], off_frac=args.staging_off_frac,
                             ramp_frac=args.staging_ramp_frac),
                git_sha=prov.get("git_revision"), torch=prov.get("torch")))
            with open(os.path.join(run_dir, "results", "run.json"), "w") as fh:
                json.dump(summary, fh, indent=2, default=str)
            with open(os.path.join(run_dir, "config", "frozen_config.yaml"), "w") as fh:
                fh.write(_frozen_config_yaml(args, arm, B, device, L, kstar_obs, x_pin, box))

            os.makedirs(os.path.join(run_dir, "arrays"), exist_ok=True)
            member_arrays = dict(
                seeds=np.asarray(seeds, dtype=np.int64), J=J, D=D, alive=live,
                turing=turing, sign_structures=np.asarray(signs, dtype=np.int64))
            if history is not None:
                member_arrays.update(history.to_arrays())
                summary["history_meta"] = history.meta()
            np.savez_compressed(os.path.join(run_dir, "arrays", "plot_arrays.npz"),
                                **member_arrays)

            # flat-scalar index row (CLAUDE.md §4: dict.update merge, run identity wins)
            row = dict(rec)
            row.update({k: v for k, v in sc.items() if not isinstance(v, (dict, list))})
            row.update({f"pop_all_{k}": v for k, v in sc["population_all"].items()})
            row.update({f"pop_ign_{k}": v for k, v in sc["population_ignited"].items()})
            row.update(ak)
            row.update(dict(run_id=run_id, arm=arm, B=B, dataset_id=args.dataset_id,
                            sample_key=args.sample_key, git_sha=prov.get("git_revision")))
            index_rows.append(row)
            runs.append(summary)
            # Written after EVERY cell, not once at the end. The first sweep was stopped
            # mid-run (the A0 arm's steady-state fallback made its last two rungs cost
            # hours) and the index had not been written at all, even though every finished
            # cell's results/run.json was on disk. An index that only exists if the whole
            # sweep survives is not an index.
            _write_index_and_summary(args.out, prov, args)
            print(f"--- {run_id}: turing {sc['n_turing']}/{B} "
                  f"({sc['turing_frac']:.3f}), distinct(all) "
                  f"{sc['population_all']['n_distinct_structures']}, "
                  f"kstar_fft_rel_err median {ak['kstar_fft_rel_err_median']:.4f}, "
                  f"{rec['member_steps_per_s']:.0f} member-steps/s, "
                  f"peak RSS {rec['peak_rss_mb']:.0f} MB", flush=True)

            del bmodel, history
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _write_index_and_summary(args.out, prov, args)
    print(f"\nwrote {os.path.join(args.out, 'results', 'phase1_summary.json')}", flush=True)
    return 0


# The D5 row, quoted from experiments/diag_fft/d5/target_reports.jsonl (git_sha 1b9ecda,
# 2026-08-12) so the comparison travels with the summary. Its BUDGET differs from this run's
# (10 seeds x 4 restarts, 2000 Adam steps + 50 LBFGS, serial, dispersion_backend='eig'), and
# only the three columns REDESIGN §1 names are comparable at all.
_D5_ROW = dict(
    source="experiments/diag_fft/d5/target_reports.jsonl", git_sha="1b9ecda",
    comparable_columns=["turing_frac", "n_distinct_structures", "kstar_fft_rel_err"],
    turing_frac=0.4, n_turing=4, n_seeds=10, n_distinct_structures=10,
    topology_consistency=0.1, mean_agreement=0.49876543209876545,
    kstar_fft_rel_err_median=0.976868414336446, kstar_fft_rel_err_mean=0.5915553846308315,
    budget="10 seeds x 4 restarts, 2000 Adam + 50 LBFGS, serial, eig backend",
    caveat="denominator is SEEDS (best-of-4 restarts), not members; REDESIGN §4.5 states "
           "it bounds neither the per-member ignition probability nor the redesign arm's")


def _write_index_and_summary(out: str, prov: dict, args) -> None:
    """Rebuild `runs.jsonl` and `results/phase1_summary.json` from EVERY cell on disk.

    Rebuilt rather than appended, and sourced from each cell's own `results/run.json`, so a
    resumed sweep produces exactly the same index as an uninterrupted one and no cell is
    double-counted. Cells from an earlier invocation are picked up automatically — which is
    what makes `--skip-existing` a resume rather than a way to lose rows.
    """
    runs, rows = [], []
    for path in sorted(glob.glob(os.path.join(out, "*", "results", "run.json"))):
        with open(path) as fh:
            summary = json.load(fh)
        runs.append(summary)
        # CLAUDE.md §4: index rows are FLAT SCALARS, built by dict.update() merge. The two
        # nested population blocks are flattened with a prefix rather than dropped; run
        # identity is applied last so it wins on any collision.
        row = {k: v for k, v in summary.items() if not isinstance(v, (dict, list))}
        for prefix, key in (("pop_all_", "population_all"),
                            ("pop_ign_", "population_ignited")):
            row.update({prefix + k: v for k, v in (summary.get(key) or {}).items()})
        row.update({f"w_{k}": v for k, v in (summary.get("weights") or {}).items()})
        row.update(dict(run_id=summary["run_id"], arm=summary["arm"], B=summary["B"]))
        rows.append(row)
    with open(os.path.join(out, "runs.jsonl"), "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    with open(os.path.join(out, "results", "phase1_summary.json"), "w") as fh:
        json.dump(dict(runs=runs, d5_baseline=_D5_ROW, provenance=prov,
                       args=vars(args)), fh, indent=2, default=str)


def _arm_weights(arm: str) -> dict:
    """The weight dict an arm actually ran with — recorded, never re-derived by a reader."""
    if arm == "a0":
        return dict(A0_WEIGHTS)
    w = dict(R2_WEIGHTS)
    if arm == "r2_noprior":
        w["param_prior"] = 0.0
    return w


def _frozen_config_yaml(args, arm, B, device, L, kstar_obs, x_pin, box) -> str:
    """The config that ACTUALLY ran, as YAML, at the tracked path .gitignore names."""
    import yaml
    return yaml.safe_dump(dict(
        arm=arm, B=B, steps=args.steps, lr=args.lr, grad_clip=args.grad_clip,
        model_seed=args.model_seed, device=str(device), dispersion_backend=args.backend,
        dataset_id=args.dataset_id, sample_key=args.sample_key, L=L,
        kstar_obs=kstar_obs, pin_xstar=(x_pin if arm != "a0" else None),
        param_boxes=({k: list(v) for k, v in box.items()}
                     if arm in ("r2", "r2_noprior") else None),
        weights=_arm_weights(arm),
        staging=dict(keys=["turing"], off_frac=args.staging_off_frac,
                     ramp_frac=args.staging_ramp_frac),
        bio_box_path=args.bio_box, history_every=args.history_every,
    ), sort_keys=True)


if __name__ == "__main__":
    sys.exit(main())
